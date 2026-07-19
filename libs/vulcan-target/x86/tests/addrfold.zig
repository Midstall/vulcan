//! Address-mode folding (Task 8), executed under qemu-i386. A load or store whose pointer is a
//! foldable `arith_imm.add(base, imm)` addresses `[base + disp32]` directly: the address-add is
//! dropped and the mem op carries a signed-32 displacement. x86-32 has no load-pair, so the win is
//! add-elision plus the folded displacement, mirroring x86-64's Task 5 integration.
//!
//! Every buffer lives IN THE GUEST as a run of consecutive `alloca` slots. `computeAllocaSlots` lays
//! same-size, same-alignment allocas out back-to-back, so `buf0 + size*j` is exactly `buf_j`. Each
//! alloca is kept alive by an instruction that references it (an init store, or the address-add whose
//! base it is), so none is dropped before the frame is sized. Loads and stores stay in program order,
//! so an init store always precedes the folded load that reads it back.
//!
//! Each test also proves the fold FIRED: `foldedMemOps` disassembles the compiled code and counts
//! memory accesses that carry a nonzero displacement off a base register other than esp. A spill,
//! alloca, or frame access addresses off esp, so filtering base != esp isolates the folded memory
//! ops. Without folding every mem op addresses `[base]` (no displacement) and the count is zero, so a
//! stuck assertion catches a fold that silently stopped firing.
//!
//! x86-32 is integer-only (bits <= 32), so unlike the x86-64/riscv64 addrfold suites there is no
//! floating-point fold case here.
//!
//! Results are checked modulo 256 (a process exit code / native return is read as one byte), so the
//! sweep values are chosen to differ in their low byte.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const h = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
const i16k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 16 } };
const i8k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 8 } };

/// Count memory accesses in `func`'s compiled code that carry a NONZERO displacement off a base
/// register other than esp. A folded constant-index load/store disassembles as `mov reg, dword ptr
/// [base + off]` and the like, with off != 0 and base != esp. Spill, alloca, and frame accesses
/// address off esp, so requiring base != esp excludes them. Zero unless address folding fired.
fn foldedMemOps(allocator: std.mem.Allocator, func: *Function) !usize {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);

    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const lb = std.mem.indexOfScalar(u8, line, '[') orelse continue;
        const rb = std.mem.indexOfScalarPos(u8, line, lb, ']') orelse continue;
        const inner = line[lb + 1 .. rb]; // e.g. "ebx + 4" or "esp + 8" or "eax"
        const base_end = std.mem.indexOfScalar(u8, inner, ' ') orelse inner.len;
        const base = inner[0..base_end];
        if (std.mem.eql(u8, base, "esp")) continue; // spill / alloca / frame, not a fold
        // A folded access carries a displacement (" + N" or " - N"); a plain `[base]` load does not.
        if (std.mem.indexOf(u8, inner, " + ") != null or std.mem.indexOf(u8, inner, " - ") != null) count += 1;
    }
    return count;
}

/// Run `func` on both the qemu-i386 and native-i386 backends (native skips off a non-i386 host),
/// asserting the low byte of the result equals `expected` on whichever backend is available.
fn expectRun(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, expected: i64) !void {
    const want: u8 = @truncate(@as(u64, @bitCast(expected)));
    var ran = false;
    for ([_]h.Backend{ h.qemu, h.native }) |backend| {
        const got = h.runFunc(io, allocator, func, args, backend) catch |e| switch (e) {
            error.SkipZigTest => continue,
            else => return e,
        };
        try std.testing.expectEqual(want, got);
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

// --- test 1: an i32 copy folds both loads and stores ------------------------------------------------

const copy_n = 4;

/// f(arg): src[j] = arg + j for j in 0..N; dst[j] = src[j] via folded loads/stores; return sum(dst).
/// The copy reads `src0 + 4j` and writes `dst0 + 4j` (both fold for j >= 1), so it exercises a folded
/// load AND a folded store, and the readback re-folds the loads. Expected = sum(arg + j) = 4*arg + 6.
fn buildCopy(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32k);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i32_t);

    var src: [copy_n]Value = undefined;
    var dst: [copy_n]Value = undefined;
    for (0..copy_n) |j| src[j] = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    for (0..copy_n) |j| dst[j] = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    // Initialize each src slot (references every src alloca so none is DCE'd) to arg + j.
    for (0..copy_n) |j| {
        const jv = try func.appendInst(e, i32_t, .{ .iconst = @intCast(j) });
        const v = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = arg, .rhs = jv } });
        try func.appendStore(e, v, src[j]); // own pointer, off 0 (not folded)
    }
    // Zero each dst slot (references every dst alloca).
    for (0..copy_n) |j| {
        const z = try func.appendInst(e, i32_t, .{ .iconst = 0 });
        try func.appendStore(e, z, dst[j]);
    }
    // Copy via folded addresses: v = load(src0 + 4j); store v to (dst0 + 4j).
    for (0..copy_n) |j| {
        const sp = if (j == 0) src[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = src[0], .imm = @intCast(j * 4) } });
        const v = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = sp } });
        const dp = if (j == 0) dst[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = dst[0], .imm = @intCast(j * 4) } });
        try func.appendStore(e, v, dp);
    }
    // Read the copy back through folded loads and sum it.
    var acc: ?Value = null;
    for (0..copy_n) |j| {
        const rp = if (j == 0) dst[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = dst[0], .imm = @intCast(j * 4) } });
        const w = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = rp } });
        acc = if (acc) |a| try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = w } }) else w;
    }
    func.setTerminator(e, .{ .ret = acc.? });
    return func;
}

test "x86-32 addrfold: constant-index i32 store then load folds and computes" {
    const allocator = std.testing.allocator;
    var func = try buildCopy(allocator);
    defer func.deinit();

    // The copy and readback fold at least the three nonzero-offset loads and three nonzero-offset
    // stores, so many folded mem ops survive (a zero here would mean folding never fired).
    try std.testing.expect(try foldedMemOps(allocator, &func) >= 3);

    for ([_]i64{ 0, 1, 7, 25, 100 }) |arg| {
        const expected: i64 = 4 * arg + 6; // sum(arg + j) for j in 0..4
        try expectRun(std.testing.io, allocator, &func, &.{arg}, expected);
    }
}

// --- test 2: byte and halfword constant-index loads fold and compute (sign + zero extension) --------

/// f(arg): buf[j] = arg + j (as an `elem`-typed slot, i.e. narrowed to it) for j in 0..4; return sum
/// of folded loads buf0 + size*j. `elem` is i8 (size 1) / i16 (size 2), so a wrong displacement
/// scale would read the wrong element. x86-32 has no `.convert` op: the narrowing to `elem`'s width
/// happens naturally at the STORE (a byte/halfword mov writes only the low bytes of the full-width
/// sum, per `intBits(func, st.value)` reading the stored value's own declared type), and the widening
/// back out happens at the LOAD (sign/zero-extended into the full 32-bit destination per `elem`'s
/// signedness). So no explicit convert is needed on either side. Expected = sum(trunc(arg + j)).
fn buildNarrowSum(allocator: std.mem.Allocator, comptime kind: ir.types.TypeKind, comptime size: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32k);
    const elem_t = try func.types.intern(kind);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i32_t);

    var buf: [4]Value = undefined;
    for (0..4) |j| buf[j] = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = elem_t } });
    for (0..4) |j| {
        // `elem_t`-typed result: the store below writes only its low bytes, truncating arg + j.
        const narrow = try func.appendArithImm(e, elem_t, .add, arg, @intCast(j));
        try func.appendStore(e, narrow, buf[j]); // own pointer, off 0
    }
    var acc: ?Value = null;
    for (0..4) |j| {
        const p = if (j == 0) buf[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf[0], .imm = @intCast(j * size) } });
        const v = try func.appendInst(e, elem_t, .{ .load = .{ .ptr = p } }); // sign/zero-extends into 32 bits
        acc = if (acc) |a| try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = v } }) else v;
    }
    func.setTerminator(e, .{ .ret = acc.? });
    return func;
}

test "x86-32 addrfold: byte/half constant-index loads fold and compute" {
    const allocator = std.testing.allocator;

    var byte_fn = try buildNarrowSum(allocator, i8k, 1);
    defer byte_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &byte_fn) >= 1);
    for ([_]i64{ 0, 1, 10, 40, 100 }) |arg| {
        var expected: i32 = 0;
        for (0..4) |j| expected += @as(i8, @truncate(arg + @as(i64, @intCast(j))));
        try expectRun(std.testing.io, allocator, &byte_fn, &.{arg}, @as(i64, expected));
    }

    var half_fn = try buildNarrowSum(allocator, i16k, 2);
    defer half_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &half_fn) >= 1);
    for ([_]i64{ 0, 1, 1000, 20000, 7 }) |arg| {
        var expected: i32 = 0;
        for (0..4) |j| expected += @as(i16, @truncate(arg + @as(i64, @intCast(j))));
        try expectRun(std.testing.io, allocator, &half_fn, &.{arg}, @as(i64, expected));
    }
}

// --- test 3: a CROSS-BLOCK folded load (the correctness invariant) ----------------------------------

const xblock_pressure = 8;

/// f(arg, cond): in ENTRY, buf1 = arg and `p = buf0 + 4` (= buf1's address, a DEAD add: its only use
/// is the load in the successor). On cond > 0, `then_b` first builds several live values (so the free
/// list would hand a temp `buf0`'s register if `buf0` were treated as dead after entry), then loads
/// [p] (folds to `4(buf0)`) and reduces. The load's pointer use must be attributed to `buf0` in the
/// SUCCESSOR block, so `buf0` stays live across the branch; miss the reroute and a temp steals its
/// register and the folded load reads a garbage address. Expected(cond>0) = arg + sum(cond + k),
/// k in 1..P; Expected(cond<=0) = cond.
fn buildCrossBlock(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32k);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const arg = try func.appendBlockParam(entry, i32_t);
    const cond = try func.appendBlockParam(entry, i32_t);

    const buf0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const buf1 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, arg, buf1); // buf1 = arg (own pointer, references buf1)
    // buf0 is referenced only by this address-add, whose result feeds solely the successor's load: a
    // dead add, so buf0's ONLY successor liveness flows through the folded load's baseOf.
    const p = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 4 } });
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = cond, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });

    // then_b: build the pressure BEFORE the folded load so a stolen base register is read by the load.
    var vals: [xblock_pressure]Value = undefined;
    for (0..xblock_pressure) |k| {
        vals[k] = try func.appendInst(then_b, i32_t, .{ .arith_imm = .{ .op = .add, .lhs = cond, .imm = @intCast(k + 1) } });
    }
    const w = try func.appendInst(then_b, i32_t, .{ .load = .{ .ptr = p } }); // folds to 4(buf0) = buf1 = arg
    var acc = w;
    for (0..xblock_pressure) |k| acc = try func.appendInst(then_b, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[k] } });
    func.setTerminator(then_b, .{ .ret = acc });
    func.setTerminator(else_b, .{ .ret = cond });
    return func;
}

test "x86-32 addrfold: a CROSS-BLOCK folded load computes correctly" {
    const allocator = std.testing.allocator;
    var func = try buildCrossBlock(allocator);
    defer func.deinit();

    try std.testing.expect(try foldedMemOps(allocator, &func) >= 1); // the successor load folded

    for ([_]i64{ 1, 0, 5, 30, 12 }) |cond| {
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
        try expectRun(std.testing.io, allocator, &func, &.{ arg, cond }, expected);
    }
}

// --- test 4: a base whose add has a NON-folded consumer keeps the add -------------------------------

/// f(arg): buf1 = arg; `p = buf0 + 4` (= buf1's address). `p` feeds the folded load [p] AND a plain
/// `p - buf0` subtraction (a non-folded use), so `p` is NOT dead and the address-add MUST still be
/// emitted. `p - buf0 = 4`, so return `load(4(buf0)) + 4 = arg + 4`. If the add were wrongly elided,
/// the subtraction would read a garbage `p` and the result would diverge.
fn buildLiveAdd(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32k);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i32_t);

    const buf0 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const buf1 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(e, arg, buf1); // buf1 = arg
    const p = try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 4 } }); // = buf1 addr
    const v = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = p } }); // folds to 4(buf0) = arg
    const d = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .sub, .lhs = p, .rhs = buf0 } }); // = 4, keeps p live
    const r = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = d } }); // arg + 4
    func.setTerminator(e, .{ .ret = r });
    return func;
}

test "x86-32 addrfold: a base used by a folded load AND another consumer keeps the add and computes" {
    const allocator = std.testing.allocator;
    var func = try buildLiveAdd(allocator);
    defer func.deinit();

    try std.testing.expect(try foldedMemOps(allocator, &func) >= 1); // the load still folded

    for ([_]i64{ 0, 1, 9, 40, 200 }) |arg| {
        try expectRun(std.testing.io, allocator, &func, &.{arg}, arg + 4);
    }
}
