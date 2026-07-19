//! x86_64 adoption Task 2: the SHARED Wimmer-Franz allocator produces EXECUTABLE x86-64 code. Each
//! test builds TWO identical functions, compiles one through the backend's own `selectFunction` (the
//! reference) and the other through `isel.compileFunctionWimmerX86` (the shared allocator + the same
//! emission), runs BOTH under qemu-x86_64 (-cpu max), and asserts the results are bit-identical
//! across many inputs. Scope is the gpr + xmm classes (scalar float AND 128/256-bit SIMD vectors),
//! including the NEW capability: a value live across a call occupies a callee-saved GPR (rbx/r12..r15)
//! via the new push/pop prologue rather than spilling. qemu is the execution oracle: a divergence
//! means the shared allocation was translated or emitted wrong. The 256-bit (AVX) cases need `-cpu
//! max`, which the harness always passes.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const encode = @import("../encode.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// Compile `func` through the shared Wimmer allocator (mutates `func`: it splits critical edges) and
/// run it under qemu with integer args, returning the low byte of rax.
fn runWimmer(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64) !u8 {
    var compiled = try isel.compileFunctionWimmerX86(allocator, func);
    defer compiled.deinit(allocator);
    return harness.runCodeInt(io, allocator, compiled.code, args, harness.qemu);
}

/// Run the reference (`selectFunction`) and the Wimmer path on two freshly-built copies of the same
/// function for every input, asserting the integer results match. `build` takes only the allocator so
/// each side gets its own untouched function (the Wimmer path mutates the IR in place).
fn expectIntMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, inputs: []const []const i64) !void {
    const allocator = std.testing.allocator;
    for (inputs) |args| {
        var ref_func = try build(allocator);
        defer ref_func.deinit();
        var wim_func = try build(allocator);
        defer wim_func.deinit();

        const ref = harness.runFunc(io, allocator, &ref_func, args, harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = runWimmer(io, allocator, &wim_func, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

// ---------------------------------------------------------------------------
// 1. Straight-line integer arithmetic.
// ---------------------------------------------------------------------------

/// f(a, b, c) = (a + b) * (b + c) - (a * c). A handful of simultaneously-live temps, no spilling: the
/// baseline that the prologue, the ABI-register param moves, and the int arithmetic translate right.
fn buildStraightLine(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);
    const c = try func.appendBlockParam(entry, i64_t);
    const ab = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const bc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = c } });
    const ac = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const prod = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = ab, .rhs = bc } });
    const res = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = ac } });
    func.setTerminator(entry, .{ .ret = res });
    return func;
}

test "wimmer-x86: straight-line int arithmetic matches" {
    const inputs = [_][]const i64{ &.{ 1, 2, 3 }, &.{ 0, 0, 0 }, &.{ -5, 7, -9 }, &.{ 100, -20, 30 }, &.{ 123456, -1, 2 } };
    try expectIntMatch(std.testing.io, buildStraightLine, &inputs);
}

// ---------------------------------------------------------------------------
// 2. Integer register-pressure kernel (forces spilling / live-range splitting).
// ---------------------------------------------------------------------------

const n_fan = 30;

/// f(n) = sum_k (n*(k+1) + k) for k in 0..30. All 30 terms are created before any is consumed, so far
/// more integer values are live at once than the 12 allocatable gpr registers: the shared allocator
/// must spill and tail-split. The reduction reloads every operand, so a wrong split/spill diverges.
fn buildIntPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    var a: [n_fan]Value = undefined;
    for (0..n_fan) |k| {
        const coeff = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        const prod = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = coeff } });
        a[k] = try func.appendArithImm(entry, i64_t, .add, prod, @intCast(k));
    }
    var sum = a[n_fan - 1];
    var k: usize = n_fan - 1;
    while (k > 0) {
        k -= 1;
        sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = a[k] } });
    }
    func.setTerminator(entry, .{ .ret = sum });
    return func;
}

test "wimmer-x86: int register pressure spills/splits and matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{7}, &.{-3}, &.{100}, &.{-1000}, &.{123456} };
    try expectIntMatch(std.testing.io, buildIntPressure, &inputs);
}

// ---------------------------------------------------------------------------
// 3. A value live across a call survives in a callee-saved register (the NEW push/pop prologue).
// ---------------------------------------------------------------------------

/// The leaf callee `g(x) = x + 2`. Built as its own function so the caller's `call` is a real
/// inter-function call that clobbers every caller-saved register.
fn buildAddTwoCallee(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const blk = try func.appendBlock();
    const x = try func.appendBlockParam(blk, i64_t);
    const r = try func.appendArithImm(blk, i64_t, .add, x, 2);
    func.setTerminator(blk, .{ .ret = r });
    return func;
}

/// The caller `f(a)`: t = a + 1 (defined BEFORE the call), cr = g(10), return t + cr = a + 13. `t` is
/// live ACROSS the call. Every caller-saved register is clobbered by the call, so the shared allocator
/// keeps `t` in a CALLEE-SAVED register (rbx/r12..r15) instead of spilling, which fires the new
/// push/pop prologue. The reference native path spills `t` across the call; both compute a + 13.
fn buildAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const t = try func.appendArithImm(entry, i64_t, .add, a, 1);
    const ten = try func.appendInst(entry, i64_t, .{ .iconst = 10 });
    const cr = try func.appendCall(entry, i64_t, "callee", &.{ten});
    const r = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = t, .rhs = cr } });
    func.setTerminator(entry, .{ .ret = r });
    return func;
}

/// Concatenate `caller_c` (entry, at offset 0) and `callee_c`, patch every call relocation in the
/// caller (all target "callee") into an intra-image rel32, and run under qemu with integer args. The
/// caller may be compiled through whichever pipeline the test chose (Wimmer or reference).
fn linkRunInt(io: std.Io, allocator: std.mem.Allocator, caller_c: *const isel.Compiled, callee_c: *const isel.Compiled, args: []const i64) !u8 {
    const code = try allocator.alloc(u8, caller_c.code.len + callee_c.code.len);
    defer allocator.free(code);
    @memcpy(code[0..caller_c.code.len], caller_c.code);
    @memcpy(code[caller_c.code.len..], callee_c.code);
    const callee_start = caller_c.code.len;
    for (caller_c.relocs) |reloc| {
        std.debug.assert(std.mem.eql(u8, reloc.symbol, "callee"));
        // rel32 for `call` (E8 disp): target - (site + 4), both within the code (equal stub shift).
        const rel: i32 = @intCast(@as(i64, @intCast(callee_start)) - @as(i64, @intCast(reloc.offset + 4)));
        std.mem.writeInt(u32, code[reloc.offset..][0..4], @bitCast(rel), .little);
    }
    return harness.runCodeInt(io, allocator, code, args, harness.qemu);
}

/// True iff `code` begins with a `push r64` of a callee-saved GPR (rbx=0x53, r12..r15 = 0x41 0x54..
/// 0x57). The Wimmer prologue pushes the used callee-saved registers FIRST (before `sub rsp`), so a
/// push at code[0] proves a value was placed in a callee-saved register, exercising the push/pop
/// prologue rather than a spill.
fn startsWithCalleeSavedPush(code: []const u8) bool {
    if (code.len >= 1 and code[0] == 0x53) return true; // push rbx
    if (code.len >= 2 and code[0] == 0x41 and code[1] >= 0x54 and code[1] <= 0x57) return true; // push r12..r15
    return false;
}

test "wimmer-x86: a value live across a call survives in a callee-saved register (new push/pop prologue)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const inputs = [_]i64{ 0, 1, 7, -3, 100, -1000, 123456 };

    // Compile the callee once (leaf, no calls). Shared by both the reference and the Wimmer caller.
    var callee = try buildAddTwoCallee(allocator);
    defer callee.deinit();
    var callee_c = isel.compile(allocator, &callee) catch |e| switch (e) {
        error.Unsupported => return error.SkipZigTest,
        else => return e,
    };
    defer callee_c.deinit(allocator);

    // The Wimmer caller MUST place `t` in a callee-saved register and emit the push/pop prologue.
    var probe = try buildAcrossCall(allocator);
    defer probe.deinit();
    var probe_c = try isel.compileFunctionWimmerX86(allocator, &probe);
    defer probe_c.deinit(allocator);
    try std.testing.expect(startsWithCalleeSavedPush(probe_c.code));

    for (inputs) |a| {
        var ref_caller = try buildAcrossCall(allocator);
        defer ref_caller.deinit();
        var wim_caller = try buildAcrossCall(allocator);
        defer wim_caller.deinit();

        var ref_caller_c = try isel.compile(allocator, &ref_caller);
        defer ref_caller_c.deinit(allocator);
        var wim_caller_c = try isel.compileFunctionWimmerX86(allocator, &wim_caller);
        defer wim_caller_c.deinit(allocator);

        const ref = linkRunInt(io, allocator, &ref_caller_c, &callee_c, &.{a}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = linkRunInt(io, allocator, &wim_caller_c, &callee_c, &.{a}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
        try std.testing.expectEqual(@as(u8, @truncate(@as(u64, @bitCast(a + 13)))), got);
    }
}

// ---------------------------------------------------------------------------
// 4. A div and a shift function (fixed-register clobbers rax/rdx/rcx per position).
// ---------------------------------------------------------------------------

/// f(a, b, c) = (a / b) + (a % b) + (c << 3) + (a >> 1). Exercises the div/rem (rax+rdx clobber) and
/// the shl/shr (rcx clobber) per-position fixed intervals: the shared allocator may place operands in
/// the clobbered registers (unlike the native pool exclusion), which the div/shift lowering stages out
/// defensively. `b` is nonzero in every input so the divide is defined.
fn buildDivShift(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);
    const c = try func.appendBlockParam(entry, i64_t);
    const q = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .div, .lhs = a, .rhs = b } });
    const rem = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .rem, .lhs = a, .rhs = b } });
    const three = try func.appendInst(entry, i64_t, .{ .iconst = 3 });
    const shl = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .shl, .lhs = c, .rhs = three } });
    const one = try func.appendInst(entry, i64_t, .{ .iconst = 1 });
    const shr = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .shr, .lhs = a, .rhs = one } });
    const s1 = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = q, .rhs = rem } });
    const s2 = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = shl } });
    const s3 = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = shr } });
    func.setTerminator(entry, .{ .ret = s3 });
    return func;
}

test "wimmer-x86: a div and a shift function matches (fixed-reg clobbers)" {
    const inputs = [_][]const i64{ &.{ 100, 7, 3 }, &.{ 40, 6, 1 }, &.{ -50, 8, 2 }, &.{ 12345, 11, -4 }, &.{ 7, 2, 5 } };
    try expectIntMatch(std.testing.io, buildDivShift, &inputs);
}

// ---------------------------------------------------------------------------
// 5. A loop-carried int sum across a pressured body, and a diamond (cross-block edge moves).
// ---------------------------------------------------------------------------

const n_body = 20;

/// f(n): acc=0; for i in 0..n: { t[k] = (i+1)*(k+1) for k in 0..20 (all live at once); acc += sum(t) };
/// return acc. Three integers cross the loop as block params (i, n, acc), so the back-edge jump is a
/// reg->reg parallel move (the cross-block Wimmer edge-move path), while the body creates 20
/// simultaneously-live temporaries (past the 12 allocatable gpr registers), so the shared allocator
/// spills and tail-splits INSIDE the body. A wrong intra-block split or a wrong back-edge move diverges.
fn buildLoopSum(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
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
    const ip1 = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    var t: [n_body]Value = undefined;
    for (0..n_body) |k| {
        const coeff = try func.appendInst(body, i64_t, .{ .iconst = @intCast(k + 1) });
        t[k] = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .mul, .lhs = ip1, .rhs = coeff } });
    }
    var s = t[n_body - 1];
    var k: usize = n_body - 1;
    while (k > 0) {
        k -= 1;
        s = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = t[k] } });
    }
    const next_acc = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = b_acc, .rhs = s } });
    const next_i = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    try func.setJump(body, header, &.{ next_i, b_n, next_acc });

    const e_acc = try func.appendBlockParam(exit, i64_t);
    func.setTerminator(exit, .{ .ret = e_acc });
    return func;
}

test "wimmer-x86: a loop-carried sum + a diamond match (cross-block edge moves)" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{3}, &.{7}, &.{12} };
    try expectIntMatch(std.testing.io, buildLoopSum, &inputs);
}

/// f(n): c = n*3 (live on BOTH arms and the join); if n > 0 -> a else b; a: va = c + 10; b: vb = c +
/// 20; m(p): return p + c. The join `m` takes a phi `p` (va from a, vb from b) resolved by moves on the
/// jump edges a->m / b->m, while `c` is defined in the entry and stays live across both arms into `m`.
fn buildDiamond(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const a_blk = try func.appendBlock();
    const b_blk = try func.appendBlock();
    const m_blk = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
    const three = try func.appendInst(entry, i64_t, .{ .iconst = 3 });
    const c = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = three } });
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = n, .rhs = zero } });
    try func.appendIf(entry, cond, .{ .target = a_blk }, .{ .target = b_blk });

    const va = try func.appendArithImm(a_blk, i64_t, .add, c, 10);
    try func.setJump(a_blk, m_blk, &.{va});

    const vb = try func.appendArithImm(b_blk, i64_t, .add, c, 20);
    try func.setJump(b_blk, m_blk, &.{vb});

    const p = try func.appendBlockParam(m_blk, i64_t);
    const r = try func.appendInst(m_blk, i64_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = c } });
    func.setTerminator(m_blk, .{ .ret = r });
    return func;
}

test "wimmer-x86: a diamond with an int value live on both paths matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{-1}, &.{5}, &.{-9}, &.{40} };
    try expectIntMatch(std.testing.io, buildDiamond, &inputs);
}

// ---------------------------------------------------------------------------
// 6. An xmm scalar-float pressure kernel (xmm split: intra-block store/reload of a scalar float).
// ---------------------------------------------------------------------------

const n_flive = 20;

/// f(a) = sum_i (a + i) for i in 0..20, f32. All 20 sums are live at once, exceeding the 13
/// allocatable xmm registers, so the shared allocator spills/splits scalar floats (the class-1
/// store/reload actions through movups). Every intermediate is an exact small integer in f32, so the
/// dependency-chain reduction is order-independent and must match the reference bit-for-bit.
fn buildFloatPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    var vals: [n_flive]Value = undefined;
    for (0..n_flive) |i| {
        const ci = try func.appendInst(entry, i32_t, .{ .iconst = @intCast(i) });
        const cf = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = ci } });
        vals[i] = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = cf } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(entry, .{ .ret = acc });
    return func;
}

test "wimmer-x86: an xmm float pressure kernel matches" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const a_vals = [_]f32{ 0.0, 1.0, 100.0, -7.0, 1234.0 };
    for (a_vals) |a_val| {
        var ref_func = try buildFloatPressure(allocator);
        defer ref_func.deinit();
        var wim_func = try buildFloatPressure(allocator);
        defer wim_func.deinit();

        const ref = harness.runFloatFunc(io, allocator, &ref_func, &.{a_val}, harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        var wim_c = try isel.compileFunctionWimmerX86(allocator, &wim_func);
        defer wim_c.deinit(allocator);
        const got = harness.runCodeFloat(io, allocator, wim_c.code, &.{a_val}, harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

// ---------------------------------------------------------------------------
// 7. SIMD vectors (128-bit xmm and 256-bit ymm) through the shared allocator.
// ---------------------------------------------------------------------------

/// Run the reference (`selectFunction`) and the Wimmer path on two fresh copies of the same function
/// for every f32 input tuple, asserting the f32 results match (the low byte of the result's bits, the
/// process exit code). `build` takes only the allocator so each side gets its own untouched function
/// (the Wimmer path mutates the IR in place). Skips (not fails) when qemu is unavailable.
fn expectFloatMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, inputs: []const []const f32) !void {
    const allocator = std.testing.allocator;
    for (inputs) |args| {
        var ref_func = try build(allocator);
        defer ref_func.deinit();
        var wim_func = try build(allocator);
        defer wim_func.deinit();

        const ref = harness.runFloatFunc(io, allocator, &ref_func, args, harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        var wim_c = try isel.compileFunctionWimmerX86(allocator, &wim_func);
        defer wim_c.deinit(allocator);
        const got = harness.runCodeFloat(io, allocator, wim_c.code, args, harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

/// f(a0..3, b0..3): va=<a>, vb=<b>, vc = va + vb, vd = vc * va (both a 128-bit addps and mulps), then
/// extract the four lanes and reduce to a scalar. Eight f32 params fit in xmm0..xmm7, so no stack args.
/// Exercises 128-bit vector values living in and moving between xmm registers on the shared path.
fn build128Arith(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const entry = try func.appendBlock();
    var ap: [4]Value = undefined;
    var bp: [4]Value = undefined;
    for (0..4) |i| ap[i] = try func.appendBlockParam(entry, t);
    for (0..4) |i| bp[i] = try func.appendBlockParam(entry, t);
    const va = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const vb = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
    const vc = try func.appendInst(entry, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    const vd = try func.appendInst(entry, v4, .{ .arith = .{ .op = .mul, .lhs = vc, .rhs = va } });
    var c: [4]Value = undefined;
    for (0..4) |i| c[i] = try func.appendInst(entry, t, .{ .extract = .{ .aggregate = vd, .index = @intCast(i) } });
    const s01 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    func.setTerminator(entry, .{ .ret = s });
    return func;
}

test "wimmer-x86: a 128-bit vector arithmetic function matches" {
    const inputs = [_][]const f32{
        &.{ 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 },
        &.{ 0.0, 1.0, -2.0, 3.0, 10.0, -20.0, 30.0, 0.5 },
        &.{ -1.5, -2.5, -3.5, -4.5, 100.0, 0.25, -0.75, 9.0 },
    };
    try expectFloatMatch(std.testing.io, build128Arith, &inputs);
}

/// A 256-bit (AVX) counterpart of `build128Arith`. Eight lanes need sixteen scalars, more than the
/// eight xmm arg registers, so the two <8 x f32> vectors are built from fconsts (no stack args). One
/// 256-bit vaddps then vmulps, extract all eight lanes (the high four via vextractf128), reduce.
fn build256Arith(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ca = [8]f32{ 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
    const da = [8]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    const t = try func.types.intern(.{ .float = .f32 });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = t } });
    const entry = try func.appendBlock();
    var af: [8]Value = undefined;
    for (0..8) |i| af[i] = try func.appendInst(entry, t, .{ .fconst = ca[i] });
    const va = try func.appendInst(entry, v8, .{ .struct_new = .{ .fields = try func.internValueList(&af) } });
    var bf: [8]Value = undefined;
    for (0..8) |i| bf[i] = try func.appendInst(entry, t, .{ .fconst = da[i] });
    const vb = try func.appendInst(entry, v8, .{ .struct_new = .{ .fields = try func.internValueList(&bf) } });
    const vc = try func.appendInst(entry, v8, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    const vd = try func.appendInst(entry, v8, .{ .arith = .{ .op = .mul, .lhs = vc, .rhs = va } });
    var e: [8]Value = undefined;
    for (0..8) |i| e[i] = try func.appendInst(entry, t, .{ .extract = .{ .aggregate = vd, .index = @intCast(i) } });
    var s = e[0];
    for (1..8) |i| s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = e[i] } });
    func.setTerminator(entry, .{ .ret = s });
    return func;
}

test "wimmer-x86: a 256-bit vector arithmetic function matches" {
    // No inputs: the vectors are constants (256-bit needs 16 lanes > 8 xmm arg regs). Runs under
    // qemu -cpu max, which exposes AVX.
    try expectFloatMatch(std.testing.io, build256Arith, &.{&.{}});
}

/// Build a register-pressure vector kernel of `lanes`-wide vectors: `count` vectors (each a broadcast
/// of a distinct constant) are all live before any is consumed, exceeding the 13 allocatable xmm
/// registers, so the shared allocator SPLITS a vector to a spill slot and reloads it. The slot store
/// and reload pick their width from the value's IR type (movups for 128-bit, vmovups for the 256-bit
/// ymm), both UNALIGNED, so the existing 32-byte slot area suffices. The vectors are summed, then all
/// lanes are extracted and reduced. A wrong-width spill or reload drops lanes and diverges.
fn buildVecPressure(allocator: std.mem.Allocator, comptime lanes: u32, comptime count: usize) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const vt = try func.types.intern(.{ .vector = .{ .len = lanes, .elem = t } });
    const entry = try func.appendBlock();
    var vs: [count]Value = undefined;
    for (0..count) |i| {
        const c = try func.appendInst(entry, t, .{ .fconst = @as(f64, @floatFromInt(i)) + 0.1 });
        var fields: [lanes]Value = undefined;
        for (0..lanes) |k| fields[k] = c;
        vs[i] = try func.appendInst(entry, vt, .{ .struct_new = .{ .fields = try func.internValueList(&fields) } });
    }
    var acc = vs[0];
    for (1..count) |i| acc = try func.appendInst(entry, vt, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vs[i] } });
    var e: [lanes]Value = undefined;
    for (0..lanes) |k| e[k] = try func.appendInst(entry, t, .{ .extract = .{ .aggregate = acc, .index = @intCast(k) } });
    var s = e[0];
    for (1..lanes) |k| s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = e[k] } });
    func.setTerminator(entry, .{ .ret = s });
    return func;
}

fn build128Pressure(allocator: std.mem.Allocator) anyerror!Function {
    return buildVecPressure(allocator, 4, 16);
}
fn build256Pressure(allocator: std.mem.Allocator) anyerror!Function {
    return buildVecPressure(allocator, 8, 16);
}

test "wimmer-x86: vector register pressure spills a vector to a slot and reloads" {
    // 128-bit (movups, 16-byte) and 256-bit (vmovups, 32-byte) both round-trip through a spill slot.
    try expectFloatMatch(std.testing.io, build128Pressure, &.{&.{}});
    try expectFloatMatch(std.testing.io, build256Pressure, &.{&.{}});
}

/// The leaf callee `g(x) = x + x`, f32. A real inter-function call so the caller's `call` clobbers
/// every caller-saved register, including all xmm.
fn buildDoubleCallee(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const x = try func.appendBlockParam(blk, t);
    const r = try func.appendInst(blk, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    func.setTerminator(blk, .{ .ret = r });
    return func;
}

/// The caller `f(a,b,c,d)`: v = <a,b,c,d> (a 128-bit vector) is defined BEFORE a call and used AFTER
/// it, so v is live ACROSS the call. xmm has NO callee-saved register, so the shared allocator must
/// SPILL v to a 32-byte slot (store before the call, reload after) rather than keep it in a register.
/// cr = g(a); return reduce(v lanes) + cr. Exercises the vector spill/reload forced by a call clobber.
fn buildVecAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const entry = try func.appendBlock();
    var ap: [4]Value = undefined;
    for (0..4) |i| ap[i] = try func.appendBlockParam(entry, t);
    const v = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const cr = try func.appendCall(entry, t, "callee", &.{ap[0]});
    var c: [4]Value = undefined;
    for (0..4) |i| c[i] = try func.appendInst(entry, t, .{ .extract = .{ .aggregate = v, .index = @intCast(i) } });
    const s01 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s0123 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s0123, .rhs = cr } });
    func.setTerminator(entry, .{ .ret = s });
    return func;
}

/// Concatenate `caller_c` (entry, at offset 0) and `callee_c`, patch the caller's "callee" relocations
/// into intra-image rel32s, and run under qemu with f32 args (result read from xmm0). The float
/// analogue of `linkRunInt`.
fn linkRunFloat(io: std.Io, allocator: std.mem.Allocator, caller_c: *const isel.Compiled, callee_c: *const isel.Compiled, fargs: []const f32) !u8 {
    const code = try allocator.alloc(u8, caller_c.code.len + callee_c.code.len);
    defer allocator.free(code);
    @memcpy(code[0..caller_c.code.len], caller_c.code);
    @memcpy(code[caller_c.code.len..], callee_c.code);
    const callee_start = caller_c.code.len;
    for (caller_c.relocs) |reloc| {
        std.debug.assert(std.mem.eql(u8, reloc.symbol, "callee"));
        const rel: i32 = @intCast(@as(i64, @intCast(callee_start)) - @as(i64, @intCast(reloc.offset + 4)));
        std.mem.writeInt(u32, code[reloc.offset..][0..4], @bitCast(rel), .little);
    }
    return harness.runCodeFloat(io, allocator, code, fargs, harness.qemu);
}

test "wimmer-x86: a vector live across a call survives (callee-saved not available for xmm, so it spills)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const inputs = [_][]const f32{
        &.{ 1.1, 2.2, 3.3, 4.4 },
        &.{ -5.0, 6.5, 7.25, -8.75 },
        &.{ 0.5, 0.25, 100.0, -3.0 },
    };

    // The callee is a leaf (no calls). Compile it once, shared by the reference and Wimmer callers.
    var callee = try buildDoubleCallee(allocator);
    defer callee.deinit();
    var callee_c = isel.compile(allocator, &callee) catch |e| switch (e) {
        error.Unsupported => return error.SkipZigTest,
        else => return e,
    };
    defer callee_c.deinit(allocator);

    for (inputs) |fargs| {
        var ref_caller = try buildVecAcrossCall(allocator);
        defer ref_caller.deinit();
        var wim_caller = try buildVecAcrossCall(allocator);
        defer wim_caller.deinit();

        var ref_caller_c = isel.compile(allocator, &ref_caller) catch |e| switch (e) {
            error.Unsupported => return error.SkipZigTest,
            else => return e,
        };
        defer ref_caller_c.deinit(allocator);
        var wim_caller_c = try isel.compileFunctionWimmerX86(allocator, &wim_caller);
        defer wim_caller_c.deinit(allocator);

        const ref = linkRunFloat(io, allocator, &ref_caller_c, &callee_c, fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = linkRunFloat(io, allocator, &wim_caller_c, &callee_c, fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

/// A loop that carries a `lanes`-wide vector accumulator ACROSS block edges (entry->header and the
/// body->header back-edge), with `count` simultaneously-live vectors in the body creating register
/// pressure. The carried vector is used at the END of the body, so it is live across the whole
/// pressured region and the shared allocator SPLITS it, making the block-param/back-edge shuffle a
/// class-1 (xmm) EDGE move whose width is picked from the value's IR type (movups for 128-bit, vmovups
/// for the 256-bit ymm). A wrong-width edge move drops the upper lanes and diverges from the reference.
/// The trip count is a constant so no gpr argument is needed (the float stub loads only xmm args).
fn buildVecLoopPressure(allocator: std.mem.Allocator, comptime lanes: u32, comptime count: usize) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const vt = try func.types.intern(.{ .vector = .{ .len = lanes, .elem = t } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    // entry: seed vacc = <0.1, ...>, i = 0, n = 5; jump header.
    const seed_c = try func.appendInst(entry, t, .{ .fconst = 0.1 });
    var seed_fields: [lanes]Value = undefined;
    for (0..lanes) |k| seed_fields[k] = seed_c;
    const vseed = try func.appendInst(entry, vt, .{ .struct_new = .{ .fields = try func.internValueList(&seed_fields) } });
    const iter0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const limit0 = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.setJump(entry, header, &.{ vseed, iter0, limit0 });

    // header(vacc, i, n): if i < n -> body else exit.
    const h_v = try func.appendBlockParam(header, vt);
    const h_i = try func.appendBlockParam(header, i32_t);
    const h_n = try func.appendBlockParam(header, i32_t);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = h_i, .rhs = h_n } });
    try func.appendIf(header, cond, .{ .target = body, .args = &.{ h_v, h_i, h_n } }, .{ .target = exit, .args = &.{h_v} });

    // body(vacc, i, n): build `count` live vectors, reduce them, ADD the carried vacc LAST (so vacc is
    // live across the whole pressured body), i+1, jump header.
    const b_v = try func.appendBlockParam(body, vt);
    const b_i = try func.appendBlockParam(body, i32_t);
    const b_n = try func.appendBlockParam(body, i32_t);
    var vs: [count]Value = undefined;
    for (0..count) |j| {
        const c = try func.appendInst(body, t, .{ .fconst = @as(f64, @floatFromInt(j)) + 0.1 });
        var fields: [lanes]Value = undefined;
        for (0..lanes) |k| fields[k] = c;
        vs[j] = try func.appendInst(body, vt, .{ .struct_new = .{ .fields = try func.internValueList(&fields) } });
    }
    var red = vs[0];
    for (1..count) |j| red = try func.appendInst(body, vt, .{ .arith = .{ .op = .add, .lhs = red, .rhs = vs[j] } });
    const vnext = try func.appendInst(body, vt, .{ .arith = .{ .op = .add, .lhs = red, .rhs = b_v } });
    const inext = try func.appendArithImm(body, i32_t, .add, b_i, 1);
    try func.setJump(body, header, &.{ vnext, inext, b_n });

    // exit(vacc): reduce the lanes to a scalar and return.
    const e_v = try func.appendBlockParam(exit, vt);
    var e: [lanes]Value = undefined;
    for (0..lanes) |k| e[k] = try func.appendInst(exit, t, .{ .extract = .{ .aggregate = e_v, .index = @intCast(k) } });
    var s = e[0];
    for (1..lanes) |k| s = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = e[k] } });
    func.setTerminator(exit, .{ .ret = s });
    return func;
}

fn build128LoopPressure(allocator: std.mem.Allocator) anyerror!Function {
    return buildVecLoopPressure(allocator, 4, 16);
}
fn build256LoopPressure(allocator: std.mem.Allocator) anyerror!Function {
    return buildVecLoopPressure(allocator, 8, 16);
}

test "wimmer-x86: a wide vector carried across loop edges under pressure uses width-correct edge moves" {
    // The carried vector crosses the entry and back edges as a class-1 edge move; 128-bit (movups) and
    // 256-bit (vmovups) must each move the full width or the reduced result diverges.
    try expectFloatMatch(std.testing.io, build128LoopPressure, &.{&.{}});
    try expectFloatMatch(std.testing.io, build256LoopPressure, &.{&.{}});
}
