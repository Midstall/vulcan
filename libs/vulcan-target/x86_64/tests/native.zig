//! Native runner: execute the shared cases.zig in-process by mapping the compiled code
//! into W^X memory and calling it directly. Valid only when the host is x86-64. On any
//! other host every case skips, since x86 machine code cannot run natively there. On an
//! x86-64 host this also exercises the JIT (coherent_jit) path.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const link = @import("../link.zig");
const mm = @import("vulcan-opt").microarch;

test "module disasm: linked functions get labels and a resolved, named call" {
    const a = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var g = ir.function.Function.init(a);
    defer g.deinit();
    const gi = try g.types.intern(i32k);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gi);
    const gm = try g.appendArithImm(gb, gi, .mul, gx, 3);
    g.setTerminator(gb, .{ .ret = gm });

    var f = ir.function.Function.init(a);
    defer f.deinit();
    const fi = try f.types.intern(i32k);
    const fb = try f.appendBlock();
    const fa = try f.appendBlockParam(fb, fi);
    const fbp = try f.appendBlockParam(fb, fi);
    const called = try f.appendCall(fb, fi, "helper", &.{fa});
    const fsum = try f.appendInst(fb, fi, .{ .arith = .{ .op = .add, .lhs = called, .rhs = fbp } });
    f.setTerminator(fb, .{ .ret = fsum });

    var module = link.Module{};
    defer module.deinit(a);
    try module.addFunction(a, "helper", &g);
    try module.addFunction(a, "main", &f);
    var linked = try link.compileModule(a, &module);
    defer linked.deinit(a);

    const syms = try a.alloc(disasm.Sym, linked.symbols.len);
    defer a.free(syms);
    for (linked.symbols, 0..) |s, i| syms[i] = .{ .name = s.name, .offset = s.offset };
    const text = try disasm.formatModule(a, linked.code, syms);
    defer a.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "helper:\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "main:\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "  <helper>") != null); // resolved call
}

test "x86-64 cases run natively in-process (skips off x86-64)" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.native);
}

test "codegen+disasm round-trip: integer add" {
    // Compile x86-64 and assert the disassembled listing: checks instruction selection and
    // register allocation at the instruction level, and runs on any host (no execution).
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const s = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(e, .{ .ret = s });

    const code = try isel.selectFunction(a, &func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);
    // The shared Wimmer allocator (SP2 production flip) keeps the two params in their ABI registers
    // and the result in rax, so the add is `x += y` in place with no shuffle moves.
    try std.testing.expectEqualStrings(
        \\0000: mov rax, rdi
        \\0003: add eax, esi
        \\0005: ret
        \\
    , text);
}

test "codegen+disasm round-trip: control flow (max via if/else)" {
    // Exercises the disassembler on real setcc/movzx/test and jcc/jmp rel32 branches.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const m = try func.appendBlock();
    const r = try func.appendBlockParam(m, i32_t);
    const c = try func.appendInst(e, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = y } });
    try func.appendIf(e, c, .{ .target = m, .args = &.{x} }, .{ .target = m, .args = &.{y} });
    func.setTerminator(m, .{ .ret = r });

    const code = try isel.selectFunction(a, &func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);
    // The shared Wimmer allocator (SP2 production flip) SPLITS the two critical edges e->m, so each
    // arm reaches the join `m` through its own forwarding block (the extra `jmp`s), where the phi
    // move places the taken value in rax before falling into the return.
    try std.testing.expectEqualStrings(
        \\0000: cmp edi, esi
        \\0002: setg al
        \\0006: movzx rax, al
        \\000a: test rax, rax
        \\000d: jne .+5
        \\0013: jmp .+14
        \\0018: jmp .+1
        \\001d: ret
        \\001e: mov rax, rdi
        \\0021: jmp .-9
        \\0026: mov rax, rsi
        \\0029: jmp .-17
        \\
    , text);
}

test "x86_64 selectFunctionForModel with an inert-fusion model is byte-identical to selectFunction" {
    // Reuses the icmp/if builder above: it is exactly the shape the cmp_branch fold (B2) targets.
    // Now that the fold reads `caps.fuse_cmp_branch` (see the cmp_branch fold tests below),
    // cascadelake-sp's real fusion table (cmp_branch on) DIVERGES from plain `selectFunction` on
    // this shape - that divergence is covered by the fold's own tests. This test keeps the
    // narrower guard that still matters: a model whose fusion table is EMPTY (caps all read
    // false) must stay byte-identical to plain, covering the off end of `capsForModel`.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const m = try func.appendBlock();
    const r = try func.appendBlockParam(m, i32_t);
    const c = try func.appendInst(e, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = y } });
    try func.appendIf(e, c, .{ .target = m, .args = &.{x} }, .{ .target = m, .args = &.{y} });
    func.setTerminator(m, .{ .ret = r });

    const plain = try isel.selectFunction(a, &func);
    defer a.free(plain);

    var inert = mm.modelFor(.@"cascadelake-sp").*; // a shallow copy: same model, empty fusion table
    inert.fusion = &.{};
    const tuned_inert = try isel.selectFunctionForModel(a, &func, &inert);
    defer a.free(tuned_inert);
    try std.testing.expectEqualSlices(u8, plain, tuned_inert);
}

// --- B2: cmp_branch fold (fuse `cmp; setcc; ...; test; jcc` into `cmp; jcc`) --------------------

const CmpOp = ir.function.CmpOp;

/// Build `f(a, b) -> i64 { if (a <op> b) return 100 else return 200 }`, with `a`/`b` typed at
/// `bits` width and `signed` signedness. This is exactly the shape `fusesIntoNextIf` targets (a
/// single-use icmp immediately followed by the `if` that tests it), so compiling the SAME
/// function plain vs cascadelake-sp-tuned exercises the unfused vs fused path on one IR.
fn buildCmpBranchIf(allocator: std.mem.Allocator, op: CmpOp, signed: bool, bits: u16) !ir.function.Function {
    var func = ir.function.Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = if (signed) .signed else .unsigned, .bits = bits } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = op, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    const yes = try func.appendInst(then_b, i64_t, .{ .iconst = 100 });
    func.setTerminator(then_b, .{ .ret = yes });
    const no = try func.appendInst(else_b, i64_t, .{ .iconst = 200 });
    func.setTerminator(else_b, .{ .ret = no });
    return func;
}

/// The golden model for `a <op> b` at `bits`/`signed`, independent of any codegen: truncates both
/// operands to `bits` (masking away anything a narrower IR type would never see) and interprets
/// them per `signed` before applying `op`. Backs the differential test's expected value so a bug
/// that made BOTH the plain and the fused path agreeably wrong would still be caught (a plain
/// fused-vs-plain comparison alone could not).
fn evalCmp(op: CmpOp, signed: bool, bits: u16, a: i64, b: i64) bool {
    const mask: u64 = if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1;
    const au: u64 = @as(u64, @bitCast(a)) & mask;
    const bu: u64 = @as(u64, @bitCast(b)) & mask;
    if (!signed) return cmpGeneric(u64, op, au, bu);
    // Sign-extend the masked `bits`-wide value back to a full i64 for a signed comparison.
    if (bits >= 64) return cmpGeneric(i64, op, a, b);
    const shift: u6 = @intCast(64 - bits);
    const ai: i64 = @as(i64, @bitCast(au << shift)) >> shift;
    const bi: i64 = @as(i64, @bitCast(bu << shift)) >> shift;
    return cmpGeneric(i64, op, ai, bi);
}

fn cmpGeneric(comptime T: type, op: CmpOp, a: T, b: T) bool {
    return switch (op) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
    };
}

/// Run `code` on `backend`, returning its result byte, or null when that backend is unavailable
/// (native off an x86-64 host, or qemu not on PATH).
fn runCmpBranchOn(io: std.Io, allocator: std.mem.Allocator, code: []const u8, args: []const i64, backend: harness.Backend) !?u8 {
    return harness.runCodeInt(io, allocator, code, args, backend) catch |e| switch (e) {
        error.SkipZigTest => null,
        else => return e,
    };
}

/// Like `runCmpBranchOn`, but returns the FULL i64 result (no mod-256 truncation): used by the
/// reduction-loop differential, whose sums grow past 255.
fn runCmpBranchOnFull(io: std.Io, allocator: std.mem.Allocator, code: []const u8, args: []const i64, backend: harness.Backend) !?i64 {
    return harness.runCodeIntFull(io, allocator, code, args, backend) catch |e| switch (e) {
        error.SkipZigTest => null,
        else => return e,
    };
}

/// Compile `func` PLAIN and tuned to cascadelake-sp (cmp_branch on), then run BOTH builds with
/// `args` under every available backend (native in-process on an x86-64 host, qemu-x86_64
/// elsewhere), asserting both equal `want` (a 100/200 low byte, per `buildCmpBranchIf`). Comparing
/// each build against the independently-computed `want` (not just against each other) also catches
/// a shared miscompile that happened to agree. Skips only if neither backend is available.
fn expectCmpBranchFused(io: std.Io, allocator: std.mem.Allocator, func: *const ir.function.Function, args: []const i64, want: u8) !void {
    const plain = try isel.selectFunction(allocator, func);
    defer allocator.free(plain);
    const tuned = try isel.selectFunctionForModel(allocator, func, mm.modelFor(.@"cascadelake-sp"));
    defer allocator.free(tuned);

    var ran = false;
    for ([_]harness.Backend{ harness.native, harness.qemu }) |backend| {
        const p = try runCmpBranchOn(io, allocator, plain, args, backend);
        const t = try runCmpBranchOn(io, allocator, tuned, args, backend);
        if (p) |pv| {
            try std.testing.expectEqual(want, pv);
            ran = true;
        }
        if (t) |tv| {
            try std.testing.expectEqual(want, tv);
            ran = true;
        }
    }
    if (!ran) return error.SkipZigTest;
}

test "x86_64 cmp_branch fold: if(icmp) is execution-equivalent to plain across ops/signs/widths" {
    const a = std.testing.allocator;
    const ops = [_]CmpOp{ .lt, .le, .gt, .ge, .eq, .ne };
    // Equal, ordinary less/greater, negative-signed, and an unsigned-wraparound pair (-1 reads as
    // all-ones: huge unsigned, -1 signed).
    const pairs = [_][2]i64{
        .{ 3, 7 },
        .{ 7, 3 },
        .{ 5, 5 },
        .{ -5, 2 },
        .{ 2, -5 },
        .{ -1, 1 },
    };
    for (ops) |op| {
        for ([_]bool{ true, false }) |signed| {
            for ([_]u16{ 32, 64 }) |bits| {
                var func = try buildCmpBranchIf(a, op, signed, bits);
                defer func.deinit();
                for (pairs) |p| {
                    const want: u8 = if (evalCmp(op, signed, bits, p[0], p[1])) 100 else 200;
                    try expectCmpBranchFused(std.testing.io, a, &func, &.{ p[0], p[1] }, want);
                }
                if (bits == 64) {
                    // High-bit i64 pair: a 32-bit `cmp` (mismeasuring only the low 32 bits) would
                    // see 0 vs 1 and get this backwards. Only a width-aware 64-bit compare is
                    // correct here, on both the fused and unfused paths.
                    const hi_a: i64 = 0x1_0000_0000;
                    const want: u8 = if (evalCmp(op, signed, bits, hi_a, 1)) 100 else 200;
                    try expectCmpBranchFused(std.testing.io, a, &func, &.{ hi_a, 1 }, want);
                }
            }
        }
    }
}

test "x86_64 cmp_branch fold: tuned build emits cmp+jcc with no setcc, plain keeps setcc+test" {
    // The fail-first signal: without the fold reading `caps.fuse_cmp_branch`, a tuned build of
    // this exact shape stays byte-identical to plain (still setcc+test+jne), so this structural
    // check is what actually distinguishes "fold implemented" from "fold absent" (the execution
    // differential above passes trivially either way, since both paths compute the same result).
    const a = std.testing.allocator;
    var func = try buildCmpBranchIf(a, .gt, true, 32);
    defer func.deinit();

    const plain = try isel.selectFunction(a, &func);
    defer a.free(plain);
    const plain_text = try disasm.format(a, plain);
    defer a.free(plain_text);
    try std.testing.expect(std.mem.indexOf(u8, plain_text, "setg") != null); // baseline: boolean materialized
    try std.testing.expect(std.mem.indexOf(u8, plain_text, "test") != null); // baseline: boolean re-tested

    const tuned = try isel.selectFunctionForModel(a, &func, mm.modelFor(.@"cascadelake-sp"));
    defer a.free(tuned);
    const tuned_text = try disasm.format(a, tuned);
    defer a.free(tuned_text);
    try std.testing.expect(std.mem.indexOf(u8, tuned_text, "cmp ") != null); // the compare survives
    try std.testing.expect(std.mem.indexOf(u8, tuned_text, "jg ") != null); // fused conditional branch (signed gt)
    try std.testing.expect(std.mem.indexOf(u8, tuned_text, "setg") == null); // no boolean materialized
    try std.testing.expect(std.mem.indexOf(u8, tuned_text, "movzx") == null); // no zero-extend of a boolean
    try std.testing.expect(std.mem.indexOf(u8, tuned_text, "test") == null); // no re-test of a boolean
}

test "x86_64 cmp_branch fold: a multi-use icmp does NOT fuse (boolean still materialized, correct result)" {
    // The icmp result feeds BOTH a select and the if condition, so it is not single-use: fusion
    // must be declined (the select would otherwise read a boolean that was never produced).
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const av = try func.appendBlockParam(entry, t);
    const bv = try func.appendBlockParam(entry, t);
    const xtb = try func.appendBlockParam(tb, t);
    const xeb = try func.appendBlockParam(eb, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = av, .rhs = bv } });
    const m = try func.appendInst(entry, t, .{ .select = .{ .cond = c, .then = av, .@"else" = bv } }); // min, uses c
    try func.appendIf(entry, c, .{ .target = tb, .args = &.{m} }, .{ .target = eb, .args = &.{m} });
    const inc = try func.appendArithImm(tb, t, .add, xtb, 1);
    func.setTerminator(tb, .{ .ret = inc });
    func.setTerminator(eb, .{ .ret = xeb });

    const tuned = try isel.selectFunctionForModel(a, &func, mm.modelFor(.@"cascadelake-sp"));
    defer a.free(tuned);
    const text = try disasm.format(a, tuned);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "setl") != null); // multi-use: fusion declined

    try harness.expectRun(std.testing.io, a, &func, &.{ 3, 7 }, 4, harness.qemu); // c true: m=min=3, tb -> 3+1
    try harness.expectRun(std.testing.io, a, &func, &.{ 7, 3 }, 3, harness.qemu); // c false: m=min=3, eb -> 3
}

test "x86_64 cmp_branch fold: an icmp not immediately before the if does NOT fuse" {
    // An instruction sits between the icmp and the if, so the icmp is not the immediately
    // preceding instruction: fusion is declined and the standard setcc;test path runs.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const av = try func.appendBlockParam(entry, t);
    const bv = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = av, .rhs = bv } });
    const sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = av, .rhs = bv } }); // intervening
    try func.appendIf(entry, c, .{ .target = tb, .args = &.{sum} }, .{ .target = eb, .args = &.{sum} });
    const xtb = try func.appendBlockParam(tb, t);
    const xeb = try func.appendBlockParam(eb, t);
    func.setTerminator(tb, .{ .ret = xtb });
    func.setTerminator(eb, .{ .ret = xeb });

    const tuned = try isel.selectFunctionForModel(a, &func, mm.modelFor(.@"cascadelake-sp"));
    defer a.free(tuned);
    const text = try disasm.format(a, tuned);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "setg") != null); // not immediately preceding -> no fusion

    try harness.expectRun(std.testing.io, a, &func, &.{ 7, 3 }, 10, harness.qemu); // both edges return a+b here
    try harness.expectRun(std.testing.io, a, &func, &.{ 3, 7 }, 10, harness.qemu);
}

// --- B3: arith_branch fold (fuse a flag-setting `add`/`sub`/`bit_and` into its consumer branch,
// eliding the `cmp` the cmp_branch fold would otherwise still emit) -----------------------------

const ArithOp = ir.function.BinOp;

/// Build `f(a, b) -> i64 { s = a <op> b; if (s <cmp> 0) return 100 else return 200 }`, register-
/// form arith at i64 width. This is exactly the shape `fusesArithIntoBranch` targets: an arith
/// immediately followed by the single-use icmp-against-0 that `fusesIntoNextIf` already fuses,
/// immediately followed by the if that tests it.
fn buildArithBranchIfReg(allocator: std.mem.Allocator, op: ArithOp, cmp: CmpOp) !ir.function.Function {
    var func = ir.function.Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const av = try func.appendBlockParam(entry, i64_t);
    const bv = try func.appendBlockParam(entry, i64_t);
    // `zero` is created BEFORE the arith so the arith sits immediately before the icmp (adjacency
    // is what `fusesArithIntoBranch` requires); the icmp's rhs may reference an `iconst` defined
    // anywhere earlier in the block.
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const s = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = op, .lhs = av, .rhs = bv } });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = cmp, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    const yes = try func.appendInst(then_b, i64_t, .{ .iconst = 100 });
    func.setTerminator(then_b, .{ .ret = yes });
    const no = try func.appendInst(else_b, i64_t, .{ .iconst = 200 });
    func.setTerminator(else_b, .{ .ret = no });
    return func;
}

/// Build `f(n) -> i64 { s = n <op> imm; if (s <cmp> 0) return 100 else return 200 }`, immediate-
/// form (`arith_imm`) arith at i64 width. `op` is `add` or `sub` only, per `fusesArithIntoBranch`'s
/// immediate scope (a `bit_and` immediate stays on the plain path).
fn buildArithBranchIfImm(allocator: std.mem.Allocator, op: ArithOp, imm: i64, cmp: CmpOp) !ir.function.Function {
    var func = ir.function.Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    // `zero` is created BEFORE the arith, for the same adjacency reason as `buildArithBranchIfReg`.
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const s = try func.appendArithImm(entry, i64_t, op, n, imm);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = cmp, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    const yes = try func.appendInst(then_b, i64_t, .{ .iconst = 100 });
    func.setTerminator(then_b, .{ .ret = yes });
    const no = try func.appendInst(else_b, i64_t, .{ .iconst = 200 });
    func.setTerminator(else_b, .{ .ret = no });
    return func;
}

/// The golden model for `a <op> b <cmp> 0`, independent of any codegen: wrapping i64 arithmetic
/// (matching two's-complement hardware semantics, since Zig's plain `+`/`-` trap on overflow)
/// then an eq/ne test against 0. Backs the differential tests' expected value so a bug that made
/// BOTH the plain and the fused path agreeably wrong would still be caught.
fn evalArithBranch(op: ArithOp, cmp: CmpOp, a: i64, b: i64) bool {
    const s: i64 = switch (op) {
        .add => a +% b,
        .sub => a -% b,
        .bit_and => a & b,
        else => unreachable, // fusesArithIntoBranch admits only add/sub/bit_and
    };
    return switch (cmp) {
        .eq => s == 0,
        .ne => s != 0,
        else => unreachable, // fusesArithIntoBranch admits only eq/ne
    };
}

/// Compile `func` PLAIN and tuned to cascadelake-sp (cmp_branch AND arith_branch on), then run
/// BOTH builds with `args` under every available backend, asserting both equal `want` (a 100/200
/// low byte, per `buildArithBranchIfReg`/`buildArithBranchIfImm`). Comparing each build against
/// the independently-computed `want` (not just against each other) catches a shared miscompile
/// that happened to agree.
fn expectArithBranchFused(io: std.Io, allocator: std.mem.Allocator, func: *const ir.function.Function, args: []const i64, want: u8) !void {
    const plain = try isel.selectFunction(allocator, func);
    defer allocator.free(plain);
    const tuned = try isel.selectFunctionForModel(allocator, func, mm.modelFor(.@"cascadelake-sp"));
    defer allocator.free(tuned);

    var ran = false;
    for ([_]harness.Backend{ harness.native, harness.qemu }) |backend| {
        const p = try runCmpBranchOn(io, allocator, plain, args, backend);
        const t = try runCmpBranchOn(io, allocator, tuned, args, backend);
        if (p) |pv| {
            try std.testing.expectEqual(want, pv);
            ran = true;
        }
        if (t) |tv| {
            try std.testing.expectEqual(want, tv);
            ran = true;
        }
    }
    if (!ran) return error.SkipZigTest;
}

test "x86_64 arith_branch fold: decrement-and-branch (n-1 != 0) is execution-equivalent to plain across a sweep" {
    const a = std.testing.allocator;
    var func = try buildArithBranchIfImm(a, .sub, 1, .ne);
    defer func.deinit();
    // n=1 -> 200 (s=0, ne false), n=5 -> 100 (s=4), n=0 -> 100 (s=-1), and an n whose decrement's
    // LOW 32 bits are zero but whose full 64-bit value is not (s = 0x1_0000_0000): only a width-
    // aware flag test gets this right, on both the fused and unfused paths.
    const ns = [_]i64{ 1, 5, 0, 0x1_0000_0001 };
    for (ns) |n| {
        const want: u8 = if (evalArithBranch(.sub, .ne, n, 1)) 100 else 200;
        try expectArithBranchFused(std.testing.io, a, &func, &.{n}, want);
    }
}

test "x86_64 arith_branch fold: add register (== 0) is execution-equivalent to plain across a sweep" {
    const a = std.testing.allocator;
    var func = try buildArithBranchIfReg(a, .add, .eq);
    defer func.deinit();
    const pairs = [_][2]i64{
        .{ 3, -3 }, // sum 0 -> eq true
        .{ 3, 4 }, // sum 7 -> eq false
        .{ 0x1_0000_0000, -0x1_0000_0000 }, // sum 0 at 64-bit width -> eq true
        .{ 0x1_0000_0000, 0 }, // sum 0x1_0000_0000: low 32 bits zero, but nonzero at 64-bit -> eq false
    };
    for (pairs) |p| {
        const want: u8 = if (evalArithBranch(.add, .eq, p[0], p[1])) 100 else 200;
        try expectArithBranchFused(std.testing.io, a, &func, &.{ p[0], p[1] }, want);
    }
}

test "x86_64 arith_branch fold: bit_and register (!= 0) is execution-equivalent to plain across a sweep" {
    const a = std.testing.allocator;
    var func = try buildArithBranchIfReg(a, .bit_and, .ne);
    defer func.deinit();
    const pairs = [_][2]i64{
        .{ 0xFF, 0xFF00 }, // and 0 -> ne false
        .{ 0xFF, 0x0F }, // and 0x0F -> ne true
        .{ 0x1_0000_0000, 0x1_0000_0000 }, // and = 0x1_0000_0000: zero at 32-bit, nonzero at 64-bit -> ne true
    };
    for (pairs) |p| {
        const want: u8 = if (evalArithBranch(.bit_and, .ne, p[0], p[1])) 100 else 200;
        try expectArithBranchFused(std.testing.io, a, &func, &.{ p[0], p[1] }, want);
    }
}

/// Whether `text` (a `disasm.format` listing) has a line containing `first` immediately followed
/// by a line containing `second` (adjacency, not just co-occurrence). Backs the structural checks
/// below: the arith_branch fold's whole point is that NOTHING (no `cmp`, no reload) sits between
/// the arith and the `jcc`.
fn hasAdjacentLines(text: []const u8, first: []const u8, second: []const u8) bool {
    var it = std.mem.splitScalar(u8, text, '\n');
    var prev: ?[]const u8 = null;
    while (it.next()) |line| {
        if (prev) |p| {
            if (std.mem.indexOf(u8, p, first) != null and std.mem.indexOf(u8, line, second) != null) return true;
        }
        prev = line;
    }
    return false;
}

test "x86_64 arith_branch fold: tuned build emits sub immediately followed by jne, no cmp at all" {
    // The fail-first signal: B2 (cmp_branch) alone still emits a separate `cmp` before the `jne`
    // (it only elides the icmp's own setcc/test, not the compare itself). This structural check
    // is what actually distinguishes "arith_branch implemented" from "cmp_branch only" (the
    // execution differentials above pass trivially either way, since both paths compute the same
    // result).
    const a = std.testing.allocator;
    var func = try buildArithBranchIfImm(a, .sub, 1, .ne);
    defer func.deinit();

    const plain = try isel.selectFunction(a, &func);
    defer a.free(plain);
    const plain_text = try disasm.format(a, plain);
    defer a.free(plain_text);
    try std.testing.expect(std.mem.indexOf(u8, plain_text, "cmp ") != null); // baseline: separate compare survives

    const tuned = try isel.selectFunctionForModel(a, &func, mm.modelFor(.@"cascadelake-sp"));
    defer a.free(tuned);
    const tuned_text = try disasm.format(a, tuned);
    defer a.free(tuned_text);
    try std.testing.expect(std.mem.indexOf(u8, tuned_text, "cmp ") == null); // no separate compare anywhere
    try std.testing.expect(hasAdjacentLines(tuned_text, "sub ", "jne ")); // sub immediately followed by jne
}

test "x86_64 arith_branch fold: a multi-use arith result does NOT fuse (cmp survives, correct result)" {
    // The sub result feeds BOTH the icmp and a second, independent use, so it is not single-use:
    // the arith_branch fold must decline (folding away the separate cmp would be fine either way
    // here, since the arith still runs, but this proves `fusesArithIntoBranch`'s single-use gate
    // is actually wired to the same predicate `emitIf` reads, not merely present on paper).
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const s = try func.appendArithImm(entry, i64_t, .sub, n, 1); // arith immediately before the icmp
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ne, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = tb, .args = &.{s} }, .{ .target = eb, .args = &.{s} }); // s used again
    const xtb = try func.appendBlockParam(tb, i64_t);
    const xeb = try func.appendBlockParam(eb, i64_t);
    func.setTerminator(tb, .{ .ret = xtb });
    func.setTerminator(eb, .{ .ret = xeb });

    const tuned = try isel.selectFunctionForModel(a, &func, mm.modelFor(.@"cascadelake-sp"));
    defer a.free(tuned);
    const text = try disasm.format(a, tuned);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp ") != null); // multi-use: arith_branch declined

    try harness.expectRun(std.testing.io, a, &func, &.{1}, 0, harness.qemu); // s=0, else edge carries s=0
    try harness.expectRun(std.testing.io, a, &func, &.{5}, 4, harness.qemu); // s=4, then edge carries s=4
}

// ---------------------------------------------------------------------------
// Full-width runners: `...Full` carries the WHOLE result via stdout (no mod-256 limit),
// unlike the plain runners above whose result is only the exit code's low byte.
// ---------------------------------------------------------------------------

test "x86_64 expectRunFull asserts the full 64-bit result, not just the low byte" {
    const a = std.testing.allocator;
    var f = ir.function.Function.init(a);
    defer f.deinit();
    const i64_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try f.appendBlock();
    const c = try f.appendInst(b, i64_t, .{ .iconst = 0x1_0000_0100 });
    f.setTerminator(b, .{ .ret = c });

    // 0x1_0000_0100 and 0x0000_0100 share the low byte (0x00) but are very different i64
    // values: the OLD low-byte runner cannot tell them apart, the Full runner must.
    try std.testing.expectEqual(@as(u8, 0x00), @as(u8, @truncate(@as(u64, @bitCast(@as(i64, 0x1_0000_0100))))));
    try std.testing.expectEqual(@as(u8, 0x00), @as(u8, @truncate(@as(u64, @bitCast(@as(i64, 0x0000_0100))))));

    var ran = false;
    for ([_]harness.Backend{ harness.native, harness.qemu }) |backend| {
        const byte = harness.runFunc(std.testing.io, a, &f, &.{}, backend) catch |e| switch (e) {
            error.SkipZigTest => continue,
            else => return e,
        };
        try std.testing.expectEqual(@as(u8, 0x00), byte); // the old runner's low byte is ambiguous
        try harness.expectRunFull(std.testing.io, a, &f, &.{}, 0x1_0000_0100, backend); // the Full runner is not
        ran = true;
    }
    if (!ran) return error.SkipZigTest; // no execution backend available (e.g. qemu absent), nothing to assert
}

test "x86_64 expectRunFloatFull asserts the exact f32 bits, not just the low byte" {
    const a = std.testing.allocator;
    // 0x3F80_0100 and 0x0000_0100 (as f32 bit patterns) share the low byte (0x00) but are very
    // different f32 values.
    const full: f32 = @bitCast(@as(u32, 0x3F80_0100));
    const collide: f32 = @bitCast(@as(u32, 0x0000_0100));
    try std.testing.expectEqual(@as(u8, 0x00), @as(u8, @truncate(@as(u32, @bitCast(full)))));
    try std.testing.expectEqual(@as(u8, 0x00), @as(u8, @truncate(@as(u32, @bitCast(collide)))));
    try std.testing.expect(full != collide);

    var f = ir.function.Function.init(a);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = x });

    // The OLD low-byte runner cannot tell `full` from `collide`: both truncate to 0x00.
    try std.testing.expectEqual(@as(u8, 0x00), try harness.runFloatFunc(std.testing.io, a, &f, &.{full}, harness.qemu));
    // The new Full runner asserts the exact bits.
    try harness.expectRunFloatFull(std.testing.io, a, &f, &.{full}, full, harness.qemu);
}

test "x86_64 expectRunDoubleFull asserts the exact f64 bits, not just the low byte" {
    const a = std.testing.allocator;
    // 0x3FF0_0000_0000_0100 and 0x0000_0000_0000_0100 (as f64 bit patterns) share the low byte
    // (0x00) but are very different f64 values.
    const full: f64 = @bitCast(@as(u64, 0x3FF0_0000_0000_0100));
    const collide: f64 = @bitCast(@as(u64, 0x0000_0000_0000_0100));
    try std.testing.expectEqual(@as(u8, 0x00), @as(u8, @truncate(@as(u64, @bitCast(full)))));
    try std.testing.expectEqual(@as(u8, 0x00), @as(u8, @truncate(@as(u64, @bitCast(collide)))));
    try std.testing.expect(full != collide);

    var f = ir.function.Function.init(a);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f64 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = x });

    // The OLD low-byte runner cannot tell `full` from `collide`: both truncate to 0x00.
    try std.testing.expectEqual(@as(u8, 0x00), try harness.runDoubleFunc(std.testing.io, a, &f, &.{full}, harness.qemu));
    // The new Full runner asserts the exact bits.
    try harness.expectRunDoubleFull(std.testing.io, a, &f, &.{full}, full, harness.qemu);
}

// ---------------------------------------------------------------------------
// Whole-pipeline differential: optimize(cascadelake-sp) -> x86_64 isel vs. the plain compile.
// ---------------------------------------------------------------------------

/// Build `f(n) -> i64 { s = 0; i = 0; while (i < n) { s += i * 3; i += 1 } ret s }`: a scalar
/// counted reduction loop (entry -> header(i,s) -> body -> back-edge to header, exit carries s).
/// Integer-only (no floats/vectors), so the vectorizer (arch-gated off for x86 anyway) has nothing
/// to do and `mm.optimize` exercises only the scalar passes (splitunroll/unroll/schedule). This is
/// exactly the canonical shape `splitunroll.recognize` requires: a single-block straight-line body
/// that is the loop's only latch, a header test `i < n` against a loop-invariant bound (`n` stays an
/// entry-scope value, never threaded through the header params, so it is never "defined in loop"),
/// a constant positive induction step, and a single accumulator (`s`) used only by its own update.
fn buildReductionLoop(allocator: std.mem.Allocator) !ir.function.Function {
    var func = ir.function.Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
    const iv0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const sv0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, sv0 });

    const h_i = try func.appendBlockParam(header, i64_t);
    const h_s = try func.appendBlockParam(header, i64_t);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = h_i, .rhs = n } });
    try func.appendIf(header, cond, .{ .target = body, .args = &.{ h_i, h_s } }, .{ .target = exit, .args = &.{h_s} });

    const b_i = try func.appendBlockParam(body, i64_t);
    const b_s = try func.appendBlockParam(body, i64_t);
    const three = try func.appendInst(body, i64_t, .{ .iconst = 3 });
    const t = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .mul, .lhs = b_i, .rhs = three } });
    const next_s = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = b_s, .rhs = t } });
    const next_i = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    try func.setJump(body, header, &.{ next_i, next_s });

    const e_s = try func.appendBlockParam(exit, i64_t);
    func.setTerminator(exit, .{ .ret = e_s });
    return func;
}

/// The golden model for `buildReductionLoop`, independent of any codegen: wrapping i64 arithmetic
/// (matching two's-complement hardware semantics), since Zig's plain `+`/`*` trap on overflow.
fn expectedReductionSum(n: i64) i64 {
    var s: i64 = 0;
    var i: i64 = 0;
    while (i < n) : (i += 1) s +%= i *% 3;
    return s;
}

test "x86_64: optimize(cascadelake) then selectFunctionForModel is execution-equivalent to plain across a counted reduction-loop sweep" {
    const a = std.testing.allocator;
    var orig = try buildReductionLoop(a);
    defer orig.deinit();
    var tuned = try buildReductionLoop(a);
    defer tuned.deinit();

    // The tuned copy actually gets rewritten: a 4-issue OoO model with a 4-inst body gives
    // splitunroll factor (4*3)/4 = 3 (>= 2), so this is not a no-op run of `optimize`.
    const changed = try mm.optimize(a, &tuned, mm.modelFor(.@"cascadelake-sp"));
    try std.testing.expect(changed);

    var diags = try ir.verify.verify(a, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    // Compile PLAIN (untouched original) and TUNED (optimized IR, model-aware isel), then run both
    // under every available backend across the sweep, each checked EXACTLY (via the Full runner,
    // not mod 256: the sums grow past 255) against the independently computed golden sum (not
    // just against each other), so a shared miscompile is still caught.
    const plain = try isel.selectFunction(a, &orig);
    defer a.free(plain);
    const tuned_code = try isel.selectFunctionForModel(a, &tuned, mm.modelFor(.@"cascadelake-sp"));
    defer a.free(tuned_code);

    const ns = [_]i64{ 0, 1, 5, 17, 64, 100 };
    for (ns) |n| {
        const want = expectedReductionSum(n);
        var ran = false;
        for ([_]harness.Backend{ harness.native, harness.qemu }) |backend| {
            const p = try runCmpBranchOnFull(std.testing.io, a, plain, &.{n}, backend);
            const t = try runCmpBranchOnFull(std.testing.io, a, tuned_code, &.{n}, backend);
            if (p) |pv| {
                try std.testing.expectEqual(want, pv);
                ran = true;
            }
            if (t) |tv| {
                try std.testing.expectEqual(want, tv);
                ran = true;
            }
        }
        if (!ran) return error.SkipZigTest;
    }
}
