//! Native backend runner: maps the compiled function into W^X memory and calls
//! it in-process. Only valid when the host is RISC-V (host == target), skips
//! otherwise. Unlike the emulator backends this does not use the UART firmware
//! (MMIO would fault natively), it calls the function and reads the return value.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const harness = @import("harness.zig");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const link = @import("../link.zig");

const Function = ir.function.Function;

test "module disasm: linked functions get labels and a resolved, named call" {
    // A two-function module (main calls helper) linked so the call relocation resolves to a
    // real `jal`. formatModule labels each function and annotates the resolved call with the
    // callee name. NB: riscv64 link.Symbol.offset is a WORD index (unlike aarch64's bytes).
    const a = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var g = Function.init(a);
    defer g.deinit();
    const gi = try g.types.intern(i32k);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gi);
    const three = try g.appendInst(gb, gi, .{ .iconst = 3 }); // riscv64 has no immediate mul
    const gm = try g.appendInst(gb, gi, .{ .arith = .{ .op = .mul, .lhs = gx, .rhs = three } });
    g.setTerminator(gb, .{ .ret = gm });

    var f = Function.init(a);
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
    for (linked.symbols, 0..) |s, i| syms[i] = .{ .name = s.name, .word = s.offset };
    const text = try disasm.formatModule(a, linked.code, syms);
    defer a.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "helper:\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "main:\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "  <helper>") != null); // the resolved jal
}

test "codegen+disasm round-trip: integer add" {
    // Compile RV64 and assert the disassembled listing. Checks instruction selection and
    // register allocation without executing, so it runs on any host.
    const a = std.testing.allocator;
    var func = Function.init(a);
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
    try std.testing.expectEqualStrings(
        \\0000: 00b502b3  add x5, x10, x11
        \\0004: 00028513  mv x10, x5
        \\0008: 00008067  ret
        \\
    , text);
}

test "codegen+disasm round-trip: control flow (max via if/else)" {
    // RV64 needs the high-profile `if` legalized to flat branches before isel (unlike
    // aarch64/x86, whose isel lowers the diamond inline), so this goes through the full
    // pipeline via harness.compileFunc. It validates the disassembler's scattered B-type and
    // J-type immediate decoding on real codegen: a fused compare-and-branch `blt .+16` plus
    // forward and BACKWARD `jal` offsets (.+20, .-12, .-20), the trickiest part of the RISC-V
    // decoder. The `x > y` icmp is the single-use condition of the immediately-following if,
    // so isel fuses it into a native `blt x11, x10` (branch to `then` when y < x, i.e. x > y)
    // instead of materializing `slt x5, x11, x10; bnez x5`. Structural only (does not run).
    const a = std.testing.allocator;
    var func = Function.init(a);
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

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\0000: 00a5c863  blt x11, x10, .+16
        \\0004: 0140006f  j .+20
        \\0008: 00028513  mv x10, x5
        \\000c: 00008067  ret
        \\0010: 00050293  mv x5, x10
        \\0014: ff5ff06f  j .-12
        \\0018: 00058293  mv x5, x11
        \\001c: fedff06f  j .-20
        \\
    , text);
}

test "codegen+disasm round-trip: fused compare-and-branch for if(icmp)" {
    // An UNSIGNED `x < y` that is the single-use condition of the immediately-following
    // if fuses into a native compare-and-branch on the two operands: no `sltu`
    // materialization and no `bnez` re-test. The THEN edge (tb) is the block emitted right
    // after the entry, so fall-through elision INVERTS the fused `bltu` to a `bgeu` that
    // targets the ELSE block and drops the trailing `jal`, falling through to THEN. This
    // proves the fused (inverted) branch is emitted. Structural only; the qemu-user
    // execution corpus (cases.zig) is the correctness oracle for the taken edge.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, u32_t);
    const y = try func.appendBlockParam(e, u32_t);
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const c = try func.appendInst(e, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    try func.appendIf(e, c, .{ .target = tb, .args = &.{} }, .{ .target = eb, .args = &.{} });
    const one = try func.appendInst(tb, u32_t, .{ .iconst = 1 });
    func.setTerminator(tb, .{ .ret = one });
    const zero = try func.appendInst(eb, u32_t, .{ .iconst = 0 });
    func.setTerminator(eb, .{ .ret = zero });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    // Fusion fired: a native unsigned compare-branch on the operands (a0=x, a1=y), inverted
    // to `bgeu` by fall-through elision (THEN is the next block), with no boolean
    // materialization or re-test.
    try std.testing.expect(std.mem.indexOf(u8, text, "bgeu x10, x11") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sltu") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bnez") == null);
}

/// A single-use icmp immediately preceding an if (unsigned min(a, b) via `a < b`), the same
/// shape `compileFuncWithCaps`'s caller below needs to prove both the flag-off fallback and the
/// flag-on (default) fused form, built once so both tests share it.
fn buildUnsignedMinIf(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, u32_t);
    const y = try func.appendBlockParam(e, u32_t);
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const c = try func.appendInst(e, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    try func.appendIf(e, c, .{ .target = tb, .args = &.{} }, .{ .target = eb, .args = &.{} });
    func.setTerminator(tb, .{ .ret = x }); // x < y -> x is the min
    func.setTerminator(eb, .{ .ret = y });
    return func;
}

test "caps.fuse_cmp_branch = false falls back to sltu;bnez, not the fused bltu (correct result)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = try buildUnsignedMinIf(a);
    defer func.deinit();

    // Compile ONCE (compileFuncWithCaps legalizes/splits-critical-edges in place, so compiling
    // the same func object twice would double-apply those passes); run the one resulting image
    // for every input via `runCode`, mirroring the relaxation tests above.
    var words = try harness.compileFuncWithCaps(a, &func, .{ .fuse_cmp_branch = false });
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    // The gate declined the fusion: the icmp materializes its boolean (`sltu`) and the if
    // re-tests it, NOT the fused `bltu` this same shape produces when the flag is on (see the
    // byte-identical-default test right below). The THEN block is emitted next, so fall-through
    // elision INVERTS the boolean re-test `bnez` to `beqz` (targeting ELSE) and drops the `jal`.
    try std.testing.expect(std.mem.indexOf(u8, text, "sltu") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "beqz") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bltu") == null);

    try std.testing.expectEqual(@as(i64, 3), try harness.runCode(io, a, words.items, &.{ 7, 3 }, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 1), try harness.runCode(io, a, words.items, &.{ 1, 2 }, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 9), try harness.runCode(io, a, words.items, &.{ 9, 9 }, harness.qemu_user)); // equal -> else (y)
}

test "caps.fuse_cmp_branch = true (the default) emits the fused bltu with no sltu" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = try buildUnsignedMinIf(a);
    defer func.deinit();

    var words = try harness.compileFuncWithCaps(a, &func, .{ .fuse_cmp_branch = true });
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    // Fusion fired and fall-through elision inverted the fused `bltu` to `bgeu` (THEN is the
    // next block), targeting ELSE with the `jal` dropped: still a native compare-branch on the
    // operands, no boolean materialization or re-test.
    try std.testing.expect(std.mem.indexOf(u8, text, "bgeu") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sltu") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bnez") == null);

    try std.testing.expectEqual(@as(i64, 3), try harness.runCode(io, a, words.items, &.{ 7, 3 }, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 1), try harness.runCode(io, a, words.items, &.{ 1, 2 }, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 9), try harness.runCode(io, a, words.items, &.{ 9, 9 }, harness.qemu_user));
}

/// Compile `func`, map it executable, and call it natively. Skips off RISC-V.
fn runNative(allocator: std.mem.Allocator, func: *Function, args: []const i64) !i64 {
    if (builtin.cpu.arch != .riscv64) return error.SkipZigTest;

    var words = try harness.compileFunc(allocator, func);
    defer words.deinit(allocator);
    const bytes = std.mem.sliceAsBytes(words.items);

    const mem = try std.posix.mmap(null, bytes.len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(mem);
    @memcpy(mem[0..bytes.len], bytes);
    const rc = std.posix.system.mprotect(mem.ptr, mem.len, .{ .READ = true, .EXEC = true });
    if (std.posix.errno(rc) != .SUCCESS) return error.ProtectFailed;
    if (builtin.cpu.arch == .riscv64) asm volatile ("fence.i" ::: .{ .memory = true });

    const ptr = mem.ptr;
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) i64, @ptrCast(ptr))(),
        1 => @as(*const fn (i64) callconv(.c) i64, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i64, i64) callconv(.c) i64, @ptrCast(ptr))(args[0], args[1]),
        else => error.Unsupported,
    };
}

test "qemu-user-riscv: a conditional branch past ±4KiB relaxes and runs both edges" {
    // A far conditional branch is the whole point of relaxation. The `then` target is
    // pushed > 4KiB away by padding the else block with a long dependency chain of adds
    // (each feeds the next, so none is dead-code-eliminated and the block cannot shrink
    // below ~1300 words). The B-type branch to `then` cannot reach that, so relaxation
    // MUST expand it to an inverted short branch (skip +8) over a `jal` to the far
    // target. We assert the relaxation fired structurally (the +8 skip), then execute
    // under qemu for inputs on BOTH edges to prove control flow is correct.
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    // Layout order follows block-append order: entry(0), else(1, the padding), then(2).
    // So `then` lands after the entire else chain, making the branch to it far.
    const entry = try func.appendBlock();
    const else_b = try func.appendBlock();
    const then_b = try func.appendBlock();

    const x = try func.appendBlockParam(entry, i32_t);
    const seven = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = seven } });
    // The else edge carries `x` into the padding block so the chain below depends on a
    // runtime value and cannot be constant-folded away (a chain of constant adds would
    // collapse to a single `li`, defeating the padding).
    const ex = try func.appendBlockParam(else_b, i32_t);
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{x} });

    // Taken edge (x == 7): return 111.
    const r_then = try func.appendInst(then_b, i32_t, .{ .iconst = 111 });
    func.setTerminator(then_b, .{ .ret = r_then });

    // Not-taken edge: a 1300-long add chain accumulating `x` each step (each add feeds
    // the next, so none is dead and the running sum is opaque to a constant-folder). It
    // pads the layout well past 4KiB and yields (1 + 1300) * x, a distinct checkable
    // result for the else edge.
    var acc = ex;
    var pad: usize = 0;
    while (pad < 1300) : (pad += 1) {
        acc = try func.appendInst(else_b, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = ex } });
    }
    func.setTerminator(else_b, .{ .ret = acc });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);

    // Relaxation fired: the far branch became an inverted short branch skipping the jal
    // (offset +8 = the instruction after the jal). No near branch in this function has an
    // 8-byte target, so `.+8` uniquely marks the relaxed skip.
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ".+8\n") != null);

    // Execute under qemu on both edges (skips cleanly if qemu is absent). Taken edge
    // (x == 7) returns 111; not-taken edge (x == 5) returns (1 + 1300) * 5 = 6505.
    try std.testing.expectEqual(@as(i64, 111), try harness.runCode(io, a, words.items, &.{7}, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 6505), try harness.runCode(io, a, words.items, &.{5}, harness.qemu_user));
}

test "qemu-user-riscv: a near conditional branch stays single-word (no relaxation)" {
    // The common case: a small if whose target is well within ±4KiB emits a single-word
    // branch unchanged by relaxation. Asserts no +8 inverted skip appears and that both
    // edges execute correctly, so the relaxation pass is a no-op on near branches.
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const else_b = try func.appendBlock();
    const then_b = try func.appendBlock();

    const x = try func.appendBlockParam(entry, i32_t);
    const seven = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = seven } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });

    const r_then = try func.appendInst(then_b, i32_t, .{ .iconst = 111 });
    func.setTerminator(then_b, .{ .ret = r_then });
    const r_else = try func.appendInst(else_b, i32_t, .{ .iconst = 222 });
    func.setTerminator(else_b, .{ .ret = r_else });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);

    const text = try disasm.format(a, words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ".+8\n") == null); // no inverted skip

    try std.testing.expectEqual(@as(i64, 111), try harness.runCode(io, a, words.items, &.{7}, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 222), try harness.runCode(io, a, words.items, &.{5}, harness.qemu_user));
}

test "qemu-user-riscv: a loop's backward conditional branch past -4KiB relaxes and runs both edges" {
    // The negative-offset twin of the forward far-branch test above: a do-while loop whose
    // back-edge conditional branch targets its OWN header, which sits more than 4KiB behind it.
    // The loop-carried state lives in two `alloca` slots (not block params) so the back edge and
    // the exit edge both carry zero jump args; that keeps `splitCriticalEdges` a no-op, so the
    // `if`'s `then` edge stays wired directly to the header block instead of being redirected
    // through a trampoline landing block (which would turn the direct branch forward and hide
    // the backward case behind an always-in-range `jal`). Each iteration pads with a 1300-long
    // dependency chain of adds on the loaded, runtime-derived `i` (each add feeds the next, so
    // none is dead and the chain cannot shrink below ~1300 words), pushing the loop body past
    // 4KiB. The B-type branch back to the header cannot reach that far behind it, so relaxation
    // MUST expand it to an inverted short branch (skip +8) over a `jal` with a NEGATIVE
    // displacement (the far jump goes backward).
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);

    // Layout order follows block-append order: entry(0), loop(1, the header AND the padded
    // body, since this is a do-while shape), done(2).
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const pi = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const pacc = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, n, pi);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.appendStore(entry, zero, pacc);
    try func.setJump(entry, loop, &.{}); // no args: loop's header carries no block params

    // Header + body in one block: `pi`/`pacc` are alloca pointers defined in the dominating
    // `entry` block, so `loop` (and its own back edge) can load/store them directly without
    // threading them through block params.
    const i_val = try func.appendInst(loop, i32_t, .{ .load = .{ .ptr = pi } });
    const acc_val = try func.appendInst(loop, i32_t, .{ .load = .{ .ptr = pacc } });
    const ni = try func.appendArithImm(loop, i32_t, .sub, i_val, 1);
    const acc1 = try func.appendInst(loop, i32_t, .{ .arith = .{ .op = .add, .lhs = acc_val, .rhs = i_val } });

    // The padding chain, exactly the forward test's technique: each add depends on the loaded
    // (runtime) `i_val` and feeds the next, so none is dead and the chain cannot be folded away.
    var acc_x = acc1;
    var pad: usize = 0;
    while (pad < 1300) : (pad += 1) {
        acc_x = try func.appendInst(loop, i32_t, .{ .arith = .{ .op = .add, .lhs = acc_x, .rhs = i_val } });
    }

    try func.appendStore(loop, ni, pi);
    try func.appendStore(loop, acc_x, pacc);
    const cmp_zero = try func.appendInst(loop, i32_t, .{ .iconst = 0 });
    const cont = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .ne, .lhs = ni, .rhs = cmp_zero } });
    // `then` (continue) targets `loop` itself: a genuine backward edge. `else` (exit) targets
    // `done`, forward. Both edges pass zero args (see above), so `then` stays wired straight to
    // `loop`'s header instead of a trampoline.
    try func.appendIf(loop, cont, .{ .target = loop, .args = &.{} }, .{ .target = done, .args = &.{} });

    const racc = try func.appendInst(done, i32_t, .{ .load = .{ .ptr = pacc } });
    func.setTerminator(done, .{ .ret = racc });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);

    // Relaxation fired: the far back-edge branch became an inverted short branch skipping the
    // jal (offset +8 = the instruction after the jal), same structural marker as the forward
    // test. And the jal it skips carries a NEGATIVE displacement, proving the relaxed jump
    // actually goes backward (to the loop header), not forward.
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ".+8\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "j .-") != null);

    // Execute under qemu (skips cleanly if qemu is absent). Each iteration accumulates
    // 1301*i (the real add plus the 1300-long padding chain, all on the same `i`), so the
    // total is 1301 * n*(n+1)/2. n=5 runs the loop 5 times: 4 back-edge continues then one
    // exit, exercising BOTH outcomes of the same relaxed branch in a single call. n=1 runs
    // it exactly once, exiting on the very first test without ever taking the back edge.
    try std.testing.expectEqual(@as(i64, 19515), try harness.runCode(io, a, words.items, &.{5}, harness.qemu_user)); // 1301 * 15
    try std.testing.expectEqual(@as(i64, 1301), try harness.runCode(io, a, words.items, &.{1}, harness.qemu_user)); // 1301 * 1
}

test "riscv64 fallthrough: an unconditional jump to the next block elides the jal" {
    // entry computes x + y and jumps to `tail`, which is emitted immediately after it. The jump
    // falls through: the sum move into `tail`'s parameter still runs, but the `jal` is elided.
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    const entry = try func.appendBlock();
    const tail = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const s = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    try func.setJump(entry, tail, &.{s});
    const p = try func.appendBlockParam(tail, i32_t);
    func.setTerminator(tail, .{ .ret = p });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    // The only control transfer left is the final `ret`: no unconditional `j` bridges the two blocks.
    try std.testing.expect(std.mem.indexOf(u8, text, "j .") == null);

    try std.testing.expectEqual(@as(i64, 7), try harness.runCode(io, a, words.items, &.{ 3, 4 }, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, -1), try harness.runCode(io, a, words.items, &.{ 4, -5 }, harness.qemu_user));
}

test "riscv64 fallthrough: an if whose else-edge targets the next block elides the jal" {
    // Append order sets layout: entry(0), else_b(1), then_b(2). The ELSE edge targets the block
    // emitted next, so elision keeps the branch to THEN and drops the `jal`, falling through to ELSE.
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const else_b = try func.appendBlock();
    const then_b = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const seven = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = seven } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const r_then = try func.appendInst(then_b, i32_t, .{ .iconst = 111 });
    func.setTerminator(then_b, .{ .ret = r_then });
    const r_else = try func.appendInst(else_b, i32_t, .{ .iconst = 222 });
    func.setTerminator(else_b, .{ .ret = r_else });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    // The un-inverted fused branch to THEN stays (rs1 = x = a0); its `jal` to ELSE is elided.
    try std.testing.expect(std.mem.indexOf(u8, text, "beq x10,") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "j .") == null);

    // Sweep: x == 7 branches to THEN (111); anything else falls through to ELSE (222).
    try std.testing.expectEqual(@as(i64, 111), try harness.runCode(io, a, words.items, &.{7}, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 222), try harness.runCode(io, a, words.items, &.{5}, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 222), try harness.runCode(io, a, words.items, &.{0}, harness.qemu_user));
}

test "riscv64 fallthrough: an if whose then-edge targets the next block inverts the branch" {
    // Append order sets layout: entry(0), then_b(1), else_b(2). The THEN edge targets the block
    // emitted next, so elision INVERTS the fused `beq` to `bne` (targeting ELSE), drops the `jal`,
    // and falls through to THEN.
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const seven = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = seven } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const r_then = try func.appendInst(then_b, i32_t, .{ .iconst = 111 });
    func.setTerminator(then_b, .{ .ret = r_then });
    const r_else = try func.appendInst(else_b, i32_t, .{ .iconst = 222 });
    func.setTerminator(else_b, .{ .ret = r_else });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    // The fused `beq` was inverted to `bne` (rs1 = x = a0), so no plain `beq` and no `jal` remain.
    try std.testing.expect(std.mem.indexOf(u8, text, "bne x10,") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "beq") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "j .") == null);

    // Sweep: x == 7 falls through to THEN (111); anything else takes the inverted branch to ELSE (222).
    try std.testing.expectEqual(@as(i64, 111), try harness.runCode(io, a, words.items, &.{7}, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 222), try harness.runCode(io, a, words.items, &.{5}, harness.qemu_user));
    try std.testing.expectEqual(@as(i64, 222), try harness.runCode(io, a, words.items, &.{0}, harness.qemu_user));
}

test "riscv64 fallthrough: block-param moves on a jump fall-through edge stay correct" {
    // entry passes its two params SWAPPED into `tail`'s params over a fall-through edge (tail is next).
    // The `jal` is elided, but the cycle-breaking swap moves must still run, so `tail` sees (b, a) and
    // returns b - a. `@"if"` edges carry no args on riscv64, so a JUMP terminator exercises this.
    const a = std.testing.allocator;
    const io = std.testing.io;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    const entry = try func.appendBlock();
    const tail = try func.appendBlock();
    const av = try func.appendBlockParam(entry, i32_t);
    const bv = try func.appendBlockParam(entry, i32_t);
    try func.setJump(entry, tail, &.{ bv, av }); // swap: p <- b, q <- a
    const p = try func.appendBlockParam(tail, i32_t);
    const q = try func.appendBlockParam(tail, i32_t);
    const d = try func.appendInst(tail, i32_t, .{ .arith = .{ .op = .sub, .lhs = p, .rhs = q } });
    func.setTerminator(tail, .{ .ret = d });

    var words = try harness.compileFunc(a, &func);
    defer words.deinit(a);
    const text = try disasm.format(a, words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "j .") == null); // fall-through, no jump word

    try std.testing.expectEqual(@as(i64, -7), try harness.runCode(io, a, words.items, &.{ 10, 3 }, harness.qemu_user)); // 3 - 10
    try std.testing.expectEqual(@as(i64, 5), try harness.runCode(io, a, words.items, &.{ 4, 9 }, harness.qemu_user)); // 9 - 4
    try std.testing.expectEqual(@as(i64, 0), try harness.runCode(io, a, words.items, &.{ 7, 7 }, harness.qemu_user));
}

test "riscv64 fallthrough: a diamond and a loop compute correctly under elision" {
    const a = std.testing.allocator;
    const io = std.testing.io;

    // DIAMOND: signed max(x, y) via an if/else that merges. Critical-edge splitting inserts the
    // trampoline blocks and every fall-through edge among them is elided where it targets the next
    // block. Execution must still select the larger value.
    {
        var func = Function.init(a);
        defer func.deinit();
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const merge = try func.appendBlock();
        const x = try func.appendBlockParam(entry, i32_t);
        const y = try func.appendBlockParam(entry, i32_t);
        const r = try func.appendBlockParam(merge, i32_t);
        const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = y } });
        try func.appendIf(entry, c, .{ .target = merge, .args = &.{x} }, .{ .target = merge, .args = &.{y} });
        func.setTerminator(merge, .{ .ret = r });

        var words = try harness.compileFunc(a, &func);
        defer words.deinit(a);
        try std.testing.expectEqual(@as(i64, 9), try harness.runCode(io, a, words.items, &.{ 9, 4 }, harness.qemu_user));
        try std.testing.expectEqual(@as(i64, 8), try harness.runCode(io, a, words.items, &.{ 3, 8 }, harness.qemu_user));
        try std.testing.expectEqual(@as(i64, 5), try harness.runCode(io, a, words.items, &.{ 5, 5 }, harness.qemu_user));
    }

    // LOOP: sum of 1..n via a do-while over two alloca slots (edges carry no args, so the back edge
    // stays wired straight to the header). The header's if falls through to the body by inverting its
    // branch (THEN is next); the total must be n*(n+1)/2.
    {
        var func = Function.init(a);
        defer func.deinit();
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const ptr_t = try func.types.intern(.ptr);
        const entry = try func.appendBlock();
        const loop = try func.appendBlock();
        const body = try func.appendBlock();
        const done = try func.appendBlock();
        const n = try func.appendBlockParam(entry, i32_t);
        const pi = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
        const psum = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
        try func.appendStore(entry, n, pi);
        const zero0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
        try func.appendStore(entry, zero0, psum);
        try func.setJump(entry, loop, &.{});

        const i_val = try func.appendInst(loop, i32_t, .{ .load = .{ .ptr = pi } });
        const zero1 = try func.appendInst(loop, i32_t, .{ .iconst = 0 });
        const c = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .gt, .lhs = i_val, .rhs = zero1 } });
        try func.appendIf(loop, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });

        const sum_val = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = psum } });
        const i_body = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = pi } });
        const nsum = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = sum_val, .rhs = i_body } });
        const ni = try func.appendArithImm(body, i32_t, .sub, i_body, 1);
        try func.appendStore(body, nsum, psum);
        try func.appendStore(body, ni, pi);
        try func.setJump(body, loop, &.{});

        const rsum = try func.appendInst(done, i32_t, .{ .load = .{ .ptr = psum } });
        func.setTerminator(done, .{ .ret = rsum });

        var words = try harness.compileFunc(a, &func);
        defer words.deinit(a);
        try std.testing.expectEqual(@as(i64, 15), try harness.runCode(io, a, words.items, &.{5}, harness.qemu_user)); // 1+2+3+4+5
        try std.testing.expectEqual(@as(i64, 55), try harness.runCode(io, a, words.items, &.{10}, harness.qemu_user));
        try std.testing.expectEqual(@as(i64, 0), try harness.runCode(io, a, words.items, &.{0}, harness.qemu_user));
    }
}

test "native-riscv: arithmetic runs in-process when the host is RISC-V" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const a = try func.appendBlockParam(e, t);
    const b = try func.appendBlockParam(e, t);
    const p = try func.appendInst(e, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const s = try func.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = a } });
    func.setTerminator(e, .{ .ret = s });
    try std.testing.expectEqual(@as(i64, 15), try runNative(allocator, &func, &.{ 3, 4 }));
}
