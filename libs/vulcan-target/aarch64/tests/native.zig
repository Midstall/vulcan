//! Native execution validation for the AArch64 backend (test-only). The host is
//! aarch64, so generated A64 code maps into W^X memory and is called directly, no
//! emulator needed. The codegen oracle: a wrong encoding or selection returns the
//! wrong value or faults.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const link = @import("../link.zig");
const jit = @import("../jit.zig");
const encode = @import("../encode.zig");

const Function = ir.function.Function;

/// Compile `func` to A64 and assert its disassembled listing equals `expected`. This round-trips
/// codegen through the disassembler, so it checks the actual instructions and register allocation
/// rather than just the run result, and works on any host since it never executes the code.
fn expectAsm(func: *const Function, expected: []const u8) !void {
    const a = std.testing.allocator;
    const code = try isel.selectFunction(a, func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);
    try std.testing.expectEqualStrings(expected, text);
}

test "codegen+disasm round-trip: integer add" {
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const s = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(e, .{ .ret = s });

    try expectAsm(&func,
        \\0000: 0b010007  add w7, w0, w1
        \\0004: aa0703e0  mov x0, x7
        \\0008: d65f03c0  ret
        \\
    );
}

test "codegen+disasm round-trip: a returned constant" {
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const v = try func.appendInst(e, i32_t, .{ .iconst = 42 });
    func.setTerminator(e, .{ .ret = v });

    try expectAsm(&func,
        \\0000: 52800547  mov w7, #0x2a
        \\0004: aa0703e0  mov x0, x7
        \\0008: d65f03c0  ret
        \\
    );
}

test "codegen+disasm round-trip: control flow (max via if/else)" {
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

    // The icmp is the single-use condition of the immediately-following if, so isel fuses
    // it into a compare-and-branch: no `cset` boolean, and the `cbnz` re-test becomes a
    // `b.gt` on the flags `cmp` set (branching to the then-edge on the icmp's condition).
    // The then-edge target `m` is the block emitted next, so the trailing `b then` is elided
    // and the then-moves fall through into `m` (fall-through branch elision). The `b else`
    // now jumps just over the then-moves. Exercises the disassembler on real branch offsets
    // and block-edge moves. The native max tests below prove this returns the same results.
    try expectAsm(&func,
        \\0000: 6b01001f  cmp w0, w1
        \\0004: 5400006c  b.gt .+12
        \\0008: aa0103e7  mov x7, x1
        \\000c: 14000002  b .+8
        \\0010: aa0003e7  mov x7, x0
        \\0014: aa0703e0  mov x0, x7
        \\0018: d65f03c0  ret
        \\
    );
}

test "codegen+disasm round-trip: scalar float add" {
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, f32_t);
    const y = try func.appendBlockParam(e, f32_t);
    const s = try func.appendInst(e, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(e, .{ .ret = s });

    try expectAsm(&func,
        \\0000: 1e212807  fadd s7, s0, s1
        \\0004: 1e6040e0  fmov d0, d7
        \\0008: d65f03c0  ret
        \\
    );
}

test "codegen+disasm round-trip: register spilling opens a frame and spills to the stack" {
    // Many simultaneously-live values exceed the register budget, forcing the allocator to
    // spill. Rather than assert an exact (allocator-dependent) listing, check the shape:
    // a stack frame is opened and there is at least one stack store and one stack reload.
    // This validates spill codegen and the disassembly of stack memory ops on real output.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    var vals: [24]ir.function.Value = undefined;
    for (0..vals.len) |i| vals[i] = try func.appendArithImm(e, i32_t, .add, x, @intCast(i + 1));
    var acc = vals[0];
    for (1..vals.len) |i| acc = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[i] } });
    func.setTerminator(e, .{ .ret = acc });

    const code = try isel.selectFunction(a, &func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "sub sp, sp, #") != null); // prologue frame
    try std.testing.expect(std.mem.indexOf(u8, text, "str ") != null); // a spill store
    try std.testing.expect(std.mem.indexOf(u8, text, "ldr ") != null); // a reload
    try std.testing.expect(std.mem.indexOf(u8, text, ", [sp") != null); // to/from a stack slot
    try std.testing.expect(std.mem.indexOf(u8, text, "ret") != null);
}

test "native: a register-pressure kernel spills and reloads to the correct result" {
    // f(a, b) = sum over k in 1..=20 of (a*k + b). All 20 products stay live until the final
    // reduction, far past the GPR pool, so the allocator must spill and later reload. Executing it
    // proves the spill/reload seams (loadOp/resultReg/storeResult, now routed through locationAt)
    // still produce the right value. With no splits, this is byte-identical to the pre-split path.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    func.setTerminator(b, .{ .ret = acc });

    // a=3, b=5: sum_{k=1..20}(3k+5) = 3*210 + 20*5 = 730.
    try expectRun(allocator, &func, &.{ 3, 5 }, 730);
    // a=-2, b=1: sum_{k=1..20}(-2k+1) = -2*210 + 20 = -400.
    try expectRun(allocator, &func, &.{ -2, 1 }, -400);
}

test "intra-block tail split reloads the correct value" {
    // f(a, b) = a*b (defined FIRST, then held live over heavy register pressure) plus
    // sum_{k=1..20}(a*k + b). The early product `t0` is an intra value whose only remaining use is
    // the very last add, so Belady/MIN spills it under pressure. With live-range splitting it
    // TAIL-SPLITS: register for the hot prefix, stack slot for the cold tail, with a store at the
    // split point. The late add reloads it from the slot, so the result is only correct if the
    // split store/reload round-trips the right bits.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    // t0 is defined early and used only at the end: a long intra live range across the pressure.
    const t0 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    // The late use of t0: without a correct tail split this reloads garbage.
    const result = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = t0 } });
    func.setTerminator(b, .{ .ret = result });

    // Meaningful-differential gate: at least one value must actually tail-split, otherwise a plain
    // whole-spill would also return the right value and the test would not exercise the new path.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    try std.testing.expect(try isel.debugSegmentCount(allocator, &func) > 0);

    // Sweep inputs (0, 1, -1, and larger magnitudes). Expected in i32 wrapping arithmetic to match
    // the target's 32-bit adds/muls exactly. result = a*b + sum_{k=1..20}(a*k + b).
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        var expected: i32 = av *% bv;
        var kk: i32 = 1;
        while (kk <= 20) : (kk += 1) expected +%= (av *% kk) +% bv;
        const got = try run(allocator, &func, &.{ av, bv });
        try std.testing.expectEqual(expected, got);
    }
}

test "second-chance reload re-homes a spilled value" {
    // f(a, b): t0 = a*b is defined FIRST and used again only at the very end. Under the 20-term
    // pressure block it TAIL-SPLITS (register prefix + slot tail). As the reduction consumes the
    // terms the register pressure DROPS, so by t0's late use a register is free again and
    // second-chance RE-HOMES t0 into it: its final use reads that register instead of reloading from
    // the slot. The result is correct only if the re-home reload round-trips the right bits.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    // t0 is defined early and used only at the end: a long intra live range across the pressure.
    const t0 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    // The late use of t0: with second-chance it reads a re-homed register, not a per-use slot reload.
    const result = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = t0 } });
    func.setTerminator(b, .{ .ret = result });

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // Meaningful-differential gate: a second-chance re-home MUST have fired (a `.reg` segment after a
    // `.slot` segment). Without it this would only exercise Task 4 tail-split + per-use reload.
    try std.testing.expect(try isel.debugReHomeCount(allocator, &func) > 0);

    // Sweep inputs (0, 1, -1, and larger magnitudes), i32 wrapping arithmetic to match the target's
    // 32-bit adds/muls exactly. result = a*b + sum_{k=1..20}(a*k + b).
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        var expected: i32 = av *% bv;
        var kk: i32 = 1;
        while (kk <= 20) : (kk += 1) expected +%= (av *% kk) +% bv;
        const got = try run(allocator, &func, &.{ av, bv });
        try std.testing.expectEqual(expected, got);
    }
}

test "second-chance declines when no register is free at the tail use (per-use reload)" {
    // A sustained-pressure kernel where second-chance CANNOT save every split value. Six early
    // `sp[i] = a*(i+1)` products are defined first and used only late, so under pressure they
    // TAIL-SPLIT (their far next use makes them the Belady victims). Twelve `res[i] = b+const`
    // values are then defined and used TWICE (once before and once after the sp uses), so they stay
    // register-resident and occupy every register ACROSS the sp uses. With the pool full at those
    // uses, second-chance has no free register to re-home some of the split sp values, so they fall
    // back to per-use slot reloads. This exercises the decline path (`frees` empty) alongside the
    // re-homes, and the result is correct only if every reload (re-homed or per-use) is right.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const nspill = 6;
    const nres = 12;
    var sp: [nspill]ir.function.Value = undefined;
    for (0..nspill) |i| {
        const c = try func.appendInst(b, t, .{ .iconst = @intCast(i + 1) });
        sp[i] = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    }
    var res: [nres]ir.function.Value = undefined;
    for (0..nres) |i| {
        const c = try func.appendInst(b, t, .{ .iconst = @intCast(100 + i) });
        res[i] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = c } });
    }
    var acc = res[0];
    for (1..nres) |i| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = res[i] } });
    for (0..nspill) |i| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = sp[i] } });
    for (0..nres) |i| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = res[i] } });
    func.setTerminator(b, .{ .ret = acc });

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // Meaningful gate: more values TAIL-SPLIT than were RE-HOMED, i.e. at least one split value found
    // no free register at its tail use and DECLINED (stayed in its slot, reloading per use). Both the
    // re-home path and the decline path are thus exercised in one kernel.
    const segs = try isel.debugSegmentCount(allocator, &func);
    const rehomes = try isel.debugReHomeCount(allocator, &func);
    try std.testing.expect(segs > rehomes);

    // result = 2*sum_i(b + 100 + i) + sum_i(a*(i+1)), evaluated in i32 wrapping arithmetic to match
    // the target's 32-bit adds/muls exactly. Sweep inputs including zero, unit, and larger magnitudes.
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        var expected: i32 = bv +% 100; // res[0]
        var i: i32 = 1;
        while (i < nres) : (i += 1) expected +%= bv +% (100 + i);
        i = 0;
        while (i < nspill) : (i += 1) expected +%= av *% (i + 1);
        i = 0;
        while (i < nres) : (i += 1) expected +%= bv +% (100 + i);
        const got = try run(allocator, &func, &.{ av, bv });
        try std.testing.expectEqual(expected, got);
    }
}

test "second-chance re-homes a ret-only value AT the terminator (terminator drain)" {
    // Regression for the terminator-position reload drain. t0 = a*b is defined FIRST and is used ONLY
    // by the `ret`, whose operand is a NON-edge-arg, so t0 is intra-splittable. Under the 20-term
    // pressure block t0 TAIL-SPLITS (register prefix + slot tail). As the reduction drains the terms a
    // register frees, and because t0's SOLE remaining use is the terminator, second-chance re-homes t0
    // with a `.reload` recorded AT the terminator position (block_end). The per-instruction drain never
    // reaches that position, so before the fix the reload is dropped and `ret` returns the stale
    // register (wrong value). The result is correct only if the new terminator drain emits the reload.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    // t0 is defined early and used only by the ret: a long intra live range across the pressure whose
    // sole use lands on the terminator position.
    const t0 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    // The reduction only exists to create and then relieve register pressure (its final `acc` is
    // deliberately unreturned). The RETURN value is t0 itself, so t0's last use is the terminator and
    // any second-chance re-home of it lands there.
    func.setTerminator(b, .{ .ret = t0 });

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // Meaningful-differential gate: t0 must actually tail-split, otherwise a whole-life register would
    // also return the right value and the terminator drain would not be exercised.
    try std.testing.expect(try isel.debugReHomeCount(allocator, &func) > 0);

    // Sweep inputs (zero, unit, negatives, larger magnitudes). result = a*b in i32 wrapping arithmetic.
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        const expected: i32 = av *% bv;
        const got = try run(allocator, &func, &.{ av, bv });
        try std.testing.expectEqual(expected, got);
    }
}

test "codegen+disasm round-trip: a call emits the ABI frame and bl" {
    // f(x) = g(x) + 1. Exercises the whole call ABI at the instruction level: a frame that
    // saves the link register, a `bl` to the callee (an unresolved relocation, so offset 0),
    // and the epilogue that restores and returns. Robust markers, not an exact listing.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const r = try func.appendCall(e, i32_t, "g", &.{x});
    const s = try func.appendArithImm(e, i32_t, .add, r, 1);
    func.setTerminator(e, .{ .ret = s });

    const code = try isel.selectFunction(a, &func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "sub sp, sp, #") != null); // prologue
    // The link-register store fuses with the first callee-saved store into a single stp (the
    // ldp/stp peephole), so the save now reads `stp x30, ...`. Still proves x30 is saved.
    try std.testing.expect(std.mem.indexOf(u8, text, "stp x30, ") != null); // save the link register
    try std.testing.expect(std.mem.indexOf(u8, text, "bl .") != null); // the call
    try std.testing.expect(std.mem.indexOf(u8, text, "ldr x30, [sp") != null); // restore the link register
    try std.testing.expect(std.mem.indexOf(u8, text, "add sp, sp, #") != null); // epilogue
    try std.testing.expect(std.mem.indexOf(u8, text, "ret") != null);
}

test "dwarf: a linked module's real functions describe as DWARF (readelf-validated)" {
    // Compile and link a 2-function module, turn its symbol table into DWARF subprograms at the
    // functions' actual addresses, emit a debug ELF, and confirm readelf decodes it. This is the
    // DWARF emitter connected to actual codegen, not hand-written addresses.
    const a = std.testing.allocator;
    const dwarf = @import("../../dwarf.zig");
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var g = Function.init(a);
    defer g.deinit();
    const gi = try g.types.intern(i32k);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gi);
    const gm = try g.appendArithImm(gb, gi, .mul, gx, 3);
    g.setTerminator(gb, .{ .ret = gm });

    var f = Function.init(a);
    defer f.deinit();
    const fi = try f.types.intern(i32k);
    const fb = try f.appendBlock();
    const fa = try f.appendBlockParam(fb, fi);
    const called = try f.appendCall(fb, fi, "helper", &.{fa});
    f.setTerminator(fb, .{ .ret = called });

    var module = link.Module{};
    defer module.deinit(a);
    try module.addFunction(a, "helper", &g);
    try module.addFunction(a, "main", &f);
    var linked = try link.compileModule(a, &module);
    defer linked.deinit(a);

    const base: u64 = 0x400000;
    const code_size: u64 = linked.code.len * 4; // aarch64 words -> bytes
    const syms = try a.alloc(dwarf.SymIn, linked.symbols.len);
    defer a.free(syms);
    for (linked.symbols, 0..) |s, i| syms[i] = .{ .name = s.name, .offset = s.offset };
    const subs = try dwarf.subprogramsFromSymbols(a, base, code_size, syms);
    defer a.free(subs);

    const elf = try dwarf.emitDebugElf(a, .{ .name = "module.glsl", .low_pc = base, .high_pc = base + code_size, .subprograms = subs });
    defer a.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "m.dbg", .data = elf });
    const res = std.process.run(a, std.testing.io, .{ .argv = &.{ "readelf", "--debug-dump=info", "m.dbg" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "helper") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "main") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_subprogram") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "400000") != null); // the real base PC
}

test "module disasm: linked functions get labels and a resolved, named call" {
    // A two-function module (main calls helper) linked so the `bl` relocation resolves.
    // formatModule labels each function at its offset and annotates the resolved call with
    // the callee name (`<helper>`), so a linked image reads as a symbolized listing.
    const a = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var g = Function.init(a);
    defer g.deinit();
    const gi = try g.types.intern(i32k);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gi);
    const gm = try g.appendArithImm(gb, gi, .mul, gx, 3);
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
    for (linked.symbols, 0..) |s, i| syms[i] = .{ .name = s.name, .word = s.offset / 4 };
    const text = try disasm.formatModule(a, linked.code, syms);
    defer a.free(text);

    // The `bl` in main resolves back to helper and is annotated with its name. The displacement
    // is -40 (not -48): main's prologue fuses its link-register + callee-saved stores into stp
    // pairs (the ldp/stp peephole), shrinking main so the backward branch to helper is shorter.
    try std.testing.expect(std.mem.indexOf(u8, text, "helper:\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "main:\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bl .-40  <helper>") != null);
}

/// A named function for `runModule`. The entry must be first.
pub const NamedFunc = struct { name: []const u8, func: *Function };

/// Compile `func` to A64, JIT-map it, and call it with `args` (each loaded into an
/// argument register). Returns the i32 result. Skips when not on aarch64.
pub fn run(allocator: std.mem.Allocator, func: *const Function, args: []const i32) !i32 {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    return callI32(&buf, args);
}

/// Compile a single function and call it, returning its f64 result (in d0).
/// Integer arguments are passed in x-registers as usual. Skips off aarch64.
pub fn runF64(allocator: std.mem.Allocator, func: *const Function, args: []const i32) !f64 {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const ptr = buf.memory.ptr;
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) f64, @ptrCast(ptr))(),
        1 => @as(*const fn (i32) callconv(.c) f64, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i32, i32) callconv(.c) f64, @ptrCast(ptr))(args[0], args[1]),
        else => error.Unsupported,
    };
}

/// Link a set of functions into one image (resolving each `bl`), JIT-map it, and
/// call the entry (the first function). Skips when not on aarch64.
pub fn runModule(allocator: std.mem.Allocator, funcs: []const NamedFunc, args: []const i32) !i32 {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    var module: link.Module = .{};
    defer module.deinit(allocator);
    for (funcs) |nf| try module.addFunction(allocator, nf.name, nf.func);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    defer buf.deinit();
    return callI32(&buf, args);
}

fn callI32(buf: *const jit.CodeBuffer, args: []const i32) !i32 {
    const ptr = buf.memory.ptr; // page-aligned, satisfies the function-pointer alignment
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) i32, @ptrCast(ptr))(),
        1 => @as(*const fn (i32) callconv(.c) i32, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i32, i32) callconv(.c) i32, @ptrCast(ptr))(args[0], args[1]),
        3 => @as(*const fn (i32, i32, i32) callconv(.c) i32, @ptrCast(ptr))(args[0], args[1], args[2]),
        else => error.Unsupported,
    };
}

fn expectRun(allocator: std.mem.Allocator, func: *Function, args: []const i32, expected: i32) !void {
    const got = try run(allocator, func, args);
    if (got != expected) {
        // A codegen bug: dump the disassembly so the wrong instruction / register is visible
        // at a glance instead of just a wrong number (the whole point of the in-tree disasm).
        dumpDisasm(allocator, func, got, expected);
        return error.TestExpectedEqual;
    }
}

/// Print the compiled function's disassembly alongside a value mismatch. Best-effort: any
/// error here is swallowed so it never masks the original test failure.
fn dumpDisasm(allocator: std.mem.Allocator, func: *const Function, got: i32, expected: i32) void {
    const code = isel.selectFunction(allocator, func) catch return;
    defer allocator.free(code);
    const text = disasm.format(allocator, code) catch return;
    defer allocator.free(text);
    std.debug.print("\ncodegen mismatch: got {d}, expected {d}. disassembly:\n{s}\n", .{ got, expected, text });
}

fn i32func(allocator: std.mem.Allocator) !Function {
    return Function.init(allocator);
}

test "native: a*b + a" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });
    try expectRun(allocator, &func, &.{ 3, 4 }, 15); // 3*4 + 3
}

test "neon: a vector crosses a block edge whole (block-param move, no truncation)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // out = cond ? in[0] : in[1], where the chosen vector reaches the merge block as a
    // <4 x f32> block parameter, so a parallel move carries it across the edge.
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const entry = try func.appendBlock();
    const out = try func.appendBlockParam(entry, ptr_t);
    const in = try func.appendBlockParam(entry, ptr_t);
    const cond = try func.appendBlockParam(entry, i32_t);
    const va = try func.appendInst(entry, v4, .{ .load = .{ .ptr = in } });
    const p1 = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = 16 } });
    const vb = try func.appendInst(entry, v4, .{ .load = .{ .ptr = p1 } });
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const m = try func.appendBlockParam(merge, v4);
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{va}) } });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{vb}) } });
    try func.appendStore(merge, m, out);
    func.setTerminator(merge, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [2][4]f32, i32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const input: [2][4]f32 align(16) = .{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 } };
    var r1: [4]f32 align(16) = .{ 0, 0, 0, 0 };
    f(&r1, &input, 1);
    try std.testing.expectEqual([4]f32{ 1, 2, 3, 4 }, r1); // chose in[0], all lanes intact
    var r0: [4]f32 align(16) = .{ 0, 0, 0, 0 };
    f(&r0, &input, 0);
    try std.testing.expectEqual([4]f32{ 5, 6, 7, 8 }, r0); // chose in[1], all lanes intact
}

test "neon: high vector pressure spills and reloads all 128 bits (no truncation)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // Load N vectors and sum them. N exceeds the FP register pool, so some vectors spill.
    // The sum is only correct if a spilled vector reloads whole (a 64-bit reload would lose
    // lanes 2 and 3).
    const N = 18;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const in = try func.appendBlockParam(blk, ptr_t);
    var v: [N]ir.function.Value = undefined;
    for (0..N) |i| {
        const p = if (i == 0) in else try func.appendInst(blk, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = @intCast(i * 16) } });
        v[i] = try func.appendInst(blk, v4, .{ .load = .{ .ptr = p } });
    }
    var s = v[0];
    for (1..N) |i| s = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = s, .rhs = v[i] } });
    try func.appendStore(blk, s, out);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [N][4]f32) callconv(.c) void;
    var input: [N][4]f32 align(16) = undefined;
    for (0..N) |i| input[i] = .{ 1, 2, 3, 4 };
    var result: [4]f32 align(16) = .{ 0, 0, 0, 0 };
    @as(Fn, @ptrCast(buf.memory.ptr))(&result, &input);
    try std.testing.expectEqual([4]f32{ N, 2 * N, 3 * N, 4 * N }, result); // every lane summed
}

fn runF32x8(allocator: std.mem.Allocator, func: *const Function, args: [8]f32) !f32 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (f32, f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

test "vectorize: 4 parallel f32 adds fuse into a NEON vector add (same result)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // s = (a0+b0) + (a1+b1) + (a2+b2) + (a3+b3): four parallel adds, then a reduction.
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    var a: [4]ir.function.Value = undefined;
    var b: [4]ir.function.Value = undefined;
    for (0..4) |i| a[i] = try func.appendBlockParam(blk, f32_t);
    for (0..4) |i| b[i] = try func.appendBlockParam(blk, f32_t);
    var c: [4]ir.function.Value = undefined;
    for (0..4) |i| c[i] = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = a[i], .rhs = b[i] } });
    var s = c[0];
    for (1..4) |i| s = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = c[i] } });
    func.setTerminator(blk, .{ .ret = s });

    const args = [8]f32{ 1, 2, 3, 4, 10, 20, 30, 40 };
    const scalar_res = try runF32x8(allocator, &func, args); // 11+22+33+44 = 110

    const changed = try opt.vectorize.run(allocator, &func);
    try std.testing.expect(changed);

    // A vector-typed arith now exists (the four scalar adds became one).
    var has_vec = false;
    for (0..func.instCount()) |i| {
        if (func.opcodeMut(@enumFromInt(i)).* == .arith) {
            const res = func.instResult(@enumFromInt(i)).?;
            if (func.types.type_kind(func.valueType(res)) == .vector) has_vec = true;
        }
    }
    try std.testing.expect(has_vec);

    const vec_res = try runF32x8(allocator, &func, args);
    try std.testing.expectEqual(scalar_res, vec_res);
    try std.testing.expectEqual(@as(f32, 110), vec_res);
}

fn loadLane(func: *Function, blk: ir.function.Block, ptr_t: ir.types.Type, f32_t: ir.types.Type, base: ir.function.Value, i: usize) !ir.function.Value {
    if (i == 0) return func.appendInst(blk, f32_t, .{ .load = .{ .ptr = base } });
    const p = try func.appendInst(blk, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = @intCast(i * 4) } });
    return func.appendInst(blk, f32_t, .{ .load = .{ .ptr = p } });
}

test "vectorize: chained (a+b)*c keeps the intermediate in a vector (pack reuse)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // s = sum_i (a_i + b_i) * c_i. The add group feeds the mul group. Pack-reuse should wire
    // the vector add's result straight into the vector mul (no re-pack between them).
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    var a: [4]ir.function.Value = undefined;
    var b: [4]ir.function.Value = undefined;
    var cc: [4]ir.function.Value = undefined;
    for (0..4) |i| a[i] = try loadLane(&func, blk, ptr_t, f32_t, pa, i);
    for (0..4) |i| b[i] = try loadLane(&func, blk, ptr_t, f32_t, pb, i);
    for (0..4) |i| cc[i] = try loadLane(&func, blk, ptr_t, f32_t, pc, i);
    var t: [4]ir.function.Value = undefined;
    for (0..4) |i| t[i] = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = a[i], .rhs = b[i] } });
    var r: [4]ir.function.Value = undefined;
    for (0..4) |i| r[i] = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .mul, .lhs = t[i], .rhs = cc[i] } });
    var s = r[0];
    for (1..4) |i| s = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = r[i] } });
    func.setTerminator(blk, .{ .ret = s });

    const av = [4]f32{ 1, 2, 3, 4 };
    const bv = [4]f32{ 10, 20, 30, 40 };
    const cv = [4]f32{ 2, 2, 2, 2 };
    const Run = struct {
        fn call(al: std.mem.Allocator, fnc: *const Function, x: *const [4]f32, y: *const [4]f32, z: *const [4]f32) !f32 {
            const code = try isel.selectFunction(al, fnc);
            defer al.free(code);
            var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
            defer buf.deinit();
            const Fn = *const fn (*const [4]f32, *const [4]f32, *const [4]f32) callconv(.c) f32;
            return @as(Fn, @ptrCast(buf.memory.ptr))(x, y, z);
        }
    };
    const scalar_res = try Run.call(allocator, &func, &av, &bv, &cv); // (11*2)+(22*2)+(33*2)+(44*2) = 220

    try std.testing.expect(try opt.vectorize.run(allocator, &func));

    // The vector mul should take the vector add's result directly (the chain stayed vector).
    var vadd_res: ?ir.function.Value = null;
    var vmul_lhs: ?ir.function.Value = null;
    for (0..func.instCount()) |i| {
        const o = func.opcodeMut(@enumFromInt(i)).*;
        if (o != .arith) continue;
        const res = func.instResult(@enumFromInt(i)).?;
        if (func.types.type_kind(func.valueType(res)) != .vector) continue;
        if (o.arith.op == .add) vadd_res = res;
        if (o.arith.op == .mul) vmul_lhs = o.arith.lhs;
    }
    try std.testing.expect(vadd_res != null and vmul_lhs != null);
    try std.testing.expectEqual(vadd_res.?, vmul_lhs.?); // pack-reuse: add's vector feeds the mul

    // The intermediate add's lanes are extracted as register ops, pack-reuse makes them dead,
    // so DCE removes them. Only the final result's 4 extracts should survive.
    var analyses = opt.pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    _ = try opt.dce.run(allocator, &func, &analyses);
    var extracts: usize = 0;
    var allocas: usize = 0;
    for (func.blockInsts(blk)) |inst| {
        switch (func.opcode(inst)) {
            .extract => extracts += 1,
            .alloca => allocas += 1,
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 4), extracts); // the dead intermediate extracts are gone
    try std.testing.expectEqual(@as(usize, 0), allocas); // pack/unpack are register ops, no stack

    const vec_res = try Run.call(allocator, &func, &av, &bv, &cv);
    try std.testing.expectEqual(scalar_res, vec_res);
    try std.testing.expectEqual(@as(f32, 220), vec_res);
}

test "neon: <4 x f32> lane-wise add/mul through pointers" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // out = a * b + a, computed a full 4-wide vector at a time (one fmul + one fadd).
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pb } });
    const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const sum = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = va } });
    try func.appendStore(blk, sum, out);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();

    const Fn = *const fn (*[4]f32, *const [4]f32, *const [4]f32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const a align(16) = [4]f32{ 1, 2, 3, 4 };
    const b align(16) = [4]f32{ 10, 20, 30, 40 };
    var result align(16) = [4]f32{ 0, 0, 0, 0 };
    f(&result, &a, &b);
    try std.testing.expectEqual([4]f32{ 11, 42, 93, 164 }, result); // a*b + a per lane
}

test "native: signed division and remainder" {
    const allocator = std.testing.allocator;
    inline for (.{ .{ ir.function.BinOp.div, @as(i32, 6) }, .{ ir.function.BinOp.rem, @as(i32, 2) } }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, t);
        const y = try func.appendBlockParam(b, t);
        const r = try func.appendInst(b, t, .{ .arith = .{ .op = case[0], .lhs = x, .rhs = y } });
        func.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &func, &.{ 20, 3 }, case[1]); // 20/3 = 6, 20%3 = 2
    }
}

test "native: shifts (left and arithmetic right)" {
    const allocator = std.testing.allocator;
    inline for (.{ .{ ir.function.BinOp.shl, @as(i32, -16) }, .{ ir.function.BinOp.shr, @as(i32, -1) } }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, t);
        const y = try func.appendBlockParam(b, t);
        const r = try func.appendInst(b, t, .{ .arith = .{ .op = case[0], .lhs = x, .rhs = y } });
        func.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &func, &.{ -4, 2 }, case[1]); // -4<<2 = -16, -4>>2 = -1 (asr)
    }
}

test "native: select picks the smaller operand" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const c = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    const m = try func.appendInst(b, t, .{ .select = .{ .cond = c, .then = x, .@"else" = y } });
    func.setTerminator(b, .{ .ret = m });
    try expectRun(allocator, &func, &.{ 7, 3 }, 3); // min(7,3)
}

test "native: subtraction yields a negative result" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const d = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = d });
    try expectRun(allocator, &func, &.{ 3, 10 }, -7);
}

test "native: constants and immediate arithmetic" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    // (x + 100) ^ 0xFF00
    const c = try func.appendArithImm(b, t, .add, x, 100);
    const mask = try func.appendInst(b, t, .{ .iconst = 0xFF00 });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .bit_xor, .lhs = c, .rhs = mask } });
    func.setTerminator(b, .{ .ret = r });
    try expectRun(allocator, &func, &.{5}, (5 + 100) ^ 0xFF00);
}

test "native: a wide constant via movz/movk" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, t, .{ .iconst = 0x1234_5678 });
    func.setTerminator(b, .{ .ret = c });
    try expectRun(allocator, &func, &.{}, 0x1234_5678);
}

test "native: max via a conditional branch to two return blocks" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = a });
    func.setTerminator(else_b, .{ .ret = b });

    try expectRun(allocator, &func, &.{ 7, 3 }, 7);
    try expectRun(allocator, &func, &.{ 3, 4 }, 4);
}

test "native: max via a merge block with parameters" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const merge = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const z = try func.appendBlockParam(merge, t);
    // if a < b -> merge(b) else merge(a): the larger flows through the param.
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = merge, .args = &.{b} }, .{ .target = merge, .args = &.{a} });
    func.setTerminator(merge, .{ .ret = z });

    try expectRun(allocator, &func, &.{ 7, 3 }, 7);
    try expectRun(allocator, &func, &.{ 3, 4 }, 4);
}

test "fused: signed min via if(icmp lt) returns the smaller (fused compare-branch)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    // The icmp immediately precedes the if and is its only use, so it fuses to `cmp; b.lt`.
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = a }); // a < b -> a is the min
    func.setTerminator(else_b, .{ .ret = b });

    try expectRun(allocator, &func, &.{ 7, 3 }, 3);
    try expectRun(allocator, &func, &.{ -5, 2 }, -5); // signed: -5 < 2
    try expectRun(allocator, &func, &.{ 4, 4 }, 4); // equal -> not <, takes else (b)
}

test "fused: unsigned max via if(icmp ult) uses the unsigned condition (b.cc, not b.lt)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // Unsigned operands: the fused branch must pick the unsigned condition (lo/hs), so a
    // large unsigned value (0xFFFFFFFF) compares GREATER than 1, unlike the signed reading.
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, u32_t);
    const b = try func.appendBlockParam(entry, u32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = b }); // a <u b -> b is the max
    func.setTerminator(else_b, .{ .ret = a });

    // a = -1 is 0xFFFFFFFF unsigned: NOT < 1, so the else edge returns a (0xFFFFFFFF == -1 as i32).
    try expectRun(allocator, &func, &.{ -1, 1 }, -1);
    try expectRun(allocator, &func, &.{ 1, 2 }, 2); // 1 <u 2 -> b = 2
    try expectRun(allocator, &func, &.{ 9, 9 }, 9); // equal -> else -> a
}

test "fused: equality branch via if(icmp eq) selects the right arm" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    const yes = try func.appendInst(then_b, t, .{ .iconst = 100 });
    func.setTerminator(then_b, .{ .ret = yes });
    const no = try func.appendInst(else_b, t, .{ .iconst = 200 });
    func.setTerminator(else_b, .{ .ret = no });

    try expectRun(allocator, &func, &.{ 5, 5 }, 100); // equal
    try expectRun(allocator, &func, &.{ 5, 6 }, 200); // not equal
}

test "fused: structural - if(icmp) emits a conditional branch and no cset boolean" {
    // Prove fusion fired: the compiled listing has a `b.<cc>` compare-branch and no `cset`
    // (the boolean materialization is skipped). Tolerant string checks, the point is that a
    // single-use integer icmp feeding an if no longer produces cset;cbnz.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = a });
    func.setTerminator(else_b, .{ .ret = b });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "cmp w0, w1") != null); // the compare
    try std.testing.expect(std.mem.indexOf(u8, text, "b.gt ") != null); // fused conditional branch
    try std.testing.expect(std.mem.indexOf(u8, text, "cset") == null); // no boolean materialized
    try std.testing.expect(std.mem.indexOf(u8, text, "cbnz") == null); // no re-test of a boolean
}

test "fused: a multi-use icmp does NOT fuse (boolean still materialized, correct result)" {
    // The icmp result feeds BOTH a select and the if condition, so it is not single-use:
    // fusion must be declined and the boolean materialized (a `cset`), or the select would
    // read a value that was never produced. The results must match the non-fused semantics.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const xtb = try func.appendBlockParam(tb, t);
    const xeb = try func.appendBlockParam(eb, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    const m = try func.appendInst(entry, t, .{ .select = .{ .cond = c, .then = a, .@"else" = b } }); // min, uses c
    try func.appendIf(entry, c, .{ .target = tb, .args = &.{m} }, .{ .target = eb, .args = &.{m} });
    const inc = try func.appendArithImm(tb, t, .add, xtb, 1);
    func.setTerminator(tb, .{ .ret = inc });
    func.setTerminator(eb, .{ .ret = xeb });

    // A multi-use icmp must keep the cset, proving the eligibility gate declines fusion here.
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "cset") != null);

    try expectRun(allocator, &func, &.{ 3, 7 }, 4); // c true: m=min=3, tb -> 3+1
    try expectRun(allocator, &func, &.{ 7, 3 }, 3); // c false: m=min=3, eb -> 3
}

test "fused: an icmp not immediately before the if does NOT fuse (intervening instruction)" {
    // An instruction sits between the icmp and the if, so the icmp is not the immediately
    // preceding instruction: fusion is declined and the standard cset;cbnz path runs.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const tb = try func.appendBlock();
    const eb = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    const sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } }); // intervening
    try func.appendIf(entry, c, .{ .target = tb, .args = &.{sum} }, .{ .target = eb, .args = &.{sum} });
    const xtb = try func.appendBlockParam(tb, t);
    const xeb = try func.appendBlockParam(eb, t);
    func.setTerminator(tb, .{ .ret = xtb }); // a > b -> a+b
    func.setTerminator(eb, .{ .ret = xeb });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "cset") != null); // not immediately preceding -> no fusion

    try expectRun(allocator, &func, &.{ 7, 3 }, 10); // both edges return a+b here
    try expectRun(allocator, &func, &.{ 3, 7 }, 10);
}

/// A single-use icmp immediately preceding an if (min(a, b) via `a < b`), the same shape as
/// the "fused: structural" test above but built once so both the `caps.fuse_cmp_branch = false`
/// and `= true` tests below share it. Returns the compiled `Compiled` so callers can disassemble
/// and JIT-run with an explicit `ModelCaps`.
fn compileMinIfWithCaps(allocator: std.mem.Allocator, func: *Function, caps: isel.ModelCaps) !isel.Compiled {
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = a }); // a < b -> a is the min
    func.setTerminator(else_b, .{ .ret = b });
    return isel.compileFunction(allocator, func, caps);
}

test "caps.fuse_cmp_branch = false falls back to cset;cbnz, not the fused cmp;b.lt (correct result)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const compiled = try compileMinIfWithCaps(allocator, &func, .{ .fuse_cmp_branch = false });
    defer allocator.free(compiled.relocs);
    defer allocator.free(compiled.lines);
    defer allocator.free(compiled.code);

    const text = try disasm.format(allocator, compiled.code);
    defer allocator.free(text);
    // The gate declined the fusion: the icmp materializes its boolean (`cset`) and the if
    // re-tests it (`cbnz`), NOT the fused `cmp; b.lt` this same shape produces when the flag
    // is on (see the byte-identical-default test right below).
    try std.testing.expect(std.mem.indexOf(u8, text, "cset") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cbnz") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.lt ") == null);

    if (builtin.cpu.arch == .aarch64) {
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer buf.deinit();
        try std.testing.expectEqual(@as(i32, 3), try callI32(&buf, &.{ 7, 3 }));
        try std.testing.expectEqual(@as(i32, -5), try callI32(&buf, &.{ -5, 2 }));
        try std.testing.expectEqual(@as(i32, 4), try callI32(&buf, &.{ 4, 4 })); // equal -> else (b)
    }
}

test "caps.fuse_cmp_branch = true (the default) emits the fused cmp;b.lt with no cset" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const compiled = try compileMinIfWithCaps(allocator, &func, .{ .fuse_cmp_branch = true });
    defer allocator.free(compiled.relocs);
    defer allocator.free(compiled.lines);
    defer allocator.free(compiled.code);

    const text = try disasm.format(allocator, compiled.code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.lt ") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cset") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cbnz") == null);

    if (builtin.cpu.arch == .aarch64) {
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer buf.deinit();
        try std.testing.expectEqual(@as(i32, 3), try callI32(&buf, &.{ 7, 3 }));
        try std.testing.expectEqual(@as(i32, -5), try callI32(&buf, &.{ -5, 2 }));
        try std.testing.expectEqual(@as(i32, 4), try callI32(&buf, &.{ 4, 4 }));
    }
}

/// Build `f(a, b) -> i(bits)` computing an `add`/`sub` whose one operand is `b << k`, with
/// the shift immediately preceding the add/sub and used exactly once - the exact shape
/// `fusesIntoNextShiftAdd` folds into a single `add/sub xd, xn, xm, lsl #k`. `shift_on_lhs`
/// puts the `(b << k)` operand on the left of the add/sub (`(b << k) op a`), else on the
/// right (`a op (b << k)`).
fn buildShiftAdd(allocator: std.mem.Allocator, bits: u16, op: ir.function.BinOp, k: i64, shift_on_lhs: bool) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = bits } });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, t);
    const b = try func.appendBlockParam(blk, t);
    const sh = try func.appendInst(blk, t, .{ .arith_imm = .{ .op = .shl, .lhs = b, .imm = k } });
    const r = if (shift_on_lhs)
        try func.appendInst(blk, t, .{ .arith = .{ .op = op, .lhs = sh, .rhs = a } })
    else
        try func.appendInst(blk, t, .{ .arith = .{ .op = op, .lhs = a, .rhs = sh } });
    func.setTerminator(blk, .{ .ret = r });
    return func;
}

test "aarch64 shift_add: x + (b<<3) folds to add-shift and computes correctly" {
    const allocator = std.testing.allocator;
    var func = try buildShiftAdd(allocator, 32, .add, 3, false); // a + (b << 3)
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    // Folded: one shifted-register add, and NO separate register shift (a plain-path `lslv`
    // would disassemble as `lsl w.., w.., w..` with no `#` immediate).
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #3") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "lsl w") == null);

    try expectRun(allocator, &func, &.{ 5, 4 }, 5 + (4 << 3)); // 37
    try expectRun(allocator, &func, &.{ 0, 1 }, 8);
    try expectRun(allocator, &func, &.{ -10, 2 }, -10 + (2 << 3)); // 6
}

test "aarch64 shift_add: x - (b<<2) folds to sub-shift (shl is the subtrahend)" {
    const allocator = std.testing.allocator;
    var func = try buildShiftAdd(allocator, 32, .sub, 2, false); // a - (b << 2)
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "sub w") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "lsl w") == null);

    try expectRun(allocator, &func, &.{ 100, 4 }, 100 - (4 << 2)); // 84
    try expectRun(allocator, &func, &.{ 0, 3 }, -(3 << 2)); // -12
}

test "aarch64 shift_add: (b<<3) + x also folds (add is commutative in the shl operand)" {
    const allocator = std.testing.allocator;
    var func = try buildShiftAdd(allocator, 32, .add, 3, true); // (b << 3) + a
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #3") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "lsl w") == null);

    try expectRun(allocator, &func, &.{ 5, 4 }, (4 << 3) + 5); // 37
    try expectRun(allocator, &func, &.{ 7, 0 }, 7);
}

/// Compile `func` with an explicit `ModelCaps` and return the `Compiled` so a caller can
/// disassemble and JIT-run it under a chosen `fuse_shift_add` setting (mirrors
/// `compileMinIfWithCaps`).
fn compileWithCaps(allocator: std.mem.Allocator, func: *const Function, caps: isel.ModelCaps) !isel.Compiled {
    return isel.compileFunction(allocator, func, caps);
}

test "aarch64 shift_add off (caps flag false) emits plain shift+add byte-identically and stays correct" {
    const allocator = std.testing.allocator;
    var func = try buildShiftAdd(allocator, 32, .add, 3, false); // a + (b << 3)
    defer func.deinit();

    const compiled = try compileWithCaps(allocator, &func, .{ .fuse_shift_add = false });
    defer allocator.free(compiled.relocs);
    defer allocator.free(compiled.lines);
    defer allocator.free(compiled.code);

    const text = try disasm.format(allocator, compiled.code);
    defer allocator.free(text);
    // The gate declined the fold: a separate register shift (`lsl w..`) materializes the
    // shifted value and a plain `add` sums it, so NO shifted-register form appears.
    try std.testing.expect(std.mem.indexOf(u8, text, "lsl w") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #") == null);

    if (builtin.cpu.arch == .aarch64) {
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer buf.deinit();
        try std.testing.expectEqual(@as(i32, 5 + (4 << 3)), try callI32(&buf, &.{ 5, 4 }));
        try std.testing.expectEqual(@as(i32, -10 + (2 << 3)), try callI32(&buf, &.{ -10, 2 }));
    }
}

test "aarch64 shift_add: shl result used twice is not folded (plain shift+add), correct result" {
    const allocator = std.testing.allocator;
    // t = b << 3; r = (a + t) + t = a + 2*(b << 3). The first add immediately follows the
    // shl, but `t` has two uses, so the single-use gate declines and both adds take the
    // plain path.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, t);
    const b = try func.appendBlockParam(blk, t);
    const sh = try func.appendInst(blk, t, .{ .arith_imm = .{ .op = .shl, .lhs = b, .imm = 3 } });
    const s1 = try func.appendInst(blk, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = sh } });
    const s2 = try func.appendInst(blk, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = sh } });
    func.setTerminator(blk, .{ .ret = s2 });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #") == null); // not folded
    try std.testing.expect(std.mem.indexOf(u8, text, "lsl w") != null); // plain register shift

    try expectRun(allocator, &func, &.{ 5, 4 }, 5 + 2 * (4 << 3)); // 69
}

test "aarch64 shift_add: (b<<2) - x is NOT folded (subShifted cannot express it), correct result" {
    const allocator = std.testing.allocator;
    var func = try buildShiftAdd(allocator, 32, .sub, 2, true); // (b << 2) - a, shl on the LHS
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    // subShifted computes rn - (rm << k), so the subtrahend must be the shifted operand.
    // Here the shift is the minuend, so the fold is declined and a plain shift + sub is used.
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "lsl w") != null);

    try expectRun(allocator, &func, &.{ 5, 4 }, (4 << 2) - 5); // 11
}

test "aarch64 shift_add: 32-bit shift beyond the width (k>31) is not folded" {
    const allocator = std.testing.allocator;
    var func = try buildShiftAdd(allocator, 32, .add, 32, false); // a + (b << 32), out of imm6 range for w-form
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    // k = 32 exceeds a 32-bit result's 0..31 range, so the fold is declined (no shifted-
    // register add) and the plain shift + add path is used.
    try std.testing.expect(std.mem.indexOf(u8, text, ", lsl #") == null);
}

test "aarch64 shift_add: a function without the pattern is byte-identical flag-on vs flag-off" {
    const allocator = std.testing.allocator;
    // A shl feeding a MUL (not an add/sub) is never foldable, so both flag settings must
    // produce byte-identical code.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, t);
    const b = try func.appendBlockParam(blk, t);
    const sh = try func.appendInst(blk, t, .{ .arith_imm = .{ .op = .shl, .lhs = b, .imm = 3 } });
    const r = try func.appendInst(blk, t, .{ .arith = .{ .op = .mul, .lhs = sh, .rhs = a } });
    func.setTerminator(blk, .{ .ret = r });

    const on = try compileWithCaps(allocator, &func, .{ .fuse_shift_add = true });
    defer allocator.free(on.relocs);
    defer allocator.free(on.lines);
    defer allocator.free(on.code);
    const off = try compileWithCaps(allocator, &func, .{ .fuse_shift_add = false });
    defer allocator.free(off.relocs);
    defer allocator.free(off.lines);
    defer allocator.free(off.code);
    try std.testing.expectEqualSlices(u32, on.code, off.code);
}

/// The three fused shapes `fusesIntoNextArith` recognizes: `add`: a*b+c -> fmadd. `sub`:
/// a*b-c -> fnmsub. `csub`: c-a*b -> fmsub (see isel.zig's `Ctx.emitFusedArith`).
const FmaShape = enum { add, sub, csub };

/// `f(a, b, c)` computing `shape` in `dbl` precision (f64 if true, else f32), with the
/// multiply immediately preceding its single consuming add/sub - exactly the shape
/// `fusesIntoNextArith` fuses into one fmadd/fmsub/fnmsub.
fn buildFmaFunc(allocator: std.mem.Allocator, dbl: bool, shape: FmaShape) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ft = try func.types.intern(.{ .float = if (dbl) .f64 else .f32 });
    const b = try func.appendBlock();
    const a_p = try func.appendBlockParam(b, ft);
    const b_p = try func.appendBlockParam(b, ft);
    const c_p = try func.appendBlockParam(b, ft);
    const prod = try func.appendInst(b, ft, .{ .arith = .{ .op = .mul, .lhs = a_p, .rhs = b_p } });
    const r = switch (shape) {
        .add => try func.appendInst(b, ft, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = c_p } }),
        .sub => try func.appendInst(b, ft, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = c_p } }),
        .csub => try func.appendInst(b, ft, .{ .arith = .{ .op = .sub, .lhs = c_p, .rhs = prod } }),
    };
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// Call a JIT-mapped `f(a, b, c) -> T` (f32 or f64, all in FP argument/return registers).
fn callFma3(comptime T: type, buf: *const jit.CodeBuffer, a: T, b: T, c: T) T {
    const ptr = buf.memory.ptr;
    return @as(*const fn (T, T, T) callconv(.c) T, @ptrCast(ptr))(a, b, c);
}

/// The unsigned integer type with T's bit width, for bit-exact (not `==`) float comparison.
fn Bits(comptime T: type) type {
    return std.meta.Int(.unsigned, @bitSizeOf(T));
}

/// Build, JIT, and run the FMA `shape` in precision `T` on the operands given as raw bit
/// patterns (exact - avoids any decimal-literal round-trip drift), and assert three things:
///   1. The JIT result is bit-identical to the hardware FMA reference `@mulAdd` (proves the
///      variant mapping and single-rounding are both correct - see the comment above each
///      call site for which `@mulAdd` form is the correct oracle for each shape).
///   2. That result DIFFERS from the naive separately-rounded computation. The operands were
///      searched specifically so fused != separate; this is what proves fusion actually fired
///      (a bug that silently fell back to separate fmul+fadd would still pass check 1 only by
///      coincidence on generic inputs, but never on these).
///   3. The emitted code contains `mnemonic` (the expected 3-source instruction) and no
///      separate fmul/fadd/fsub for the fused pair.
fn checkFma(comptime T: type, shape: FmaShape, a_bits: Bits(T), b_bits: Bits(T), c_bits: Bits(T), want: T, mnemonic: []const u8) !void {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildFmaFunc(allocator, T == f64, shape);
    defer func.deinit();
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();

    const a: T = @bitCast(a_bits);
    const b: T = @bitCast(b_bits);
    const c: T = @bitCast(c_bits);
    const got = callFma3(T, &buf, a, b, c);
    try std.testing.expectEqual(@as(Bits(T), @bitCast(want)), @as(Bits(T), @bitCast(got)));

    const naive = switch (shape) {
        .add => a * b + c,
        .sub => a * b - c,
        .csub => c - a * b,
    };
    try std.testing.expect(@as(Bits(T), @bitCast(naive)) != @as(Bits(T), @bitCast(got)));

    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, mnemonic) != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fmul") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fadd") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fsub") == null);
}

// Operand triples below were found by random search specifically because the separately-
// rounded computation (a*b then +/-c as two instructions) differs from the fused one, so
// each test would fail if fusion did not fire, or fired with the wrong variant.

test "fma: scalar f32 a*b+c matches @mulAdd bit-exactly and fuses to fmadd" {
    const a: f32 = @bitCast(@as(u32, 0xc40ac54c));
    const b: f32 = @bitCast(@as(u32, 0x43a8f6ea));
    const c: f32 = @bitCast(@as(u32, 0xc28c5708));
    try checkFma(f32, .add, 0xc40ac54c, 0x43a8f6ea, 0xc28c5708, @mulAdd(f32, a, b, c), "fmadd");
}

test "fma: scalar f32 a*b-c matches @mulAdd bit-exactly and fuses to fnmsub" {
    const a: f32 = @bitCast(@as(u32, 0xc44a1c46));
    const b: f32 = @bitCast(@as(u32, 0x44559447));
    const c: f32 = @bitCast(@as(u32, 0x43b99ce8));
    try checkFma(f32, .sub, 0xc44a1c46, 0x44559447, 0x43b99ce8, @mulAdd(f32, a, b, -c), "fnmsub");
}

test "fma: scalar f32 c-a*b matches @mulAdd bit-exactly and fuses to fmsub" {
    const a: f32 = @bitCast(@as(u32, 0xc44a1c46));
    const b: f32 = @bitCast(@as(u32, 0x44559447));
    const c: f32 = @bitCast(@as(u32, 0x43b99ce8));
    try checkFma(f32, .csub, 0xc44a1c46, 0x44559447, 0x43b99ce8, @mulAdd(f32, -a, b, c), "fmsub");
}

test "fma: scalar f64 a*b+c matches @mulAdd bit-exactly and fuses to fmadd" {
    const a: f64 = @bitCast(@as(u64, 0x404e1b3cea04d64a));
    const b: f64 = @bitCast(@as(u64, 0x406635dc84d3c228));
    const c: f64 = @bitCast(@as(u64, 0xc084ec8dcc159ab8));
    try checkFma(f64, .add, 0x404e1b3cea04d64a, 0x406635dc84d3c228, 0xc084ec8dcc159ab8, @mulAdd(f64, a, b, c), "fmadd");
}

test "fma: scalar f64 a*b-c matches @mulAdd bit-exactly and fuses to fnmsub" {
    const a: f64 = @bitCast(@as(u64, 0x40891218249a6341));
    const b: f64 = @bitCast(@as(u64, 0x407a4101d38c144d));
    const c: f64 = @bitCast(@as(u64, 0xc08c85a4101e5516));
    try checkFma(f64, .sub, 0x40891218249a6341, 0x407a4101d38c144d, 0xc08c85a4101e5516, @mulAdd(f64, a, b, -c), "fnmsub");
}

test "fma: scalar f64 c-a*b matches @mulAdd bit-exactly and fuses to fmsub" {
    const a: f64 = @bitCast(@as(u64, 0x40891218249a6341));
    const b: f64 = @bitCast(@as(u64, 0x407a4101d38c144d));
    const c: f64 = @bitCast(@as(u64, 0xc08c85a4101e5516));
    try checkFma(f64, .csub, 0x40891218249a6341, 0x407a4101d38c144d, 0xc08c85a4101e5516, @mulAdd(f64, -a, b, c), "fmsub");
}

test "fma: a multi-use mul does NOT fuse (separate fmul+fadd, correct result)" {
    // The product feeds the fusible add AND is read again afterward, so it is not single-use:
    // fusion must be declined for both consumers and the multiply stays materialized.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ft = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, ft);
    const b = try func.appendBlockParam(blk, ft);
    const c = try func.appendBlockParam(blk, ft);
    const prod = try func.appendInst(blk, ft, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const s = try func.appendInst(blk, ft, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = c } }); // a*b+c, fusible shape...
    const r = try func.appendInst(blk, ft, .{ .arith = .{ .op = .add, .lhs = s, .rhs = prod } }); // ...but prod is reused here
    func.setTerminator(blk, .{ .ret = r });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "fmul") != null); // the multiply is still materialized
    try std.testing.expect(std.mem.indexOf(u8, text, "fadd") != null); // and added separately, twice
    try std.testing.expect(std.mem.indexOf(u8, text, "fmadd") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fmsub") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fnmsub") == null);

    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const got = callFma3(f32, &buf, 2.0, 3.0, 5.0); // prod=6, s=6+5=11, r=11+6=17
    try std.testing.expectEqual(@as(f32, 17.0), got);
}

/// The two vector shapes `fusesIntoNextArith` recognizes for NEON FMLA/FMLS: `add`:
/// a*b+c -> fmla. `csub`: c-a*b -> fmls. The third scalar shape, a*b-c, has no single NEON
/// instruction (FMLA/FMLS only ever add or subtract the product, never negate the whole
/// result), so `fusesIntoNextArith` never fuses it for a vector - it stays separate
/// fmul+fsub and is covered by its own non-fusing test below.
const VecFmaShape = enum { add, csub };

/// `f(out, pa, pb, pc)` loading three `<4 x f32>` vectors through pointers, computing
/// `shape` a full vector at a time, and storing the result through `out` - mirrors the
/// "neon: <4 x f32> lane-wise add/mul through pointers" test's pointer-argument style, with
/// the multiply immediately preceding its single consuming add/sub so `fusesIntoNextArith`
/// fuses it into one fmla/fmls.
fn buildVecFmaFunc(allocator: std.mem.Allocator, shape: VecFmaShape) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pb } });
    const vc = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pc } });
    const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const r = switch (shape) {
        .add => try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = vc } }),
        .csub => try func.appendInst(blk, v4, .{ .arith = .{ .op = .sub, .lhs = vc, .rhs = prod } }),
    };
    try func.appendStore(blk, r, out);
    func.setTerminator(blk, .{ .ret = null });
    return func;
}

/// Asserts `code` contains a word matching `template`'s fixed opcode bits (rd/rn/rm masked
/// off, since the allocator picks the actual registers) - mirrors `expectHasDot`.
fn expectHasVecWord(code: []const u32, template: u32) !void {
    const reg_mask: u32 = 0x001F03FF; // rd[4:0] | rn[9:5] | rm[20:16]
    const fixed = template & ~reg_mask;
    for (code) |w| if (w & ~reg_mask == fixed) return;
    return error.TestExpectedEqual;
}

/// Asserts `code` contains no word matching `template`'s fixed opcode bits.
fn expectNoVecWord(code: []const u32, template: u32) !void {
    const reg_mask: u32 = 0x001F03FF;
    const fixed = template & ~reg_mask;
    for (code) |w| try std.testing.expect(w & ~reg_mask != fixed);
}

/// Build, JIT, and run the vector FMA `shape` on four `<4 x f32>`-worth of raw bit-pattern
/// operand triples (one per lane - exact, avoids any decimal-literal round-trip drift), and
/// assert per lane:
///   1. The JIT result is bit-identical to the hardware FMA reference `@mulAdd` (proves the
///      fused instruction and its operand order are both correct).
///   2. That result DIFFERS from the naive separately-rounded computation, proving fusion
///      actually fired (these triples are the same ones the scalar fma tests above verified
///      diverge between fused and separate rounding).
/// Also asserts structurally that the emitted code contains the fused instruction and NOT a
/// separate fmul+fadd/fsub pair.
fn checkVecFma(shape: VecFmaShape, abits: [4]u32, bbits: [4]u32, cbits: [4]u32, mnemonic_template: u32) !void {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildVecFmaFunc(allocator, shape);
    defer func.deinit();
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);

    try expectHasVecWord(code, mnemonic_template);
    try expectNoVecWord(code, encode.fmulVec(.x0, .x0, .x0));
    try expectNoVecWord(code, encode.faddVec(.x0, .x0, .x0));
    try expectNoVecWord(code, encode.fsubVec(.x0, .x0, .x0));

    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [4]f32, *const [4]f32, *const [4]f32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);

    var a: [4]f32 align(16) = undefined;
    var b: [4]f32 align(16) = undefined;
    var c: [4]f32 align(16) = undefined;
    for (0..4) |i| {
        a[i] = @bitCast(abits[i]);
        b[i] = @bitCast(bbits[i]);
        c[i] = @bitCast(cbits[i]);
    }
    var got: [4]f32 align(16) = undefined;
    f(&got, &a, &b, &c);

    for (0..4) |i| {
        const want = switch (shape) {
            .add => @mulAdd(f32, a[i], b[i], c[i]),
            .csub => @mulAdd(f32, -a[i], b[i], c[i]),
        };
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got[i])));
        const naive = switch (shape) {
            .add => a[i] * b[i] + c[i],
            .csub => c[i] - a[i] * b[i],
        };
        try std.testing.expect(@as(u32, @bitCast(naive)) != @as(u32, @bitCast(got[i])));
    }
}

test "neon fma: vector f32 a*b+c matches @mulAdd bit-exactly per lane and fuses to fmla" {
    // Same operand triple (verified to diverge between fused and separate rounding by the
    // scalar "fma: scalar f32 a*b+c" test above) replicated across all four lanes, so every
    // lane would fail without fusion, not just one.
    try checkVecFma(
        .add,
        .{0xc40ac54c} ** 4,
        .{0x43a8f6ea} ** 4,
        .{0xc28c5708} ** 4,
        encode.fmlaVec(.x0, .x0, .x0),
    );
}

test "neon fma: vector f32 c-a*b matches @mulAdd bit-exactly per lane and fuses to fmls" {
    // Same operand triple as the scalar "fma: scalar f32 c-a*b" test above.
    try checkVecFma(
        .csub,
        .{0xc44a1c46} ** 4,
        .{0x44559447} ** 4,
        .{0x43b99ce8} ** 4,
        encode.fmlsVec(.x0, .x0, .x0),
    );
}

test "neon fma: a multi-use vector mul does NOT fuse (separate fmul+fadd, correct result)" {
    // Mirrors "fma: a multi-use mul does NOT fuse" above but for a vector: the product feeds
    // the fusible add AND is read again afterward, so it is not single-use and stays
    // materialized as a separate fmul + two fadds.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pb } });
    const vc = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pc } });
    const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const s = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = vc } }); // a*b+c, fusible shape...
    const r = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = s, .rhs = prod } }); // ...but prod is reused here
    try func.appendStore(blk, r, out);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    try expectHasVecWord(code, encode.fmulVec(.x0, .x0, .x0)); // the multiply is still materialized
    try expectNoVecWord(code, encode.fmlaVec(.x0, .x0, .x0));
    try expectNoVecWord(code, encode.fmlsVec(.x0, .x0, .x0));

    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [4]f32, *const [4]f32, *const [4]f32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const a align(16) = [4]f32{ 2, 2, 2, 2 };
    const b align(16) = [4]f32{ 3, 3, 3, 3 };
    const c align(16) = [4]f32{ 5, 5, 5, 5 };
    var got: [4]f32 align(16) = undefined;
    f(&got, &a, &b, &c); // prod=6, s=6+5=11, r=11+6=17 per lane
    try std.testing.expectEqual([4]f32{ 17, 17, 17, 17 }, got);
}

test "neon fma: vector a*b-c does NOT fuse (no single NEON instruction expresses it)" {
    // fusesIntoNextArith must reject this shape for a vector (unlike scalar, which fuses it
    // to fnmsub): NEON FMLA/FMLS only ever add or subtract the product, never negate the
    // whole result, so this must stay a separate fmul + fsub and still compute correctly.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    var func = try buildVecFmaFuncSub(alloc);
    defer func.deinit();

    const code = try isel.selectFunction(alloc, &func);
    defer alloc.free(code);
    try expectHasVecWord(code, encode.fmulVec(.x0, .x0, .x0));
    try expectHasVecWord(code, encode.fsubVec(.x0, .x0, .x0));
    try expectNoVecWord(code, encode.fmlaVec(.x0, .x0, .x0));
    try expectNoVecWord(code, encode.fmlsVec(.x0, .x0, .x0));

    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [4]f32, *const [4]f32, *const [4]f32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const a align(16) = [4]f32{ 2, 3, 4, 5 };
    const b align(16) = [4]f32{ 10, 10, 10, 10 };
    const c align(16) = [4]f32{ 1, 1, 1, 1 };
    var got: [4]f32 align(16) = undefined;
    f(&got, &a, &b, &c); // a*b - c per lane: 19, 29, 39, 49
    try std.testing.expectEqual([4]f32{ 19, 29, 39, 49 }, got);
}

/// `f(out, pa, pb, pc) = a*b - c`, a full vector at a time - the one shape that must never
/// fuse for a vector (see `fusesIntoNextArith`).
fn buildVecFmaFuncSub(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pb } });
    const vc = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pc } });
    const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const r = try func.appendInst(blk, v4, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = vc } }); // a*b - c
    try func.appendStore(blk, r, out);
    func.setTerminator(blk, .{ .ret = null });
    return func;
}

test "native: a non-leaf function calls another and uses the result" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee(a) = a * 2   (leaf)
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(t_kind);
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        const two = try callee.appendInst(b, t, .{ .iconst = 2 });
        const r = try callee.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = two } });
        callee.setTerminator(b, .{ .ret = r });
    }
    // caller(x) = callee(x) + 1   (non-leaf: saves fp/lr + a callee-saved reg)
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(t_kind);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const call = try caller.appendCall(b, t, "callee", &.{x});
        const r = try caller.appendArithImm(b, t, .add, call, 1);
        caller.setTerminator(b, .{ .ret = r });
    }

    // caller(5) = (5*2) + 1 = 11.
    try std.testing.expectEqual(@as(i32, 11), try runModule(allocator, &.{
        .{ .name = "caller", .func = &caller },
        .{ .name = "callee", .func = &callee },
    }, &.{5}));
}

test "native: a non-leaf function with a value live across a call" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a   (leaf)
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(t_kind);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    // f(x) = dbl(x) + x   (x is live across the call, so it must survive in a
    // callee-saved register)
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const t = try f.types.intern(t_kind);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const d = try f.appendCall(b, t, "dbl", &.{x});
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = d, .rhs = x } });
        f.setTerminator(b, .{ .ret = r });
    }

    // f(10) = (10+10) + 10 = 30.
    try std.testing.expectEqual(@as(i32, 30), try runModule(allocator, &.{
        .{ .name = "f", .func = &f },
        .{ .name = "dbl", .func = &dbl },
    }, &.{10}));
}

test "native: a counted loop sums 0..n (back-edge)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    try expectRun(allocator, &func, &.{5}, 10); // 0+1+2+3+4
}

test "native: loop-header alignment pads with nops but never changes the result" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    const unaligned = try isel.selectFunction(allocator, &func);
    defer allocator.free(unaligned);
    const aligned = try isel.selectFunctionAligned(allocator, &func, 32);
    defer allocator.free(aligned);

    // The loop header (`loop`, with a real back-edge from `body`) gets padded to a
    // 32-byte boundary, so the aligned build is strictly longer than the unaligned one.
    try std.testing.expect(aligned.len > unaligned.len);

    var unaligned_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(unaligned));
    defer unaligned_buf.deinit();
    var aligned_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(aligned));
    defer aligned_buf.deinit();

    // Alignment is a placement hint only: every input must produce the identical result
    // whether the loop header was padded or not.
    for ([_]i32{ 0, 1, 5, 20 }) |input| {
        const want = try callI32(&unaligned_buf, &.{input});
        const got = try callI32(&aligned_buf, &.{input});
        try std.testing.expectEqual(want, got);
    }
}

test "native: selectFunctionForModel fires the alignment hook from ampere-altra without changing results" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    const plain = try isel.selectFunction(allocator, &func);
    defer allocator.free(plain);
    const model = opt.microarch.modelFor(.@"ampere-altra");
    const tuned = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(tuned);

    // ampere-altra's fetch_align (32) is above the 4-byte no-op threshold, so the model
    // seam pads the loop header, same as calling selectFunctionAligned directly: the
    // model-compiled build is never shorter than the plain one.
    try std.testing.expect(tuned.len >= plain.len);
    try std.testing.expect(tuned.len > plain.len);

    var plain_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(plain));
    defer plain_buf.deinit();
    var tuned_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(tuned));
    defer tuned_buf.deinit();

    // The model seam only changes where code lands, never what it computes.
    for ([_]i32{ 0, 1, 5, 20 }) |input| {
        const want = try callI32(&plain_buf, &.{input});
        const got = try callI32(&tuned_buf, &.{input});
        try std.testing.expectEqual(want, got);
    }
}

test "native: register spilling under high pressure (leaf)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const p0 = try func.appendBlockParam(e, t);
    const p1 = try func.appendBlockParam(e, t);

    // 20 values all live at once (defined before any is consumed) exhausts the
    // ~10-register leaf pool, forcing spills. Folding them back exercises reloads.
    var vals: [20]ir.function.Value = undefined;
    for (&vals) |*v| v.* = try func.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(e, .{ .ret = acc });

    // Each value is p0 + p1 = 2, summing 20 of them = 40.
    try expectRun(allocator, &func, &.{ 1, 1 }, 40);
}

test "native: register spilling under high pressure (non-leaf)" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // id(a) = a
    var id = Function.init(allocator);
    defer id.deinit();
    {
        const t = try id.types.intern(t_kind);
        const b = try id.appendBlock();
        const a = try id.appendBlockParam(b, t);
        id.setTerminator(b, .{ .ret = a });
    }
    // f(p0, p1): 15 values live across a call to id, then folded. The call makes
    // this non-leaf (callee-saved pool of ~10), so the long-lived values spill.
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const t = try f.types.intern(t_kind);
        const e = try f.appendBlock();
        const p0 = try f.appendBlockParam(e, t);
        const p1 = try f.appendBlockParam(e, t);
        var vals: [15]ir.function.Value = undefined;
        for (&vals) |*v| v.* = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
        const r = try f.appendCall(e, t, "id", &.{p0}); // clobbers caller-saved regs
        var acc = r;
        for (vals) |v| acc = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
        f.setTerminator(e, .{ .ret = acc });
    }
    // f(1,1) = id(1) + 15*(1+1) = 1 + 30 = 31.
    try std.testing.expectEqual(@as(i32, 31), try runModule(allocator, &.{
        .{ .name = "f", .func = &f },
        .{ .name = "id", .func = &id },
    }, &.{ 1, 1 }));
}

test "native: a call with ten arguments (stack args)" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee(a0..a9) = a0 + a1 + ... + a9 (a8, a9 arrive on the stack)
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(t_kind);
        const b = try callee.appendBlock();
        var ps: [10]ir.function.Value = undefined;
        for (&ps) |*p| p.* = try callee.appendBlockParam(b, t);
        var sum = ps[0];
        for (ps[1..]) |p| sum = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = p } });
        callee.setTerminator(b, .{ .ret = sum });
    }
    // caller() = callee(1, 2, ..., 10)
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(t_kind);
        const b = try caller.appendBlock();
        var args: [10]ir.function.Value = undefined;
        for (&args, 0..) |*a, i| a.* = try caller.appendInst(b, t, .{ .iconst = @intCast(i + 1) });
        const r = try caller.appendCall(b, t, "callee", &args);
        caller.setTerminator(b, .{ .ret = r });
    }
    // 1 + 2 + ... + 10 = 55.
    try std.testing.expectEqual(@as(i32, 55), try runModule(allocator, &.{
        .{ .name = "caller", .func = &caller },
        .{ .name = "callee", .func = &callee },
    }, &.{}));
}

test "native: alloca stores and reloads through a stack frame" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, t);
    const slot = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(e, x, slot);
    const v = try func.appendInst(e, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(e, .{ .ret = v });
    try expectRun(allocator, &func, &.{42}, 42);
}

test "native: sub-word store and sign-extending load (i8)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const a = try func.appendBlockParam(e, i8_t);
    const slot = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i8_t } });
    try func.appendStore(e, a, slot); // strb (low byte)
    const v = try func.appendInst(e, i8_t, .{ .load = .{ .ptr = slot } }); // ldrsb (sign-extend)
    func.setTerminator(e, .{ .ret = v });
    // 200 stored as a byte is 0xC8, loaded with ldrsb it sign-extends to -56.
    try expectRun(allocator, &func, &.{200}, -56);
}

test "native: a stack slot survives a call (alloca in a non-leaf frame)" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // id(a) = a   (leaf)
    var id = Function.init(allocator);
    defer id.deinit();
    {
        const t = try id.types.intern(t_kind);
        const b = try id.appendBlock();
        const a = try id.appendBlockParam(b, t);
        id.setTerminator(b, .{ .ret = a });
    }
    // f(x): slot = alloca, *slot = x, r = id(x), return *slot + r
    // (the alloca lives above the saved registers in the non-leaf frame)
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const t = try f.types.intern(t_kind);
        const ptr_t = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, x, slot);
        const r = try f.appendCall(b, t, "id", &.{x});
        const v = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        const sum = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = r } });
        f.setTerminator(b, .{ .ret = sum });
    }
    // f(5) = 5 + id(5) = 10.
    try std.testing.expectEqual(@as(i32, 10), try runModule(allocator, &.{
        .{ .name = "f", .func = &f },
        .{ .name = "id", .func = &id },
    }, &.{5}));
}

test "native: f64 constant returned in d0" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, f64_t, .{ .fconst = 3.5 });
    func.setTerminator(b, .{ .ret = c });
    try std.testing.expectEqual(@as(f64, 3.5), try runF64(allocator, &func, &.{}));
}

test "native: f64 arithmetic (add/sub/mul/div)" {
    const allocator = std.testing.allocator;
    const cases = .{
        .{ ir.function.BinOp.add, @as(f64, 1.5), @as(f64, 2.5), @as(f64, 4.0) },
        .{ ir.function.BinOp.sub, @as(f64, 5.0), @as(f64, 1.5), @as(f64, 3.5) },
        .{ ir.function.BinOp.mul, @as(f64, 1.5), @as(f64, 4.0), @as(f64, 6.0) },
        .{ ir.function.BinOp.div, @as(f64, 7.0), @as(f64, 2.0), @as(f64, 3.5) },
    };
    inline for (cases) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const f64_t = try func.types.intern(.{ .float = .f64 });
        const b = try func.appendBlock();
        const x = try func.appendInst(b, f64_t, .{ .fconst = case[1] });
        const y = try func.appendInst(b, f64_t, .{ .fconst = case[2] });
        const r = try func.appendInst(b, f64_t, .{ .arith = .{ .op = case[0], .lhs = x, .rhs = y } });
        func.setTerminator(b, .{ .ret = r });
        try std.testing.expectEqual(case[3], try runF64(allocator, &func, &.{}));
    }
}

test "native: int<->f64 conversions" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const fx = try func.appendInst(b, f64_t, .{ .convert = .{ .value = x } }); // scvtf
    const half = try func.appendInst(b, f64_t, .{ .fconst = 0.5 });
    const prod = try func.appendInst(b, f64_t, .{ .arith = .{ .op = .mul, .lhs = fx, .rhs = half } });
    const sum = try func.appendInst(b, f64_t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = half } });
    const r = try func.appendInst(b, i32_t, .{ .convert = .{ .value = sum } }); // fcvtzs
    func.setTerminator(b, .{ .ret = r });
    // int(7.0 * 0.5 + 0.5) = int(4.0) = 4.
    try expectRun(allocator, &func, &.{7}, 4);
}

test "native: f32 arithmetic then narrow to int" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendInst(b, f32_t, .{ .fconst = 2.5 });
    const y = try func.appendInst(b, f32_t, .{ .fconst = 1.5 });
    const s = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const r = try func.appendInst(b, i32_t, .{ .convert = .{ .value = s } });
    func.setTerminator(b, .{ .ret = r });
    try expectRun(allocator, &func, &.{}, 4); // int(2.5 + 1.5) = 4
}

test "native: f64 compare and select" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const a = try func.appendInst(b, f64_t, .{ .fconst = 1.0 });
    const c = try func.appendInst(b, f64_t, .{ .fconst = 2.0 });
    const lt = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = c } }); // fcmp
    const then = try func.appendInst(b, f64_t, .{ .fconst = 3.5 });
    const els = try func.appendInst(b, f64_t, .{ .fconst = 9.5 });
    const m = try func.appendInst(b, f64_t, .{ .select = .{ .cond = lt, .then = then, .@"else" = els } });
    func.setTerminator(b, .{ .ret = m });
    try std.testing.expectEqual(@as(f64, 3.5), try runF64(allocator, &func, &.{})); // 1.0 < 2.0 -> 3.5
}

//
// This aarch64 host has FEAT_FP16 (fphp/asimdhp), so the native H-form ops execute. Each kernel
// is JIT-run twice: once through the NATIVE path (`selectFunctionForModel` with the fp16 model,
// which sets caps.fp16 -> ldr h / H-form arith / str h, single-rounded, no widen/narrow) and once
// through the base-ISA EMULATION (`selectFunction`, fp16=false -> the f32 widening in an S reg
// with per-op fcvt rounding). Both must equal `@as(f16, ...)`. The DIVISION question is resolved:
// an exhaustive on-host sweep over every finite f16 pair showed native single-rounded `fdiv h`
// equals the emulation's f32-then-round for ALL inputs (f16's 10-bit mantissa makes double
// rounding through f32 always safe), and Zig's `@as(f16, a/b)` matches both, so `@as(f16, ...)`
// is the correct reference for every op including div.

/// Compile `func` to A64 words, choosing the native FEAT_FP16 path (`selectFunctionForModel` with
/// the fp16-capable ampere-altra model) or the emulation (`selectFunction`). Caller owns the slice.
fn selectF16(allocator: std.mem.Allocator, func: *const Function, native: bool) ![]u32 {
    if (native) return isel.selectFunctionForModel(allocator, func, opt.microarch.modelFor(.@"ampere-altra"));
    return isel.selectFunction(allocator, func);
}

/// JIT-run `void k(*f16 out, *const f16 a, *const f16 b): out = a <op> b`, exercising the native
/// half load (ldr h), H-form arithmetic, and store (str h) end to end. Returns the stored half.
fn runF16Bin(allocator: std.mem.Allocator, op: ir.function.BinOp, a: f16, b: f16, native: bool) !f16 {
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pb } });
    const r = try func.appendInst(blk, f16_t, .{ .arith = .{ .op = op, .lhs = va, .rhs = vb } });
    try func.appendStore(blk, r, out);
    func.setTerminator(blk, .{ .ret = null });
    const code = try selectF16(allocator, &func, native);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*f16, *const f16, *const f16) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    var result: f16 = 0;
    f(&result, &a, &b);
    return result;
}

test "native: FEAT_FP16 half add/sub/mul/div bit-exact vs @as(f16), native and emulation agree" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // The model actually reaches the native path (a guard against a silently-inert feature bit).
    try std.testing.expect(opt.microarch.modelFor(.@"ampere-altra").features.aarch64.fp16);

    const vals = [_]f16{ 0.5, 1.5, -2.25, 3.0, 7.0, 0.1, 10.5, -0.333, 1234.0, 0.0009765625 };
    const ops = [_]ir.function.BinOp{ .add, .sub, .mul, .div };
    for (ops) |op| {
        for (vals) |a| for (vals) |b| {
            if (op == .div and b == 0) continue;
            const ref: f16 = switch (op) {
                .add => a + b,
                .sub => a - b,
                .mul => a * b,
                .div => a / b,
                else => unreachable, // only the four arithmetic ops are enumerated above
            };
            const nat = try runF16Bin(allocator, op, a, b, true);
            const emu = try runF16Bin(allocator, op, a, b, false);
            try std.testing.expectEqual(@as(u16, @bitCast(ref)), @as(u16, @bitCast(nat)));
            try std.testing.expectEqual(@as(u16, @bitCast(ref)), @as(u16, @bitCast(emu)));
        };
    }

    // The gate is real: the native path emits strictly fewer words than the emulation because it
    // drops the per-op fcvt widen/narrow around the load, arithmetic, and store.
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pb } });
    const r = try func.appendInst(blk, f16_t, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    try func.appendStore(blk, r, out);
    func.setTerminator(blk, .{ .ret = null });
    const nat_code = try selectF16(allocator, &func, true);
    defer allocator.free(nat_code);
    const emu_code = try selectF16(allocator, &func, false);
    defer allocator.free(emu_code);
    try std.testing.expect(nat_code.len < emu_code.len);
}

test "native: FEAT_FP16 conversions, int<->f16, and fconst bit-exact vs @as(f16)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const ptr_k = ir.types.TypeKind{ .float = .f16 };

    inline for (.{ true, false }) |native| {
        // fconst: out = @as(f16, 3.5) via native `fmov h, w` (emulation: the S-held widening).
        {
            var func = Function.init(allocator);
            defer func.deinit();
            const ptr_t = try func.types.intern(.ptr);
            const f16_t = try func.types.intern(ptr_k);
            const blk = try func.appendBlock();
            const out = try func.appendBlockParam(blk, ptr_t);
            const c = try func.appendInst(blk, f16_t, .{ .fconst = 3.5 });
            try func.appendStore(blk, c, out);
            func.setTerminator(blk, .{ .ret = null });
            const code = try selectF16(allocator, &func, native);
            defer allocator.free(code);
            var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
            defer buf.deinit();
            const f: *const fn (*f16) callconv(.c) void = @ptrCast(buf.memory.ptr);
            var got: f16 = 0;
            f(&got);
            try std.testing.expectEqual(@as(u16, @bitCast(@as(f16, 3.5))), @as(u16, @bitCast(got)));
        }

        // int -> f16 (scvtf h) then f16 -> int (fcvtzs w, h), round-tripping i32 x through a half.
        {
            var func = Function.init(allocator);
            defer func.deinit();
            const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
            const f16_t = try func.types.intern(ptr_k);
            const blk = try func.appendBlock();
            const x = try func.appendBlockParam(blk, i32_t);
            const fx = try func.appendInst(blk, f16_t, .{ .convert = .{ .value = x } }); // scvtf h
            const back = try func.appendInst(blk, i32_t, .{ .convert = .{ .value = fx } }); // fcvtzs w, h
            func.setTerminator(blk, .{ .ret = back });
            const code = try selectF16(allocator, &func, native);
            defer allocator.free(code);
            var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
            defer buf.deinit();
            const f: *const fn (i32) callconv(.c) i32 = @ptrCast(buf.memory.ptr);
            for ([_]i32{ 0, 1, 7, 42, 100, 1000, -3, -37 }) |x_in| {
                const ref: i32 = @intFromFloat(@as(f16, @floatFromInt(x_in)));
                try std.testing.expectEqual(ref, f(x_in));
            }
        }

        // f16 <-> f32: `f32 k(*const f16 a): return (f32) a` and `void k(*f16 out, f32 x): out = (f16) x`.
        {
            var widen = Function.init(allocator);
            defer widen.deinit();
            const ptr_t = try widen.types.intern(.ptr);
            const f16_t = try widen.types.intern(ptr_k);
            const f32_t = try widen.types.intern(.{ .float = .f32 });
            const blk = try widen.appendBlock();
            const pa = try widen.appendBlockParam(blk, ptr_t);
            const av = try widen.appendInst(blk, f16_t, .{ .load = .{ .ptr = pa } });
            const w = try widen.appendInst(blk, f32_t, .{ .convert = .{ .value = av } }); // fcvt s, h
            widen.setTerminator(blk, .{ .ret = w });
            const wcode = try selectF16(allocator, &widen, native);
            defer allocator.free(wcode);
            var wbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(wcode));
            defer wbuf.deinit();
            const wf: *const fn (*const f16) callconv(.c) f32 = @ptrCast(wbuf.memory.ptr);

            var narrow = Function.init(allocator);
            defer narrow.deinit();
            const nptr_t = try narrow.types.intern(.ptr);
            const nf16_t = try narrow.types.intern(ptr_k);
            const nf32_t = try narrow.types.intern(.{ .float = .f32 });
            const nblk = try narrow.appendBlock();
            const nout = try narrow.appendBlockParam(nblk, nptr_t);
            const nx = try narrow.appendBlockParam(nblk, nf32_t);
            const nn = try narrow.appendInst(nblk, nf16_t, .{ .convert = .{ .value = nx } }); // fcvt h, s
            try narrow.appendStore(nblk, nn, nout);
            narrow.setTerminator(nblk, .{ .ret = null });
            const ncode = try selectF16(allocator, &narrow, native);
            defer allocator.free(ncode);
            var nbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(ncode));
            defer nbuf.deinit();
            const nf: *const fn (*f16, f32) callconv(.c) void = @ptrCast(nbuf.memory.ptr);

            const nx_in: f32 = 12345.678; // an f32 that must round to nearest-even half
            for ([_]f16{ 0.5, -2.25, 3.0, 0.1, 65504.0 }) |a| {
                try std.testing.expectEqual(@as(f32, a), wf(&a)); // widen is exact
                var got: f16 = 0;
                nf(&got, nx_in);
                // Reference rounds the SAME f32 the kernel narrows (a single f32->f16 round).
                const ref: f16 = @floatCast(nx_in);
                try std.testing.expectEqual(@as(u16, @bitCast(ref)), @as(u16, @bitCast(got)));
            }
        }

        // f16 <-> f64: `f64 k(*const f16 a): return (f64) a` and `void k(*f16 out, f64 x): out = (f16) x`.
        {
            var widen = Function.init(allocator);
            defer widen.deinit();
            const ptr_t = try widen.types.intern(.ptr);
            const f16_t = try widen.types.intern(ptr_k);
            const f64_t = try widen.types.intern(.{ .float = .f64 });
            const blk = try widen.appendBlock();
            const pa = try widen.appendBlockParam(blk, ptr_t);
            const av = try widen.appendInst(blk, f16_t, .{ .load = .{ .ptr = pa } });
            const w = try widen.appendInst(blk, f64_t, .{ .convert = .{ .value = av } }); // fcvt d, h
            widen.setTerminator(blk, .{ .ret = w });
            const wcode = try selectF16(allocator, &widen, native);
            defer allocator.free(wcode);
            var wbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(wcode));
            defer wbuf.deinit();
            const wf: *const fn (*const f16) callconv(.c) f64 = @ptrCast(wbuf.memory.ptr);

            var narrow = Function.init(allocator);
            defer narrow.deinit();
            const nptr_t = try narrow.types.intern(.ptr);
            const nf16_t = try narrow.types.intern(ptr_k);
            const nf64_t = try narrow.types.intern(.{ .float = .f64 });
            const nblk = try narrow.appendBlock();
            const nout = try narrow.appendBlockParam(nblk, nptr_t);
            const nx = try narrow.appendBlockParam(nblk, nf64_t);
            const nn = try narrow.appendInst(nblk, nf16_t, .{ .convert = .{ .value = nx } }); // fcvt h, d
            try narrow.appendStore(nblk, nn, nout);
            narrow.setTerminator(nblk, .{ .ret = null });
            const ncode = try selectF16(allocator, &narrow, native);
            defer allocator.free(ncode);
            var nbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(ncode));
            defer nbuf.deinit();
            const nf: *const fn (*f16, f64) callconv(.c) void = @ptrCast(nbuf.memory.ptr);

            const nx_in: f64 = 0.30000000000000004; // an f64 that must round once directly d -> h
            for ([_]f16{ 0.5, -2.25, 3.0, 0.1, 65504.0 }) |a| {
                try std.testing.expectEqual(@as(f64, a), wf(&a)); // widen is exact
                var got: f16 = 0;
                nf(&got, nx_in);
                // Reference rounds the SAME f64 the kernel narrows (a single f64->f16 round).
                const ref: f16 = @floatCast(nx_in);
                try std.testing.expectEqual(@as(u16, @bitCast(ref)), @as(u16, @bitCast(got)));
            }
        }
    }
}

test "native: a call passing f64 arguments in v-registers" {
    const allocator = std.testing.allocator;
    const f64_kind = ir.types.TypeKind{ .float = .f64 };

    // addf(a, b) = a + b
    var addf = Function.init(allocator);
    defer addf.deinit();
    {
        const t = try addf.types.intern(f64_kind);
        const b = try addf.appendBlock();
        const a = try addf.appendBlockParam(b, t);
        const bb = try addf.appendBlockParam(b, t);
        const r = try addf.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bb } });
        addf.setTerminator(b, .{ .ret = r });
    }
    // caller() = int(addf(1.5, 2.5))
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(f64_kind);
        const i32_t = try caller.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try caller.appendBlock();
        const c1 = try caller.appendInst(b, t, .{ .fconst = 1.5 });
        const c2 = try caller.appendInst(b, t, .{ .fconst = 2.5 });
        const s = try caller.appendCall(b, t, "addf", &.{ c1, c2 });
        const r = try caller.appendInst(b, i32_t, .{ .convert = .{ .value = s } });
        caller.setTerminator(b, .{ .ret = r });
    }
    // int(1.5 + 2.5) = int(4.0) = 4.
    try std.testing.expectEqual(@as(i32, 4), try runModule(allocator, &.{
        .{ .name = "caller", .func = &caller },
        .{ .name = "addf", .func = &addf },
    }, &.{}));
}

test "jit: compile a module and call functions by name" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a   (leaf)
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(t_kind);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    // caller(x) = dbl(x) + 1
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(t_kind);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = r });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "dbl", &dbl);

    var compiled = try jit.compileModule(allocator, &module);
    defer compiled.deinit(allocator);

    // Both functions are individually callable by name through the JIT.
    const caller_fn = compiled.funcPointer(*const fn (i32) callconv(.c) i32, "caller").?;
    try std.testing.expectEqual(@as(i32, 11), caller_fn(5)); // dbl(5) + 1
    const dbl_fn = compiled.funcPointer(*const fn (i32) callconv(.c) i32, "dbl").?;
    try std.testing.expectEqual(@as(i32, 14), dbl_fn(7));
    try std.testing.expect(compiled.funcPointer(*const fn () callconv(.c) i32, "missing") == null);
}

test "pipeline: an optimized function runs correctly on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // f(x) = (2 + 3) * 4 + x, with a dead x*x. Constant folding collapses the
    // arithmetic to 20 and DCE drops the dead product, then aarch64 codegen runs it.
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const c3 = try func.appendInst(b, t, .{ .iconst = 3 });
    const c4 = try func.appendInst(b, t, .{ .iconst = 4 });
    const a = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c2, .rhs = c3 } });
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c4 } });
    _ = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } }); // dead
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = r });

    try std.testing.expect(try opt.optimize(allocator, &func));
    try expectRun(allocator, &func, &.{22}, 42); // 20 + 22
}

test "pipeline: inlining composes with aarch64 codegen" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // madd(a, b) = a*b + a
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const cb = try callee.appendBlock();
        const a = try callee.appendBlockParam(cb, t);
        const bb = try callee.appendBlockParam(cb, t);
        const prod = try callee.appendInst(cb, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
        const sum = try callee.appendInst(cb, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
        callee.setTerminator(cb, .{ .ret = sum });
    }
    // f() = madd(2, 3)
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const b = try caller.appendBlock();
    const c2 = try caller.appendInst(b, t, .{ .iconst = 2 });
    const c3 = try caller.appendInst(b, t, .{ .iconst = 3 });
    const r = try caller.appendCall(b, t, "madd", &.{ c2, c3 });
    caller.setTerminator(b, .{ .ret = r });

    const Lk = struct {
        callee: *const Function,
        fn get(ctx: *anyopaque, name: []const u8) ?*const Function {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return if (std.mem.eql(u8, name, "madd")) self.callee else null;
        }
    };
    var lk = Lk{ .callee = &callee };
    const lookup = opt.inlining.Lookup{ .context = &lk, .func = Lk.get };
    try std.testing.expect(try opt.inlining.run(allocator, &caller, lookup));
    _ = try opt.optimize(allocator, &caller); // fold across the inlined call
    for (caller.blockInsts(b)) |inst| try std.testing.expect(caller.opcode(inst) != .call);

    try expectRun(allocator, &caller, &.{}, 8); // madd(2,3) = 8, after inlining
}

test "pipeline: LTO across modules then aarch64 codegen" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var src = opt.lto.Module.init(allocator);
    {
        // helper(a, b) = a*b + a
        var f = Function.init(allocator);
        const t = try f.types.intern(i32k);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const bb = try f.appendBlockParam(b, t);
        const prod = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
        const sum = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
        f.setTerminator(b, .{ .ret = sum });
        try src.add("helper", f);
        // entry(x) = helper(x, x) + 1
        var g = Function.init(allocator);
        const gt = try g.types.intern(i32k);
        const gb = try g.appendBlock();
        const x = try g.appendBlockParam(gb, gt);
        const call = try g.appendCall(gb, gt, "helper", &.{ x, x });
        const r = try g.appendArithImm(gb, gt, .add, call, 1);
        g.setTerminator(gb, .{ .ret = r });
        try src.add("entry", g);
    }
    const blob = try opt.lto.encode(allocator, &src);
    src.deinit();
    defer allocator.free(blob);
    var module = try opt.lto.decode(allocator, blob);
    defer module.deinit();
    _ = try opt.lto.link(allocator, &module, &.{"entry"});
    try std.testing.expectEqual(@as(usize, 1), module.count()); // helper inlined + pruned

    // entry(x) = x*x + x + 1.  x=4: 16 + 4 + 1 = 21.
    try expectRun(allocator, module.get("entry").?, &.{4}, 21);
}

test "object+ld: emit ELF .o, link it, and JIT-run the result" {
    const allocator = std.testing.allocator;
    const object = @import("../object.zig");
    const ld = @import("../ld.zig");
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a   (leaf)
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    // caller(x) = dbl(x) + 1   (a cross-function CALL26 relocation)
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = r });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "dbl", &dbl);

    // Module -> ELF .o -> the linker -> image.
    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var image = try ld.linkObjects(allocator, &.{obj}, 0);
    defer image.deinit(allocator);

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // JIT-map the linked image and call through it.
    var buf = try jit.CodeBuffer.map(image.code);
    defer buf.deinit();
    const caller_fn = buf.entry(*const fn (i32) callconv(.c) i32, image.addressOf("caller").?);
    try std.testing.expectEqual(@as(i32, 11), caller_fn(5)); // dbl(5) + 1
    const dbl_fn = buf.entry(*const fn (i32) callconv(.c) i32, image.addressOf("dbl").?);
    try std.testing.expectEqual(@as(i32, 14), dbl_fn(7));
}

test "spirv: lower a SPIR-V function to IR and run it natively (x*y + x)" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");

    // Hand-assembled SPIR-V: int f(int x, int y) { return x*y + x }
    // ids: int=1, fnty=2, f=3, x=4, y=5, entry=6, prod=7, sum=8.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    const op = spirv.opcodes;
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(allocator, op.IAdd, &.{ 1, 8, 7, 4 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // SPIR-V -> Vulcan IR -> AArch64 -> run on the host: f(3, 4) = 3*4 + 3 = 15.
    try expectRun(allocator, &func, &.{ 3, 4 }, 15);
}

test "spirv: an optimized SPIR-V function runs natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const op = spirv.opcodes;

    // int f(int x) { return (x + 7) * x }  (a constant plus two ops)
    // ids: int=1, fnty=2, c7=3, f=4, x=5, entry=6, sum=7, prod=8.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 1, 3, 7 });
    try b.emit(allocator, op.Function, &.{ 1, 4, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IAdd, &.{ 1, 7, 5, 3 });
    try b.emit(allocator, op.IMul, &.{ 1, 8, 7, 5 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    _ = try opt.optimize(allocator, &func); // the IR composes with the optimizer
    // f(5) = (5 + 7) * 5 = 60.
    try expectRun(allocator, &func, &.{5}, 60);
}

test "spirv: int<->float conversions lowered and run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const op = spirv.opcodes;

    // int f(int x) { return int(float(x) * 2.5) }
    // ids: int=1, float=2, fnty=3, c2.5=4, f=5, x=6, entry=7, fx=8, scaled=9, r=10.
    var b = try spirv.binary.Builder.init(allocator, 11);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeFunction, &.{ 3, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 4, 0x40200000 }); // 2.5f
    try b.emit(allocator, op.Function, &.{ 1, 5, 0, 3 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 6 });
    try b.emit(allocator, op.Label, &.{7});
    try b.emit(allocator, op.ConvertSToF, &.{ 2, 8, 6 }); // fx = float(x)
    try b.emit(allocator, op.FMul, &.{ 2, 9, 8, 4 }); // scaled = fx * 2.5
    try b.emit(allocator, op.ConvertFToS, &.{ 1, 10, 9 }); // r = int(scaled)
    try b.emit(allocator, op.ReturnValue, &.{10});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    // f(4) = int(4.0 * 2.5) = int(10.0) = 10.
    try expectRun(allocator, &func, &.{4}, 10);
}

test "native: 64-bit pointer arithmetic into a stack array (base + i*4)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const arr_t = try func.types.intern(.{ .array = .{ .len = 8, .elem = i32_t } });

    // f(i) { int buf[8], int* p = buf + i, *p = i*10 + 1, return *p }
    // Exercises a 32-byte array alloca, a 64-bit pointer add, and load/store at a
    // computed address. The byte offset is i*4 (element stride).
    const e = try func.appendBlock();
    const i = try func.appendBlockParam(e, i32_t);
    const buf = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = arr_t } });
    const off = try func.appendArithImm(e, i32_t, .shl, i, 2); // i*4 bytes
    const p = try func.appendInst(e, ptr_t, .{ .arith = .{ .op = .add, .lhs = buf, .rhs = off } }); // 64-bit add
    const ten_i = try func.appendInst(e, i32_t, .{ .iconst = 10 });
    const scaled = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .mul, .lhs = i, .rhs = ten_i } });
    const val = try func.appendArithImm(e, i32_t, .add, scaled, 1);
    try func.appendStore(e, val, p);
    const got = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(e, .{ .ret = got });

    // f(3) = 3*10 + 1 = 31, written at buf+12 and read back.
    try expectRun(allocator, &func, &.{3}, 31);
}

/// Run a lowered compute kernel `main(invocation_id, buf_ptr)` natively against a
/// real buffer, mutating it in place. The signature matches the SPIR-V frontend's
/// synthesized entry params (the invocation id, then the storage-buffer pointer).
fn runCompute(allocator: std.mem.Allocator, func: *const Function, invocation_id: i32, buf: []i32) !void {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var cb = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer cb.deinit();
    const main_fn = cb.entry(*const fn (i32, [*]i32) callconv(.c) void, 0);
    main_fn(invocation_id, buf.ptr);
}

test "spirv compute: storage-buffer shader runs natively (data[gid] *= 2)" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // void main() { data[gl_GlobalInvocationID.x] = data[gid] * 2 }
    var b = try spirv.binary.Builder.init(allocator, 23);
    defer b.deinit(allocator);
    try b.emit(allocator, o.Decorate, &.{ 14, o.Decoration.builtin, o.BuiltIn.global_invocation_id });
    try b.emit(allocator, o.TypeVoid, &.{1});
    try b.emit(allocator, o.TypeInt, &.{ 2, 32, 1 });
    try b.emit(allocator, o.TypeInt, &.{ 3, 32, 0 });
    try b.emit(allocator, o.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, o.TypePointer, &.{ 5, o.StorageClass.input, 4 });
    try b.emit(allocator, o.TypePointer, &.{ 6, o.StorageClass.input, 3 });
    try b.emit(allocator, o.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, o.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, o.TypePointer, &.{ 9, o.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, o.TypePointer, &.{ 10, o.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, o.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, o.Constant, &.{ 3, 12, 0 });
    try b.emit(allocator, o.Constant, &.{ 2, 13, 2 });
    try b.emit(allocator, o.Variable, &.{ 5, 14, o.StorageClass.input });
    try b.emit(allocator, o.Variable, &.{ 9, 15, o.StorageClass.storage_buffer });
    try b.emit(allocator, o.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, o.Label, &.{17});
    try b.emit(allocator, o.AccessChain, &.{ 6, 18, 14, 12 });
    try b.emit(allocator, o.Load, &.{ 3, 19, 18 });
    try b.emit(allocator, o.AccessChain, &.{ 10, 20, 15, 12, 19 });
    try b.emit(allocator, o.Load, &.{ 2, 21, 20 });
    try b.emit(allocator, o.IMul, &.{ 2, 22, 21, 13 });
    try b.emit(allocator, o.Store, &.{ 20, 22 });
    try b.emit(allocator, o.Return, &.{});
    try b.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // Run one invocation (thread 2) over a real buffer: only data[2] doubles.
    var data = [_]i32{ 10, 20, 30, 40 };
    try runCompute(allocator, &func, 2, &data);
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 60, 40 }, &data);

    // A different thread index hits a different element.
    var data2 = [_]i32{ 5, 6, 7, 8 };
    try runCompute(allocator, &func, 0, &data2);
    try std.testing.expectEqualSlices(i32, &.{ 10, 6, 7, 8 }, &data2);
}

/// Build f(a, b) = a <op> b with the given signedness, lower its division to
/// division-free IR, and run it natively. Validates the lowering against the
/// host's real divide.
fn runLoweredDiv(allocator: std.mem.Allocator, signedness: std.builtin.Signedness, bop: ir.function.BinOp, a: i32, b: i32) !i32 {
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = signedness, .bits = 32 } });
    const blk = try func.appendBlock();
    const x = try func.appendBlockParam(blk, t);
    const y = try func.appendBlockParam(blk, t);
    const r = try func.appendInst(blk, t, .{ .arith = .{ .op = bop, .lhs = x, .rhs = y } });
    func.setTerminator(blk, .{ .ret = r });
    try std.testing.expect(try opt.lowerdiv.run(allocator, &func));
    return run(allocator, &func, &.{ a, b });
}

test "lowerdiv: unsigned division and remainder run correctly" {
    const allocator = std.testing.allocator;
    const U = std.builtin.Signedness.unsigned;
    try std.testing.expectEqual(@as(i32, 6), try runLoweredDiv(allocator, U, .div, 20, 3));
    try std.testing.expectEqual(@as(i32, 2), try runLoweredDiv(allocator, U, .rem, 20, 3));
    try std.testing.expectEqual(@as(i32, 14), try runLoweredDiv(allocator, U, .div, 100, 7));
    try std.testing.expectEqual(@as(i32, 15), try runLoweredDiv(allocator, U, .div, 255, 16));
    // The high bit must be treated as magnitude, not sign: 0x80000000 / 2.
    try std.testing.expectEqual(@as(i32, @bitCast(@as(u32, 0x40000000))), try runLoweredDiv(allocator, U, .div, @bitCast(@as(u32, 0x80000000)), 2));
}

test "lowerdiv: signed division and remainder run correctly (round toward zero)" {
    const allocator = std.testing.allocator;
    const S = std.builtin.Signedness.signed;
    try std.testing.expectEqual(@as(i32, 6), try runLoweredDiv(allocator, S, .div, 20, 3));
    try std.testing.expectEqual(@as(i32, -6), try runLoweredDiv(allocator, S, .div, -20, 3));
    try std.testing.expectEqual(@as(i32, -6), try runLoweredDiv(allocator, S, .div, 20, -3));
    try std.testing.expectEqual(@as(i32, 6), try runLoweredDiv(allocator, S, .div, -20, -3));
    // Remainder takes the sign of the dividend.
    try std.testing.expectEqual(@as(i32, -2), try runLoweredDiv(allocator, S, .rem, -20, 3));
    try std.testing.expectEqual(@as(i32, 2), try runLoweredDiv(allocator, S, .rem, 20, -3));
}

test "spirv: unary negate and bitwise not run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x) { return -x }  (OpSNegate)
    {
        var b = try spirv.binary.Builder.init(allocator, 8);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
        try b.emit(allocator, o.Label, &.{5});
        try b.emit(allocator, o.SNegate, &.{ 1, 6, 4 });
        try b.emit(allocator, o.ReturnValue, &.{6});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{7}, -7);
    }
    // int f(int x) { return ~x }  (OpNot, ~5 = -6)
    {
        var b = try spirv.binary.Builder.init(allocator, 8);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
        try b.emit(allocator, o.Label, &.{5});
        try b.emit(allocator, o.Not, &.{ 1, 6, 4 });
        try b.emit(allocator, o.ReturnValue, &.{6});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{5}, -6);
    }
}

test "spirv: GLSL clamp and abs (OpExtInst) run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x) { return clamp(x, 2, 8) }  (SClamp, set operand is ignored)
    // ids: int=1, fnty=2, c2=3, c8=4, f=5, x=6, entry=7, r=8.
    {
        var b = try spirv.binary.Builder.init(allocator, 9);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Constant, &.{ 1, 3, 2 });
        try b.emit(allocator, o.Constant, &.{ 1, 4, 8 });
        try b.emit(allocator, o.Function, &.{ 1, 5, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 6 });
        try b.emit(allocator, o.Label, &.{7});
        try b.emit(allocator, o.ExtInst, &.{ 1, 8, 99, o.Glsl.s_clamp, 6, 3, 4 });
        try b.emit(allocator, o.ReturnValue, &.{8});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{5}, 5); // in range
        var f2 = try spirv.lowerModule(allocator, b.words.items);
        defer f2.deinit();
        try expectRun(allocator, &f2, &.{1}, 2); // clamped up to lo
        var f3 = try spirv.lowerModule(allocator, b.words.items);
        defer f3.deinit();
        try expectRun(allocator, &f3, &.{10}, 8); // clamped down to hi
    }
    // int f(int x) { return abs(x) }  (SAbs)
    {
        var b = try spirv.binary.Builder.init(allocator, 8);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
        try b.emit(allocator, o.Label, &.{5});
        try b.emit(allocator, o.ExtInst, &.{ 1, 6, 99, o.Glsl.s_abs, 4 });
        try b.emit(allocator, o.ReturnValue, &.{6});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{-7}, 7);
        var f2 = try spirv.lowerModule(allocator, b.words.items);
        defer f2.deinit();
        try expectRun(allocator, &f2, &.{7}, 7);
    }
}

test "spirv compute: multi-member buffer struct runs natively (data[i] *= scale)" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // buffer Buf { uint scale, uint data[] }  // member 0 at offset 0, member 1 at 4
    // void main() { uint i = gid.x, data[i] = data[i] * scale }
    // ids: void=1 uint=2 v3=3 pInV3=4 pInU=5 arr=6 Buf=7 pSbBuf=8 pSbU=9 voidfn=10
    //      u0=11 u1=12 gid=13 buf=14 main=15 entry=16 xptr=17 i=18 sptr=19 scale=20
    //      eptr=21 v=22 v2=23.
    var b = try spirv.binary.Builder.init(allocator, 24);
    defer b.deinit(allocator);
    // Annotations (before the types they decorate).
    try b.emit(allocator, o.MemberDecorate, &.{ 7, 0, o.Decoration.offset, 0 }); // scale @0
    try b.emit(allocator, o.MemberDecorate, &.{ 7, 1, o.Decoration.offset, 4 }); // data @4
    try b.emit(allocator, o.Decorate, &.{ 6, o.Decoration.array_stride, 4 });
    try b.emit(allocator, o.Decorate, &.{ 13, o.Decoration.builtin, o.BuiltIn.global_invocation_id });
    // Types.
    try b.emit(allocator, o.TypeVoid, &.{1});
    try b.emit(allocator, o.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, o.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, o.TypePointer, &.{ 4, o.StorageClass.input, 3 });
    try b.emit(allocator, o.TypePointer, &.{ 5, o.StorageClass.input, 2 });
    try b.emit(allocator, o.TypeRuntimeArray, &.{ 6, 2 });
    try b.emit(allocator, o.TypeStruct, &.{ 7, 2, 6 }); // { uint scale, uint data[] }
    try b.emit(allocator, o.TypePointer, &.{ 8, o.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, o.TypePointer, &.{ 9, o.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, o.TypeFunction, &.{ 10, 1 });
    try b.emit(allocator, o.Constant, &.{ 2, 11, 0 });
    try b.emit(allocator, o.Constant, &.{ 2, 12, 1 });
    try b.emit(allocator, o.Variable, &.{ 4, 13, o.StorageClass.input });
    try b.emit(allocator, o.Variable, &.{ 8, 14, o.StorageClass.storage_buffer });
    try b.emit(allocator, o.Function, &.{ 1, 15, 0, 10 });
    try b.emit(allocator, o.Label, &.{16});
    try b.emit(allocator, o.AccessChain, &.{ 5, 17, 13, 11 }); // &gid.x
    try b.emit(allocator, o.Load, &.{ 2, 18, 17 }); // i
    try b.emit(allocator, o.AccessChain, &.{ 9, 19, 14, 11 }); // &buf.scale (member 0)
    try b.emit(allocator, o.Load, &.{ 2, 20, 19 }); // scale
    try b.emit(allocator, o.AccessChain, &.{ 9, 21, 14, 12, 18 }); // &buf.data[i] (member 1 + i)
    try b.emit(allocator, o.Load, &.{ 2, 22, 21 }); // data[i]
    try b.emit(allocator, o.IMul, &.{ 2, 23, 22, 20 }); // data[i] * scale
    try b.emit(allocator, o.Store, &.{ 21, 23 });
    try b.emit(allocator, o.Return, &.{});
    try b.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // Buffer layout: [scale=3, data[0]=10, data[1]=20, data[2]=30].
    var data = [_]i32{ 3, 10, 20, 30 };
    try runCompute(allocator, &func, 1, &data); // thread 1: data[1] = 20 * 3 = 60
    try std.testing.expectEqualSlices(i32, &.{ 3, 10, 60, 30 }, &data);

    var data2 = [_]i32{ 3, 10, 20, 30 };
    try runCompute(allocator, &func, 0, &data2); // thread 0: data[0] = 10 * 3 = 30
    try std.testing.expectEqualSlices(i32, &.{ 3, 30, 20, 30 }, &data2);
}

test "spirv vectors: construct, component-wise add, extract run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x, int y) { ivec2 a = ivec2(x,y), ivec2 b = ivec2(y,x),
    //                       ivec2 c = a + b, return c.x }   // = x + y
    // ids: int=1 v2=2 fnty=3 f=4 x=5 y=6 entry=7 a=8 b=9 c=10 r=11.
    var bld = try spirv.binary.Builder.init(allocator, 12);
    defer bld.deinit(allocator);
    try bld.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
    try bld.emit(allocator, o.TypeVector, &.{ 2, 1, 2 });
    try bld.emit(allocator, o.TypeFunction, &.{ 3, 1, 1, 1 });
    try bld.emit(allocator, o.Function, &.{ 1, 4, 0, 3 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 5 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 6 });
    try bld.emit(allocator, o.Label, &.{7});
    try bld.emit(allocator, o.CompositeConstruct, &.{ 2, 8, 5, 6 }); // a = (x, y)
    try bld.emit(allocator, o.CompositeConstruct, &.{ 2, 9, 6, 5 }); // b = (y, x)
    try bld.emit(allocator, o.IAdd, &.{ 2, 10, 8, 9 }); // c = a + b (component-wise)
    try bld.emit(allocator, o.CompositeExtract, &.{ 1, 11, 10, 0 }); // c.x
    try bld.emit(allocator, o.ReturnValue, &.{11});
    try bld.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, bld.words.items);
    defer func.deinit();
    try expectRun(allocator, &func, &.{ 3, 4 }, 7); // c.x = x + y = 7
}

test "spirv vectors: VectorShuffle swaps components" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x, int y) { ivec2 a = ivec2(x,y), ivec2 b = a.yx, return b.x }  // = y
    var bld = try spirv.binary.Builder.init(allocator, 12);
    defer bld.deinit(allocator);
    try bld.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
    try bld.emit(allocator, o.TypeVector, &.{ 2, 1, 2 });
    try bld.emit(allocator, o.TypeFunction, &.{ 3, 1, 1, 1 });
    try bld.emit(allocator, o.Function, &.{ 1, 4, 0, 3 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 5 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 6 });
    try bld.emit(allocator, o.Label, &.{7});
    try bld.emit(allocator, o.CompositeConstruct, &.{ 2, 8, 5, 6 }); // a = (x, y)
    try bld.emit(allocator, o.VectorShuffle, &.{ 2, 9, 8, 8, 1, 0 }); // b = a.yx = (y, x)
    try bld.emit(allocator, o.CompositeExtract, &.{ 1, 10, 9, 0 }); // b.x
    try bld.emit(allocator, o.ReturnValue, &.{10});
    try bld.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, bld.words.items);
    defer func.deinit();
    try expectRun(allocator, &func, &.{ 3, 4 }, 4); // b.x = y = 4
}

test "uefi: IR -> aarch64 -> PE32+ image, and the embedded code runs" {
    const allocator = std.testing.allocator;
    const pe = @import("../../pe.zig");

    // A trivial efi_main-like function: returns 42 (a constant EFI_STATUS).
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, t, .{ .iconst = 42 });
    func.setTerminator(b, .{ .ret = c });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "efi_main", &func);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);

    const entry = linked.addressOf("efi_main").?;
    const img = try pe.writeUefiImage(allocator, std.mem.sliceAsBytes(linked.code), linked.code.len * 4, entry, .aarch64);
    defer allocator.free(img);

    // A well-formed UEFI image: MZ / PE / AArch64 machine / EFI_APPLICATION subsystem.
    try std.testing.expectEqual(@as(u8, 'M'), img[0]);
    const lfanew = std.mem.readInt(u32, img[0x3c..0x40], .little);
    try std.testing.expectEqual(@as(u16, 0xAA64), std.mem.readInt(u16, img[lfanew + 4 ..][0..2], .little));
    const opt_hdr = img[lfanew + 4 + 20 ..];
    try std.testing.expectEqual(@as(u16, 10), std.mem.readInt(u16, opt_hdr[68..70], .little)); // EFI_APPLICATION

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // The .text bytes are the real function: extract from PointerToRawData and run.
    const sect = img[lfanew + 4 + 20 + 240 ..];
    const praw = std.mem.readInt(u32, sect[20..24], .little);
    const text = img[praw..][0 .. linked.code.len * 4];
    var buf = try jit.CodeBuffer.map(text);
    defer buf.deinit();
    const fp = buf.entry(*const fn () callconv(.c) i32, entry);
    try std.testing.expectEqual(@as(i32, 42), fp());
}

test "object+ld+exec: link two functions into a runnable ELF and execute it natively" {
    const allocator = std.testing.allocator;
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // executes the AArch64 ELF directly
    const object = @import("../object.zig");
    const ld = @import("../ld.zig");
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a, main(x) = dbl(x) + 2.  main(20) = 42.
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(i32k);
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const d = try main.appendCall(b, t, "dbl", &.{x});
        const r = try main.appendArithImm(b, t, .add, d, 2);
        main.setTerminator(b, .{ .ret = r });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "dbl", &dbl);
    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    const base: u64 = 0x400000;
    var image = try ld.linkObjects(allocator, &.{obj}, base);
    defer image.deinit(allocator);

    // A tiny entry stub: set the argument, call main, then exit with its result. main sits
    // right past the 16-byte stub. bl is at offset 4.
    const main_off: i64 = @intCast(image.addressOf("main").? - base);
    const stub = [_]u32{
        encode.movz(.x0, 20, 0), // x0 = 20
        encode.bl(@intCast((16 + main_off) - 4)), // bl main (it sits past the 16-byte stub)
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(allocator, program.items, program.items.len, base, base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound, error.AccessDenied => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // dbl(20) + 2
        else => return error.BackendFailed,
    }
}

test "native: unsigned div/shr/compare and unsigned int->float" {
    const allocator = std.testing.allocator;

    { // udiv: 0xFFFFFFFF / 2 = 0x7FFFFFFF (signed sdiv would give 0)
        var f = Function.init(allocator);
        defer f.deinit();
        const u = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, u);
        const d = try f.appendBlockParam(b, u);
        const r = try f.appendInst(b, u, .{ .arith = .{ .op = .div, .lhs = a, .rhs = d } });
        f.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &f, &.{ -1, 2 }, 0x7FFFFFFF);
    }
    { // lsr: 0x80000000 >> 1 = 0x40000000 (signed asr would give 0xC0000000)
        var f = Function.init(allocator);
        defer f.deinit();
        const u = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, u);
        const r = try f.appendArithImm(b, u, .shr, a, 1);
        f.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &f, &.{@bitCast(@as(u32, 0x80000000))}, 0x40000000);
    }
    { // unsigned compare: (-1 as u32) <u 1 is false
        var f = Function.init(allocator);
        defer f.deinit();
        const u = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const bool_t = try f.types.intern(.bool);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, u);
        const d = try f.appendBlockParam(b, u);
        const c = try f.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = d } });
        const one = try f.appendInst(b, u, .{ .iconst = 1 });
        const zero = try f.appendInst(b, u, .{ .iconst = 0 });
        const r = try f.appendInst(b, u, .{ .select = .{ .cond = c, .then = one, .@"else" = zero } });
        f.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &f, &.{ -1, 1 }, 0);
        try expectRun(allocator, &f, &.{ 1, 2 }, 1);
    }
    { // unsigned int -> f64: (f64)(u32)0xFFFFFFFF = 4294967295.0 (signed would be -1.0)
        var f = Function.init(allocator);
        defer f.deinit();
        const u = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f64t = try f.types.intern(.{ .float = .f64 });
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, u);
        const r = try f.appendInst(b, f64t, .{ .convert = .{ .value = a } });
        f.setTerminator(b, .{ .ret = r });
        try std.testing.expectEqual(@as(f64, 4294967295.0), try runF64(allocator, &f, &.{-1}));
    }
}

test "disasm: source-annotated listing of a real -g object (full DWARF-read pipeline)" {
    // cc -g a real function, then run the whole vulcan-disasm pipeline on it: findText ->
    // sectionByName(.debug_line) -> decodeLine -> formatElfWithLines, and confirm the source-line
    // markers land in the listing. Host must be aarch64 for cc to emit an aarch64 object.
    const a = std.testing.allocator;
    const io = std.testing.io;
    const dwarf = @import("../../dwarf.zig");
    const elf_read = @import("../../elf_read.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "add.c", .data = "int add(int a, int b) {\n  int c = a + b;\n  return c;\n}\n" });
    const cc = std.process.run(a, io, .{ .argv = &.{ "cc", "-gdwarf-4", "-O0", "-c", "add.c", "-o", "add.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer a.free(cc.stdout);
    defer a.free(cc.stderr);
    if (cc.term != .exited or cc.term.exited != 0) return error.SkipZigTest;
    const obj = try tmp.dir.readFileAlloc(io, "add.o", a, .limited(4 << 20));
    defer a.free(obj);

    const t = try elf_read.findText(obj);
    if (t.machine != elf_read.EM_AARCH64) return error.SkipZigTest; // non-aarch64 host
    const dl = (try elf_read.sectionByName(obj, ".debug_line")) orelse return error.SkipZigTest;
    const rows = try dwarf.decodeLine(a, dl);
    defer a.free(rows);

    var lines: std.ArrayList(disasm.AddrLine) = .empty;
    defer lines.deinit(a);
    for (rows) |r| if (!r.end_sequence) try lines.append(a, .{ .addr = r.address, .line = r.line });
    std.mem.sort(disasm.AddrLine, lines.items, {}, struct {
        fn lt(_: void, x: disasm.AddrLine, y: disasm.AddrLine) bool {
            return x.addr < y.addr;
        }
    }.lt);

    const words = try a.alloc(u32, t.bytes.len / 4);
    defer a.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little);
    const listing = try disasm.formatElfWithLines(a, words, t.addr, &.{}, lines.items);
    defer a.free(listing);

    // The real source's body lines appear as markers interleaved in the disassembly.
    try std.testing.expect(std.mem.indexOf(u8, listing, "; line 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, listing, "; line 3") != null);
}

test "pipeline: algebraic identities simplify then run correctly on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // Every subterm is an identity: x*1=x, x-x=0, x&x=x, x^x=0. So the function computes
    // ((x*1) + (x-x)) + ((x&x) - (x^x)) = (x + 0) + (x - 0) = 2x. simplify must preserve that.
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const one = try func.appendInst(b, t, .{ .iconst = 1 });
    const m1 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = one } }); // x
    const sxx = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = x } }); // 0
    const axx = try func.appendInst(b, t, .{ .arith = .{ .op = .bit_and, .lhs = x, .rhs = x } }); // x
    const xxx = try func.appendInst(b, t, .{ .arith = .{ .op = .bit_xor, .lhs = x, .rhs = x } }); // 0
    const l = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = m1, .rhs = sxx } }); // x
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = axx, .rhs = xxx } }); // x
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = l, .rhs = r } }); // 2x
    func.setTerminator(b, .{ .ret = sum });

    try std.testing.expect(try opt.optimize(allocator, &func));
    try expectRun(allocator, &func, &.{21}, 42); // 2 * 21
}

test "pipeline: strength reduction (mul/div/rem by powers of two) runs correctly on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // Unsigned so div/rem reduce to shift/mask: f(x) = (x*4) + (x/2) + (x%2).
    // For x = 10: 40 + 5 + 0 = 45. After strength: (x<<2) + (x>>1) + (x&1).
    const t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const m = try func.appendArithImm(b, t, .mul, x, 4); // x << 2
    const d = try func.appendArithImm(b, t, .div, x, 2); // x >> 1
    const r = try func.appendArithImm(b, t, .rem, x, 2); // x & 1
    const s1 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = m, .rhs = d } });
    const s2 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = r } });
    func.setTerminator(b, .{ .ret = s2 });

    try std.testing.expect(try opt.optimize(allocator, &func));
    try expectRun(allocator, &func, &.{10}, 45);
}

test "pipeline: a constant-condition select folds away and runs correctly on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // f(a, b) = (1 < 2) ? a : b. constfold resolves 1<2 to true, then simplify folds the select to a.
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bb = try func.appendBlockParam(b, t);
    const c1 = try func.appendInst(b, t, .{ .iconst = 1 });
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const cmp = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = c1, .rhs = c2 } });
    const sel = try func.appendInst(b, t, .{ .select = .{ .cond = cmp, .then = a, .@"else" = bb } });
    func.setTerminator(b, .{ .ret = sel });

    try std.testing.expect(try opt.optimize(allocator, &func));
    try expectRun(allocator, &func, &.{ 7, 9 }, 7); // picks a = 7
}

test "pipeline: branch folding drops a dead arm and runs correctly on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const tv = try func.appendBlockParam(then_b, t);
    const ev = try func.appendBlockParam(else_b, t);
    const rv = try func.appendBlockParam(merge, t);
    const c1 = try func.appendInst(entry, t, .{ .iconst = 1 });
    const c2 = try func.appendInst(entry, t, .{ .iconst = 2 });
    const cmp = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = c1, .rhs = c2 } }); // false
    try func.appendIf(entry, cmp, .{ .target = then_b, .args = &.{x} }, .{ .target = else_b, .args = &.{x} });
    const t100 = try func.appendArithImm(then_b, t, .add, tv, 100); // dead path
    try func.setJump(then_b, merge, &.{t100});
    const e1 = try func.appendArithImm(else_b, t, .add, ev, 1);
    try func.setJump(else_b, merge, &.{e1});
    func.setTerminator(merge, .{ .ret = rv });

    // The full default pipeline: constfold makes cmp constant-false, branchfold turns entry's if into
    // `jump else_b` (leaving then_b dead), and GVN/LICM/DCE run over the CFG that still contains the
    // unreachable, param-carrying then_b, which the now-reachability-aware analyses tolerate.
    try std.testing.expect(try opt.optimize(allocator, &func));
    try expectRun(allocator, &func, &.{41}, 42); // takes the else path: x + 1
}

/// Build a function whose result `opt.optimize` must leave unchanged. The equivalence tests below
/// each run the unoptimized and optimized forms and assert they agree on real hardware.
fn eqIdentitiesStrength(func: *Function) !void {
    // (x*1) + (x-x) + (x*4) = 5x  (arith identities + strength reduction)
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const m1 = try func.appendArithImm(b, t, .mul, x, 1);
    const sxx = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = x } });
    const m4 = try func.appendArithImm(b, t, .mul, x, 4);
    const s1 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = m1, .rhs = sxx } });
    const s2 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = m4 } });
    func.setTerminator(b, .{ .ret = s2 });
}

fn eqConstSelect(func: *Function) !void {
    // (1 < 2) ? x*2 : x*3  -> 2x  (constfold + select fold + strength)
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const c1 = try func.appendInst(b, t, .{ .iconst = 1 });
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const cmp = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = c1, .rhs = c2 } });
    const m2 = try func.appendArithImm(b, t, .mul, x, 2);
    const m3 = try func.appendArithImm(b, t, .mul, x, 3);
    const sel = try func.appendInst(b, t, .{ .select = .{ .cond = cmp, .then = m2, .@"else" = m3 } });
    func.setTerminator(b, .{ .ret = sel });
}

fn eqConstBranch(func: *Function) !void {
    // if (2 > 1) return x+10 else return x-10  -> x+10  (branch folding)
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const bb = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const av = try func.appendBlockParam(a, t);
    const bv = try func.appendBlockParam(bb, t);
    const c1 = try func.appendInst(entry, t, .{ .iconst = 1 });
    const c2 = try func.appendInst(entry, t, .{ .iconst = 2 });
    const cmp = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = c2, .rhs = c1 } });
    try func.appendIf(entry, cmp, .{ .target = a, .args = &.{x} }, .{ .target = bb, .args = &.{x} });
    const ap = try func.appendArithImm(a, t, .add, av, 10);
    func.setTerminator(a, .{ .ret = ap });
    const bm = try func.appendArithImm(bb, t, .sub, bv, 10);
    func.setTerminator(bb, .{ .ret = bm });
}

fn eqSelfCmpSelect(func: *Function) !void {
    // (x == x) ? x+1 : x-1  -> x+1  (self-comparison fold + select fold)
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const cmp = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = x } });
    const p = try func.appendArithImm(b, t, .add, x, 1);
    const q = try func.appendArithImm(b, t, .sub, x, 1);
    const sel = try func.appendInst(b, t, .{ .select = .{ .cond = cmp, .then = p, .@"else" = q } });
    func.setTerminator(b, .{ .ret = sel });
}

test "optimization is semantics-preserving: opt vs non-opt agree on aarch64" {
    const allocator = std.testing.allocator;
    const builders = [_]*const fn (*Function) anyerror!void{
        eqIdentitiesStrength, eqConstSelect, eqConstBranch, eqSelfCmpSelect,
    };
    const args = [_]i32{ 0, 1, 7, 100, -3 };
    for (builders) |build| {
        for (args) |arg| {
            var f0 = Function.init(allocator);
            defer f0.deinit();
            try build(&f0);
            const reference = try run(allocator, &f0, &.{arg});

            var f1 = Function.init(allocator);
            defer f1.deinit();
            try build(&f1);
            _ = try opt.optimize(allocator, &f1);
            const optimized = try run(allocator, &f1, &.{arg});

            try std.testing.expectEqual(reference, optimized);
        }
    }
}

test "multi-block inlining preserves semantics on aarch64 (callee has a loop)" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee sumto(n) = 0+1+...+(n-1), a counted loop: 4 blocks with a back-edge (body -> loop).
    const buildCallee = struct {
        fn go(a: std.mem.Allocator) !Function {
            var f = Function.init(a);
            const t = try f.types.intern(i32k);
            const bt = try f.types.intern(.bool);
            const entry = try f.appendBlock();
            const loop = try f.appendBlock();
            const body = try f.appendBlock();
            const done = try f.appendBlock();
            const n = try f.appendBlockParam(entry, t);
            const i = try f.appendBlockParam(loop, t);
            const acc = try f.appendBlockParam(loop, t);
            const bi = try f.appendBlockParam(body, t);
            const bacc = try f.appendBlockParam(body, t);
            const racc = try f.appendBlockParam(done, t);
            const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
            try f.setJump(entry, loop, &.{ zero, zero });
            const cmp = try f.appendInst(loop, bt, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
            try f.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
            const ni = try f.appendArithImm(body, t, .add, bi, 1);
            const nacc = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
            try f.setJump(body, loop, &.{ ni, nacc });
            f.setTerminator(done, .{ .ret = racc });
            return f;
        }
    }.go;
    // caller f(x) = sumto(x) + 100  (code after the call exercises the continuation split).
    const buildCaller = struct {
        fn go(a: std.mem.Allocator) !Function {
            var f = Function.init(a);
            const t = try f.types.intern(i32k);
            const b = try f.appendBlock();
            const x = try f.appendBlockParam(b, t);
            const s = try f.appendCall(b, t, "sumto", &.{x});
            const r = try f.appendArithImm(b, t, .add, s, 100);
            f.setTerminator(b, .{ .ret = r });
            return f;
        }
    }.go;

    const Lk = struct {
        callee: *const Function,
        fn get(ctx: *anyopaque, name: []const u8) ?*const Function {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return if (std.mem.eql(u8, name, "sumto")) self.callee else null;
        }
    };

    for ([_]i32{ 0, 1, 5, 10 }) |arg| {
        // Reference: the linked module (a real call), non-inlined.
        var callee0 = try buildCallee(allocator);
        defer callee0.deinit();
        var caller0 = try buildCaller(allocator);
        defer caller0.deinit();
        const reference = try runModule(allocator, &.{ .{ .name = "f", .func = &caller0 }, .{ .name = "sumto", .func = &callee0 } }, &.{arg});

        // Inlined: clone the callee's blocks (loop and all) into the caller, then run the one function.
        var callee1 = try buildCallee(allocator);
        defer callee1.deinit();
        var caller1 = try buildCaller(allocator);
        defer caller1.deinit();
        var lk = Lk{ .callee = &callee1 };
        try std.testing.expect(try opt.inlining.run(allocator, &caller1, .{ .context = &lk, .func = Lk.get }));
        for (0..caller1.blockCount()) |bi| for (caller1.blockInsts(@enumFromInt(bi))) |inst| {
            try std.testing.expect(caller1.opcode(inst) != .call); // the call was inlined away
        };
        const inlined = try run(allocator, &caller1, &.{arg});

        try std.testing.expectEqual(reference, inlined);
    }
}

/// Builds `out.<4 x i32> = dot(*zero_ptr, *a_ptr, *b_ptr)`: load the zero accumulator,
/// the two `<16 x i8>`/`<16 x u8>` operands (signedness picked by `signed`), dot them,
/// store the `<4 x i32>` result. Mirrors the existing pointer-argument NEON tests above
/// (e.g. the block-edge and high-pressure vector tests) rather than building the int8
/// operands through allocas + a store loop.
fn dotFunc(allocator: std.mem.Allocator, signed: bool) !Function {
    var func = Function.init(allocator);
    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const acc_t = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = if (signed) .signed else .unsigned, .bits = 8 } });
    const data_t = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });
    const entry = try func.appendBlock();
    const out = try func.appendBlockParam(entry, ptr_t);
    const zero_ptr = try func.appendBlockParam(entry, ptr_t);
    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const acc = try func.appendInst(entry, acc_t, .{ .load = .{ .ptr = zero_ptr } });
    const va = try func.appendInst(entry, data_t, .{ .load = .{ .ptr = a_ptr } });
    const vb = try func.appendInst(entry, data_t, .{ .load = .{ .ptr = b_ptr } });
    const result = try func.appendDot(entry, acc, va, vb);
    try func.appendStore(entry, result, out);
    func.setTerminator(entry, .{ .ret = null });
    return func;
}

/// Asserts `code` contains an SDOT/UDOT word. Register fields (rd/rn/rm) vary with
/// whatever the allocator picked, so the check masks them off and compares only the
/// fixed opcode bits (the same field layout `sdot`/`udot` themselves encode).
fn expectHasDot(code: []const u32, signed: bool) !void {
    const reg_mask: u32 = 0x001F03FF; // rd[4:0] | rn[9:5] | rm[20:16]
    const fixed = (if (signed) encode.sdot(.x0, .x0, .x0) else encode.udot(.x0, .x0, .x0)) & ~reg_mask;
    for (code) |w| if (w & ~reg_mask == fixed) return;
    return error.TestExpectedEqual; // no sdot/udot word found in the emitted code
}

test "neon: SDOT computes the INT8 dot-product-accumulate (signed, matches a scalar reference)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try dotFunc(allocator, true);
    defer func.deinit();
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    try expectHasDot(code, true);

    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]i32, *const [4]i32, *const [16]i8, *const [16]i8) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);

    const as = [_][16]i8{
        .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 },
        .{ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16 },
        .{ 127, -128, 127, -128, 1, -1, 0, 0, 100, -100, 50, -50, 25, -25, 10, -10 },
        .{ -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128 },
    };
    const bs = [_][16]i8{
        .{ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 },
        .{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        .{ -128, 127, -128, 127, 1, -1, 0, 0, -100, 100, -50, 50, -25, 25, -10, 10 },
        .{ 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 },
    };
    const zero: [4]i32 align(16) = .{ 0, 0, 0, 0 };
    for (as, bs) |a, b| {
        var scalar_ref: i32 = 0;
        for (0..16) |i| scalar_ref += @as(i32, a[i]) * @as(i32, b[i]);
        const aa: [16]i8 align(16) = a;
        const bb: [16]i8 align(16) = b;
        var out: [4]i32 align(16) = undefined;
        f(&out, &zero, &aa, &bb);
        const sum = out[0] + out[1] + out[2] + out[3];
        try std.testing.expectEqual(scalar_ref, sum);
    }
}

test "neon: UDOT computes the INT8 dot-product-accumulate (unsigned, matches a scalar reference)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try dotFunc(allocator, false);
    defer func.deinit();
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    try expectHasDot(code, false);

    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]i32, *const [4]i32, *const [16]u8, *const [16]u8) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);

    const as = [_][16]u8{
        .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 },
        .{ 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240 },
        .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200 },
    };
    const bs = [_][16]u8{
        .{ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 },
        .{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        .{ 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200 },
        .{ 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200 },
    };
    const zero: [4]i32 align(16) = .{ 0, 0, 0, 0 };
    for (as, bs) |a, b| {
        var scalar_ref: i32 = 0;
        for (0..16) |i| scalar_ref += @as(i32, a[i]) * @as(i32, b[i]);
        const aa: [16]u8 align(16) = a;
        const bb: [16]u8 align(16) = b;
        var out: [4]i32 align(16) = undefined;
        f(&out, &zero, &aa, &bb);
        const sum = out[0] + out[1] + out[2] + out[3];
        try std.testing.expectEqual(scalar_ref, sum);
    }
}

//
// f16 is emulated: an f16 value lives in an S register as its f32 WIDENING (a value exactly
// representable in half), and the boundaries round via base-ISA `fcvt` (no FEAT_FP16): a load
// is `ldr h; fcvt s,h`, a store is `fcvt h,s; str h`, every arithmetic result and narrowing
// convert rounds to nearest-even half with `fcvt`. These tests JIT the code on this aarch64
// host and assert the result bit-matches Zig's own `@as(f16, ...)` reference (Zig lowers f16
// ops the same way: promote to f32, operate, round back to half). f16 crosses the JIT boundary
// only through MEMORY (the 2-byte IEEE-half layout Zig's `f16` also uses), never in an argument
// register, so these validate the emulation itself. Vulcan's own register convention for an f16
// is the f32 widening, consistent across calls, so this is not the C half-format ABI.

fn f16Bits(x: f16) u16 {
    return @bitCast(x);
}

/// JIT a binary f16 op done through memory (out = a <op> b, each an f16 in memory) and return
/// the stored half so the caller compares its bits to Zig's f16 reference.
fn runF16Binary(allocator: std.mem.Allocator, op: ir.function.BinOp, a: f16, b: f16) !f16 {
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pout = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pb } });
    const r = try func.appendInst(blk, f16_t, .{ .arith = .{ .op = op, .lhs = va, .rhs = vb } });
    try func.appendStore(blk, r, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const f16, *const f16, *f16) callconv(.c) void;
    var out: f16 = 0;
    @as(Fn, @ptrCast(buf.memory.ptr))(&a, &b, &out);
    return out;
}

test "f16 load/store round-trips a half value bit-exact" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // out = *in: `ldr h; fcvt s,h` on the load, `fcvt h,s; str h` on the store. The value is
    // already an exact half, so its 16 bits must survive the round-trip unchanged.
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const pin = try func.appendBlockParam(blk, ptr_t);
    const pout = try func.appendBlockParam(blk, ptr_t);
    const v = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pin } });
    try func.appendStore(blk, v, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const f16, *f16) callconv(.c) void;
    // A spread of halves: normal, negative, zero, the largest finite half, a subnormal.
    for ([_]f16{ 1.5, -2.25, 0.0, 65504.0, 0.00006103515625 }) |x| {
        var in: f16 = x;
        var out: f16 = 0;
        @as(Fn, @ptrCast(buf.memory.ptr))(&in, &out);
        try std.testing.expectEqual(f16Bits(x), f16Bits(out));
    }
}

test "f16 add/sub/mul/div match Zig's per-op half rounding (bit-exact)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const cases = [_]struct { a: f16, b: f16 }{
        .{ .a = 1.5, .b = 2.25 },
        .{ .a = 3.140625, .b = 0.5 },
        .{ .a = 100.0, .b = 7.0 },
        .{ .a = 0.1, .b = 0.2 }, // 0.1 and 0.2 are not exact in f16, so the results round
    };
    for (cases) |c| {
        // The reference (`c.a + c.b`, etc.) is Zig's own f16 arithmetic: promote to f32,
        // operate, round to nearest-even half. The emulation must produce the same bits.
        try std.testing.expectEqual(f16Bits(c.a + c.b), f16Bits(try runF16Binary(allocator, .add, c.a, c.b)));
        try std.testing.expectEqual(f16Bits(c.a - c.b), f16Bits(try runF16Binary(allocator, .sub, c.a, c.b)));
        try std.testing.expectEqual(f16Bits(c.a * c.b), f16Bits(try runF16Binary(allocator, .mul, c.a, c.b)));
        try std.testing.expectEqual(f16Bits(c.a / c.b), f16Bits(try runF16Binary(allocator, .div, c.a, c.b)));
    }
}

test "f16 multiply rounds its result to nearest-even half (not a raw f32 product)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // a*a whose exact f32 product is NOT representable in f16, so the half result must round.
    // This proves the arith path narrows to half, rather than leaving the f32 product in place.
    const a: f16 = 1.0009765625; // 1 + 2^-10, itself exactly representable
    const got = try runF16Binary(allocator, .mul, a, a);
    try std.testing.expectEqual(f16Bits(a * a), f16Bits(got)); // matches Zig's f16 multiply
    const exact_f32: f32 = @as(f32, a) * @as(f32, a);
    try std.testing.expect(@as(f32, got) != exact_f32); // rounding to half changed the value
    try std.testing.expectEqual(f16Bits(@as(f16, @floatCast(exact_f32))), f16Bits(got));
}

test "f16 chained multiply rounds every intermediate to half" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // r = (a*b)*c. Each multiply must round its result to half before the next consumes it
    // (fp mul/add fusion is disabled for f16, and each op re-rounds). Compared against Zig's
    // step-by-step f16 chain.
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    const pout = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pb } });
    const vc = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pc } });
    const ab = try func.appendInst(blk, f16_t, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const abc = try func.appendInst(blk, f16_t, .{ .arith = .{ .op = .mul, .lhs = ab, .rhs = vc } });
    try func.appendStore(blk, abc, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const f16, *const f16, *const f16, *f16) callconv(.c) void;
    var a: f16 = 1.0009765625;
    var b: f16 = 1.0029296875;
    var c: f16 = 1.0048828125;
    var out: f16 = 0;
    @as(Fn, @ptrCast(buf.memory.ptr))(&a, &b, &c, &out);
    const ref: f16 = (a * b) * c; // Zig: rounds a*b to half, then rounds (that*c) to half
    try std.testing.expectEqual(f16Bits(ref), f16Bits(out));
}

test "convert f16 -> f32 widens exactly" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const pin = try func.appendBlockParam(blk, ptr_t);
    const pout = try func.appendBlockParam(blk, ptr_t);
    const v = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pin } });
    const w = try func.appendInst(blk, f32_t, .{ .convert = .{ .value = v } }); // f16 -> f32
    try func.appendStore(blk, w, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const f16, *f32) callconv(.c) void;
    for ([_]f16{ 3.140625, -0.5, 65504.0 }) |x| {
        var in: f16 = x;
        var out: f32 = 0;
        @as(Fn, @ptrCast(buf.memory.ptr))(&in, &out);
        try std.testing.expectEqual(@as(f32, x), out); // widening a half to f32 is exact
    }
}

test "convert f32 -> f16 rounds to nearest-even half (proves it is not a bare copy)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const pin = try func.appendBlockParam(blk, ptr_t);
    const pout = try func.appendBlockParam(blk, ptr_t);
    const v = try func.appendInst(blk, f32_t, .{ .load = .{ .ptr = pin } });
    const w = try func.appendInst(blk, f16_t, .{ .convert = .{ .value = v } }); // f32 -> f16, ROUNDS
    try func.appendStore(blk, w, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const f32, *f16) callconv(.c) void;
    // 3.14159 is not representable in f16: it rounds to 3.140625 (bits 0x4248). Were the convert
    // a bare `fmov` (the old `sd == dd` trap), the stored half would carry f32 bits instead.
    var in: f32 = 3.14159;
    var out: f16 = 0;
    @as(Fn, @ptrCast(buf.memory.ptr))(&in, &out);
    try std.testing.expectEqual(f16Bits(@as(f16, @floatCast(in))), f16Bits(out));
    try std.testing.expect(@as(f32, out) != in); // rounding to half changed the value
    try std.testing.expectEqual(@as(u16, 0x4248), f16Bits(out));
}

test "convert int <-> f16 rounds int->f16 and truncates f16->int" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // int -> f16: 2049 is not exactly representable in f16 (the step at 2^11 is 2), so it rounds
    // to 2048. scvtf lands in the S view, then the result rounds to nearest-even half.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const p_t = try func.types.intern(.ptr);
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const f16_t = try func.types.intern(.{ .float = .f16 });
        const blk = try func.appendBlock();
        const x = try func.appendBlockParam(blk, i32_t);
        const pout = try func.appendBlockParam(blk, p_t);
        const w = try func.appendInst(blk, f16_t, .{ .convert = .{ .value = x } }); // i32 -> f16
        try func.appendStore(blk, w, pout);
        func.setTerminator(blk, .{ .ret = null });

        const code = try isel.selectFunction(allocator, &func);
        defer allocator.free(code);
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
        defer buf.deinit();
        const Fn = *const fn (i32, *f16) callconv(.c) void;
        var out: f16 = 0;
        @as(Fn, @ptrCast(buf.memory.ptr))(2049, &out);
        try std.testing.expectEqual(f16Bits(@as(f16, @floatFromInt(@as(i32, 2049)))), f16Bits(out));
        try std.testing.expectEqual(@as(f16, 2048.0), out); // rounded down to the even step
        // A small value that IS exactly representable converts without change.
        @as(Fn, @ptrCast(buf.memory.ptr))(-7, &out);
        try std.testing.expectEqual(@as(f16, -7.0), out);
    }
    // f16 -> int: fcvtzs from the S-held half, round toward zero (truncate).
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const p_t = try func.types.intern(.ptr);
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const f16_t = try func.types.intern(.{ .float = .f16 });
        const blk = try func.appendBlock();
        const pin = try func.appendBlockParam(blk, p_t);
        const v = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = pin } });
        const r = try func.appendInst(blk, i32_t, .{ .convert = .{ .value = v } }); // f16 -> i32
        func.setTerminator(blk, .{ .ret = r });

        const code = try isel.selectFunction(allocator, &func);
        defer allocator.free(code);
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
        defer buf.deinit();
        const Fn = *const fn (*const f16) callconv(.c) i32;
        var in: f16 = 3.5;
        try std.testing.expectEqual(@as(i32, 3), @as(Fn, @ptrCast(buf.memory.ptr))(&in));
        in = -3.9;
        try std.testing.expectEqual(@as(i32, -3), @as(Fn, @ptrCast(buf.memory.ptr))(&in));
    }
}

test "f16 constant materializes as its half-rounded f32 widening" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const pout = try func.appendBlockParam(blk, ptr_t);
    // 3.14159 rounds to the half 3.140625: the fconst must store the ROUNDED half, not the f64.
    const k = try func.appendInst(blk, f16_t, .{ .fconst = 3.14159 });
    try func.appendStore(blk, k, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*f16) callconv(.c) void;
    var out: f16 = 0;
    @as(Fn, @ptrCast(buf.memory.ptr))(&out);
    try std.testing.expectEqual(f16Bits(@as(f16, @floatCast(@as(f64, 3.14159)))), f16Bits(out));
}

test "f16 survives register spilling bit-exact (held as its f32 widening in a 16-byte slot)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // Load N f16 values that are ALL live at once, then left-fold add them. N far exceeds the
    // FP register pool, so many f16 values spill and reload. An f16 lives in an S register as
    // its f32 widening and spills through the uniform scalar-fpr slot (`str d`/`ldr d`, the
    // value in the low 32 bits), so a spilled half must reload with its exact value intact.
    const N = 40;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const blk = try func.appendBlock();
    const in = try func.appendBlockParam(blk, ptr_t);
    const pout = try func.appendBlockParam(blk, ptr_t);
    var v: [N]ir.function.Value = undefined;
    for (0..N) |i| {
        const p = if (i == 0) in else try func.appendInst(blk, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = @intCast(i * 2) } });
        v[i] = try func.appendInst(blk, f16_t, .{ .load = .{ .ptr = p } });
    }
    var s = v[0];
    for (1..N) |i| s = try func.appendInst(blk, f16_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = v[i] } });
    try func.appendStore(blk, s, pout);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const [N]f16, *f16) callconv(.c) void;
    var input: [N]f16 = undefined;
    for (0..N) |i| input[i] = @floatCast(@as(f32, @floatFromInt(i)) * 0.1);
    var out: f16 = 0;
    @as(Fn, @ptrCast(buf.memory.ptr))(&input, &out);
    // The reference folds in the SAME order with per-op half rounding (f16 add is not
    // associative, so the order must match the IR's left fold).
    var ref: f16 = input[0];
    for (1..N) |i| ref = ref + input[i];
    try std.testing.expectEqual(f16Bits(ref), f16Bits(out));
}

test "f32/f64 through the same memory paths are unchanged by the f16 work (regression)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // out = (*a + *b) as f32, and separately as f64: the non-f16 float load/store/arith paths
    // must be byte-identical to before (the f16 branches never touch f32/f64).
    inline for (.{ f32, f64 }) |T| {
        var func = Function.init(allocator);
        defer func.deinit();
        const ptr_t = try func.types.intern(.ptr);
        const ft = try func.types.intern(.{ .float = if (T == f32) .f32 else .f64 });
        const blk = try func.appendBlock();
        const pa = try func.appendBlockParam(blk, ptr_t);
        const pb = try func.appendBlockParam(blk, ptr_t);
        const pout = try func.appendBlockParam(blk, ptr_t);
        const va = try func.appendInst(blk, ft, .{ .load = .{ .ptr = pa } });
        const vb = try func.appendInst(blk, ft, .{ .load = .{ .ptr = pb } });
        const r = try func.appendInst(blk, ft, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
        try func.appendStore(blk, r, pout);
        func.setTerminator(blk, .{ .ret = null });

        const code = try isel.selectFunction(allocator, &func);
        defer allocator.free(code);
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
        defer buf.deinit();
        const Fn = *const fn (*const T, *const T, *T) callconv(.c) void;
        var a: T = 1.25;
        var b: T = 2.5;
        var out: T = 0;
        @as(Fn, @ptrCast(buf.memory.ptr))(&a, &b, &out);
        try std.testing.expectEqual(@as(T, 3.75), out);
    }
}

// --- ldp/stp peephole: targeted execution validation (Task 4) -----------------------------------
//
// The always-on `pairMemory` pass fuses adjacent same-base consecutive ldr/str into ldp/stp. These
// tests originally forced fusion onto the callee-saved register runs a NON-LEAF function saves in its
// prologue and restores in its epilogue (consecutive same-base 8-byte-spaced stores/loads), because
// at the time the data-load/store path always addressed `[base, #0]`. Address folding (Task 3) now
// also lets constant-index data loads/stores fold to `[base, #off]` and fuse, so the copy tests below
// fuse their data accesses too. The callee-saved-run fusion still fires; the counts stay >= 1.
//
// ldp/stp are deliberately NOT `peephole.decodeMem` forms (that decoder returns null for them by
// design, treating a fused word as opaque), so a fused pair is detected here by matching the exact
// base opcodes of the four GPR ldp/stp encoders (ldpOffX/stpOffX/ldpOffW/stpOffW in encode.zig).
// Their fixed class bits live in [31:22]. imm7/rt2/rn/rt1 are all below bit 22, so masking with
// 0xFFC00000 isolates the class cleanly.
const MemPairKind = enum { none, ldp, stp };

fn memPairKind(word: u32) MemPairKind {
    return switch (word & 0xFFC00000) {
        0xA9400000, 0x29400000 => .ldp, // ldpOffX / ldpOffW
        0xA9000000, 0x29000000 => .stp, // stpOffX / stpOffW
        else => .none,
    };
}

/// Count the fused ldp/stp words in a compiled function. Returns `.{ ldp, stp }`.
fn countMemPairs(allocator: std.mem.Allocator, func: *const Function) ![2]usize {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var counts = [2]usize{ 0, 0 };
    for (code) |w| switch (memPairKind(w)) {
        .none => {},
        .ldp => counts[0] += 1,
        .stp => counts[1] += 1,
    };
    return counts;
}

test "ldp/stp: memPairKind matches only the four GPR pair encoders and rejects ldr/str and non-mem" {
    // Guards the emission probe itself: it must fire on real ldp/stp words and stay silent on the
    // ldr/str forms it is meant to distinguish from, otherwise the emission assertions below are
    // meaningless. (This is the by-construction check the brief asks for.)
    try std.testing.expectEqual(MemPairKind.ldp, memPairKind(encode.ldpOffX(.x0, .x1, .x2, 16)));
    try std.testing.expectEqual(MemPairKind.stp, memPairKind(encode.stpOffX(.x0, .x1, .x2, 16)));
    try std.testing.expectEqual(MemPairKind.ldp, memPairKind(encode.ldpOffW(.x0, .x1, .x2, 8)));
    try std.testing.expectEqual(MemPairKind.stp, memPairKind(encode.stpOffW(.x0, .x1, .x2, 8)));
    // The individual (unfused) memory ops must NOT be miscounted as pairs.
    try std.testing.expectEqual(MemPairKind.none, memPairKind(encode.ldrOff(.x0, .x1, 16)));
    try std.testing.expectEqual(MemPairKind.none, memPairKind(encode.strOff(.x0, .x1, 16)));
    try std.testing.expectEqual(MemPairKind.none, memPairKind(encode.ldrW(.x0, .x1, 8)));
    try std.testing.expectEqual(MemPairKind.none, memPairKind(encode.strW(.x0, .x1, 8)));
    try std.testing.expectEqual(MemPairKind.none, memPairKind(encode.add64(.x0, .x1, .x2)));
    try std.testing.expectEqual(MemPairKind.none, memPairKind(encode.nop()));
}

test "ldp/stp: a leaf non-fusable function emits zero pairs" {
    // Meaningfulness anchor for the emission counts below: a trivial leaf `x + 1` has no callee-saved
    // run and no fusable adjacency, so the probe must count exactly zero pairs. If this were nonzero,
    // the `>= 1` assertions in the fusing tests would prove nothing.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const r = try func.appendArithImm(b, t, .add, x, 1);
    func.setTerminator(b, .{ .ret = r });
    const counts = try countMemPairs(allocator, &func);
    try std.testing.expectEqual(@as(usize, 0), counts[0]);
    try std.testing.expectEqual(@as(usize, 0), counts[1]);
}

test "ldp/stp: a spill/reload heavy function still computes correctly and emits ldp/stp" {
    // f(x) = g(x) + sum_{i=1..24}(g(x) + i), with g(x) = 3x. `main` is non-leaf (it calls g), so
    // its GPR pool is x19-x28 (callee-saved) from the start, not just on overflow. Those callee-saved
    // registers get saved in the prologue and restored in the epilogue as consecutive same-base
    // stores/loads, which pairMemory fuses into stp then ldp. The 24 live partials only decide how
    // many registers (and so how many pairs) are needed, not whether fusion fires.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var g = Function.init(allocator);
    defer g.deinit();
    const gt = try g.types.intern(t_kind);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gt);
    const gm = try g.appendArithImm(gb, gt, .mul, gx, 3);
    g.setTerminator(gb, .{ .ret = gm });

    var main = Function.init(allocator);
    defer main.deinit();
    const mt = try main.types.intern(t_kind);
    const mb = try main.appendBlock();
    const mx = try main.appendBlockParam(mb, mt);
    const c = try main.appendCall(mb, mt, "g", &.{mx});
    var vals: [24]ir.function.Value = undefined;
    for (0..24) |i| vals[i] = try main.appendArithImm(mb, mt, .add, c, @intCast(i + 1));
    var acc = vals[0];
    for (1..24) |i| acc = try main.appendInst(mb, mt, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[i] } });
    main.setTerminator(mb, .{ .ret = acc });

    // Emission: the always-on pass runs inside compileFunction, so a fused prologue/epilogue must be
    // present. At least one stp (prologue) and one ldp (epilogue).
    const counts = try countMemPairs(allocator, &main);
    try std.testing.expect(counts[0] >= 1); // ldp
    try std.testing.expect(counts[1] >= 1); // stp

    // Correctness: link main+g so the `bl` resolves, then run across a sweep. Fusing the callee-saved
    // save/restore is only correct if every paired register round-trips its bits, so a wrong fusion
    // would corrupt the reduction and change the result. Expected = 24*(3x) + sum_{i=1..24} i.
    const funcs = [_]NamedFunc{ .{ .name = "main", .func = &main }, .{ .name = "g", .func = &g } };
    const inputs = [_]i32{ 0, 1, 2, 3, -1, 5, 10, -7 };
    for (inputs) |x| {
        var expected: i32 = 0;
        var i: i32 = 1;
        while (i <= 24) : (i += 1) expected +%= (3 *% x) +% i;
        const got = try runModule(allocator, &funcs, &.{x});
        try std.testing.expectEqual(expected, got);
    }
}

test "ldp/stp: a struct-copy of consecutive words matches and emits ldp/stp" {
    // Copy 12 consecutive i64 words in[j] -> out[j]. All 12 loaded values are held live across a call
    // to a helper (making the function non-leaf), so they occupy callee-saved registers that the
    // prologue saves and the epilogue restores as consecutive same-base runs. pairMemory fuses those
    // into stp/ldp, and the copy is only correct if each paired save/restore preserves its 64 bits.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const n = 12;
    const i64_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 64 } };

    // Side-effect-free helper the copy calls purely to become non-leaf.
    var g = Function.init(allocator);
    defer g.deinit();
    const gt = try g.types.intern(i64_kind);
    const gb = try g.appendBlock();
    const gz = try g.appendInst(gb, gt, .{ .iconst = 0 });
    g.setTerminator(gb, .{ .ret = gz });

    var main = Function.init(allocator);
    defer main.deinit();
    const t = try main.types.intern(i64_kind);
    const ptr_t = try main.types.intern(.ptr);
    const b = try main.appendBlock();
    const out = try main.appendBlockParam(b, ptr_t);
    const in = try main.appendBlockParam(b, ptr_t);
    var loaded: [n]ir.function.Value = undefined;
    for (0..n) |i| {
        const p = if (i == 0) in else try main.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = @intCast(i * 8) } });
        loaded[i] = try main.appendInst(b, t, .{ .load = .{ .ptr = p } });
    }
    _ = try main.appendCall(b, t, "g", &.{});
    for (0..n) |i| {
        const p = if (i == 0) out else try main.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = out, .imm = @intCast(i * 8) } });
        try main.appendStore(b, loaded[i], p);
    }
    main.setTerminator(b, .{ .ret = null });

    // Emission: at least one stp (prologue) and one ldp (epilogue).
    const counts = try countMemPairs(allocator, &main);
    try std.testing.expect(counts[0] >= 1); // ldp
    try std.testing.expect(counts[1] >= 1); // stp

    // Correctness: link main+g, map, and copy a distinctly-valued source. A dropped/lane-truncated
    // fusion of the callee-saved run would corrupt one of the 12 in-flight words and mismatch here.
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "g", &g);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    defer buf.deinit();
    const Fn = *const fn (*[n]i64, *const [n]i64) callconv(.c) void;
    var src: [n]i64 = undefined;
    for (0..n) |i| src[i] = @as(i64, 0x1122_3344_0000_0000) + @as(i64, @intCast(i)) * 7 - 3;
    var dst: [n]i64 = undefined;
    @memset(&dst, -1);
    @as(Fn, @ptrCast(buf.memory.ptr))(&dst, &src);
    try std.testing.expectEqualSlices(i64, &src, &dst);
}

test "ldp/stp: adjacent scalar loads summed match" {
    // Load a[0..4] (consecutive i32 words) and sum them. Under address folding the constant-index
    // loads fold to `[base, #4j]` and the adjacent ones fuse to ldp, so this reconfirms that a plain
    // adjacent-load kernel still computes the right value across a sweep of inputs after folding.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    var lanes: [4]ir.function.Value = undefined;
    for (0..4) |i| {
        const p = if (i == 0) base else try func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = @intCast(i * 4) } });
        lanes[i] = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    }
    var acc = lanes[0];
    for (1..4) |i| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = lanes[i] } });
    func.setTerminator(b, .{ .ret = acc });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const [4]i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const cases = [_][4]i32{
        .{ 1, 2, 3, 4 },
        .{ 0, 0, 0, 0 },
        .{ -10, 20, -30, 40 },
        .{ 100, -1, -1, -1 },
    };
    for (cases) |arr| {
        const expected = arr[0] +% arr[1] +% arr[2] +% arr[3];
        try std.testing.expectEqual(expected, f(&arr));
    }
}

// --- address-mode folding: execution validation (Task 3) ----------------------------------------
//
// A load/store whose pointer is a foldable `arith_imm.add(base, imm)` now addresses `[base, #imm]`
// directly: the address-add is dropped and the now-adjacent offset-form loads/stores present the
// shape `pairMemory` fuses into ldp/stp. These tests JIT-run the folded code on this Ampere host and
// assert both the computed values AND that folding actually fired (address-adds gone, pairs emitted).

/// Count the register-form add words (`add xd, xn, xm` and `add wd, wn, wm`, LSL #0) in a compiled
/// function. `arith_imm.add` lowers to the register form (its immediate is materialized in a scratch),
/// so an address-add shows up here; when it folds into a load/store displacement it disappears. A
/// pure constant-index copy therefore has ZERO adds once folded. The add shifted-register family
/// fixes every bit except Rm[20:16]/Rn[9:5]/Rd[4:0] (and shift 0 keeps imm6/shift-type clear), so
/// masking those register fields off isolates the class (0x8B for 64-bit, 0x0B for 32-bit).
fn countAdds(allocator: std.mem.Allocator, func: *const Function) !usize {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var count: usize = 0;
    for (code) |w| switch (w & 0xFFE0FC00) {
        0x8B000000, 0x0B000000 => count += 1,
        else => {},
    };
    return count;
}

test "addrfold: constant-index i64 copy folds and adjacent pairs emit ldp/stp" {
    // out[j] = in[j] for j in 0..4, each addressed through `in + 8*j` / `out + 8*j`. Every address-add
    // folds into the ldr/str displacement and dies, leaving four adjacent `ldr x,[in,#8j]` and four
    // adjacent `str x,[out,#8j]`, which pairMemory fuses into ldp/stp. Proves the fold (no add words,
    // pairs present) AND the data (the copy is correct across a sweep).
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const n = 4;
    const i64_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 64 } };
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i64_kind);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    var loaded: [n]ir.function.Value = undefined;
    for (0..n) |i| {
        const p = if (i == 0) in else try func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = @intCast(i * 8) } });
        loaded[i] = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    }
    for (0..n) |i| {
        const p = if (i == 0) out else try func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = out, .imm = @intCast(i * 8) } });
        try func.appendStore(b, loaded[i], p);
    }
    func.setTerminator(b, .{ .ret = null });

    // The fold eliminated every address-add, so none survive in the machine code (this is a pure
    // copy with no other arithmetic, so a nonzero count would mean an address-add failed to fold).
    try std.testing.expectEqual(@as(usize, 0), try countAdds(allocator, &func));
    // The folded, now-adjacent loads/stores fuse into at least one ldp and one stp.
    const counts = try countMemPairs(allocator, &func);
    try std.testing.expect(counts[0] >= 1); // ldp
    try std.testing.expect(counts[1] >= 1); // stp

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[n]i64, *const [n]i64) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const cases = [_][n]i64{
        .{ 1, 2, 3, 4 },
        .{ 0, -1, 0x7FFF_FFFF_FFFF_FFFF, -0x8000_0000_0000_0000 },
        .{ 0x1122_3344_5566_7788, -3, 42, 0x0102_0304_0506_0708 },
    };
    for (cases) |src| {
        var dst: [n]i64 = undefined;
        @memset(&dst, -1);
        f(&dst, &src);
        try std.testing.expectEqualSlices(i64, &src, &dst);
    }
}

test "addrfold: byte and halfword constant-index loads fold and compute correctly" {
    // Sum four consecutive i8 (offsets 0..3) then four consecutive i16 (offsets 0,2,4,6). The byte
    // loads fold to `ldrsb [base,#off]` (Task-2 u12 unscaled encoders) and the halfword loads to
    // `ldrsh [base,#off]` (Task-2 u13 scaled-by-2 encoders). A wrong offset scale would read the
    // wrong element and mismatch.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const i8_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 8 } };
    const i16_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 16 } };
    const i32_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var byte_fn = Function.init(allocator);
    defer byte_fn.deinit();
    {
        const i8_t = try byte_fn.types.intern(i8_kind);
        const i32_t = try byte_fn.types.intern(i32_kind);
        const ptr_t = try byte_fn.types.intern(.ptr);
        const b = try byte_fn.appendBlock();
        const base = try byte_fn.appendBlockParam(b, ptr_t);
        var acc: ?ir.function.Value = null;
        for (0..4) |i| {
            const p = if (i == 0) base else try byte_fn.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = @intCast(i) } });
            const v = try byte_fn.appendInst(b, i8_t, .{ .load = .{ .ptr = p } });
            const w = try byte_fn.appendInst(b, i32_t, .{ .convert = .{ .value = v } });
            acc = if (acc) |a| try byte_fn.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = w } }) else w;
        }
        byte_fn.setTerminator(b, .{ .ret = acc.? });
    }

    var half_fn = Function.init(allocator);
    defer half_fn.deinit();
    {
        const i16_t = try half_fn.types.intern(i16_kind);
        const i32_t = try half_fn.types.intern(i32_kind);
        const ptr_t = try half_fn.types.intern(.ptr);
        const b = try half_fn.appendBlock();
        const base = try half_fn.appendBlockParam(b, ptr_t);
        var acc: ?ir.function.Value = null;
        for (0..4) |i| {
            const p = if (i == 0) base else try half_fn.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = @intCast(i * 2) } });
            const v = try half_fn.appendInst(b, i16_t, .{ .load = .{ .ptr = p } });
            const w = try half_fn.appendInst(b, i32_t, .{ .convert = .{ .value = v } });
            acc = if (acc) |a| try half_fn.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = w } }) else w;
        }
        half_fn.setTerminator(b, .{ .ret = acc.? });
    }

    const byte_code = try isel.selectFunction(allocator, &byte_fn);
    defer allocator.free(byte_code);
    var byte_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(byte_code));
    defer byte_buf.deinit();
    const ByteFn = *const fn (*const [4]i8) callconv(.c) i32;
    const bf: ByteFn = @ptrCast(byte_buf.memory.ptr);
    const byte_cases = [_][4]i8{ .{ 1, 2, 3, 4 }, .{ -1, -2, -3, -4 }, .{ 127, -128, 5, -5 } };
    for (byte_cases) |arr| {
        const expected: i32 = @as(i32, arr[0]) + arr[1] + arr[2] + arr[3];
        try std.testing.expectEqual(expected, bf(&arr));
    }

    const half_code = try isel.selectFunction(allocator, &half_fn);
    defer allocator.free(half_code);
    var half_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(half_code));
    defer half_buf.deinit();
    const HalfFn = *const fn (*const [4]i16) callconv(.c) i32;
    const hf: HalfFn = @ptrCast(half_buf.memory.ptr);
    const half_cases = [_][4]i16{ .{ 10, 20, 30, 40 }, .{ -100, 200, -300, 400 }, .{ 32767, -32768, 1, -1 } };
    for (half_cases) |arr| {
        const expected: i32 = @as(i32, arr[0]) + arr[1] + arr[2] + arr[3];
        try std.testing.expectEqual(expected, hf(&arr));
    }
}

test "addrfold: fp32 and fp64 constant-index loads fold" {
    // Read a float field at a constant byte offset: an f32 at +4 (folds to `ldr s,[base,#4]`) and an
    // f64 at +8 (folds to `ldr d,[base,#8]`). Returns the field, so a wrong displacement mismatches.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var f32_fn = Function.init(allocator);
    defer f32_fn.deinit();
    {
        const f32_t = try f32_fn.types.intern(.{ .float = .f32 });
        const ptr_t = try f32_fn.types.intern(.ptr);
        const b = try f32_fn.appendBlock();
        const base = try f32_fn.appendBlockParam(b, ptr_t);
        const p = try f32_fn.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = 4 } });
        const v = try f32_fn.appendInst(b, f32_t, .{ .load = .{ .ptr = p } });
        f32_fn.setTerminator(b, .{ .ret = v });
    }

    var f64_fn = Function.init(allocator);
    defer f64_fn.deinit();
    {
        const f64_t = try f64_fn.types.intern(.{ .float = .f64 });
        const ptr_t = try f64_fn.types.intern(.ptr);
        const b = try f64_fn.appendBlock();
        const base = try f64_fn.appendBlockParam(b, ptr_t);
        const p = try f64_fn.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = 8 } });
        const v = try f64_fn.appendInst(b, f64_t, .{ .load = .{ .ptr = p } });
        f64_fn.setTerminator(b, .{ .ret = v });
    }

    const f32_code = try isel.selectFunction(allocator, &f32_fn);
    defer allocator.free(f32_code);
    var f32_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(f32_code));
    defer f32_buf.deinit();
    const F32Fn = *const fn (*const [2]f32) callconv(.c) f32;
    const ff: F32Fn = @ptrCast(f32_buf.memory.ptr);
    const f32_cases = [_][2]f32{ .{ 1.5, -2.25 }, .{ 0.0, 3.14 }, .{ -7.0, 42.0 } };
    for (f32_cases) |arr| try std.testing.expectEqual(arr[1], ff(&arr));

    const f64_code = try isel.selectFunction(allocator, &f64_fn);
    defer allocator.free(f64_code);
    var f64_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(f64_code));
    defer f64_buf.deinit();
    const F64Fn = *const fn (*const [2]f64) callconv(.c) f64;
    const df: F64Fn = @ptrCast(f64_buf.memory.ptr);
    const f64_cases = [_][2]f64{ .{ 1.5, -2.25 }, .{ 0.0, 3.141592653589793 }, .{ -7.0, 42.0 } };
    for (f64_cases) |arr| try std.testing.expectEqual(arr[1], df(&arr));
}

test "addrfold: a CROSS-BLOCK folded load computes correctly" {
    // p = base + 16 is computed in the ENTRY block; a conditional branch reaches then_b, which under
    // register pressure builds 20 simultaneously-live values BEFORE loading [p] and reducing them.
    // The load folds to `[base,#16]`, so the reroute must attribute its pointer use to `base` in the
    // SUCCESSOR block: only then does base stay live across then_b (spilled/kept off the reused pool)
    // instead of being treated as dead after entry, whose register the 20 values would then steal,
    // reading a garbage address. A missing cross-block reroute miscompiles here.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const pressure = 20;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const base = try func.appendBlockParam(entry, ptr_t);
    const cond = try func.appendBlockParam(entry, t);
    const p = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = 16 } });
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = cond, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    // then_b: build the pressure BEFORE the folded load so a stolen base register is read by the load.
    var vals: [pressure]ir.function.Value = undefined;
    for (0..pressure) |i| vals[i] = try func.appendInst(then_b, t, .{ .arith_imm = .{ .op = .add, .lhs = cond, .imm = @intCast(i + 1) } });
    const w = try func.appendInst(then_b, t, .{ .load = .{ .ptr = p } }); // folds to [base, #16]
    var acc = w;
    for (0..pressure) |i| acc = try func.appendInst(then_b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[i] } });
    func.setTerminator(then_b, .{ .ret = acc });
    func.setTerminator(else_b, .{ .ret = cond });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const [8]i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    var arr = [8]i32{ 10, 20, 30, 40, 50, 60, 70, 80 }; // arr[4] sits at byte offset 16
    const conds = [_]i32{ 1, 0, -1, 5, 100, -7 };
    for (conds) |cv| {
        // Perturb arr[4] per case so a stale (wrong) base address could not accidentally match.
        arr[4] = 50 +% cv;
        var expected: i32 = undefined;
        if (cv > 0) {
            expected = arr[4];
            var k: i32 = 1;
            while (k <= pressure) : (k += 1) expected +%= cv +% k;
        } else {
            expected = cv;
        }
        try std.testing.expectEqual(expected, f(&arr, cv));
    }
}

test "addrfold: a loop-invariant base folded-loaded in the header survives the body across the back-edge" {
    // base is loop-invariant (from entry) and folded-loaded in the LOOP HEADER (`w = [base+4]`), then
    // the body builds 20 live values before jumping back. This specifically exercises the SECOND
    // liveness reroute (markUsedBitset, feeding extendLiveRanges' backward dataflow): only if the
    // header's folded load attributes its pointer use to `base` does base stay live across the body
    // and the back-edge. Miss that reroute and the body reuses base's register, so the NEXT
    // iteration's header load reads a garbage address and the accumulation diverges.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const pressure = 20;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const base = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, t);
    // Compute the folded address in ENTRY. `p` is a dead add (its only use is the header load), so it
    // claims no register. base's ONLY in-loop liveness therefore flows through the header load's
    // baseOf reroute, which is precisely what markUsedBitset/extendLiveRanges must carry across the
    // back-edge. (If p lived in the loop, its own operand scan would mark base and mask the reroute.)
    const p = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = 4 } });
    const iv0 = try func.appendInst(entry, t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ iv0, acc0 });
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const w = try func.appendInst(loop, t, .{ .load = .{ .ptr = p } }); // folds to [base, #4]
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc, w } }, .{ .target = done, .args = &.{acc} });
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const bw = try func.appendBlockParam(body, t);
    var s = bw;
    for (0..pressure) |k| {
        const vk = try func.appendInst(body, t, .{ .arith_imm = .{ .op = .add, .lhs = bi, .imm = @intCast(k + 1) } });
        s = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = vk } });
    }
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = s } });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, nacc });
    const racc = try func.appendBlockParam(done, t);
    func.setTerminator(done, .{ .ret = racc });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const [8]i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    var arr = [8]i32{ 10, 20, 30, 40, 50, 60, 70, 80 }; // arr[1] sits at byte offset 4
    const trips = [_]i32{ 0, 1, 2, 4, 7 };
    for (trips) |nn| {
        arr[1] = 20 +% nn; // vary the folded-load target per case
        var expected: i32 = 0;
        var it: i32 = 0;
        while (it < nn) : (it += 1) {
            var si: i32 = arr[1];
            var k: i32 = 1;
            while (k <= pressure) : (k += 1) si +%= it +% k;
            expected +%= si;
        }
        try std.testing.expectEqual(expected, f(&arr, nn));
    }
}

test "addrfold: a base used by a folded load AND another consumer keeps the add and still computes" {
    // p = base + 8 feeds BOTH a folded load (`[base,#8]`) AND a second address-add p2 = p + 8. The
    // load folds, but p has a surviving non-mem use (p2), so p is NOT dead: its add stays materialized
    // (`add x,base,#8`). p2 IS dead (its only use is the second folded load `[p,#8]`). Returns
    // arr[2] + arr[4]. Asserts both the not-dead add survives and the value is correct.
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = 8 } });
    const w1 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } }); // folds to [base, #8] -> arr[2]
    const p2 = try func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = p, .imm = 8 } });
    const w2 = try func.appendInst(b, t, .{ .load = .{ .ptr = p2 } }); // folds to [p, #8] -> arr[4]
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = w1, .rhs = w2 } });
    func.setTerminator(b, .{ .ret = sum });

    // p is NOT dead (p2 uses it beyond the folded load), so its address-add stays materialized. Two
    // register-form adds survive: p's `base + 8` and the final `w1 + w2`. Were p wrongly dropped as
    // dead, w2 would load through an uncomputed base, caught by the value sweep below. (Test 1, where
    // every add is dead, folds to zero adds, so this count genuinely separates the two cases.)
    try std.testing.expect(try countAdds(allocator, &func) >= 2);

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*const [8]i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const cases = [_][8]i32{
        .{ 1, 2, 3, 4, 5, 6, 7, 8 },
        .{ -10, -20, -30, -40, -50, -60, -70, -80 },
        .{ 0, 0, 100, 0, 200, 0, 0, 0 },
    };
    for (cases) |arr| {
        const expected = arr[2] +% arr[4];
        try std.testing.expectEqual(expected, f(&arr));
    }
}

// -- arith-branch fold (flag-setting arith + b.cc) ---------------------------------------------
// The fold rewrites `s = arith(a, b); c = icmp ==/!= (s, 0); if (c)` into one flag-setting S-form
// (`adds`/`subs`/`ands`, or the immediate `addsImm`/`subsImm`) whose Z flag is (s == 0) plus a
// `b.eq`/`b.ne`, eliding the separate `cmp #0`. The icmp is skipped by the compare-and-branch
// fold it composes with. These build the exact adjacent triple and JIT-run it on this aarch64 host.

/// `f(a, b) -> i(bits)` computing `s = a <arith_op> b; if (s <cmp_op> 0) then_val else els_val`,
/// with the constant 0 materialized first so the arith sits immediately before the icmp, which
/// sits immediately before the if (the exact adjacency `fusesArithIntoBranch` folds). `s` is
/// read only by the icmp (single-use), so the fold is eligible.
fn buildArithBranchReg(
    allocator: std.mem.Allocator,
    bits: u16,
    arith_op: ir.function.BinOp,
    cmp_op: ir.function.CmpOp,
    then_val: i64,
    els_val: i64,
) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = bits } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = arith_op, .lhs = a, .rhs = b } });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = cmp_op, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = els_b });
    const tv = try func.appendInst(then_b, t, .{ .iconst = then_val });
    func.setTerminator(then_b, .{ .ret = tv });
    const ev = try func.appendInst(els_b, t, .{ .iconst = els_val });
    func.setTerminator(els_b, .{ .ret = ev });
    return func;
}

/// Like `buildArithBranchReg` but the arith is the immediate form `s = n <arith_op> imm`
/// (the loop `n - 1` decrement), so the fold emits `subsImm`/`addsImm`.
fn buildArithBranchImm(
    allocator: std.mem.Allocator,
    bits: u16,
    arith_op: ir.function.BinOp,
    imm: i64,
    cmp_op: ir.function.CmpOp,
    then_val: i64,
    els_val: i64,
) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = bits } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const s = try func.appendInst(entry, t, .{ .arith_imm = .{ .op = arith_op, .lhs = n, .imm = imm } });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = cmp_op, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = els_b });
    const tv = try func.appendInst(then_b, t, .{ .iconst = then_val });
    func.setTerminator(then_b, .{ .ret = tv });
    const ev = try func.appendInst(els_b, t, .{ .iconst = els_val });
    func.setTerminator(els_b, .{ .ret = ev });
    return func;
}

test "aarch64 arith_branch: decrement-and-branch (n-1; if n!=0) folds to subs+b.ne and computes correctly" {
    const allocator = std.testing.allocator;
    var func = try buildArithBranchImm(allocator, 32, .sub, 1, .ne, 100, 200); // (n-1) != 0 ? 100 : 200
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    // Folded: the immediate S-form sets Z = (n-1 == 0) and `b.ne` branches on it, with NO
    // separate `cmp` (the plain compare-and-branch would emit `cmp` before the `b.ne`).
    try std.testing.expect(std.mem.indexOf(u8, text, "subs w") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.ne ") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") == null);

    try expectRun(allocator, &func, &.{1}, 200); // 1-1 == 0 -> else
    try expectRun(allocator, &func, &.{5}, 100); // 5-1 != 0 -> then
    try expectRun(allocator, &func, &.{0}, 100); // 0-1 == -1 != 0 -> then
    try expectRun(allocator, &func, &.{-3}, 100);
}

test "aarch64 arith_branch: 32-bit decrement fold is execution-equivalent to the flag-off compile" {
    const allocator = std.testing.allocator;
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // Same pattern as the fold test above, compiled BOTH ways (S-form fold vs plain arith + a
    // separate `cmp`) and JIT-run side by side: the two must agree on every input even though
    // their machine code differs, i.e. the fold changes instruction selection, never behavior.
    var func = try buildArithBranchImm(allocator, 32, .sub, 1, .ne, 100, 200); // (n-1) != 0 ? 100 : 200
    defer func.deinit();

    const on = try compileWithCaps(allocator, &func, .{ .fuse_arith_branch = true });
    defer allocator.free(on.relocs);
    defer allocator.free(on.lines);
    defer allocator.free(on.code);
    const off = try compileWithCaps(allocator, &func, .{ .fuse_arith_branch = false });
    defer allocator.free(off.relocs);
    defer allocator.free(off.lines);
    defer allocator.free(off.code);

    var buf_on = try jit.CodeBuffer.map(std.mem.sliceAsBytes(on.code));
    defer buf_on.deinit();
    var buf_off = try jit.CodeBuffer.map(std.mem.sliceAsBytes(off.code));
    defer buf_off.deinit();

    for ([_]i32{ 1, 5, 0, -3, 100 }) |n| {
        const got_on = try callI32(&buf_on, &.{n});
        const got_off = try callI32(&buf_off, &.{n});
        try std.testing.expectEqual(got_off, got_on);
    }
}

test "aarch64 arith_branch: add-and-branch (a+b; if ==0) folds to adds+b.eq" {
    const allocator = std.testing.allocator;
    var func = try buildArithBranchReg(allocator, 32, .add, .eq, 100, 200); // (a+b) == 0 ? 100 : 200
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "adds w") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.eq ") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") == null);

    try expectRun(allocator, &func, &.{ 3, -3 }, 100); // sum 0 -> then
    try expectRun(allocator, &func, &.{ 1, 1 }, 200); // sum 2 -> else
    try expectRun(allocator, &func, &.{ 0, 0 }, 100);
}

test "aarch64 arith_branch: and-and-branch (a&b; if !=0) folds to ands+b.ne" {
    const allocator = std.testing.allocator;
    var func = try buildArithBranchReg(allocator, 32, .bit_and, .ne, 100, 200); // (a&b) != 0 ? 100 : 200
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "ands w") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.ne ") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") == null);

    try expectRun(allocator, &func, &.{ 6, 3 }, 100); // 6&3 == 2 != 0 -> then
    try expectRun(allocator, &func, &.{ 4, 1 }, 200); // 4&1 == 0 -> else
    try expectRun(allocator, &func, &.{ 7, 7 }, 100);
}

test "aarch64 arith_branch off (flag false) emits arith+cmp+branch, no S-form for the branch" {
    const allocator = std.testing.allocator;
    var func = try buildArithBranchImm(allocator, 32, .sub, 1, .ne, 100, 200);
    defer func.deinit();

    const compiled = try compileWithCaps(allocator, &func, .{ .fuse_arith_branch = false });
    defer allocator.free(compiled.relocs);
    defer allocator.free(compiled.lines);
    defer allocator.free(compiled.code);

    const text = try disasm.format(allocator, compiled.code);
    defer allocator.free(text);
    // The gate declined the arith-branch fold: the decrement materializes with a plain `sub`
    // (no flag-setting `subs` Rd), and the compare-and-branch fold still fires with a separate
    // `cmp` before the `b.ne`. So NO S-form is used for the branch.
    try std.testing.expect(std.mem.indexOf(u8, text, "subs") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sub w") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.ne ") != null);

    if (builtin.cpu.arch == .aarch64) {
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer buf.deinit();
        try std.testing.expectEqual(@as(i32, 200), try callI32(&buf, &.{1}));
        try std.testing.expectEqual(@as(i32, 100), try callI32(&buf, &.{5}));
        try std.testing.expectEqual(@as(i32, 100), try callI32(&buf, &.{0}));
    }
}

test "aarch64 arith_branch: arith result used twice is NOT folded (plain add + cmp), correct result" {
    const allocator = std.testing.allocator;
    // `s = a + b` feeds the icmp AND is returned on both edges, so it has more than one use and
    // the single-use gate declines: `s` materializes plainly and the compare-and-branch keeps a
    // separate `cmp`. The result is `a + b` regardless of which edge is taken.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{s} }, .{ .target = els_b, .args = &.{s} });
    const xt = try func.appendBlockParam(then_b, t);
    func.setTerminator(then_b, .{ .ret = xt });
    const xe = try func.appendBlockParam(els_b, t);
    func.setTerminator(els_b, .{ .ret = xe });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "adds w") == null); // not folded
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") != null); // separate compare kept

    try expectRun(allocator, &func, &.{ 3, 4 }, 7);
    try expectRun(allocator, &func, &.{ 3, -3 }, 0);
}

test "aarch64 arith_branch: lt comparison is NOT folded (needs N/V), plain cmp;b.lt, correct result" {
    const allocator = std.testing.allocator;
    var func = try buildArithBranchReg(allocator, 32, .sub, .lt, 100, 200); // (a-b) < 0 ? 100 : 200
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    // Only eq/ne fold (the S-form's Z flag is exactly result==0). `lt` stays on the plain
    // compare-and-branch path (`cmp; b.lt`), so no flag-setting `subs` Rd is used for the branch.
    try std.testing.expect(std.mem.indexOf(u8, text, "subs") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b.lt ") != null);

    try expectRun(allocator, &func, &.{ 1, 5 }, 100); // 1-5 = -4 < 0 -> then
    try expectRun(allocator, &func, &.{ 5, 1 }, 200); // 5-1 = 4 -> else
    try expectRun(allocator, &func, &.{ 4, 4 }, 200); // 0 -> else
}

test "aarch64 arith_branch: an instruction between the arith and the icmp blocks the fold, correct result" {
    const allocator = std.testing.allocator;
    // `mid = a - b` sits between `s = a + b` and the icmp, so the arith is no longer at if_idx-2:
    // the adjacency gate declines and `s` materializes plainly. `s` is still single-use, `mid` is
    // returned on both edges, so the result is `a - b`.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const mid = try func.appendInst(entry, t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } }); // intervening
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{mid} }, .{ .target = els_b, .args = &.{mid} });
    const xt = try func.appendBlockParam(then_b, t);
    func.setTerminator(then_b, .{ .ret = xt });
    const xe = try func.appendBlockParam(els_b, t);
    func.setTerminator(els_b, .{ .ret = xe });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "adds w") == null); // not folded (non-adjacent)
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") != null);

    try expectRun(allocator, &func, &.{ 5, 3 }, 2); // returns a - b
    try expectRun(allocator, &func, &.{ 2, 10 }, -8);
}

test "aarch64 arith_branch: arith result live past the branch stays correct (not folded)" {
    const allocator = std.testing.allocator;
    // `s = a + b` is compared against 0 AND consumed in the then-block (`s + 1`), so it is live
    // past the branch (multi-use). The fold declines, but the value must remain correct either
    // way. Returns `a+b+1` when `a+b != 0`, else 0.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ne, .lhs = s, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{s} }, .{ .target = els_b });
    const xt = try func.appendBlockParam(then_b, t);
    const r = try func.appendInst(then_b, t, .{ .arith_imm = .{ .op = .add, .lhs = xt, .imm = 1 } });
    func.setTerminator(then_b, .{ .ret = r });
    const ev = try func.appendInst(els_b, t, .{ .iconst = 0 });
    func.setTerminator(els_b, .{ .ret = ev });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "adds w") == null); // multi-use -> not folded

    try expectRun(allocator, &func, &.{ 3, 4 }, 8); // 7 != 0 -> 7 + 1
    try expectRun(allocator, &func, &.{ 3, -3 }, 0); // 0 -> else
}

test "aarch64 arith_branch: a function without the pattern is byte-identical flag-on vs flag-off" {
    const allocator = std.testing.allocator;
    // A compare-and-branch whose icmp compares two params (no arith feeding it) is never an
    // arith-branch, so both flag settings must produce byte-identical code.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = els_b });
    func.setTerminator(then_b, .{ .ret = a });
    func.setTerminator(els_b, .{ .ret = b });

    const on = try compileWithCaps(allocator, &func, .{ .fuse_arith_branch = true });
    defer allocator.free(on.relocs);
    defer allocator.free(on.lines);
    defer allocator.free(on.code);
    const off = try compileWithCaps(allocator, &func, .{ .fuse_arith_branch = false });
    defer allocator.free(off.relocs);
    defer allocator.free(off.lines);
    defer allocator.free(off.code);
    try std.testing.expectEqualSlices(u32, on.code, off.code);
}

test "aarch64 arith_branch: 64-bit decrement is NOT folded (encode.cmp is 32-bit only), computes correctly" {
    const allocator = std.testing.allocator;
    // The surrounding icmp lowering (both the unfused icmp path and the fused compare-and-branch
    // path this fold lives inside) always compares with `encode.cmp`, which is 32-bit only. A
    // 64-bit S-form here would set Z from all 64 bits, disagreeing with that 32-bit `cmp` on
    // values whose low 32 bits are zero but high bits are not (e.g. 0x1_0000_0000: equal to 0
    // under a 32-bit `cmp`, not-equal under a 64-bit `subs`). `fusesArithIntoBranch` declines
    // whenever the arith result is wide (see its doc comment), so the decrement materializes with
    // a plain `sub x..` and the compare-and-branch fold keeps its separate `cmp`.
    var func = try buildArithBranchImm(allocator, 64, .sub, 1, .ne, 100, 200); // (n-1) != 0 ? 100 : 200
    defer func.deinit();

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "subs") == null); // no flag-setting S-form
    try std.testing.expect(std.mem.indexOf(u8, text, "sub x") != null); // plain 64-bit materialization
    try std.testing.expect(std.mem.indexOf(u8, text, "cmp") != null); // compare-and-branch keeps its cmp
    try std.testing.expect(std.mem.indexOf(u8, text, "b.ne ") != null);

    if (builtin.cpu.arch == .aarch64) {
        var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
        defer buf.deinit();
        const f: *const fn (i64) callconv(.c) i64 = @ptrCast(buf.memory.ptr);
        try std.testing.expectEqual(@as(i64, 200), f(1)); // 1-1 == 0 -> else
        try std.testing.expectEqual(@as(i64, 100), f(5)); // 5-1 != 0 -> then
        try std.testing.expectEqual(@as(i64, 100), f(0)); // 0-1 == -1 != 0 -> then
        try std.testing.expectEqual(@as(i64, 100), f(-3));
    }
}

/// Count non-overlapping occurrences of `needle` in `haystack`. Used to assert how many branches of
/// a given shape a disassembled listing contains (an elided branch drops the count).
fn countSubstr(haystack: []const u8, needle: []const u8) usize {
    var count: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, i, needle)) |at| {
        count += 1;
        i = at + needle.len;
    }
    return count;
}

test "aarch64 fallthrough: unconditional jump to the next block elides the branch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
    const tail = try func.appendBlock();
    const r = try func.appendBlockParam(tail, t);
    func.setTerminator(entry, .{ .jump = .{ .target = tail, .args = try func.internValueList(&.{s}) } });
    func.setTerminator(tail, .{ .ret = r });

    // `tail` is the block emitted right after `entry`, so the jump falls through: no `b`.
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expectEqual(@as(usize, 0), countSubstr(text, "  b ."));

    // 2*a arrives through the block-param move that stays on the fall-through edge.
    for ([_]i32{ 0, 1, -1, 7, -13, 1000 }) |v| {
        try expectRun(allocator, &func, &.{v}, v *% 2);
    }
}

test "aarch64 fallthrough: conditional then-edge next elides b-then" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    const then_b = try func.appendBlock(); // emitted next, so the then-edge falls through
    const else_b = try func.appendBlock();
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .ret = one });
    const zero = try func.appendInst(else_b, t, .{ .iconst = 0 });
    func.setTerminator(else_b, .{ .ret = zero });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    // The fused `b.gt` (branch to the then-label) stays and the `b else` stays, but the trailing
    // `b then` is elided: exactly one unconditional branch remains.
    try std.testing.expect(std.mem.indexOf(u8, text, "b.gt ") != null);
    try std.testing.expectEqual(@as(usize, 1), countSubstr(text, "  b ."));

    const cases = [_][2]i32{ .{ 3, 1 }, .{ 1, 3 }, .{ 5, 5 }, .{ -2, -7 }, .{ -7, -2 } };
    for (cases) |cc| {
        try expectRun(allocator, &func, &.{ cc[0], cc[1] }, if (cc[0] > cc[1]) 1 else 0);
    }
}

test "aarch64 fallthrough: conditional else-edge next inverts and falls through" {
    const allocator = std.testing.allocator;

    // Fused path: the else-edge is next, so the fused `b.gt then` inverts to `b.le else` and the
    // else-moves fall through. Exactly one unconditional branch (`b then`) remains.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const a = try func.appendBlockParam(entry, t);
        const b = try func.appendBlockParam(entry, t);
        const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        const else_b = try func.appendBlock(); // emitted next, so the else-edge falls through
        const then_b = try func.appendBlock();
        try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
        const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
        func.setTerminator(then_b, .{ .ret = one });
        const zero = try func.appendInst(else_b, t, .{ .iconst = 0 });
        func.setTerminator(else_b, .{ .ret = zero });

        const code = try isel.selectFunction(allocator, &func);
        defer allocator.free(code);
        const text = try disasm.format(allocator, code);
        defer allocator.free(text);
        try std.testing.expect(std.mem.indexOf(u8, text, "b.le ") != null); // inverted condition
        try std.testing.expect(std.mem.indexOf(u8, text, "b.gt ") == null);
        try std.testing.expectEqual(@as(usize, 1), countSubstr(text, "  b ."));

        const cases = [_][2]i32{ .{ 3, 1 }, .{ 1, 3 }, .{ 5, 5 }, .{ -2, -7 }, .{ -7, -2 } };
        for (cases) |cc| {
            try expectRun(allocator, &func, &.{ cc[0], cc[1] }, if (cc[0] > cc[1]) 1 else 0);
        }
    }

    // Boolean (non-fused) path: a bare bool condition normally branches with `cbnz then`; inverting
    // for the else-fall-through emits `cbz else`. Verifies the cbz encoder on the inverted boolean path.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const flag = try func.appendBlockParam(entry, bool_t);
        const else_b = try func.appendBlock(); // emitted next, so the else-edge falls through
        const then_b = try func.appendBlock();
        try func.appendIf(entry, flag, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
        const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
        func.setTerminator(then_b, .{ .ret = one });
        const zero = try func.appendInst(else_b, t, .{ .iconst = 0 });
        func.setTerminator(else_b, .{ .ret = zero });

        const code = try isel.selectFunction(allocator, &func);
        defer allocator.free(code);
        const text = try disasm.format(allocator, code);
        defer allocator.free(text);
        try std.testing.expect(std.mem.indexOf(u8, text, "cbz ") != null); // inverted boolean branch
        try std.testing.expect(std.mem.indexOf(u8, text, "cbnz ") == null);

        for ([_]i32{ 0, 1 }) |flag_val| {
            try expectRun(allocator, &func, &.{flag_val}, if (flag_val != 0) 1 else 0);
        }
    }
}

test "aarch64 fallthrough: block-param moves on the fall-through edge stay correct" {
    const allocator = std.testing.allocator;

    // Unconditional fall-through carrying TWO block-param args (p = a+1, q = 2a). Result p+q = 3a+1.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const entry = try func.appendBlock();
        const a = try func.appendBlockParam(entry, t);
        const p = try func.appendInst(entry, t, .{ .arith_imm = .{ .op = .add, .lhs = a, .imm = 1 } });
        const q = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        const tail = try func.appendBlock();
        const pp = try func.appendBlockParam(tail, t);
        const qq = try func.appendBlockParam(tail, t);
        func.setTerminator(entry, .{ .jump = .{ .target = tail, .args = try func.internValueList(&.{ p, q }) } });
        const sum = try func.appendInst(tail, t, .{ .arith = .{ .op = .add, .lhs = pp, .rhs = qq } });
        func.setTerminator(tail, .{ .ret = sum });
        for ([_]i32{ 0, 1, -1, 4, -9, 250 }) |v| {
            try expectRun(allocator, &func, &.{v}, (3 *% v) +% 1);
        }
    }

    // Conditional THEN-edge fall-through carrying an arg: max via `if a>b then then_b(a) else else_b(b)`,
    // then_b emitted next. The then-edge's arg `a` moves into the then-param on the fall-through path.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const a = try func.appendBlockParam(entry, t);
        const b = try func.appendBlockParam(entry, t);
        const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        const then_b = try func.appendBlock(); // next -> then-edge falls through
        const tv = try func.appendBlockParam(then_b, t);
        const else_b = try func.appendBlock();
        const ev = try func.appendBlockParam(else_b, t);
        try func.appendIf(entry, c, .{ .target = then_b, .args = &.{a} }, .{ .target = else_b, .args = &.{b} });
        func.setTerminator(then_b, .{ .ret = tv });
        func.setTerminator(else_b, .{ .ret = ev });
        const cases = [_][2]i32{ .{ 3, 1 }, .{ 1, 3 }, .{ 5, 5 }, .{ -2, -7 }, .{ -7, -2 }, .{ 100, -100 } };
        for (cases) |cc| {
            try expectRun(allocator, &func, &.{ cc[0], cc[1] }, if (cc[0] > cc[1]) cc[0] else cc[1]);
        }
    }

    // Conditional ELSE-edge fall-through carrying an arg (the inverted layout): same max, but else_b
    // is emitted next. The else-edge's arg `b` moves into the else-param on the fall-through path.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const a = try func.appendBlockParam(entry, t);
        const b = try func.appendBlockParam(entry, t);
        const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        const else_b = try func.appendBlock(); // next -> else-edge falls through (inverted branch)
        const ev = try func.appendBlockParam(else_b, t);
        const then_b = try func.appendBlock();
        const tv = try func.appendBlockParam(then_b, t);
        try func.appendIf(entry, c, .{ .target = then_b, .args = &.{a} }, .{ .target = else_b, .args = &.{b} });
        func.setTerminator(then_b, .{ .ret = tv });
        func.setTerminator(else_b, .{ .ret = ev });
        const cases = [_][2]i32{ .{ 3, 1 }, .{ 1, 3 }, .{ 5, 5 }, .{ -2, -7 }, .{ -7, -2 }, .{ 100, -100 } };
        for (cases) |cc| {
            try expectRun(allocator, &func, &.{ cc[0], cc[1] }, if (cc[0] > cc[1]) cc[0] else cc[1]);
        }
    }
}

test "aarch64 fallthrough: a diamond and a loop compute correctly" {
    const allocator = std.testing.allocator;

    // Diamond: entry -> (then_b | else_b) -> merge(param) -> ret. else_b's jump to merge also falls
    // through (merge is emitted right after else_b), so both the then-edge elision and an
    // unconditional fall-through with a block-param arg are exercised in one function. Result = max.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const a = try func.appendBlockParam(entry, t);
        const b = try func.appendBlockParam(entry, t);
        const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        const then_b = try func.appendBlock();
        const else_b = try func.appendBlock();
        const merge = try func.appendBlock();
        const m = try func.appendBlockParam(merge, t);
        try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
        func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{a}) } });
        func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{b}) } });
        func.setTerminator(merge, .{ .ret = m });
        const cases = [_][2]i32{ .{ 3, 1 }, .{ 1, 3 }, .{ 5, 5 }, .{ -8, -2 }, .{ 42, 41 } };
        for (cases) |cc| {
            try expectRun(allocator, &func, &.{ cc[0], cc[1] }, if (cc[0] > cc[1]) cc[0] else cc[1]);
        }
    }

    // Counting loop: sum_{i=0}^{n-1} i. entry falls through to the header, the header's then-edge
    // (body) falls through, and body's back-edge to the header is a real backward `b`.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const n = try func.appendBlockParam(entry, t);
        const header = try func.appendBlock();
        const i = try func.appendBlockParam(header, t);
        const acc = try func.appendBlockParam(header, t);
        const body = try func.appendBlock();
        const exit = try func.appendBlock();
        const zero_i = try func.appendInst(entry, t, .{ .iconst = 0 });
        const zero_acc = try func.appendInst(entry, t, .{ .iconst = 0 });
        func.setTerminator(entry, .{ .jump = .{ .target = header, .args = try func.internValueList(&.{ zero_i, zero_acc }) } });
        const c = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
        try func.appendIf(header, c, .{ .target = body, .args = &.{} }, .{ .target = exit, .args = &.{} });
        const next_i = try func.appendInst(body, t, .{ .arith_imm = .{ .op = .add, .lhs = i, .imm = 1 } });
        const acc2 = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = i } });
        func.setTerminator(body, .{ .jump = .{ .target = header, .args = try func.internValueList(&.{ next_i, acc2 }) } });
        func.setTerminator(exit, .{ .ret = acc });
        for ([_]i32{ 0, 1, 2, 5, 10, 100 }) |nv| {
            var expected: i32 = 0;
            var k: i32 = 0;
            while (k < nv) : (k += 1) expected +%= k;
            try expectRun(allocator, &func, &.{nv}, expected);
        }
    }
}
