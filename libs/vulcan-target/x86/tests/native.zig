//! Native runner: execute the shared cases.zig in-process by mapping the compiled code
//! into W^X memory and calling it. Valid only when the host is i386. On any other host
//! every case skips, since i386 machine code cannot run natively there. On an i386 host
//! it also exercises the JIT (coherent_jit) path.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const link = @import("../link.zig");

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

test "i386 cases run natively in-process (skips off i386)" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.native);
}

test "codegen+disasm round-trip: integer add" {
    // Compile i386 and assert the disassembled listing (instruction selection + register
    // allocation), no execution so it runs on any host. cdecl args come off the stack.
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
    try std.testing.expectEqualStrings(
        \\0000: mov eax, dword ptr [esp + 4]
        \\0007: mov ecx, dword ptr [esp + 8]
        \\000e: mov edx, eax
        \\0010: add edx, ecx
        \\0012: mov eax, edx
        \\0014: ret
        \\
    , text);
}

test "codegen+disasm round-trip: control flow (max via if/else)" {
    // Exercises the disassembler on real setcc/movzx/test and jcc/jmp rel32 branches. The Wimmer
    // pipeline splits the two critical edges into m, so each arm is a short forwarding block that
    // jumps to the shared `ret`. r is homed in eax and x already lives in eax, so the then-arm needs
    // no move; the else-arm copies y (ecx) into eax. Both paths return the larger of x, y.
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
    try std.testing.expectEqualStrings(
        \\0000: mov eax, dword ptr [esp + 4]
        \\0007: mov ecx, dword ptr [esp + 8]
        \\000e: cmp eax, ecx
        \\0010: setg dl
        \\0013: movzx edx, dl
        \\0016: test edx, edx
        \\0018: jne .+5
        \\001e: jmp .+11
        \\0023: jmp .+1
        \\0028: ret
        \\0029: jmp .-6
        \\002e: mov eax, ecx
        \\0030: jmp .-13
        \\
    , text);
}

test "codegen+disasm round-trip: alloca/store/load at displacement 0 (no qemu needed)" {
    // Smoke-checks the new load/store lowering by disassembly, so it runs even where
    // qemu-i386 is unavailable: alloca an i32 slot, store the argument, load it back, return
    // it. Confirms the [reg+0] mov encoders (Task 5) are actually wired into `.alloca`/
    // `.load`/`.store` (Task 6), not just callable.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const slot = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(e, x, slot);
    const r = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(e, .{ .ret = r });

    const code = try isel.selectFunction(a, &func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\0000: sub esp, 16
        \\0006: mov eax, dword ptr [esp + 20]
        \\000d: lea ecx, [esp]
        \\0014: mov dword ptr [ecx], eax
        \\001a: mov eax, dword ptr [ecx]
        \\0020: add esp, 16
        \\0026: ret
        \\
    , text);
}

test "Wimmer path: add-two-args compiles and returns via eax" {
    // The test-only shared-Wimmer entry (`compileFunctionWimmerX86`) compiles a two-parameter
    // add. cdecl params come off the stack into the allocator-chosen homes. Structural only (a
    // full execution differential is Task 4): assert it compiles and ends in `ret`.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const s = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(e, .{ .ret = s });

    var compiled = try isel.compileFunctionWimmerX86(a, &func);
    defer compiled.deinit(a);
    try std.testing.expect(compiled.code.len > 0);
    try std.testing.expectEqual(@as(u8, 0xC3), compiled.code[compiled.code.len - 1]); // ret
}

test "Wimmer path: more than four live ints compiles with spilling" {
    // Five values (a plus v1..v4) are simultaneously live when v5 is defined, exceeding the
    // four-register pool {eax,ecx,edx,esi}, so the shared allocator must spill. Assert the
    // spilling path compiles cleanly through the test-only Wimmer entry and ends in `ret`.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const p = try func.appendBlockParam(e, i32_t);
    const v1 = try func.appendArithImm(e, i32_t, .add, p, 1);
    const v2 = try func.appendArithImm(e, i32_t, .add, p, 2);
    const v3 = try func.appendArithImm(e, i32_t, .add, p, 3);
    const v4 = try func.appendArithImm(e, i32_t, .add, p, 4);
    const v5 = try func.appendArithImm(e, i32_t, .add, p, 5);
    const s1 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = v1, .rhs = v2 } });
    const s2 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = v3 } });
    const s3 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = v4 } });
    const s4 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = s3, .rhs = v5 } });
    func.setTerminator(e, .{ .ret = s4 });

    var compiled = try isel.compileFunctionWimmerX86(a, &func);
    defer compiled.deinit(a);
    try std.testing.expect(compiled.code.len > 0);
    try std.testing.expectEqual(@as(u8, 0xC3), compiled.code[compiled.code.len - 1]); // ret
}

test "Wimmer path: an esi-homed boolean stages setcc through ebx (byte-addressability)" {
    // Force the icmp result to live in esi (which has NO low byte): a and b occupy eax/ecx as
    // the compare operands, m1 occupies edx, so the boolean lands in the only remaining
    // allocatable register, esi. A direct `setcc esi` would wrongly encode DH (the latent bug
    // the cutover fixes), so the emitter must stage the boolean through the byte-addressable ebx
    // (bl) and then move it to esi. Structural (disasm) proof; full execution is Task 4.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const m1 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    // The boolean (unused) is forced into esi by the surrounding register pressure.
    _ = try func.appendInst(e, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    const s1 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = m1, .rhs = x } });
    const s2 = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = y } });
    func.setTerminator(e, .{ .ret = s2 });

    var compiled = try isel.compileFunctionWimmerX86(a, &func);
    defer compiled.deinit(a);
    const text = try disasm.format(a, compiled.code);
    defer a.free(text);
    // The boolean is homed in esi, so the setcc must target the byte-addressable ebx (bl) and the
    // value is moved into esi afterward.
    try std.testing.expect(std.mem.indexOf(u8, text, "setl bl") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "mov esi, ebx") != null);
    // And it must NOT setcc a non-byte register: esi in an 8-bit slot encodes as DH, which is the
    // latent miscompile this staging fixes.
    try std.testing.expect(std.mem.indexOf(u8, text, "setl dh") == null);
}

test "Wimmer path: a div whose divisor is homed in eax is copied out before the clobber" {
    // Two params with no other pressure: x (param0) homes in eax, y (param1) in ecx (the pool's
    // first two slots, {eax,ecx,edx,esi}). `div(y, x)` puts x (in eax) in the DIVISOR position.
    // The naive lowering loads the dividend (y) into eax right on top of it, then reads "the
    // divisor" back out of eax, silently dividing y by itself instead of by x. The guard must
    // copy x out of eax into the reload scratch (edi) BEFORE eax is overwritten with the
    // dividend, and idiv must read that copy, not eax. Structural (disasm) proof; a full
    // execution differential is Task 4.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const d = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .div, .lhs = y, .rhs = x } });
    func.setTerminator(e, .{ .ret = d });

    var compiled = try isel.compileFunctionWimmerX86(a, &func);
    defer compiled.deinit(a);
    const text = try disasm.format(a, compiled.code);
    defer a.free(text);
    // The divisor (x, in eax) is saved into edi before the dividend clobbers eax.
    try std.testing.expect(std.mem.indexOf(u8, text, "mov edi, eax") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "mov eax, ecx") != null);
    const idx_save_divisor = std.mem.indexOf(u8, text, "mov edi, eax").?;
    const idx_load_dividend = std.mem.indexOf(u8, text, "mov eax, ecx").?;
    try std.testing.expect(idx_save_divisor < idx_load_dividend);
    try std.testing.expect(std.mem.indexOf(u8, text, "idiv edi") != null);
    // The miscompile signature: dividing by whatever now sits in eax (the dividend, post-clobber)
    // instead of the real divisor.
    try std.testing.expect(std.mem.indexOf(u8, text, "idiv eax") == null);
}

test "Wimmer path: a shift whose lhs is homed in ecx is copied out before the clobber" {
    // x (param0) homes in eax, y (param1) in ecx. `shl(y, x)` puts y (in ecx) in the SHIFT-LHS
    // position and x (in eax) as the shift count. The naive lowering moves the count into ecx
    // right on top of the lhs, then shifts using the (stale) register name that now holds the
    // count, i.e. shifts the count by itself. The guard must copy y out of ecx into the byte
    // scratch (ebx) BEFORE ecx is overwritten with the count, and the shift must read that copy.
    // Structural (disasm) proof; a full execution differential is Task 4.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const d = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .shl, .lhs = y, .rhs = x } });
    func.setTerminator(e, .{ .ret = d });

    var compiled = try isel.compileFunctionWimmerX86(a, &func);
    defer compiled.deinit(a);
    const text = try disasm.format(a, compiled.code);
    defer a.free(text);
    // The lhs (y, in ecx) is saved into ebx before the shift count clobbers ecx.
    try std.testing.expect(std.mem.indexOf(u8, text, "mov ebx, ecx") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "mov ecx, eax") != null);
    const idx_save_lhs = std.mem.indexOf(u8, text, "mov ebx, ecx").?;
    const idx_load_count = std.mem.indexOf(u8, text, "mov ecx, eax").?;
    try std.testing.expect(idx_save_lhs < idx_load_count);
    try std.testing.expect(std.mem.indexOf(u8, text, "shl edx, cl") != null);
    // The miscompile signature: shifting the value the (now-corrupted) ecx-homed register name
    // reads AFTER the count overwrote it, instead of the preserved lhs.
    try std.testing.expect(std.mem.indexOf(u8, text, "mov edx, ecx") == null);
}

test "codegen+disasm round-trip: an i8 store stages through ebx (byte-addressability)" {
    // The 8-bit store arm UNCONDITIONALLY stages its value through ebx (bl) before
    // `mov byte ptr [...], bl`, rather than special-casing whether the source register happens to
    // be byte-addressable (esi/edi have no 8-bit form, so a source homed there would otherwise
    // misencode). Here the value is homed in eax, and the store still routes through ebx. Confirms
    // the byte-addressability staging, not just the [reg+0] store path in isolation.
    const a = std.testing.allocator;
    var func = ir.function.Function.init(a);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i8_t);
    const slot = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i8_t } });
    try func.appendStore(e, x, slot);
    const r = try func.appendInst(e, i8_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(e, .{ .ret = r });

    const code = try isel.selectFunction(a, &func);
    defer a.free(code);
    const text = try disasm.format(a, code);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\0000: sub esp, 16
        \\0006: mov eax, dword ptr [esp + 20]
        \\000d: lea ecx, [esp]
        \\0014: mov ebx, eax
        \\0016: mov byte ptr [ecx], bl
        \\001c: movsx eax, byte ptr [ecx]
        \\0023: add esp, 16
        \\0029: ret
        \\
    , text);
}

// --- Task 3: address fold under register pressure through the fold-aware Wimmer pipeline -------------
//
// The shared allocator is fold-BLIND: it reads only the raw IR operands. A foldable
// `p = arith_imm.add(base, imm); load/store(p)` leaves `p` (the add's result) DEAD once the mem op
// folds to `disp(base)`, but a fold-blind Wimmer compile would let `base` die at the add and reuse its
// register across the pressured region, then emit `disp(base)` reading the STALE register (the aarch64
// SP1 trap). `applyFoldRewriteX86` repoints each folded mem op's pointer to `base` and DCEs the dead
// adds BEFORE `wimmer.allocate`, so the allocator keeps `base` live to the mem op while the SAME
// analysis still drives the `disp(base)` emission. This test builds the fold-under-pressure shape,
// compiles it through the fold-aware Wimmer pipeline, asserts the fold actually FIRED in the emitted
// bytes, and (under qemu-i386 or an i386 host) asserts the result equals a hand-computed GROUND TRUTH.
// A fold-blind allocation would read a stale base and diverge from that ground truth.

const fold_pressure = 8;

/// f(arg, cond): in ENTRY, `buf1 = arg` and two DEAD adds off buf0 (`pl = buf0 + 4` = buf1's address,
/// `ps = buf0 + 8` = buf2's address), each used ONLY by a successor mem op, so both fold. On cond > 0,
/// `then_b` first builds `fold_pressure` live temporaries (more than the four-register pool, so a
/// fold-blind allocator would have handed buf0's register to a temp by the time the folded accesses
/// run), then the folded store to `ps` and the folded load from `pl`. buf0 must stay live across the
/// whole pressured region for both folded accesses to read the right slot. Allocas are i32 (4 bytes),
/// laid out back-to-back, so `buf0 + 4` = buf1 and `buf0 + 8` = buf2. Expected(cond>0) = arg + cond +
/// sum_{k=1..P}(cond + k); Expected(cond<=0) = cond.
fn buildFoldUnderPressure(allocator: std.mem.Allocator) !ir.function.Function {
    var func = ir.function.Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const arg = try func.appendBlockParam(entry, i32_t);
    const cond = try func.appendBlockParam(entry, i32_t);

    const buf0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const buf1 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const buf2 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, arg, buf1); // buf1 = arg (own pointer, off 0, references buf1)
    // Two DEAD adds off buf0: each result feeds ONLY a successor mem op, so buf0's successor liveness
    // flows solely through the folded accesses' base.
    const pl = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 4 } }); // = buf1
    const ps = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 8 } }); // = buf2
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = cond, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });

    // then_b: build the pressure temporaries FIRST, then the folded store/load, so a fold-blind
    // allocator would already have handed buf0's register to a temp by the time the folded accesses run.
    var vals: [fold_pressure]ir.function.Value = undefined;
    for (0..fold_pressure) |k| {
        vals[k] = try func.appendInst(then_b, i32_t, .{ .arith_imm = .{ .op = .add, .lhs = cond, .imm = @intCast(k + 1) } });
    }
    try func.appendStore(then_b, cond, ps); // folded store to [buf0 + 8] = buf2
    const w = try func.appendInst(then_b, i32_t, .{ .load = .{ .ptr = pl } }); // folded load [buf0 + 4] = buf1 = arg
    const rb = try func.appendInst(then_b, i32_t, .{ .load = .{ .ptr = buf2 } }); // read buf2 back (own ptr) = cond
    var acc = try func.appendInst(then_b, i32_t, .{ .arith = .{ .op = .add, .lhs = w, .rhs = rb } }); // arg + cond
    for (0..fold_pressure) |k| acc = try func.appendInst(then_b, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[k] } });
    func.setTerminator(then_b, .{ .ret = acc });
    func.setTerminator(else_b, .{ .ret = cond });
    return func;
}

/// Count memory accesses in already-compiled `code` that carry a NONZERO displacement off a base
/// register other than esp: a folded `[base + off]` (off != 0, base != esp). Spill, alloca, and frame
/// accesses address off esp, so requiring base != esp isolates the folds. Zero unless address folding
/// fired. Mirrors the `foldedMemOps` disasm probe in `tests/addrfold.zig`, but works on already-compiled
/// bytes so it can inspect the Wimmer-fold output directly.
fn foldedMemOpsInCode(allocator: std.mem.Allocator, code: []const u8) !usize {
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const lb = std.mem.indexOfScalar(u8, line, '[') orelse continue;
        const rb = std.mem.indexOfScalarPos(u8, line, lb, ']') orelse continue;
        const inner = line[lb + 1 .. rb];
        const base_end = std.mem.indexOfScalar(u8, inner, ' ') orelse inner.len;
        if (std.mem.eql(u8, inner[0..base_end], "esp")) continue; // spill / alloca / frame, not a fold
        if (std.mem.indexOf(u8, inner, " + ") != null or std.mem.indexOf(u8, inner, " - ") != null) count += 1;
    }
    return count;
}

test "Wimmer path: address fold under register pressure fires and computes the ground truth" {
    const a = std.testing.allocator;
    const io = std.testing.io;

    // The fold MUST fire in the fold-aware Wimmer output: at least the folded load and the folded store
    // survive as `[base + disp]` off a non-esp register. A zero here means the rewrite elided the fold or
    // the pipeline stayed fold-blind.
    var probe = try buildFoldUnderPressure(a);
    defer probe.deinit();
    var probe_c = try isel.compileFunctionWimmerX86Fold(a, &probe);
    defer probe_c.deinit(a);
    try std.testing.expect(try foldedMemOpsInCode(a, probe_c.code) >= 2);

    const inputs = [_][]const i64{ &.{ 21, 1 }, &.{ 20, 0 }, &.{ 25, 5 }, &.{ 50, 30 }, &.{ 32, 12 }, &.{ 7, -4 } };
    for (inputs) |args| {
        // Ground truth: on cond > 0 the folded load reads arg and the folded store/read-back gives cond,
        // then the P=fold_pressure temporaries (cond+k, k=1..P) are summed in: arg + cond +
        // sum_{k=1..P}(cond + k). On cond <= 0 the else arm returns cond. i32 wrapping, read mod 256.
        const arg: i32 = @intCast(args[0]);
        const cond: i32 = @intCast(args[1]);
        var exp: i32 = cond;
        if (cond > 0) {
            exp = arg +% cond;
            for (0..fold_pressure) |k| exp +%= cond +% @as(i32, @intCast(k + 1));
        }
        const want: u8 = @truncate(@as(u32, @bitCast(exp)));

        var wim_func = try buildFoldUnderPressure(a);
        defer wim_func.deinit();
        var wim_c = try isel.compileFunctionWimmerX86Fold(a, &wim_func);
        defer wim_c.deinit(a);

        // Execute the fold-aware Wimmer bytes on whichever i386 backend is available (qemu-i386, else a
        // native i386 host); a fold-blind allocation reads a stale base and diverges from `want`.
        var ran = false;
        for ([_]harness.Backend{ harness.qemu, harness.native }) |backend| {
            const got = harness.runCodeInt(io, a, wim_c.code, args, backend) catch |e| switch (e) {
                error.SkipZigTest => continue,
                else => return e,
            };
            try std.testing.expectEqual(want, got);
            ran = true;
        }
        if (!ran) return error.SkipZigTest;
    }
}
