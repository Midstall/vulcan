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
        \\0000: mov esi, dword ptr [esp + 4]
        \\0007: mov edx, dword ptr [esp + 8]
        \\000e: mov ecx, esi
        \\0010: add ecx, edx
        \\0012: mov eax, ecx
        \\0014: ret
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
    try std.testing.expectEqualStrings(
        \\0000: mov esi, dword ptr [esp + 4]
        \\0007: mov edx, dword ptr [esp + 8]
        \\000e: cmp esi, edx
        \\0010: setg cl
        \\0013: movzx ecx, cl
        \\0016: test ecx, ecx
        \\0018: jne .+7
        \\001e: mov ecx, edx
        \\0020: jmp .+7
        \\0025: mov ecx, esi
        \\0027: jmp .+0
        \\002c: mov eax, ecx
        \\002e: ret
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
        \\0006: mov esi, dword ptr [esp + 20]
        \\000d: lea edx, [esp]
        \\0014: mov dword ptr [edx], esi
        \\001a: mov esi, dword ptr [edx]
        \\0020: mov eax, esi
        \\0022: add esp, 16
        \\0028: ret
        \\
    , text);
}

test "codegen+disasm round-trip: an 8-bit store stages through ebx (byte-addressability)" {
    // esi (the value's assigned register here) has no 8-bit form, so an i8 store must move
    // it into ebx (bl) before `mov byte ptr [...], bl`. Confirms the byte-addressability
    // staging described in Task 6, not just the [reg+0] store path in isolation.
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
        \\0006: mov esi, dword ptr [esp + 20]
        \\000d: lea edx, [esp]
        \\0014: mov ebx, esi
        \\0016: mov byte ptr [edx], bl
        \\001c: movsx esi, byte ptr [edx]
        \\0023: mov eax, esi
        \\0025: add esp, 16
        \\002b: ret
        \\
    , text);
}
