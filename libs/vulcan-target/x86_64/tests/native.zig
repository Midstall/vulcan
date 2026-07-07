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
    try std.testing.expectEqualStrings(
        \\0000: mov r9, rdi
        \\0003: mov r8, rsi
        \\0006: mov rdi, r9
        \\0009: add rdi, r8
        \\000c: mov rax, rdi
        \\000f: ret
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
        \\0000: mov r9, rdi
        \\0003: mov r8, rsi
        \\0006: cmp r9, r8
        \\0009: setg dil
        \\000d: movzx rdi, dil
        \\0011: test rdi, rdi
        \\0014: jne .+8
        \\001a: mov rdi, r8
        \\001d: jmp .+8
        \\0022: mov rdi, r9
        \\0025: jmp .+0
        \\002a: mov rax, rdi
        \\002d: ret
        \\
    , text);
}
