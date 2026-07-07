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
    // J-type immediate decoding on real codegen: a `bne .+16` plus forward and BACKWARD
    // `jal` offsets (.+20, .-12, .-20), the trickiest part of the RISC-V decoder.
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
        \\0000: 00a5a2b3  slt x5, x11, x10
        \\0004: 00029863  bnez x5, .+16
        \\0008: 0140006f  j .+20
        \\000c: 00028513  mv x10, x5
        \\0010: 00008067  ret
        \\0014: 00050293  mv x5, x10
        \\0018: ff5ff06f  j .-12
        \\001c: 00058293  mv x5, x11
        \\0020: fedff06f  j .-20
        \\
    , text);
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
