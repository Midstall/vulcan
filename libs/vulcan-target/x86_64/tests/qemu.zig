//! qemu-x86_64 runner: execute the shared cases.zig under qemu-x86_64 user mode. The
//! harness wraps each function in a static Linux ELF and QEMU runs it. Skips when
//! qemu-x86_64 is not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const cases = @import("cases.zig");
const harness = @import("harness.zig");
const link = @import("../link.zig");
const isel = @import("../isel.zig");

const Function = ir.function.Function;

test "x86-64 cases run under qemu-x86_64" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu);
}

// A binary128 operation has no SSE form; the soft-fp pass lowers it to a libgcc call before
// isel, so the backend must emit a call relocation against the undefined soft-fp symbol. The
// in-memory JIT linker leaves that symbol unresolved (error.UndefinedSymbol); the object path
// writes it as an undefined symbol the final system link resolves from libgcc/compiler-rt.
test "an f128 add compiles to an undefined __addtf3 soft-fp call relocation" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var compiled = try isel.compile(allocator, &f);
    defer compiled.deinit(allocator);

    var addtf3: usize = 0;
    for (compiled.relocs) |rel| {
        if (std.mem.eql(u8, rel.symbol, "__addtf3")) {
            addtf3 += 1;
            try std.testing.expectEqual(isel.Kind.call, rel.kind);
        }
    }
    try std.testing.expectEqual(@as(usize, 1), addtf3);
}

// An f128 `select` is a conditional data move, not an arithmetic libcall, so the softfp pass
// leaves it in place. x86-64 lowers select as a two-armed branch whose arms move the whole
// value with 128-bit movups (see `selectInto`), so an f128 select compiles natively (no
// libcall, no error). This is the x86-64 counterpart of aarch64's fail-closed f128 select,
// whose `fcsel` has no 128-bit form.
test "an f128 select compiles natively on x86-64 with no soft-fp call" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const i1_t = try f.types.intern(.bool);
    const b = try f.appendBlock();
    const cond = try f.appendBlockParam(b, i1_t);
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, t, .{ .select = .{ .cond = cond, .then = x, .@"else" = y } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var compiled = try isel.compile(allocator, &f);
    defer compiled.deinit(allocator);
    try std.testing.expect(compiled.code.len > 0);
    for (compiled.relocs) |rel| try std.testing.expect(!std.mem.startsWith(u8, rel.symbol, "__")); // no soft-fp call
}

test "an unreachable block that uses a reachable value compiles and the reachable path runs" {
    // The exact shape that tripped the shared allocator's SSA def-in-range assert before
    // `neutralizeUnreachable` was adopted: a value DEFINED in the reachable entry is USED by a block
    // NO reachable block branches to. The production `compile`/`selectFunction` must neutralize the
    // orphan block, tolerate its emptied (no-instruction, null-terminator) form in emission, and still
    // return the reachable sum. Skips when qemu-x86_64 is not on PATH.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    // entry: s = x + y ; ret s.
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i64_t);
    const y = try func.appendBlockParam(entry, i64_t);
    const s = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(s) });

    // The unreachable block: it USES `s` (a reachable value) yet nothing branches to it.
    const dead = try func.appendBlock();
    const d = try func.appendInst(dead, i64_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = s } });
    func.setTerminator(dead, .{ .ret = ir.function.Ret.one(d) });

    // Compiles without crashing AND the reachable path executes correctly (5 + 3 == 8).
    try harness.expectRunFull(std.testing.io, a, &func, &.{ 5, 3 }, 8, harness.qemu);
}

test "an indirect call with 7 integer args passes the 7th on the stack (qemu-x86_64)" {
    // The bug: the x86-64 indirect-call path (`call_indirect`) failed closed on any
    // seventh integer argument, since System V has only 6 integer arg registers and the
    // path had no stack-argument code. A wasm `call_indirect` prepends a hidden context
    // pointer, so six wasm params become seven machine args, and this was reached in the
    // wild on x86-64 while aarch64 (eight arg registers) never hit it. The direct-call
    // path already homed stack args; this proves the indirect path now does too.
    //
    // main() computes the address of callee and calls it indirectly with 7 args
    // {1,2,3,4,5,6,100}. callee returns a1 - a2 + a3 - a4 + a5 - a6 + a7, an
    // order-sensitive sum: dropping or misplacing the stack arg (100) or any register
    // arg changes the result. Expected: 1 - 2 + 3 - 4 + 5 - 6 + 100 = 97.
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i64k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 64 } };

    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i64k);
        const b = try callee.appendBlock();
        var a: [7]ir.function.Value = undefined;
        for (&a) |*p| p.* = try callee.appendBlockParam(b, t);
        const s1 = try callee.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = a[0], .rhs = a[1] } });
        const s2 = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = a[2] } });
        const s3 = try callee.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = s2, .rhs = a[3] } });
        const s4 = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s3, .rhs = a[4] } });
        const s5 = try callee.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = s4, .rhs = a[5] } });
        const s6 = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s5, .rhs = a[6] } });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(s6) });
    }

    var main_f = Function.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i64k);
        const ptr_t = try main_f.types.intern(.ptr);
        const b = try main_f.appendBlock();
        const vals = [_]i64{ 1, 2, 3, 4, 5, 6, 100 };
        var args: [7]ir.function.Value = undefined;
        for (&args, vals) |*arg, v| arg.* = try main_f.appendInst(b, t, .{ .iconst = v });
        const tgt = try main_f.appendGlobalAddr(b, ptr_t, "callee");
        const r = try main_f.appendCallIndirect(b, t, tgt, &args);
        main_f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main_f);
    try module.addFunction(allocator, "callee", &callee);

    try std.testing.expectEqual(@as(u8, 97), try harness.runModuleData(io, allocator, &module, &.{}, harness.qemu));
}

test "a rodata global read via global_addr returns its value under qemu-x86_64" {
    // main() -> *(&K), K a rodata i32 constant. Proves isel's `global_addr` arm
    // (`lea rd, [rip+disp32]`) end to end: a real `.pcrel_lea` reloc, resolved (here,
    // statically by `runModuleData`) to K's address, executed on real x86-64 silicon
    // under qemu-x86_64.
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var main_f = Function.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i32k);
        const ptr_t = try main_f.types.intern(.ptr);
        const b = try main_f.appendBlock();
        const g = try main_f.appendGlobalAddr(b, ptr_t, "K");
        const v = try main_f.appendInst(b, t, .{ .load = .{ .ptr = g } });
        main_f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    const k_bytes = [_]u8{ 42, 0, 0, 0 }; // i32 42, little-endian
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main_f);
    try module.addData(allocator, "K", &k_bytes);

    try std.testing.expectEqual(@as(u8, 42), try harness.runModuleData(io, allocator, &module, &.{}, harness.qemu));
}
