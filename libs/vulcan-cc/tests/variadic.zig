//! This file tests the frontend foundation for variadic (`...`) support. It covers the
//! lexer and parser plumbing, `ctype.FuncType.is_variadic`, `__builtin_va_list`, the built-in
//! `<stdarg.h>`, the IR `Call`/`CallIndirect` `is_variadic`/`num_fixed` fields, and default
//! argument promotion. No backend reads any of this data yet. A later stage adds that. This
//! file proves the plumbing works end to end for a fixed set of examples. It also proves the
//! plumbing changes NOTHING for a fixed-arity program (see the byte-identity note at the
//! bottom of this file).

const std = @import("std");
const builtin = @import("builtin");
const cc = @import("vulcan-cc");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const ld = @import("vulcan-link");

// Step 1: lexing `"int f(int, ...);"` produces ONE `.ellipsis` token for the three dots.
// It does not produce three separate `.dot` tokens.
test "lexing '...' yields a single .ellipsis token" {
    const allocator = std.testing.allocator;
    const toks = try cc.lexer.tokenize(allocator, "int f(int, ...);");
    defer cc.lexer.freeTokens(allocator, toks);
    const expected = [_]cc.lexer.Kind{
        .kw_int, .ident, .lparen, .kw_int, .comma, .ellipsis, .rparen, .semicolon, .eof,
    };
    try std.testing.expectEqual(expected.len, toks.len);
    for (expected, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

// Step 2: `preproc.preprocess` scans `...` the same way. This is the full pipeline that
// `parser.parse` runs, not just `lexer.tokenize`. `preproc.zig` has its own scanner, and it
// reuses `lexer.matchOperator` directly, so this behavior should already work. This test
// confirms it.
test "preproc scans '...' as a single .ellipsis PToken" {
    const allocator = std.testing.allocator;
    const toks = try cc.preproc.preprocess(allocator, "int f(int, ...);", .{});
    defer cc.lexer.freeTokens(allocator, toks);
    try std.testing.expectEqual(cc.lexer.Kind.ellipsis, toks[5].kind);
}

// Step 3: this test exercises `parseParams`'s `...` handling, through the public `parser.parse`.
test "a trailing '...' after a named param marks a FuncDecl variadic" {
    const allocator = std.testing.allocator;
    var unit = try cc.parser.parse(allocator, "int f(int a, ...);", cc.layout.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 1), unit.func_decls.len);
    try std.testing.expectEqual(@as(usize, 1), unit.func_decls[0].params.len);
    try std.testing.expect(unit.func_decls[0].is_variadic);
}

test "no trailing '...' leaves a FuncDecl non-variadic" {
    const allocator = std.testing.allocator;
    var unit = try cc.parser.parse(allocator, "int g(int a);", cc.layout.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 1), unit.func_decls.len);
    try std.testing.expect(!unit.func_decls[0].is_variadic);
}

test "'...' with no named parameter first is rejected" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.parser.parse(allocator, "int h(...);", cc.layout.host(), .{}));
}

test "'...' must be the last parameter" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.parser.parse(allocator, "int j(int a, ..., int b);", cc.layout.host(), .{}));
}

// A DEFINITION (not just a bodyless prototype) marks `Func.is_variadic` the same way.
test "a variadic function DEFINITION marks Func.is_variadic" {
    const allocator = std.testing.allocator;
    var unit = try cc.parser.parse(allocator, "int f(int a, ...){ return a; }", cc.layout.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 1), unit.funcs.len);
    try std.testing.expectEqual(@as(usize, 1), unit.funcs[0].params.len);
    try std.testing.expect(unit.funcs[0].is_variadic);
}

// Step 6: `stdarg.zig`'s bytes resolve for `#include <stdarg.h>`. Its `typedef` line
// (naming `__builtin_va_list`) survives preprocessing into the final token stream. `va_list`
// itself is NOT a preprocessor-level rewrite. It is an ordinary `typedef` name, a PARSER
// concept, and the preprocessor has no notion of typedefs. This test checks only the header's
// own tokens. A later stage makes `parser.parse` recognize `__builtin_va_list`/`va_list` as a
// usable type.
fn stdargOnlyResolve(ctx: ?*anyopaque, name: []const u8, is_system: bool, includer_dir: ?[]const u8, next_after: ?[]const u8) cc.preproc.Error!?cc.preproc.ResolvedFile {
    _ = ctx;
    _ = includer_dir;
    _ = next_after;
    if (cc.stdarg.resolve(name, is_system)) |b| return .{ .identity = "stdarg.h", .bytes = b };
    return null;
}

test "#include <stdarg.h> resolves to stdarg.zig's bytes" {
    const allocator = std.testing.allocator;
    const resolver: cc.preproc.IncludeResolver = .{ .resolveFn = stdargOnlyResolve };
    const toks = try cc.preproc.preprocess(allocator, "#include <stdarg.h>\nva_list x;", .{ .resolver = resolver });
    defer cc.lexer.freeTokens(allocator, toks);
    // `stdarg.h` follows gcc's own header protocol (see `stdarg.zig`). A PLAIN
    // (non-`__need___va_list`) include emits TWO typedef lines: `typedef __builtin_va_list
    // __gnuc_va_list;` then `typedef __gnuc_va_list va_list;`, ahead of the user's own line. The
    // `#ifndef`/`#define`/`#endif` guards and the four `va_*` `#define`s are directives. The
    // preprocessor consumes them into the macro table and never emits them as tokens.
    try std.testing.expectEqual(cc.lexer.Kind.kw_typedef, toks[0].kind);
    try std.testing.expectEqualStrings("__builtin_va_list", toks[1].text);
    try std.testing.expectEqualStrings("__gnuc_va_list", toks[2].text);
    try std.testing.expectEqual(cc.lexer.Kind.semicolon, toks[3].kind);
    try std.testing.expectEqual(cc.lexer.Kind.kw_typedef, toks[4].kind);
    try std.testing.expectEqualStrings("__gnuc_va_list", toks[5].text);
    try std.testing.expectEqualStrings("va_list", toks[6].text);
    try std.testing.expectEqual(cc.lexer.Kind.semicolon, toks[7].kind);
    // The including source's own line, unexpanded.
    try std.testing.expectEqualStrings("va_list", toks[8].text);
    try std.testing.expectEqualStrings("x", toks[9].text);
    try std.testing.expectEqual(cc.lexer.Kind.semicolon, toks[10].kind);
    try std.testing.expectEqual(cc.lexer.Kind.eof, toks[11].kind);
}

// Step 8: this test compiles a real variadic call end to end, then inspects the lowered IR.
// `printf`'s prototype has ONE fixed param (`fmt`). The call passes a `char` as the second
// (variadic) argument. `defaultArgPromote` must widen it to `int` (C's default argument
// promotion), since there is no declared parameter type to convert against past the fixed
// ones. The `call` op must carry `is_variadic = true` and `num_fixed = 1`.
test "a variadic call default-arg-promotes its variadic args and sets num_fixed" {
    const allocator = std.testing.allocator;
    var module = try cc.compile(allocator,
        \\int printf(const char *fmt, ...);
        \\int f(char c) {
        \\    return printf("x", c);
        \\}
    );
    defer module.deinit(allocator);

    const func = module.find("f").?;
    var found = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const op = func.opcode(inst);
            if (op != .call) continue;
            found = true;
            const c = op.call;
            try std.testing.expect(c.is_variadic);
            try std.testing.expectEqual(@as(u32, 1), c.num_fixed);
            const args = func.valueList(c.args);
            try std.testing.expectEqual(@as(usize, 2), args.len);
            // args[1] is the promoted `c`. It must be i32 (default-promoted), not the
            // original i8 `char`.
            const arg1_ty = func.valueType(args[1]);
            const printed = try std.fmt.allocPrint(allocator, "{f}", .{func.types.fmt(arg1_ty)});
            defer allocator.free(printed);
            try std.testing.expectEqualStrings("i32", printed);
        }
    }
    try std.testing.expect(found);
}

// A call to a variadic function DEFINED IN THIS TU (not just a prototype like `printf` in
// Step 8) must carry the same `is_variadic`/`num_fixed`. This test exercises the
// `findFunc(l.funcs, name)` branch in `lower.zig`'s `.call` arm, rather than the
// `findFuncDecl` branch that Step 8 covers.
test "a call to a same-TU variadic DEFINITION sets is_variadic and num_fixed" {
    const allocator = std.testing.allocator;
    var module = try cc.compile(allocator,
        \\int sum(int n, ...) { return n; }
        \\int f(void) { return sum(1, 2, 3); }
    );
    defer module.deinit(allocator);

    const func = module.find("f").?;
    var found = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const op = func.opcode(inst);
            if (op != .call) continue;
            found = true;
            const c = op.call;
            try std.testing.expect(c.is_variadic);
            try std.testing.expectEqual(@as(u32, 1), c.num_fixed);
            // `n` (the one fixed param) plus the two variadic args `2, 3`.
            const args = func.valueList(c.args);
            try std.testing.expectEqual(@as(usize, 3), args.len);
        }
    }
    try std.testing.expect(found);
}

// This test covers an INDIRECT call through a variadic function-pointer PARAMETER. A variadic
// function-pointer TYPE parses fine. `parseFuncDeclaratorTail` (shared by the bare function
// declarator and the grouped pointer-to-function form) copies `parseParams`'s `is_variadic`
// onto the `FuncType` either way. So `int (*fp)(int, ...)` works as a parameter declarator
// with no adaptation needed. The call then goes through `lower.zig`'s indirect path (`fp` is
// a local or param, so it shadows any same-named function and never reaches
// `findFunc`/`findFuncDecl`). This path must set `is_variadic`/`num_fixed` on the
// `call_indirect` op the same way the direct-call paths do.
test "an indirect call through a variadic function pointer sets is_variadic and num_fixed" {
    const allocator = std.testing.allocator;
    var module = try cc.compile(allocator,
        \\int f(int (*fp)(int, ...)) { return fp(1, 2, 3); }
    );
    defer module.deinit(allocator);

    const func = module.find("f").?;
    var found = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const op = func.opcode(inst);
            if (op != .call_indirect) continue;
            found = true;
            const c = op.call_indirect;
            try std.testing.expect(c.is_variadic);
            try std.testing.expectEqual(@as(u32, 1), c.num_fixed);
            const args = func.valueList(c.args);
            try std.testing.expectEqual(@as(usize, 3), args.len);
        }
    }
    try std.testing.expect(found);
}

// A variadic DEFINITION's own lowered `Function` (not just a CALL to one) carries
// `is_variadic`/`num_fixed_params`. Its body's `__builtin_va_start`/`_arg`/`_end` lower to the
// matching IR ops. This end-to-end test would have caught a real bug: `num_fixed_params` was
// set from `fn_ast.params.len` for EVERY function, fixed-arity included, instead of staying
// `0` for a non-variadic one. See the contrasting `add` test right below, which pins the
// fixed-arity side.
test "a variadic DEFINITION lowers is_variadic/num_fixed_params and its stdarg ops" {
    const allocator = std.testing.allocator;
    var module = try cc.compile(allocator,
        \\int sum(int n, ...){
        \\    __builtin_va_list ap;
        \\    __builtin_va_start(ap, n);
        \\    int x = __builtin_va_arg(ap, int);
        \\    __builtin_va_end(ap);
        \\    return x;
        \\}
    );
    defer module.deinit(allocator);

    const func = module.find("sum").?;
    try std.testing.expect(func.is_variadic);
    try std.testing.expectEqual(@as(u32, 1), func.num_fixed_params);

    var saw_start = false;
    var saw_arg = false;
    var saw_end = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .va_start => saw_start = true,
                .va_arg => saw_arg = true,
                .va_end => saw_end = true,
                else => {},
            }
        }
    }
    try std.testing.expect(saw_start);
    try std.testing.expect(saw_arg);
    try std.testing.expect(saw_end);
}

// The contrasting fixed-arity case. This test pins `num_fixed_params == 0` for a
// NON-variadic function. The field's own doc comment on `Function` says it must stay `0`
// unless `is_variadic` is true. Later backend code trusts that invariant.
test "a non-variadic DEFINITION lowers is_variadic=false and num_fixed_params=0" {
    const allocator = std.testing.allocator;
    var module = try cc.compile(allocator, "int add(int a, int b){ return a + b; }");
    defer module.deinit(allocator);

    const func = module.find("add").?;
    try std.testing.expect(!func.is_variadic);
    try std.testing.expectEqual(@as(u32, 0), func.num_fixed_params);
}

// Step 9: byte-identity guard. A FIXED-ARITY program must compile to the EXACT SAME object
// bytes on this branch as on base commit de487fa (the external-linkage foundation this work
// builds on). Every new field on `Call`/`CallIndirect`/`FuncType`/`Func`/`FuncDecl` defaults
// to the non-variadic value, so no existing (fixed-arity) construction site's behavior can
// have changed. This check runs OUTSIDE `zig build test`, because a unit test cannot compare
// a `.o`'s bytes against a different git commit's checkout:
//
//   git worktree add /tmp/vcc-base-de487fa de487fa
//   (cd /tmp/vcc-base-de487fa && zig build vcc 2>&1 | tail -5)
//   /tmp/vcc-base-de487fa/zig-out/bin/vcc <fixed-arity sample>.c -o base.o
//   zig build vcc && zig-out/bin/vcc <same sample>.c -o work.o
//   sha256sum base.o work.o   # must match

// ============================================================================================
// This section proves the backend variadic CALL side, end to end against REAL glibc.
//
// Each arch compiles a translation unit whose `main` CALLS the real libc variadic `printf`,
// links it against the actual `libc.so.6`, and runs the result under the real glibc `ld.so`
// (natively on the aarch64 host, under `qemu-<arch>` for the cross targets):
//
//     int printf(const char *fmt, ...);
//     int main(void) { return printf(<fmt>, <arg>); }   // returns the char count printf wrote
//
// A tiny HAND-ASSEMBLED `_start` (the ELF entry) calls `main`, then the REAL libc `exit` with
// `main`'s return value. The entry is hand-written, not vcc-compiled, for two reasons:
//   1. A vcc function prologue assumes a normal CALL entry (`rsp % 16 == 8` on x86-64), but the
//      kernel enters `_start` at `rsp % 16 == 0`. A vcc `_start` would misalign every call
//      below it. glibc's `printf` uses `movaps` (16-byte-aligned) when it saves the xmm
//      argument registers, and that faults on a misaligned stack. The hand `_start` calls
//      `main` at the kernel's own alignment, so `main`'s prologue realigns correctly.
//   2. Calling the real libc `exit` (instead of a raw exit syscall) flushes the stdio buffers
//      before `_exit`, so the captured pipe stdout is not lost. The process exit code is
//      `printf`'s return value (the character count), a second independent correctness check.
//
// Three argument shapes exercise every arch's new call-side machinery:
//   * `printf("%d\n", 7)`     -> the integer path (all arches; x86-64 also proves `AL = 0`).
//   * `printf("%s\n", "hi")`  -> a pointer (string-literal) variadic arg.
//   * `printf("%.1f\n", 3.5)` -> the `double` path: x86-64 `AL = 1` (one xmm arg), riscv64's
//                                lp64d anonymous-double-in-a-register rule, aarch64's AAPCS64
//                                v-register. i386's double-on-stack placement is byte-correct
//                                (confirmed by disassembly) but does NOT run here against real
//                                glibc: a `double` argument makes i386 `main` push an odd
//                                number of 4-byte words, so the `call printf` lands on a stack
//                                misaligned by 8 (a PRE-EXISTING i386 backend gap: its frame
//                                does not force the SysV 16-byte call alignment, unrelated to
//                                variadic support and equally breaks any i386 SSE libc call).
//                                i386 int/string DO run: their even push count happens to stay
//                                aligned.
//
// An arch skips cleanly when its glibc `ld.so`/`libc.so.6` or its `qemu` is absent, the same
// way `external_linkage.zig` does.

/// A vcc-compilable program: `main` returns `printf(fmt, arg)`'s char count (a hand `_start`
/// supplies the entry and the libc `exit`). `arg` is spliced verbatim (`7`, `"hi"`, `3.5`).
fn printfProgram(comptime fmt: []const u8, comptime arg: []const u8) []const u8 {
    return "int printf(const char *fmt, ...);\n" ++
        "int main(void) { return printf(\"" ++ fmt ++ "\", " ++ arg ++ "); }\n";
}

/// Emit an already-lowered `mod` as a relocatable object for `arch` (no optimizer, matching the
/// plain vcc driver). Shared by the host-layout `vccObj` and the target-layout `vccObjForTarget`.
/// Caller owns the bytes.
fn emitObj(allocator: std.mem.Allocator, arch: ld.Arch, mod: *cc.Module) ![]u8 {
    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });

    var objdata: std.ArrayList(target.native.ObjData) = .empty;
    defer objdata.deinit(allocator);
    for (mod.data) |d| try objdata.append(allocator, .{
        .name = d.name,
        .bytes = d.bytes,
        .kind = switch (d.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = d.size,
    });

    return target.native.writeObjectDataFor(allocator, arch, mfs.items, objdata.items);
}

/// Compile `source` for `arch` into a relocatable object (no optimizer, matching the plain vcc
/// driver). Defines `main`, imports `printf` as an undefined symbol. Caller owns the bytes.
fn vccObj(allocator: std.mem.Allocator, arch: ld.Arch, source: []const u8) ![]u8 {
    var mod = try cc.compile(allocator, source);
    defer mod.deinit(allocator);
    return emitObj(allocator, arch, &mod);
}

/// `ld.Arch` and `cc.layout.Arch` are distinct enums with the same member names (their orders
/// differ), so map by name.
fn layoutForLdArch(arch: ld.Arch) cc.layout.Arch {
    return switch (arch) {
        .aarch64 => .aarch64,
        .riscv64 => .riscv64,
        .x86_64 => .x86_64,
        .x86 => .x86,
    };
}

/// Like `vccObj`, but builds the frontend types from `arch`'s OWN target layout, not the build
/// host's. This is what makes `__builtin_va_list` take the TARGET arch's ABI shape, so the
/// va_list this object hands to real glibc `vsnprintf` byte-matches what glibc expects.
fn vccObjForTarget(allocator: std.mem.Allocator, arch: ld.Arch, source: []const u8) ![]u8 {
    var mod = try cc.compileForTarget(allocator, source, cc.layout.forArch(layoutForLdArch(arch)), .{});
    defer mod.deinit(allocator);
    return emitObj(allocator, arch, &mod);
}

// --- hand-assembled `_start` objects (call `main`, then libc `exit(main())`) ----------------

// x86-64 `call rel32` relocation (`R_X86_64_PLT32`) and i386 `call rel32` (`R_386_PC32`).
const R_X86_64_PLT32: u32 = 4;
const R_386_PC32: u32 = 2;

/// One symbol / RELA relocation for a raw ELF64 object.
const Sym64 = struct { name: []const u8, value: u64 = 0, shndx: u16, info: u8, size: u64 = 0 };
const Rel64 = struct { offset: u64, sym: u32, typ: u32, addend: i64 };
/// One symbol / `SHT_REL` relocation for a raw ELF32 object.
const Sym32 = struct { name: []const u8, value: u32 = 0, shndx: u16, info: u8, size: u32 = 0 };
const Rel32 = struct { offset: u32, sym: u32, typ: u32 };

/// Emit a minimal ELF64 x86-64 relocatable object (`ET_REL`, `EM_X86_64`). Copied from
/// `external_linkage.zig`'s `writeRawObject64`. x86-64's `object.zig` exposes only the
/// IR-driven `writeModule`, so a hand-assembled `_start` needs this low-level writer.
fn writeRawObject64(allocator: std.mem.Allocator, text: []const u8, syms: []const Sym64, rels: []const Rel64) ![]u8 {
    const w = std.mem.writeInt;
    const has_rela = rels.len > 0;
    const text_ndx: u16 = 1;
    const symtab_ndx: u16 = 2;
    const strtab_ndx: u16 = 3;
    const rela_ndx: u16 = 4;
    const shstrtab_ndx: u16 = if (has_rela) 5 else 4;
    const nsections: u16 = if (has_rela) 6 else 5;

    var strtab: std.ArrayList(u8) = .empty;
    defer strtab.deinit(allocator);
    try strtab.append(allocator, 0);
    var name_offs = try allocator.alloc(u32, syms.len);
    defer allocator.free(name_offs);
    for (syms, 0..) |s, i| {
        name_offs[i] = @intCast(strtab.items.len);
        try strtab.appendSlice(allocator, s.name);
        try strtab.append(allocator, 0);
    }

    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24);
    for (syms, 0..) |s, i| {
        var e: [24]u8 = @splat(0);
        w(u32, e[0..4], name_offs[i], .little);
        e[4] = s.info;
        e[5] = 0;
        w(u16, e[6..8], s.shndx, .little);
        w(u64, e[8..16], s.value, .little);
        w(u64, e[16..24], s.size, .little);
        try symtab.appendSlice(allocator, &e);
    }

    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (rels) |r| {
        var e: [24]u8 = @splat(0);
        w(u64, e[0..8], r.offset, .little);
        w(u64, e[8..16], (@as(u64, r.sym) << 32) | r.typ, .little);
        w(i64, e[16..24], r.addend, .little);
        try rela.appendSlice(allocator, &e);
    }

    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const symtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const rela_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rela.text\x00");
    const shstrtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    const alignUp = struct {
        fn f(v: usize, a: usize) usize {
            return std.mem.alignForward(usize, v, a);
        }
    }.f;
    var off: usize = 64;
    off = alignUp(off, 16);
    const text_off = off;
    off += text.len;
    off = alignUp(off, 8);
    const symtab_off = off;
    off += symtab.items.len;
    const strtab_off = off;
    off += strtab.items.len;
    off = alignUp(off, 8);
    const rela_off = off;
    if (has_rela) off += rela.items.len;
    const shstr_off = off;
    off += shstr.items.len;
    off = alignUp(off, 8);
    const sh_off = off;
    const total = sh_off + @as(usize, nsections) * 64;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2;
    buf[5] = 1;
    buf[6] = 1;
    w(u16, buf[16..18], 1, .little); // ET_REL
    w(u16, buf[18..20], 62, .little); // EM_X86_64
    w(u32, buf[20..24], 1, .little);
    w(u64, buf[40..48], sh_off, .little);
    w(u16, buf[52..54], 64, .little);
    w(u16, buf[58..60], 64, .little);
    w(u16, buf[60..62], nsections, .little);
    w(u16, buf[62..64], shstrtab_ndx, .little);

    @memcpy(buf[text_off..][0..text.len], text);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
    if (has_rela) @memcpy(buf[rela_off..][0..rela.items.len], rela.items);
    @memcpy(buf[shstr_off..][0..shstr.items.len], shstr.items);

    const putShdr = struct {
        fn f(b: []u8, at: usize, name: u32, typ: u32, flags: u64, o: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
            const ww = std.mem.writeInt;
            ww(u32, b[at + 0 ..][0..4], name, .little);
            ww(u32, b[at + 4 ..][0..4], typ, .little);
            ww(u64, b[at + 8 ..][0..8], flags, .little);
            ww(u64, b[at + 24 ..][0..8], o, .little);
            ww(u64, b[at + 32 ..][0..8], size, .little);
            ww(u32, b[at + 40 ..][0..4], sh_link, .little);
            ww(u32, b[at + 44 ..][0..4], sh_info, .little);
            ww(u64, b[at + 48 ..][0..8], addralign, .little);
            ww(u64, b[at + 56 ..][0..8], entsize, .little);
        }
    }.f;
    const SHT_PROGBITS: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_STRTAB: u32 = 3;
    const SHT_RELA: u32 = 4;
    const SHF_ALLOC: u64 = 0x2;
    const SHF_EXECINSTR: u64 = 0x4;
    const SHF_INFO_LINK: u64 = 0x40;

    putShdr(buf, sh_off + text_ndx * 64, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + symtab_ndx * 64, symtab_name, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 8, 24);
    putShdr(buf, sh_off + strtab_ndx * 64, strtab_name, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    if (has_rela) putShdr(buf, sh_off + rela_ndx * 64, rela_name, SHT_RELA, SHF_INFO_LINK, rela_off, rela.items.len, symtab_ndx, text_ndx, 8, 24);
    putShdr(buf, sh_off + shstrtab_ndx * 64, shstrtab_name, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);
    return buf;
}

/// Emit a minimal ELF32 i386 relocatable object (`ET_REL`, `EM_386`), `SHT_REL` (in-field
/// addend). Copied from `external_linkage.zig`'s `writeRawObject32`.
fn writeRawObject32(allocator: std.mem.Allocator, text: []const u8, syms: []const Sym32, rels: []const Rel32) ![]u8 {
    const w = std.mem.writeInt;
    const has_rel = rels.len > 0;
    const text_ndx: u16 = 1;
    const symtab_ndx: u16 = 2;
    const strtab_ndx: u16 = 3;
    const rel_ndx: u16 = 4;
    const shstrtab_ndx: u16 = if (has_rel) 5 else 4;
    const nsections: u16 = if (has_rel) 6 else 5;

    var strtab: std.ArrayList(u8) = .empty;
    defer strtab.deinit(allocator);
    try strtab.append(allocator, 0);
    var name_offs = try allocator.alloc(u32, syms.len);
    defer allocator.free(name_offs);
    for (syms, 0..) |s, i| {
        name_offs[i] = @intCast(strtab.items.len);
        try strtab.appendSlice(allocator, s.name);
        try strtab.append(allocator, 0);
    }

    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 16);
    for (syms, 0..) |s, i| {
        var e: [16]u8 = @splat(0);
        w(u32, e[0..4], name_offs[i], .little);
        w(u32, e[4..8], s.value, .little);
        w(u32, e[8..12], s.size, .little);
        e[12] = s.info;
        e[13] = 0;
        w(u16, e[14..16], s.shndx, .little);
        try symtab.appendSlice(allocator, &e);
    }

    var rel: std.ArrayList(u8) = .empty;
    defer rel.deinit(allocator);
    for (rels) |r| {
        var e: [8]u8 = @splat(0);
        w(u32, e[0..4], r.offset, .little);
        w(u32, e[4..8], (r.sym << 8) | r.typ, .little);
        try rel.appendSlice(allocator, &e);
    }

    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const symtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const rel_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rel.text\x00");
    const shstrtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    const alignUp = struct {
        fn f(v: usize, a: usize) usize {
            return std.mem.alignForward(usize, v, a);
        }
    }.f;
    var off: usize = 52;
    off = alignUp(off, 16);
    const text_off = off;
    off += text.len;
    off = alignUp(off, 4);
    const symtab_off = off;
    off += symtab.items.len;
    const strtab_off = off;
    off += strtab.items.len;
    off = alignUp(off, 4);
    const rel_off = off;
    if (has_rel) off += rel.items.len;
    const shstr_off = off;
    off += shstr.items.len;
    off = alignUp(off, 4);
    const sh_off = off;
    const total = sh_off + @as(usize, nsections) * 40;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1;
    buf[6] = 1;
    w(u16, buf[16..18], 1, .little); // ET_REL
    w(u16, buf[18..20], 3, .little); // EM_386
    w(u32, buf[20..24], 1, .little);
    w(u32, buf[32..36], @intCast(sh_off), .little);
    w(u16, buf[40..42], 52, .little);
    w(u16, buf[46..48], 40, .little);
    w(u16, buf[48..50], nsections, .little);
    w(u16, buf[50..52], shstrtab_ndx, .little);

    @memcpy(buf[text_off..][0..text.len], text);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
    if (has_rel) @memcpy(buf[rel_off..][0..rel.items.len], rel.items);
    @memcpy(buf[shstr_off..][0..shstr.items.len], shstr.items);

    const putShdr = struct {
        fn f(b: []u8, at: usize, name: u32, typ: u32, flags: u32, o: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u32, entsize: u32) void {
            const ww = std.mem.writeInt;
            ww(u32, b[at + 0 ..][0..4], name, .little);
            ww(u32, b[at + 4 ..][0..4], typ, .little);
            ww(u32, b[at + 8 ..][0..4], flags, .little);
            ww(u32, b[at + 16 ..][0..4], @intCast(o), .little);
            ww(u32, b[at + 20 ..][0..4], @intCast(size), .little);
            ww(u32, b[at + 24 ..][0..4], sh_link, .little);
            ww(u32, b[at + 28 ..][0..4], sh_info, .little);
            ww(u32, b[at + 32 ..][0..4], addralign, .little);
            ww(u32, b[at + 36 ..][0..4], entsize, .little);
        }
    }.f;
    const SHT_PROGBITS: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_STRTAB: u32 = 3;
    const SHT_REL: u32 = 9;
    const SHF_ALLOC: u32 = 0x2;
    const SHF_EXECINSTR: u32 = 0x4;
    const SHF_INFO_LINK: u32 = 0x40;

    putShdr(buf, sh_off + text_ndx * 40, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + symtab_ndx * 40, symtab_name, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 4, 16);
    putShdr(buf, sh_off + strtab_ndx * 40, strtab_name, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    if (has_rel) putShdr(buf, sh_off + rel_ndx * 40, rel_name, SHT_REL, SHF_INFO_LINK, rel_off, rel.items.len, symtab_ndx, text_ndx, 4, 8);
    putShdr(buf, sh_off + shstrtab_ndx * 40, shstrtab_name, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);
    return buf;
}

/// The hand `_start` for `arch`: `main` then libc `exit(main())`. Both `main` and `exit` are
/// undefined imports the linker binds (intra-link `main`, `libc.so.6` PLT `exit`). The entry is
/// reached at the kernel's stack alignment, so the `call main` keeps `main` correctly aligned.
fn startObj(allocator: std.mem.Allocator, arch: ld.Arch) ![]u8 {
    switch (arch) {
        .aarch64 => {
            // bl main ; bl exit  (w0 holds main's return, which is exit's argument).
            const encode = target.aarch64.encode;
            const object = target.aarch64.object;
            const words = [_]u32{ encode.bl(0), encode.bl(0) };
            var text: [words.len * 4]u8 = undefined;
            for (words, 0..) |wd, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], wd, .little);
            const symbols = [_]object.Symbol{
                .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
                .{ .name = "main", .kind = .func, .defined = false, .section = .text },
                .{ .name = "exit", .kind = .func, .defined = false, .section = .text },
            };
            const relocs = [_]object.Reloc{
                .{ .offset = 0, .symbol = 1, .type = .call26 }, // bl main  @ byte 0
                .{ .offset = 4, .symbol = 2, .type = .call26 }, // bl exit  @ byte 4
            };
            return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
        },
        .riscv64 => {
            // auipc/jalr main ; auipc/jalr exit  (a0 holds main's return = exit's argument).
            const encode = target.riscv64.encode;
            const object = target.riscv64.object;
            const words = [_]u32{
                encode.auipc(.x1, 0), encode.jalr(.x1, .x1, 0),
                encode.auipc(.x1, 0), encode.jalr(.x1, .x1, 0),
            };
            var text: [words.len * 4]u8 = undefined;
            for (words, 0..) |wd, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], wd, .little);
            const symbols = [_]object.Symbol{
                .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
                .{ .name = "main", .kind = .func, .defined = false, .section = .text },
                .{ .name = "exit", .kind = .func, .defined = false, .section = .text },
            };
            const relocs = [_]object.Reloc{
                .{ .offset = 0, .symbol = 1, .type = .call }, // auipc/jalr main @ byte 0
                .{ .offset = 8, .symbol = 2, .type = .call }, // auipc/jalr exit @ byte 8
            };
            return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
        },
        .x86_64 => {
            //   e8 rel32   call main   @0  (rel32 @1)
            //   89 c7      mov edi,eax @5  (main's return -> exit's first arg)
            //   e8 rel32   call exit   @7  (rel32 @8)
            const text = [_]u8{ 0xe8, 0, 0, 0, 0, 0x89, 0xc7, 0xe8, 0, 0, 0, 0 };
            const syms = [_]Sym64{
                .{ .name = "_start", .shndx = 1, .info = 0x12 },
                .{ .name = "main", .shndx = 0, .info = 0x10 },
                .{ .name = "exit", .shndx = 0, .info = 0x10 },
            };
            const rels = [_]Rel64{
                .{ .offset = 1, .sym = 2, .typ = R_X86_64_PLT32, .addend = -4 },
                .{ .offset = 8, .sym = 3, .typ = R_X86_64_PLT32, .addend = -4 },
            };
            return writeRawObject64(allocator, &text, &syms, &rels);
        },
        .x86 => {
            //   83 e4 f0        and esp,-16   @0   (kernel entry is 16-aligned, keep it so)
            //   e8 fc ff ff ff  call main     @3   (rel32 @4, in-field addend -4)
            //   83 ec 0c        sub esp,12    @8   (pad so the pushed exit arg keeps 16-align)
            //   50              push eax      @11  (main's return -> exit's cdecl stack arg)
            //   e8 fc ff ff ff  call exit     @12  (rel32 @13)
            const text = [_]u8{
                0x83, 0xe4, 0xf0,
                0xe8, 0xfc, 0xff,
                0xff, 0xff, 0x83,
                0xec, 0x0c, 0x50,
                0xe8, 0xfc, 0xff,
                0xff, 0xff,
            };
            const syms = [_]Sym32{
                .{ .name = "_start", .shndx = 1, .info = 0x12 },
                .{ .name = "main", .shndx = 0, .info = 0x10 },
                .{ .name = "exit", .shndx = 0, .info = 0x10 },
            };
            const rels = [_]Rel32{
                .{ .offset = 4, .sym = 2, .typ = R_386_PC32 },
                .{ .offset = 13, .sym = 3, .typ = R_386_PC32 },
            };
            return writeRawObject32(allocator, &text, &syms, &rels);
        },
    }
}

// --- glibc / qemu discovery (mirrors external_linkage.zig) ----------------------------------

/// Locate a real glibc file named `soname` anywhere in the Nix store. Allocated path or null.
fn findFile(allocator: std.mem.Allocator, soname: []const u8) !?[]u8 {
    const cmd = try std.fmt.allocPrint(allocator, "find /nix/store -maxdepth 3 -name {s} 2>/dev/null | head -1", .{soname});
    defer allocator.free(cmd);
    const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ "sh", "-c", cmd } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Locate a `qemu-<suffix>` user-mode emulator on PATH, or null if absent.
fn findQemu(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const cmd = try std.fmt.allocPrint(allocator, "command -v {s}", .{name});
    defer allocator.free(cmd);
    const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ "sh", "-c", cmd } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// The per-arch runtime facts a printf end-to-end test needs: the target, the glibc `ld.so`
/// soname to embed as PT_INTERP, and either native execution (this host is aarch64) or a
/// `qemu-<name>` prefix.
const ArchCase = struct {
    arch: ld.Arch,
    interp_soname: []const u8,
    qemu: ?[]const u8, // null = run natively (host arch)
};

/// Compile `source` for `case.arch`, link `main` + the hand `_start` against the real
/// `libc.so.6` beside the glibc `ld.so`, run the dynexe under that `ld.so` (natively or via
/// qemu), and assert the child exits `expected_code` (printf's char count) with exactly
/// `expected_stdout`. Skips cleanly when the interpreter or qemu is absent.
fn runPrintfCase(allocator: std.mem.Allocator, io: std.Io, case: ArchCase, source: []const u8, expected_stdout: []const u8, expected_code: u8) !void {
    const interp = (try findFile(allocator, case.interp_soname)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = if (case.qemu) |q| (try findQemu(allocator, q)) orelse return error.SkipZigTest else null;
    defer if (qemu) |q| allocator.free(q);

    // `libc.so.6` sits in the same lib dir as the interpreter for these Nix glibc outputs.
    const libdir = std.fs.path.dirname(interp) orelse return error.SkipZigTest;
    const libc_path = try std.fs.path.join(allocator, &.{ libdir, "libc.so.6" });
    defer allocator.free(libc_path);
    const libc_bytes = std.Io.Dir.cwd().readFileAlloc(io, libc_path, allocator, .limited(128 * 1024 * 1024)) catch return error.SkipZigTest;
    defer allocator.free(libc_bytes);

    const main_obj = try vccObj(allocator, case.arch, source);
    defer allocator.free(main_obj);
    const start_obj = try startObj(allocator, case.arch);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libc_bytes } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libc.so.6"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe", .data = dynexe, .flags = .{ .permissions = .executable_file } });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", libdir); // let ld.so find libc.so.6 (and its own deps)

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    if (qemu) |q| try argv.append(allocator, q);
    try argv.append(allocator, "./dynexe");

    const run = std.process.run(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // loader / qemu absent
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);

    switch (run.term) {
        .exited => |code| try std.testing.expectEqual(expected_code, code),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqualStrings(expected_stdout, run.stdout);
}

/// The three arches whose vcc `main` keeps the SysV 16-byte call alignment real glibc `printf`
/// needs for every argument shape (aarch64 native, x86-64/riscv64 under qemu).
const aligned_cases = [_]ArchCase{
    .{ .arch = .aarch64, .interp_soname = "ld-linux-aarch64.so.1", .qemu = null },
    .{ .arch = .x86_64, .interp_soname = "ld-linux-x86-64.so.2", .qemu = "qemu-x86_64" },
    .{ .arch = .riscv64, .interp_soname = "ld-linux-riscv64-lp64d.so.1", .qemu = "qemu-riscv64" },
};

// `printf("%d\n", 7)` -> "7\n" (2 chars): the integer variadic path.
test "variadic call side: printf(\"%d\") of an int prints 7 through real glibc" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // this suite runs on the aarch64 host
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const src = printfProgram("%d\\n", "7");
    for (aligned_cases) |case| try runPrintfCase(allocator, io, case, src, "7\n", 2);
}

// `printf("%s\n", "hi")` -> "hi\n" (3 chars): a pointer (string-literal) variadic arg.
test "variadic call side: printf(\"%s\") of a string prints hi through real glibc" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const src = printfProgram("%s\\n", "\"hi\"");
    for (aligned_cases) |case| try runPrintfCase(allocator, io, case, src, "hi\n", 3);
}

// `printf("%.1f\n", 3.5)` -> "3.5\n" (4 chars): the `double` variadic path. x86-64 AL=1,
// riscv64 anonymous-double-in-GPR, aarch64 v-register. i386 is covered by the probe test
// below. Real glibc `printf` is unreachable on i386 for reasons unrelated to variadic. See
// the module header and the i386 probe test.
test "variadic call side: printf(\"%.1f\") of a double prints 3.5 through real glibc" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const src = printfProgram("%.1f\\n", "3.5");
    for (aligned_cases) |case| try runPrintfCase(allocator, io, case, src, "3.5\n", 4);
}

// --- i386 double-on-stack, verified against a custom variadic reader (no glibc) --------------
//
// i386 cannot reach real glibc `printf` at runtime. `printf` uses SSE (`movaps`), which needs
// a 16-byte-aligned stack. vcc's i386 frame does not force the SysV call alignment (a
// pre-existing i386 gap, unrelated to variadic support). It breaks ANY i386 SSE libc call, and
// real glibc i386 stdio also faults without `__libc_start_main`. So i386's `double` placement is
// proven a different way: a HAND-ASSEMBLED variadic reader `.so` (`probe(int tag, ...)`) that
// only reads its stack arguments (no SSE, no stdio, no alignment need). `probe(7, 3.5)` returns
// 42 only if the i386 call pushed `tag = 7` and the 8 bytes of the `double 3.5` in the right
// order: low word `0x00000000` at the lower address, high word `0x400C0000` above it, the exact
// `push high; push low` sequence this backend uses. A wrong order or a lost word yields 0.

/// `libprobe.so`'s object: `probe` reads its cdecl stack args and returns 42 only if `tag == 7`
/// and the following 8 bytes are the little-endian `double 3.5` (`0x400C000000000000`), else 0.
///   b8 00000000              mov eax, 0                 @0   (default: mismatch)
///   83 7c 24 04 07           cmp dword [esp+4], 7       @5   (tag)
///   75 16                    jne end                    @10
///   83 7c 24 08 00           cmp dword [esp+8], 0       @12  (3.5 low word)
///   75 0f                    jne end                    @17
///   81 7c 24 0c 00000c40     cmp dword [esp+12],0x400C0000 @19 (3.5 high word)
///   75 05                    jne end                    @27
///   b8 2a000000              mov eax, 42                @29  (match)
///   c3                       ret  (end)                 @34
fn probeObjX86(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0xb8, 0x00, 0x00, 0x00, 0x00,
        0x83, 0x7c, 0x24, 0x04, 0x07,
        0x75, 0x16, 0x83, 0x7c, 0x24,
        0x08, 0x00, 0x75, 0x0f, 0x81,
        0x7c, 0x24, 0x0c, 0x00, 0x00,
        0x0c, 0x40, 0x75, 0x05, 0xb8,
        0x2a, 0x00, 0x00, 0x00, 0xc3,
    };
    const syms = [_]Sym32{.{ .name = "probe", .shndx = 1, .info = 0x12 }};
    return writeRawObject32(allocator, &text, &syms, &.{});
}

/// A RAW-exit i386 `_start` (no libc): `call main; mov ebx,eax; mov eax,1; int 0x80`. Used by
/// the probe test, which never touches libc, so a raw `exit` syscall (no stdio flush) is enough.
///   e8 fcffffff  call main   @0  (rel32 @1, in-field addend -4)
///   89 c3        mov ebx,eax @5
///   b8 01000000  mov eax,1   @7  (SYS_exit)
///   cd 80        int 0x80    @12
fn startRawObjX86(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{ 0xe8, 0xfc, 0xff, 0xff, 0xff, 0x89, 0xc3, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xcd, 0x80 };
    const syms = [_]Sym32{
        .{ .name = "_start", .shndx = 1, .info = 0x12 },
        .{ .name = "main", .shndx = 0, .info = 0x10 },
    };
    const rels = [_]Rel32{.{ .offset = 1, .sym = 2, .typ = R_386_PC32 }};
    return writeRawObject32(allocator, &text, &syms, &rels);
}

test "variadic call side: i386 pushes a double arg in the right order (custom reader .so)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findFile(allocator, "ld-linux.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // `probe` is variadic (`tag` fixed, the `double` anonymous), so the vcc call site exercises
    // exactly the i386 float-on-stack path.
    const main_obj = try vccObj(allocator, .x86, "int probe(int tag, ...); int main(void){ return probe(7, 3.5); }");
    defer allocator.free(main_obj);
    const probe_obj = try probeObjX86(allocator);
    defer allocator.free(probe_obj);
    const libprobe = try ld.linkDynamic(allocator, &.{.{ .object = probe_obj }}, .{ .mode = .shared, .soname = "libprobe.so" });
    defer allocator.free(libprobe);
    const start_obj = try startRawObjX86(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libprobe } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libprobe.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libprobe.so", .data = libprobe });
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe", .data = dynexe, .flags = .{ .permissions = .executable_file } });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./dynexe" },
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    switch (run.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code),
        else => return error.TestUnexpectedResult,
    }
}

// ============================================================================================
// This section proves the x86_64 variadic DEFINE side, end to end under qemu-x86_64.
//
// A vcc-compiled variadic FUNCTION (`int sum(int n, ...)`, `double favg(int n, ...)`) reads its
// unnamed arguments with `__builtin_va_start`/`__builtin_va_arg`/`__builtin_va_end`. The isel now
// emits the System V register-save-area prologue (176 bytes, GP spill + AL-gated movaps xmm save)
// and expands each `va_arg` into the gp_offset/fp_offset walk with a register/overflow branch.
//
// Two things are self-contained (no libc): a raw `exit` syscall entry, and the argument setup.
//   * `sum` INT case: the caller must pass MORE than the six System V GP argument registers so
//     the `va_arg` OVERFLOW path (reading from `overflow_arg_area`, the incoming stack args)
//     actually runs. The vcc CALL side does not yet emit stack arguments (a known gap, unrelated
//     to this define-side test), so a tiny HAND-ASSEMBLED `_start` places `n=10` plus ten int
//     varargs (five in rsi..r9, five on the stack, SysV-aligned) and calls vcc's `sum`, then
//     `exit`s with the returned total. `sum(10, 10,9,8,7,3,1,1,1,1,1) == 42`.
//   * `favg` DOUBLE case: eight `double` varargs stay in xmm0..xmm7 (no fp stack args, so the vcc
//     CALL side handles it), which exercises the AL-gated `movaps` save and the fp_offset register
//     walk. A full vcc TU (`favg` + a `main` that calls it and casts to int) is linked with a plain
//     `call main; exit` entry. `(int)favg(8, 5,5,5,5,5,5,5,7) == 42`.
//
// Each case is also checked against a gcc-differential oracle: the SAME C source compiled with
// `zig cc -target x86_64-linux-musl -static` (a real System V compiler, and for `sum` a real
// stack-argument caller) and run under qemu. The check asserts the oracle's exit matches vcc's
// (both 42). The suite skips cleanly when `qemu-x86_64` (or `zig cc`) is absent, never a faked
// pass.

/// `_start` that hand-places the arguments for `sum(10, 10,9,8,7,3,1,1,1,1,1)` per System V and
/// `exit`s with the returned total. n -> rdi, the first five varargs -> rsi,rdx,rcx,r8,r9, the last
/// five -> the stack (SysV: first stack arg at the lowest address, rsp 16-aligned at the `call`),
/// AL = 0 (no xmm args). Bytes verified against `zig cc`'s assembler. The `call sum` rel32 sits at
/// byte 59 and binds to the intra-link `sum` (undefined import) via an `R_X86_64_PLT32` reloc.
fn startCallSumObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0x48, 0xc7, 0xc7, 0x0a, 0x00, 0x00, 0x00, // mov rdi, 10   (n)
        0x48, 0xc7, 0xc6, 0x0a, 0x00, 0x00, 0x00, // mov rsi, 10
        0x48, 0xc7, 0xc2, 0x09, 0x00, 0x00, 0x00, // mov rdx, 9
        0x48, 0xc7, 0xc1, 0x08, 0x00, 0x00, 0x00, // mov rcx, 8
        0x49, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov r8, 7
        0x49, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00, // mov r9, 3
        0x48, 0x83, 0xec, 0x08, // sub rsp, 8     (pad: five 8-byte pushes + 8 keep 16-alignment)
        0x6a, 0x01, // push 1
        0x6a, 0x01, // push 1
        0x6a, 0x01, // push 1
        0x6a, 0x01, // push 1
        0x6a, 0x01, // push 1
        0x31, 0xc0, // xor eax, eax   (AL = 0: no xmm args)
        0xe8, 0x00, 0x00, 0x00, 0x00, // call sum   (rel32 @ byte 59)
        0x89, 0xc7, // mov edi, eax   (total -> exit's first arg)
        0xb8, 0x3c, 0x00, 0x00, 0x00, // mov eax, 60  (SYS_exit)
        0x0f, 0x05, // syscall
    };
    const syms = [_]Sym64{
        .{ .name = "_start", .shndx = 1, .info = 0x12 },
        .{ .name = "sum", .shndx = 0, .info = 0x10 }, // undefined import (intra-link)
    };
    const rels = [_]Rel64{.{ .offset = 59, .sym = 2, .typ = R_X86_64_PLT32, .addend = -4 }};
    return writeRawObject64(allocator, &text, &syms, &rels);
}

/// `_start` = `call main; mov edi,eax; mov eax,60; syscall`. This is the plain self-contained
/// entry (a raw `exit` syscall, no libc) for the vcc TUs whose own `main` sets up the call. The
/// `call main` rel32 is at byte 1, an `R_X86_64_PLT32` reloc to the intra-link `main`.
fn startCallMainObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0xe8, 0x00, 0x00, 0x00, 0x00, // call main
        0x89, 0xc7, // mov edi, eax
        0xb8, 0x3c, 0x00, 0x00, 0x00, // mov eax, 60
        0x0f, 0x05, // syscall
    };
    const syms = [_]Sym64{
        .{ .name = "_start", .shndx = 1, .info = 0x12 },
        .{ .name = "main", .shndx = 0, .info = 0x10 },
    };
    const rels = [_]Rel64{.{ .offset = 1, .sym = 2, .typ = R_X86_64_PLT32, .addend = -4 }};
    return writeRawObject64(allocator, &text, &syms, &rels);
}

/// Link `start_obj` + `main_obj` into a plain STATIC x86_64 executable (a static link, no
/// PT_INTERP, no dynamic segment). Write it to a fresh tmp dir, run it under `qemu`, and return
/// the child's exit code. Skips when qemu is absent mid-run (never masks a wrong exit).
fn linkRunStaticX86(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, start_obj: []const u8, main_obj: []const u8) !u8 {
    var image = try ld.linkInputs(allocator, &.{ .{ .object = start_obj }, .{ .object = main_obj } }, 0x400000);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(.x86_64, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./a.out" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// The gcc-differential oracle: compile `source` with `zig cc` for x86_64 (`-static`, a real
/// System V compiler, and for the `sum` program, a real stack-argument caller). Run it under
/// `qemu`, and return its exit code. Returns null when `zig cc` cannot build (e.g. it is
/// unavailable), so the caller skips rather than fakes a pass.
fn oracleExitX86(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, source: []const u8) !?u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "o.c", .data = source });

    const build = std.process.run(allocator, io, .{
        .argv = &.{ "zig", "cc", "-target", "x86_64-linux-musl", "-static", "-O0", "-o", "oracle", "o.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null, // zig absent
        else => return e,
    };
    defer allocator.free(build.stdout);
    defer allocator.free(build.stderr);
    switch (build.term) {
        .exited => |code| if (code != 0) return null, // could not build the oracle -> skip
        else => return null,
    }

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./oracle" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// The vcc translation unit under test for the INT overflow case: just `sum` (the hand `_start`
/// supplies the overflowing caller). Kept minimal so the isel emits exactly the variadic define-side
/// prologue + `va_arg` walk under test.
const sum_src =
    \\int sum(int n, ...) {
    \\    int t = 0;
    \\    __builtin_va_list ap;
    \\    __builtin_va_start(ap, n);
    \\    for (int i = 0; i < n; i++) t = t + __builtin_va_arg(ap, int);
    \\    __builtin_va_end(ap);
    \\    return t;
    \\}
;

/// The gcc oracle's program for the INT case: the SAME `sum`, plus a C `main` that calls it with the
/// SAME ten varargs the hand `_start` uses (the real compiler handles the stack-argument passing).
const sum_oracle_src = sum_src ++
    \\
    \\int main(void) { return sum(10, 10, 9, 8, 7, 3, 1, 1, 1, 1, 1); }
;

test "x86_64: vcc variadic sum() reads its int varargs (register + overflow) to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccObj(allocator, .x86_64, sum_src);
    defer allocator.free(main_obj);
    const start_obj = try startCallSumObj(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticX86(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    // gcc-differential: the same C `sum` compiled by a real SysV compiler (with a real stack-arg
    // caller) exits with the same total.
    if (try oracleExitX86(allocator, io, qemu, sum_oracle_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

/// The vcc TU for the DOUBLE case: `favg` sums its `double` varargs, and `main` calls it with eight
/// doubles (all in xmm0..xmm7, so the vcc CALL side needs no fp stack args). `5*7 + 7 == 42.0`. The
/// result is turned into an exit code with a `double` COMPARE (not an `(int)` cast of a call result,
/// which the frontend does not yet lower), so the test stays focused on the variadic define side.
const favg_src =
    \\double favg(int n, ...) {
    \\    double t = 0;
    \\    __builtin_va_list ap;
    \\    __builtin_va_start(ap, n);
    \\    for (int i = 0; i < n; i++) t = t + __builtin_va_arg(ap, double);
    \\    __builtin_va_end(ap);
    \\    return t;
    \\}
    \\int main(void) { return favg(8, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.0) == 42.0 ? 42 : 0; }
;

test "x86_64: vcc variadic favg() reads its double varargs (xmm save) to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccObj(allocator, .x86_64, favg_src);
    defer allocator.free(main_obj);
    const start_obj = try startCallMainObj(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticX86(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    if (try oracleExitX86(allocator, io, qemu, favg_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

// ============================================================================================
// This section proves the aarch64 variadic DEFINE side, end to end NATIVELY on the aarch64 host.
//
// A vcc-compiled variadic FUNCTION (`int sum(int n, ...)`, `double favg(int n, ...)`) reads its
// unnamed arguments with `__builtin_va_start`/`_arg`/`_end`. The isel now emits the AAPCS64
// register-save-area prologue (a 64-byte x0..x7 GP block + a 128-byte q0..q7 VR block, both filled
// unconditionally) and expands each `va_arg` into the `__gr_offs`/`__vr_offs` walk with a
// register/overflow branch (`__*_top + __*_offs` while the offset is negative, else `__stack`).
//
// Both cases are self-contained (no libc): a hand `_start` (a raw `exit` syscall entry) plus the
// argument setup, linked with the static linker into a plain aarch64 executable that runs
// NATIVELY (this host is aarch64, no qemu):
//   * `sum` INT case: the caller must pass MORE than the seven x1..x7 varargs registers (x0 holds
//     the named `n`) so the `va_arg` OVERFLOW path (reading `__stack`, the incoming stack args)
//     actually runs. The vcc CALL side does not yet emit stack arguments (a known gap, unrelated
//     to this define-side test), so a hand `_start` places `n=10` plus ten int varargs (seven in
//     x1..x7, three on the stack, 16-byte aligned at the `bl`) and calls vcc's `sum`, then `exit`s
//     with the total. `sum(10, 10,9,8,7,3,1,1,1,1,1) == 42`.
//   * `favg` DOUBLE case: eight `double` varargs stay in v0..v7 (no fp stack args, so the vcc CALL
//     side handles it), exercising the VR save area and the `__vr_offs` register walk. A full vcc TU
//     (`favg` + a `main` that calls it and compares to 42.0) is linked with a plain `bl main; exit`
//     entry. `favg(8, 5,5,5,5,5,5,5,7) == 42.0`.
//
// Each case is also checked against a gcc-differential oracle: the SAME C source compiled with the
// host `cc` (a real AAPCS64 compiler, and for `sum` a real stack-argument caller) and run natively.
// The check asserts the oracle's exit matches vcc's (both 42). The suite skips cleanly when `cc` is
// absent, never a faked pass.

/// A hand `_start` that places the arguments for `sum(10, 10,9,8,7,3,1,1,1,1,1)` per AAPCS64 and
/// `exit`s natively with the returned total. `n` -> x0, the first seven varargs -> x1..x7, the last
/// three -> the stack (first stack arg at the lowest address, sp 16-aligned at the `bl`). Built with
/// the aarch64 backend's own `object.write`, so the `bl sum` is a real `R_AARCH64_CALL26` relocation
/// to the intra-link `sum` (an undefined import).
fn startSumAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.aarch64.encode;
    const object = target.aarch64.object;
    const words = [_]u32{
        encode.movz(.x0, 10, 0), // n = 10
        encode.movz(.x1, 10, 0),
        encode.movz(.x2, 9, 0),
        encode.movz(.x3, 8, 0),
        encode.movz(.x4, 7, 0),
        encode.movz(.x5, 3, 0),
        encode.movz(.x6, 1, 0),
        encode.movz(.x7, 1, 0),
        encode.subImm64(.zr, .zr, 32), // sub sp, sp, #32  (three 8-byte stack args + 8 pad, 16-aligned)
        encode.movz(.x9, 1, 0), // the three stack varargs are all 1
        encode.strOff(.x9, .zr, 0), // [sp, #0]
        encode.strOff(.x9, .zr, 8), // [sp, #8]
        encode.strOff(.x9, .zr, 16), // [sp, #16]
        encode.bl(0), // bl sum  (word 13, byte 52), patched by the CALL26 reloc
        encode.addImm64(.zr, .zr, 32), // add sp, sp, #32
        encode.movz(.x8, 93, 0), // x8 = 93 (SYS_exit); x0 already holds sum's return
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |wd, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], wd, .little);
    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "sum", .kind = .func, .defined = false, .section = .text },
    };
    const relocs = [_]object.Reloc{.{ .offset = 52, .symbol = 1, .type = .call26 }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// A plain aarch64 `_start` = `bl main; mov x8,#93; svc #0`. This is the self-contained
/// raw-`exit` entry (no libc) for a vcc TU whose own `main` sets up the call. The `bl main`
/// (word 0, byte 0) binds to the intra-link `main` via an `R_AARCH64_CALL26` reloc.
fn startMainRawAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.aarch64.encode;
    const object = target.aarch64.object;
    const words = [_]u32{ encode.bl(0), encode.movz(.x8, 93, 0), encode.svc(0) };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |wd, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], wd, .little);
    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "main", .kind = .func, .defined = false, .section = .text },
    };
    const relocs = [_]object.Reloc{.{ .offset = 0, .symbol = 1, .type = .call26 }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// Link `start_obj` + `main_obj` into a plain STATIC aarch64 executable (a static link). Write
/// it to a fresh tmp dir, run it NATIVELY (this host is aarch64), and return the child's exit
/// code. Skips when execution is impossible mid-run (never masks a wrong exit).
fn linkRunStaticAArch64(allocator: std.mem.Allocator, io: std.Io, start_obj: []const u8, main_obj: []const u8) !u8 {
    var image = try ld.linkInputs(allocator, &.{ .{ .object = start_obj }, .{ .object = main_obj } }, 0x400000);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(.aarch64, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });

    const run = std.process.run(allocator, io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// The gcc-differential oracle: compile `source` with the host `cc` (a real AAPCS64 compiler, and for
/// `sum` a real stack-argument caller), run it NATIVELY, and return its exit code. `cc` accepts the
/// `__builtin_va_*` spellings directly (no `<stdarg.h>` needed), so the vcc source compiles verbatim.
/// Returns null when `cc` cannot build (e.g. it is absent), so the caller skips rather than fakes.
fn oracleExitNative(allocator: std.mem.Allocator, io: std.Io, source: []const u8) !?u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "o.c", .data = source });

    const build = std.process.run(allocator, io, .{
        .argv = &.{ "cc", "-O0", "-o", "oracle", "o.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null, // cc absent
        else => return e,
    };
    defer allocator.free(build.stdout);
    defer allocator.free(build.stderr);
    switch (build.term) {
        .exited => |code| if (code != 0) return null, // could not build the oracle -> skip
        else => return null,
    }

    const run = std.process.run(allocator, io, .{
        .argv = &.{"./oracle"},
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

test "aarch64: vcc variadic sum() reads its int varargs (register + overflow) to exit 42 natively" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // native aarch64 host only
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const main_obj = try vccObj(allocator, .aarch64, sum_src);
    defer allocator.free(main_obj);
    const start_obj = try startSumAArch64(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticAArch64(allocator, io, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    // gcc-differential: the same C `sum` compiled by the host AAPCS64 compiler (with a real
    // stack-arg caller) exits with the same total.
    if (try oracleExitNative(allocator, io, sum_oracle_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

test "aarch64: vcc variadic favg() reads its double varargs (VR save) to exit 42 natively" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const main_obj = try vccObj(allocator, .aarch64, favg_src);
    defer allocator.free(main_obj);
    const start_obj = try startMainRawAArch64(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticAArch64(allocator, io, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    if (try oracleExitNative(allocator, io, favg_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

// This section proves the riscv64 variadic DEFINE side (LP64D), end to end under qemu-riscv64.
//
// A vcc-compiled variadic FUNCTION (`int sum(int n, ...)`, `double favg(int n, ...)`) reads its
// unnamed arguments through the LP64D `va_list` (a plain `void*` walking one contiguous 8-byte-slot
// run: the a0..a7 save block, then the incoming stack args). Two cases:
//   * `sum` INT case: the caller passes MORE than the seven a1..a7 vararg registers (a0 holds the
//     fixed `n`), so the va_arg walk crosses from the register save block into the incoming stack
//     args. The two must be contiguous. The vcc CALL side does not yet emit stack arguments (a
//     known gap, unrelated to the define side), so a HAND-ASSEMBLED `_start` places the ten
//     varargs (seven in a1..a7, three on the stack, sp 16-aligned at the call) and `exit`s with
//     the total. `sum(10, 10,9,8,7,3,1,1,1,1,1) == 42`.
//   * `favg` DOUBLE case: this specifically proves the LP64D rule that anonymous doubles arrive in
//     the INTEGER registers a1..a7 (raw bits, via the caller's fmv.x.d), get spilled by the define
//     side, and are read back by `va_arg(ap, double)` with a plain `fld` from the slot. Seven doubles
//     stay in a1..a7 (no stack vararg, so the vcc CALL side handles the call), so a full vcc TU
//     (`favg` + a `main` that calls it and compares to 42.0) links with a plain `call main; exit`
//     entry. `favg(7, 6,6,6,6,6,6,6) == 42.0`.
//
// Each case is also checked against a gcc-differential oracle: the SAME C source compiled with
// `zig cc -target riscv64-linux-musl -static` (a real LP64D compiler, and for `sum` a real stack-
// argument caller) and run under qemu. The check asserts the oracle's exit matches vcc's (both
// 42). The suite skips cleanly when `qemu-riscv64` (or `zig cc`) is absent, never a faked pass.

/// A hand `_start` that places the arguments for `sum(10, 10,9,8,7,3,1,1,1,1,1)` per LP64D and
/// `exit`s (raw syscall) with the returned total. n -> a0, the first seven varargs -> a1..a7, the
/// last three -> the stack (LP64D: first stack arg at the lowest address, sp 16-aligned at the call).
/// The `auipc/jalr sum` pair (word 13, byte 52) binds to the intra-link `sum` (an undefined import)
/// via a `.call` relocation.
fn startCallSumRiscv(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.riscv64.encode;
    const object = target.riscv64.object;
    const words = [_]u32{
        encode.addi(.x10, .x0, 10), // a0 = 10   (n)
        encode.addi(.x11, .x0, 10), // a1 = 10
        encode.addi(.x12, .x0, 9), // a2 = 9
        encode.addi(.x13, .x0, 8), // a3 = 8
        encode.addi(.x14, .x0, 7), // a4 = 7
        encode.addi(.x15, .x0, 3), // a5 = 3
        encode.addi(.x16, .x0, 1), // a6 = 1
        encode.addi(.x17, .x0, 1), // a7 = 1
        encode.addi(.x2, .x2, -32), // sp -= 32 (16-aligned stack-arg area, three 8-byte slots + pad)
        encode.addi(.x5, .x0, 1), // t0 = 1  (the three stack varargs)
        encode.sd(.x5, .x2, 0), // [sp+0]  = 1
        encode.sd(.x5, .x2, 8), // [sp+8]  = 1
        encode.sd(.x5, .x2, 16), // [sp+16] = 1
        encode.auipc(.x1, 0), // call sum   (word 13, byte 52)
        encode.jalr(.x1, .x1, 0),
        encode.addi(.x2, .x2, 32), // sp += 32 (restore before exit)
        encode.addi(.x17, .x0, 93), // a7 = 93 (SYS_exit); a0 already holds sum's return
        encode.ecall(),
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |wd, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], wd, .little);
    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "sum", .kind = .func, .defined = false, .section = .text },
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 52, .symbol = 1, .type = .call }, // auipc/jalr sum @ byte 52
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// A hand `_start` = `call main; exit(main())` (raw syscall, no libc) for the vcc TU whose own `main`
/// sets up the `favg` call. The `auipc/jalr main` pair (word 0, byte 0) binds to the intra-link
/// `main` (an undefined import).
fn startCallMainRiscv(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.riscv64.encode;
    const object = target.riscv64.object;
    const words = [_]u32{
        encode.auipc(.x1, 0), // call main  (word 0, byte 0)
        encode.jalr(.x1, .x1, 0),
        encode.addi(.x17, .x0, 93), // a7 = 93 (SYS_exit); a0 already holds main's return
        encode.ecall(),
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |wd, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], wd, .little);
    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "main", .kind = .func, .defined = false, .section = .text },
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 0, .symbol = 1, .type = .call }, // auipc/jalr main @ byte 0
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// Link `start_obj` + `main_obj` into a plain STATIC riscv64 executable (a static link: no
/// PT_INTERP, no dynamic segment). Write it to a fresh tmp dir, run it under `qemu`, and return
/// the child's exit code. Skips when qemu is absent mid-run (never masks a wrong exit).
fn linkRunStaticRiscv(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, start_obj: []const u8, main_obj: []const u8) !u8 {
    var image = try ld.linkInputs(allocator, &.{ .{ .object = start_obj }, .{ .object = main_obj } }, 0x400000);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(.riscv64, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./a.out" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// The gcc-differential oracle: compile `source` with `zig cc` for riscv64 (`-static`, a real
/// LP64D compiler, and for the `sum` program, a real stack-argument caller). Run it under
/// `qemu`, and return its exit code. Returns null when `zig cc` cannot build (e.g. it is
/// unavailable), so the caller skips rather than fakes a pass.
fn oracleExitRiscv(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, source: []const u8) !?u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "o.c", .data = source });

    const build = std.process.run(allocator, io, .{
        .argv = &.{ "zig", "cc", "-target", "riscv64-linux-musl", "-static", "-O0", "-o", "oracle", "o.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null, // zig absent
        else => return e,
    };
    defer allocator.free(build.stdout);
    defer allocator.free(build.stderr);
    switch (build.term) {
        .exited => |code| if (code != 0) return null, // could not build the oracle -> skip
        else => return null,
    }

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./oracle" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// The vcc TU for the riscv64 DOUBLE case: `favg` sums its `double` varargs, and `main` calls it with
/// SEVEN doubles (all in a1..a7 per LP64D's anonymous-double-in-a-register rule, so the vcc CALL side
/// needs no stack vararg). `6*7 == 42.0`. The result is turned into an exit code with a `double`
/// COMPARE (not an `(int)` cast of a call result, which the frontend does not yet lower), so the test
/// stays focused on the variadic define side.
const favg_rv_src =
    \\double favg(int n, ...) {
    \\    double t = 0;
    \\    __builtin_va_list ap;
    \\    __builtin_va_start(ap, n);
    \\    for (int i = 0; i < n; i++) t = t + __builtin_va_arg(ap, double);
    \\    __builtin_va_end(ap);
    \\    return t;
    \\}
    \\int main(void) { return favg(7, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0) == 42.0 ? 42 : 0; }
;

test "riscv64: vcc variadic sum() reads its int varargs (register + stack overflow) to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccObj(allocator, .riscv64, sum_src);
    defer allocator.free(main_obj);
    const start_obj = try startCallSumRiscv(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticRiscv(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    // gcc-differential: the same C `sum` compiled by a real LP64D compiler (with a real stack-arg
    // caller) exits with the same total.
    if (try oracleExitRiscv(allocator, io, qemu, sum_oracle_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

test "riscv64: vcc variadic favg() reads its double varargs (int-reg spill, fld read-back) to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccObj(allocator, .riscv64, favg_rv_src);
    defer allocator.free(main_obj);
    const start_obj = try startCallMainRiscv(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticRiscv(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    if (try oracleExitRiscv(allocator, io, qemu, favg_rv_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

// ============================================================================================
// This section proves the i386 (cdecl) variadic DEFINE side, end to end under qemu-i386.
//
// cdecl passes EVERY argument on the CALLER's stack (no argument registers at all), so
// `va_list` is a plain `char*` walking it. The isel emits NO register-save-area and NO
// prologue spill. `va_start` seeds the pointer just past the last named parameter. `va_arg`
// reads through it and bumps it by the argument's own 4-byte-rounded size. `va_end` is a
// no-op.
//
//   * `sum` INT case: the vcc CALL side already pushes every cdecl argument correctly, so
//     unlike the other three arches this needs NO hand-assembled argument-setup `_start` at
//     all. `sum` AND its `main` caller are BOTH compiled by vcc, linked with a plain
//     raw-syscall `_start` (`call main; exit(main())`, already defined above as
//     `startRawObjX86`), and run under qemu. `sum` reads its THREE variadic args with three
//     STRAIGHT-LINE `__builtin_va_arg` calls rather than a loop. This backend's `lowerInst` has
//     NO arm for the IR `convert` op at all (a pre-existing gap unrelated to variadic support.
//     The other three arches' isel all implement it). A raw, unoptimized `while`/`for`
//     condition always lowers through that op (`convert i32, (icmp)` before the `!= 0` branch
//     test). Straight-line code has no such condition, so it exercises `va_start` and THREE
//     consecutive `va_arg` advances without depending on that unrelated gap. The call is also
//     capped at four total arguments, a second, separate, also pre-existing register-allocator
//     capacity limit (see `sum_x86_32_src`'s own doc comment). `sum(3, 20,14,8) == 42`.
//   * DOUBLE case: x86-32 has NO float register class at all (see the isel module header). This
//     is a pre-existing gap unrelated to variadic support (ANY `double` local or arithmetic
//     already fails to compile on this backend, confirmed by the last test below). So
//     `__builtin_va_arg(ap, double)` cannot produce a USABLE result here, but its POINTER
//     ADVANCE (the 8-byte stride this backend uses) is still real machine code that runs.
//     `afterdbl` walks PAST a discarded `double` vararg and then reads the `int` immediately
//     following it. This only returns the right value if the `double` was recognized as an
//     8-byte (not 4-byte) stride, exercising exactly the requirement without needing a float
//     register. `afterdbl(1, 3.5, 42) == 42`.
//
// Each case is also checked against a gcc-differential oracle: the SAME C source compiled with
// `zig cc -target x86-linux-musl -static` (a real cdecl compiler) and run under qemu. The check
// asserts the oracle's exit matches vcc's. The suite skips cleanly when `qemu-i386` (or `zig
// cc`) is absent, never a faked pass.

/// The gcc-differential oracle: compile `source` with `zig cc` for i386 (`-static`, a real
/// cdecl compiler), run it under `qemu-i386`, and return its exit code. Returns null when `zig
/// cc` cannot build (e.g. it is unavailable), so the caller skips rather than fakes a pass.
fn oracleExitX86_32(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, source: []const u8) !?u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "o.c", .data = source });

    const build = std.process.run(allocator, io, .{
        .argv = &.{ "zig", "cc", "-target", "x86-linux-musl", "-static", "-O0", "-o", "oracle", "o.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null, // zig absent
        else => return e,
    };
    defer allocator.free(build.stdout);
    defer allocator.free(build.stderr);
    switch (build.term) {
        .exited => |code| if (code != 0) return null, // could not build the oracle -> skip
        else => return null,
    }

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./oracle" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// Link `start_obj` + `main_obj` into a plain STATIC i386 executable (a static link: no
/// PT_INTERP, no dynamic segment). Write it to a fresh tmp dir, run it under `qemu-i386`,
/// and return the child's exit code. Skips when qemu is absent mid-run (never masks a wrong exit).
fn linkRunStaticX86_32(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, start_obj: []const u8, main_obj: []const u8) !u8 {
    var image = try ld.linkInputs(allocator, &.{ .{ .object = start_obj }, .{ .object = main_obj } }, 0x400000);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(.x86, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });

    const run = std.process.run(allocator, io, .{
        .argv = &.{ qemu, "./a.out" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |code| code,
        else => error.TestUnexpectedResult,
    };
}

/// The vcc TU for the i386 INT case: `sum` reads THREE variadic ints with three STRAIGHT-LINE
/// `__builtin_va_arg` calls. See the module comment above for why not a loop: a pre-existing gap
/// in this backend's `convert` support, unrelated to variadic support, which a raw loop
/// condition always needs. `t = t + __builtin_va_arg(...)` is an accumulate-and-forget shape,
/// rather than several named locals live at once, and it keeps register pressure low.
/// `vccObj` runs no optimizer, so each statement reloads `t` from its stack slot fresh, matching
/// every other arch's own `sum` shape. The CALL site is capped at four total arguments (`n`
/// plus three varargs): the shared Wimmer allocator's spill fallback hits its own separate,
/// pre-existing capacity limit (also unrelated to variadic support) on x86-32's four-register
/// gpr pool, once a single `call` needs five or more simultaneously-materialized argument
/// values. This is confirmed with a plain non-variadic `sum(int, ...)` call of increasing arity.
/// Each statement still walks a FRESH `va_list` position, so this genuinely exercises repeated
/// `va_arg` advances, not just one. `20+14+8 == 42`.
const sum_x86_32_src =
    \\int sum(int n, ...) {
    \\    __builtin_va_list ap;
    \\    __builtin_va_start(ap, n);
    \\    int t = __builtin_va_arg(ap, int);
    \\    t = t + __builtin_va_arg(ap, int);
    \\    t = t + __builtin_va_arg(ap, int);
    \\    __builtin_va_end(ap);
    \\    return t;
    \\}
    \\int main(void) { return sum(3, 20, 14, 8); }
;

test "i386: vcc variadic sum() reads its int varargs (all on the cdecl stack) to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // Unlike the other three arches, i386's cdecl CALL side already pushes every argument (no
    // register-passed arguments at all). So `sum` AND its caller are BOTH compiled by vcc. No
    // hand-assembled argument setup is needed, just the raw-syscall `_start`
    // (`startRawObjX86`, defined above for the call-side probe test).
    const main_obj = try vccObj(allocator, .x86, sum_x86_32_src);
    defer allocator.free(main_obj);
    const start_obj = try startRawObjX86(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticX86_32(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    // gcc-differential: the same C `sum` compiled by a real cdecl compiler exits with the same total.
    if (try oracleExitX86_32(allocator, io, qemu, sum_x86_32_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

/// The vcc TU for the i386 DOUBLE case (see the module comment above for why this is not a real
/// `favg`): `afterdbl` discards a `double` vararg (just to walk past it) then reads the `int`
/// immediately following it on the stack. This only returns the right value if the `double` was
/// recognized as an 8-byte stride, not a 4-byte one. That is exactly the requirement this test
/// exercises, without needing a float register (which this backend does not have at all).
const afterdbl_src =
    \\int afterdbl(int n, ...) {
    \\    __builtin_va_list ap;
    \\    __builtin_va_start(ap, n);
    \\    __builtin_va_arg(ap, double);
    \\    int t = __builtin_va_arg(ap, int);
    \\    __builtin_va_end(ap);
    \\    return t;
    \\}
    \\int main(void) { return afterdbl(1, 3.5, 42); }
;

test "i386: vcc variadic afterdbl() advances its va_list past a double vararg by 8 bytes (not 4) to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccObj(allocator, .x86, afterdbl_src);
    defer allocator.free(main_obj);
    const start_obj = try startRawObjX86(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticX86_32(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(@as(u8, 42), got);

    if (try oracleExitX86_32(allocator, io, qemu, afterdbl_src)) |oracle| {
        try std.testing.expectEqual(got, oracle);
    }
}

// A real `favg` that USES its `double` vararg in arithmetic (mirroring the other three arches'
// own test) is rejected CLEANLY on i386, not silently miscompiled. x86-32 has no float register
// class at all (a pre-existing backend gap unrelated to variadic support). `double t = 0;`'s
// own alloca already fails `typeSize` before any variadic-specific code runs.
test "i386: a double favg() that uses __builtin_va_arg(ap, double) in arithmetic is rejected cleanly, not miscompiled" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, vccObj(allocator, .x86, favg_src));
}

// ============================================================================================
// This section proves the x86-32 `.convert` isel arm. `lowerInst` had NO arm for the IR
// `convert` op at all (a pre-existing gap unrelated to variadic support, noted above near the
// i386 `sum` test and the real-glibc skip test below). So ANY raw, unoptimized C
// assignment between two different integer widths (e.g. `int i = some_char;`) failed to
// compile on this backend with `error.Unsupported`. Each case below is checked against `zig cc
// -target x86-linux-musl -static` (a real cdecl compiler) run under the same qemu-i386, so a
// wrong sign/zero-extension choice shows up as a mismatched exit code, not just "it compiled".

/// WIDENING, signed source: a negative `char` widened to `int` must sign-extend (`movsx`).
/// `c = -5` widens to `i = -5`. The process exits with its LOW BYTE (`-5 & 0xff = 251`). This
/// is the same truncation `exit()` itself performs, and it matches gcc's own exit code for
/// this source.
const convert_sext_byte_src =
    \\int f(void) { char c = -5; int i = c; return i; }
    \\int main(void) { return f(); }
;

/// WIDENING, unsigned source: an `unsigned char` widened to `int` must zero-extend (`movzx`),
/// never sign-extend. `c = 200` stays `200`, fitting a single exit byte unchanged.
const convert_zext_byte_src =
    \\int f(void) { unsigned char c = 200; int i = c; return i; }
    \\int main(void) { return f(); }
;

/// WIDENING, 16-bit signed source: a negative `short` widened to `int` must sign-extend
/// through the 16-bit form, not the 8-bit form. `s = -1000` widens to `i = -1000`. The exit
/// byte is its low byte (`-1000` as `0xFFFFFC18`, low byte `0x18 = 24`).
const convert_sext_word_src =
    \\int f(void) { short s = -1000; int i = s; return i; }
    \\int main(void) { return f(); }
;

/// NARROWING: casting a wider `int` down to `char` keeps the low bits verbatim, with NO
/// sign/zero-extension at all. `x = 300` narrows to `c = 300 & 0xff = 44` (already in the
/// signed `char` range, so it reads back unchanged when widened again to `int`).
const convert_narrow_src =
    \\int f(void) { int x = 300; char c = (char)x; return c; }
    \\int main(void) { return f(); }
;

/// Compile `source` for i386, link it with the raw-syscall `_start`, run it under `qemu`, and
/// require the exit code to be `want`. It checks both vcc's own compile AND (when `zig cc` is
/// available) the real cdecl oracle, so a wrong extension choice cannot pass by accident.
fn expectConvertCase(allocator: std.mem.Allocator, io: std.Io, qemu: []const u8, source: []const u8, want: u8) !void {
    const main_obj = try vccObj(allocator, .x86, source);
    defer allocator.free(main_obj);
    const start_obj = try startRawObjX86(allocator);
    defer allocator.free(start_obj);

    const got = try linkRunStaticX86_32(allocator, io, qemu, start_obj, main_obj);
    try std.testing.expectEqual(want, got);

    if (try oracleExitX86_32(allocator, io, qemu, source)) |oracle| {
        try std.testing.expectEqual(want, oracle);
    }
}

test "i386: .convert sign-extends a negative signed char widened to int (movsx byte)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);
    try expectConvertCase(allocator, io, qemu, convert_sext_byte_src, 251);
}

test "i386: .convert zero-extends an unsigned char widened to int (movzx byte)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);
    try expectConvertCase(allocator, io, qemu, convert_zext_byte_src, 200);
}

test "i386: .convert sign-extends a negative short widened to int (movsx word)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);
    try expectConvertCase(allocator, io, qemu, convert_sext_word_src, 24);
}

test "i386: .convert narrows int to char by keeping the low bits (plain move)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);
    try expectConvertCase(allocator, io, qemu, convert_narrow_src, 44);
}

// A `.convert` touching a 64-bit width still fails closed: x86-32 has no 64-bit register, and
// this backend has no multi-register wide-int support, so `long long` reaching a convert must
// be rejected rather than silently mislowered into a 32-bit truncation.
test "i386: a 64-bit-wide .convert still fails closed (no multi-register wide-int support)" {
    const allocator = std.testing.allocator;
    const src =
        \\int f(void) { long long x = 5; int y = (int)x; return y; }
    ;
    try std.testing.expectError(error.Unsupported, vccObj(allocator, .x86, src));
}

// ============================================================================================
// This section is the ABI-compat proof: our own `va_start`-built `va_list` is handed to the
// REAL glibc `vsnprintf` on every arch. This is the check that justifies the ABI-exact design.
// A wrong `va_list` byte layout, or a wrong argument-passing rule for the `va_list` value
// itself, fails HERE even though every self-contained `sum`/`favg` test above passed (those
// never left our own code).
//
// The program (per arch, compiled with `vccObjForTarget` so `__builtin_va_list` takes the TARGET
// arch's ABI shape, NOT the host's):
//
//     int vsnprintf(char*, unsigned long, const char*, __builtin_va_list);
//     int myprintf(char *buf, const char *fmt, ...) {
//         __builtin_va_list ap; __builtin_va_start(ap, fmt);
//         int r = vsnprintf(buf, 64, fmt, ap);
//         __builtin_va_end(ap); return r;
//     }
//     int main(void){ char b[64]; myprintf(b, <fmt>, <args>); return <buffer check> ? 42 : 1; }
//
// `myprintf` is a vcc variadic DEFINITION: it builds `ap` with `va_start`, then passes `ap` to the
// non-variadic `vsnprintf` prototype. How `ap` reaches `vsnprintf` is the crux, and it differs by
// ABI. All four give glibc the SAME thing (a pointer to the walker or register-save block):
//   * x86_64 / aarch64: `va_list` is `__va_list_tag[1]` (array-of-1). Passing `ap` DECAYS to a
//     pointer to the 24B/32B struct (C array-decay). AAPCS64 passes a >16B composite indirectly
//     (by pointer) anyway, so this is exactly what glibc's aarch64 `vsnprintf` reads.
//   * riscv64 / i386: `va_list` is a scalar pointer (`void*` / `char*`). Passing `ap` passes the
//     pointer VALUE (the walker), NOT its address.
// The `vsnprintf` 4th parameter `__builtin_va_list` likewise decays (array arches) or stays scalar
// (pointer arches), so the prototype parses and the argument types line up on both sides.
//
// Link real `libc.so.6` for `vsnprintf`, run under the real glibc `ld.so` (aarch64 natively,
// x86_64/riscv64 under qemu). This is the SAME harness the printf tests above use. i386 is NOT
// run here. `compileForTarget(.x86)` switches the whole frontend to ILP32 and hits pre-existing
// x86-32 gaps: a Wimmer 5+-live-arg spill cap, and no float-value support (the `.convert` isel
// gap closed elsewhere in this file no longer applies here). i386 also could not reach real
// glibc in the printf tests above (frame alignment, no `__libc_start_main`). See the skip test
// below.

/// The vcc source for this proof. `main_body` is spliced in as `main`'s body so the int and
/// float cases share the `vsnprintf` prototype + `myprintf` definition verbatim. The output
/// buffer `b` is a FILE-SCOPE array, not a `main` local. A local `char b[64]` needs an
/// array-typed stack alloca, which the riscv64 isel only sizes inside a variadic function (a
/// pre-existing gap unrelated to variadic support. `main` is not variadic). A global buffer
/// routes through the data path instead and is irrelevant to the va_list ABI this test proves
/// (glibc writes into `b` either way).
fn vsnprintfProgram(comptime main_body: []const u8) []const u8 {
    return "int vsnprintf(char*, unsigned long, const char*, __builtin_va_list);\n" ++
        "char b[64];\n" ++
        "int myprintf(char *buf, const char *fmt, ...) {\n" ++
        "    __builtin_va_list ap;\n" ++
        "    __builtin_va_start(ap, fmt);\n" ++
        "    int r = vsnprintf(buf, 64, fmt, ap);\n" ++
        "    __builtin_va_end(ap);\n" ++
        "    return r;\n" ++
        "}\n" ++
        "int main(void) {\n" ++ main_body ++ "\n}\n";
}

/// The int/string case: `myprintf(b, "%d-%s-%d", 1, "x", 2)` fills `b` with `"1-x-2"`, so bytes 0,
/// 2, 4 are `'1'`, `'x'`, `'2'`. Returns 42 only if the buffer matches (a byte-exact ABI proof).
const vsnprintf_int_src = vsnprintfProgram(
    \\    myprintf(b, "%d-%s-%d", 1, "x", 2);
    \\    return (b[0] == '1' && b[2] == 'x' && b[4] == '2') ? 42 : 1;
);

/// The float case: `myprintf(b, "%.1f", 2.5)` fills `b` with `"2.5"`. The `2.5` double literal is a
/// variadic argument (default-arg-promoted to `double`), so this proves the fp side of the va_list
/// too. Returns 42 only if bytes 0, 1, 2 are `'2'`, `'.'`, `'5'`.
const vsnprintf_float_src = vsnprintfProgram(
    \\    myprintf(b, "%.1f", 2.5);
    \\    return (b[0] == '2' && b[1] == '.' && b[2] == '5') ? 42 : 1;
);

/// Compile `source` for `case.arch` with the TARGET layout, link `main` + the hand `_start` against
/// the real `libc.so.6` (for `vsnprintf`), run the dynexe under that glibc `ld.so` (natively or via
/// qemu), and assert the child exits 42 (the buffer-check success code). Mirrors `runPrintfCase`,
/// but compiles for-target (so the va_list matches the target ABI) and produces no stdout. Skips
/// cleanly when the interpreter or qemu is absent.
fn runVsnprintfCase(allocator: std.mem.Allocator, io: std.Io, case: ArchCase, source: []const u8) !void {
    const interp = (try findFile(allocator, case.interp_soname)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = if (case.qemu) |q| (try findQemu(allocator, q)) orelse return error.SkipZigTest else null;
    defer if (qemu) |q| allocator.free(q);

    const libdir = std.fs.path.dirname(interp) orelse return error.SkipZigTest;
    const libc_path = try std.fs.path.join(allocator, &.{ libdir, "libc.so.6" });
    defer allocator.free(libc_path);
    const libc_bytes = std.Io.Dir.cwd().readFileAlloc(io, libc_path, allocator, .limited(128 * 1024 * 1024)) catch return error.SkipZigTest;
    defer allocator.free(libc_bytes);

    const main_obj = try vccObjForTarget(allocator, case.arch, source);
    defer allocator.free(main_obj);
    const start_obj = try startObj(allocator, case.arch);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libc_bytes } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libc.so.6"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe", .data = dynexe, .flags = .{ .permissions = .executable_file } });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", libdir);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    if (qemu) |q| try argv.append(allocator, q);
    try argv.append(allocator, "./dynexe");

    const run = std.process.run(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);

    switch (run.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code),
        else => return error.TestUnexpectedResult,
    }
}

test "our va_list into real glibc vsnprintf(\"%d-%s-%d\") exits 42 (aarch64/x86_64/riscv64)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // this suite runs on the aarch64 host
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for (aligned_cases) |case| try runVsnprintfCase(allocator, io, case, vsnprintf_int_src);
}

test "our va_list into real glibc vsnprintf(\"%.1f\") exits 42 (aarch64/x86_64/riscv64)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for (aligned_cases) |case| try runVsnprintfCase(allocator, io, case, vsnprintf_float_src);
}

// i386 CANNOT run this real-glibc proof. `compileForTarget(.x86)` switches the frontend to ILP32,
// exposing pre-existing x86-32 backend gaps unrelated to variadic support: a Wimmer
// 5+-live-argument spill cap, and no float-value support (the `.convert` isel gap closed
// elsewhere in this file no longer applies here). Real glibc `vsnprintf` is itself unreachable
// on i386 for the reasons documented on the printf and i386 sum tests above (SSE frame
// alignment, and no `__libc_start_main`). This is a documented skip, never a faked pass. The
// i386 va_list byte layout is instead proven by the self-contained i386 define-side tests
// above and the i386 custom-reader probe test. This test records the skip explicitly so the
// suite states it.
test "i386 real-glibc vsnprintf is a documented skip (ILP32 frontend gaps + no libc reach)" {
    return error.SkipZigTest;
}

// A variadic CALL-SITE through a function pointer (`call_indirect` with `is_variadic = true`)
// is NOT a supported shape. The x86_64/riscv64 `.call_indirect` isel arms lack the AL-count and
// anonymous-float-to-integer-register placement rules their DIRECT `.call` arms carry, so
// emitting one would silently miscompile rather than fail. Both arms now fail closed with
// `error.Unsupported` on `is_variadic`. This test proves the guard, not a supported feature
// (full variadic-through-function-pointer support is out of scope for now).
test "a variadic call through a function pointer fails closed on x86_64 and riscv64 (not miscompiled)" {
    const allocator = std.testing.allocator;
    const src =
        \\int f(int (*fp)(int, ...)) { return fp(1, 2); }
    ;
    try std.testing.expectError(error.Unsupported, vccObj(allocator, .x86_64, src));
    try std.testing.expectError(error.Unsupported, vccObj(allocator, .riscv64, src));
}

// The guard above must not over-reject. A NON-variadic call through a function pointer keeps
// compiling on both arches. The existing `external_linkage.zig` fn-pointer tests already cover
// this end to end. This test pins it at the same `vccObj` layer the guard test above uses.
test "a non-variadic call through a function pointer still compiles on x86_64 and riscv64" {
    const allocator = std.testing.allocator;
    const src =
        \\int f(int (*fp)(int, int)) { return fp(1, 2); }
    ;
    const obj64 = try vccObj(allocator, .x86_64, src);
    allocator.free(obj64);
    const objrv = try vccObj(allocator, .riscv64, src);
    allocator.free(objrv);
}

// ============================================================================================
// This test proves the WHOLE VCC stack composes on a REAL header. The header-parsing proof in
// tests/preproc_glibc.zig proved VCC PARSES real glibc `stdio.h`. This proves a program that
// `#include <stdio.h>` and calls `printf` COMPILES, LINKS against real glibc, and RUNS.
// `printf`'s prototype comes FROM the parsed header (a variadic declaration), so the call goes
// through the variadic path, now fed by the real header instead of a hand-written prototype.
// This is the end-to-end payoff of the whole compiler stack at once.
//
//     #include <stdio.h>
//     int main(void){ printf("hello\n"); return 0; }
//
// Compiled per target with `compileForTarget` (the arch's own layout, so `__builtin_va_list`
// and the type widths match) PLUS an `FsResolver` over that arch's glibc dev include dir and the
// `SystemPredef` (`__GNUC__=4.2`, LP64) the header-parsing proof uses. Linked against the real
// `libc.so.6`, run under the real glibc `ld.so` (aarch64 natively, x86_64/riscv64 under qemu).
// This is the same harness the printf tests above use. It asserts stdout is exactly "hello\n"
// and the exit code is 0 (`main`'s `return 0`, distinct from the earlier tests that return
// printf's char count).

/// Runs `sh -c script`, returns trimmed stdout or null (never an error, so a probe that finds
/// nothing cleanly skips). Mirrors `preproc_glibc.zig`'s `shOutput`.
fn shOutput(allocator: std.mem.Allocator, script: []const u8) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ "sh", "-c", script } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Locate the HOST (aarch64) glibc dev include dir (the one with a real `stdio.h`). Same search
/// order as `preproc_glibc.zig`'s `findHostGlibcIncludeDir`: `VCC_GLIBC_INCLUDE`, then real gcc's
/// own `-E -v` system-include list, then a non-cross `/nix/store/*-glibc-*-dev/include`, then the
/// one store path confirmed present when this was written. Null (clean skip) if none pans out.
fn findHostGlibcIncludeDir(allocator: std.mem.Allocator) !?[]u8 {
    const script =
        \\if [ -n "${VCC_GLIBC_INCLUDE:-}" ] && [ -f "$VCC_GLIBC_INCLUDE/stdio.h" ]; then
        \\  echo "$VCC_GLIBC_INCLUDE"; exit 0
        \\fi
        \\if command -v gcc >/dev/null 2>&1; then
        \\  d=$(echo | gcc -E -v -xc - 2>&1 | grep -E '^ .*/glibc-[0-9][^/]*-dev/include$' | tail -1 | sed -e 's/^ //')
        \\  if [ -n "$d" ] && [ -f "$d/stdio.h" ]; then echo "$d"; exit 0; fi
        \\fi
        \\for d in /nix/store/*-glibc-[0-9]*-dev/include; do
        \\  if [ -f "$d/stdio.h" ]; then echo "$d"; exit 0; fi
        \\done
        \\known=/nix/store/11azz9spqc81walgg0mq9wjxrgbi4xqz-glibc-2.42-61-dev/include
        \\if [ -f "$known/stdio.h" ]; then echo "$known"; exit 0; fi
        \\exit 1
    ;
    return shOutput(allocator, script);
}

/// Locate a CROSS glibc dev include dir for `triple`. Mirrors `preproc_glibc.zig`'s
/// `findCrossGlibcIncludeDir`. Null (clean skip) when this Nix profile has no such tree.
fn findCrossGlibcIncludeDir(allocator: std.mem.Allocator, triple: []const u8) !?[]u8 {
    const script = try std.fmt.allocPrint(allocator,
        \\for d in /nix/store/*-glibc-{s}-*-dev/include; do
        \\  if [ -f "$d/stdio.h" ]; then echo "$d"; exit 0; fi
        \\done
        \\exit 1
    , .{triple});
    defer allocator.free(script);
    return shOutput(allocator, script);
}

/// The glibc dev include dir for `arch`: the HOST tree for aarch64 (this build host), the cross
/// tree for the others. Null (clean skip) when absent.
fn findGlibcIncludeForArch(allocator: std.mem.Allocator, arch: ld.Arch) !?[]u8 {
    return switch (arch) {
        .aarch64 => findHostGlibcIncludeDir(allocator),
        .x86_64 => findCrossGlibcIncludeDir(allocator, "x86_64-unknown-linux-gnu"),
        .riscv64 => findCrossGlibcIncludeDir(allocator, "riscv64-unknown-linux-gnu"),
        .x86 => findCrossGlibcIncludeDir(allocator, "i686-unknown-linux-gnu"),
    };
}

/// The `SystemPredef` this proof uses: `__GNUC__=4.2` (so glibc's `__GNUC_PREREQ` gates pick
/// the GNU-extension branches its headers rely on) on an LP64 target. Matches
/// `preproc_glibc.zig`'s `systemPredef`.
fn capstoneSystemPredef(arch: cc.layout.Arch) cc.preproc.SystemPredef {
    return .{ .arch = arch, .gnuc_major = 4, .gnuc_minor = 2, .gnuc_patch = 0, .long_bits = 64, .ptr_bits = 64, .char_signed = false };
}

/// Compile `source` (which `#include`s real glibc headers) for `arch` into a relocatable object,
/// resolving those includes off `glibc_include` via an `FsResolver` and seeding the `SystemPredef`
/// so the `__GNUC__`-gated header branches expand. Uses the arch's OWN target layout, so the
/// `printf` call's variadic ABI matches the target. Caller owns the bytes.
fn vccObjWithHeaders(allocator: std.mem.Allocator, io: std.Io, arch: ld.Arch, source: []const u8, glibc_include: []const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const dirs = [_][]const u8{glibc_include};
    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, &dirs);
    const lay_arch = layoutForLdArch(arch);
    var mod = try cc.compileForTarget(allocator, source, cc.layout.forArch(lay_arch), .{
        .resolver = fsr.asResolver(),
        .system = capstoneSystemPredef(lay_arch),
    });
    defer mod.deinit(allocator);
    return emitObj(allocator, arch, &mod);
}

/// The printf-hello program (`printf` declared FROM the real header). C source, with the C-level
/// `\n` written as `\\n` so the compiler sees a two-char escape inside the string literal.
const printf_hello_src =
    "#include <stdio.h>\n" ++
    "int main(void){ printf(\"hello\\n\"); return 0; }\n";

/// Like `runPrintfCase`, but the source `#include <stdio.h>` (compiled with `vccObjWithHeaders`),
/// so `printf` comes from the REAL parsed header. Skips cleanly when the glibc dev headers, the
/// interpreter, or qemu are absent. Asserts stdout "hello\n" and exit 0.
fn runPrintfHelloCase(allocator: std.mem.Allocator, io: std.Io, case: ArchCase) !void {
    const glibc_include = (try findGlibcIncludeForArch(allocator, case.arch)) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    const interp = (try findFile(allocator, case.interp_soname)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = if (case.qemu) |q| (try findQemu(allocator, q)) orelse return error.SkipZigTest else null;
    defer if (qemu) |q| allocator.free(q);

    const libdir = std.fs.path.dirname(interp) orelse return error.SkipZigTest;
    const libc_path = try std.fs.path.join(allocator, &.{ libdir, "libc.so.6" });
    defer allocator.free(libc_path);
    const libc_bytes = std.Io.Dir.cwd().readFileAlloc(io, libc_path, allocator, .limited(128 * 1024 * 1024)) catch return error.SkipZigTest;
    defer allocator.free(libc_bytes);

    const main_obj = try vccObjWithHeaders(allocator, io, case.arch, printf_hello_src, glibc_include);
    defer allocator.free(main_obj);
    const start_obj = try startObj(allocator, case.arch);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libc_bytes } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libc.so.6"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe", .data = dynexe, .flags = .{ .permissions = .executable_file } });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", libdir);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    if (qemu) |q| try argv.append(allocator, q);
    try argv.append(allocator, "./dynexe");

    const run = std.process.run(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);

    switch (run.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 0), code),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqualStrings("hello\n", run.stdout);
}

// This test: `#include <stdio.h>` + `printf("hello\n")` compiles from the REAL header,
// links real glibc, and prints "hello\n". aarch64 (host) MUST run whenever the glibc dev tree is
// present. x86_64/riscv64 run under qemu when their cross glibc and qemu are present, else skip.
test "#include <stdio.h> + printf(\"hello\") runs through real glibc (aarch64/x86_64/riscv64)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // this suite runs on the aarch64 host
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Each arch skips INDEPENDENTLY (its cross glibc dev tree or qemu may be absent, e.g. no
    // x86_64 cross headers on a given profile), so one arch's absent tooling never masks
    // another's genuine run. A run that reaches the child asserts stdout/exit INSIDE
    // `runPrintfHelloCase`, so a real failure still propagates. The whole test skips only when NO
    // arch could run at all (glibc entirely absent).
    var any_ran = false;
    for (aligned_cases) |case| {
        runPrintfHelloCase(allocator, io, case) catch |e| switch (e) {
            error.SkipZigTest => continue,
            else => return e,
        };
        any_ran = true;
    }
    if (!any_ran) return error.SkipZigTest;
}

// ============================================================================================
// This test: `div_t div(int, int)`, a REAL glibc export that returns an 8-byte struct BY VALUE,
// proves the WHOLE struct-by-value RETURN ABI (classification through register placement)
// against glibc across the real toolchain boundary. Every struct-return test so far (elsewhere
// in this suite) calls a `mk`/`take` DEFINED in the same vcc translation unit, so a
// caller/callee mismatch could cancel out invisibly (both sides share one bug). This test calls
// a function vcc never compiled at all. The callee is real, gcc-compiled glibc, so only a
// byte-exact match with the SAME calling convention gcc itself emits can pass.
//
// Linux glibc's `div_t` is `struct { int quot; int rem; }` (8 bytes: `quot` at offset 0, `rem`
// at offset 4). It is ONE integer eightbyte, so the real ABI (SysV x86-64, AAPCS64, RISC-V
// lp64d, all agree here) returns it in a SINGLE integer return register (rax / x0 / a0), the
// whole 8 bytes packed verbatim: `quot`'s 4 bytes in the low word, `rem`'s 4 bytes in the high
// word. `div(17, 5)` computes `{ .quot = 3, .rem = 2 }`, so a correct read of both fields off a
// correctly-placed register gives `r.quot * 10 + r.rem == 32`. A wrong register, a wrong
// eightbyte width, or a swapped quot/rem byte order all give a DIFFERENT exit code. This test
// is a single-bit-precise ABI checksum, not just "it linked and ran".

/// Hand-declares `div_t`/`div` (self-contained, no header parse needed to prove the ABI).
/// `main` receives the struct-returning call's result the same way the struct-return tests
/// elsewhere in this suite do (`div_t r = div(17, 5);`), except `div` is an UNDEFINED symbol
/// here, resolved at link time against real `libc.so.6`. vcc never lowered a body for it.
const div_src =
    \\typedef struct { int quot; int rem; } div_t;
    \\div_t div(int numer, int denom);
    \\int main(void) { div_t r = div(17, 5); return r.quot * 10 + r.rem; }
;

/// `ldiv_t ldiv(long, long)` is glibc's 64-bit sibling: `struct { long quot; long rem; }`, 16
/// bytes, TWO integer eightbytes. It returns as a PAIR (rax:rdx / x0:x1 / a0:a1), not the single
/// register `div_t` uses. This proves the two-eightbyte integer struct-return path (the
/// `struct Q{long a;long b;}` shape used elsewhere in this suite) against real glibc too,
/// distinct from `div_t`'s one-register path. `ldiv(17, 5)` also gives `{ .quot = 3, .rem = 2
/// }`, so the same `*10 + rem == 32` check applies. The `long` arithmetic is narrowed to `int`
/// only at the very end (`main`'s return type), never inside a struct member access. The
/// pre-existing member-cast gap that other struct-by-value tests route around does not apply
/// here, since this is a plain local narrowing conversion.
const ldiv_src =
    \\typedef struct { long quot; long rem; } ldiv_t;
    \\ldiv_t ldiv(long numer, long denom);
    \\int main(void) { ldiv_t r = ldiv(17, 5); return (int)(r.quot * 10 + r.rem); }
;

/// Compile `source` for `case.arch` with the TARGET layout, link `main` + the hand `_start`
/// against the real `libc.so.6` (for `div`/`ldiv`), run the dynexe under that glibc `ld.so`
/// (natively or via qemu), and assert the child exits `want`. Mirrors `runVsnprintfCase`, the
/// SAME real-glibc harness, parameterized on the expected exit code so it serves both the
/// `div_t` and `ldiv_t` tests. Skips cleanly when the interpreter or qemu is absent.
fn runDivCase(allocator: std.mem.Allocator, io: std.Io, case: ArchCase, source: []const u8, want: u8) !void {
    const interp = (try findFile(allocator, case.interp_soname)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = if (case.qemu) |q| (try findQemu(allocator, q)) orelse return error.SkipZigTest else null;
    defer if (qemu) |q| allocator.free(q);

    const libdir = std.fs.path.dirname(interp) orelse return error.SkipZigTest;
    const libc_path = try std.fs.path.join(allocator, &.{ libdir, "libc.so.6" });
    defer allocator.free(libc_path);
    const libc_bytes = std.Io.Dir.cwd().readFileAlloc(io, libc_path, allocator, .limited(128 * 1024 * 1024)) catch return error.SkipZigTest;
    defer allocator.free(libc_bytes);

    const main_obj = try vccObjForTarget(allocator, case.arch, source);
    defer allocator.free(main_obj);
    const start_obj = try startObj(allocator, case.arch);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libc_bytes } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libc.so.6"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe", .data = dynexe, .flags = .{ .permissions = .executable_file } });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", libdir);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    if (qemu) |q| try argv.append(allocator, q);
    try argv.append(allocator, "./dynexe");

    const run = std.process.run(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);

    switch (run.term) {
        .exited => |code| try std.testing.expectEqual(want, code),
        else => return error.TestUnexpectedResult,
    }
}

test "real glibc div(17,5) struct-return (div_t, one integer eightbyte) exits 32 (aarch64/x86_64/riscv64)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // this suite runs on the aarch64 host
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Each arch skips INDEPENDENTLY (mirrors the printf-hello test above). A run that reaches
    // the child asserts the exit code INSIDE `runDivCase`, so a genuine ABI mismatch still fails
    // the test loudly. It never gets masked by another arch's absent tooling.
    var any_ran = false;
    for (aligned_cases) |case| {
        runDivCase(allocator, io, case, div_src, 32) catch |e| switch (e) {
            error.SkipZigTest => continue,
            else => return e,
        };
        any_ran = true;
    }
    if (!any_ran) return error.SkipZigTest;
}

test "real glibc ldiv(17,5) struct-return (ldiv_t, two integer eightbytes rax:rdx/x0:x1/a0:a1) exits 32 (aarch64/x86_64/riscv64)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var any_ran = false;
    for (aligned_cases) |case| {
        runDivCase(allocator, io, case, ldiv_src, 32) catch |e| switch (e) {
            error.SkipZigTest => continue,
            else => return e,
        };
        any_ran = true;
    }
    if (!any_ran) return error.SkipZigTest;
}

// i386's `div_t`/`ldiv_t` are also memory/eax-returned (i386's `.sret` convention makes every
// struct return, `div_t` included, go through a hidden pointer instead), but i386 EXECUTION of
// ANY struct-by-value program is blocked by the SAME pre-existing gap the other i386
// struct-by-value tests already document: a struct local is an `array{i64}` storage blob the
// integer-only 32-bit backend cannot size or load. This is a documented skip, not a faked pass.
// i386's struct-return SHAPE (the hidden pointer) is proven at the IR level by
// `external_linkage.zig`'s i386 sret test.
test "i386 real-glibc div_t is a documented skip (pre-existing struct-local gap)" {
    return error.SkipZigTest;
}
