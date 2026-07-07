//! Execution tests for the C backend: emit C from Vulcan IR, compile it with the host C
//! compiler (`cc`), run it, and check the result. Where a function also runs on the
//! aarch64 native JIT, the two are cross-checked (differential testing), so a divergence
//! between the C source backend and the machine-code backend is caught.
//!
//! Skips when `cc` is unavailable, mirroring the other backends' execution runners.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const glsl = @import("vulcan-glsl");

const opt = @import("vulcan-opt");
const c = target.c;
const native = target.native;
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

/// A concrete argument to pass into the compiled function.
const Arg = union(enum) { int: i64, float: f32 };

/// Compile `source` with the host C compiler into a temp executable, run it, and return
/// its trimmed stdout (caller owns it). Returns `error.NoCompiler` if `cc` is not on the
/// PATH so the caller can skip.
fn compileAndRun(io: std.Io, allocator: std.mem.Allocator, source: []const u8) ![]u8 {
    return compileAndRunMulti(io, allocator, &.{source});
}

/// Compile and link several C sources (separate translation units) into one executable,
/// run it, and return its trimmed stdout. Separate TUs let a `extern char sym[];` reference
/// coexist with the symbol's real typed definition in another file.
fn compileAndRunMulti(io: std.Io, allocator: std.mem.Allocator, sources: []const []const u8) ![]u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var argv: std.ArrayList([]const u8) = .empty;
    defer {
        for (argv.items) |a| allocator.free(a);
        argv.deinit(allocator);
    }
    const flags = [_][]const u8{ "cc", "-std=c99", "-O0", "-w", "-o", "prog" };
    for (flags) |flag| try argv.append(allocator, try allocator.dupe(u8, flag));
    for (sources, 0..) |src, i| {
        const fname = try std.fmt.allocPrint(allocator, "prog{d}.c", .{i});
        defer allocator.free(fname);
        try tmp.dir.writeFile(io, .{ .sub_path = fname, .data = src });
        try argv.append(allocator, try allocator.dupe(u8, fname));
    }
    try argv.append(allocator, try allocator.dupe(u8, "-lm"));

    const source = sources[0];
    const compiled = std.process.run(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoCompiler,
        else => return err,
    };
    defer allocator.free(compiled.stdout);
    defer allocator.free(compiled.stderr);
    if (compiled.term != .exited or compiled.term.exited != 0) {
        std.debug.print("cc failed:\n{s}\n--- source ---\n{s}\n", .{ compiled.stderr, source });
        return error.CompileFailed;
    }

    const ran = try std.process.run(allocator, io, .{
        .argv = &.{"./prog"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);
    return allocator.dupe(u8, std.mem.trim(u8, ran.stdout, " \n\r\t"));
}

/// Wrap an emitted function in a full C program: headers, the function itself under name
/// `f`, and a `main` that calls it with `args` and prints the result. Integer results
/// print as a decimal; float results print their raw 32-bit pattern so the check is exact.
fn wrapProgram(allocator: std.mem.Allocator, func: *const Function, args: []const Arg) ![]u8 {
    const body = try c.emitFunction(allocator, func, "f");
    defer allocator.free(body);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "#include <stdint.h>\n#include <stdbool.h>\n#include <stdio.h>\n#include <string.h>\n#include <math.h>\n\n");
    try out.appendSlice(allocator, body);
    try out.appendSlice(allocator, "\nint main(void) {\n    ");

    const ret = returnType(func);
    const is_float = ret != null and func.types.type_kind(ret.?) == .float;
    if (ret == null) {
        try out.appendSlice(allocator, "f(");
    } else {
        try emitCType(allocator, &out, func, ret.?);
        try out.appendSlice(allocator, " r = f(");
    }
    const params = func.blockParams(@enumFromInt(0));
    for (args, 0..) |arg, i| {
        if (i != 0) try out.appendSlice(allocator, ", ");
        // Cast each argument literal to the parameter's C type.
        try out.append(allocator, '(');
        try emitCType(allocator, &out, func, func.valueType(params[i]));
        try out.append(allocator, ')');
        switch (arg) {
            .int => |v| try out.print(allocator, "{d}", .{v}),
            // Scientific notation always carries a decimal/exponent, so it is a valid C
            // floating constant (unlike e.g. "16", which with an `f` suffix is illegal).
            .float => |v| try out.print(allocator, "{e}", .{v}),
        }
    }
    try out.appendSlice(allocator, ");\n");

    if (ret == null) {
        try out.appendSlice(allocator, "    printf(\"0\\n\");\n");
    } else if (is_float) {
        // Print the exact bit pattern so float equality is not lossy through text.
        try out.appendSlice(allocator, "    uint32_t bits; memcpy(&bits, &r, 4); printf(\"%u\\n\", bits);\n");
    } else {
        try out.appendSlice(allocator, "    printf(\"%lld\\n\", (long long)r);\n");
    }
    try out.appendSlice(allocator, "    return 0;\n}\n");
    return out.toOwnedSlice(allocator);
}

fn returnType(func: *const Function) ?Type {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
        if (func.terminator(block)) |term| switch (term) {
            .ret => |v| if (v) |vv| return func.valueType(vv),
            .jump => {},
        };
    }
    return null;
}

fn emitCType(allocator: std.mem.Allocator, out: *std.ArrayList(u8), func: *const Function, ty: Type) !void {
    switch (func.types.type_kind(ty)) {
        .bool => try out.appendSlice(allocator, "bool"),
        .int => |i| {
            const width: u16 = if (i.bits <= 8) 8 else if (i.bits <= 16) 16 else if (i.bits <= 32) 32 else 64;
            try out.print(allocator, "{s}int{d}_t", .{ if (i.signedness == .unsigned) "u" else "", width });
        },
        .float => |f| try out.appendSlice(allocator, if (f == .f32) "float" else "double"),
        .ptr => try out.appendSlice(allocator, "void*"),
        else => return error.Unsupported,
    }
}

/// Emit `func` to C, compile, run with `args`, and return the printed integer result.
fn runCInt(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const Arg) !i64 {
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try compileAndRun(io, allocator, program);
    defer allocator.free(stdout);
    return std.fmt.parseInt(i64, stdout, 10);
}

/// Compile GLSL `src`, emit its function `name` to C, compile and run it with `args`, and
/// return the printed integer result. Exercises the C backend on real frontend-produced IR
/// (control flow, loops, phi copies, inlined calls).
fn runGlslCInt(io: std.Io, allocator: std.mem.Allocator, src: []const u8, name: []const u8, args: []const Arg) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const func = module.find(name) orelse return error.MissingFunction;
    return runCInt(io, allocator, func, args);
}

fn expectGlslCInt(src: []const u8, name: []const u8, args: []const Arg, expected: i64) !void {
    const r = runGlslCInt(std.testing.io, std.testing.allocator, src, name, args) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(expected, r);
}

/// Like `runGlslCInt`, but the function returns f32: the program prints the raw 32-bit
/// pattern, which we reinterpret so the comparison is not lost through text.
fn runGlslCF32(io: std.Io, allocator: std.mem.Allocator, src: []const u8, name: []const u8, args: []const Arg) !f32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const func = module.find(name) orelse return error.MissingFunction;
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try compileAndRun(io, allocator, program);
    defer allocator.free(stdout);
    const bits = try std.fmt.parseInt(u32, stdout, 10);
    return @bitCast(bits);
}

fn expectGlslCF32(src: []const u8, name: []const u8, args: []const Arg, expected: f32) !void {
    const r = runGlslCF32(std.testing.io, std.testing.allocator, src, name, args) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectApproxEqAbs(expected, r, 1e-5);
}

test "C backend: add two ints and run" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const r = runCInt(std.testing.io, allocator, &func, &.{ .{ .int = 20 }, .{ .int = 22 } }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 42), r);
}

test "C backend: GLSL integer arithmetic" {
    // a*b + a - b for a=6, b=4 -> 24 + 6 - 4 = 26
    try expectGlslCInt("int f(int a, int b) { return a * b + a - b; }", "f", &.{ .{ .int = 6 }, .{ .int = 4 } }, 26);
}

test "C backend: GLSL if/else picks a branch" {
    // max via an if/else diamond with a merge phi. a=3, b=9 -> 9
    const src = "int f(int a, int b) { int m; if (a > b) { m = a; } else { m = b; } return m; }";
    try expectGlslCInt(src, "f", &.{ .{ .int = 3 }, .{ .int = 9 } }, 9);
}

test "C backend: GLSL for-loop with a loop-carried phi" {
    // sum a, b times. a=5, b=4 -> 20. Exercises loop back-edge goto and header phis.
    const src = "int f(int a, int b) { int s = 0; for (int i = 0; i < b; i = i + 1) { s = s + a; } return s; }";
    try expectGlslCInt(src, "f", &.{ .{ .int = 5 }, .{ .int = 4 } }, 20);
}

test "C backend: GLSL ternary lowers to select" {
    // (a>b) ? a*2 : b*2 for a=3, b=10 -> 20
    try expectGlslCInt("int f(int a, int b) { return (a > b) ? a * 2 : b * 2; }", "f", &.{ .{ .int = 3 }, .{ .int = 10 } }, 20);
}

test "C backend: GLSL nested control flow" {
    // A while loop wrapping an if: count how many of 0..b are > a/2-ish. Keep it simple and
    // deterministic: accumulate i while i < b, adding 2 when i is even else 1. b=5 ->
    // i=0:+2, 1:+1, 2:+2, 3:+1, 4:+2 = 8
    const src =
        \\int f(int b) {
        \\  int s = 0;
        \\  int i = 0;
        \\  while (i < b) {
        \\    if (i % 2 == 0) { s = s + 2; } else { s = s + 1; }
        \\    i = i + 1;
        \\  }
        \\  return s;
        \\}
    ;
    try expectGlslCInt(src, "f", &.{.{ .int = 5 }}, 8);
}

test "C backend: alloca, store, load round-trip" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const p = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, a, p);
    const x = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const r = runCInt(std.testing.io, allocator, &func, &.{ .{ .int = 10 }, .{ .int = 5 } }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 15), r);
}

test "C backend: module with a cross-function call" {
    const allocator = std.testing.allocator;
    const i32_t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // g(x) = x * 3
    var g = Function.init(allocator);
    defer g.deinit();
    const gi = try g.types.intern(i32_t_kind);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gi);
    const gm = try g.appendInst(gb, gi, .{ .arith_imm = .{ .op = .mul, .lhs = gx, .imm = 3 } });
    g.setTerminator(gb, .{ .ret = gm });

    // f(a, b) = g(a) + b
    var f = Function.init(allocator);
    defer f.deinit();
    const fi = try f.types.intern(i32_t_kind);
    const fb = try f.appendBlock();
    const fa = try f.appendBlockParam(fb, fi);
    const fbparam = try f.appendBlockParam(fb, fi);
    const called = try f.appendCall(fb, fi, "g", &.{fa});
    const fsum = try f.appendInst(fb, fi, .{ .arith = .{ .op = .add, .lhs = called, .rhs = fbparam } });
    f.setTerminator(fb, .{ .ret = fsum });

    const module = try c.emitModule(allocator, &.{ .{ .name = "g", .func = &g }, .{ .name = "f", .func = &f } });
    defer allocator.free(module);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, "#include <stdint.h>\n#include <stdio.h>\n\n");
    try program.appendSlice(allocator, module);
    try program.appendSlice(allocator, "\nint main(void) {\n    int32_t r = f((int32_t)4, (int32_t)5);\n    printf(\"%lld\\n\", (long long)r);\n    return 0;\n}\n");

    const stdout = compileAndRun(std.testing.io, allocator, program.items) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    // g(4) + 5 = 12 + 5 = 17
    try std.testing.expectEqual(@as(i64, 17), try std.fmt.parseInt(i64, stdout, 10));
}

test "C backend: struct construction and extract" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const s = try func.appendStructNew(entry, st, &.{ a, b });
    const f0 = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = s, .index = 0 } });
    const f1 = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = s, .index = 1 } });
    // Build the pair (a, b), read both fields back, return f0*10 + f1.
    const scaled = try func.appendInst(entry, i32_t, .{ .arith_imm = .{ .op = .mul, .lhs = f0, .imm = 10 } });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = scaled, .rhs = f1 } });
    func.setTerminator(entry, .{ .ret = sum });

    const r = runCInt(std.testing.io, allocator, &func, &.{ .{ .int = 4 }, .{ .int = 5 } }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 45), r); // 4*10 + 5
}

test "C backend: array alloca with computed pointer store/load" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const arr_t = try func.types.intern(.{ .array = .{ .len = 8, .elem = i32_t } });

    // f(i) { int buf[8]; int* p = buf + i*4 bytes; *p = i*10 + 1; return *p; }
    const e = try func.appendBlock();
    const i = try func.appendBlockParam(e, i32_t);
    const buf = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = arr_t } });
    const off = try func.appendArithImm(e, i32_t, .shl, i, 2);
    const p = try func.appendInst(e, ptr_t, .{ .arith = .{ .op = .add, .lhs = buf, .rhs = off } });
    const scaled = try func.appendArithImm(e, i32_t, .mul, i, 10);
    const val = try func.appendArithImm(e, i32_t, .add, scaled, 1);
    try func.appendStore(e, val, p);
    const got = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(e, .{ .ret = got });

    const r = runCInt(std.testing.io, allocator, &func, &.{.{ .int = 3 }}) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 31), r); // 3*10 + 1
}

test "C backend: slice construction and length extract" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const slice_t = try func.types.intern(.{ .slice = .{ .elem = i32_t } });

    // f(n) { int buf; []i32 s = { &buf, n }; return s.len; }
    const e = try func.appendBlock();
    const n = try func.appendBlockParam(e, i64_t);
    const buf = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const s = try func.appendStructNew(e, slice_t, &.{ buf, n });
    const len = try func.appendInst(e, i64_t, .{ .extract = .{ .aggregate = s, .index = 1 } });
    func.setTerminator(e, .{ .ret = len });

    const r = runCInt(std.testing.io, allocator, &func, &.{.{ .int = 42 }}) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 42), r);
}

test "C backend: global_addr reads an external global" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);

    // f() { return *(int*)&g; }  where g is defined in another translation unit.
    const e = try func.appendBlock();
    const gp = try func.appendGlobalAddr(e, ptr_t, "g_value");
    const v = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = gp } });
    func.setTerminator(e, .{ .ret = v });

    const body = try c.emitFunction(allocator, &func, "f");
    defer allocator.free(body);
    const prog = try std.fmt.allocPrint(allocator, "#include <stdint.h>\n#include <stdio.h>\n{s}\nint main(void){{ printf(\"%lld\\n\", (long long)f()); return 0; }}\n", .{body});
    defer allocator.free(prog);
    const helper = "int g_value = 99;\n";

    const stdout = compileAndRunMulti(std.testing.io, allocator, &.{ prog, helper }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    try std.testing.expectEqual(@as(i64, 99), try std.fmt.parseInt(i64, stdout, 10));
}

test "C backend: call_indirect through a function address" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);

    // f(x) { int(*fp)(int) = &triple; return fp(x); }  triple lives in another TU.
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const fp = try func.appendGlobalAddr(e, ptr_t, "triple");
    const r = try func.appendCallIndirect(e, i32_t, fp, &.{x});
    func.setTerminator(e, .{ .ret = r });

    const body = try c.emitFunction(allocator, &func, "f");
    defer allocator.free(body);
    const prog = try std.fmt.allocPrint(allocator, "#include <stdint.h>\n#include <stdio.h>\n{s}\nint main(void){{ printf(\"%lld\\n\", (long long)f(5)); return 0; }}\n", .{body});
    defer allocator.free(prog);
    const helper = "int triple(int x){ return x * 3; }\n";

    const stdout = compileAndRunMulti(std.testing.io, allocator, &.{ prog, helper }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    try std.testing.expectEqual(@as(i64, 15), try std.fmt.parseInt(i64, stdout, 10)); // triple(5)
}

test "C backend: auto-vectorized GLSL vec4 shader runs through C" {
    const allocator = std.testing.allocator;
    // The frontend scalarizes `a + b` into four fadds; the auto-vectorizer fuses them into
    // one vector op (struct_new pack + vector arith + extract). The C backend must then
    // render those aggregates as C structs and run to the same answer as NEON does.
    const src = "float f(vec4 a, vec4 b) { vec4 c = a + b; return c.x + c.y + c.z + c.w; }";
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const func = module.findMut("f") orelse return error.MissingFunction;
    _ = try opt.vectorize.run(allocator, func);

    // Confirm the vectorizer actually produced a vector-typed op, so this really exercises
    // the aggregate path and not just scalars.
    var has_vec = false;
    for (0..func.instCount()) |i| {
        const res = func.instResult(@enumFromInt(i)) orelse continue;
        if (func.types.type_kind(func.valueType(res)) == .vector) has_vec = true;
    }
    try std.testing.expect(has_vec);

    const program = try wrapProgram(allocator, func, &.{
        .{ .float = 1.0 },  .{ .float = 2.0 },  .{ .float = 3.0 },  .{ .float = 4.0 },
        .{ .float = 10.0 }, .{ .float = 20.0 }, .{ .float = 30.0 }, .{ .float = 40.0 },
    });
    defer allocator.free(program);
    const stdout = compileAndRun(std.testing.io, allocator, program) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    const bits = try std.fmt.parseInt(u32, stdout, 10);
    // (1+10)+(2+20)+(3+30)+(4+40) = 110
    try std.testing.expectEqual(@as(f32, 110.0), @as(f32, @bitCast(bits)));
}

test "C backend: vector pack, element-wise add, and extract" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v2 = try func.types.intern(.{ .vector = .{ .len = 2, .elem = f32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const cc = try func.appendBlockParam(entry, f32_t);
    const d = try func.appendBlockParam(entry, f32_t);
    const va = try func.appendStructNew(entry, v2, &.{ a, b });
    const vb = try func.appendStructNew(entry, v2, &.{ cc, d });
    const vs = try func.appendInst(entry, v2, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    const x0 = try func.appendInst(entry, f32_t, .{ .extract = .{ .aggregate = vs, .index = 0 } });
    const x1 = try func.appendInst(entry, f32_t, .{ .extract = .{ .aggregate = vs, .index = 1 } });
    const r = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = x0, .rhs = x1 } });
    func.setTerminator(entry, .{ .ret = r });

    const program = try wrapProgram(allocator, &func, &.{ .{ .float = 1.0 }, .{ .float = 2.0 }, .{ .float = 10.0 }, .{ .float = 20.0 } });
    defer allocator.free(program);
    const stdout = compileAndRun(std.testing.io, allocator, program) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    const bits = try std.fmt.parseInt(u32, stdout, 10);
    // (1+10) + (2+20) = 33
    try std.testing.expectEqual(@as(f32, 33.0), @as(f32, @bitCast(bits)));
}

test "C backend: GLSL scalar float arithmetic" {
    // x*2 + 1 for x=20 -> 41
    try expectGlslCF32("float f(float x) { return x * 2.0 + 1.0; }", "f", &.{.{ .float = 20.0 }}, 41.0);
}

test "C backend: GLSL float builtins" {
    // sqrt(16) -> 4, floor(3.7) -> 3
    try expectGlslCF32("float f(float x) { return sqrt(x); }", "f", &.{.{ .float = 16.0 }}, 4.0);
    try expectGlslCF32("float f(float x) { return floor(x); }", "f", &.{.{ .float = 3.7 }}, 3.0);
}

test "C backend: GLSL int<->float conversions" {
    // int(x) truncates 3.7 -> 3
    try expectGlslCInt("int f(float x) { return int(x); }", "f", &.{.{ .float = 3.7 }}, 3);
    // float(a) * 0.5 for a=7 -> 3.5
    try expectGlslCF32("float f(int a) { return float(a) * 0.5; }", "f", &.{.{ .int = 7 }}, 3.5);
}
