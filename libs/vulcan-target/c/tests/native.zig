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

/// Whether the host `cc` accepts the `_Float128` type. Apple clang on aarch64-darwin does not
/// (that target has no 128-bit floating type at all: `long double` is 64-bit and there is no
/// `_Float128`/`__float128`), so the binary128 C-oracle test skips there rather than failing.
fn ccSupportsFloat128(io: std.Io, allocator: std.mem.Allocator) bool {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    tmp.dir.writeFile(io, .{ .sub_path = "probe.c", .data = "_Float128 x;\n" }) catch return false;
    const r = std.process.run(allocator, io, .{
        .argv = &.{ "cc", "-std=c99", "-w", "-fsyntax-only", "probe.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch return false;
    defer allocator.free(r.stdout);
    defer allocator.free(r.stderr);
    return r.term == .exited and r.term.exited == 0;
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
        // Print the exact bit pattern so float equality is not lossy through text. Width
        // depends on the float kind: an f16 result is 2 bytes, f32 is 4, f64 is 8.
        switch (func.types.type_kind(ret.?).float) {
            .f16 => try out.appendSlice(allocator, "    uint16_t bits; memcpy(&bits, &r, 2); printf(\"%u\\n\", (unsigned)bits);\n"),
            .f32 => try out.appendSlice(allocator, "    uint32_t bits; memcpy(&bits, &r, 4); printf(\"%u\\n\", bits);\n"),
            .f64 => try out.appendSlice(allocator, "    uint64_t bits; memcpy(&bits, &r, 8); printf(\"%llu\\n\", (unsigned long long)bits);\n"),
            // An f128 result is 16 bytes; print the two 64-bit halves, high first.
            .f128 => try out.appendSlice(allocator, "    unsigned long long hi; unsigned long long lo; memcpy(&hi, (char*)&r + 8, 8); memcpy(&lo, (char*)&r, 8); printf(\"%llx%016llx\\n\", hi, lo);\n"),
        }
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
            .ret => |r| if (r.count > 0) return func.valueType(r.values[0]),
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
        .float => |f| try out.appendSlice(allocator, switch (f) {
            .f16 => "_Float16",
            .f32 => "float",
            .f64 => "double",
            .f128 => "_Float128",
        }),
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

/// Like `runCInt`, but `func` returns f16: the program prints the raw 16-bit half bit
/// pattern (see `wrapProgram`'s per-kind bit-print), which we reinterpret so the comparison
/// is bit-exact rather than lossy through text.
fn runCF16(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const Arg) !f16 {
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try compileAndRun(io, allocator, program);
    defer allocator.free(stdout);
    const bits = try std.fmt.parseInt(u16, stdout, 10);
    return @bitCast(bits);
}

/// Like `runCF16`, but the function returns f128: the program prints the result's two 64-bit
/// halves as hex (high first, low zero-padded to 16 digits), which we reassemble into the raw
/// 128-bit pattern so the comparison is bit-exact.
fn runCQuad(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const Arg) !u128 {
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try compileAndRun(io, allocator, program);
    defer allocator.free(stdout);
    // "<hi:x><lo:016x>": the low 64 bits are always the last 16 hex digits (zero-padded); the
    // high 64 bits are whatever precedes them.
    if (stdout.len < 16) return error.BadOutput;
    const split = stdout.len - 16;
    const hi = try std.fmt.parseInt(u64, stdout[0..split], 16);
    const lo = try std.fmt.parseInt(u64, stdout[split..], 16);
    return (@as(u128, hi) << 64) | lo;
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
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(sum) });

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
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const p = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, a, p);
    const x = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = b } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(sum) });

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
    g.setTerminator(gb, .{ .ret = ir.function.Ret.one(gm) });

    // f(a, b) = g(a) + b
    var f = Function.init(allocator);
    defer f.deinit();
    const fi = try f.types.intern(i32_t_kind);
    const fb = try f.appendBlock();
    const fa = try f.appendBlockParam(fb, fi);
    const fbparam = try f.appendBlockParam(fb, fi);
    const called = try f.appendCall(fb, fi, "g", &.{fa});
    const fsum = try f.appendInst(fb, fi, .{ .arith = .{ .op = .add, .lhs = called, .rhs = fbparam } });
    f.setTerminator(fb, .{ .ret = ir.function.Ret.one(fsum) });

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
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(sum) });

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
    const ptr_t = try func.types.ptrGlobal();
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
    func.setTerminator(e, .{ .ret = ir.function.Ret.one(got) });

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
    const ptr_t = try func.types.ptrGlobal();
    const slice_t = try func.types.intern(.{ .slice = .{ .elem = i32_t } });

    // f(n) { int buf; []i32 s = { &buf, n }; return s.len; }
    const e = try func.appendBlock();
    const n = try func.appendBlockParam(e, i64_t);
    const buf = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const s = try func.appendStructNew(e, slice_t, &.{ buf, n });
    const len = try func.appendInst(e, i64_t, .{ .extract = .{ .aggregate = s, .index = 1 } });
    func.setTerminator(e, .{ .ret = ir.function.Ret.one(len) });

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
    const ptr_t = try func.types.ptrGlobal();

    // f() { return *(int*)&g; }  where g is defined in another translation unit.
    const e = try func.appendBlock();
    const gp = try func.appendGlobalAddr(e, ptr_t, "g_value");
    const v = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = gp } });
    func.setTerminator(e, .{ .ret = ir.function.Ret.one(v) });

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
    const ptr_t = try func.types.ptrGlobal();

    // f(x) { int(*fp)(int) = &triple; return fp(x); }  triple lives in another TU.
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const fp = try func.appendGlobalAddr(e, ptr_t, "triple");
    const r = try func.appendCallIndirect(e, i32_t, fp, &.{x});
    func.setTerminator(e, .{ .ret = ir.function.Ret.one(r) });

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
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(r) });

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

test "C backend: f16 multiply compiled+run with cc, bit-exact against Zig's own f16 multiply" {
    // f(a, b) = a * b, both f16. `_Float16 * _Float16` in the emitted C is what proves out the
    // whole f16 lowering end to end: real `cc` compiles it, real hardware runs it, and the
    // result is compared bit-for-bit (not just approximately) to Zig's own half-precision `*`,
    // which is the ground truth for "what should this half multiply produce".
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f16_t);
    const b = try func.appendBlockParam(entry, f16_t);
    const r = try func.appendInst(entry, f16_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(r) });

    const cases = [_]struct { a: f16, b: f16 }{
        .{ .a = 1.5, .b = 2.25 }, // exact in f16: a plain sanity case, no rounding involved
        // 1 + 2^-10: itself exact in f16, but its SQUARE is not representable in half, so the
        // multiply's result must round to nearest-even, not just widen the raw f32 product.
        .{ .a = 1.0009765625, .b = 1.0009765625 },
        .{ .a = -100.0, .b = 7.0 },
    };
    for (cases) |case| {
        const got = runCF16(std.testing.io, allocator, &func, &.{
            .{ .float = case.a },
            .{ .float = case.b },
        }) catch |err| switch (err) {
            error.NoCompiler => return error.SkipZigTest,
            else => return err,
        };
        const want: f16 = case.a * case.b; // Zig's own half-precision multiply: the oracle
        try std.testing.expectEqual(@as(u16, @bitCast(want)), @as(u16, @bitCast(got)));
    }
    // The rounding case actually rounds: the exact f32 product differs from the half result,
    // so this is really exercising round-to-nearest-even and not silently widening to f32.
    const exact_f32: f32 = @as(f32, cases[1].a) * @as(f32, cases[1].b);
    try std.testing.expect(@as(f32, cases[1].a * cases[1].b) != exact_f32);
}

test "C backend: i64->f128 multiply compiled+run with cc, bit-exact against Zig's own f128" {
    // f(a, b) = (_Float128)a * (_Float128)b, a and b i64. Real `cc` compiles the emitted
    // `_Float128` widen-and-multiply and its host libgcc/compiler-rt runs it; the 128-bit
    // result is compared bit-for-bit against Zig's own f128. This is the numeric ground truth
    // behind the machine backends, which lower the SAME widen and multiply to the SAME soft-fp
    // symbols (`__floatditf`, `__multf3`). Integer arguments carry their value exactly (unlike
    // the f32-carried float args), so the case below can force a product that only a real
    // binary128 keeps: two integers just under 2^41 whose product needs ~81 mantissa bits,
    // more than f64's 53 but well inside f128's 112.
    const allocator = std.testing.allocator;
    // Apple clang (aarch64-darwin) has no `_Float128`, so the emitted C would not compile there.
    // Skip cleanly when the host cc cannot accept the type.
    if (!ccSupportsFloat128(std.testing.io, allocator)) return error.SkipZigTest;
    var func = Function.init(allocator);
    defer func.deinit();

    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const f128_t = try func.types.intern(.{ .float = .f128 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);
    const fa = try func.appendInst(entry, f128_t, .{ .convert = .{ .value = a } });
    const fb = try func.appendInst(entry, f128_t, .{ .convert = .{ .value = b } });
    const r = try func.appendInst(entry, f128_t, .{ .arith = .{ .op = .mul, .lhs = fa, .rhs = fb } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(r) });

    const cases = [_]struct { a: i64, b: i64 }{
        .{ .a = 6, .b = 7 }, // tiny sanity case
        .{ .a = (1 << 40) + 1, .b = (1 << 40) + 1 }, // product = 2^80 + 2^41 + 1: needs > 53 mantissa bits
        .{ .a = -123456789, .b = 987654321 },
    };
    for (cases) |case| {
        const got = runCQuad(std.testing.io, allocator, &func, &.{
            .{ .int = case.a },
            .{ .int = case.b },
        }) catch |err| switch (err) {
            error.NoCompiler => return error.SkipZigTest,
            else => return err,
        };
        const want: f128 = @as(f128, @floatFromInt(case.a)) * @as(f128, @floatFromInt(case.b));
        try std.testing.expectEqual(@as(u128, @bitCast(want)), got);
    }
    // The middle case genuinely needs binary128: its f128 product differs from the f64 product
    // (a and b are exact in f64, but their product is not), so a backend computing in f64 would
    // fail the bit-exact check above.
    const af: f64 = @floatFromInt(cases[1].a);
    const bf: f64 = @floatFromInt(cases[1].b);
    const p128: f128 = @as(f128, @floatFromInt(cases[1].a)) * @as(f128, @floatFromInt(cases[1].b));
    try std.testing.expect(p128 != @as(f128, af * bf));
}
