//! Execution tests for the JS backend: emit JavaScript from Vulcan IR, run it with the
//! host's Node.js, and check the result. Where a function also runs on the aarch64 native
//! JIT / the C backend, the answers must agree, so a JS-vs-native divergence is caught.
//!
//! Node is resolved from PATH, and failing that from the Nix store, so the tests run in
//! this environment; if none is found they skip.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const glsl = @import("vulcan-glsl");
const opt = @import("vulcan-opt");

const js = target.js;
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

const Arg = union(enum) { int: i64, float: f32 };

/// Locate a Node.js binary: PATH first, then a Nix-store nodejs. Returns the path (caller
/// owns it) or `error.NoEngine` to skip.
fn findNode(io: std.Io, allocator: std.mem.Allocator) ![]u8 {
    const script =
        "command -v node 2>/dev/null || ls /nix/store/*nodejs-slim-2*/bin/node 2>/dev/null | head -n1 || ls /nix/store/*nodejs-2*/bin/node 2>/dev/null | head -n1";
    const res = std.process.run(allocator, io, .{ .argv = &.{ "sh", "-c", script } }) catch |err| switch (err) {
        error.FileNotFound => return error.NoEngine,
        else => return err,
    };
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    const path = std.mem.trim(u8, res.stdout, " \n\r\t");
    if (path.len == 0) return error.NoEngine;
    return allocator.dupe(u8, path);
}

/// Run a JS `program` with Node, returning its trimmed stdout (caller owns it).
fn runJs(io: std.Io, allocator: std.mem.Allocator, program: []const u8) ![]u8 {
    const node = try findNode(io, allocator);
    defer allocator.free(node);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "prog.js", .data = program });

    const ran = std.process.run(allocator, io, .{
        .argv = &.{ node, "prog.js" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoEngine,
        else => return err,
    };
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);
    if (ran.term != .exited or ran.term.exited != 0) {
        std.debug.print("node failed:\n{s}\n--- source ---\n{s}\n", .{ ran.stderr, program });
        return error.RunFailed;
    }
    return allocator.dupe(u8, std.mem.trim(u8, ran.stdout, " \n\r\t"));
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

/// Wrap `func` (named `f`) with the runtime and a driver that calls it with `args` and
/// prints the result: an integer decimal, a boolean 0/1, or a float's raw 32-bit pattern.
fn wrapProgram(allocator: std.mem.Allocator, func: *const Function, args: []const Arg) ![]u8 {
    const body = try js.emitFunction(allocator, func, "f");
    defer allocator.free(body);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, js.runtime_preamble);
    try out.append(allocator, '\n');
    try out.appendSlice(allocator, body);
    try out.appendSlice(allocator, "\nconst __r = f(");
    for (args, 0..) |arg, i| {
        if (i != 0) try out.appendSlice(allocator, ", ");
        switch (arg) {
            .int => |v| try out.print(allocator, "{d}n", .{v}),
            .float => |v| try out.print(allocator, "{e}", .{v}),
        }
    }
    try out.appendSlice(allocator, ");\n");

    const ret = returnType(func);
    if (ret != null and func.types.type_kind(ret.?) == .float and func.types.type_kind(ret.?).float == .f32) {
        // Print the exact 32-bit pattern so float equality survives the round-trip.
        try out.appendSlice(allocator, "const __bv = new DataView(new ArrayBuffer(4)); __bv.setFloat32(0, __r, true); console.log(__bv.getUint32(0, true));\n");
    } else if (ret != null and func.types.type_kind(ret.?) == .bool) {
        try out.appendSlice(allocator, "console.log(__r ? \"1\" : \"0\");\n");
    } else {
        try out.appendSlice(allocator, "console.log(__r.toString());\n");
    }
    return out.toOwnedSlice(allocator);
}

fn runJsInt(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const Arg) !i64 {
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try runJs(io, allocator, program);
    defer allocator.free(stdout);
    return std.fmt.parseInt(i64, stdout, 10);
}

fn runJsF32(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const Arg) !f32 {
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try runJs(io, allocator, program);
    defer allocator.free(stdout);
    return @bitCast(try std.fmt.parseInt(u32, stdout, 10));
}

/// Run and return an f16-returning function's result as an f64: `wrapProgram` has no f16
/// case, so it falls to the generic `console.log(__r.toString())` branch. A JS Number's
/// `toString()` is defined to print the shortest decimal that parses back to the exact same
/// double, so parsing it back here recovers the exact f16-as-double value with no precision
/// loss, unlike the f32 path which needs the raw-bits trick to survive Number's rounding.
fn runJsF16(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const Arg) !f64 {
    const program = try wrapProgram(allocator, func, args);
    defer allocator.free(program);
    const stdout = try runJs(io, allocator, program);
    defer allocator.free(stdout);
    return std.fmt.parseFloat(f64, stdout);
}

fn expectJsF16(func: *const Function, args: []const Arg, expected: f64) !void {
    const r = runJsF16(std.testing.io, std.testing.allocator, func, args) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    // Every value here is an exact f16 value widened to f64, so the comparison is exact, not
    // approximate: a mismatch means a rounding-mode divergence between Math.f16round and Zig's
    // f16, not a benign precision difference.
    try std.testing.expectEqual(expected, r);
}

// GLSL-driven runners: exercise the JS backend on real frontend-produced IR.

fn runGlslJsInt(io: std.Io, allocator: std.mem.Allocator, src: []const u8, nm: []const u8, args: []const Arg) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const func = module.find(nm) orelse return error.MissingFunction;
    return runJsInt(io, allocator, func, args);
}

fn runGlslJsF32(io: std.Io, allocator: std.mem.Allocator, src: []const u8, nm: []const u8, args: []const Arg) !f32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const func = module.find(nm) orelse return error.MissingFunction;
    return runJsF32(io, allocator, func, args);
}

fn expectGlslJsInt(src: []const u8, nm: []const u8, args: []const Arg, expected: i64) !void {
    const r = runGlslJsInt(std.testing.io, std.testing.allocator, src, nm, args) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(expected, r);
}

fn expectGlslJsF32(src: []const u8, nm: []const u8, args: []const Arg, expected: f32) !void {
    const r = runGlslJsF32(std.testing.io, std.testing.allocator, src, nm, args) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectApproxEqAbs(expected, r, 1e-5);
}

test "JS backend: add two ints and run" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const r = runJsInt(std.testing.io, allocator, &func, &.{ .{ .int = 20 }, .{ .int = 22 } }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 42), r);
}

test "JS backend: 32-bit integer wrapping" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    // a * a for a large a must wrap to 32 bits like the native backends, not grow unbounded.
    const sq = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = a } });
    func.setTerminator(entry, .{ .ret = sq });

    const r = runJsInt(std.testing.io, allocator, &func, &.{.{ .int = 100000 }}) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    // 100000*100000 = 10000000000; as i32 that wraps to 1410065408.
    try std.testing.expectEqual(@as(i64, 1410065408), r);
}

test "JS backend: GLSL integer arithmetic" {
    try expectGlslJsInt("int f(int a, int b) { return a * b + a - b; }", "f", &.{ .{ .int = 6 }, .{ .int = 4 } }, 26);
}

test "JS backend: GLSL if/else picks a branch" {
    const src = "int f(int a, int b) { int m; if (a > b) { m = a; } else { m = b; } return m; }";
    try expectGlslJsInt(src, "f", &.{ .{ .int = 3 }, .{ .int = 9 } }, 9);
}

test "JS backend: GLSL for-loop with a loop-carried phi" {
    const src = "int f(int a, int b) { int s = 0; for (int i = 0; i < b; i = i + 1) { s = s + a; } return s; }";
    try expectGlslJsInt(src, "f", &.{ .{ .int = 5 }, .{ .int = 4 } }, 20);
}

test "JS backend: GLSL ternary lowers to select" {
    try expectGlslJsInt("int f(int a, int b) { return (a > b) ? a * 2 : b * 2; }", "f", &.{ .{ .int = 3 }, .{ .int = 10 } }, 20);
}

test "JS backend: GLSL nested control flow" {
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
    try expectGlslJsInt(src, "f", &.{.{ .int = 5 }}, 8);
}

test "JS backend: GLSL scalar float arithmetic" {
    try expectGlslJsF32("float f(float x) { return x * 2.0 + 1.0; }", "f", &.{.{ .float = 20.0 }}, 41.0);
}

test "JS backend: GLSL float builtins" {
    try expectGlslJsF32("float f(float x) { return sqrt(x); }", "f", &.{.{ .float = 16.0 }}, 4.0);
    try expectGlslJsF32("float f(float x) { return floor(x); }", "f", &.{.{ .float = 3.7 }}, 3.0);
}

test "JS backend: GLSL int<->float conversions" {
    try expectGlslJsInt("int f(float x) { return int(x); }", "f", &.{.{ .float = 3.7 }}, 3);
    try expectGlslJsF32("float f(int a) { return float(a) * 0.5; }", "f", &.{.{ .int = 7 }}, 3.5);
}

// f16 differentials: emit an f16 kernel, run it on Node's `Math.f16round`, and require an
// exact match against Zig's own `@as(f16, ...)` (widened to f64 for the comparison, since
// every f16 value is exact in f64). These are the node-executed proof that removing the
// js.zig f16 gate did not just stop rejecting f16 but actually lowers it correctly.

test "JS backend: f16 multiply rounds a non-half-representable product" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const entry = try func.appendBlock();
    // Both operands are fconst (not block params), so the exact-half rounding of the input
    // literals happens on both the JS and the Zig side identically; only the product's
    // rounding is under test.
    const a = try func.appendInst(entry, f16_t, .{ .fconst = 1.1 });
    const b = try func.appendInst(entry, f16_t, .{ .fconst = 1.1 });
    const prod = try func.appendInst(entry, f16_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = prod });

    const ah: f16 = 1.1;
    const bh: f16 = 1.1;
    const expected: f16 = ah * bh; // Zig's own f16 multiply: the oracle.
    try expectJsF16(&func, &.{}, @as(f64, expected));
}

test "JS backend: f16 add" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const entry = try func.appendBlock();
    const a = try func.appendInst(entry, f16_t, .{ .fconst = 0.1 });
    const b = try func.appendInst(entry, f16_t, .{ .fconst = 0.2 });
    const sum = try func.appendInst(entry, f16_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const ah: f16 = 0.1;
    const bh: f16 = 0.2;
    const expected: f16 = ah + bh;
    try expectJsF16(&func, &.{}, @as(f64, expected));
}

test "JS backend: f32 -> f16 convert rounds a value f16 cannot hold exactly" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, f32_t);
    const h = try func.appendInst(entry, f16_t, .{ .convert = .{ .value = x } });
    func.setTerminator(entry, .{ .ret = h });

    const pi_f32: f32 = 3.14159274; // not exactly representable in f16
    const expected: f16 = @floatCast(pi_f32);
    try expectJsF16(&func, &.{.{ .float = pi_f32 }}, @as(f64, expected));
}

test "JS backend: int -> f16 convert rounds a value f16 cannot hold exactly" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const h = try func.appendInst(entry, f16_t, .{ .convert = .{ .value = n } });
    func.setTerminator(entry, .{ .ret = h });

    // 12345 needs 14 significant bits; f16 keeps only 11 (1 implicit + 10 explicit), so the
    // conversion must round, not truncate silently.
    const n_val: i64 = 12345;
    const expected: f16 = @floatFromInt(n_val);
    try expectJsF16(&func, &.{.{ .int = n_val }}, @as(f64, expected));
}

test "JS backend: alloca, store, load round-trip" {
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

    const r = runJsInt(std.testing.io, allocator, &func, &.{ .{ .int = 10 }, .{ .int = 5 } }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 15), r);
}

test "JS backend: array alloca with computed pointer store/load" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const arr_t = try func.types.intern(.{ .array = .{ .len = 8, .elem = i32_t } });

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

    const r = runJsInt(std.testing.io, allocator, &func, &.{.{ .int = 3 }}) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 31), r); // 3*10 + 1
}

test "JS backend: struct construction and extract" {
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
    const scaled = try func.appendArithImm(entry, i32_t, .mul, f0, 10);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = scaled, .rhs = f1 } });
    func.setTerminator(entry, .{ .ret = sum });

    const r = runJsInt(std.testing.io, allocator, &func, &.{ .{ .int = 4 }, .{ .int = 5 } }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 45), r);
}

test "JS backend: slice construction and length extract" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const slice_t = try func.types.intern(.{ .slice = .{ .elem = i32_t } });

    const e = try func.appendBlock();
    const n = try func.appendBlockParam(e, i64_t);
    const buf = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const s = try func.appendStructNew(e, slice_t, &.{ buf, n });
    const len = try func.appendInst(e, i64_t, .{ .extract = .{ .aggregate = s, .index = 1 } });
    func.setTerminator(e, .{ .ret = len });

    const r = runJsInt(std.testing.io, allocator, &func, &.{.{ .int = 42 }}) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(i64, 42), r);
}

test "JS backend: vector pack, element-wise add, and extract" {
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

    const got = runJsF32(std.testing.io, allocator, &func, &.{ .{ .float = 1.0 }, .{ .float = 2.0 }, .{ .float = 10.0 }, .{ .float = 20.0 } }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(f32, 33.0), got); // (1+10)+(2+20)
}

test "JS backend: auto-vectorized GLSL vec4 shader runs through JS" {
    const allocator = std.testing.allocator;
    const src = "float f(vec4 a, vec4 b) { vec4 c = a + b; return c.x + c.y + c.z + c.w; }";
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const func = module.findMut("f") orelse return error.MissingFunction;
    _ = try opt.vectorize.run(allocator, func);

    var has_vec = false;
    for (0..func.instCount()) |i| {
        const res = func.instResult(@enumFromInt(i)) orelse continue;
        if (func.types.type_kind(func.valueType(res)) == .vector) has_vec = true;
    }
    try std.testing.expect(has_vec);

    const got = runJsF32(std.testing.io, allocator, func, &.{
        .{ .float = 1.0 },  .{ .float = 2.0 },  .{ .float = 3.0 },  .{ .float = 4.0 },
        .{ .float = 10.0 }, .{ .float = 20.0 }, .{ .float = 30.0 }, .{ .float = 40.0 },
    }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(f32, 110.0), got);
}

test "JS backend: module with a cross-function call" {
    const allocator = std.testing.allocator;
    const i32_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var g = Function.init(allocator);
    defer g.deinit();
    const gi = try g.types.intern(i32_kind);
    const gb = try g.appendBlock();
    const gx = try g.appendBlockParam(gb, gi);
    const gm = try g.appendArithImm(gb, gi, .mul, gx, 3);
    g.setTerminator(gb, .{ .ret = gm });

    var f = Function.init(allocator);
    defer f.deinit();
    const fi = try f.types.intern(i32_kind);
    const fb = try f.appendBlock();
    const fa = try f.appendBlockParam(fb, fi);
    const fbp = try f.appendBlockParam(fb, fi);
    const called = try f.appendCall(fb, fi, "g", &.{fa});
    const fsum = try f.appendInst(fb, fi, .{ .arith = .{ .op = .add, .lhs = called, .rhs = fbp } });
    f.setTerminator(fb, .{ .ret = fsum });

    const module = try js.emitModule(allocator, &.{ .{ .name = "g", .func = &g }, .{ .name = "f", .func = &f } });
    defer allocator.free(module);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, module);
    try program.appendSlice(allocator, "\nconsole.log(f(4n, 5n).toString());\n");

    const stdout = runJs(std.testing.io, allocator, program.items) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    try std.testing.expectEqual(@as(i64, 17), try std.fmt.parseInt(i64, stdout, 10)); // g(4)+5
}

test "JS backend: call_indirect through a global function reference" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);

    // f(x) { fp = &triple; return fp(x); }  triple is defined by the driver.
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const fp = try func.appendGlobalAddr(e, ptr_t, "triple");
    const r = try func.appendCallIndirect(e, i32_t, fp, &.{x});
    func.setTerminator(e, .{ .ret = r });

    const body = try js.emitFunction(allocator, &func, "f");
    defer allocator.free(body);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, js.runtime_preamble);
    try program.appendSlice(allocator, "\nfunction triple(x){ return BigInt.asIntN(32, x * 3n); }\n");
    try program.appendSlice(allocator, body);
    try program.appendSlice(allocator, "\nconsole.log(f(5n).toString());\n");

    const stdout = runJs(std.testing.io, allocator, program.items) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    try std.testing.expectEqual(@as(i64, 15), try std.fmt.parseInt(i64, stdout, 10)); // triple(5)
}

test "JS backend: global_addr reads an external data global" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);

    // f() { return *(int*)&g_value; }  g_value is a byte offset the driver sets up.
    const e = try func.appendBlock();
    const gp = try func.appendGlobalAddr(e, ptr_t, "g_value");
    const v = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = gp } });
    func.setTerminator(e, .{ .ret = v });

    const body = try js.emitFunction(allocator, &func, "f");
    defer allocator.free(body);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, js.runtime_preamble);
    // Place 99 into the shared memory and bind g_value to its offset.
    try program.appendSlice(allocator, "\nglobalThis.g_value = __alloca(4); __dv.setInt32(Number(g_value), 99, true);\n");
    try program.appendSlice(allocator, body);
    try program.appendSlice(allocator, "\nconsole.log(f().toString());\n");

    const stdout = runJs(std.testing.io, allocator, program.items) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(stdout);
    try std.testing.expectEqual(@as(i64, 99), try std.fmt.parseInt(i64, stdout, 10));
}
