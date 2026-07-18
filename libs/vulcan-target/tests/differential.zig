//! Three-way differential tests: compile the same GLSL function through the native JIT (host
//! machine code), the C backend (compiled with `cc`), and the JS backend (run with Node.js),
//! and require all three to agree. Three independent implementations with very different value
//! models (native ints, C ints, JS BigInt) is the strongest bug-catcher: a divergence in any
//! one backend surfaces without hand-computing an expected value.
//!
//! Skips gracefully when the host cannot JIT natively, or when `cc` / Node.js is unavailable.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const glsl = @import("vulcan-glsl");

const engine = @import("vulcan-wasm"); // the Wasm JIT engine (runs Wasm-target output in-process)
const native = target.native;
const c = target.c;
const js = target.js;
const wasm = target.wasm;

const F2 = *const fn (i32, i32) callconv(.c) i32;
const F1 = *const fn (i32) callconv(.c) i32;
const FF2 = *const fn (f32, f32) callconv(.c) f32;
const FF1 = *const fn (f32) callconv(.c) f32;

fn hostCanJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .riscv64 => true,
        else => false,
    };
}

/// Compile GLSL `src`, JIT its function `f` for the host, and call it with `args`.
fn runNative(allocator: std.mem.Allocator, src: []const u8, comptime Fn: type, args: anytype) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return @call(.auto, buf.entry(Fn, 0), args);
}

/// Compile GLSL `src` to C, build a driver that calls `f(args)` and prints the result, compile
/// with `cc`, run it, and return the printed integer. `error.NoCompiler` if `cc` is absent.
fn runC(io: std.Io, allocator: std.mem.Allocator, src: []const u8, args: []const i64) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try c.emitFunction(allocator, f, "f");
    defer allocator.free(body);

    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, "#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n#include <stdio.h>\n#include <math.h>\n");
    try prog.appendSlice(allocator, body);
    try prog.appendSlice(allocator, "\nint main(void){ printf(\"%lld\\n\", (long long)f(");
    for (args, 0..) |v, i| {
        if (i != 0) try prog.appendSlice(allocator, ", ");
        try prog.print(allocator, "(int32_t){d}", .{v});
    }
    try prog.appendSlice(allocator, ")); return 0; }\n");

    const out = try compileAndRunC(io, allocator, prog.items);
    defer allocator.free(out);
    return std.fmt.parseInt(i64, out, 10);
}

/// Compile GLSL `src` to JS, run it with Node.js calling `f(args)`, and return the printed
/// integer. `error.NoEngine` if no Node is found.
fn runJs(io: std.Io, allocator: std.mem.Allocator, src: []const u8, args: []const i64) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try js.emitFunction(allocator, f, "f");
    defer allocator.free(body);

    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, js.runtime_preamble);
    try prog.append(allocator, '\n');
    try prog.appendSlice(allocator, body);
    try prog.appendSlice(allocator, "\nconsole.log(f(");
    for (args, 0..) |v, i| {
        if (i != 0) try prog.appendSlice(allocator, ", ");
        try prog.print(allocator, "{d}n", .{v});
    }
    try prog.appendSlice(allocator, ").toString());\n");

    const out = try runNodeProgram(io, allocator, prog.items);
    defer allocator.free(out);
    return std.fmt.parseInt(i64, out, 10);
}

/// Compile GLSL `src` through the Wasm target, instantiate it with the Wasm JIT engine, and
/// call `f(args)` in-process. A completely separate codegen path (IR -> Wasm bytecode) executed
/// on a stack-machine runtime, so it exercises the most different backend of the four.
fn runWasm2(allocator: std.mem.Allocator, src: []const u8, a: i32, b: i32) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var wmod = wasm.link.Module.init(allocator);
    defer wmod.deinit();
    try wmod.addFunction("f", f);
    var obj = try wasm.object.writeModule(allocator, &wmod);
    defer obj.deinit(allocator);
    var inst = try engine.Instance.instantiate(allocator, obj.module, &.{});
    defer inst.deinit();
    return @as(i64, try inst.call2(i32, i32, i32, "f", a, b));
}

fn runWasm1(allocator: std.mem.Allocator, src: []const u8, a: i32) !i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var wmod = wasm.link.Module.init(allocator);
    defer wmod.deinit();
    try wmod.addFunction("f", f);
    var obj = try wasm.object.writeModule(allocator, &wmod);
    defer obj.deinit(allocator);
    var inst = try engine.Instance.instantiate(allocator, obj.module, &.{});
    defer inst.deinit();
    return @as(i64, try inst.call1(i32, i32, "f", a));
}

/// The f32 analogues: the engine's typed `call` is generic, and the aarch64 C ABI passes f32
/// in the same FP registers the JIT'd Wasm function uses, so f32 in/out works directly.
fn runWasmF2(allocator: std.mem.Allocator, src: []const u8, a: f32, b: f32) !u32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var wmod = wasm.link.Module.init(allocator);
    defer wmod.deinit();
    try wmod.addFunction("f", f);
    var obj = try wasm.object.writeModule(allocator, &wmod);
    defer obj.deinit(allocator);
    var inst = try engine.Instance.instantiate(allocator, obj.module, &.{});
    defer inst.deinit();
    return @bitCast(try inst.call2(f32, f32, f32, "f", a, b));
}

fn wasmBatchF2(allocator: std.mem.Allocator, src: []const u8, aa: []const f32, bb: []const f32, out: []u32) !void {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var wmod = wasm.link.Module.init(allocator);
    defer wmod.deinit();
    try wmod.addFunction("f", f);
    var obj = try wasm.object.writeModule(allocator, &wmod);
    defer obj.deinit(allocator);
    var inst = try engine.Instance.instantiate(allocator, obj.module, &.{});
    defer inst.deinit();
    for (aa, bb, 0..) |a, b, i| out[i] = @bitCast(try inst.call2(f32, f32, f32, "f", a, b));
}

/// Assert native == C == JS == Wasm for a two-int-argument function `f`. Skips if a tool is missing.
fn diff2(src: []const u8, a: i32, b: i32) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    if (!hostCanJit()) return error.SkipZigTest;

    const nres = try runNative(allocator, src, F2, .{ a, b });
    const cres = runC(io, allocator, src, &.{ a, b }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    const jres = runJs(io, allocator, src, &.{ a, b }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    const wres = try runWasm2(allocator, src, a, b);
    try std.testing.expectEqual(nres, cres);
    try std.testing.expectEqual(nres, jres);
    try std.testing.expectEqual(nres, wres);
}

/// Assert native == C == JS == Wasm for a one-int-argument function `f`.
fn diff1(src: []const u8, a: i32) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    if (!hostCanJit()) return error.SkipZigTest;

    const nres = try runNative(allocator, src, F1, .{a});
    const cres = runC(io, allocator, src, &.{a}) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    const jres = runJs(io, allocator, src, &.{a}) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    const wres = try runWasm1(allocator, src, a);
    try std.testing.expectEqual(nres, cres);
    try std.testing.expectEqual(nres, jres);
    try std.testing.expectEqual(nres, wres);
}

/// Run `src`'s `f` natively and return its f32 result. The float runners print the exact 32-bit
/// pattern so the cross-backend compare stays bit-exact.
fn runNativeF(allocator: std.mem.Allocator, src: []const u8, comptime Fn: type, args: anytype) !f32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return @call(.auto, buf.entry(Fn, 0), args);
}

fn runCF(io: std.Io, allocator: std.mem.Allocator, src: []const u8, args: []const f32) !f32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try c.emitFunction(allocator, f, "f");
    defer allocator.free(body);

    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, "#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n#include <stdio.h>\n#include <math.h>\n");
    try prog.appendSlice(allocator, body);
    try prog.appendSlice(allocator, "\nint main(void){ float r = f(");
    for (args, 0..) |v, i| {
        if (i != 0) try prog.appendSlice(allocator, ", ");
        try prog.print(allocator, "(float){e}", .{@as(f64, v)}); // exact f64 widening, legal C literal
    }
    try prog.appendSlice(allocator, "); uint32_t b; memcpy(&b, &r, 4); printf(\"%u\\n\", b); return 0; }\n");

    const out = try compileAndRunC(io, allocator, prog.items);
    defer allocator.free(out);
    return @bitCast(try std.fmt.parseInt(u32, out, 10));
}

fn runJsF(io: std.Io, allocator: std.mem.Allocator, src: []const u8, args: []const f32) !f32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try js.emitFunction(allocator, f, "f");
    defer allocator.free(body);

    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, js.runtime_preamble);
    try prog.append(allocator, '\n');
    try prog.appendSlice(allocator, body);
    try prog.appendSlice(allocator, "\nconst __r = f(");
    for (args, 0..) |v, i| {
        if (i != 0) try prog.appendSlice(allocator, ", ");
        try prog.print(allocator, "{e}", .{@as(f64, v)});
    }
    try prog.appendSlice(allocator, ");\nconst __b = new DataView(new ArrayBuffer(4)); __b.setFloat32(0, __r, true); console.log(__b.getUint32(0, true));\n");

    const out = try runNodeProgram(io, allocator, prog.items);
    defer allocator.free(out);
    return @bitCast(try std.fmt.parseInt(u32, out, 10));
}

/// Assert native == C == JS (bit-exact) for a two-float-argument f32 function. Use only ops
/// that are correctly-rounded on all three (arith/div/sqrt/floor/min/max), never transcendentals.
fn diffF2(src: []const u8, a: f32, b: f32) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    if (!hostCanJit()) return error.SkipZigTest;
    const nres = try runNativeF(allocator, src, FF2, .{ a, b });
    const cres = runCF(io, allocator, src, &.{ a, b }) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    const jres = runJsF(io, allocator, src, &.{ a, b }) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    const wbits = try runWasmF2(allocator, src, a, b);
    try std.testing.expectEqual(@as(u32, @bitCast(nres)), @as(u32, @bitCast(cres)));
    try std.testing.expectEqual(@as(u32, @bitCast(nres)), @as(u32, @bitCast(jres)));
    try std.testing.expectEqual(@as(u32, @bitCast(nres)), wbits);
}

fn diffF1(src: []const u8, a: f32) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    if (!hostCanJit()) return error.SkipZigTest;
    const nres = try runNativeF(allocator, src, FF1, .{a});
    const cres = runCF(io, allocator, src, &.{a}) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    const jres = runJsF(io, allocator, src, &.{a}) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expectEqual(@as(u32, @bitCast(nres)), @as(u32, @bitCast(cres)));
    try std.testing.expectEqual(@as(u32, @bitCast(nres)), @as(u32, @bitCast(jres)));
}

// Fuzzing bakes many random inputs into one C/JS program so each shader compiles once.
const fuzz_n = 64;

/// Run `f(a,b)` for every input pair through the Wasm target + engine, in-process. One
/// instantiate, then `fuzz_n` calls, so it stays fast like native.
fn wasmBatch2(allocator: std.mem.Allocator, src: []const u8, aa: []const i32, bb: []const i32, out: []i64) !void {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var wmod = wasm.link.Module.init(allocator);
    defer wmod.deinit();
    try wmod.addFunction("f", f);
    var obj = try wasm.object.writeModule(allocator, &wmod);
    defer obj.deinit(allocator);
    var inst = try engine.Instance.instantiate(allocator, obj.module, &.{});
    defer inst.deinit();
    for (aa, bb, 0..) |a, b, i| out[i] = try inst.call2(i32, i32, i32, "f", a, b);
}

/// Run `f(a,b)` for every input pair natively, in-process (fast, no subprocess).
fn nativeBatch2(allocator: std.mem.Allocator, src: []const u8, aa: []const i32, bb: []const i32, out: []i64) !void {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const fp = buf.entry(F2, 0);
    for (aa, bb, 0..) |a, b, i| out[i] = fp(a, b);
}

/// Emit a C program that computes `f(A[i], B[i])` for all baked inputs and prints each; run it
/// once and return the results. Caller owns the slice.
fn cBatch2(io: std.Io, allocator: std.mem.Allocator, src: []const u8, aa: []const i32, bb: []const i32) ![]i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try c.emitFunction(allocator, f, "f");
    defer allocator.free(body);

    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, "#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n#include <stdio.h>\n#include <math.h>\n");
    try prog.appendSlice(allocator, body);
    try emitIntArray(allocator, &prog, "A", aa);
    try emitIntArray(allocator, &prog, "B", bb);
    try prog.print(allocator, "\nint main(void){{ for (int i = 0; i < {d}; i++) printf(\"%lld\\n\", (long long)f(A[i], B[i])); return 0; }}\n", .{aa.len});

    const out = try compileAndRunC(io, allocator, prog.items);
    defer allocator.free(out);
    return parseLines(allocator, out);
}

/// The JS analogue: bake the inputs, print each result, run once.
fn jsBatch2(io: std.Io, allocator: std.mem.Allocator, src: []const u8, aa: []const i32, bb: []const i32) ![]i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try js.emitFunction(allocator, f, "f");
    defer allocator.free(body);

    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, js.runtime_preamble);
    try prog.append(allocator, '\n');
    try prog.appendSlice(allocator, body);
    try prog.appendSlice(allocator, "\nconst A = [");
    for (aa, 0..) |v, i| try prog.print(allocator, "{s}{d}n", .{ if (i != 0) @as([]const u8, ", ") else "", v });
    try prog.appendSlice(allocator, "];\nconst B = [");
    for (bb, 0..) |v, i| try prog.print(allocator, "{s}{d}n", .{ if (i != 0) @as([]const u8, ", ") else "", v });
    try prog.print(allocator, "];\nfor (let i = 0; i < {d}; i++) console.log(f(A[i], B[i]).toString());\n", .{aa.len});

    const out = try runNodeProgram(io, allocator, prog.items);
    defer allocator.free(out);
    return parseLines(allocator, out);
}

fn emitIntArray(allocator: std.mem.Allocator, prog: *std.ArrayList(u8), name: []const u8, vals: []const i32) !void {
    try prog.print(allocator, "\nint32_t {s}[] = {{", .{name});
    for (vals, 0..) |v, i| try prog.print(allocator, "{s}{d}", .{ if (i != 0) @as([]const u8, ",") else "", v });
    try prog.appendSlice(allocator, "};");
}

fn emitFloatArray(allocator: std.mem.Allocator, prog: *std.ArrayList(u8), name: []const u8, vals: []const f32) !void {
    // Print the EXACT f64 widening of each f32 (not `{e}` of the f32, which is only the shortest
    // f32-round-trip decimal - that parses to the right f32 in C but to a slightly-off f64 in
    // JS, drifting the result by a ULP). The f64 widening parses exactly in both.
    try prog.print(allocator, "\nfloat {s}[] = {{", .{name});
    for (vals, 0..) |v, i| try prog.print(allocator, "{s}{e}", .{ if (i != 0) @as([]const u8, ",") else "", @as(f64, v) });
    try prog.appendSlice(allocator, "};");
}

fn nativeBatchF2(allocator: std.mem.Allocator, src: []const u8, aa: []const f32, bb: []const f32, out: []u32) !void {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const fp = buf.entry(FF2, 0);
    for (aa, bb, 0..) |a, b, i| out[i] = @bitCast(fp(a, b));
}

fn cBatchF2(io: std.Io, allocator: std.mem.Allocator, src: []const u8, aa: []const f32, bb: []const f32) ![]i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try c.emitFunction(allocator, f, "f");
    defer allocator.free(body);
    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, "#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n#include <stdio.h>\n#include <math.h>\n");
    try prog.appendSlice(allocator, body);
    try emitFloatArray(allocator, &prog, "A", aa);
    try emitFloatArray(allocator, &prog, "B", bb);
    try prog.print(allocator, "\nint main(void){{ for (int i = 0; i < {d}; i++) {{ float r = f(A[i], B[i]); uint32_t b; memcpy(&b, &r, 4); printf(\"%u\\n\", b); }} return 0; }}\n", .{aa.len});
    const out = try compileAndRunC(io, allocator, prog.items);
    defer allocator.free(out);
    return parseLines(allocator, out);
}

fn jsBatchF2(io: std.Io, allocator: std.mem.Allocator, src: []const u8, aa: []const f32, bb: []const f32) ![]i64 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find("f") orelse return error.MissingFunction;
    const body = try js.emitFunction(allocator, f, "f");
    defer allocator.free(body);
    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(allocator);
    try prog.appendSlice(allocator, js.runtime_preamble);
    try prog.append(allocator, '\n');
    try prog.appendSlice(allocator, body);
    try prog.appendSlice(allocator, "\nconst A = [");
    for (aa, 0..) |v, i| try prog.print(allocator, "{s}{e}", .{ if (i != 0) @as([]const u8, ", ") else "", @as(f64, v) });
    try prog.appendSlice(allocator, "];\nconst B = [");
    for (bb, 0..) |v, i| try prog.print(allocator, "{s}{e}", .{ if (i != 0) @as([]const u8, ", ") else "", @as(f64, v) });
    // NB: `__bv`, not `__dv` - the runtime preamble already declares a `const __dv`.
    try prog.print(allocator, "];\nconst __bv = new DataView(new ArrayBuffer(4));\nfor (let i = 0; i < {d}; i++) {{ __bv.setFloat32(0, f(A[i], B[i]), true); console.log(__bv.getUint32(0, true)); }}\n", .{aa.len});
    const out = try runNodeProgram(io, allocator, prog.items);
    defer allocator.free(out);
    return parseLines(allocator, out);
}

/// Fuzz a two-float-arg f32 function bit-exactly. Inputs are finite and normal (a in
/// [-100,100], b in [0.5,100] so nothing produces NaN/inf and division never blows up), and the
/// shader must use only correctly-rounded ops (no transcendentals).
fn fuzzF2(src: []const u8, seed: u64) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    if (!hostCanJit()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var aa: [fuzz_n]f32 = undefined;
    var bb: [fuzz_n]f32 = undefined;
    for (0..fuzz_n) |i| {
        aa[i] = (rng.float(f32) * 2.0 - 1.0) * 100.0;
        bb[i] = rng.float(f32) * 99.5 + 0.5;
    }

    var nbits: [fuzz_n]u32 = undefined;
    try nativeBatchF2(allocator, src, &aa, &bb, &nbits);
    var wbits: [fuzz_n]u32 = undefined;
    try wasmBatchF2(allocator, src, &aa, &bb, &wbits);
    const cres = cBatchF2(io, allocator, src, &aa, &bb) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(cres);
    const jres = jsBatchF2(io, allocator, src, &aa, &bb) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(jres);

    for (0..fuzz_n) |i| {
        if (@as(i64, nbits[i]) != cres[i] or @as(i64, nbits[i]) != jres[i] or nbits[i] != wbits[i]) {
            std.debug.print("\nfuzzF divergence on f({e}, {e}): native=0x{x} C=0x{x} JS=0x{x} wasm=0x{x}\n", .{ aa[i], bb[i], nbits[i], cres[i], jres[i], wbits[i] });
            return error.FuzzDivergence;
        }
    }
}

fn parseLines(allocator: std.mem.Allocator, text: []const u8) ![]i64 {
    var list: std.ArrayList(i64) = .empty;
    errdefer list.deinit(allocator);
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \r\t");
        if (t.len == 0) continue;
        try list.append(allocator, try std.fmt.parseInt(i64, t, 10));
    }
    return list.toOwnedSlice(allocator);
}

/// Fuzz a two-int-arg function: `fuzz_n` random inputs through native, C, and JS, all must
/// agree. `b` is kept in [1, hi] so division/remainder shaders never divide by zero (and never
/// hit the INT_MIN/-1 overflow), and both stay in i32 range so nothing is target-defined.
fn fuzz2(src: []const u8, seed: u64) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    if (!hostCanJit()) return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var aa: [fuzz_n]i32 = undefined;
    var bb: [fuzz_n]i32 = undefined;
    for (0..fuzz_n) |i| {
        aa[i] = rng.intRangeAtMost(i32, -100000, 100000);
        bb[i] = rng.intRangeAtMost(i32, 1, 100000);
    }

    var nres: [fuzz_n]i64 = undefined;
    try nativeBatch2(allocator, src, &aa, &bb, &nres);
    var wres: [fuzz_n]i64 = undefined;
    try wasmBatch2(allocator, src, &aa, &bb, &wres);
    const cres = cBatch2(io, allocator, src, &aa, &bb) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(cres);
    const jres = jsBatch2(io, allocator, src, &aa, &bb) catch |err| switch (err) {
        error.NoEngine => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(jres);

    try std.testing.expectEqual(@as(usize, fuzz_n), cres.len);
    try std.testing.expectEqual(@as(usize, fuzz_n), jres.len);
    for (0..fuzz_n) |i| {
        if (nres[i] != cres[i] or nres[i] != jres[i] or nres[i] != wres[i]) {
            std.debug.print("\nfuzz divergence on f({d}, {d}): native={d} C={d} JS={d} wasm={d}\n", .{ aa[i], bb[i], nres[i], cres[i], jres[i], wres[i] });
            return error.FuzzDivergence;
        }
    }
}

fn compileAndRunC(io: std.Io, allocator: std.mem.Allocator, source: []const u8) ![]u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "prog.c", .data = source });
    const compiled = std.process.run(allocator, io, .{
        // `-ffp-contract=off`: the IR (and thus native + JS) does SEPARATE rounded float ops, so
        // C must not fuse `a*b+c` into an FMA (which rounds once, diverging by up to a ULP). The
        // float fuzzer caught exactly this.
        .argv = &.{ "cc", "-std=c99", "-O0", "-w", "-ffp-contract=off", "-o", "prog", "prog.c", "-lm" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoCompiler,
        else => return err,
    };
    defer allocator.free(compiled.stdout);
    defer allocator.free(compiled.stderr);
    if (compiled.term != .exited or compiled.term.exited != 0) return error.CompileFailed;
    const ran = try std.process.run(allocator, io, .{ .argv = &.{"./prog"}, .cwd = .{ .dir = tmp.dir } });
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);
    return allocator.dupe(u8, std.mem.trim(u8, ran.stdout, " \n\r\t"));
}

fn findNode(io: std.Io, allocator: std.mem.Allocator) ![]u8 {
    const script = "command -v node 2>/dev/null || ls /nix/store/*nodejs-slim-2*/bin/node 2>/dev/null | head -n1 || ls /nix/store/*nodejs-2*/bin/node 2>/dev/null | head -n1";
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

fn runNodeProgram(io: std.Io, allocator: std.mem.Allocator, source: []const u8) ![]u8 {
    const node = try findNode(io, allocator);
    defer allocator.free(node);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "prog.js", .data = source });
    const ran = std.process.run(allocator, io, .{ .argv = &.{ node, "prog.js" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.NoEngine,
        else => return err,
    };
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);
    if (ran.term != .exited or ran.term.exited != 0) return error.RunFailed;
    return allocator.dupe(u8, std.mem.trim(u8, ran.stdout, " \n\r\t"));
}

// The tests below sweep the GLSL feature surface and require native, C, and JS to all agree.
test "differential: integer arithmetic" {
    try diff2("int f(int a, int b) { return a * b + a - b; }", 6, 4);
    try diff2("int f(int a, int b) { return (a + b) * (a - b); }", 9, 5);
    try diff2("int f(int a, int b) { return a / b + a % b; }", 23, 4);
}

test "differential: bitwise and shifts" {
    try diff2("int f(int a, int b) { return (a & b) | (a ^ b); }", 12, 10);
    try diff2("int f(int a, int b) { return (a << 2) + (b >> 1); }", 5, 20);
}

test "differential: control flow" {
    try diff2("int f(int a, int b) { int m; if (a > b) { m = a; } else { m = b; } return m; }", 3, 9);
    try diff2("int f(int a, int b) { int s = 0; for (int i = 0; i < b; i = i + 1) { s = s + a; } return s; }", 5, 4);
    try diff2("int f(int a, int b) { return (a > b) ? a * 2 : b * 2; }", 3, 10);
}

test "differential: nested control flow" {
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
    try diff1(src, 5);
    try diff1(src, 8);
}

test "differential: builtins via integers" {
    try diff2("int f(int a, int b) { return max(a, b) - min(a, b); }", 7, 3);
    try diff1("int f(int a) { return abs(a); }", -42);
}

test "differential: float arithmetic (bit-exact)" {
    // Only correctly-rounded f32 ops, so native (hardware), C (float), and JS (fround) match
    // bit-for-bit. Values with non-terminating binary expansions exercise the rounding.
    try diffF2("float f(float a, float b) { return a * b + a; }", 1.5, 3.25);
    try diffF2("float f(float a, float b) { return a / b; }", 1.0, 3.0);
    try diffF2("float f(float a, float b) { return (a + b) * (a - b) / b; }", 7.1, 2.9);
    try diffF1("float f(float x) { return x * 0.1 + 0.2; }", 0.7);
}

test "fuzz differential: 64 random inputs, native == C == JS" {
    // Well-defined-for-all-i32 ops only: wrapping arithmetic, truncating division (b >= 1), and
    // branch/builtin selection. Random inputs catch overflow-wrap and edge cases fixed tests miss.
    try fuzz2("int f(int a, int b) { return a * b + a - b; }", 0x1234);
    try fuzz2("int f(int a, int b) { return a / b + a % b; }", 0x5678);
    try fuzz2("int f(int a, int b) { return (a > b) ? a * 2 : b * 2; }", 0x9abc);
    try fuzz2("int f(int a, int b) { return max(a, b) - min(a, b); }", 0xdef0);
    try fuzz2("int f(int a, int b) { int m = a; if (b > a) { m = b; } return m * m; }", 0x2468);
}

test "fuzz differential: 64 random float inputs, native == C == JS (bit-exact)" {
    // Correctly-rounded f32 ops over random finite/normal inputs. sqrt(a*a+b*b) is always a
    // non-negative finite argument. Empirically confirms the double-rounding (Figueroa) bit-
    // exactness of native (hardware) vs C (float) vs JS (fround) across a wide input space.
    //
    // `a * b + a - b` and `a * a + b * b` are each a single-use multiply feeding straight into
    // an add/sub - since Vulcan now permits fp-contraction, the aarch64 native backend fuses
    // exactly that shape into one fmadd (one rounding), which the unfused C/JS/Wasm references
    // never do. That divergence is intentional (see aarch64/isel.zig's fusesIntoNextArith and
    // its dedicated bit-exact-vs-@mulAdd tests), not a codegen bug, so it does not belong in
    // this bit-exact cross-backend contract. `p - p` / `bb - bb` is exactly 0.0 for every finite
    // input, so it changes no value; it exists purely to give the product a second use, which
    // is exactly the condition that keeps `fusesIntoNextArith` from firing.
    try fuzzF2("float f(float a, float b) { float p = a * b; return p + a - b + (p - p); }", 0x1357);
    try fuzzF2("float f(float a, float b) { return a / b; }", 0x2468);
    try fuzzF2("float f(float a, float b) { return (a + b) * (a - b); }", 0x369c);
    try fuzzF2("float f(float a, float b) { float aa = a * a; float bb = b * b; return sqrt(aa + bb) + (bb - bb); }", 0x48ad);
}

test "differential: float builtins and conversions (bit-exact)" {
    try diffF1("float f(float x) { return sqrt(x); }", 2.0); // correctly-rounded on all three
    try diffF1("float f(float x) { return floor(x) + ceil(x); }", 3.7);
    try diffF2("float f(float a, float b) { return min(a, b) + max(a, b); }", 4.5, 1.25);
    try diffF1("float f(float x) { return float(int(x)) + x; }", 5.9); // f32<->i32 round-trip
    try diffF2("float f(float a, float b) { return (a > b) ? a : b; }", 2.5, 9.5); // select
}
