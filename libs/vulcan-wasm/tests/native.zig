//! Execution tests for the Wasm frontend: assemble a Wasm module in-process, lower it
//! to Vulcan IR, JIT it for the host via `vulcan-target.native`, run it. Self-contained
//! (no external wasm tooling), runs natively in-process.

const std = @import("std");
const wasm = @import("vulcan-wasm");
const native = @import("vulcan-target").native;
const ir = @import("vulcan-ir");
const wtarget = @import("vulcan-target").wasm;
const glsl = @import("vulcan-glsl");

const valtype_i32: u8 = 0x7F;
const valtype_i64: u8 = 0x7E;

/// Load `bytes`, JIT the export `name`, and call it with `args` (0-2 of them).
fn runi(allocator: std.mem.Allocator, bytes: []const u8, name: []const u8, args: []const i64) !i64 {
    var module = try wasm.load(allocator, bytes);
    defer module.deinit(allocator);
    const f = module.find(name) orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return switch (args.len) {
        0 => buf.entry(*const fn () callconv(.c) i64, 0)(),
        1 => buf.entry(*const fn (i64) callconv(.c) i64, 0)(args[0]),
        2 => buf.entry(*const fn (i64, i64) callconv(.c) i64, 0)(args[0], args[1]),
        else => error.Unsupported,
    };
}

/// Unsigned LEB128 append.
fn lebU(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: usize) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(v & 0x7F);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try out.append(allocator, byte);
        if (v == 0) break;
    }
}

/// Append a section: id, then the LEB length of `content`, then `content`.
fn section(out: *std.ArrayList(u8), allocator: std.mem.Allocator, id: u8, content: []const u8) !void {
    try out.append(allocator, id);
    try lebU(out, allocator, content.len);
    try out.appendSlice(allocator, content);
}

/// Build a one-function module exporting `name`, with the given param/result
/// valtypes and instruction `body` (the trailing `end` is added here). No extra
/// locals. Caller owns the bytes.
fn oneFunctionModule(
    allocator: std.mem.Allocator,
    params: []const u8,
    results: []const u8,
    body: []const u8,
    name: []const u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });

    // Type section: one functype.
    var ty: std.ArrayList(u8) = .empty;
    defer ty.deinit(allocator);
    try lebU(&ty, allocator, 1);
    try ty.append(allocator, 0x60);
    try lebU(&ty, allocator, params.len);
    try ty.appendSlice(allocator, params);
    try lebU(&ty, allocator, results.len);
    try ty.appendSlice(allocator, results);
    try section(&out, allocator, 1, ty.items);

    // Function section: function 0 has type 0.
    var fn_sec: std.ArrayList(u8) = .empty;
    defer fn_sec.deinit(allocator);
    try lebU(&fn_sec, allocator, 1);
    try lebU(&fn_sec, allocator, 0);
    try section(&out, allocator, 3, fn_sec.items);

    // Export section: name -> func 0.
    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    try lebU(&ex, allocator, 1);
    try lebU(&ex, allocator, name.len);
    try ex.appendSlice(allocator, name);
    try ex.append(allocator, 0x00); // kind = func
    try lebU(&ex, allocator, 0);
    try section(&out, allocator, 7, ex.items);

    // Code section: one body with zero local groups, then the instructions + end.
    var body_full: std.ArrayList(u8) = .empty;
    defer body_full.deinit(allocator);
    try lebU(&body_full, allocator, 0); // local groups
    try body_full.appendSlice(allocator, body);
    try body_full.append(allocator, 0x0B); // end

    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, 1); // one function body
    try lebU(&code, allocator, body_full.items.len);
    try code.appendSlice(allocator, body_full.items);
    try section(&out, allocator, 10, code.items);

    return out.toOwnedSlice(allocator);
}

test "wasm: lower and JIT a numeric function (x*2 + 1)" {
    const allocator = std.testing.allocator;
    // f(x: i32) -> i32 = (local.get 0) * 2 + 1
    const body = [_]u8{
        0x20, 0x00, // local.get 0
        0x41, 0x02, // i32.const 2
        0x6C, // i32.mul
        0x41, 0x01, // i32.const 1
        0x6A, // i32.add
    };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);

    var module = try wasm.load(allocator, bytes);
    defer module.deinit(allocator);

    const f = module.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const fp = buf.entry(*const fn (i64) callconv(.c) i64, 0);
    try std.testing.expectEqual(@as(i64, 41), fp(20)); // 20*2 + 1
}

test "wasm: local.set reassigns a parameter" {
    const allocator = std.testing.allocator;
    // f(x) = { x = x + 5, then x * 2 }
    const body = [_]u8{ 0x20, 0x00, 0x41, 0x05, 0x6A, 0x21, 0x00, 0x20, 0x00, 0x41, 0x02, 0x6C };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 16), try runi(allocator, bytes, "f", &.{3})); // (3+5)*2
}

test "wasm: signed division and remainder" {
    const allocator = std.testing.allocator;
    const div = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x6D }; // a / b (div_s)
    const dbytes = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &div, "f");
    defer allocator.free(dbytes);
    try std.testing.expectEqual(@as(i64, 6), try runi(allocator, dbytes, "f", &.{ 20, 3 }));

    const rem = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x6F }; // a % b (rem_s)
    const rbytes = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &rem, "f");
    defer allocator.free(rbytes);
    try std.testing.expectEqual(@as(i64, 2), try runi(allocator, rbytes, "f", &.{ 20, 3 }));
}

test "wasm: unsigned compare differs from signed (the coerce path)" {
    const allocator = std.testing.allocator;
    // f(a, b) = a <_u b. With a = -1 (unsigned max), -1 <_u 1 is false (0).
    const ltu = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x49 }; // i32.lt_u
    const ubytes = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &ltu, "f");
    defer allocator.free(ubytes);
    try std.testing.expectEqual(@as(i64, 0), try runi(allocator, ubytes, "f", &.{ -1, 1 }));
    try std.testing.expectEqual(@as(i64, 1), try runi(allocator, ubytes, "f", &.{ 1, 2 }));

    // The signed compare on the same inputs gives the opposite: -1 <_s 1 is true (1).
    const lts = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x48 }; // i32.lt_s
    const sbytes = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &lts, "f");
    defer allocator.free(sbytes);
    try std.testing.expectEqual(@as(i64, 1), try runi(allocator, sbytes, "f", &.{ -1, 1 }));
}

test "wasm: i64 arithmetic across the 32-bit boundary" {
    const allocator = std.testing.allocator;
    const body = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x7C }; // i64.add
    const bytes = try oneFunctionModule(allocator, &.{ valtype_i64, valtype_i64 }, &.{valtype_i64}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 0x100000001), try runi(allocator, bytes, "f", &.{ 0x100000000, 1 }));
}

test "wasm: i32.eqz" {
    const allocator = std.testing.allocator;
    const body = [_]u8{ 0x20, 0x00, 0x45 }; // i32.eqz
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 1), try runi(allocator, bytes, "f", &.{0}));
    try std.testing.expectEqual(@as(i64, 0), try runi(allocator, bytes, "f", &.{5}));
}

const FnSpec = struct { params: []const u8, results: []const u8, body: []const u8, export_name: ?[]const u8 };

/// Build a module from several function specs. Function `i` has type `i`. Exported
/// functions get an export entry under their name. Caller owns the bytes.
fn multiFunctionModule(allocator: std.mem.Allocator, funcs: []const FnSpec) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });

    var ty: std.ArrayList(u8) = .empty;
    defer ty.deinit(allocator);
    try lebU(&ty, allocator, funcs.len);
    for (funcs) |f| {
        try ty.append(allocator, 0x60);
        try lebU(&ty, allocator, f.params.len);
        try ty.appendSlice(allocator, f.params);
        try lebU(&ty, allocator, f.results.len);
        try ty.appendSlice(allocator, f.results);
    }
    try section(&out, allocator, 1, ty.items);

    var fn_sec: std.ArrayList(u8) = .empty;
    defer fn_sec.deinit(allocator);
    try lebU(&fn_sec, allocator, funcs.len);
    for (0..funcs.len) |i| try lebU(&fn_sec, allocator, i);
    try section(&out, allocator, 3, fn_sec.items);

    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    var nexp: usize = 0;
    for (funcs) |f| {
        if (f.export_name != null) nexp += 1;
    }
    try lebU(&ex, allocator, nexp);
    for (funcs, 0..) |f, i| {
        if (f.export_name) |name| {
            try lebU(&ex, allocator, name.len);
            try ex.appendSlice(allocator, name);
            try ex.append(allocator, 0x00);
            try lebU(&ex, allocator, i);
        }
    }
    try section(&out, allocator, 7, ex.items);

    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, funcs.len);
    for (funcs) |f| {
        var body_full: std.ArrayList(u8) = .empty;
        defer body_full.deinit(allocator);
        try lebU(&body_full, allocator, 0);
        try body_full.appendSlice(allocator, f.body);
        try body_full.append(allocator, 0x0B);
        try lebU(&code, allocator, body_full.items.len);
        try code.appendSlice(allocator, body_full.items);
    }
    try section(&out, allocator, 10, code.items);
    return out.toOwnedSlice(allocator);
}

/// Link and JIT a whole loaded module, then call its export `name` with `args`.
fn runModule(allocator: std.mem.Allocator, bytes: []const u8, name: []const u8, args: []const i64) !i64 {
    var module = try wasm.load(allocator, bytes);
    defer module.deinit(allocator);
    var mfs: std.ArrayList(native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (module.functions) |*lf| try mfs.append(allocator, .{ .name = lf.name, .func = &lf.func });
    var jitted = try native.jitModule(allocator, mfs.items);
    defer jitted.deinit();
    const fp = jitted.entry(*const fn (i64) callconv(.c) i64, name) orelse return error.MissingFunction;
    return switch (args.len) {
        1 => fp(args[0]),
        else => error.Unsupported,
    };
}

test "wasm: call across functions (linked module)" {
    const allocator = std.testing.allocator;
    // helper(x) = x*x + 1, main(x) = helper(x)
    const helper_body = [_]u8{ 0x20, 0x00, 0x20, 0x00, 0x6C, 0x41, 0x01, 0x6A };
    const main_body = [_]u8{ 0x20, 0x00, 0x10, 0x01 }; // local.get 0, call 1
    const bytes = try multiFunctionModule(allocator, &.{
        .{ .params = &.{valtype_i32}, .results = &.{valtype_i32}, .body = &main_body, .export_name = "main" },
        .{ .params = &.{valtype_i32}, .results = &.{valtype_i32}, .body = &helper_body, .export_name = "helper" },
    });
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 37), try runModule(allocator, bytes, "main", &.{6})); // 6*6 + 1
}

test "wasm: value live across a call" {
    const allocator = std.testing.allocator;
    // helper(x) = x*x + 1, main(x) = helper(x) + x  (x is live across the call)
    const helper_body = [_]u8{ 0x20, 0x00, 0x20, 0x00, 0x6C, 0x41, 0x01, 0x6A };
    const main_body = [_]u8{ 0x20, 0x00, 0x10, 0x01, 0x20, 0x00, 0x6A }; // local.get0, call1, local.get0, add
    const bytes = try multiFunctionModule(allocator, &.{
        .{ .params = &.{valtype_i32}, .results = &.{valtype_i32}, .body = &main_body, .export_name = "main" },
        .{ .params = &.{valtype_i32}, .results = &.{valtype_i32}, .body = &helper_body, .export_name = "helper" },
    });
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 31), try runModule(allocator, bytes, "main", &.{5})); // (5*5+1) + 5
}

/// Like oneFunctionModule but with declared locals (one group per valtype in `locals`).
fn funcModule(allocator: std.mem.Allocator, params: []const u8, results: []const u8, locals: []const u8, body: []const u8, name: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    var ty: std.ArrayList(u8) = .empty;
    defer ty.deinit(allocator);
    try lebU(&ty, allocator, 1);
    try ty.append(allocator, 0x60);
    try lebU(&ty, allocator, params.len);
    try ty.appendSlice(allocator, params);
    try lebU(&ty, allocator, results.len);
    try ty.appendSlice(allocator, results);
    try section(&out, allocator, 1, ty.items);
    var fn_sec: std.ArrayList(u8) = .empty;
    defer fn_sec.deinit(allocator);
    try lebU(&fn_sec, allocator, 1);
    try lebU(&fn_sec, allocator, 0);
    try section(&out, allocator, 3, fn_sec.items);
    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    try lebU(&ex, allocator, 1);
    try lebU(&ex, allocator, name.len);
    try ex.appendSlice(allocator, name);
    try ex.append(allocator, 0x00);
    try lebU(&ex, allocator, 0);
    try section(&out, allocator, 7, ex.items);
    var body_full: std.ArrayList(u8) = .empty;
    defer body_full.deinit(allocator);
    try lebU(&body_full, allocator, locals.len); // one group per local
    for (locals) |vt| {
        try lebU(&body_full, allocator, 1);
        try body_full.append(allocator, vt);
    }
    try body_full.appendSlice(allocator, body);
    try body_full.append(allocator, 0x0B);
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, 1);
    try lebU(&code, allocator, body_full.items.len);
    try code.appendSlice(allocator, body_full.items);
    try section(&out, allocator, 10, code.items);
    return out.toOwnedSlice(allocator);
}

test "wasm: if/else selects a value through a local" {
    const allocator = std.testing.allocator;
    // f(x) = { local r, if (x) r = 10 else r = 20, then r }
    const body = [_]u8{
        0x20, 0x00, // local.get 0 (cond)
        0x04, 0x40, // if (empty type)
        0x41, 0x0A, 0x21, 0x01, // r = 10
        0x05, // else
        0x41, 0x14, 0x21, 0x01, // r = 20
        0x0B, // end
        0x20, 0x01, // local.get r
    };
    const bytes = try funcModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 10), try runi(allocator, bytes, "f", &.{1}));
    try std.testing.expectEqual(@as(i64, 20), try runi(allocator, bytes, "f", &.{0}));
}

test "wasm: counting loop (sum 1..n) with block/loop/br_if/br" {
    const allocator = std.testing.allocator;
    // locals: i (1), acc (2). i=1, loop { if i>n break, acc+=i, i+=1 }, then acc
    const body = [_]u8{
        0x41, 0x01, 0x21, 0x01, // i = 1
        0x02, 0x40, // block
        0x03, 0x40, // loop
        0x20, 0x01, 0x20, 0x00, 0x4A, 0x0D, 0x01, // if i >_s n: br 1 (break)
        0x20, 0x02, 0x20, 0x01, 0x6A, 0x21, 0x02, // acc += i
        0x20, 0x01, 0x41, 0x01, 0x6A, 0x21, 0x01, // i += 1
        0x0C, 0x00, // br 0 (continue)
        0x0B, // end loop
        0x0B, // end block
        0x20, 0x02, // local.get acc
    };
    const bytes = try funcModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &.{ valtype_i32, valtype_i32 }, &body, "sum");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 15), try runi(allocator, bytes, "sum", &.{5}));
    try std.testing.expectEqual(@as(i64, 55), try runi(allocator, bytes, "sum", &.{10}));
    try std.testing.expectEqual(@as(i64, 0), try runi(allocator, bytes, "sum", &.{0}));
}

const valtype_f32: u8 = 0x7D;
const valtype_f64: u8 = 0x7C;

fn runF32_2(allocator: std.mem.Allocator, bytes: []const u8, name: []const u8, a: f32, b: f32) !f32 {
    var module = try wasm.load(allocator, bytes);
    defer module.deinit(allocator);
    const f = module.find(name) orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return buf.entry(*const fn (f32, f32) callconv(.c) f32, 0)(a, b);
}

fn runF32_0(allocator: std.mem.Allocator, bytes: []const u8, name: []const u8) !f32 {
    var module = try wasm.load(allocator, bytes);
    defer module.deinit(allocator);
    const f = module.find(name) orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return buf.entry(*const fn () callconv(.c) f32, 0)();
}

test "wasm: f32 arithmetic" {
    const allocator = std.testing.allocator;
    // g(a, b) = (a + b) * a
    const body = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x92, 0x20, 0x00, 0x94 };
    const bytes = try oneFunctionModule(allocator, &.{ valtype_f32, valtype_f32 }, &.{valtype_f32}, &body, "g");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(f32, 10.0), try runF32_2(allocator, bytes, "g", 2.0, 3.0)); // (2+3)*2
}

test "wasm: f32 const + add" {
    const allocator = std.testing.allocator;
    // k() = 1.5 + 2.0 = 3.5
    const body = [_]u8{ 0x43, 0x00, 0x00, 0xC0, 0x3F, 0x43, 0x00, 0x00, 0x00, 0x40, 0x92 };
    const bytes = try oneFunctionModule(allocator, &.{}, &.{valtype_f32}, &body, "k");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(f32, 3.5), try runF32_0(allocator, bytes, "k"));
}

test "wasm: int<->float conversions round-trip" {
    const allocator = std.testing.allocator;
    // h(x) = i32.trunc_f32_s(f32.convert_i32_s(x))  (identity for integers)
    const body = [_]u8{ 0x20, 0x00, 0xB2, 0xA8 };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "h");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i32, 7), @as(i32, @truncate(try runi(allocator, bytes, "h", &.{7}))));
    try std.testing.expectEqual(@as(i32, -3), @as(i32, @truncate(try runi(allocator, bytes, "h", &.{-3}))));
}

test "wasm: i64.extend_i32_s sign-extends" {
    const allocator = std.testing.allocator;
    // e(x) = i64.extend_i32_s(x). For -1 (i32), result is -1 (i64) = 0xFFFFFFFFFFFFFFFF.
    const body = [_]u8{ 0x20, 0x00, 0xAC };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i64}, &body, "e");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, -1), try runi(allocator, bytes, "e", &.{-1}));
    try std.testing.expectEqual(@as(i64, 5), try runi(allocator, bytes, "e", &.{5}));
}

/// funcModule plus a Memory section declaring `min_pages` pages.
fn memFuncModule(allocator: std.mem.Allocator, params: []const u8, results: []const u8, locals: []const u8, body: []const u8, name: []const u8, min_pages: u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    var ty: std.ArrayList(u8) = .empty;
    defer ty.deinit(allocator);
    try lebU(&ty, allocator, 1);
    try ty.append(allocator, 0x60);
    try lebU(&ty, allocator, params.len);
    try ty.appendSlice(allocator, params);
    try lebU(&ty, allocator, results.len);
    try ty.appendSlice(allocator, results);
    try section(&out, allocator, 1, ty.items);
    var fn_sec: std.ArrayList(u8) = .empty;
    defer fn_sec.deinit(allocator);
    try lebU(&fn_sec, allocator, 1);
    try lebU(&fn_sec, allocator, 0);
    try section(&out, allocator, 3, fn_sec.items);
    // Memory section: one memory, min `min_pages`, no max.
    try section(&out, allocator, 5, &.{ 0x01, 0x00, min_pages });
    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    try lebU(&ex, allocator, 1);
    try lebU(&ex, allocator, name.len);
    try ex.appendSlice(allocator, name);
    try ex.append(allocator, 0x00);
    try lebU(&ex, allocator, 0);
    try section(&out, allocator, 7, ex.items);
    var body_full: std.ArrayList(u8) = .empty;
    defer body_full.deinit(allocator);
    try lebU(&body_full, allocator, locals.len);
    for (locals) |vt| {
        try lebU(&body_full, allocator, 1);
        try body_full.append(allocator, vt);
    }
    try body_full.appendSlice(allocator, body);
    try body_full.append(allocator, 0x0B);
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, 1);
    try lebU(&code, allocator, body_full.items.len);
    try code.appendSlice(allocator, body_full.items);
    try section(&out, allocator, 10, code.items);
    return out.toOwnedSlice(allocator);
}

test "wasm: linear memory store then load" {
    const allocator = std.testing.allocator;
    // rw(addr, val) = { mem[addr] = val, then mem[addr] }   (memarg align=2 offset=0)
    const body = [_]u8{
        0x20, 0x00, 0x20, 0x01, 0x36, 0x02, 0x00, // i32.store mem[addr]=val
        0x20, 0x00, 0x28, 0x02, 0x00, // i32.load mem[addr]
    };
    const bytes = try memFuncModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &.{}, &body, "rw", 1);
    defer allocator.free(bytes);
    var inst = try wasm.Instance.instantiate(allocator, bytes, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 12345), try inst.call2(i32, i32, i32, "rw", 16, 12345));
    try std.testing.expectEqual(@as(u32, 12345), std.mem.readInt(u32, inst.memory[16..20], .little)); // landed in the buffer
}

test "wasm: value-carrying if (if result i32)" {
    const allocator = std.testing.allocator;
    // f(x) = if (x) (result i32) { 100 } else { 200 }  (100 and 200 as signed LEB)
    const body = [_]u8{ 0x20, 0x00, 0x04, 0x7F, 0x41, 0xE4, 0x00, 0x05, 0x41, 0xC8, 0x01, 0x0B };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 100), try runi(allocator, bytes, "f", &.{1}));
    try std.testing.expectEqual(@as(i64, 200), try runi(allocator, bytes, "f", &.{0}));
}

test "wasm: select (max)" {
    const allocator = std.testing.allocator;
    // f(a, b) = (a >_s b) ? a : b
    const body = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x20, 0x00, 0x20, 0x01, 0x4A, 0x1B };
    const bytes = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 7), try runi(allocator, bytes, "f", &.{ 3, 7 }));
    try std.testing.expectEqual(@as(i64, 9), try runi(allocator, bytes, "f", &.{ 9, 2 }));
}

test "wasm: value-carrying block with br" {
    const allocator = std.testing.allocator;
    // f(x) = block (result i32) { local.get x, i32.const 1, i32.add, br 0 } + 0
    const body = [_]u8{ 0x02, 0x7F, 0x20, 0x00, 0x41, 0x01, 0x6A, 0x0C, 0x00, 0x0B };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 6), try runi(allocator, bytes, "f", &.{5})); // (5+1) via br
}

test "wasm: br_table (switch)" {
    const allocator = std.testing.allocator;
    // switch(x){0->10, 1->20, default->30} via nested blocks
    const body = [_]u8{
        0x02, 0x40, 0x02, 0x40, 0x02, 0x40, // block b2, b1, b0
        0x20, 0x00, // local.get x
        0x0E, 0x02, 0x00, 0x01, 0x02, // br_table [0,1] default 2
        0x0B, 0x41, 0x0A, 0x0F, // end b0: const 10, return
        0x0B, 0x41, 0x14, 0x0F, // end b1: const 20, return
        0x0B, 0x41, 0x1E, 0x0F, // end b2: const 30, return
    };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "sw");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 10), try runi(allocator, bytes, "sw", &.{0}));
    try std.testing.expectEqual(@as(i64, 20), try runi(allocator, bytes, "sw", &.{1}));
    try std.testing.expectEqual(@as(i64, 30), try runi(allocator, bytes, "sw", &.{2}));
    try std.testing.expectEqual(@as(i64, 30), try runi(allocator, bytes, "sw", &.{7}));
}

test "wasm: i32.rotl / rotr" {
    const allocator = std.testing.allocator;
    // rotl(x, n)
    const rl = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x77 };
    const rlb = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &rl, "f");
    defer allocator.free(rlb);
    // rotl(0x12345678, 8) = 0x34567812
    try std.testing.expectEqual(@as(i32, @bitCast(@as(u32, 0x34567812))), @as(i32, @truncate(try runi(allocator, rlb, "f", &.{ 0x12345678, 8 }))));
    // rotl(x, 0) = x  (the n==0 edge case)
    try std.testing.expectEqual(@as(i32, 0x12345678), @as(i32, @truncate(try runi(allocator, rlb, "f", &.{ 0x12345678, 0 }))));

    const rr = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x78 };
    const rrb = try oneFunctionModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &rr, "f");
    defer allocator.free(rrb);
    // rotr(0x12345678, 8) = 0x78123456
    try std.testing.expectEqual(@as(i32, @bitCast(@as(u32, 0x78123456))), @as(i32, @truncate(try runi(allocator, rrb, "f", &.{ 0x12345678, 8 }))));
}

test "wasm: i32.extend8_s / extend16_s" {
    const allocator = std.testing.allocator;
    const e8 = [_]u8{ 0x20, 0x00, 0xC0 };
    const e8b = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &e8, "f");
    defer allocator.free(e8b);
    try std.testing.expectEqual(@as(i32, -1), @as(i32, @truncate(try runi(allocator, e8b, "f", &.{0xFF})))); // 0xFF -> -1
    try std.testing.expectEqual(@as(i32, 1), @as(i32, @truncate(try runi(allocator, e8b, "f", &.{0x01}))));
    const e16 = [_]u8{ 0x20, 0x00, 0xC1 };
    const e16b = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &e16, "f");
    defer allocator.free(e16b);
    try std.testing.expectEqual(@as(i32, -1), @as(i32, @truncate(try runi(allocator, e16b, "f", &.{0xFFFF}))));
}

test "wasm: f32.min / max" {
    const allocator = std.testing.allocator;
    const mn = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x96 };
    const mnb = try oneFunctionModule(allocator, &.{ valtype_f32, valtype_f32 }, &.{valtype_f32}, &mn, "f");
    defer allocator.free(mnb);
    try std.testing.expectEqual(@as(f32, 2.0), try runF32_2(allocator, mnb, "f", 2.0, 5.0));
    const mx = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x97 };
    const mxb = try oneFunctionModule(allocator, &.{ valtype_f32, valtype_f32 }, &.{valtype_f32}, &mx, "f");
    defer allocator.free(mxb);
    try std.testing.expectEqual(@as(f32, 5.0), try runF32_2(allocator, mxb, "f", 2.0, 5.0));
}

/// One-function module with one mutable i32 global initialized to `init`.
fn globalFuncModule(allocator: std.mem.Allocator, params: []const u8, results: []const u8, body: []const u8, name: []const u8, init: i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    var ty: std.ArrayList(u8) = .empty;
    defer ty.deinit(allocator);
    try lebU(&ty, allocator, 1);
    try ty.append(allocator, 0x60);
    try lebU(&ty, allocator, params.len);
    try ty.appendSlice(allocator, params);
    try lebU(&ty, allocator, results.len);
    try ty.appendSlice(allocator, results);
    try section(&out, allocator, 1, ty.items);
    var fn_sec: std.ArrayList(u8) = .empty;
    defer fn_sec.deinit(allocator);
    try lebU(&fn_sec, allocator, 1);
    try lebU(&fn_sec, allocator, 0);
    try section(&out, allocator, 3, fn_sec.items);
    // Global section: one mutable i32 = init (i32.const init, end).
    var g: std.ArrayList(u8) = .empty;
    defer g.deinit(allocator);
    try lebU(&g, allocator, 1); // one global
    try g.append(allocator, 0x7F); // i32
    try g.append(allocator, 0x01); // mutable
    try g.append(allocator, 0x41); // i32.const
    {
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(allocator);
        var v: i64 = init;
        // signed LEB of init
        while (true) {
            const byte: u8 = @intCast(@as(u64, @bitCast(v)) & 0x7F);
            v >>= 7;
            const done = (v == 0 and (byte & 0x40) == 0) or (v == -1 and (byte & 0x40) != 0);
            try g.append(allocator, if (done) byte else byte | 0x80);
            if (done) break;
        }
    }
    try g.append(allocator, 0x0B); // end
    try section(&out, allocator, 6, g.items);
    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    try lebU(&ex, allocator, 1);
    try lebU(&ex, allocator, name.len);
    try ex.appendSlice(allocator, name);
    try ex.append(allocator, 0x00);
    try lebU(&ex, allocator, 0);
    try section(&out, allocator, 7, ex.items);
    var body_full: std.ArrayList(u8) = .empty;
    defer body_full.deinit(allocator);
    try lebU(&body_full, allocator, 0);
    try body_full.appendSlice(allocator, body);
    try body_full.append(allocator, 0x0B);
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, 1);
    try lebU(&code, allocator, body_full.items.len);
    try code.appendSlice(allocator, body_full.items);
    try section(&out, allocator, 10, code.items);
    return out.toOwnedSlice(allocator);
}

test "wasm: mutable global get/set" {
    const allocator = std.testing.allocator;
    // incr(x) = { g = g + x, then g }   with global g initialized to 100
    const body = [_]u8{ 0x23, 0x00, 0x20, 0x00, 0x6A, 0x24, 0x00, 0x23, 0x00 };
    const bytes = try globalFuncModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "incr", 100);
    defer allocator.free(bytes);
    var inst = try wasm.Instance.instantiate(allocator, bytes, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i64, 100), inst.globals[0]); // initialized from the module
    try std.testing.expectEqual(@as(i32, 105), try inst.call1(i32, i32, "incr", 5)); // g: 100 -> 105
    try std.testing.expectEqual(@as(i32, 115), try inst.call1(i32, i32, "incr", 10)); // g: 105 -> 115 (persists)
}

/// memFuncModule plus an active data segment writing `data` at offset 0.
fn dataFuncModule(allocator: std.mem.Allocator, params: []const u8, results: []const u8, body: []const u8, name: []const u8, data: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    var ty: std.ArrayList(u8) = .empty;
    defer ty.deinit(allocator);
    try lebU(&ty, allocator, 1);
    try ty.append(allocator, 0x60);
    try lebU(&ty, allocator, params.len);
    try ty.appendSlice(allocator, params);
    try lebU(&ty, allocator, results.len);
    try ty.appendSlice(allocator, results);
    try section(&out, allocator, 1, ty.items);
    var fn_sec: std.ArrayList(u8) = .empty;
    defer fn_sec.deinit(allocator);
    try lebU(&fn_sec, allocator, 1);
    try lebU(&fn_sec, allocator, 0);
    try section(&out, allocator, 3, fn_sec.items);
    try section(&out, allocator, 5, &.{ 0x01, 0x00, 0x01 }); // memory, min 1
    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    try lebU(&ex, allocator, 1);
    try lebU(&ex, allocator, name.len);
    try ex.appendSlice(allocator, name);
    try ex.append(allocator, 0x00);
    try lebU(&ex, allocator, 0);
    try section(&out, allocator, 7, ex.items);
    var body_full: std.ArrayList(u8) = .empty;
    defer body_full.deinit(allocator);
    try lebU(&body_full, allocator, 0);
    try body_full.appendSlice(allocator, body);
    try body_full.append(allocator, 0x0B);
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, 1);
    try lebU(&code, allocator, body_full.items.len);
    try code.appendSlice(allocator, body_full.items);
    try section(&out, allocator, 10, code.items);
    // Data section: one active segment at offset 0.
    var d: std.ArrayList(u8) = .empty;
    defer d.deinit(allocator);
    try lebU(&d, allocator, 1); // one segment
    try d.append(allocator, 0x00); // active, memory 0
    try d.appendSlice(allocator, &.{ 0x41, 0x00, 0x0B }); // i32.const 0, end
    try lebU(&d, allocator, data.len);
    try d.appendSlice(allocator, data);
    try section(&out, allocator, 11, d.items);
    return out.toOwnedSlice(allocator);
}

test "wasm: data segment initializes memory" {
    const allocator = std.testing.allocator;
    // f(addr) = i32.load mem[addr]   (the data segment pre-fills memory)
    const body = [_]u8{ 0x20, 0x00, 0x28, 0x02, 0x00 };
    const data = [_]u8{ 0x2A, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00 }; // 42 at [0], 99 at [4]
    const bytes = try dataFuncModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "rd", &data);
    defer allocator.free(bytes);
    var inst = try wasm.Instance.instantiate(allocator, bytes, &.{}); // applies the data segment
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 42), try inst.call1(i32, i32, "rd", 0)); // mem[0]
    try std.testing.expectEqual(@as(i32, 99), try inst.call1(i32, i32, "rd", 4)); // mem[4]
}

test "wasm: sized memory load/store (8/16-bit)" {
    const allocator = std.testing.allocator;
    // f(addr, val) = { i32.store8 mem[addr]=val, then i32.load8_u mem[addr] }
    const body = [_]u8{
        0x20, 0x00, 0x20, 0x01, 0x3A, 0x00, 0x00, // i32.store8
        0x20, 0x00, 0x2D, 0x00, 0x00, // i32.load8_u
    };
    const bytes = try memFuncModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &.{}, &body, "rw8", 1);
    defer allocator.free(bytes);
    var inst = try wasm.Instance.instantiate(allocator, bytes, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 0xAB), try inst.call2(i32, i32, i32, "rw8", 10, 0x1234AB)); // store8 keeps low byte
    try std.testing.expectEqual(@as(u8, 0xAB), inst.memory[10]);
    try std.testing.expectEqual(@as(u8, 0), inst.memory[11]); // only one byte written

    // load8_s sign-extends: store 0xFF, load8_s -> -1
    const body_s = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x3A, 0x00, 0x00, 0x20, 0x00, 0x2C, 0x00, 0x00 };
    const bytes_s = try memFuncModule(allocator, &.{ valtype_i32, valtype_i32 }, &.{valtype_i32}, &.{}, &body_s, "rw8s", 1);
    defer allocator.free(bytes_s);
    var inst_s = try wasm.Instance.instantiate(allocator, bytes_s, &.{});
    defer inst_s.deinit();
    try std.testing.expectEqual(@as(i32, -1), try inst_s.call2(i32, i32, i32, "rw8s", 0, 0xFF));
}

test "wasm: f32.abs" {
    const allocator = std.testing.allocator;
    const body = [_]u8{ 0x20, 0x00, 0x8B };
    const bytes = try oneFunctionModule(allocator, &.{valtype_f32}, &.{valtype_f32}, &body, "a");
    defer allocator.free(bytes);
    var module = try wasm.load(allocator, bytes);
    defer module.deinit(allocator);
    var jit = try native.jitFunction(allocator, module.find("a").?);
    defer jit.deinit();
    const fp = jit.entry(*const fn (f32) callconv(.c) f32, 0);
    try std.testing.expectEqual(@as(f32, 3.5), fp(-3.5));
    try std.testing.expectEqual(@as(f32, 3.5), fp(3.5));
}

test "wasm: i32 clz / ctz / popcnt" {
    const allocator = std.testing.allocator;
    inline for (.{
        .{ .op = 0x67, .in = 1, .want = 31 }, // clz(1) = 31
        .{ .op = 0x67, .in = 0, .want = 32 }, // clz(0) = 32
        .{ .op = 0x67, .in = 0x00FF0000, .want = 8 }, // clz
        .{ .op = 0x68, .in = 8, .want = 3 }, // ctz(8) = 3
        .{ .op = 0x68, .in = 0, .want = 32 }, // ctz(0) = 32
        .{ .op = 0x69, .in = 0xFF, .want = 8 }, // popcnt(0xFF) = 8
        .{ .op = 0x69, .in = 0xFFFFFFFF, .want = 32 }, // popcnt(-1) = 32
        .{ .op = 0x69, .in = 0, .want = 0 },
    }) |c| {
        const body = [_]u8{ 0x20, 0x00, c.op };
        const bytes = try oneFunctionModule(allocator, &.{valtype_i32}, &.{valtype_i32}, &body, "f");
        defer allocator.free(bytes);
        try std.testing.expectEqual(@as(i32, c.want), @as(i32, @truncate(try runi(allocator, bytes, "f", &.{c.in}))));
    }
}

test "wasm: i64 popcnt" {
    const allocator = std.testing.allocator;
    const body = [_]u8{ 0x20, 0x00, 0x7B };
    const bytes = try oneFunctionModule(allocator, &.{valtype_i64}, &.{valtype_i64}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 40), try runi(allocator, bytes, "f", &.{0xFFFFFFFFFF})); // 40 ones
}

test "wasm: f32 sqrt/ceil/floor/trunc/nearest" {
    const allocator = std.testing.allocator;
    inline for (.{
        .{ .op = 0x91, .in = 9.0, .want = 3.0 }, // sqrt(9) = 3
        .{ .op = 0x8D, .in = 2.3, .want = 3.0 }, // ceil
        .{ .op = 0x8E, .in = 2.7, .want = 2.0 }, // floor
        .{ .op = 0x8F, .in = -2.7, .want = -2.0 }, // trunc (toward zero)
        .{ .op = 0x90, .in = 2.5, .want = 2.0 }, // nearest (ties to even)
        .{ .op = 0x90, .in = 3.5, .want = 4.0 },
    }) |c| {
        const body = [_]u8{ 0x20, 0x00, c.op };
        const bytes = try oneFunctionModule(allocator, &.{valtype_f32}, &.{valtype_f32}, &body, "f");
        defer allocator.free(bytes);
        var module = try wasm.load(allocator, bytes);
        defer module.deinit(allocator);
        var jit = try native.jitFunction(allocator, module.find("f").?);
        defer jit.deinit();
        try std.testing.expectEqual(@as(f32, c.want), jit.entry(*const fn (f32) callconv(.c) f32, 0)(c.in));
    }
}

test "wasm: f32.copysign and reinterpret" {
    const allocator = std.testing.allocator;
    // copysign(3.0, -1.0) = -3.0
    const cs = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x98 };
    const csb = try oneFunctionModule(allocator, &.{ valtype_f32, valtype_f32 }, &.{valtype_f32}, &cs, "c");
    defer allocator.free(csb);
    var m1 = try wasm.load(allocator, csb);
    defer m1.deinit(allocator);
    var j1 = try native.jitFunction(allocator, m1.find("c").?);
    defer j1.deinit();
    try std.testing.expectEqual(@as(f32, -3.0), j1.entry(*const fn (f32, f32) callconv(.c) f32, 0)(3.0, -1.0));
    try std.testing.expectEqual(@as(f32, 3.0), j1.entry(*const fn (f32, f32) callconv(.c) f32, 0)(3.0, 1.0));

    // i32.reinterpret_f32(1.0) = 0x3F800000
    const ri = [_]u8{ 0x20, 0x00, 0xBC };
    const rib = try oneFunctionModule(allocator, &.{valtype_f32}, &.{valtype_i32}, &ri, "r");
    defer allocator.free(rib);
    var m2 = try wasm.load(allocator, rib);
    defer m2.deinit(allocator);
    var j2 = try native.jitFunction(allocator, m2.find("r").?);
    defer j2.deinit();
    try std.testing.expectEqual(@as(i32, @bitCast(@as(u32, 0x3F800000))), j2.entry(*const fn (f32) callconv(.c) i32, 0)(1.0));
}

test "wasm: call_indirect through a function table" {
    const allocator = std.testing.allocator;
    // type0 = (i32)->i32, type1 = (i32,i32)->i32
    // func0 double(x)=x*2, func1 triple(x)=x*3, func2 dispatch(sel,x)=table[sel](x)
    // table = [double, triple]
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    // Type section: 2 types
    try section(&out, allocator, 1, &.{ 0x02, 0x60, 0x01, 0x7F, 0x01, 0x7F, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F });
    // Function section: func0:type0, func1:type0, func2:type1
    try section(&out, allocator, 3, &.{ 0x03, 0x00, 0x00, 0x01 });
    // Table section: 1 funcref table, min 2
    try section(&out, allocator, 4, &.{ 0x01, 0x70, 0x00, 0x02 });
    // Export: dispatch = func 2
    var ex: std.ArrayList(u8) = .empty;
    defer ex.deinit(allocator);
    try lebU(&ex, allocator, 1);
    try lebU(&ex, allocator, "dispatch".len);
    try ex.appendSlice(allocator, "dispatch");
    try ex.append(allocator, 0x00);
    try lebU(&ex, allocator, 2);
    try section(&out, allocator, 7, ex.items);
    // Element: segment 0, offset 0, [func0, func1]
    try section(&out, allocator, 9, &.{ 0x01, 0x00, 0x41, 0x00, 0x0B, 0x02, 0x00, 0x01 });
    // Code: 3 bodies
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(allocator);
    try lebU(&code, allocator, 3);
    const bodies = [_][]const u8{
        &.{ 0x00, 0x20, 0x00, 0x41, 0x02, 0x6C, 0x0B }, // double
        &.{ 0x00, 0x20, 0x00, 0x41, 0x03, 0x6C, 0x0B }, // triple
        &.{ 0x00, 0x20, 0x01, 0x20, 0x00, 0x11, 0x00, 0x00, 0x0B }, // dispatch: x, sel, call_indirect
    };
    for (bodies) |b| {
        try lebU(&code, allocator, b.len);
        try code.appendSlice(allocator, b);
    }
    try section(&out, allocator, 10, code.items);

    var inst = try wasm.Instance.instantiate(allocator, out.items, &.{}); // fills the table
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 10), try inst.call2(i32, i32, i32, "dispatch", 0, 5)); // double(5)
    try std.testing.expectEqual(@as(i32, 15), try inst.call2(i32, i32, i32, "dispatch", 1, 5)); // triple(5)
}

const Host = struct {
    fn addOne(x: i32) callconv(.c) i32 {
        return x + 1;
    }
};

test "wasm: imported host function" {
    const allocator = std.testing.allocator;
    // (import "env" "addone" (func (param i32)(result i32)))
    // (func $run (export "run") (param x i32)(result i32) (local.get x)(call $addone))
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });
    try section(&out, allocator, 1, &.{ 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F }); // type0 = (i32)->i32
    try section(&out, allocator, 2, &.{ 0x01, 0x03, 'e', 'n', 'v', 0x06, 'a', 'd', 'd', 'o', 'n', 'e', 0x00, 0x00 }); // import
    try section(&out, allocator, 3, &.{ 0x01, 0x00 }); // func0 (run): type0
    try section(&out, allocator, 7, &.{ 0x01, 0x03, 'r', 'u', 'n', 0x00, 0x01 }); // export run = funcidx 1
    try section(&out, allocator, 10, &.{ 0x01, 0x06, 0x00, 0x20, 0x00, 0x10, 0x00, 0x0B }); // run: local.get 0, call 0

    // Instantiate, binding the import to the host function.
    var inst = try wasm.Instance.instantiate(allocator, out.items, &.{@intFromPtr(&Host.addOne)});
    defer inst.deinit();
    try std.testing.expectEqualStrings("env.addone", inst.module.imports[0]);
    try std.testing.expectEqual(@as(i32, 6), try inst.call1(i32, i32, "run", 5)); // addone(5)
    try std.testing.expectEqual(@as(i32, 43), try inst.call1(i32, i32, "run", 42));
}

test "wasm: bulk memory (memory.fill + memory.copy) under the engine" {
    const allocator = std.testing.allocator;
    // (memory 1) (func (export "test") (result i32)
    //   memory.fill(0, 65, 4), memory.copy(8, 0, 4), i32.load(8))  -> 0x41414141
    const bytes = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, 0x03,
        0x02, 0x01, 0x00, 0x05, 0x03, 0x01, 0x00, 0x01, 0x07, 0x08, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x00, 0x00, 0x0a, 0x1d, 0x01, 0x1b, 0x00, 0x41, 0x00, 0x41, 0xc1, 0x00, 0x41, 0x04, 0xfc, 0x0b,
        0x00, 0x41, 0x08, 0x41, 0x00, 0x41, 0x04, 0xfc, 0x0a, 0x00, 0x00, 0x41, 0x08, 0x28, 0x02, 0x00,
        0x0b,
    };
    var inst = try wasm.Instance.instantiate(allocator, &bytes, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 0x41414141), try inst.call0(i32, "test"));
}

test "wasm: unreachable code after return is skipped" {
    const allocator = std.testing.allocator;
    // (func (result i32) (i32.const 42) (return) (i32.const 99))  -- the 99 is dead code
    const body = [_]u8{ 0x41, 0x2A, 0x0F, 0x41, 0x63 };
    const bytes = try oneFunctionModule(allocator, &.{}, &.{valtype_i32}, &body, "f");
    defer allocator.free(bytes);
    try std.testing.expectEqual(@as(i64, 42), try runi(allocator, bytes, "f", &.{}));
}

/// Build a `(i32, i32) -> i32` function that extracts one field of a two-field
/// struct built from its two params, so the return value proves the extract index.
fn structPickFn(func: *ir.function.Function, index: u32) !void {
    const t_i32 = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{ t_i32, t_i32 } });
    const b0 = try func.appendBlock();
    const a = try func.appendBlockParam(b0, t_i32);
    const b = try func.appendBlockParam(b0, t_i32);
    const v = try func.appendStructNew(b0, st, &.{ a, b });
    const field = try func.appendInst(b0, t_i32, .{ .extract = .{ .aggregate = v, .index = index } });
    func.setTerminator(b0, .{ .ret = field });
}

test "wasm target: struct_new + extract lowers and round-trips" {
    const allocator = std.testing.allocator;

    // fst(a, b) returns field 0 (a), snd(a, b) returns field 1 (b).
    var fst = ir.function.Function.init(allocator);
    defer fst.deinit();
    try structPickFn(&fst, 0);
    var snd = ir.function.Function.init(allocator);
    defer snd.deinit();
    try structPickFn(&snd, 1);

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("fst", &fst);
    try m.addFunction("snd", &snd);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 3), try inst.call2(i32, i32, i32, "fst", 3, 7));
    try std.testing.expectEqual(@as(i32, 7), try inst.call2(i32, i32, i32, "snd", 3, 7));
    try std.testing.expectEqual(@as(i32, 20), try inst.call2(i32, i32, i32, "snd", 10, 20));
}

test "wasm target: if/else selection (max) round-trips" {
    const allocator = std.testing.allocator;

    // max(a, b): a diamond where both if arms jump to a merge block carrying the
    // larger value as a phi parameter.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, t);
    const b = try f.appendBlockParam(entry, t);
    const merge = try f.appendBlock();
    const r = try f.appendBlockParam(merge, t);
    const c = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try f.appendIf(entry, c, .{ .target = merge, .args = &.{a} }, .{ .target = merge, .args = &.{b} });
    f.setTerminator(merge, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("max", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 4), try inst.call2(i32, i32, i32, "max", 3, 4));
    try std.testing.expectEqual(@as(i32, 7), try inst.call2(i32, i32, i32, "max", 7, 2));
}

test "wasm target: nested if/else (sign) round-trips" {
    const allocator = std.testing.allocator;

    // sign(x): x>0 -> 1, else (x<0 -> -1, else 0). The inner if's merge is the outer
    // merge, so the emitter must not emit it twice or at the wrong nesting.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const x = try f.appendBlockParam(entry, t);
    const neg = try f.appendBlock();
    const merge = try f.appendBlock();
    const r = try f.appendBlockParam(merge, t);

    const z0 = try f.appendInst(entry, t, .{ .iconst = 0 });
    const c1 = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = z0 } });
    const p1 = try f.appendInst(entry, t, .{ .iconst = 1 });
    try f.appendIf(entry, c1, .{ .target = merge, .args = &.{p1} }, .{ .target = neg, .args = &.{} });
    try f.addAttr(.{ .block = entry }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(merge) } } });

    const z1 = try f.appendInst(neg, t, .{ .iconst = 0 });
    const c2 = try f.appendInst(neg, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = z1 } });
    const m1 = try f.appendInst(neg, t, .{ .iconst = -1 });
    const z2 = try f.appendInst(neg, t, .{ .iconst = 0 });
    try f.appendIf(neg, c2, .{ .target = merge, .args = &.{m1} }, .{ .target = merge, .args = &.{z2} });

    f.setTerminator(merge, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("sign", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 1), try inst.call1(i32, i32, "sign", 42));
    try std.testing.expectEqual(@as(i32, -1), try inst.call1(i32, i32, "sign", -7));
    try std.testing.expectEqual(@as(i32, 0), try inst.call1(i32, i32, "sign", 0));
}

test "wasm target: loop (sum 1..n) round-trips" {
    const allocator = std.testing.allocator;

    // sum(n) = 1 + 2 + ... + n, a header/body/exit loop with phi-carried acc and i.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const n = try f.appendBlockParam(entry, t);
    const header = try f.appendBlock();
    const sum = try f.appendBlockParam(header, t);
    const i = try f.appendBlockParam(header, t);
    const body = try f.appendBlock();
    const exit = try f.appendBlock();
    const r = try f.appendBlockParam(exit, t);

    const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
    const one = try f.appendInst(entry, t, .{ .iconst = 1 });
    try f.setJump(entry, header, &.{ zero, one });

    const cond = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .le, .lhs = i, .rhs = n } });
    try f.appendIf(header, cond, .{ .target = body, .args = &.{} }, .{ .target = exit, .args = &.{sum} });
    // Mark the structured loop: merge (exit) and continue (the back-edge block).
    try f.addAttr(.{ .block = header }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(exit) } } });
    try f.addAttr(.{ .block = header }, .{ .custom = .{ .namespace = "cf", .key = "continue", .value = .{ .int = @intFromEnum(body) } } });

    const sum2 = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = i } });
    const inext = try f.appendArithImm(body, t, .add, i, 1);
    try f.setJump(body, header, &.{ sum2, inext });

    f.setTerminator(exit, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("sum", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 15), try inst.call1(i32, i32, "sum", 5));
    try std.testing.expectEqual(@as(i32, 55), try inst.call1(i32, i32, "sum", 10));
}

test "wasm target: if nested inside a loop round-trips" {
    const allocator = std.testing.allocator;

    // clampSum(n): acc=0; for i in 1..n { if acc < 100 acc += i }  (an if in the body,
    // whose merge is the loop's continue block).
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const n = try f.appendBlockParam(entry, t);
    const header = try f.appendBlock();
    const acc = try f.appendBlockParam(header, t);
    const i = try f.appendBlockParam(header, t);
    const body = try f.appendBlock();
    const add = try f.appendBlock();
    const bmerge = try f.appendBlock();
    const acc3 = try f.appendBlockParam(bmerge, t);
    const exit = try f.appendBlock();
    const accf = try f.appendBlockParam(exit, t);

    const acc0 = try f.appendInst(entry, t, .{ .iconst = 0 });
    const istart = try f.appendInst(entry, t, .{ .iconst = 1 });
    try f.setJump(entry, header, &.{ acc0, istart });

    const cond = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .le, .lhs = i, .rhs = n } });
    try f.appendIf(header, cond, .{ .target = body, .args = &.{} }, .{ .target = exit, .args = &.{acc} });
    try f.addAttr(.{ .block = header }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(exit) } } });
    try f.addAttr(.{ .block = header }, .{ .custom = .{ .namespace = "cf", .key = "continue", .value = .{ .int = @intFromEnum(bmerge) } } });

    const hundred = try f.appendInst(body, t, .{ .iconst = 100 });
    const c2 = try f.appendInst(body, bool_t, .{ .icmp = .{ .op = .lt, .lhs = acc, .rhs = hundred } });
    try f.appendIf(body, c2, .{ .target = add, .args = &.{} }, .{ .target = bmerge, .args = &.{acc} });
    try f.addAttr(.{ .block = body }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(bmerge) } } });

    const acc2 = try f.appendInst(add, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = i } });
    try f.setJump(add, bmerge, &.{acc2});

    const inext = try f.appendArithImm(bmerge, t, .add, i, 1);
    try f.setJump(bmerge, header, &.{ acc3, inext });

    f.setTerminator(exit, .{ .ret = accf });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("clampSum", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 15), try inst.call1(i32, i32, "clampSum", 5));
    try std.testing.expectEqual(@as(i32, 105), try inst.call1(i32, i32, "clampSum", 20));
}

test "wasm target: i64 arithmetic round-trips" {
    const allocator = std.testing.allocator;
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, t);
    const b = try f.appendBlockParam(entry, t);
    const s = try f.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    f.setTerminator(entry, .{ .ret = s });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("add64", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    // A value that needs the full 64 bits (would truncate on a 32-bit path).
    try std.testing.expectEqual(@as(i64, 0x1_0000_0001), try inst.call2(i64, i64, i64, "add64", 0x1_0000_0000, 1));
}

test "wasm target: f64 arithmetic round-trips" {
    const allocator = std.testing.allocator;
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f64 });
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, t);
    const b = try f.appendBlockParam(entry, t);
    const s = try f.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    f.setTerminator(entry, .{ .ret = s });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("muld", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f64, 8.0), try inst.call2(f64, f64, f64, "muld", 2.5, 3.2));
}

test "wasm target: narrow i8 store/load sign-extends round-trips" {
    const allocator = std.testing.allocator;
    // s8(x: i8) = load_s8(store8(x)): stores the low byte, reloads sign-extended.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const i8t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr = try f.types.intern(.ptr);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, i8t);
    const slot = try f.appendInst(b, ptr, .{ .alloca = .{ .elem = i8t } });
    try f.appendStore(b, x, slot);
    const r = try f.appendInst(b, i8t, .{ .load = .{ .ptr = slot } });
    f.setTerminator(b, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("s8", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    // 200 = 0xC8, stored as a byte, reloaded as a signed i8 = -56.
    try std.testing.expectEqual(@as(i32, -56), try inst.call1(i32, i32, "s8", 200));
    try std.testing.expectEqual(@as(i32, 50), try inst.call1(i32, i32, "s8", 50));
}

test "wasm target: unary reinterpret (f64<->i64) and sqrt round-trip" {
    const allocator = std.testing.allocator;

    // rt(x: f64) = reinterpret_f64(reinterpret_i64(x)) == x (exercises the f64 reinterpret).
    var f_rt = ir.function.Function.init(allocator);
    defer f_rt.deinit();
    {
        const f64t = try f_rt.types.intern(.{ .float = .f64 });
        const i64t = try f_rt.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
        const b = try f_rt.appendBlock();
        const x = try f_rt.appendBlockParam(b, f64t);
        const bits = try f_rt.appendInst(b, i64t, .{ .unary = .{ .op = .reinterpret, .value = x } });
        const back = try f_rt.appendInst(b, f64t, .{ .unary = .{ .op = .reinterpret, .value = bits } });
        f_rt.setTerminator(b, .{ .ret = back });
    }
    // sqrtd(x: f64) = sqrt(x).
    var f_sq = ir.function.Function.init(allocator);
    defer f_sq.deinit();
    {
        const f64t = try f_sq.types.intern(.{ .float = .f64 });
        const b = try f_sq.appendBlock();
        const x = try f_sq.appendBlockParam(b, f64t);
        const r = try f_sq.appendInst(b, f64t, .{ .unary = .{ .op = .sqrt, .value = x } });
        f_sq.setTerminator(b, .{ .ret = r });
    }

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("rt", &f_rt);
    try m.addFunction("sqrtd", &f_sq);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f64, 3.14159), try inst.call1(f64, f64, "rt", 3.14159));
    try std.testing.expectEqual(@as(f64, 4.0), try inst.call1(f64, f64, "sqrtd", 16.0));
}

test "wasm target: unsigned vs signed compare round-trips" {
    const allocator = std.testing.allocator;
    // ltu(a, b) = a <_u b, using an unsigned i32. -1 (0xFFFFFFFF) <_u 1 is false.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const u32t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, u32t);
    const b = try f.appendBlockParam(entry, u32t);
    const c = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    // widen bool to i32 for the ABI: return c ? 1 : 0 via select on constants.
    const one = try f.appendInst(entry, u32t, .{ .iconst = 1 });
    const zero = try f.appendInst(entry, u32t, .{ .iconst = 0 });
    const r = try f.appendInst(entry, u32t, .{ .select = .{ .cond = c, .then = one, .@"else" = zero } });
    f.setTerminator(entry, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("ltu", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 0), try inst.call2(i32, i32, i32, "ltu", -1, 1)); // 0xFFFFFFFF <u 1 = false
    try std.testing.expectEqual(@as(i32, 1), try inst.call2(i32, i32, i32, "ltu", 1, 2));
}

test "wasm target: unsigned div and shr use the unsigned wasm ops" {
    const allocator = std.testing.allocator;
    const u32t_of = struct {
        fn t(f: *ir.function.Function) !ir.types.Type {
            return f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        }
    }.t;

    { // divu(a, b) = a / b unsigned. 0xFFFFFFFF / 2 = 0x7FFFFFFF (unsigned), signed -1/2 = 0.
        var f = ir.function.Function.init(allocator);
        defer f.deinit();
        const u = try u32t_of(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, u);
        const d = try f.appendBlockParam(b, u);
        const r = try f.appendInst(b, u, .{ .arith = .{ .op = .div, .lhs = a, .rhs = d } });
        f.setTerminator(b, .{ .ret = r });
        var m = wtarget.link.Module.init(allocator);
        defer m.deinit();
        try m.addFunction("divu", &f);
        var linked = try wtarget.link.compileModule(allocator, &m);
        defer linked.deinit(allocator);
        var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
        defer inst.deinit();
        try std.testing.expectEqual(@as(i32, 0x7FFFFFFF), try inst.call2(i32, i32, i32, "divu", -1, 2));
    }
    { // shru(a) = a >> 1 logical. 0x80000000 >> 1 = 0x40000000 (unsigned), arithmetic = 0xC0000000.
        var f = ir.function.Function.init(allocator);
        defer f.deinit();
        const u = try u32t_of(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, u);
        const r = try f.appendArithImm(b, u, .shr, a, 1);
        f.setTerminator(b, .{ .ret = r });
        var m = wtarget.link.Module.init(allocator);
        defer m.deinit();
        try m.addFunction("shru", &f);
        var linked = try wtarget.link.compileModule(allocator, &m);
        defer linked.deinit(allocator);
        var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
        defer inst.deinit();
        try std.testing.expectEqual(@as(i32, 0x40000000), try inst.call1(i32, i32, "shru", @bitCast(@as(u32, 0x80000000))));
    }
}

test "wasm target: i32 -> i64 sign-extend conversion round-trips" {
    const allocator = std.testing.allocator;
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const i32t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, i32t);
    const w = try f.appendInst(b, i64t, .{ .convert = .{ .value = x } }); // i32 -> i64
    f.setTerminator(b, .{ .ret = w });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("sx", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i64, -5), try inst.call1(i64, i32, "sx", -5)); // sign-extended
    try std.testing.expectEqual(@as(i64, 7), try inst.call1(i64, i32, "sx", 7));
}

test "wasm target: i64 -> i32 wrap and f32 <-> f64 conversions round-trip" {
    const allocator = std.testing.allocator;
    var f_wrap = ir.function.Function.init(allocator);
    defer f_wrap.deinit();
    {
        const i32t = try f_wrap.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const i64t = try f_wrap.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
        const b = try f_wrap.appendBlock();
        const x = try f_wrap.appendBlockParam(b, i64t);
        const w = try f_wrap.appendInst(b, i32t, .{ .convert = .{ .value = x } }); // i64 -> i32 wrap
        f_wrap.setTerminator(b, .{ .ret = w });
    }
    var f_pd = ir.function.Function.init(allocator);
    defer f_pd.deinit();
    {
        // widen(x: f32) -> f64 then narrow back, times 2 to make truncation visible: (f32)((f64)x * 2)
        const f32t = try f_pd.types.intern(.{ .float = .f32 });
        const f64t = try f_pd.types.intern(.{ .float = .f64 });
        const b = try f_pd.appendBlock();
        const x = try f_pd.appendBlockParam(b, f32t);
        const wide = try f_pd.appendInst(b, f64t, .{ .convert = .{ .value = x } }); // f32 -> f64
        const two = try f_pd.appendInst(b, f64t, .{ .fconst = 2.0 });
        const scaled = try f_pd.appendInst(b, f64t, .{ .arith = .{ .op = .mul, .lhs = wide, .rhs = two } });
        const narrow = try f_pd.appendInst(b, f32t, .{ .convert = .{ .value = scaled } }); // f64 -> f32
        f_pd.setTerminator(b, .{ .ret = narrow });
    }

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("wrap", &f_wrap);
    try m.addFunction("pd", &f_pd);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    // 0x1_0000_0007 wraps to 7.
    try std.testing.expectEqual(@as(i32, 7), try inst.call1(i32, i64, "wrap", 0x1_0000_0007));
    try std.testing.expectEqual(@as(f32, 5.0), try inst.call1(f32, f32, "pd", 2.5));
}

/// Differential test: compile the same GLSL `float f(float)` through the aarch64
/// native backend and through the wasm target, and require they agree on `x`. A
/// mismatch is a bug in one backend that neither's own tests would catch alone.
fn diffGlslF32(allocator: std.mem.Allocator, src: []const u8, name: []const u8, x: f32) !void {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find(name) orelse return error.MissingFunction;

    // Native aarch64 path.
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const native_result = buf.entry(*const fn (f32) callconv(.c) f32, 0)(x);

    // Wasm target path (IR -> wasm -> frontend re-lower -> native JIT).
    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction(name, f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    const wasm_result = try inst.call1(f32, f32, name, x);

    std.testing.expectEqual(native_result, wasm_result) catch |e| {
        std.debug.print("\ndiff {s}({d}): native={d} wasm={d}\n", .{ name, x, native_result, wasm_result });
        return e;
    };
}

/// Float-argument, int-result differential (e.g. int(x) conversions).
fn diffGlslF2I(allocator: std.mem.Allocator, src: []const u8, name: []const u8, x: f32) !void {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find(name) orelse return error.MissingFunction;

    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const native_result = buf.entry(*const fn (f32) callconv(.c) i32, 0)(x);

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction(name, f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    const wasm_result = try inst.call1(i32, f32, name, x);

    std.testing.expectEqual(native_result, wasm_result) catch |e| {
        std.debug.print("\ndiff {s}({d}): native={d} wasm={d}\n", .{ name, x, native_result, wasm_result });
        return e;
    };
}

test "wasm target vs aarch64: float->int conversion (saturation)" {
    const allocator = std.testing.allocator;
    const src = "int f(float x){ return int(x); }";
    // In-range agree; the interesting cases are out of i32 range, where aarch64 fcvtzs
    // saturates. The wasm target must match (saturating trunc), not trap.
    for ([_]f32{ 2.7, -3.9, 0.0, 100.5, -100.5, 3.0e9, -3.0e9, 1.0e30 }) |x| {
        try diffGlslF2I(allocator, src, "f", x);
    }
}

/// Integer version of the differential test.
fn diffGlslI32(allocator: std.mem.Allocator, src: []const u8, name: []const u8, x: i32) !void {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find(name) orelse return error.MissingFunction;

    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const native_result = buf.entry(*const fn (i32) callconv(.c) i32, 0)(x);

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction(name, f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    const wasm_result = try inst.call1(i32, i32, name, x);

    std.testing.expectEqual(native_result, wasm_result) catch |e| {
        std.debug.print("\ndiff {s}({d}): native={d} wasm={d}\n", .{ name, x, native_result, wasm_result });
        return e;
    };
}

test "wasm target vs aarch64: differential over GLSL scalar functions" {
    const allocator = std.testing.allocator;
    const fcases = [_]struct { src: []const u8, name: []const u8, xs: []const f32 }{
        .{ .src = "float f(float x){ return x*x - 2.0*x + 1.0; }", .name = "f", .xs = &.{ 0.0, 1.0, 3.5, -2.0 } },
        .{ .src = "float f(float x){ return clamp(x, 0.0, 1.0); }", .name = "f", .xs = &.{ -1.0, 0.3, 2.0 } },
        .{ .src = "float f(float x){ return abs(x) + sqrt(abs(x)); }", .name = "f", .xs = &.{ 4.0, -9.0, 0.25 } },
        .{ .src = "float f(float x){ return x < 0.0 ? -x : x*2.0; }", .name = "f", .xs = &.{ -3.0, 5.0 } },
        .{ .src = "float f(float x){ return floor(x) + fract(x); }", .name = "f", .xs = &.{ 3.75, -1.25 } },
        .{ .src = "float f(float x){ return mix(1.0, 3.0, x); }", .name = "f", .xs = &.{ 0.0, 0.5, 1.0 } },
        .{ .src = "float f(float x){ float s=0.0; for(int i=0;i<3;i=i+1) s=s+x; return s; }", .name = "f", .xs = &.{ 2.0, -1.5 } },
        .{ .src = "float f(float x){ return sign(x) + step(0.5, x); }", .name = "f", .xs = &.{ -2.0, 0.0, 0.75 } },
        .{ .src = "float f(float x){ return mod(x, 3.0); }", .name = "f", .xs = &.{ 7.0, -1.0, 3.0 } },
        .{ .src = "float f(float x){ return max(min(x, 4.0), -4.0) / 2.0; }", .name = "f", .xs = &.{ 10.0, -10.0, 1.0 } },
    };
    inline for (fcases) |c| {
        for (c.xs) |x| try diffGlslF32(allocator, c.src, c.name, x);
    }

    const icases = [_]struct { src: []const u8, name: []const u8, xs: []const i32 }{
        .{ .src = "int f(int x){ return x*x + x; }", .name = "f", .xs = &.{ 0, 3, -4, 100 } },
        .{ .src = "int f(int x){ return x / 3 + x % 3; }", .name = "f", .xs = &.{ 10, -10, 7 } },
        .{ .src = "int f(int x){ if (x > 0) return 1; else if (x < 0) return -1; return 0; }", .name = "f", .xs = &.{ 5, -5, 0 } },
        .{ .src = "int f(int x){ int c=0; for(int i=0;i<x;i=i+1) if (i>1) c=c+i; return c; }", .name = "f", .xs = &.{ 5, 2, 0 } },
        .{ .src = "int f(int x){ return (x << 2) | (x & 3); }", .name = "f", .xs = &.{ 5, -1, 0 } },
        .{ .src = "int f(int x){ return ~x + (x ^ 15); }", .name = "f", .xs = &.{ 5, -1, 0, 255 } },
        .{ .src = "int f(int x){ return 1 << x; }", .name = "f", .xs = &.{ 0, 1, 15, 30, 31 } }, // 1<<31 = INT_MIN
        .{ .src = "int f(int x){ return x * x; }", .name = "f", .xs = &.{ 46341, 100000, -46341 } }, // i32 mul overflow wraps
        .{ .src = "int f(int x){ return x + 2000000000; }", .name = "f", .xs = &.{ 2000000000, -1, 500000000 } }, // add overflow wraps
        .{ .src = "int f(int x){ return x >> 31; }", .name = "f", .xs = &.{ -1, 1, -2147483648 } }, // arithmetic shift, sign fill
    };
    inline for (icases) |c| {
        for (c.xs) |x| try diffGlslI32(allocator, c.src, c.name, x);
    }
}

test "wasm target vs aarch64: differential over GLSL switch statements" {
    const allocator = std.testing.allocator;
    // A switch desugars to a nested if/else chain in the frontend; this checks the wasm
    // backend lowers that chain the same as aarch64 across matched, grouped, default, and
    // unmatched selectors.
    const src =
        \\int f(int x){
        \\  int r = 0;
        \\  switch (x) {
        \\    case 0: r = 10; break;
        \\    case 1:
        \\    case 2: r = 20; break;
        \\    case 3: r = 30; break;
        \\    default: r = 99; break;
        \\  }
        \\  return r;
        \\}
    ;
    for ([_]i32{ 0, 1, 2, 3, 7, -1 }) |x| try diffGlslI32(allocator, src, "f", x);

    // A switch nested inside a loop, with continue targeting the loop (skip i == 2).
    const src2 =
        \\int f(int n){
        \\  int sum = 0;
        \\  for (int i = 0; i < n; i = i + 1) {
        \\    switch (i) {
        \\      case 2: continue;
        \\      default: break;
        \\    }
        \\    sum = sum + i;
        \\  }
        \\  return sum;
        \\}
    ;
    for ([_]i32{ 0, 3, 5, 8 }) |n| try diffGlslI32(allocator, src2, "f", n);
}

test "wasm target vs aarch64: break and continue inside an if inside a loop" {
    const allocator = std.testing.allocator;
    // These exercise a `br` from inside a wasm `if` out to an enclosing loop scope, which
    // needs the if nesting counted in the branch depth, and the loop's continue block being
    // a real branch target (both paths of the if reconverge on it).
    const brk = "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ if(i==3) break; s=s+i; } return s; }";
    for ([_]i32{ 0, 2, 5, 10 }) |n| try diffGlslI32(allocator, brk, "f", n);

    const cont = "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ if(i==2) continue; s=s+i; } return s; }";
    for ([_]i32{ 0, 2, 5, 8 }) |n| try diffGlslI32(allocator, cont, "f", n);
}

test "wasm target vs aarch64: integer composites (arrays, structs)" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u8{
        // Integer array: constant-indexed stores + unrolled-loop read.
        "int f(int x){ int a[3]; a[0]=x; a[1]=x+1; a[2]=x+2; int s=0; for(int i=0;i<3;i=i+1){ s=s+a[i]; } return s; }",
        // Struct with int members.
        "struct S { int a; int b; }; int f(int x){ S s; s.a = x*2; s.b = x+7; return s.a - s.b; }",
        // Array of ivec2, accumulate components.
        "int f(int x){ ivec2 a[2]; a[0]=ivec2(x,1); a[1]=ivec2(2,x); int s=0; for(int i=0;i<2;i=i+1){ s = s + a[i].x*10 + a[i].y; } return s; }",
        // Uninitialized int array then partial fill (zero-init of the rest must be int 0).
        "int f(int x){ int a[4]; a[1]=x; return a[0]*1000 + a[1]; }",
    };
    inline for (cases) |src| {
        for ([_]i32{ 0, 1, 3, -2, 100 }) |x| try diffGlslI32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: integer vectors through all control flow" {
    const allocator = std.testing.allocator;
    // Integer-vector loop/if/inline phis were a class of native-codegen bug (f32-typed phi
    // params). Sweep ivec through every control-flow shape to catch stragglers.
    const cases = [_][]const u8{
        // ivec through if/else merge (mergeVal).
        "int f(int x){ ivec2 v; if(x > 0){ v = ivec2(x, 1); } else { v = ivec2(0, x); } return v.x*10 + v.y; }",
        // ivec through a ternary.
        "int f(int x){ ivec2 v = (x > 5) ? ivec2(x, 2) : ivec2(2, x); return v.x*10 + v.y; }",
        // ivec accumulator through a loop with a conditional update.
        "int f(int n){ ivec2 acc = ivec2(0, 0); for(int i=0;i<n;i=i+1){ if(i%2==0) acc = acc + ivec2(i, 1); else acc = acc + ivec2(1, i); } return acc.x*100 + acc.y; }",
        // ivec through nested loops.
        "int f(int n){ ivec2 acc = ivec2(0,0); for(int i=0;i<n;i=i+1){ for(int j=0;j<2;j=j+1){ acc = acc + ivec2(i, j); } } return acc.x*100 + acc.y; }",
        // uvec accumulator (unsigned) through a loop.
        "int f(int n){ uvec2 acc = uvec2(0u, 0u); for(int i=0;i<n;i=i+1){ acc = acc + uvec2(uint(i), 1u); } return int(acc.x*100u + acc.y); }",
        // ivec through inlined helper with ivec params inside a loop.
        "int dot2(ivec2 a, ivec2 b){ return a.x*b.x + a.y*b.y; } int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ s = s + dot2(ivec2(i,1), ivec2(2,i)); } return s; }",
    };
    inline for (cases) |src| {
        for ([_]i32{ 0, 1, 2, 3, 5, 8, -1 }) |x| try diffGlslI32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: complex programs combining features" {
    const allocator = std.testing.allocator;
    // Realistic mixes: control flow + vectors + integer + builtins, where feature
    // interactions (not single ops) are the likely bug source.
    const icases = [_][]const u8{
        // Loop accumulating an integer with a switch inside, min/max on the result.
        "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ switch(i%3){ case 0: s=s+i; break; case 1: s=s-1; break; default: s=s+bitCount(i); } } return clamp(s, -50, 50); }",
        // Nested loops with break/continue, integer vectors, and bitwise reduce.
        "int f(int n){ ivec2 acc = ivec2(0,0); for(int i=0;i<n;i=i+1){ if(i==5) break; for(int j=0;j<3;j=j+1){ if(j==i) continue; acc = acc + ivec2(i, j); } } return acc.x*100 + acc.y; }",
        // Bit manipulation pipeline.
        "int f(int x){ int r = bitfieldReverse(x); r = bitfieldExtract(r, 8, 16); return findMSB(r) + bitCount(r & 0xFF); }",
        // Fixed-point pack/unpack round-trip with arithmetic.
        "int f(int x){ float t = float(x & 0xFF) / 255.0; uint p = packUnorm4x8(vec4(t, t*0.5, 1.0-t, 1.0)); vec4 v = unpackUnorm4x8(p); return int((v.x + v.y + v.z + v.w) * 63.75 + 0.5); }",
    };
    inline for (icases) |src| {
        for ([_]i32{ 0, 1, 3, 5, 7, 12, 200, -1, 0x12345678 }) |x| try diffGlslI32(allocator, src, "f", x);
    }
    const fcases = [_][]const u8{
        // Vector math + control flow producing a float.
        "float f(float x){ vec3 v = vec3(x, x*0.5, 1.0); float acc = 0.0; for(int i=0;i<3;i=i+1){ if(v[i] > 0.5) acc = acc + sqrt(abs(v[i])); } return acc; }",
        // Nested conditionals selecting between builtin results.
        "float f(float x){ float r; if(x < 0.0){ r = -sqrt(-x); } else if(x < 1.0){ r = mix(0.0, 1.0, x); } else { r = floor(x) + fract(x)*2.0; } return clamp(r, -10.0, 10.0); }",
    };
    inline for (fcases) |src| {
        for ([_]f32{ -4.0, -0.5, 0.0, 0.3, 0.75, 2.5, 9.9 }) |x| try diffGlslF32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: float builtins" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u8{
        "float f(float x){ return sqrt(x*x + 1.0); }",
        "float f(float x){ return floor(x) + ceil(x*0.5); }",
        "float f(float x){ return fract(x) - trunc(x*0.25); }",
        "float f(float x){ return abs(x) + sign(x); }",
        "float f(float x){ return clamp(x, -1.0, 1.0) + mix(2.0, 4.0, fract(x)); }",
        "float f(float x){ return mod(x, 3.0) + step(0.5, fract(x)); }",
        "float f(float x){ return min(x, 2.0) * max(x, -2.0); }",
        "float f(float x){ return smoothstep(0.0, 1.0, x); }",
    };
    inline for (cases) |src| {
        for ([_]f32{ -3.25, -1.0, 0.0, 0.5, 1.75, 4.0, 7.3 }) |x| try diffGlslF32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: matrices" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u8{
        // mat2 * vec2, read a component.
        "float f(float x){ mat2 m = mat2(1.0, 2.0, 3.0, 4.0); vec2 r = m * vec2(x, 1.0); return r.x + r.y; }",
        // mat3 construction and mat*vec.
        "float f(float x){ mat3 m = mat3(x, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, x); vec3 r = m * vec3(1.0, 2.0, 3.0); return r.x + r.y + r.z; }",
        // mat2 * mat2 then apply.
        "float f(float x){ mat2 a = mat2(x, 1.0, 0.0, x); mat2 b = mat2(2.0, 0.0, 1.0, 2.0); mat2 c = a * b; vec2 r = c * vec2(1.0, 1.0); return r.x + r.y; }",
    };
    inline for (cases) |src| {
        for ([_]f32{ 0.0, 1.0, 2.5, -1.5 }) |x| try diffGlslF32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: integer bit builtins and pack/unpack" {
    const allocator = std.testing.allocator;
    // Bit builtins (SWAR + shifts/select) and pack/unpack (nearest-round + float<->int
    // convert). The round op and float->uint convert are the likely untested wasm spots.
    const cases = [_][]const u8{
        "int f(int x){ return bitCount(x); }",
        "int f(int x){ return findLSB(x); }",
        "int f(int x){ return findMSB(x); }",
        "int f(int x){ return findMSB(uint(x)); }",
        "int f(int x){ return bitfieldReverse(x); }",
        "int f(int x){ return bitfieldExtract(x, 4, 8); }",
        "int f(int x){ return int(bitfieldInsert(uint(x), uint(15), 4, 4)); }",
        "int f(int x){ float fx = float(x) * 0.01; return int(packUnorm4x8(vec4(fx, 0.0, 0.5, 1.0))); }",
        "int f(int x){ float fx = float(x) * 0.01; return int(packSnorm4x8(vec4(fx, -0.5, 1.0, -1.0))); }",
        "int f(int x){ vec4 v = unpackUnorm4x8(uint(x)); return int((v.x + v.z) * 255.0 + 0.5); }",
        "int f(int x){ vec4 v = unpackSnorm4x8(uint(x)); return int(v.y * 100.0); }",
    };
    inline for (cases) |src| {
        for ([_]i32{ 0, 1, 3, 0xFF, -1, 0x12345678, 0x663300FF }) |x| try diffGlslI32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: integer vectors, unsigned ops, and bit reinterpret" {
    const allocator = std.testing.allocator;
    // Cross-validate the new GLSL integer/unsigned-vector and reinterpret features on the
    // wasm backend. The reinterpret (unary op) and unsigned shift/divide are the likely
    // untested spots.
    const cases = [_][]const u8{
        // ivec integer division (truncates, unlike float).
        "int f(int x){ ivec3 v = ivec3(x, 3, 0); return v.x / v.y; }",
        // ivec component-wise add + bitwise.
        "int f(int x){ ivec2 a = ivec2(x,2); ivec2 b = ivec2(3,x); ivec2 c = a + b; return (c.x << 4) | (c.y & 7); }",
        // uvec unsigned right-shift (int->uint reinterpret + logical shift).
        "int f(int x){ uvec2 v = uvec2(x, 1); return int(v.x >> v.y); }",
        // uvec unsigned division of a high-bit value.
        "int f(int x){ uvec2 v = uvec2(x, 2); return int(v.x / v.y); }",
        // bit reinterpret round-trip through a float register.
        "int f(int x){ return floatBitsToInt(intBitsToFloat(x)); }",
        // float->int bits of a computed float.
        "int f(int x){ float g = float(x) * 0.5; return floatBitsToInt(g); }",
    };
    inline for (cases) |src| {
        for ([_]i32{ 0, 1, 3, 5, -1, -2, 1065353216 }) |x| try diffGlslI32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: deeply nested control flow" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u8{
        // Nested loops, break in the inner loop (breaks only the inner).
        "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ for(int j=0;j<n;j=j+1){ if(j==2) break; s=s+j; } s=s+i; } return s; }",
        // Nested loops, continue in the inner loop.
        "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ for(int j=0;j<n;j=j+1){ if(j==1) continue; s=s+j; } } return s; }",
        // break guarded by a doubly-nested if.
        "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ if(i>0){ if(i==3) break; } s=s+i; } return s; }",
        // Early return from inside an if inside a loop.
        "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ if(i==3) return s; s=s+i; } return s; }",
        // continue in the inner, break in the outer, both if-guarded.
        "int f(int n){ int s=0; for(int i=0;i<n;i=i+1){ if(i==4) break; for(int j=0;j<n;j=j+1){ if(j==i) continue; s=s+1; } } return s; }",
        // while loop with a break and a continue in if/else arms.
        "int f(int n){ int s=0; int i=0; while(i<n){ i=i+1; if(i==2) continue; if(i==5) break; s=s+i; } return s; }",
        // if/else where each arm loops (branch depth differs per arm).
        "int f(int n){ int s=0; if(n>0){ for(int i=0;i<n;i=i+1) s=s+i; } else { for(int i=0;i<3;i=i+1) s=s-i; } return s; }",
    };
    inline for (cases) |src| {
        for ([_]i32{ 0, 1, 3, 5, 8 }) |n| try diffGlslI32(allocator, src, "f", n);
    }
}

test "wasm target vs aarch64: float control flow with phis" {
    const allocator = std.testing.allocator;
    // Float-typed loop/if bodies build float block-param phis, a different codegen path
    // than the integer cases: accumulators, conditional accumulation, and early exit.
    const cases = [_][]const u8{
        "float f(float x){ float s=0.0; for(int i=0;i<5;i=i+1){ if(x>0.0) s=s+x; else s=s-x; } return s; }",
        "float f(float x){ float a=x; for(int i=0;i<4;i=i+1){ a=a*0.5; if(a<0.1) break; } return a; }",
        "float f(float x){ float s=0.0; for(int i=0;i<6;i=i+1){ if(x<0.0) continue; s=s+x*float(i); } return s; }",
        "float f(float x){ float r; if(x>1.0){ r=x*x; } else { r=x+x; } return r; }",
    };
    inline for (cases) |src| {
        for ([_]f32{ -2.0, 0.0, 0.5, 3.0 }) |x| try diffGlslF32(allocator, src, "f", x);
    }
}

test "wasm target vs aarch64: large function body (256+ byte encoding)" {
    const allocator = std.testing.allocator;
    // A long straight-line body compiles to well over 256 bytes of wasm, exercising the
    // multi-byte LEB128 body-length encoding end to end (compile AND run correct).
    const src =
        \\int f(int x){
        \\  int a=x; a=a+1; a=a*2; a=a-3; a=a+4; a=a*5; a=a-6; a=a+7; a=a*8; a=a-9;
        \\  a=a+10; a=a*11; a=a-12; a=a+13; a=a*14; a=a-15; a=a+16; a=a*17; a=a-18;
        \\  a=a+19; a=a*2; a=a-21; a=a+22; a=a*2; a=a-24; a=a+25; a=a*2; a=a-27;
        \\  a=a+28; a=a*2; a=a-30; a=a+31; a=a*2; a=a-33; a=a+34; a=a*2; a=a-36;
        \\  return a;
        \\}
    ;
    for ([_]i32{ 0, 1, -1, 7 }) |x| try diffGlslI32(allocator, src, "f", x);
}

test "wasm target: GLSL scalar function compiles and runs end-to-end" {
    const allocator = std.testing.allocator;
    // min/max lower to float compare + select, exercising the whole pipeline:
    // GLSL -> IR -> wasm target -> frontend re-lower -> native JIT -> run.
    const src =
        \\float clampf(float x) { return min(max(x, 0.0), 1.0); }
    ;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find("clampf") orelse return error.MissingFunction;

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("clampf", f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f32, 0.5), try inst.call1(f32, f32, "clampf", 0.5));
    try std.testing.expectEqual(@as(f32, 0.0), try inst.call1(f32, f32, "clampf", -2.0));
    try std.testing.expectEqual(@as(f32, 1.0), try inst.call1(f32, f32, "clampf", 3.0));
}

test "wasm target: GLSL if-inside-loop runs end-to-end" {
    const allocator = std.testing.allocator;
    // countPos(n): count i in [0, n) with i > 2. An if nested in a loop with a
    // conditional accumulator (phi through both the loop and the selection).
    const src =
        \\int countPos(int n) { int c = 0; for (int i = 0; i < n; i = i + 1) { if (i > 2) c = c + 1; } return c; }
    ;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find("countPos") orelse return error.MissingFunction;

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("countPos", f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 2), try inst.call1(i32, i32, "countPos", 5)); // i=3,4
    try std.testing.expectEqual(@as(i32, 0), try inst.call1(i32, i32, "countPos", 3));
}

test "wasm target: GLSL loop runs end-to-end" {
    const allocator = std.testing.allocator;
    // A for-loop sum: the frontend produces a header/body/exit loop with
    // cf.merge/cf.continue attrs, exercising the wasm target's loop emitter.
    const src =
        \\int sumn(int n) { int s = 0; for (int i = 1; i <= n; i = i + 1) s = s + i; return s; }
    ;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find("sumn") orelse return error.MissingFunction;

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("sumn", f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 15), try inst.call1(i32, i32, "sumn", 5));
    try std.testing.expectEqual(@as(i32, 55), try inst.call1(i32, i32, "sumn", 10));
}

test "wasm target: GLSL function with control flow runs end-to-end" {
    const allocator = std.testing.allocator;
    // An if with an early return: the frontend produces a real multi-block CFG with
    // cf.merge attrs, exercising the wasm target's structured control-flow emitter.
    const src =
        \\float absf(float x) { if (x < 0.0) return -x; return x; }
    ;
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find("absf") orelse return error.MissingFunction;

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("absf", f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f32, 3.0), try inst.call1(f32, f32, "absf", -3.0));
    try std.testing.expectEqual(@as(f32, 2.0), try inst.call1(f32, f32, "absf", 2.0));
}

test "wasm target: f32 select (min) round-trips" {
    const allocator = std.testing.allocator;
    // fmin(a, b) = (a < b) ? a : b, a float-operand select.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const ft = try f.types.intern(.{ .float = .f32 });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, ft);
    const b = try f.appendBlockParam(entry, ft);
    const lt = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    const r = try f.appendInst(entry, ft, .{ .select = .{ .cond = lt, .then = a, .@"else" = b } });
    f.setTerminator(entry, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("fmin", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f32, 2.5), try inst.call2(f32, f32, f32, "fmin", 2.5, 7.5));
    try std.testing.expectEqual(@as(f32, 1.0), try inst.call2(f32, f32, f32, "fmin", 4.0, 1.0));
}

test "wasm target: f32 arithmetic round-trips" {
    const allocator = std.testing.allocator;

    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const ft = try f.types.intern(.{ .float = .f32 });
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, ft);
    const b = try f.appendBlockParam(entry, ft);
    const sum = try f.appendInst(entry, ft, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    f.setTerminator(entry, .{ .ret = sum });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("faddf", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f32, 3.75), try inst.call2(f32, f32, f32, "faddf", 1.5, 2.25));
}

test "wasm target: direct call resolves by name not interning order" {
    const allocator = std.testing.allocator;

    // decoy(x)=x+1 at index 0, target(x)=x+100 at index 1, main(x)=target(x) at index 2.
    // In `main`, "target" is the first interned symbol (id 0). If the backend used the
    // symbol id as the function index it would call decoy (index 0) and return x+1.
    var decoy = ir.function.Function.init(allocator);
    defer decoy.deinit();
    var target = ir.function.Function.init(allocator);
    defer target.deinit();
    var mainf = ir.function.Function.init(allocator);
    defer mainf.deinit();
    inline for (.{ .{ &decoy, 1 }, .{ &target, 100 } }) |pair| {
        const fp = pair[0];
        const t = try fp.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try fp.appendBlock();
        const x = try fp.appendBlockParam(b, t);
        const r = try fp.appendArithImm(b, t, .add, x, pair[1]);
        fp.setTerminator(b, .{ .ret = r });
    }
    {
        const t = try mainf.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try mainf.appendBlock();
        const x = try mainf.appendBlockParam(b, t);
        const r = try mainf.appendCall(b, t, "target", &.{x});
        mainf.setTerminator(b, .{ .ret = r });
    }

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("decoy", &decoy);
    try m.addFunction("target", &target);
    try m.addFunction("main", &mainf);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 105), try inst.call1(i32, i32, "main", 5));
}

test "wasm target: int->float convert with mixed-type locals round-trips" {
    const allocator = std.testing.allocator;

    // scale(n: i32) -> f32 = float(n) * 2.5. Mixes an i32 param with f32 values, which
    // exercises the grouped-by-valtype local layout and float zero-init.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const it = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ft = try f.types.intern(.{ .float = .f32 });
    const entry = try f.appendBlock();
    const n = try f.appendBlockParam(entry, it);
    const nf = try f.appendInst(entry, ft, .{ .convert = .{ .value = n } });
    const k = try f.appendInst(entry, ft, .{ .fconst = 2.5 });
    const r = try f.appendInst(entry, ft, .{ .arith = .{ .op = .mul, .lhs = nf, .rhs = k } });
    f.setTerminator(entry, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("scale", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(f32, 10.0), try inst.call1(f32, i32, "scale", 4));
    try std.testing.expectEqual(@as(f32, -7.5), try inst.call1(f32, i32, "scale", -3));
}

test "wasm target: stack frame + control flow + cross-block alloca" {
    const allocator = std.testing.allocator;
    // absStore(x): slot = alloca; if x < 0 { store -x } else { store x }; return *slot.
    // The alloca (sp-relative) is written from two branches and read in the merge.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr = try f.types.intern(.ptr);
    const entry = try f.appendBlock();
    const x = try f.appendBlockParam(entry, t);
    const neg = try f.appendBlock();
    const pos = try f.appendBlock();
    const merge = try f.appendBlock();

    const slot = try f.appendInst(entry, ptr, .{ .alloca = .{ .elem = t } });
    const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
    const isneg = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = zero } });
    try f.appendIf(entry, isneg, .{ .target = neg, .args = &.{} }, .{ .target = pos, .args = &.{} });
    try f.addAttr(.{ .block = entry }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(merge) } } });

    const nx = try f.appendInst(neg, t, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = x } });
    try f.appendStore(neg, nx, slot);
    try f.setJump(neg, merge, &.{});

    try f.appendStore(pos, x, slot);
    try f.setJump(pos, merge, &.{});

    const r = try f.appendInst(merge, t, .{ .load = .{ .ptr = slot } });
    f.setTerminator(merge, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("absStore", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 5), try inst.call1(i32, i32, "absStore", -5));
    try std.testing.expectEqual(@as(i32, 3), try inst.call1(i32, i32, "absStore", 3));
}

test "wasm target: cross-call allocas do not alias (stack pointer)" {
    const allocator = std.testing.allocator;
    const i32t_of = struct {
        fn t(f: *ir.function.Function) !ir.types.Type {
            return f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        }
    }.t;

    // callee(): allocate a slot, store 7, return it. Writes to the shared stack.
    var callee = ir.function.Function.init(allocator);
    defer callee.deinit();
    {
        const t = try i32t_of(&callee);
        const ptr = try callee.types.intern(.ptr);
        const b = try callee.appendBlock();
        const slot = try callee.appendInst(b, ptr, .{ .alloca = .{ .elem = t } });
        const seven = try callee.appendInst(b, t, .{ .iconst = 7 });
        try callee.appendStore(b, seven, slot);
        const r = try callee.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        callee.setTerminator(b, .{ .ret = r });
    }
    // caller(): store 1000 into its own slot, call callee(), reload the slot, add.
    // With static offsets callee's store would clobber the slot (14). With a stack
    // pointer the frames are disjoint (1007).
    var caller = ir.function.Function.init(allocator);
    defer caller.deinit();
    {
        const t = try i32t_of(&caller);
        const ptr = try caller.types.intern(.ptr);
        const b = try caller.appendBlock();
        const slot = try caller.appendInst(b, ptr, .{ .alloca = .{ .elem = t } });
        const k = try caller.appendInst(b, t, .{ .iconst = 1000 });
        try caller.appendStore(b, k, slot);
        const c = try caller.appendCall(b, t, "callee", &.{});
        const reloaded = try caller.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        const sum = try caller.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = reloaded, .rhs = c } });
        caller.setTerminator(b, .{ .ret = sum });
    }

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("callee", &callee);
    try m.addFunction("caller", &caller);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 1007), try inst.call0(i32, "caller"));
}

test "wasm target: alloca + store/load through linear memory round-trips" {
    const allocator = std.testing.allocator;

    // memsum(a, b): store a and b into two distinct alloca slots, load them back, add.
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr = try f.types.intern(.ptr);
    const entry = try f.appendBlock();
    const a = try f.appendBlockParam(entry, t);
    const b = try f.appendBlockParam(entry, t);
    const p = try f.appendInst(entry, ptr, .{ .alloca = .{ .elem = t } });
    const q = try f.appendInst(entry, ptr, .{ .alloca = .{ .elem = t } });
    try f.appendStore(entry, a, p);
    try f.appendStore(entry, b, q);
    const x = try f.appendInst(entry, t, .{ .load = .{ .ptr = p } });
    const y = try f.appendInst(entry, t, .{ .load = .{ .ptr = q } });
    const sum = try f.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    f.setTerminator(entry, .{ .ret = sum });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("memsum", &f);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 30), try inst.call2(i32, i32, i32, "memsum", 10, 20));
    try std.testing.expectEqual(@as(i32, 7), try inst.call2(i32, i32, i32, "memsum", 5, 2));
}

/// Build a `(i32) -> i32` function computing `x * k` via arith_imm.
fn scaleFn(func: *ir.function.Function, k: i64) !void {
    const t_i32 = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const x = try func.appendBlockParam(b0, t_i32);
    const r = try func.appendInst(b0, t_i32, .{ .arith_imm = .{ .op = .mul, .lhs = x, .imm = k } });
    func.setTerminator(b0, .{ .ret = r });
}

test "wasm target: call_indirect lowers and round-trips" {
    const allocator = std.testing.allocator;

    // func0 double(x)=x*2, func1 triple(x)=x*3, func2 dispatch(sel,x)=table[sel](x).
    var f_double = ir.function.Function.init(allocator);
    defer f_double.deinit();
    try scaleFn(&f_double, 2);
    var f_triple = ir.function.Function.init(allocator);
    defer f_triple.deinit();
    try scaleFn(&f_triple, 3);

    var f_disp = ir.function.Function.init(allocator);
    defer f_disp.deinit();
    const t_i32 = try f_disp.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try f_disp.appendBlock();
    const sel = try f_disp.appendBlockParam(b0, t_i32);
    const x = try f_disp.appendBlockParam(b0, t_i32);
    const r = try f_disp.appendCallIndirect(b0, t_i32, sel, &.{x});
    f_disp.setTerminator(b0, .{ .ret = r });

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("double", &f_double);
    try m.addFunction("triple", &f_triple);
    try m.addFunction("dispatch", &f_disp);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    try std.testing.expectEqual(@as(i32, 10), try inst.call2(i32, i32, i32, "dispatch", 0, 5));
    try std.testing.expectEqual(@as(i32, 15), try inst.call2(i32, i32, i32, "dispatch", 1, 5));
}

/// Differential runner for a hand-built IR module (memory ops and calls, which the GLSL
/// path never emits): build once, run the `entry` function on aarch64 natively and through
/// the wasm target, and require the results match.
fn diffIRModule(allocator: std.mem.Allocator, funcs: []const native.ModuleFunction, entry: []const u8, arg: i32) !void {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;

    var jm = try native.jitModule(allocator, funcs);
    defer jm.deinit();
    const nat = jm.entry(*const fn (i32) callconv(.c) i32, entry).?(arg);

    var m = wtarget.link.Module.init(allocator);
    defer m.deinit();
    for (funcs) |ff| try m.addFunction(ff.name, ff.func);
    var linked = try wtarget.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);
    var inst = try wasm.Instance.instantiate(allocator, linked.module, &.{});
    defer inst.deinit();
    const w = try inst.call1(i32, i32, entry, arg);

    std.testing.expectEqual(nat, w) catch |e| {
        std.debug.print("\ndiffIR {s}({d}): native={d} wasm={d}\n", .{ entry, arg, nat, w });
        return e;
    };
}

fn i32ty(f: *ir.function.Function) !ir.types.Type {
    return f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "wasm vs aarch64: multiple allocas do not alias" {
    const allocator = std.testing.allocator;
    // f(x): a=x+1 in slot A, b=x*3 in slot B, return load(A)*load(B) - load(A).
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const t = try i32ty(&f);
    const ptr = try f.types.intern(.ptr);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const sa = try f.appendInst(b, ptr, .{ .alloca = .{ .elem = t } });
    const sb = try f.appendInst(b, ptr, .{ .alloca = .{ .elem = t } });
    const va = try f.appendArithImm(b, t, .add, x, 1);
    const vb = try f.appendArithImm(b, t, .mul, x, 3);
    try f.appendStore(b, va, sa);
    try f.appendStore(b, vb, sb);
    const la = try f.appendInst(b, t, .{ .load = .{ .ptr = sa } });
    const lb = try f.appendInst(b, t, .{ .load = .{ .ptr = sb } });
    const prod = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = la, .rhs = lb } });
    const res = try f.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = la } });
    f.setTerminator(b, .{ .ret = res });

    for ([_]i32{ 0, 1, 4, -2, 10 }) |x_val| {
        try diffIRModule(allocator, &.{.{ .name = "f", .func = &f }}, "f", x_val);
    }
}

test "wasm vs aarch64: nested calls with arguments" {
    const allocator = std.testing.allocator;
    // add3(p) = p+3; dbl(p) = p*2; f(x) = dbl(add3(x)) = (x+3)*2.
    var add3 = ir.function.Function.init(allocator);
    defer add3.deinit();
    {
        const t = try i32ty(&add3);
        const b = try add3.appendBlock();
        const p = try add3.appendBlockParam(b, t);
        const r = try add3.appendArithImm(b, t, .add, p, 3);
        add3.setTerminator(b, .{ .ret = r });
    }
    var dbl = ir.function.Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try i32ty(&dbl);
        const b = try dbl.appendBlock();
        const p = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendArithImm(b, t, .mul, p, 2);
        dbl.setTerminator(b, .{ .ret = r });
    }
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    {
        const t = try i32ty(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const c1 = try f.appendCall(b, t, "add3", &.{x});
        const c2 = try f.appendCall(b, t, "dbl", &.{c1});
        f.setTerminator(b, .{ .ret = c2 });
    }
    const funcs = [_]native.ModuleFunction{
        .{ .name = "add3", .func = &add3 },
        .{ .name = "dbl", .func = &dbl },
        .{ .name = "f", .func = &f },
    };
    for ([_]i32{ 0, 1, 5, -4 }) |x_val| try diffIRModule(allocator, &funcs, "f", x_val);
}

test "wasm vs aarch64: value live across a call and a memory slot" {
    const allocator = std.testing.allocator;
    // helper(p) = p*10; f(x): t=x+5 (lives across the call), c=helper(x),
    // store t into a slot, return load(slot)+c = (x+5) + x*10.
    var helper = ir.function.Function.init(allocator);
    defer helper.deinit();
    {
        const t = try i32ty(&helper);
        const b = try helper.appendBlock();
        const p = try helper.appendBlockParam(b, t);
        const r = try helper.appendArithImm(b, t, .mul, p, 10);
        helper.setTerminator(b, .{ .ret = r });
    }
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    {
        const t = try i32ty(&f);
        const ptr = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const tv = try f.appendArithImm(b, t, .add, x, 5);
        const slot = try f.appendInst(b, ptr, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, tv, slot);
        const c = try f.appendCall(b, t, "helper", &.{x});
        const lt = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        const res = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = lt, .rhs = c } });
        f.setTerminator(b, .{ .ret = res });
    }
    const funcs = [_]native.ModuleFunction{
        .{ .name = "helper", .func = &helper },
        .{ .name = "f", .func = &f },
    };
    for ([_]i32{ 0, 1, 7, -3, 100 }) |x_val| try diffIRModule(allocator, &funcs, "f", x_val);
}
