//! Execution tests for the Wasm frontend: assemble a Wasm module in-process, lower it
//! to Vulcan IR, JIT it for the host via `vulcan-target.native`, run it. Self-contained
//! (no external wasm tooling), runs natively in-process.

const std = @import("std");
const wasm = @import("vulcan-wasm");
const native = @import("vulcan-target").native;

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
