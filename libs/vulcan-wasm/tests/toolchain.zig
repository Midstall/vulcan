//! Execution tests over Wasm modules that a real toolchain produced: `zig cc` drives clang
//! and lld, and the result goes through the loader, the JIT, and the WASI runtime.
//!
//! Every other Wasm test in this repo assembles its module byte by byte. That is how a fault
//! that stopped every linker-produced module from loading stayed invisible: no module from a
//! compiler had ever run. A hand-built module only contains the shapes its author thought of,
//! and the null table slot at index 0 was not one of them.
//!
//! The modules are built at test time, not checked in. `zig cc` is part of the toolchain that
//! builds this test, it needs no network, and it caches its output, so a rebuild costs
//! milliseconds. A checked-in Wasm blob is the worse option: a reviewer cannot read it in a
//! diff, and it cannot follow the linker whose output shape is the thing under test. The
//! `zig` binary comes through `build_options`, so the test uses the same one that built it and
//! does not depend on the PATH.
//!
//! A test skips, and never fails, when the toolchain cannot produce a Wasm module.

const std = @import("std");
const wasm = @import("vulcan-wasm");
const build_options = @import("build_options");

/// Compile `source` into `tmp/m.wasm` with `zig cc` and return the module bytes, which the
/// caller owns. `flags` carries the target and the link options. Returns `error.NoToolchain`
/// when `zig` cannot be run at all, so the caller skips instead of failing.
fn buildModule(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp: *std.testing.TmpDir,
    source: []const u8,
    flags: []const []const u8,
) ![]u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "m.c", .data = source });

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &.{ build_options.zig_exe, "cc" });
    try argv.appendSlice(allocator, flags);
    try argv.appendSlice(allocator, &.{ "m.c", "-o", "m.wasm" });

    const built = std.process.run(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoToolchain,
        else => return err,
    };
    defer allocator.free(built.stdout);
    defer allocator.free(built.stderr);
    if (built.term != .exited or built.term.exited != 0) {
        // The toolchain is present but rejected our own source, which is a real failure.
        std.debug.print("zig cc failed:\n{s}\n--- source ---\n{s}\n", .{ built.stderr, source });
        return error.CompileFailed;
    }

    return tmp.dir.readFileAlloc(io, "m.wasm", allocator, .limited(64 * 1024 * 1024));
}

/// The path of `tmp/m.wasm` relative to the test's working directory, which is the repo
/// root. A child process needs a path, not the open directory handle.
fn modulePath(allocator: std.mem.Allocator, tmp: *const std.testing.TmpDir) ![]u8 {
    return std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path, "m.wasm" });
}

/// The host side of the module's one import.
fn addOne(_: ?*anyopaque, x: i32) callconv(.c) i32 {
    return x + 1;
}

test "a module clang and lld produced instantiates and dispatches through its table" {
    // The C here compiles to the shape that used to be refused: an imported function, plus a
    // static array of function pointers, which lld turns into an element segment that starts
    // at table offset 1. Slot 0 stays empty so that a null function pointer keeps the value
    // zero and stays distinct from a real one.
    //
    // `tbl` holds table indices in linear memory, so the guest picks the slot itself. The
    // test therefore does not depend on which slot lld gave each function.
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const source =
        \\__attribute__((import_module("env"), import_name("addone")))
        \\extern int addone(int x);
        \\typedef int (*fp)(int);
        \\static int dbl(int x) { return x * 2; }
        \\static int tri(int x) { return x * 3; }
        \\static fp tbl[2] = { dbl, tri };
        \\__attribute__((export_name("dispatch")))
        \\int dispatch(int sel, int x) { return addone(tbl[sel & 1](x)); }
        \\
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = buildModule(allocator, io, &tmp, source, &.{
        "-target", "wasm32-freestanding", "-nostdlib", "-Wl,--no-entry", "-Wl,--strip-all", "-O2",
    }) catch |err| switch (err) {
        error.NoToolchain => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(bytes);

    var inst = try wasm.Instance.instantiate(allocator, bytes, &.{@intFromPtr(&addOne)});
    defer inst.deinit();

    // The shape under test. If a future linker fills slot 0, these are the lines that say so.
    try std.testing.expect(inst.module.table.len >= 3); // the null slot plus the two functions
    try std.testing.expectEqual(@as(?u32, null), inst.module.table[0]);
    try std.testing.expectEqual(@as(usize, 0), inst.table[0]);

    try std.testing.expectEqual(@as(i32, 11), try inst.call2(i32, i32, i32, "dispatch", 0, 5)); // addone(dbl(5))
    try std.testing.expectEqual(@as(i32, 16), try inst.call2(i32, i32, i32, "dispatch", 1, 5)); // addone(tri(5))
}

test "vulcan-wasm prints the output of a wasm32-wasi program built by zig cc" {
    // The whole path end to end: clang and wasi-libc produce the module, lld lays out its
    // table and data segments, the loader reads it, the JIT compiles it for the host, and the
    // WASI runtime carries `printf` out to fd 1. The CLI comes from `build_options`, so this
    // runs the freshly built binary.
    //
    // The program is built at `-Os`. The default `-O0` build of the same source also runs and
    // prints the same line, but it is 920 KB of unoptimized wasi-libc, and the JIT needs
    // about four minutes for it, which is too slow for a test. `-O1` and above are quick.
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const source =
        \\#include <stdio.h>
        \\int main(void) { printf("hello from wasi\n"); return 0; }
        \\
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = buildModule(allocator, io, &tmp, source, &.{ "-target", "wasm32-wasi", "-Os" }) catch |err| switch (err) {
        error.NoToolchain => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(bytes);
    // The bytes only prove a module came out. The CLI opens the same file for itself.
    try std.testing.expect(std.mem.startsWith(u8, bytes, "\x00asm"));

    const path = try modulePath(allocator, &tmp);
    defer allocator.free(path);
    const ran = try std.process.run(allocator, io, .{ .argv = &.{ build_options.vulcan_wasm_bin, path } });
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);

    try std.testing.expectEqual(std.process.Child.Term{ .exited = 0 }, ran.term);
    try std.testing.expectEqualStrings("hello from wasi\n", ran.stdout);
}
