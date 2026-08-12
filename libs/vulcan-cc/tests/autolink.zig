//! This test proves the default-executable autolink. It runs the linked program end to end.
//! It drives the real `vcc` binary. The `build_options` module supplies the binary path, so
//! the test always runs the freshly built driver. It runs the driver exactly as a user
//! would: `vcc hello.c -o hello`, with no hand-supplied crt object, `-lc`, `-I`, or
//! `--dynamic-linker`. The driver discovers the host `crt1.o`, `libc.so.6`, and the per-arch
//! dynamic linker. It links a non-PIE `ET_EXEC` binary around them and embeds the loader as
//! `PT_INTERP`. So `./hello` runs on its own. The test checks that its stdout is exactly
//! `Hello, world!\n`. The test skips cleanly, and never fails, when the host toolchain is
//! absent. The host toolchain is a real `gcc` plus a discoverable `crt1.o`, `libc.so.6`, and
//! loader. Another test skips the same way when no host glibc development tree is present.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// This is the per-host dynamic-linker soname that the driver embeds. It mirrors
/// `vcc.zig`'s `interpSoname`. It is re-declared here because the driver is a separate
/// module. The toolchain probe below gates on the same loader that the linked `./hello`
/// binary will need.
fn hostInterpSoname() ?[]const u8 {
    return switch (builtin.cpu.arch) {
        .aarch64 => "ld-linux-aarch64.so.1",
        .x86_64 => "ld-linux-x86-64.so.2",
        .riscv64 => "ld-linux-riscv64-lp64d.so.1",
        .x86 => "ld-linux.so.2",
        else => null,
    };
}

/// Returns true when the host C runtime that autolink needs is discoverable. This means a
/// real `gcc` whose `crt1.o` directory also holds `libc.so.6` and the host loader. This is
/// the same all-three-in-one-directory rule that the driver's `discoverToolchain` applies.
/// A `false` result means the test skips rather than fails. The toolchain is missing, not VCC.
fn toolchainPresent(allocator: std.mem.Allocator, io: std.Io, soname: []const u8) bool {
    const script = std.fmt.allocPrint(allocator,
        \\command -v gcc >/dev/null 2>&1 || exit 1
        \\c=$(gcc -print-file-name=crt1.o 2>/dev/null)
        \\case "$c" in /*) ;; *) exit 1;; esac
        \\d=$(dirname "$c")
        \\[ -f "$d/crt1.o" ] && [ -f "$d/libc.so.6" ] && [ -f "$d/{s}" ] || exit 1
        \\exit 0
    , .{soname}) catch return false;
    defer allocator.free(script);
    const proc = std.process.run(allocator, io, .{ .argv = &.{ "sh", "-c", script } }) catch return false;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    return switch (proc.term) {
        .exited => |code| code == 0,
        else => false,
    };
}

test "vcc hello.c -o hello autolinks the host crt + libc + loader, and ./hello prints" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const soname = hostInterpSoname() orelse return error.SkipZigTest;
    if (!toolchainPresent(allocator, io, soname)) return error.SkipZigTest;

    // This is a real printf hello program. The driver compiles it, links it, and runs it.
    // Every path below is relative to the test's own working directory, the repo root. So no
    // child process ever needs its working directory changed. `build_options.vcc_bin` is a
    // repo-relative path from `getEmittedBin`. A path that contains a `/` resolves against
    // the working directory, not `PATH`. `.zig-cache/tmp/<name>` is the tmp dir's on-disk
    // path. It matches the path used by other driver-wiring tests in this repo.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{
        .sub_path = "hello.c",
        .data = "#include <stdio.h>\nint main(void){ printf(\"Hello, world!\\n\"); return 0; }\n",
    });
    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);
    const src_path = try std.fs.path.join(allocator, &.{ dir_path, "hello.c" });
    defer allocator.free(src_path);
    const out_path = try std.fs.path.join(allocator, &.{ dir_path, "hello" });
    defer allocator.free(out_path);

    // Run `vcc <tmp>/hello.c -o <tmp>/hello`. Supply no crt object, `-lc`, `-I`, or
    // `--dynamic-linker`.
    const compile = try std.process.run(allocator, io, .{
        .argv = &.{ build_options.vcc_bin, src_path, "-o", out_path },
    });
    defer allocator.free(compile.stdout);
    defer allocator.free(compile.stderr);
    switch (compile.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("vcc autolink failed (exit {d}):\n{s}\n", .{ code, compile.stderr });
            return error.TestUnexpectedResult;
        },
        else => return error.TestUnexpectedResult,
    }

    // This step is the real test: run the linked binary. Its `PT_INTERP` is embedded, so it
    // runs on its own. Retry only on a non-clean process exit, such as a spawn race. Never
    // retry in a way that could mask a wrong-output regression.
    var attempt: usize = 0;
    while (true) {
        attempt += 1;
        const run = try std.process.run(allocator, io, .{ .argv = &.{out_path} });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        switch (run.term) {
            .exited => |code| {
                try std.testing.expectEqual(@as(u8, 0), code);
                try std.testing.expectEqualStrings("Hello, world!\n", run.stdout);
                return;
            },
            else => {
                if (attempt >= 5) return error.TestUnexpectedResult;
                std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
                continue;
            },
        }
    }
}
