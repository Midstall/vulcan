//! Shared x86-64 execution-test harness, parameterized by a `Backend`. cases.zig builds
//! IR functions and asserts results through `expectRun`. Each runner supplies its backend.
//! qemu.zig builds a static ELF and runs it under qemu-x86_64. native.zig maps the code
//! into W^X memory and calls it in-process (only on an x86-64 host).
//!
//! The plain runners (`runFunc`/`expectRun` and friends) are checked modulo 256 (a process
//! exit code is the low byte), so test values must differ in their low byte.
//!
//! The parallel `...Full` runners (`runFuncFull`/`expectRunFull`, `runFloatFuncFull`,
//! `runDoubleFuncFull`) carry the WHOLE result back over the child's stdout instead of the
//! exit-code low byte, so a test can assert an exact i64/f32/f64 value with no mod-256 limit.
//! They are additive: the plain runners and the ELF/stub bytes they build stay unchanged.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const encode = @import("../encode.zig");
const isel = @import("../isel.zig");
const link = @import("../link.zig");
const elf = @import("../elf.zig");
const jit = @import("../../coherent_jit.zig");

const Function = ir.function.Function;
const Reg = encode.Reg;

/// How a test runs compiled code. `qemu_cmd` runs a built ELF under an emulator. `native`
/// calls the code in-process (valid only when the host is x86-64).
pub const Backend = struct {
    name: []const u8,
    qemu_cmd: ?[]const u8 = null,
    native: bool = false,
};

pub const qemu = Backend{ .name = "qemu-x86_64", .qemu_cmd = "qemu-x86_64" };
pub const native = Backend{ .name = "native-x86_64", .native = true };

/// Load each integer argument into its System V register (rdi, rsi, rdx, rcx, r8, r9),
/// appending onto `s`. Shared by `buildStub` and `buildStubFull`.
fn appendIntArgs(allocator: std.mem.Allocator, s: *std.ArrayList(u8), args: []const i64) std.mem.Allocator.Error!void {
    const arg_regs = [_]Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
    // A full 64-bit immediate load (not the i32-immediate `movImm`): an i64-typed test argument
    // may carry a high-bit value (e.g. 0x1_0000_0000, used to prove a 64-bit-width compare), which
    // does not fit `movImm`'s i32 field.
    for (args, 0..) |a, i| try s.appendSlice(allocator, encode.movImm64(arg_regs[i], @bitCast(a)).slice());
}

/// Append the full-width result tail: `push rax` (the function's result, already in `rax`),
/// `write(1, rsp, nbytes)`, then `exit(0)`. Used by the `...Full` stubs so a test can read the
/// WHOLE result back from the child's stdout instead of only an exit-code low byte.
fn appendWriteResultTail(allocator: std.mem.Allocator, s: *std.ArrayList(u8), nbytes: i32) std.mem.Allocator.Error!void {
    try s.appendSlice(allocator, encode.pushReg(.rax).slice()); // result bytes now at [rsp, rsp+8)
    try s.appendSlice(allocator, encode.movImm(.rdi, 1, true).slice()); // fd = stdout
    try s.appendSlice(allocator, encode.movReg(.rsi, .rsp).slice()); // buf = rsp
    try s.appendSlice(allocator, encode.movImm(.rdx, nbytes, true).slice()); // count = nbytes
    try s.appendSlice(allocator, encode.movImm(.rax, 1, true).slice()); // sys_write
    try s.appendSlice(allocator, encode.syscall().slice());
    try s.appendSlice(allocator, encode.movImm(.rax, 60, true).slice()); // sys_exit
    try s.appendSlice(allocator, encode.xorr(.rdi, .rdi, false).slice()); // status = 0
    try s.appendSlice(allocator, encode.syscall().slice());
}

/// Build the entry stub: load each argument into its System V register, `call` the
/// code right after the stub, then `exit(result)`.
fn buildStub(allocator: std.mem.Allocator, args: []const i64) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    try appendIntArgs(allocator, &s, args);

    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());

    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
    return s.toOwnedSlice(allocator);
}

/// Like `buildStub`, but the tail writes the full 8-byte result to stdout and exits 0,
/// instead of exiting with its low byte.
fn buildStubFull(allocator: std.mem.Allocator, args: []const i64) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    try appendIntArgs(allocator, &s, args);

    var tail: std.ArrayList(u8) = .empty;
    defer tail.deinit(allocator);
    try appendWriteResultTail(allocator, &tail, 8);

    try s.appendSlice(allocator, encode.callRel(@intCast(tail.items.len)).slice());
    try s.appendSlice(allocator, tail.items);
    return s.toOwnedSlice(allocator);
}

/// Run a single function with `args` and return the low byte of its result.
pub fn runFunc(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, backend: Backend) !u8 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    return runCode(io, allocator, code, 0, args, backend);
}

/// Run a single function with `args` and return its FULL i64 result (no mod-256 truncation).
pub fn runFuncFull(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, backend: Backend) !i64 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    return runCodeFull(io, allocator, code, 0, args, backend);
}

/// Link `module` and run its `main` (which must be the first function, at offset 0).
pub fn runModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, backend: Backend) !u8 {
    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    const entry = linked.addressOf("main") orelse return error.UndefinedSymbol;
    std.debug.assert(entry == 0); // the stub calls the code at offset 0
    return runCode(io, allocator, linked.code, 0, args, backend);
}

/// Link `module` and run its `main` with f32 `fargs`, returning the low byte of the f32
/// result's bits (qemu only, like `runFloatFunc`).
pub fn runFloatModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, fargs: []const f32, backend: Backend) !u8 {
    if (backend.qemu_cmd == null) return error.SkipZigTest;
    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    const entry = linked.addressOf("main") orelse return error.UndefinedSymbol;
    std.debug.assert(entry == 0);
    const stub = try buildFloatStub(allocator, fargs);
    defer allocator.free(stub);
    return runProgram(io, allocator, stub, linked.code, backend);
}

pub fn expectRunFloatModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, fargs: []const f32, expected: f32, backend: Backend) !void {
    const want: u8 = @truncate(@as(u32, @bitCast(expected)));
    try std.testing.expectEqual(want, try runFloatModule(io, allocator, module, fargs, backend));
}

/// Call `code` in-process (native x86-64 host only) with `args`, returning the callee's full i64.
/// Shared by `runCode` (which truncates to the low byte) and `runCodeFull` (which keeps all 64 bits).
fn callNative(code: []const u8, entry_off: usize, args: []const i64) !i64 {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    var buf = try jit.CodeBuffer.map(code);
    defer buf.deinit();
    return switch (args.len) {
        0 => buf.entry(*const fn () callconv(.c) i64, entry_off)(),
        1 => buf.entry(*const fn (i64) callconv(.c) i64, entry_off)(args[0]),
        2 => buf.entry(*const fn (i64, i64) callconv(.c) i64, entry_off)(args[0], args[1]),
        3 => buf.entry(*const fn (i64, i64, i64) callconv(.c) i64, entry_off)(args[0], args[1], args[2]),
        else => return error.Unsupported,
    };
}

/// Run `code` (entered at `entry_off`) with `args`. For the native backend the code
/// is called in-process, otherwise it is wrapped in an ELF and run under qemu.
fn runCode(io: std.Io, allocator: std.mem.Allocator, code: []const u8, entry_off: usize, args: []const i64, backend: Backend) !u8 {
    if (backend.native) return @truncate(@as(u64, @bitCast(try callNative(code, entry_off, args))));

    const stub = try buildStub(allocator, args);
    defer allocator.free(stub);
    return runProgram(io, allocator, stub, code, backend);
}

/// Like `runCode`, but returns the FULL i64 result (no mod-256 truncation): the native path
/// returns the callee's i64 return value directly, the qemu path uses the full-width stub and
/// reads all 8 result bytes back from stdout.
fn runCodeFull(io: std.Io, allocator: std.mem.Allocator, code: []const u8, entry_off: usize, args: []const i64, backend: Backend) !i64 {
    if (backend.native) return callNative(code, entry_off, args);

    const stub = try buildStubFull(allocator, args);
    defer allocator.free(stub);
    const raw = try runProgramFull(io, allocator, stub, code, backend, 8);
    return @bitCast(raw);
}

/// Build a static ELF from `stub ++ code`, run it under qemu, and hand back the process's
/// exit term plus everything it wrote to stdout (caller owns `stdout`, must free it). Shared
/// by `runProgram` (exit-code low byte) and `runProgramFull` (whole result via stdout).
fn runImage(io: std.Io, allocator: std.mem.Allocator, stub: []const u8, code: []const u8, backend: Backend) !struct { term: std.process.Child.Term, stdout: []u8 } {
    const program = try allocator.alloc(u8, stub.len + code.len);
    defer allocator.free(program);
    @memcpy(program[0..stub.len], stub);
    @memcpy(program[stub.len..], code);
    const image = try elf.writeExec(allocator, program, 0);
    defer allocator.free(image);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.elf", .data = image, .flags = .{ .permissions = .executable_file } });
    const result = std.process.run(allocator, io, .{
        // `-cpu max` exposes AVX (and the rest) so the 256-bit YMM path can run. The default
        // qemu64 model has no AVX. SSE-only cases are unaffected.
        .argv = &.{ backend.qemu_cmd.?, "-cpu", "max", "a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stderr);
    return .{ .term = result.term, .stdout = result.stdout };
}

/// Wrap `stub ++ code` in a static ELF and run it under qemu, returning the exit code.
fn runProgram(io: std.Io, allocator: std.mem.Allocator, stub: []const u8, code: []const u8, backend: Backend) !u8 {
    const img = try runImage(io, allocator, stub, code, backend);
    defer allocator.free(img.stdout);
    return switch (img.term) {
        .exited => |c| c,
        else => error.BackendFailed,
    };
}

/// Like `runProgram`, but for a full-width stub (see `appendWriteResultTail`): the process
/// must exit 0 (a nonzero exit means the code under test faulted before reaching the write),
/// and the result is the `nbytes` the tail wrote to stdout, read back little-endian.
fn runProgramFull(io: std.Io, allocator: std.mem.Allocator, stub: []const u8, code: []const u8, backend: Backend, nbytes: usize) !u64 {
    const img = try runImage(io, allocator, stub, code, backend);
    defer allocator.free(img.stdout);
    switch (img.term) {
        .exited => |c| if (c != 0) return error.BackendFailed,
        else => return error.BackendFailed,
    }
    // The full-width stub tail writes exactly `nbytes` to stdout then exits; the code under
    // test itself never touches stdout, so this is always available.
    std.debug.assert(img.stdout.len >= nbytes);
    var v: u64 = 0;
    for (0..nbytes) |i| v |= @as(u64, img.stdout[i]) << @intCast(i * 8);
    return v;
}

/// Run raw compiled `code` (entered at offset 0) with integer `args`, returning the low byte of
/// its result. Used by the shared-Wimmer differential tests, which compile through
/// `isel.compileFunctionWimmerX86` (bytes, not a `Function`) and diff against `runFunc`.
pub fn runCodeInt(io: std.Io, allocator: std.mem.Allocator, code: []const u8, args: []const i64, backend: Backend) !u8 {
    return runCode(io, allocator, code, 0, args, backend);
}

/// Like `runCodeInt`, but returns the FULL i64 result (no mod-256 truncation).
pub fn runCodeIntFull(io: std.Io, allocator: std.mem.Allocator, code: []const u8, args: []const i64, backend: Backend) !i64 {
    return runCodeFull(io, allocator, code, 0, args, backend);
}

/// Run raw compiled `code` with f32 `fargs` (loaded into xmm0.., result read from xmm0), returning
/// the low byte of the f32 result's bits. qemu only, like `runFloatFunc`.
pub fn runCodeFloat(io: std.Io, allocator: std.mem.Allocator, code: []const u8, fargs: []const f32, backend: Backend) !u8 {
    if (backend.qemu_cmd == null) return error.SkipZigTest;
    const stub = try buildFloatStub(allocator, fargs);
    defer allocator.free(stub);
    return runProgram(io, allocator, stub, code, backend);
}

/// Load each f32 argument's bits into xmm0,xmm1,.. (via a general register), appending onto
/// `s`. Shared by `buildFloatStub` and `buildFloatStubFull`.
fn appendFloatArgs(allocator: std.mem.Allocator, s: *std.ArrayList(u8), fargs: []const f32) std.mem.Allocator.Error!void {
    const xmm_args = [_]encode.Xmm{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };
    for (fargs, 0..) |fa, i| {
        try s.appendSlice(allocator, encode.movImm(.rax, @bitCast(@as(u32, @bitCast(fa))), true).slice());
        try s.appendSlice(allocator, encode.movdToXmm(xmm_args[i], .rax).slice());
    }
}

/// The float entry stub: load each f32 argument's bits into xmm0,xmm1,..., `call` the code,
/// move the f32 result (xmm0) back to a general register, and `exit` with its low byte.
fn buildFloatStub(allocator: std.mem.Allocator, fargs: []const f32) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    try appendFloatArgs(allocator, &s, fargs);
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movdFromXmm(.rax, .xmm0).slice()); // f32 result bits -> eax
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());
    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
    return s.toOwnedSlice(allocator);
}

/// Like `buildFloatStub`, but the tail writes the full 4-byte f32 result to stdout and exits
/// 0, instead of exiting with its low byte.
fn buildFloatStubFull(allocator: std.mem.Allocator, fargs: []const f32) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    try appendFloatArgs(allocator, &s, fargs);

    var tail: std.ArrayList(u8) = .empty;
    defer tail.deinit(allocator);
    try tail.appendSlice(allocator, encode.movdFromXmm(.rax, .xmm0).slice()); // f32 result bits -> eax
    try appendWriteResultTail(allocator, &tail, 4);

    try s.appendSlice(allocator, encode.callRel(@intCast(tail.items.len)).slice());
    try s.appendSlice(allocator, tail.items);
    return s.toOwnedSlice(allocator);
}

/// Run a scalar-float function with f32 `fargs` and return the low byte of its f32 result's
/// bits (qemu only, a process exit code is one byte, so test values must differ there).
pub fn runFloatFunc(io: std.Io, allocator: std.mem.Allocator, func: *const Function, fargs: []const f32, backend: Backend) !u8 {
    if (backend.qemu_cmd == null) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    const stub = try buildFloatStub(allocator, fargs);
    defer allocator.free(stub);
    return runProgram(io, allocator, stub, code, backend);
}

/// Assert a scalar-float function returns `expected` (checked by the low byte of its bits).
pub fn expectRunFloat(io: std.Io, allocator: std.mem.Allocator, func: *const Function, fargs: []const f32, expected: f32, backend: Backend) !void {
    const want: u8 = @truncate(@as(u32, @bitCast(expected)));
    try std.testing.expectEqual(want, try runFloatFunc(io, allocator, func, fargs, backend));
}

/// Run a scalar-float function with f32 `fargs` and return its FULL f32 result (no mod-256
/// limit on its bits). Qemu only, like `runFloatFunc`.
pub fn runFloatFuncFull(io: std.Io, allocator: std.mem.Allocator, func: *const Function, fargs: []const f32, backend: Backend) !f32 {
    if (backend.qemu_cmd == null) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    const stub = try buildFloatStubFull(allocator, fargs);
    defer allocator.free(stub);
    const raw = try runProgramFull(io, allocator, stub, code, backend, 4);
    return @bitCast(@as(u32, @truncate(raw)));
}

/// Assert a scalar-float function returns exactly `expected` (its full bits, not just the low
/// byte).
pub fn expectRunFloatFull(io: std.Io, allocator: std.mem.Allocator, func: *const Function, fargs: []const f32, expected: f32, backend: Backend) !void {
    try std.testing.expectEqual(expected, try runFloatFuncFull(io, allocator, func, fargs, backend));
}

/// Load each f64 argument's 64 bits into xmm0.. via movq, appending onto `s`. Shared by
/// `buildDoubleStub` and `buildDoubleStubFull`.
fn appendDoubleArgs(allocator: std.mem.Allocator, s: *std.ArrayList(u8), dargs: []const f64) std.mem.Allocator.Error!void {
    const xmm_args = [_]encode.Xmm{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };
    for (dargs, 0..) |da, i| {
        try s.appendSlice(allocator, encode.movImm64(.rax, @bitCast(da)).slice());
        try s.appendSlice(allocator, encode.movqToXmm(xmm_args[i], .rax).slice());
    }
}

/// Like `buildFloatStub` but for f64: load each argument's 64 bits into xmm0.. via movq, and
/// read the f64 result's bits out of xmm0 with movq.
fn buildDoubleStub(allocator: std.mem.Allocator, dargs: []const f64) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    try appendDoubleArgs(allocator, &s, dargs);
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movqFromXmm(.rax, .xmm0).slice()); // f64 result bits -> rax
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());
    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
    return s.toOwnedSlice(allocator);
}

/// Like `buildDoubleStub`, but the tail writes the full 8-byte f64 result to stdout and exits
/// 0, instead of exiting with its low byte.
fn buildDoubleStubFull(allocator: std.mem.Allocator, dargs: []const f64) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    try appendDoubleArgs(allocator, &s, dargs);

    var tail: std.ArrayList(u8) = .empty;
    defer tail.deinit(allocator);
    try tail.appendSlice(allocator, encode.movqFromXmm(.rax, .xmm0).slice()); // f64 result bits -> rax
    try appendWriteResultTail(allocator, &tail, 8);

    try s.appendSlice(allocator, encode.callRel(@intCast(tail.items.len)).slice());
    try s.appendSlice(allocator, tail.items);
    return s.toOwnedSlice(allocator);
}

/// Run a scalar-double function with f64 `dargs`, returning the low byte of the f64 result's
/// bits (qemu only).
pub fn runDoubleFunc(io: std.Io, allocator: std.mem.Allocator, func: *const Function, dargs: []const f64, backend: Backend) !u8 {
    if (backend.qemu_cmd == null) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    const stub = try buildDoubleStub(allocator, dargs);
    defer allocator.free(stub);
    return runProgram(io, allocator, stub, code, backend);
}

pub fn expectRunDouble(io: std.Io, allocator: std.mem.Allocator, func: *const Function, dargs: []const f64, expected: f64, backend: Backend) !void {
    const want: u8 = @truncate(@as(u64, @bitCast(expected)));
    try std.testing.expectEqual(want, try runDoubleFunc(io, allocator, func, dargs, backend));
}

/// Run a scalar-double function with f64 `dargs` and return its FULL f64 result (no mod-256
/// limit on its bits). Qemu only, like `runDoubleFunc`.
pub fn runDoubleFuncFull(io: std.Io, allocator: std.mem.Allocator, func: *const Function, dargs: []const f64, backend: Backend) !f64 {
    if (backend.qemu_cmd == null) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    const stub = try buildDoubleStubFull(allocator, dargs);
    defer allocator.free(stub);
    const raw = try runProgramFull(io, allocator, stub, code, backend, 8);
    return @bitCast(raw);
}

/// Assert a scalar-double function returns exactly `expected` (its full bits, not just the low
/// byte).
pub fn expectRunDoubleFull(io: std.Io, allocator: std.mem.Allocator, func: *const Function, dargs: []const f64, expected: f64, backend: Backend) !void {
    try std.testing.expectEqual(expected, try runDoubleFuncFull(io, allocator, func, dargs, backend));
}

pub fn expectRun(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, expected: i64, backend: Backend) !void {
    const want: u8 = @truncate(@as(u64, @bitCast(expected)));
    try std.testing.expectEqual(want, try runFunc(io, allocator, func, args, backend));
}

/// Assert a function returns exactly `expected` (the FULL i64 result, no mod-256 limit).
pub fn expectRunFull(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, expected: i64, backend: Backend) !void {
    try std.testing.expectEqual(expected, try runFuncFull(io, allocator, func, args, backend));
}

pub fn expectRunModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, expected: i64, backend: Backend) !void {
    const want: u8 = @truncate(@as(u64, @bitCast(expected)));
    try std.testing.expectEqual(want, try runModule(io, allocator, module, args, backend));
}

pub fn i32type(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}
