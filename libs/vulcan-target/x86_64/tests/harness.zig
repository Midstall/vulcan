//! Shared x86-64 execution-test harness, parameterized by a `Backend`. cases.zig builds
//! IR functions and asserts results through `expectRun`. Each runner supplies its backend.
//! qemu.zig builds a static ELF and runs it under qemu-x86_64. native.zig maps the code
//! into W^X memory and calls it in-process (only on an x86-64 host).
//!
//! Results are checked modulo 256 (a process exit code is the low byte), so test values
//! must differ in their low byte.

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

/// Build the entry stub: load each argument into its System V register, `call` the
/// code right after the stub, then `exit(result)`.
fn buildStub(allocator: std.mem.Allocator, args: []const i64) std.mem.Allocator.Error![]u8 {
    const arg_regs = [_]Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    for (args, 0..) |a, i| try s.appendSlice(allocator, encode.movImm(arg_regs[i], @intCast(a)).slice());

    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());

    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
    return s.toOwnedSlice(allocator);
}

/// Run a single function with `args` and return the low byte of its result.
pub fn runFunc(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, backend: Backend) !u8 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    return runCode(io, allocator, code, 0, args, backend);
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

/// Run `code` (entered at `entry_off`) with `args`. For the native backend the code
/// is called in-process, otherwise it is wrapped in an ELF and run under qemu.
fn runCode(io: std.Io, allocator: std.mem.Allocator, code: []const u8, entry_off: usize, args: []const i64, backend: Backend) !u8 {
    if (backend.native) {
        if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
        var buf = try jit.CodeBuffer.map(code);
        defer buf.deinit();
        const result: i64 = switch (args.len) {
            0 => buf.entry(*const fn () callconv(.c) i64, entry_off)(),
            1 => buf.entry(*const fn (i64) callconv(.c) i64, entry_off)(args[0]),
            2 => buf.entry(*const fn (i64, i64) callconv(.c) i64, entry_off)(args[0], args[1]),
            3 => buf.entry(*const fn (i64, i64, i64) callconv(.c) i64, entry_off)(args[0], args[1], args[2]),
            else => return error.Unsupported,
        };
        return @truncate(@as(u64, @bitCast(result)));
    }

    const stub = try buildStub(allocator, args);
    defer allocator.free(stub);
    return runProgram(io, allocator, stub, code, backend);
}

/// Wrap `stub ++ code` in a static ELF and run it under qemu, returning the exit code.
fn runProgram(io: std.Io, allocator: std.mem.Allocator, stub: []const u8, code: []const u8, backend: Backend) !u8 {
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
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return switch (result.term) {
        .exited => |c| c,
        else => error.BackendFailed,
    };
}

/// The float entry stub: load each f32 argument's bits into xmm0,xmm1,..., `call` the code,
/// move the f32 result (xmm0) back to a general register, and `exit` with its low byte.
fn buildFloatStub(allocator: std.mem.Allocator, fargs: []const f32) std.mem.Allocator.Error![]u8 {
    const xmm_args = [_]encode.Xmm{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    for (fargs, 0..) |fa, i| {
        try s.appendSlice(allocator, encode.movImm(.rax, @bitCast(@as(u32, @bitCast(fa)))).slice());
        try s.appendSlice(allocator, encode.movdToXmm(xmm_args[i], .rax).slice());
    }
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movdFromXmm(.rax, .xmm0).slice()); // f32 result bits -> eax
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());
    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
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

/// Like `buildFloatStub` but for f64: load each argument's 64 bits into xmm0.. via movq, and
/// read the f64 result's bits out of xmm0 with movq.
fn buildDoubleStub(allocator: std.mem.Allocator, dargs: []const f64) std.mem.Allocator.Error![]u8 {
    const xmm_args = [_]encode.Xmm{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    for (dargs, 0..) |da, i| {
        try s.appendSlice(allocator, encode.movImm64(.rax, @bitCast(da)).slice());
        try s.appendSlice(allocator, encode.movqToXmm(xmm_args[i], .rax).slice());
    }
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movqFromXmm(.rax, .xmm0).slice()); // f64 result bits -> rax
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());
    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
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

pub fn expectRun(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, expected: i64, backend: Backend) !void {
    const want: u8 = @truncate(@as(u64, @bitCast(expected)));
    try std.testing.expectEqual(want, try runFunc(io, allocator, func, args, backend));
}

pub fn expectRunModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, expected: i64, backend: Backend) !void {
    const want: u8 = @truncate(@as(u64, @bitCast(expected)));
    try std.testing.expectEqual(want, try runModule(io, allocator, module, args, backend));
}

pub fn i32type(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}
