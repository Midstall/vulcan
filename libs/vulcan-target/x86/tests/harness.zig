//! Shared i386 (cdecl) execution-test harness, parameterized by a `Backend`. cases.zig
//! builds IR functions and asserts results through `expectRun`. qemu.zig runs a static
//! ELF under qemu-i386 and native.zig calls the code in-process (only on an i386 host).
//! Results are checked modulo 256 (a process exit code is the low byte).

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const encode = @import("../encode.zig");
const isel = @import("../isel.zig");
const link = @import("../link.zig");
const elf = @import("../elf.zig");
const jit = @import("../../coherent_jit.zig");

const Function = ir.function.Function;

pub const Backend = struct {
    name: []const u8,
    qemu_cmd: ?[]const u8 = null,
    native: bool = false,
};

pub const qemu = Backend{ .name = "qemu-i386", .qemu_cmd = "qemu-i386" };
pub const native = Backend{ .name = "native-i386", .native = true };

/// The entry stub: push the cdecl arguments right-to-left, `call` the code after the
/// stub, then `exit(result)` via the i386 syscall gate.
fn buildStub(allocator: std.mem.Allocator, args: []const i64) std.mem.Allocator.Error![]u8 {
    var s: std.ArrayList(u8) = .empty;
    errdefer s.deinit(allocator);
    var k: usize = args.len;
    while (k > 0) {
        k -= 1;
        try s.appendSlice(allocator, encode.pushImm(@intCast(args[k])).slice());
    }
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.ebx, .eax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.eax, 1).slice());
    try exitseq.appendSlice(allocator, encode.int80().slice());
    try s.appendSlice(allocator, encode.callRel(@intCast(exitseq.items.len)).slice());
    try s.appendSlice(allocator, exitseq.items);
    return s.toOwnedSlice(allocator);
}

pub fn runFunc(io: std.Io, allocator: std.mem.Allocator, func: *const Function, args: []const i64, backend: Backend) !u8 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    return runCode(io, allocator, code, args, backend);
}

/// Link `module` and run its `main` (the first function, at offset 0).
pub fn runModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, backend: Backend) !u8 {
    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    const entry = linked.addressOf("main") orelse return error.UndefinedSymbol;
    std.debug.assert(entry == 0); // the stub calls the code at offset 0
    return runCode(io, allocator, linked.code, args, backend);
}

fn runCode(io: std.Io, allocator: std.mem.Allocator, code: []const u8, args: []const i64, backend: Backend) !u8 {
    if (backend.native) {
        if (builtin.cpu.arch != .x86) return error.SkipZigTest;
        var buf = try jit.CodeBuffer.map(code);
        defer buf.deinit();
        const result: i32 = switch (args.len) {
            0 => buf.entry(*const fn () callconv(.c) i32, 0)(),
            1 => buf.entry(*const fn (i32) callconv(.c) i32, 0)(@intCast(args[0])),
            2 => buf.entry(*const fn (i32, i32) callconv(.c) i32, 0)(@intCast(args[0]), @intCast(args[1])),
            else => return error.Unsupported,
        };
        return @truncate(@as(u32, @bitCast(result)));
    }

    const stub = try buildStub(allocator, args);
    defer allocator.free(stub);
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
        .argv = &.{ backend.qemu_cmd.?, "a.elf" },
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

/// Run raw compiled `code` (entered at offset 0) with integer `args`, returning the low byte of its
/// result. Used by the shared-Wimmer differential tests, which compile through
/// `isel.compileFunctionWimmerX86`/`...Fold` (bytes, not a `Function`) and diff against `runFunc`.
pub fn runCodeInt(io: std.Io, allocator: std.mem.Allocator, code: []const u8, args: []const i64, backend: Backend) !u8 {
    return runCode(io, allocator, code, args, backend);
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
