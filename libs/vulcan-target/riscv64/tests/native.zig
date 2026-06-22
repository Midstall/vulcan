//! Native backend runner: maps the compiled function into W^X memory and calls
//! it in-process. Only valid when the host is RISC-V (host == target), skips
//! otherwise. Unlike the emulator backends this does not use the UART firmware
//! (MMIO would fault natively), it calls the function and reads the return value.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const harness = @import("harness.zig");

const Function = ir.function.Function;

/// Compile `func`, map it executable, and call it natively. Skips off RISC-V.
fn runNative(allocator: std.mem.Allocator, func: *Function, args: []const i64) !i64 {
    if (builtin.cpu.arch != .riscv64) return error.SkipZigTest;

    var words = try harness.compileFunc(allocator, func);
    defer words.deinit(allocator);
    const bytes = std.mem.sliceAsBytes(words.items);

    const mem = try std.posix.mmap(null, bytes.len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(mem);
    @memcpy(mem[0..bytes.len], bytes);
    const rc = std.posix.system.mprotect(mem.ptr, mem.len, .{ .READ = true, .EXEC = true });
    if (std.posix.errno(rc) != .SUCCESS) return error.ProtectFailed;
    if (builtin.cpu.arch == .riscv64) asm volatile ("fence.i" ::: .{ .memory = true });

    const ptr = mem.ptr;
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) i64, @ptrCast(ptr))(),
        1 => @as(*const fn (i64) callconv(.c) i64, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i64, i64) callconv(.c) i64, @ptrCast(ptr))(args[0], args[1]),
        else => error.Unsupported,
    };
}

test "native-riscv: arithmetic runs in-process when the host is RISC-V" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const a = try func.appendBlockParam(e, t);
    const b = try func.appendBlockParam(e, t);
    const p = try func.appendInst(e, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const s = try func.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = a } });
    func.setTerminator(e, .{ .ret = s });
    try std.testing.expectEqual(@as(i64, 15), try runNative(allocator, &func, &.{ 3, 4 }));
}
