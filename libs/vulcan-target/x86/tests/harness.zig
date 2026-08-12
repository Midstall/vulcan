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

/// Like `runModule`, but `module` may also carry rodata/data/bss globals referenced from
/// code via `global_addr` (see `link.Module.addData`/`addWritable`/`addBss`).
///
/// The qemu path lays out `linked.code` followed by the rodata and writable-data bytes
/// (bss reserved as zeroed filler - no test here writes to one) as ONE flat blob, then
/// resolves every carried-forward `.abs32` relocation against the blob's REAL runtime
/// address: unlike x86-64's rip-relative `.pcrel_lea` (where only the relative shift
/// between site and target matters, so the constant offset of the stub/ELF header
/// cancels out), i386's `mov rd, imm32` needs the actual absolute address, so the site's
/// known load address (`elf.load_addr + elf.code_offset + stub.len`, exactly where
/// `elf.writeExec` places `stub ++ image`) is folded in here.
///
/// The native path (real only on an i386 host, essentially never true for this test
/// suite's aarch64/x86-64 hosts) instead delegates to the shared `native.jitModuleData`,
/// which already links against `link.zig`'s `applyGlobalReloc` generically - the
/// harness's raw `runCode` has no data-section support and does not run this path.
pub fn runModuleData(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, backend: Backend) !u8 {
    if (backend.native) {
        if (builtin.cpu.arch != .x86) return error.SkipZigTest;
        const native_target = @import("../../native.zig");
        var funcs: std.ArrayList(native_target.ModuleFunction) = .empty;
        defer funcs.deinit(allocator);
        for (module.funcs.items) |e| try funcs.append(allocator, .{ .name = e.name, .func = e.func });
        var datas: std.ArrayList(native_target.ModuleData) = .empty;
        defer datas.deinit(allocator);
        for (module.data.items) |d| try datas.append(allocator, .{ .name = d.name, .bytes = d.bytes, .kind = d.kind, .size = d.size, .relocs = d.relocs });

        var jm = try native_target.jitModuleData(allocator, funcs.items, datas.items);
        defer jm.deinit();
        const f = jm.entry(*const fn () callconv(.c) i32, "main") orelse return error.UndefinedSymbol;
        std.debug.assert(args.len == 0); // every current data test takes no arguments
        return @truncate(@as(u32, @bitCast(f())));
    }

    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    const entry = linked.addressOf("main") orelse return error.UndefinedSymbol;
    std.debug.assert(entry == 0); // the stub calls the code at offset 0

    // Each section's data starts right after the previous one, all within the same
    // blob as the code (`rodata_base`/`data_base` below double as that section's
    // starting byte offset within the final image).
    var rodata_len: usize = 0;
    var data_len: usize = 0;
    var bss_len: usize = 0;
    for (linked.data) |d| switch (d.kind) {
        .rodata => rodata_len = @max(rodata_len, d.off + d.size),
        .data => data_len = @max(data_len, d.off + d.size),
        .bss => bss_len = @max(bss_len, d.off + d.size),
    };
    const rodata_base = linked.code.len;
    const data_base = rodata_base + rodata_len;
    const bss_base = data_base + data_len;
    const image_len = bss_base + bss_len;

    const image = try allocator.alloc(u8, image_len);
    defer allocator.free(image);
    @memset(image, 0);
    @memcpy(image[0..linked.code.len], linked.code);
    for (linked.data) |d| switch (d.kind) {
        .rodata => @memcpy(image[rodata_base + d.off ..][0..d.size], d.bytes),
        .data => @memcpy(image[data_base + d.off ..][0..d.size], d.bytes),
        .bss => {},
    };

    const stub = try buildStub(allocator, args);
    defer allocator.free(stub);
    // The vaddr of `image[0]` once `stub ++ image` is wrapped by `elf.writeExec` and
    // mapped at `elf.load_addr` (the whole file, `elf.code_offset` in).
    const image_base: u32 = elf.load_addr + @as(u32, @intCast(elf.code_offset + stub.len));

    for (linked.relocs) |r| {
        const target_off = blk: {
            for (linked.data) |d| if (std.mem.eql(u8, d.name, r.symbol)) break :blk switch (d.kind) {
                .rodata => rodata_base + d.off,
                .data => data_base + d.off,
                .bss => bss_base + d.off,
            };
            if (linked.addressOf(r.symbol)) |off| break :blk off; // degenerate: names a function
            return error.UndefinedSymbol;
        };
        const target_addr: u32 = image_base + @as(u32, @intCast(target_off));
        std.mem.writeInt(u32, image[r.offset..][0..4], target_addr, .little);
    }

    const program = try allocator.alloc(u8, stub.len + image.len);
    defer allocator.free(program);
    @memcpy(program[0..stub.len], stub);
    @memcpy(program[stub.len..], image);
    return runProgram(io, allocator, program, backend);
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
    return runProgram(io, allocator, program, backend);
}

/// Wrap an already-built `stub ++ code` (or `stub ++ image`) program into a static ELF
/// and run it under qemu, returning the process's exit code (the low byte of the
/// callee's result). Shared by `runCode` and `runModuleData`.
fn runProgram(io: std.Io, allocator: std.mem.Allocator, program: []const u8, backend: Backend) !u8 {
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
