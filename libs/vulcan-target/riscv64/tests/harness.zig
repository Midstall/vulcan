//! Shared RISC-V execution-test harness. Compiles IR to RISC-V machine code,
//! wraps it in a minimal flat firmware ELF (an entry stub loads the arguments,
//! calls the function, and writes the returned a0 to a UART), and runs that ELF
//! on a configurable backend (emulator/simulator/QEMU). The same cases (cases.zig)
//! run on every backend. Each runner file (river.zig, ...) supplies its own.
//! Unavailable or incompatible backends skip via `error.SkipZigTest`.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("../encode.zig");
const emit = @import("../emit.zig");
const isel = @import("../isel.zig");
const schedule = @import("../schedule.zig");
const link = @import("../link.zig");

const Function = ir.function.Function;

/// The program loads at the canonical DRAM base. The entry stub sits first.
pub const load_address: u64 = 0x80000000;
/// ns16550a UART base. Its transmit-holding register is at offset 0.
pub const uart_base: u64 = 0x10000000;
/// Stack pointer the stub sets, in a region whose bit 31 is clear (so `lui`
/// materializes it without sign-extending into a negative, faulting address).
const stack_top: u64 = 0x40080000;

/// How to run a firmware ELF. A backend either builds the argv to execute it, or
/// is marked incompatible (the MMIO-UART firmware doesn't fit that machine yet).
pub const Backend = struct {
    name: []const u8,
    incompatible: bool = false,
    reason: []const u8 = "",
    /// Build the full argv to run the firmware at `elf_path`. Caller owns it.
    buildArgv: *const fn (allocator: std.mem.Allocator, elf_path: []const u8) std.mem.Allocator.Error![]const []const u8,
};

fn riverArgv(allocator: std.mem.Allocator, elf_path: []const u8) std.mem.Allocator.Error![]const []const u8 {
    // rc1-m is RV64GC + Zba/Zbb/Zbs: covers everything Vulcan emits (incl. F/D and
    // the Zbb rev8 the endianness engine uses). The UART sends to stdout.
    return allocator.dupe([]const u8, &.{
        "river-emulator",           "--core",
        "rc1-m",                    "--memory",
        "main:0x80000000:1M:dram",  "--memory",
        "stack:0x40000000:1M:dram", "--device",
        "uart:uart:0x10000000",     "--device-option",
        "uart.input.empty=1",       "--firmware",
        elf_path,                   "--max-cycles",
        "20000",
    });
}

fn unusedArgv(allocator: std.mem.Allocator, elf_path: []const u8) std.mem.Allocator.Error![]const []const u8 {
    _ = elf_path;
    return allocator.dupe([]const u8, &.{});
}

/// River functional emulator (Midstall's CPU): the reference backend.
pub const river = Backend{ .name = "river-emulator", .buildArgv = riverArgv };

/// Spike (the RISC-V ISA simulator). Incompatible for now: it has no MMIO 16550
/// UART at 0x10000000. Output goes through HTIF tohost, so the firmware's UART
/// tail would produce nothing. Needs a tohost stub variant.
pub const spike = Backend{ .name = "spike", .incompatible = true, .reason = "firmware uses MMIO UART, spike uses HTIF tohost", .buildArgv = unusedArgv };

/// QEMU (`qemu-system-riscv64 -M virt`). Incompatible for now: the firmware
/// self-loops with no exit, so QEMU would run forever. Needs a finisher/SBI
/// shutdown in the stub.
pub const qemu = Backend{ .name = "qemu-system-riscv64", .incompatible = true, .reason = "firmware never exits, QEMU needs a shutdown device", .buildArgv = unusedArgv };

fn argReg(i: usize) encode.Reg {
    return @enumFromInt(@as(u5, @intCast(10 + i)));
}

fn loadImmInto(allocator: std.mem.Allocator, words: *std.ArrayList(u32), reg: encode.Reg, val: i64) std.mem.Allocator.Error!void {
    if (val >= -2048 and val <= 2047) {
        try words.append(allocator, encode.addi(reg, .x0, @intCast(val)));
    } else {
        const bits: u32 = @bitCast(@as(i32, @intCast(val)));
        const hi: u20 = @truncate((bits +% 0x800) >> 12);
        const lo: i12 = @bitCast(@as(u12, @truncate(bits)));
        try words.append(allocator, encode.lui(reg, hi));
        try words.append(allocator, encode.addi(reg, reg, lo));
    }
}

/// The entry stub: set sp, enable the FPU, load arguments into a0.., call the
/// function placed right after, write the returned a0 to the UART as 8 LE bytes,
/// and self-loop. Caller owns the result.
pub fn buildStub(allocator: std.mem.Allocator, args: []const i64) std.mem.Allocator.Error![]u32 {
    var w: std.ArrayList(u32) = .empty;
    errdefer w.deinit(allocator);

    try w.append(allocator, encode.lui(.x2, @intCast(stack_top >> 12))); // sp
    try w.append(allocator, encode.lui(.x6, 6)); // x6 = 0x6000 (mstatus.FS = Dirty)
    try w.append(allocator, encode.csrrs(.x0, 0x300, .x6));
    for (args, 0..) |arg, i| try loadImmInto(allocator, &w, argReg(i), arg);
    const call_idx = w.items.len;
    try w.append(allocator, encode.jal(.x1, 0));
    try w.append(allocator, encode.lui(.x5, @intCast(uart_base >> 12)));
    try w.append(allocator, encode.addi(.x7, .x10, 0));
    for (0..8) |_| {
        try w.append(allocator, encode.sb(.x7, .x5, 0));
        try w.append(allocator, encode.srli(.x7, .x7, 8));
    }
    try w.append(allocator, encode.jal(.x0, 0)); // j .
    const fn_off: i21 = @intCast((w.items.len - call_idx) * 4);
    w.items[call_idx] = encode.jal(.x1, fn_off);
    return w.toOwnedSlice(allocator);
}

/// Wrap a code image into a flat rv64 firmware ELF loaded at `entry`, entering at
/// `entry`. Delegates to the linker's production ELF writer. Caller owns it.
pub fn writeElf(allocator: std.mem.Allocator, code: []const u8, entry: u64) std.mem.Allocator.Error![]u8 {
    const ld = @import("../ld.zig");
    return ld.writeElfExec(allocator, code, code.len, entry, entry);
}

/// Compile `func` through the full pipeline to RISC-V words (entry at word 0).
pub fn compileFunc(allocator: std.mem.Allocator, func: *Function) !std.ArrayList(u32) {
    try ir.legalize.legalize(allocator, func);
    try isel.splitCriticalEdges(allocator, func);
    try schedule.scheduleFunction(allocator, func);
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var list: std.ArrayList(u32) = .empty;
    try list.appendSlice(allocator, code);
    return list;
}

/// Compile `func`, run it on `backend` with `args`, and return a0.
pub fn runFunc(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64, backend: Backend) !i64 {
    var words = try compileFunc(allocator, func);
    defer words.deinit(allocator);
    return runCode(io, allocator, words.items, args, backend);
}

/// Like `runFunc`, but for a whole linked module (entry first).
pub fn runModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, backend: Backend) !i64 {
    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    return runCode(io, allocator, linked.code, args, backend);
}

/// Run an already-linked code image (entry at word 0) on `backend`.
pub fn runCode(io: std.Io, allocator: std.mem.Allocator, code: []const u32, args: []const i64, backend: Backend) !i64 {
    if (backend.incompatible) return error.SkipZigTest;

    const stub = try buildStub(allocator, args);
    defer allocator.free(stub);
    const program = try allocator.alloc(u32, stub.len + code.len);
    defer allocator.free(program);
    @memcpy(program[0..stub.len], stub);
    @memcpy(program[stub.len..], code);

    const bytes = try emit.emitBytes(allocator, program);
    defer allocator.free(bytes);
    const elf = try writeElf(allocator, bytes, load_address);
    defer allocator.free(elf);

    // Write the firmware into a unique temp directory and run the backend with its
    // cwd set there, so the argv just names the file (no cache-path assumptions).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "firmware.elf", .data = elf });

    const argv = try backend.buildArgv(allocator, "firmware.elf");
    defer allocator.free(argv);
    const result = std.process.run(allocator, io, .{ .argv = argv, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.stdout.len < 8) {
        std.debug.print("{s}: stdout too short ({d} bytes):\nstdout: {s}\nstderr: {s}\n", .{ backend.name, result.stdout.len, result.stdout, result.stderr });
        return error.BackendFailed;
    }
    const tail = result.stdout[result.stdout.len - 8 ..];
    return @bitCast(std.mem.readInt(u64, tail[0..8], .little));
}

pub fn expectRun(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64, expected: i64, backend: Backend) !void {
    try std.testing.expectEqual(expected, try runFunc(io, allocator, func, args, backend));
}

test "buildStub sets sp, loads args, calls the function, and emits a UART tail" {
    const allocator = std.testing.allocator;
    const stub = try buildStub(allocator, &.{ 3, 4 });
    defer allocator.free(stub);
    try std.testing.expectEqual(encode.lui(.x2, 0x40080), stub[0]);
    try std.testing.expectEqual(encode.addi(.x10, .x0, 3), stub[3]); // a0 = 3
    try std.testing.expectEqual(encode.addi(.x11, .x0, 4), stub[4]); // a1 = 4
    try std.testing.expectEqual(encode.jal(.x0, 0), stub[stub.len - 1]); // j .
}

test "writeElf produces a valid rv64 ELF header pointing at the code" {
    const allocator = std.testing.allocator;
    const code = [_]u8{ 0x11, 0x22, 0x33, 0x44 };
    const elf = try writeElf(allocator, &code, 0);
    defer allocator.free(elf);
    try std.testing.expectEqualSlices(u8, &.{ 0x7f, 'E', 'L', 'F', 2, 1, 1 }, elf[0..7]);
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, elf[16..18], .little)); // ET_EXEC
    try std.testing.expectEqual(@as(u16, 243), std.mem.readInt(u16, elf[18..20], .little)); // EM_RISCV
    try std.testing.expectEqualSlices(u8, &code, elf[0x1000..]); // code at a page-aligned file offset
}
