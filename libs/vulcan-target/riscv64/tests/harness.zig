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
const compress = @import("../compress.zig");
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
    /// A Linux user-mode backend (e.g. qemu-riscv64): the program is a plain static ELF whose entry
    /// stub calls the function, writes the 8-byte result to stdout via the `write` syscall, and exits
    /// via `exit`. When false, the firmware path (MMIO-UART, self-loop) is used instead.
    user_mode: bool = false,
    /// RVC-compress the program before running it, but ONLY when it is provably self-contained (a
    /// single function, or a linked module with no pending relocations) so compression cannot shift a
    /// relocation target it does not know about. Validates the compressor across the whole corpus.
    compress_rvc: bool = false,
    /// Build the full argv to run the ELF at `elf_path`. Caller owns it.
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

fn qemuUserArgv(allocator: std.mem.Allocator, elf_path: []const u8) std.mem.Allocator.Error![]const []const u8 {
    // qemu-riscv64 user mode runs the static ELF directly under the Linux syscall ABI.
    return allocator.dupe([]const u8, &.{ "qemu-riscv64", elf_path });
}

fn qemuUserCpuMaxArgv(allocator: std.mem.Allocator, elf_path: []const u8) std.mem.Allocator.Error![]const []const u8 {
    // `-cpu max` turns on every optional extension qemu implements, including Zfh (native f16). The
    // default (no `-cpu`) is RV64GC with NO Zfh, so the native half instructions would fault there;
    // this backend is used only by the native Zfh differential tests.
    return allocator.dupe([]const u8, &.{ "qemu-riscv64", "-cpu", "max", elf_path });
}

/// River functional emulator (Midstall's CPU): the reference backend.
pub const river = Backend{ .name = "river-emulator", .buildArgv = riverArgv };

/// QEMU user-mode (`qemu-riscv64`): runs the same case corpus as a plain Linux static ELF, so the
/// codegen executes on any dev machine with qemu even when River/Spike are absent. Skips if qemu is
/// not on PATH.
pub const qemu_user = Backend{ .name = "qemu-riscv64", .user_mode = true, .buildArgv = qemuUserArgv };

/// Like `qemu_user`, but adds `-cpu max` so Zfh (native f16) is enabled. Used by the native f16
/// differential tests, which emit real half instructions that the default RV64GC CPU rejects.
pub const qemu_user_cpumax = Backend{ .name = "qemu-riscv64 -cpu max", .user_mode = true, .buildArgv = qemuUserCpuMaxArgv };

/// Like `qemu_user`, but RVC-compresses every self-contained case first, so the whole corpus doubles
/// as an execution test of the compressor on real, diverse codegen. Skips if qemu is absent.
pub const qemu_user_rvc = Backend{ .name = "qemu-riscv64", .user_mode = true, .compress_rvc = true, .buildArgv = qemuUserArgv };

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

/// User-mode entry stub: load arguments into a0.., call the function placed right after, then write
/// the returned a0 to stdout as 8 LE bytes via the `write` syscall and `exit(0)`. The kernel gives a
/// valid sp and an enabled FPU, so (unlike the firmware stub) neither is set up here. Caller owns it.
pub fn buildUserStub(allocator: std.mem.Allocator, args: []const i64) std.mem.Allocator.Error![]u32 {
    var w: std.ArrayList(u32) = .empty;
    errdefer w.deinit(allocator);

    for (args, 0..) |arg, i| try loadImmInto(allocator, &w, argReg(i), arg);
    const call_idx = w.items.len;
    try w.append(allocator, encode.jal(.x1, 0)); // call the function (offset patched below)
    // a0 now holds the result. Spill it to the stack and write those 8 bytes to fd 1.
    try w.append(allocator, encode.addi(.x2, .x2, -16)); // sp -= 16
    try w.append(allocator, encode.sd(.x10, .x2, 0)); // sd a0, 0(sp)
    try w.append(allocator, encode.addi(.x10, .x0, 1)); // a0 = 1 (stdout)
    try w.append(allocator, encode.addi(.x11, .x2, 0)); // a1 = sp (buffer)
    try w.append(allocator, encode.addi(.x12, .x0, 8)); // a2 = 8 (length)
    try w.append(allocator, encode.addi(.x17, .x0, 64)); // a7 = 64 (write)
    try w.append(allocator, encode.ecall());
    try w.append(allocator, encode.addi(.x10, .x0, 0)); // a0 = 0 (status)
    try w.append(allocator, encode.addi(.x17, .x0, 93)); // a7 = 93 (exit)
    try w.append(allocator, encode.ecall());
    const fn_off: i21 = @intCast((w.items.len - call_idx) * 4);
    w.items[call_idx] = encode.jal(.x1, fn_off);
    return w.toOwnedSlice(allocator);
}

/// Load a 32-bit pattern into `reg` via `lui`+`addi` (RV64 sign-extends the `addi` result
/// through bit 63; callers either want that directly, or immediately mask/shift it away).
fn loadImm32Into(allocator: std.mem.Allocator, words: *std.ArrayList(u32), reg: encode.Reg, bits: u32) std.mem.Allocator.Error!void {
    const hi: u20 = @truncate((bits +% 0x800) >> 12);
    const lo: i12 = @bitCast(@as(u12, @truncate(bits)));
    try words.append(allocator, encode.lui(reg, hi));
    try words.append(allocator, encode.addi(reg, reg, lo));
}

/// Load an arbitrary 64-bit bit pattern into `reg` (`scratch` must differ from `reg`). Unlike
/// `loadImmInto`, which only handles i32-range values, this carries a full 64-bit pattern -
/// needed for f64 args' raw bits. Splits into high/low 32-bit halves, each built with the
/// standard `lui`+`addi` sequence: the high half is shifted into place (its sign-extension
/// artifacts land above bit 63 and are discarded by the shift), and the low half is masked back
/// down to exactly 32 bits (clearing its own sign-extension artifacts) before the two are ORed.
fn loadImm64Into(allocator: std.mem.Allocator, words: *std.ArrayList(u32), reg: encode.Reg, scratch: encode.Reg, bits: u64) std.mem.Allocator.Error!void {
    try loadImm32Into(allocator, words, reg, @truncate(bits >> 32));
    try words.append(allocator, encode.slli(reg, reg, 32));
    try loadImm32Into(allocator, words, scratch, @truncate(bits));
    try words.append(allocator, encode.slli(scratch, scratch, 32));
    try words.append(allocator, encode.srli(scratch, scratch, 32));
    try words.append(allocator, encode.or_(reg, reg, scratch));
}

/// User-mode entry stub for a SCALAR float function: loads `dbl`-precision args (given as raw
/// bit patterns - f64 bits, or f32 bits in the low 32 bits) into fa0.., calls the function, and
/// writes the fa0 result's bits (zero-extended to 8 bytes for an f32 result) to stdout via
/// `write`, then `exit(0)`. Mirrors `buildUserStub`, but for the hardware float ABI (fa0.. args,
/// fa0 result) instead of the integer one: FMA fusion needs real float args/results, which
/// `buildUserStub`'s GPR-only ABI cannot carry. `x5`/`x6` are used as scratch (caller-saved,
/// unused by any arg or the tail). Caller owns the result.
pub fn buildUserStubFloat(allocator: std.mem.Allocator, dbl: bool, fargs: []const u64) std.mem.Allocator.Error![]u32 {
    var w: std.ArrayList(u32) = .empty;
    errdefer w.deinit(allocator);

    for (fargs, 0..) |bits, i| {
        try loadImm64Into(allocator, &w, .x5, .x6, bits);
        const freg: encode.FReg = @enumFromInt(@as(u5, @intCast(10 + i)));
        try w.append(allocator, if (dbl) encode.fmv_d_x(freg, .x5) else encode.fmv_w_x(freg, .x5));
    }
    const call_idx = w.items.len;
    try w.append(allocator, encode.jal(.x1, 0)); // call the function (offset patched below)
    // fa0 now holds the result. Spill it to the stack (as a double or a single) and reload as a
    // plain 64-bit integer so the tail below - identical to buildUserStub's - can write it out.
    // Zeroing the slot first means an f32 result's untouched upper 4 bytes read back as 0, not
    // stack garbage.
    try w.append(allocator, encode.addi(.x2, .x2, -16)); // sp -= 16
    try w.append(allocator, encode.sd(.x0, .x2, 0));
    try w.append(allocator, if (dbl) encode.fsd(.f10, .x2, 0) else encode.fsw(.f10, .x2, 0));
    try w.append(allocator, encode.ld(.x10, .x2, 0)); // a0 = the result's bits
    try w.append(allocator, encode.sd(.x10, .x2, 0)); // sd a0, 0(sp) (the write buffer)
    try w.append(allocator, encode.addi(.x10, .x0, 1)); // a0 = 1 (stdout)
    try w.append(allocator, encode.addi(.x11, .x2, 0)); // a1 = sp (buffer)
    try w.append(allocator, encode.addi(.x12, .x0, 8)); // a2 = 8 (length)
    try w.append(allocator, encode.addi(.x17, .x0, 64)); // a7 = 64 (write)
    try w.append(allocator, encode.ecall());
    try w.append(allocator, encode.addi(.x10, .x0, 0)); // a0 = 0 (status)
    try w.append(allocator, encode.addi(.x17, .x0, 93)); // a7 = 93 (exit)
    try w.append(allocator, encode.ecall());
    const fn_off: i21 = @intCast((w.items.len - call_idx) * 4);
    w.items[call_idx] = encode.jal(.x1, fn_off);
    return w.toOwnedSlice(allocator);
}

/// Compile `func`, run it under a user-mode `backend` (e.g. `qemu_user`) with scalar float args,
/// and return the raw bits `fa0` held on return (an f64's bits, or an f32's bits zero-extended).
/// Mirrors `runFunc`, but for a function using the hardware float ABI (fa0.. args/result)
/// instead of the integer one `runFunc`/`runProgram` assume - only a user-mode backend has a
/// float-ABI stub today.
pub fn runFuncFloat(io: std.Io, allocator: std.mem.Allocator, func: *Function, dbl: bool, fargs: []const u64, backend: Backend) !u64 {
    var words = try compileFunc(allocator, func);
    defer words.deinit(allocator);
    return runCompiledFloat(io, allocator, words.items, dbl, fargs, backend);
}

/// Run pre-compiled float-ABI code (entry at word 0) under a user-mode `backend`, returning the
/// raw bits `fa0` held on return. The run half of `runFuncFloat`, split out so a caller can supply
/// code produced by `isel.selectFunction` alone (skipping the scheduler) when it must preserve a
/// specific liveness shape - e.g. a scalar-float-spill stress test whose deliberately wide live
/// range a scheduler could otherwise narrow.
pub fn runCompiledFloat(io: std.Io, allocator: std.mem.Allocator, code: []const u32, dbl: bool, fargs: []const u64, backend: Backend) !u64 {
    if (backend.incompatible) return error.SkipZigTest;
    if (!backend.user_mode) return error.Unsupported; // only buildUserStubFloat exists so far

    const stub = try buildUserStubFloat(allocator, dbl, fargs);
    defer allocator.free(stub);
    const program = try allocator.alloc(u32, stub.len + code.len);
    defer allocator.free(program);
    @memcpy(program[0..stub.len], stub);
    @memcpy(program[stub.len..], code);

    const bytes = try emit.emitBytes(allocator, program);
    defer allocator.free(bytes);
    const user_base: u64 = 0x10000;
    const elf = try (@import("../ld.zig")).writeElfExec(allocator, bytes, bytes.len, user_base, user_base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "firmware.elf", .data = elf, .flags = .{ .permissions = .executable_file } });

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
    return std.mem.readInt(u64, tail[0..8], .little);
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

/// Like `compileFunc`, but selects for a specific microarch `model` (via `selectFunctionForModel`),
/// so a model-gated capability (e.g. Zfh native f16) reaches the backend. The pipeline is otherwise
/// identical.
pub fn compileFuncForModel(allocator: std.mem.Allocator, func: *Function, model: *const @import("vulcan-opt").microarch.Model) !std.ArrayList(u32) {
    try ir.legalize.legalize(allocator, func);
    try isel.splitCriticalEdges(allocator, func);
    try schedule.scheduleFunction(allocator, func);
    const code = try isel.selectFunctionForModel(allocator, func, model);
    defer allocator.free(code);
    var list: std.ArrayList(u32) = .empty;
    try list.appendSlice(allocator, code);
    return list;
}

/// Compile `func`, run it on `backend` with `args`, and return a0. A single function is self-
/// contained (any `call` would need linking), so RVC compression is safe when the backend asks.
pub fn runFunc(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64, backend: Backend) !i64 {
    var words = try compileFunc(allocator, func);
    defer words.deinit(allocator);
    return runProgram(io, allocator, words.items, args, backend, backend.compress_rvc);
}

/// Like `runFunc`, but for a whole linked module (entry first). Compression is only safe when the
/// module has no pending relocations (every call resolved to an internal `jal`, no external data).
pub fn runModule(io: std.Io, allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, backend: Backend) !i64 {
    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    const can_compress = backend.compress_rvc and linked.relocs.len == 0;
    return runProgram(io, allocator, linked.code, args, backend, can_compress);
}

/// Run an already-linked code image (entry at word 0) on `backend`.
pub fn runCode(io: std.Io, allocator: std.mem.Allocator, code: []const u32, args: []const i64, backend: Backend) !i64 {
    return runProgram(io, allocator, code, args, backend, false);
}

fn runProgram(io: std.Io, allocator: std.mem.Allocator, code: []const u32, args: []const i64, backend: Backend, compress_it: bool) !i64 {
    if (backend.incompatible) return error.SkipZigTest;

    const stub = if (backend.user_mode) try buildUserStub(allocator, args) else try buildStub(allocator, args);
    defer allocator.free(stub);
    const program = try allocator.alloc(u32, stub.len + code.len);
    defer allocator.free(program);
    @memcpy(program[0..stub.len], stub);
    @memcpy(program[stub.len..], code);

    // Self-contained programs may be RVC-compressed (stub + body are all PC-relative); compress
    // recomputes the stub's call jal for the shrunk layout.
    const bytes = if (compress_it) try compress.compress(allocator, program) else try emit.emitBytes(allocator, program);
    defer allocator.free(bytes);
    // User-mode: a static ELF at a low base, entered at the stub. Firmware: flat image at DRAM base.
    const user_base: u64 = 0x10000;
    const elf = if (backend.user_mode)
        try (@import("../ld.zig")).writeElfExec(allocator, bytes, bytes.len, user_base, user_base)
    else
        try writeElf(allocator, bytes, load_address);
    defer allocator.free(elf);

    // Write the ELF into a unique temp directory and run the backend with its cwd set there, so the
    // argv just names the file (no cache-path assumptions). User-mode qemu needs the file executable.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    if (backend.user_mode)
        try tmp.dir.writeFile(io, .{ .sub_path = "firmware.elf", .data = elf, .flags = .{ .permissions = .executable_file } })
    else
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
