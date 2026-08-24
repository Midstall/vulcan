//! This test proves that `native.writeObjectDataFor` really cross-emits a runnable
//! object for EACH of the 4 targets, regardless of which arch this test binary itself
//! runs on (aarch64, given the dev host). The test compiles the same arch-independent
//! vulcan-ir `int main(void) { return 42; }` once. Then, for every target, it emits
//! the `.o` file with `writeObjectDataFor`, links it with `vulcan-link.linkObjects`,
//! and prepends a tiny hand-assembled entry stub. The stub calls `main` and exits with
//! its return value, the same stub shape each backend's own `link_native`/`native`
//! tests already use. The test wraps the result in a runnable ELF with
//! `vulcan-link.writeElfExec`, then executes it: natively for aarch64 (the host), and
//! under `qemu-<arch>` for the other three. Each of the 4 must actually RUN, not skip,
//! and exit 42 on a host with all 3 qemu binaries present. An arch whose qemu is
//! missing skips cleanly instead of failing.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const ld = @import("vulcan-link");

const Function = ir.function.Function;

/// `int main(void) { return 42; }` as vulcan-ir: one block, no params, `ret iconst 42`.
/// This is arch-independent. The same `Function` feeds every backend's
/// `writeObjectDataFor` branch below.
fn buildMain(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try f.appendBlock();
    const c = try f.appendInst(b, t, .{ .iconst = 42 });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(c) });
    return f;
}

/// Run `argv[0]`, already on PATH, or a native `./a.elf`, against `elf` written to a
/// fresh tmp dir, and return its exit code. Returns `error.SkipZigTest` when the
/// runner, `qemu-<arch>` when `argv[0]` names one, is not installed.
fn runElf(allocator: std.mem.Allocator, io: std.Io, elf: []const u8, argv: []const []const u8) !u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });

    const proc = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound, error.AccessDenied => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    return switch (proc.term) {
        .exited => |code| code,
        else => {
            std.debug.print("cross_target run term: {any}\nstderr: {s}\n", .{ proc.term, proc.stderr });
            return error.BackendFailed;
        },
    };
}

test "cross-target: writeObjectDataFor(.aarch64, ...) emits+links+runs to exit 42" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Executes the AArch64 ELF directly, which needs a Linux host: the image is a
    // Linux ELF with a Linux svc exit, and a darwin aarch64 host cannot exec it.
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .aarch64, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.aarch64.encode;
    const base: u64 = 0x400000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + 12);
    defer image.deinit(allocator);

    // stub: bl main, then movz x8, #93, then svc #0. This exits with exit(x0), where x0
    // already holds main's return value (AAPCS64). The stub is 12 bytes and sits right
    // before `image.code`. `bl` is the very first instruction (pc == base), so its offset
    // to `main` is simply `main_addr - base`. This needs no extra stub-length term, unlike
    // a `bl` sitting later in the stub.
    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const stub = [_]u32{
        encode.bl(@intCast(main_addr - @as(i64, @intCast(base)))),
        encode.movz(.x8, 93, 0),
        encode.svc(0),
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.aarch64, allocator, program.items, program.items.len, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{"./a.elf"}) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "cross-target: writeObjectDataFor(.x86_64, ...) emits+links+runs to exit 42 (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .x86_64, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.x86_64.encode;
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice()); // rdi = main's return (rax)
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice()); // rax = 60 (exit)
    try exitseq.appendSlice(allocator, encode.syscall().slice());
    const stub_len: u64 = 5 + exitseq.items.len; // call rel32 (5) ++ exitseq

    const base: u64 = 0x10000000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + 5)));
    var stub: std.ArrayList(u8) = .empty;
    defer stub.deinit(allocator);
    try stub.appendSlice(allocator, encode.callRel(rel).slice());
    try stub.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, stub.items.len);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, stub.items);
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.x86_64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-x86_64", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "cross-target: writeObjectDataFor(.x86, ...) emits+links+runs to exit 42 (qemu-i386)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .x86, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.x86.encode;
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.ebx, .eax).slice()); // ebx = main's return (eax)
    try exitseq.appendSlice(allocator, encode.movImm(.eax, 1).slice()); // eax = 1 (exit)
    try exitseq.appendSlice(allocator, encode.int80().slice());
    const stub_len: u64 = 5 + exitseq.items.len;

    const base: u64 = 0x08048000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + 5)));
    var stub: std.ArrayList(u8) = .empty;
    defer stub.deinit(allocator);
    try stub.appendSlice(allocator, encode.callRel(rel).slice());
    try stub.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, stub.items.len);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, stub.items);
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.x86, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-i386", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "cross-target: writeObjectDataFor(.riscv64, ...) emits+links+runs to exit 42 (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .riscv64, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.riscv64.encode;
    const stub_len: u64 = 12; // jal main (4 bytes), li a7,93 (4 bytes), ecall (4 bytes)

    const base: u64 = 0x10000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    // main's return (i32) is already in a0 (x10), the RISC-V calling convention's
    // integer return register. This matches `exit(a0)`'s expectation directly, so no
    // register move is needed. This mirrors aarch64's x0.
    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const jal_off: i21 = @intCast(main_addr - @as(i64, @intCast(base)));
    const stub = [_]u32{
        encode.jal(.x1, jal_off),
        encode.addi(.x17, .x0, 93), // a7 = 93 (exit)
        encode.ecall(),
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);
    try std.testing.expectEqual(stub_len, @as(u64, 12));

    const elf = try ld.writeElfExec(.riscv64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-riscv64", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}
