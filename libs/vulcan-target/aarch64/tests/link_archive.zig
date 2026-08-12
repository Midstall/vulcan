//! `.a` archive input coverage for the shared linker (`vulcan-link`). Builds real
//! ELF64 `.o` members via `object.writeModule`, hand-assembles a minimal `ar` archive
//! containing them, and exercises `link.linkInputs`: a referenced member must be
//! pulled and resolved, an unreferenced member must NOT be pulled, and a symbol that
//! no archive member defines must still surface `error.UndefinedSymbol`. The host is
//! aarch64, so the pulled-and-linked program is wrapped in a runnable ELF and
//! executed directly (no emulator needed), same as `native.zig`'s
//! `object+ld+exec` tests.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const object = @import("../object.zig");
const link = @import("../link.zig");
const encode = @import("../encode.zig");
const ld = @import("vulcan-link");

const Function = ir.function.Function;

/// A minimal `ar` (common-format) archive: `!<arch>\n` magic, then one 60-byte
/// header per member (name blank-padded and `/`-terminated, decimal size, "`\n"
/// terminator) followed by the member's bytes, 2-byte aligned.
fn buildArchive(allocator: std.mem.Allocator, members: []const ld.Member) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, "!<arch>\n");
    for (members) |m| {
        var hdr: [60]u8 = undefined;
        @memset(&hdr, ' ');
        const name_field = try std.fmt.allocPrint(allocator, "{s}/", .{m.name});
        defer allocator.free(name_field);
        std.debug.assert(name_field.len <= 16);
        @memcpy(hdr[0..name_field.len], name_field);
        var size_buf: [10]u8 = undefined;
        const size_str = try std.fmt.bufPrint(&size_buf, "{d}", .{m.bytes.len});
        @memcpy(hdr[48..][0..size_str.len], size_str);
        hdr[58] = '`';
        hdr[59] = '\n';
        try buf.appendSlice(allocator, &hdr);
        try buf.appendSlice(allocator, m.bytes);
        if (m.bytes.len % 2 == 1) try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

/// `helper() = 42` as its own one-function module -> ELF `.o` bytes, i.e. what a real
/// `helper.o` archive member would contain.
fn buildHelperObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var helper = Function.init(allocator);
    defer helper.deinit();
    const t = try helper.types.intern(i32k);
    const b = try helper.appendBlock();
    const v = try helper.appendInst(b, t, .{ .iconst = 42 });
    helper.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "helper", &helper);
    return object.writeModule(allocator, &module);
}

/// `dead() = 7` as its own one-function module: never referenced by `main`, so a
/// correct archive-aware linker must not pull it in.
fn buildDeadObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var dead = Function.init(allocator);
    defer dead.deinit();
    const t = try dead.types.intern(i32k);
    const b = try dead.appendBlock();
    const v = try dead.appendInst(b, t, .{ .iconst = 7 });
    dead.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "dead", &dead);
    return object.writeModule(allocator, &module);
}

/// `main() = helper()` as its own one-function module, referencing `helper` as an
/// external symbol that only the archive can satisfy.
fn buildMainObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var main = Function.init(allocator);
    defer main.deinit();
    const t = try main.types.intern(i32k);
    const b = try main.appendBlock();
    const r = try main.appendCall(b, t, "helper", &.{});
    main.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    return object.writeModule(allocator, &module);
}

test "linkInputs pulls a referenced archive member and resolves the call, natively runs to exit 42" {
    const allocator = std.testing.allocator;

    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);
    const helper_o = try buildHelperObj(allocator);
    defer allocator.free(helper_o);
    const dead_o = try buildDeadObj(allocator);
    defer allocator.free(dead_o);

    const archive_bytes = try buildArchive(allocator, &.{
        .{ .name = "helper.o", .bytes = helper_o },
        .{ .name = "dead.o", .bytes = dead_o },
    });
    defer allocator.free(archive_bytes);

    const base: u64 = 0x400000;
    var image = try ld.linkInputs(allocator, &.{
        .{ .object = main_o },
        .{ .archive = archive_bytes },
    }, base);
    defer image.deinit(allocator);

    // `helper` was pulled and resolved; `dead` (unreferenced) was not pulled.
    try std.testing.expect(image.addressOf("helper") != null);
    try std.testing.expect(image.addressOf("dead") == null);

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // executes the AArch64 ELF directly

    // A tiny entry stub: call main, then exit with its result. main sits right past
    // the 12-byte stub; bl is the first instruction (site offset 0), x0 already
    // holds main's return value (AAPCS64) once it returns.
    const main_off: i64 = @intCast(image.addressOf("main").? - base);
    const stub = [_]u32{
        encode.bl(@intCast(12 + main_off)), // bl main
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.aarch64, allocator, program.items, program.items.len, base, base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound, error.AccessDenied => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // helper() = 42
        else => return error.BackendFailed,
    }
}

/// `helper2() = 99` as its own one-function module.
fn buildHelper2Obj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var helper2 = Function.init(allocator);
    defer helper2.deinit();
    const t = try helper2.types.intern(i32k);
    const b = try helper2.appendBlock();
    const v = try helper2.appendInst(b, t, .{ .iconst = 99 });
    helper2.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "helper2", &helper2);
    return object.writeModule(allocator, &module);
}

/// `helper() = helper2()` as its own one-function module: an inter-member dependency,
/// so pulling `helper` out of an archive only satisfies `main`'s call and itself needs
/// another archive member (`helper2`) pulled in a later fixpoint sweep.
fn buildHelperCallsHelper2Obj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var helper = Function.init(allocator);
    defer helper.deinit();
    const t = try helper.types.intern(i32k);
    const b = try helper.appendBlock();
    const r = try helper.appendCall(b, t, "helper2", &.{});
    helper.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "helper", &helper);
    return object.writeModule(allocator, &module);
}

test "linkInputs archive input whose members are all unreferenced returns an empty image without panicking" {
    const allocator = std.testing.allocator;

    const helper_o = try buildHelperObj(allocator);
    defer allocator.free(helper_o);
    const dead_o = try buildDeadObj(allocator);
    defer allocator.free(dead_o);

    // No `.object` input, and nothing in the pull set references either member: the
    // fixpoint pulls nothing, so `included` stays empty. This must return the same
    // empty image the zero-object path returns, not panic on an empty first element.
    const archive_bytes = try buildArchive(allocator, &.{
        .{ .name = "helper.o", .bytes = helper_o },
        .{ .name = "dead.o", .bytes = dead_o },
    });
    defer allocator.free(archive_bytes);

    const base: u64 = 0x400000;
    var image = try ld.linkInputs(allocator, &.{
        .{ .archive = archive_bytes },
    }, base);
    defer image.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), image.code.len);
    try std.testing.expectEqual(@as(u64, 0), image.memsz);
    try std.testing.expect(image.addressOf("helper") == null);
    try std.testing.expect(image.addressOf("dead") == null);
}

test "linkInputs fixpoint pulls a two-hop inter-member archive dependency (main -> helper -> helper2)" {
    const allocator = std.testing.allocator;

    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);
    const helper_o = try buildHelperCallsHelper2Obj(allocator);
    defer allocator.free(helper_o);
    const helper2_o = try buildHelper2Obj(allocator);
    defer allocator.free(helper2_o);

    // Member order matters: `helper2.o` is listed BEFORE `helper.o`, so within a
    // single linear sweep over `candidates.items`, `helper2` is visited (and skipped,
    // still unreferenced) before `helper` is visited and pulled (which is what first
    // makes `helper2` needed). A single-pass fixpoint would stop right there and
    // never pull `helper2`, leaving `error.UndefinedSymbol`; only a second sweep (the
    // `while (changed)` loop) picks it up. This is what proves the fixpoint iterates.
    const archive_bytes = try buildArchive(allocator, &.{
        .{ .name = "helper2.o", .bytes = helper2_o },
        .{ .name = "helper.o", .bytes = helper_o },
    });
    defer allocator.free(archive_bytes);

    const base: u64 = 0x400000;
    var image = try ld.linkInputs(allocator, &.{
        .{ .object = main_o },
        .{ .archive = archive_bytes },
    }, base);
    defer image.deinit(allocator);

    try std.testing.expect(image.addressOf("helper") != null);
    try std.testing.expect(image.addressOf("helper2") != null);

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // executes the AArch64 ELF directly

    // Same entry stub shape as the one-hop test above: call main, then exit with its
    // result. main sits right past the 12-byte stub.
    const main_off: i64 = @intCast(image.addressOf("main").? - base);
    const stub = [_]u32{
        encode.bl(@intCast(12 + main_off)), // bl main
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.aarch64, allocator, program.items, program.items.len, base, base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound, error.AccessDenied => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 99), code), // helper() -> helper2() = 99
        else => return error.BackendFailed,
    }
}

test "linkInputs surfaces UndefinedSymbol when no archive member satisfies the call" {
    const allocator = std.testing.allocator;

    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);
    const dead_o = try buildDeadObj(allocator);
    defer allocator.free(dead_o);

    // The archive has a member, but it does not define `helper`.
    const archive_bytes = try buildArchive(allocator, &.{
        .{ .name = "dead.o", .bytes = dead_o },
    });
    defer allocator.free(archive_bytes);

    const base: u64 = 0x400000;
    try std.testing.expectError(error.UndefinedSymbol, ld.linkInputs(allocator, &.{
        .{ .object = main_o },
        .{ .archive = archive_bytes },
    }, base));
}
