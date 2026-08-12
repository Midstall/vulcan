//! This file tests external direct calls. It proves them end to end against a REAL glibc
//! `ld.so` on all four targets. The C frontend compiles the translation unit
//!   extern int add(int a, int b); int main(void){ return add(1, 41); }
//! Here `add` is a bodyless DECLARATION. It is defined in another object or `.so`. So the vcc
//! object emits `add` as an undefined (`SHN_UNDEF`) symbol, and the linker binds it to
//! `libadd.so`'s `add` export through a real PLT import. Each arch links three inputs.
//! These are the vcc `main.o` (defines `main`, imports `add`), a tiny hand-assembled `_start.o`
//! (defines `_start`, imports `main`, exits with `main`'s return value), and `libadd.so`
//! (an ET_DYN exporting `add`). The test links them with `ld.linkDynamic`, then runs the dynexe
//! under the real ld.so: natively on aarch64, under `qemu-<arch>` for the cross targets.
//! `add(1, 41) = 42`, so the process exits with code 42. Every arch skips cleanly if its
//! interpreter (or qemu) is not present. Before this feature, the vcc compile failed:
//! `add` was not in `l.funcs`, so it gave `error.Unsupported`. The frontend change makes the
//! external call lower to `appendCall`.

const std = @import("std");
const builtin = @import("builtin");
const cc = @import("vulcan-cc");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const ld = @import("vulcan-link");
const opt = @import("vulcan-opt");

/// Run `options` up to a few times and assert the child's exit code equals `expected_exit`.
/// This is a local copy of `vulcan-target/tests/run_helper.zig`'s `runExpectExit`. That file
/// lives in a different module, so this file cannot `@import` it directly. The function
/// retries ONLY on a non-`.exited` term (a spawn race, per that helper's doc). It asserts
/// immediately on a clean exit, so a genuine wrong-exit-code regression can never stay hidden.
fn runExpectExit(allocator: std.mem.Allocator, io: std.Io, options: std.process.RunOptions, expected_exit: u8) !void {
    var attempt: usize = 0;
    while (true) {
        attempt += 1;
        const run = try std.process.run(allocator, io, options);
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        switch (run.term) {
            .exited => |code| {
                try std.testing.expectEqual(expected_exit, code);
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

/// The translation unit under test: a call to the external `add` declared by a prototype.
const extern_src = "extern int add(int a, int b); int main(void){ return add(1, 41); }";

/// The translation unit under test for the external data read: a READ of the external data
/// global `counter`. It is declared `extern`, defined in another object or `.so`, and never
/// defined in this TU.
const counter_src = "extern int counter; int main(void){ return counter; }";

/// The SAME-TU function-pointer program: `add` is DEFINED in the same object as `main`. So
/// nothing is dynamically imported at all. `fp = add;` takes a plain intra-object symbol
/// address, and `fp(1, 41)` is a genuine `call_indirect`. This isolates the new x86 and
/// riscv64 indirect-call ISEL (aarch64 and x86-64 already had it) from every dynamic-linking
/// concern that the `extern_src` and `counter_src` tests exercise.
const fnptr_same_tu_src = "int add(int a, int b){ return a + b; } int main(void){ int (*fp)(int, int) = add; return fp(1, 41); }";

/// The EXTERNAL-via-pointer program: `add` is an `extern` prototype. It is defined in another
/// object or `.so`, and never defined in this TU, the same as `extern_src`. But unlike
/// `extern_src`, it is never called BY NAME. The only reference is `fp = add;`'s address-of.
/// The call itself goes through `fp`, a genuine indirect `call_indirect`.
const fnptr_extern_src = "extern int add(int a, int b); int main(void){ int (*fp)(int, int) = add; return fp(1, 41); }";

/// Compile `source` with the C frontend and emit a relocatable object for `arch` via
/// `writeObjectDataFor`. This is the same object-emit entry point that `tests/native.zig` and
/// the `vcc` driver use. The result defines `main`. It references, as an undefined symbol,
/// whatever external symbol `source` declares `extern` and never defines (`add` for
/// `extern_src`, `counter` for `counter_src`). The caller owns the returned bytes.
fn vccMainObj(allocator: std.mem.Allocator, arch: ld.Arch, source: []const u8) ![]u8 {
    return vccMainObjOpt(allocator, arch, source, false);
}

/// Like `vccMainObj`, but this optionally runs the optimizer's `mem2reg` (and friends)
/// on every function first, when `optimize` is true. The function-pointer tests need this.
/// Before optimization, a function-pointer LOCAL (`int (*fp)(int, int) = add;`) is a
/// `ptr`-typed `alloca`. x86-32's raw (un-optimized) object-emit path does not support this:
/// `typeSize` there is int-only. This is a PRE-EXISTING gap, unrelated to indirect calls.
/// Even a plain `int *p = &x;` local hits the same `error.Unsupported` on x86 today.
/// `mem2reg` promotes `fp` straight to an SSA value (it is never address-taken). This sidesteps
/// that gap entirely, while still genuinely exercising the new `.call_indirect` codegen this
/// feature adds: `fp`'s value flows directly into `call_indirect`'s target operand either way.
/// Every pre-existing (non-function-pointer) caller keeps `optimize = false`, so output stays
/// byte-identical.
fn vccMainObjOpt(allocator: std.mem.Allocator, arch: ld.Arch, source: []const u8, optimize: bool) ![]u8 {
    var mod = try cc.compile(allocator, source);
    defer mod.deinit(allocator);
    if (optimize) {
        for (mod.funcs) |*nf| _ = try opt.optimize(allocator, &nf.func);
    }

    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });

    // Map the frontend's data objects to the arch-independent `ObjData`. This TU has no
    // globals, so the list is empty, but the mapping stays faithful in case the TU grows.
    var objdata: std.ArrayList(target.native.ObjData) = .empty;
    defer objdata.deinit(allocator);
    for (mod.data) |d| try objdata.append(allocator, .{
        .name = d.name,
        .bytes = d.bytes,
        .kind = switch (d.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = d.size,
    });

    return target.native.writeObjectDataFor(allocator, arch, mfs.items, objdata.items);
}

/// `ld.Arch` and `cc.layout.Arch` are distinct enums with the same member names, but different
/// orders. So this maps by name. It mirrors `tests/variadic.zig`'s `layoutForLdArch`.
fn layoutForLdArch(arch: ld.Arch) cc.layout.Arch {
    return switch (arch) {
        .aarch64 => .aarch64,
        .riscv64 => .riscv64,
        .x86_64 => .x86_64,
        .x86 => .x86,
    };
}

/// Like `vccMainObj`, but builds the frontend types from `arch`'s OWN target layout, instead of
/// the build host's layout. This matters for struct-by-value. The ABI classification (register,
/// `.memory_stack`, or `.memory_ref`) and the `.memory_stack` chunk width are TARGET facts. So a
/// cross-arch struct test must reach the frontend through here. Otherwise it classifies with the
/// aarch64 host ABI, and never exercises the target's `.memory_stack` path. The caller owns the
/// bytes.
fn vccMainObjForTarget(allocator: std.mem.Allocator, arch: ld.Arch, source: []const u8) ![]u8 {
    var mod = try cc.compileForTarget(allocator, source, cc.layout.forArch(layoutForLdArch(arch)), .{});
    defer mod.deinit(allocator);

    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });

    var objdata: std.ArrayList(target.native.ObjData) = .empty;
    defer objdata.deinit(allocator);
    for (mod.data) |d| try objdata.append(allocator, .{
        .name = d.name,
        .bytes = d.bytes,
        .kind = switch (d.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        },
        .size = d.size,
    });

    return target.native.writeObjectDataFor(allocator, arch, mfs.items, objdata.items);
}

/// Locate a real glibc interpreter named `soname` in the Nix store. This is the loader to
/// embed in the dynexe. Returns an allocated absolute path, or null if none is found (the
/// caller then skips). Mirrors each dynamic test's `findGlibcInterp` and `findCrossInterp`.
fn findInterp(allocator: std.mem.Allocator, soname: []const u8) !?[]u8 {
    const cmd = try std.fmt.allocPrint(allocator, "find /nix/store -maxdepth 3 -name {s} 2>/dev/null | head -1", .{soname});
    defer allocator.free(cmd);
    const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ "sh", "-c", cmd } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Locate a `qemu-<suffix>` user-mode emulator on PATH, or null if absent (so the caller
/// skips cleanly). Mirrors each cross dynamic test's `findQemu`.
fn findQemu(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const cmd = try std.fmt.allocPrint(allocator, "command -v {s}", .{name});
    defer allocator.free(cmd);
    const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ "sh", "-c", cmd } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Write the shared library (as `so_name`, its own soname, for example `libadd.so` or
/// `libcounter.so`) and `dynexe` to a fresh tmp dir. Run the dynexe under the real ld.so,
/// prefixed by `runner` (for example `qemu-riscv64`, or empty for a native run). Assert
/// `exit(42)`. `LD_LIBRARY_PATH="."` lets ld.so find the library by soname in the cwd. The
/// on-disk file name must match the `DT_NEEDED` soname baked into `dynexe`. That is why the
/// `so_name` parameter exists.
fn writeAndRun(allocator: std.mem.Allocator, io: std.Io, so_name: []const u8, shared_so: []const u8, dynexe: []const u8, runner: ?[]const u8) !void {
    // Structural: an ET_EXEC (or ET_DYN PIE, not used here) dynamic executable.
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = so_name, .data = shared_so });
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe", .data = dynexe, .flags = .{ .permissions = .executable_file } });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    if (runner) |r| try argv.append(allocator, r);
    try argv.append(allocator, "./dynexe");

    runExpectExit(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // loader/qemu absent
        else => return e,
    };
}

/// Link `start_obj` and `main_obj` into a plain STATIC executable for `arch`, using
/// `ld.linkInputs` and `ld.writeExecutable`. This is the static path: no PT_INTERP or dynamic
/// segment at all, unlike `writeAndRun`'s dynamic executable plus real `ld.so`. Write the result
/// to a fresh tmp dir, and run it through `qemu-<arch>` when `runner` is non-null, or natively
/// when null. Assert `exit(42)`. The SAME-TU function-pointer test uses this helper: `add` is
/// DEFINED in `main_obj`'s own object, so there is nothing to dynamically resolve. This isolates
/// the new indirect-call ISEL from every dynamic-linking concern the `writeAndRun` tests exercise.
fn linkStaticAndRun(allocator: std.mem.Allocator, io: std.Io, arch: ld.Arch, start_obj: []const u8, main_obj: []const u8, runner: ?[]const u8) !void {
    var image = try ld.linkInputs(allocator, &.{ .{ .object = start_obj }, .{ .object = main_obj } }, 0x400000);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(arch, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    if (runner) |r| try argv.append(allocator, r);
    try argv.append(allocator, "./a.out");

    runExpectExit(allocator, io, .{
        .argv = argv.items,
        .cwd = .{ .dir = tmp.dir },
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // qemu absent
        else => return e,
    };
}

// --- aarch64 (native) ------------------------------------------------------------------

/// `libadd.so`'s object: a leaf `add(a, b) = a + b` (AAPCS64), `add` a defined global func.
fn addObjAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.aarch64.encode;
    const object = target.aarch64.object;
    const words = [_]u32{ encode.add(.x0, .x0, .x1), encode.ret() };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);
    const symbols = [_]object.Symbol{.{ .name = "add", .value = 0, .kind = .func, .defined = true, .section = .text }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &.{} });
}

/// `_start.o`: `bl main; mov x8, 93; svc 0`. This calls the intra-link `main` (an undefined
/// import bound to the vcc object), and exits with its return value (already in `x0`).
fn startObjAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.aarch64.encode;
    const object = target.aarch64.object;
    const words = [_]u32{ encode.bl(0), encode.movz(.x8, 93, 0), encode.svc(0) };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);
    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "main", .kind = .func, .defined = false, .section = .text }, // intra-link import
    };
    const relocs = [_]object.Reloc{.{ .offset = 0, .symbol = 1, .type = .call26 }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

test "aarch64: vcc external call to add() links against a .so and the REAL glibc ld.so runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // native execution
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-aarch64.so.1")) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    const add_obj = try addObjAArch64(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const main_obj = try vccMainObj(allocator, .aarch64, extern_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjAArch64(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, null);
}

/// `libcounter.so`'s object: a writable `.data` global `counter = 42` (STT_OBJECT), plus a
/// single `ret` so `.text` stays non-empty (the emitter and placement logic expect a `.text`
/// section). `counter` is a defined, global data symbol at `.data` offset 0. This mirrors
/// `buildCounterObj` in `aarch64/tests/dynamic.zig`.
fn counterObjAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.aarch64.encode;
    const object = target.aarch64.object;
    const words = [_]u32{encode.ret()};
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);
    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);
    const symbols = [_]object.Symbol{.{ .name = "counter", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data }};
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &.{} });
}

test "aarch64: vcc external data read of counter (extern, via GOT) links against a .so and the REAL glibc ld.so runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // native execution
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-aarch64.so.1")) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    const counter_obj = try counterObjAArch64(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{ .mode = .shared, .soname = "libcounter.so" });
    defer allocator.free(libcounter_so);

    const main_obj = try vccMainObj(allocator, .aarch64, counter_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjAArch64(allocator); // calls main, exits with its result
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libcounter.so", libcounter_so, dynexe, null);
}

// --- x86_64 (qemu) ---------------------------------------------------------------------

// x86-64 relocation code for a `call rel32` (`R_X86_64_PLT32`).
const R_X86_64_PLT32: u32 = 4;

/// One symbol for a raw ELF64 `.symtab` (see the x86_64 dynamic test's `Sym`).
const Sym64 = struct { name: []const u8, value: u64 = 0, shndx: u16, info: u8, size: u64 = 0 };
/// One RELA relocation against `.text` (see the x86_64 dynamic test's `Rel`).
const Rel64 = struct { offset: u64, sym: u32, typ: u32, addend: i64 };

/// Emit a minimal ELF64 x86-64 relocatable object (`ET_REL`, `EM_X86_64`). Copied from
/// `x86_64/tests/dynamic.zig`'s `writeRawObject`. x86-64's `object.zig` exposes only the
/// IR-driven `writeModule`, so a hand-assembled `_start` needs this low-level writer.
fn writeRawObject64(allocator: std.mem.Allocator, text: []const u8, syms: []const Sym64, rels: []const Rel64) ![]u8 {
    const w = std.mem.writeInt;
    const has_rela = rels.len > 0;
    const text_ndx: u16 = 1;
    const symtab_ndx: u16 = 2;
    const strtab_ndx: u16 = 3;
    const rela_ndx: u16 = 4;
    const shstrtab_ndx: u16 = if (has_rela) 5 else 4;
    const nsections: u16 = if (has_rela) 6 else 5;

    var strtab: std.ArrayList(u8) = .empty;
    defer strtab.deinit(allocator);
    try strtab.append(allocator, 0);
    var name_offs = try allocator.alloc(u32, syms.len);
    defer allocator.free(name_offs);
    for (syms, 0..) |s, i| {
        name_offs[i] = @intCast(strtab.items.len);
        try strtab.appendSlice(allocator, s.name);
        try strtab.append(allocator, 0);
    }

    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24);
    for (syms, 0..) |s, i| {
        var e: [24]u8 = @splat(0);
        w(u32, e[0..4], name_offs[i], .little);
        e[4] = s.info;
        e[5] = 0;
        w(u16, e[6..8], s.shndx, .little);
        w(u64, e[8..16], s.value, .little);
        w(u64, e[16..24], s.size, .little);
        try symtab.appendSlice(allocator, &e);
    }

    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (rels) |r| {
        var e: [24]u8 = @splat(0);
        w(u64, e[0..8], r.offset, .little);
        w(u64, e[8..16], (@as(u64, r.sym) << 32) | r.typ, .little);
        w(i64, e[16..24], r.addend, .little);
        try rela.appendSlice(allocator, &e);
    }

    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const symtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const rela_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rela.text\x00");
    const shstrtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    const alignUp = struct {
        fn f(v: usize, a: usize) usize {
            return std.mem.alignForward(usize, v, a);
        }
    }.f;
    var off: usize = 64;
    off = alignUp(off, 16);
    const text_off = off;
    off += text.len;
    off = alignUp(off, 8);
    const symtab_off = off;
    off += symtab.items.len;
    const strtab_off = off;
    off += strtab.items.len;
    off = alignUp(off, 8);
    const rela_off = off;
    if (has_rela) off += rela.items.len;
    const shstr_off = off;
    off += shstr.items.len;
    off = alignUp(off, 8);
    const sh_off = off;
    const total = sh_off + @as(usize, nsections) * 64;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2;
    buf[5] = 1;
    buf[6] = 1;
    w(u16, buf[16..18], 1, .little); // ET_REL
    w(u16, buf[18..20], 62, .little); // EM_X86_64
    w(u32, buf[20..24], 1, .little);
    w(u64, buf[40..48], sh_off, .little);
    w(u16, buf[52..54], 64, .little);
    w(u16, buf[58..60], 64, .little);
    w(u16, buf[60..62], nsections, .little);
    w(u16, buf[62..64], shstrtab_ndx, .little);

    @memcpy(buf[text_off..][0..text.len], text);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
    if (has_rela) @memcpy(buf[rela_off..][0..rela.items.len], rela.items);
    @memcpy(buf[shstr_off..][0..shstr.items.len], shstr.items);

    const putShdr = struct {
        fn f(b: []u8, at: usize, name: u32, typ: u32, flags: u64, o: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
            const ww = std.mem.writeInt;
            ww(u32, b[at + 0 ..][0..4], name, .little);
            ww(u32, b[at + 4 ..][0..4], typ, .little);
            ww(u64, b[at + 8 ..][0..8], flags, .little);
            ww(u64, b[at + 24 ..][0..8], o, .little);
            ww(u64, b[at + 32 ..][0..8], size, .little);
            ww(u32, b[at + 40 ..][0..4], sh_link, .little);
            ww(u32, b[at + 44 ..][0..4], sh_info, .little);
            ww(u64, b[at + 48 ..][0..8], addralign, .little);
            ww(u64, b[at + 56 ..][0..8], entsize, .little);
        }
    }.f;
    const SHT_PROGBITS: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_STRTAB: u32 = 3;
    const SHT_RELA: u32 = 4;
    const SHF_ALLOC: u64 = 0x2;
    const SHF_EXECINSTR: u64 = 0x4;
    const SHF_INFO_LINK: u64 = 0x40;

    putShdr(buf, sh_off + text_ndx * 64, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + symtab_ndx * 64, symtab_name, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 8, 24);
    putShdr(buf, sh_off + strtab_ndx * 64, strtab_name, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    if (has_rela) putShdr(buf, sh_off + rela_ndx * 64, rela_name, SHT_RELA, SHF_INFO_LINK, rela_off, rela.items.len, symtab_ndx, text_ndx, 8, 24);
    putShdr(buf, sh_off + shstrtab_ndx * 64, shstrtab_name, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);
    return buf;
}

/// `libadd.so`'s object: `add(a, b) = a + b` (SysV: `edi`+`esi` -> `eax`).
fn addObjX86_64(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{ 0x89, 0xf8, 0x01, 0xf0, 0xc3 }; // mov eax,edi ; add eax,esi ; ret
    const syms = [_]Sym64{.{ .name = "add", .shndx = 1, .info = 0x12 }};
    return writeRawObject64(allocator, &text, &syms, &.{});
}

/// `_start.o`: `call main; mov edi,eax; mov eax,60; syscall`. This calls the intra-link `main`
/// (undefined import), and exits with its return value.
///   e8 00 00 00 00   call main      @0  (rel32 @1, PLT32 reloc, addend -4)
///   89 c7            mov edi, eax    @5
///   b8 3c 00 00 00   mov eax, 60     @7  (SYS_exit)
///   0f 05            syscall         @12
fn startObjX86_64(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0xe8, 0x00, 0x00, 0x00, 0x00,
        0x89, 0xc7, 0xb8, 0x3c, 0x00,
        0x00, 0x00, 0x0f, 0x05,
    };
    const syms = [_]Sym64{
        .{ .name = "_start", .shndx = 1, .info = 0x12 },
        .{ .name = "main", .shndx = 0, .info = 0x10 }, // undefined import (intra-link)
    };
    const rels = [_]Rel64{.{ .offset = 1, .sym = 2, .typ = R_X86_64_PLT32, .addend = -4 }};
    return writeRawObject64(allocator, &text, &syms, &rels);
}

test "x86_64: vcc external call to add() links against a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-x86-64.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const add_obj = try addObjX86_64(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const main_obj = try vccMainObj(allocator, .x86_64, extern_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, qemu);
}

/// `libcounter.so`'s object via the IR-driven `link.Module` data API: an EXPORTED writable
/// `.data` global `counter = 42` (STT_OBJECT), plus a never-called leaf function so `.text`
/// stays non-empty. This mirrors `buildCounterObj` in `x86_64/tests/dynamic.zig`. x86-64's
/// `object.zig` only exposes the IR-driven `writeModule`, not a raw `object.write`.
fn counterObjX86_64(allocator: std.mem.Allocator) ![]u8 {
    const F = ir.function.Function;
    var dummy = F.init(allocator);
    defer dummy.deinit();
    {
        const t = try dummy.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try dummy.appendBlock();
        const x = try dummy.appendBlockParam(b, t);
        dummy.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }

    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);

    var module: target.x86_64.link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "_lib_dummy", &dummy);
    try module.addWritable(allocator, "counter", &data);
    return target.x86_64.object.writeModule(allocator, &module);
}

test "x86_64: vcc external data read of counter (extern, via GOT) links against a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-x86-64.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const counter_obj = try counterObjX86_64(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{ .mode = .shared, .soname = "libcounter.so" });
    defer allocator.free(libcounter_so);

    const main_obj = try vccMainObj(allocator, .x86_64, counter_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator); // calls main, exits with its result
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libcounter.so", libcounter_so, dynexe, qemu);
}

// --- x86 / i386 (qemu) -----------------------------------------------------------------

// i386 relocation code for a `call rel32` (`R_386_PC32`, in-field addend REL).
const R_386_PC32: u32 = 2;

/// One symbol for a raw ELF32 `.symtab` (see the x86 dynamic test's `Sym`).
const Sym32 = struct { name: []const u8, value: u32 = 0, shndx: u16, info: u8, size: u32 = 0 };
/// One `SHT_REL` relocation against `.text` (see the x86 dynamic test's `Rel`).
const Rel32 = struct { offset: u32, sym: u32, typ: u32 };

/// Emit a minimal ELF32 i386 relocatable object (`ET_REL`, `EM_386`). Copied from
/// `x86/tests/dynamic.zig`'s `writeRawObject32`, a data-less variant. It uses `Elf32_Sym` and
/// `Elf32_Rel` with `SHT_REL` (no addend field). The caller owns the returned bytes.
fn writeRawObject32(allocator: std.mem.Allocator, text: []const u8, syms: []const Sym32, rels: []const Rel32) ![]u8 {
    const w = std.mem.writeInt;
    const has_rel = rels.len > 0;
    const text_ndx: u16 = 1;
    const symtab_ndx: u16 = 2;
    const strtab_ndx: u16 = 3;
    const rel_ndx: u16 = 4;
    const shstrtab_ndx: u16 = if (has_rel) 5 else 4;
    const nsections: u16 = if (has_rel) 6 else 5;

    var strtab: std.ArrayList(u8) = .empty;
    defer strtab.deinit(allocator);
    try strtab.append(allocator, 0);
    var name_offs = try allocator.alloc(u32, syms.len);
    defer allocator.free(name_offs);
    for (syms, 0..) |s, i| {
        name_offs[i] = @intCast(strtab.items.len);
        try strtab.appendSlice(allocator, s.name);
        try strtab.append(allocator, 0);
    }

    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 16);
    for (syms, 0..) |s, i| {
        var e: [16]u8 = @splat(0);
        w(u32, e[0..4], name_offs[i], .little);
        w(u32, e[4..8], s.value, .little);
        w(u32, e[8..12], s.size, .little);
        e[12] = s.info;
        e[13] = 0;
        w(u16, e[14..16], s.shndx, .little);
        try symtab.appendSlice(allocator, &e);
    }

    var rel: std.ArrayList(u8) = .empty;
    defer rel.deinit(allocator);
    for (rels) |r| {
        var e: [8]u8 = @splat(0);
        w(u32, e[0..4], r.offset, .little);
        w(u32, e[4..8], (r.sym << 8) | r.typ, .little);
        try rel.appendSlice(allocator, &e);
    }

    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const symtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const rel_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rel.text\x00");
    const shstrtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    const alignUp = struct {
        fn f(v: usize, a: usize) usize {
            return std.mem.alignForward(usize, v, a);
        }
    }.f;
    var off: usize = 52;
    off = alignUp(off, 16);
    const text_off = off;
    off += text.len;
    off = alignUp(off, 4);
    const symtab_off = off;
    off += symtab.items.len;
    const strtab_off = off;
    off += strtab.items.len;
    off = alignUp(off, 4);
    const rel_off = off;
    if (has_rel) off += rel.items.len;
    const shstr_off = off;
    off += shstr.items.len;
    off = alignUp(off, 4);
    const sh_off = off;
    const total = sh_off + @as(usize, nsections) * 40;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1;
    buf[6] = 1;
    w(u16, buf[16..18], 1, .little); // ET_REL
    w(u16, buf[18..20], 3, .little); // EM_386
    w(u32, buf[20..24], 1, .little);
    w(u32, buf[32..36], @intCast(sh_off), .little);
    w(u16, buf[40..42], 52, .little);
    w(u16, buf[46..48], 40, .little);
    w(u16, buf[48..50], nsections, .little);
    w(u16, buf[50..52], shstrtab_ndx, .little);

    @memcpy(buf[text_off..][0..text.len], text);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
    if (has_rel) @memcpy(buf[rel_off..][0..rel.items.len], rel.items);
    @memcpy(buf[shstr_off..][0..shstr.items.len], shstr.items);

    const putShdr = struct {
        fn f(b: []u8, at: usize, name: u32, typ: u32, flags: u32, o: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u32, entsize: u32) void {
            const ww = std.mem.writeInt;
            ww(u32, b[at + 0 ..][0..4], name, .little);
            ww(u32, b[at + 4 ..][0..4], typ, .little);
            ww(u32, b[at + 8 ..][0..4], flags, .little);
            ww(u32, b[at + 16 ..][0..4], @intCast(o), .little);
            ww(u32, b[at + 20 ..][0..4], @intCast(size), .little);
            ww(u32, b[at + 24 ..][0..4], sh_link, .little);
            ww(u32, b[at + 28 ..][0..4], sh_info, .little);
            ww(u32, b[at + 32 ..][0..4], addralign, .little);
            ww(u32, b[at + 36 ..][0..4], entsize, .little);
        }
    }.f;
    const SHT_PROGBITS: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_STRTAB: u32 = 3;
    const SHT_REL: u32 = 9;
    const SHF_ALLOC: u32 = 0x2;
    const SHF_EXECINSTR: u32 = 0x4;
    const SHF_INFO_LINK: u32 = 0x40;

    putShdr(buf, sh_off + text_ndx * 40, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + symtab_ndx * 40, symtab_name, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 4, 16);
    putShdr(buf, sh_off + strtab_ndx * 40, strtab_name, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    if (has_rel) putShdr(buf, sh_off + rel_ndx * 40, rel_name, SHT_REL, SHF_INFO_LINK, rel_off, rel.items.len, symtab_ndx, text_ndx, 4, 8);
    putShdr(buf, sh_off + shstrtab_ndx * 40, shstrtab_name, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);
    return buf;
}

/// `libadd.so`'s object: `add(a, b) = a + b` (i386 cdecl: args on the stack -> `eax`).
fn addObjX86(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{ 0x8b, 0x44, 0x24, 0x04, 0x03, 0x44, 0x24, 0x08, 0xc3 }; // mov eax,[esp+4]; add eax,[esp+8]; ret
    const syms = [_]Sym32{.{ .name = "add", .shndx = 1, .info = 0x12 }};
    return writeRawObject32(allocator, &text, &syms, &.{});
}

/// `_start.o`: `call main; mov ebx,eax; mov eax,1; int 0x80`. This calls the intra-link `main`
/// (undefined import), and exits with its return value (i386 exit via `int 0x80`).
///   e8 fc ff ff ff   call main     @0  (rel32 @1, PC32 reloc, in-field addend -4. This is
///                                       the i386 `SHT_REL` convention the backend and linker
///                                       share, so the PC-relative `disp = S - P - 4` comes out
///                                       right.)
///   89 c3            mov ebx, eax   @5
///   b8 01 00 00 00   mov eax, 1     @7  (SYS_exit)
///   cd 80            int 0x80       @12
fn startObjX86(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0xe8, 0xfc, 0xff, 0xff, 0xff,
        0x89, 0xc3, 0xb8, 0x01, 0x00,
        0x00, 0x00, 0xcd, 0x80,
    };
    const syms = [_]Sym32{
        .{ .name = "_start", .shndx = 1, .info = 0x12 },
        .{ .name = "main", .shndx = 0, .info = 0x10 }, // undefined import (intra-link)
    };
    const rels = [_]Rel32{.{ .offset = 1, .sym = 2, .typ = R_386_PC32 }};
    return writeRawObject32(allocator, &text, &syms, &rels);
}

test "x86 (i386): vcc external call to add() links against an ELF32 .so and the REAL cross ld.so runs it to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const add_obj = try addObjX86(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const main_obj = try vccMainObj(allocator, .x86, extern_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, qemu);
}

/// `libcounter.so`'s object via the IR-driven `link.Module` data API: an EXPORTED writable
/// `.data` global `counter = 42` (STT_OBJECT), plus a never-called leaf function so `.text`
/// stays non-empty. This mirrors the x86_64 counterpart above. x86's `object.zig` likewise
/// only exposes the IR-driven `writeModule`, not a raw `object.write`.
fn counterObjX86(allocator: std.mem.Allocator) ![]u8 {
    const F = ir.function.Function;
    var dummy = F.init(allocator);
    defer dummy.deinit();
    {
        const t = try dummy.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try dummy.appendBlock();
        const x = try dummy.appendBlockParam(b, t);
        dummy.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }

    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);

    var module: target.x86.link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "_lib_dummy", &dummy);
    try module.addWritable(allocator, "counter", &data);
    return target.x86.object.writeModule(allocator, &module);
}

test "x86 (i386): vcc external data read of counter (extern, via GOT) links against an ELF32 .so and the REAL cross ld.so runs it to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const counter_obj = try counterObjX86(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{ .mode = .shared, .soname = "libcounter.so" });
    defer allocator.free(libcounter_so);

    const main_obj = try vccMainObj(allocator, .x86, counter_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86(allocator); // calls main, exits with its result
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libcounter.so", libcounter_so, dynexe, qemu);
}

// --- riscv64 (qemu) --------------------------------------------------------------------

/// `libadd.so`'s object: `add(a, b) = a + b` (lp64: `a0`+`a1` -> `a0`).
fn addObjRiscv64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.riscv64.encode;
    const object = target.riscv64.object;
    const words = [_]u32{ encode.add(.x10, .x10, .x11), encode.jalr(.x0, .x1, 0) }; // add a0,a0,a1 ; ret
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);
    const symbols = [_]object.Symbol{.{ .name = "add", .value = 0, .kind = .func, .defined = true, .section = .text }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &.{} });
}

/// `_start.o`: `auipc ra; jalr ra; addi a7,93; ecall`. This calls the intra-link `main`
/// (an undefined `R_RISCV_CALL` import), and exits with its return value (already in `a0`).
///   auipc ra, 0       @0   (R_RISCV_CALL -> main)
///   jalr  ra, ra, 0   @4
///   addi  a7, x0, 93  @8   (SYS_exit)
///   ecall             @12
fn startObjRiscv64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.riscv64.encode;
    const object = target.riscv64.object;
    const words = [_]u32{
        encode.auipc(.x1, 0),
        encode.jalr(.x1, .x1, 0),
        encode.addi(.x17, .x0, 93),
        encode.ecall(),
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);
    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "main", .kind = .func, .defined = false, .section = .text }, // intra-link import
    };
    const relocs = [_]object.Reloc{.{ .offset = 0, .symbol = 1, .type = .call }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// Scan an ELF relocatable object's `.symtab` (ELFCLASS32 or 64) for a symbol named `name`
/// that is UNDEFINED (`st_shndx == SHN_UNDEF == 0`). This proves the FRONTEND lowering
/// emitted the external `add` as an undefined reference in the vcc object, independent of
/// whether the linker can bind it downstream.
fn objectHasUndefinedSymbol(bytes: []const u8, name: []const u8) bool {
    const r = std.mem.readInt;
    if (bytes.len < 64 or !std.mem.eql(u8, bytes[0..4], "\x7fELF")) return false;
    const is64 = bytes[4] == 2;
    const shoff: u64 = if (is64) r(u64, bytes[40..48], .little) else r(u32, bytes[32..36], .little);
    const shentsize: u16 = if (is64) r(u16, bytes[58..60], .little) else r(u16, bytes[46..48], .little);
    const shnum: u16 = if (is64) r(u16, bytes[60..62], .little) else r(u16, bytes[48..50], .little);
    const SHT_SYMTAB: u32 = 2;
    var i: u16 = 0;
    while (i < shnum) : (i += 1) {
        const sh = bytes[@intCast(shoff + @as(u64, i) * shentsize)..];
        if (r(u32, sh[4..8], .little) != SHT_SYMTAB) continue;
        const sh_offset: u64 = if (is64) r(u64, sh[24..32], .little) else r(u32, sh[16..20], .little);
        const sh_size: u64 = if (is64) r(u64, sh[32..40], .little) else r(u32, sh[20..24], .little);
        const sh_link: u32 = if (is64) r(u32, sh[40..44], .little) else r(u32, sh[24..28], .little);
        const link_sh = bytes[@intCast(shoff + @as(u64, @intCast(sh_link)) * shentsize)..];
        const str_off: u64 = if (is64) r(u64, link_sh[24..32], .little) else r(u32, link_sh[16..20], .little);
        const entsz: u64 = if (is64) 24 else 16;
        var so: u64 = 0;
        while (so + entsz <= sh_size) : (so += entsz) {
            const sym = bytes[@intCast(sh_offset + so)..];
            const st_name = r(u32, sym[0..4], .little);
            const st_shndx: u16 = if (is64) r(u16, sym[6..8], .little) else r(u16, sym[14..16], .little);
            if (st_shndx != 0) continue; // not SHN_UNDEF
            const nm = std.mem.sliceTo(bytes[@intCast(str_off + st_name)..], 0);
            if (std.mem.eql(u8, nm, name)) return true;
        }
    }
    return false;
}

test "riscv64: vcc external call to add() links against a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-riscv64-lp64d.so.1")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObj(allocator, .riscv64, extern_src);
    defer allocator.free(main_obj);

    // The frontend deliverable: the external call lowers to an undefined `add`
    // reference in the vcc object. This holds on riscv64 exactly as on the other arches. The
    // `.call` lowering arm is arch-independent, so assert it directly.
    try std.testing.expect(objectHasUndefinedSymbol(main_obj, "add"));

    const add_obj = try addObjRiscv64(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    // `main_obj`'s `jal add` (VCC's riscv64 codegen lowers every direct call to a single
    // near `jal`, with a +/-1MiB reach) is classified as a PLT-bindable function import.
    // `resolve.isCallReloc` treats riscv64's `.jal` the same as `.call`. The linker redirects
    // it to a synthesized PLT entry (`riscv64.redirectCallNear`), exactly like the other three
    // arches' single-instruction call forms.
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, qemu);
}

/// `libcounter.so`'s object: a writable `.data` global `counter = 42` (STT_OBJECT), plus a
/// single `ret` so `.text` stays non-empty. This mirrors `buildCounterObj` in
/// `riscv64/tests/dynamic.zig`.
fn counterObjRiscv64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.riscv64.encode;
    const object = target.riscv64.object;
    const words = [_]u32{encode.jalr(.x0, .x1, 0)}; // ret
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);
    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);
    const symbols = [_]object.Symbol{.{ .name = "counter", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data }};
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &.{} });
}

test "riscv64: vcc external data read of counter (extern, via GOT) links against a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-riscv64-lp64d.so.1")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObj(allocator, .riscv64, counter_src);
    defer allocator.free(main_obj);

    // The frontend deliverable: the extern `counter` reference lowers to an
    // undefined data symbol reference in the vcc object. This mirrors the earlier
    // `objectHasUndefinedSymbol` check for the call case.
    try std.testing.expect(objectHasUndefinedSymbol(main_obj, "counter"));

    const counter_obj = try counterObjRiscv64(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{ .mode = .shared, .soname = "libcounter.so" });
    defer allocator.free(libcounter_so);

    const start_obj = try startObjRiscv64(allocator); // calls main, exits with its result
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libcounter.so", libcounter_so, dynexe, qemu);
}

// --- Function pointers -------------------------------------------------------------------
//
// Two programs, per arch. (a) SAME-TU (`fnptr_same_tu_src`): `add` is defined in the same
// object as `main`. A plain STATIC link (`linkStaticAndRun`, with no `.so` and no real `ld.so`
// at all) isolates the new indirect-call ISEL. (x86 and riscv64 gained `.call_indirect` with
// this feature; aarch64 and x86-64 already had it.) (b) EXTERNAL-via-pointer
// (`fnptr_extern_src`): `add` is `extern`, defined only in `libadd.so`, and NEVER called by
// name (the only reference is `fp = add;`'s address-of). This reuses each arch's existing
// `addObj*` and `libadd.so` fixture from the external-call tests above, linked dynamically
// (`writeAndRun`) exactly like `extern_src`, but through `fnptr_extern_src`.

test "aarch64: same-TU function pointer call natively runs to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // native execution
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const main_obj = try vccMainObjOpt(allocator, .aarch64, fnptr_same_tu_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjAArch64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .aarch64, start_obj, main_obj, null);
}

test "aarch64: function pointer to an extern function links against a .so and the REAL glibc ld.so runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // native execution
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-aarch64.so.1")) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    const add_obj = try addObjAArch64(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const main_obj = try vccMainObjOpt(allocator, .aarch64, fnptr_extern_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjAArch64(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, null);
}

test "x86_64: same-TU function pointer call runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjOpt(allocator, .x86_64, fnptr_same_tu_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "x86_64: function pointer to an extern function links against a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-x86-64.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const add_obj = try addObjX86_64(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const main_obj = try vccMainObjOpt(allocator, .x86_64, fnptr_extern_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, qemu);
}

test "x86 (i386): same-TU function pointer call runs to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjOpt(allocator, .x86, fnptr_same_tu_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86, start_obj, main_obj, qemu);
}

test "x86 (i386): function pointer to an extern function links against an ELF32 .so and the REAL cross ld.so runs it to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux.so.2")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-i386")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const add_obj = try addObjX86(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const main_obj = try vccMainObjOpt(allocator, .x86, fnptr_extern_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, qemu);
}

test "riscv64: same-TU function pointer call runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjOpt(allocator, .riscv64, fnptr_same_tu_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

/// This test targets a regression: HIGH ARGUMENT PRESSURE for `.call_indirect`'s target
/// staging. `callit`'s function-pointer parameter `fp` is its FIRST int parameter, so it is
/// entry-pinned to `a0` (riscv64's lp64 ABI). Since nothing else in `callit` needs `a0`, it
/// stays resident there for the rest of the function, under the shared Wimmer-Franz allocator.
/// (`a0`..`a7` sit outside the general allocatable pool. A value only occupies one of them via
/// a FIXED assignment, entry params included, and nothing forces a move-out when it is not
/// needed.) `callit` then makes an 8-argument indirect call through `fp`. Its FIRST argument
/// (`a`, the literal `1`) is entry-pinned to `a1`, and gets moved into `a0` as part of the
/// call's own argument setup. That move lands directly on the register the buggy, pre-fix code
/// would still be reading the target address out of. Without the fix, `reloadInt` returned
/// `cl.target`'s raw resident register verbatim, never staged into a scratch register before
/// the argument moves. So that argument move clobbers `fp`'s address in `a0` before the `jalr`,
/// and the call jumps through whatever integer the argument setup left behind, instead of
/// `add8`'s address. This is not merely a wrong answer: it is a jalr through a small integer,
/// which crashes (SIGILL or SIGSEGV) under qemu, long before any `exit`. With the fix, the
/// target is staged into the dedicated `spill_scratch1` register (x8, outside a0-a7 and outside
/// `parallelMoveInt`'s own `spill_scratch0`/x6 cycle-break scratch) before any argument move
/// runs, so it survives untouched through to the `jalr`. `add8(1..7, 14) = 1+2+3+4+5+6+7+14 =
/// 42`, so the process exits with code 42.
const fnptr_high_pressure_src =
    \\int add8(int a,int b,int c,int d,int e,int f,int g,int h){
    \\    return a+b+c+d+e+f+g+h;
    \\}
    \\int callit(int (*fp)(int,int,int,int,int,int,int,int), int a,int b,int c,int d,int e,int f,int g){
    \\    return fp(a,b,c,d,e,f,g,14);
    \\}
    \\int main(void){
    \\    return callit(add8, 1,2,3,4,5,6,7);
    \\}
;

test "riscv64: function pointer call under high argument pressure (a0-a7 all live) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjOpt(allocator, .riscv64, fnptr_high_pressure_src, true);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

test "riscv64: function pointer to an extern function links against a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findInterp(allocator, "ld-linux-riscv64-lp64d.so.1")) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjOpt(allocator, .riscv64, fnptr_extern_src, true);
    defer allocator.free(main_obj);

    const add_obj = try addObjRiscv64(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{ .mode = .shared, .soname = "libadd.so" });
    defer allocator.free(libadd_so);

    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = main_obj }, .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    try writeAndRun(allocator, io, "libadd.so", libadd_so, dynexe, qemu);
}

// --- Stack arguments and `.memory_stack` struct decomposition ----------------------------

/// A direct proof, independent of structs: a plain function with EIGHT integer parameters.
/// The System V x86_64 ABI passes the first six in rdi..r9, and the last two on the stack. So
/// this exercises the net-new caller stack-arg placement (a local `sub rsp` window, stores to
/// [rsp+k*8], `add rsp`), and the callee stack-arg home ([rsp+frame+8+k*8] into each
/// parameter's location). `1+2+3+4+5+6+7+14 == 42`, so `_start` exits 42 when the whole path
/// is correct.
const sum8_src = "int sum8(int a,int b,int c,int d,int e,int f,int g,int h){ return a+b+c+d+e+f+g+h; } int main(void){ return sum8(1,2,3,4,5,6,7,14); }";

/// On i386, an 8-byte struct is i386's `.memory_stack` class (i386 passes every struct by
/// value on the stack). The frontend decomposes it into two 4-byte chunks (i386 is a 32-bit,
/// integer-only backend, so a general chunk is one 32-bit word), which cdecl pushes positionally.
/// `20 + 22 == 42`.
const struct_stack_i386_src = "struct P { int x; int y; }; int take(struct P p){ return p.x + p.y; } int main(void){ struct P p; p.x = 20; p.y = 22; return take(p); }";

/// On x86_64, combining both proofs above: a 64-byte struct is over 16 bytes, so its class is
/// `.memory_stack` (SysV MEMORY, passed by value on the stack). The frontend decomposes it into
/// EIGHT 8-byte chunks. The first six land in rdi..r9, and chunks 6 and 7 become x86_64 stack
/// arguments. So a single struct-by-value argument drives the new stack-arg path on BOTH sides.
/// `take` reads a field in chunk 0 (a register chunk) and a field in chunk 7 (a stack-homed
/// chunk). `20 + 22 == 42`.
const struct_stack_x86_64_src = "struct Huge { int v[16]; }; int take(struct Huge h){ return h.v[0] + h.v[15]; } int main(void){ struct Huge h; h.v[0] = 20; h.v[15] = 22; return take(h); }";

test "x86_64: a call with EIGHT integer args passes two on the stack and runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, sum8_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "x86 (i386): a struct-by-value arg decomposes into two 32-bit .memory_stack chunks (frontend)" {
    // i386 passes every struct by value on the stack (`.memory_stack`). The frontend decomposes
    // an 8-byte struct into two GENERAL-register-wide (32-bit) integer chunks. This asserts that
    // decomposition at the IR level. `take`'s entry block takes exactly two 32-bit integer
    // parameters, NOT the aarch64 host's single register i64 (`.registers`), and NOT a
    // `.memory_ref` pointer. That is the one i386-specific half of the proof above (a word-wide
    // chunk, 4 bytes here versus 8 on x86_64). i386 EXECUTION of a struct-by-value program is
    // blocked by a SEPARATE, pre-existing i386 backend gap: a struct local is an `array{ i64 }`
    // storage blob, which the integer-only 32-bit backend cannot size or load. So the proof
    // stops at the frontend, the part this test owns. The x86_64 struct test below runs the SAME
    // `.memory_stack` decomposition engine end to end under qemu.
    const allocator = std.testing.allocator;

    var mod = try cc.compileForTarget(allocator, struct_stack_i386_src, cc.layout.forArch(.x86), .{});
    defer mod.deinit(allocator);

    var found = false;
    for (mod.funcs) |*nf| {
        if (!std.mem.eql(u8, nf.name, "take")) continue;
        found = true;
        const params = nf.func.blockParams(@enumFromInt(0));
        try std.testing.expectEqual(@as(usize, 2), params.len);
        for (params) |p| {
            const k = nf.func.types.type_kind(nf.func.valueType(p));
            try std.testing.expect(k == .int);
            try std.testing.expectEqual(@as(u16, 32), k.int.bits);
        }
    }
    try std.testing.expect(found);
}

test "x86_64: an over-16-byte struct-by-value arg overflows onto the stack and runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, struct_stack_x86_64_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

// --- Register struct RETURNS on x86_64 (rax:rdx) and riscv64 (a0:a1) ---------------------

/// A struct RETURNED BY VALUE in registers, driving both the single-register and the
/// register-pair return paths in one program. `struct P` is 8 bytes (two ints), one integer
/// eightbyte, so it comes back in the first return register (rax / a0). `struct Q` is 16 bytes
/// (two longs), two integer eightbytes, so it comes back in the return pair (rax:rdx / a0:a1).
/// The callee places each eightbyte, and `main`'s caller stores the return registers into the
/// destination slot, then reads the fields back. `4 + 6 + 12 + 20 == 42`.
const struct_ret_regs_src =
    "struct P { int x; int y; };" ++
    " struct Q { long a; long b; };" ++
    " struct P mkp(void){ struct P p; p.x = 4; p.y = 6; return p; }" ++
    " struct Q mkq(void){ struct Q q; q.a = 12; q.b = 20; return q; }" ++
    " int main(void){ struct P p = mkp(); struct Q q = mkq(); return p.x + p.y + q.a + q.b; }";

test "x86_64: a struct returned by value in rax:rdx runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, struct_ret_regs_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: a struct returned by value in a0:a1 runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, struct_ret_regs_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

// --- FLOAT-class struct RETURNS on x86_64 (SSE) and riscv64 (FP-struct) ------------------

/// A struct with FLOAT-class eightbytes RETURNED BY VALUE, driving the all-float and the mixed
/// integer+float return paths in one program. `struct F` is two doubles, so it comes back in two
/// FP registers (xmm0:xmm1 on x86_64, fa0:fa1 on riscv64). `struct M` is `{int; double}`, a mixed
/// return, so its integer eightbyte comes back in the first integer register (rax / a0) and its
/// float eightbyte in the first FP register (xmm0 / fa0). The callee places each eightbyte into
/// the return register of its bank, and `main`'s caller stores each into the destination slot,
/// then reads the fields back. `(10.5 + 11.5) + (5 + 15.0) == 22 + 20 == 42`.
const struct_ret_float_src =
    "struct F { double a; double b; };" ++
    " struct M { int i; double d; };" ++
    " struct F mkf(void){ struct F v; v.a = 10.5; v.b = 11.5; return v; }" ++
    " struct M mkm(void){ struct M v; v.i = 5; v.d = 15.0; return v; }" ++
    " int main(void){ struct F f = mkf(); struct M m = mkm(); return (int)(f.a + f.b) + (int)(m.i + m.d); }";

test "x86_64: a float-class struct returned by value in SSE registers runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, struct_ret_float_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: a float-class struct returned by value in FP registers runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, struct_ret_float_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

/// A struct of two `float`s returned by value. On riscv64's lp64d ABI each `float` field is its
/// own FP-struct eightbyte in its own `fa` register (fa0 at offset 0, fa1 at offset 4), so the
/// caller stores each back with a 4-byte `fsw` (not the 8-byte `fsd` a `double` return uses).
/// `20.0 + 22.0 == 42`.
const struct_ret_two_floats_src =
    "struct FF { float a; float b; };" ++
    " struct FF mk(void){ struct FF v; v.a = 20.0f; v.b = 22.0f; return v; }" ++
    " int main(void){ struct FF v = mk(); return (int)(v.a + v.b); }";

test "riscv64: a two-float struct returned by value uses 4-byte fsw stores and runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, struct_ret_two_floats_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

// --- Hidden-pointer (`.sret`) struct RETURNS on x86_64 and riscv64 -----------------------

/// A struct LARGER than 16 bytes returned BY VALUE. `struct Big` is 24 bytes (three longs), so
/// its return plan is `.sret`: the caller allocates the destination slot and passes its address
/// as a hidden result pointer (an ordinary first integer argument on both targets, rdi / a0), and
/// the callee copies its return value through it. On x86_64 and riscv64, the frontend's uniform
/// hidden-pointer prepend, plus the count-1 return echo, need NO backend change. (The hidden
/// pointer lands positionally in the first argument register, and the return echoes it in rax /
/// a0.) The aarch64 path, which routes the hidden pointer through x8, is proven natively by
/// `native.zig`. `10 + 12 + 20 == 42`.
const struct_ret_sret_src =
    "struct Big { long a; long b; long c; };" ++
    " struct Big mk(void){ struct Big b; b.a = 10; b.b = 12; b.c = 20; return b; }" ++
    " int main(void){ struct Big r = mk(); return r.a + r.b + r.c; }";

test "x86_64: an over-16-byte struct returned by value through the hidden pointer runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, struct_ret_sret_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: an over-16-byte struct returned by value through the hidden pointer runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, struct_ret_sret_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

test "x86 (i386): an over-16-byte struct return threads a hidden result pointer (frontend)" {
    // i386 returns EVERY struct through a hidden result pointer (`.sret`). The frontend marks the
    // struct-returning function `sret` and prepends the hidden pointer as its FIRST entry-block
    // parameter (a plain pointer). This asserts that at the IR level. i386 EXECUTION of a
    // struct-returning program is blocked by the SAME pre-existing i386 backend gap noted above: a
    // struct local is an `array{ i64 }` storage blob the integer-only 32-bit backend cannot size.
    // So the proof stops at the frontend, the part this test owns. The x86_64 and riscv64 sret
    // tests above run the SAME hidden-pointer engine end to end under qemu, and aarch64 runs it
    // natively.
    const allocator = std.testing.allocator;

    var mod = try cc.compileForTarget(allocator, struct_ret_sret_src, cc.layout.forArch(.x86), .{});
    defer mod.deinit(allocator);

    var found = false;
    for (mod.funcs) |*nf| {
        if (!std.mem.eql(u8, nf.name, "mk")) continue;
        found = true;
        try std.testing.expect(nf.func.sret);
        const params = nf.func.blockParams(@enumFromInt(0));
        try std.testing.expectEqual(@as(usize, 1), params.len); // only the hidden pointer (mk is void-param)
        try std.testing.expect(nf.func.types.type_kind(nf.func.valueType(params[0])) == .ptr);
    }
    try std.testing.expect(found);
}

// --- Float-class (`.sse`) struct ARGUMENTS ------------------------------------------------

/// A struct with two `double` fields is `.sse` on every target here: two 8-byte SSE eightbytes
/// on x86_64 (xmm0/xmm1), an AAPCS64 HFA of two doubles on aarch64 (d0/d1, proven natively by
/// `native.zig`), and RISC-V lp64d's FP-struct flatten (fa0/fa1). The frontend loads each
/// eightbyte as an `f64` value from its offset and appends it, so the backend routes it to the
/// next FP argument register with NO backend change. The final cast to `int` keeps the exit
/// code exact. `20.0 + 22.0 == 42`.
const struct_sse_arg_src = "struct F { double a; double b; }; int take(struct F v){ return (int)(v.a + v.b); } int main(void){ struct F v; v.a = 20.0; v.b = 22.0; return take(v); }";

test "x86_64: a struct-by-value arg with two SSE eightbytes (xmm0/xmm1) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, struct_sse_arg_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: a struct-by-value arg flattened into two FP registers (fa0/fa1) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, struct_sse_arg_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

/// A MIXED struct (one integer eightbyte plus one SSE eightbyte) exercises the interleave. The
/// `int` field routes through the general-purpose argument counter. The `double` field routes
/// through the FP one, in eightbyte order. The `double`-to-`int` narrowing goes through a plain
/// local (`int di = v.d;`), rather than an explicit `(int)v.d` cast. Casting a struct MEMBER
/// ACCESS directly hits a separate, PRE-EXISTING `error.Unsupported`, unrelated to this feature.
/// This was confirmed on a struct with a single `int` field and no struct-by-value argument at
/// all. It also fails for `(long)v.i`. `3*10 + 12 == 42`.
const struct_sse_mixed_arg_src = "struct M { int i; double d; }; int take(struct M v){ int di = v.d; return v.i * 10 + di; } int main(void){ struct M v; v.i = 3; v.d = 12.0; return take(v); }";

test "x86_64: a mixed integer+SSE struct-by-value arg (GPR + xmm) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, struct_sse_mixed_arg_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: a mixed integer+SSE struct-by-value arg (GPR + FPR) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, struct_sse_mixed_arg_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

// --- Cross-arch float-struct differential, and the whole-feature integration -------------

/// A `struct {double x, y;}` RETURNED by one call and PASSED to another. This combines the
/// `.sse`-eightbyte arg path and the `.sse`-eightbyte return path, sharing one destination slot,
/// in a single program instead of two separate ones. `native.zig` runs the SAME two-function
/// shape natively on the aarch64 host (its gcc-diff twin). This proves it cross-arch under qemu
/// (x86_64 xmm0/xmm1, riscv64 fa0/fa1). `20.0 + 22.0 == 42`. `double r = addv(v); int c =
/// (int)r;` casts a LOCAL, not the call itself. This routes around a pre-existing parser gap,
/// unrelated to struct-by-value. See the `m4d_integration_src` doc comment below for the root
/// cause.
const float_struct_roundtrip_src =
    "struct F { double x; double y; };" ++
    " struct F make(double x, double y){ struct F v; v.x = x; v.y = y; return v; }" ++
    " double addv(struct F v){ return v.x + v.y; }" ++
    " int main(void){ struct F v = make(20.0, 22.0); double r = addv(v); return (int)r; }";

test "x86_64: a float-class struct made by one call and consumed by another runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, float_struct_roundtrip_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: a float-class struct made by one call and consumed by another runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, float_struct_roundtrip_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}

/// The WHOLE-FEATURE integration, cross-arch: a small all-integer struct (`.registers`), a
/// large over-16-byte struct (`.memory_ref`/`.memory_stack` arg, `.sret` return), an all-double
/// HFA/FP-struct, and a mixed int+double struct, each made by a struct-RETURNING call and
/// consumed by a struct-by-value ARGUMENT, composed in one program. `native.zig` runs the exact
/// same shape natively on the aarch64 host (gcc-diff). This is its qemu cross-arch twin, where a
/// hardcoded expected exit stands in for the gcc oracle (unavailable when cross-compiling).
/// `sumSmall(mkSmall(1,2))=3`, `sumBig(mkBig(10,10,11))=31`, `(int)sumF(mkF(1.5,2.5))=4`, and
/// `(int)sumMixed(mkMixed(3,1.5))=4`. `3 + 31 + 4 + 4 == 42`.
///
/// `double fc = sumF(...); int c = (int)fc;` casts a plain LOCAL, not the call itself. This
/// works around a PRE-EXISTING parser gap, unrelated to struct-by-value, that this integration
/// test surfaced while bisecting an unrelated failure. A cast's operand parses through
/// `parseAtom` (see `parser.zig`'s `.lparen`/cast-vs-parenthesized-expr case), which stops
/// BEFORE the postfix chain (`parsePrimary`'s `()`/`[]`/`.`/`->` loop). So `(int)g()` misparses
/// as `((int)g)()` (cast `g` first, THEN call the cast result) instead of C's `(int)(g())`.
/// This is the SAME root cause as the already-documented "cast of a struct member access" gap
/// that the earlier float-struct-argument tests route around the identical way (`(double)v.i`
/// misparses as `((double)v).i`). It is confirmed broader here: it reproduces with NO structs at
/// all, on a bare `(int)g()`. This is reported, but NOT fixed. It is out of scope: it needs a
/// general parser precedence fix, not a struct-by-value ABI change.
const m4d_integration_src =
    "struct Small{int x; int y;};" ++
    " struct Big{long a; long b; long c;};" ++
    " struct F{double x; double y;};" ++
    " struct Mixed{int i; double d;};" ++
    " struct Small mkSmall(int x, int y){ struct Small s; s.x=x; s.y=y; return s; }" ++
    " int sumSmall(struct Small s){ return s.x + s.y; }" ++
    " struct Big mkBig(long a, long b, long c){ struct Big g; g.a=a; g.b=b; g.c=c; return g; }" ++
    " long sumBig(struct Big g){ return g.a + g.b + g.c; }" ++
    " struct F mkF(double x, double y){ struct F v; v.x=x; v.y=y; return v; }" ++
    " double sumF(struct F v){ return v.x + v.y; }" ++
    " struct Mixed mkMixed(int i, double d){ struct Mixed m; m.i=i; m.d=d; return m; }" ++
    " double sumMixed(struct Mixed m){ double di = m.i; return di + m.d; }" ++
    " int main(void){" ++
    // This accumulates into `r` immediately, instead of keeping four separate live locals for
    // a final combined sum. `native.zig`'s comment on the same shape explains why: a separate,
    // pre-existing Wimmer register-allocator spill limit under heavy simultaneous liveness,
    // unrelated to struct-by-value correctness.
    "   int r = sumSmall(mkSmall(1, 2));" ++
    "   long b = sumBig(mkBig(10, 10, 11));" ++
    "   r = r + (int)b;" ++
    "   double fc = sumF(mkF(1.5, 2.5));" ++
    "   r = r + (int)fc;" ++
    "   double fd = sumMixed(mkMixed(3, 1.5));" ++
    "   r = r + (int)fd;" ++
    "   return r;" ++
    " }";

test "x86_64: the struct-by-value integration (small/large/float/mixed structs, args and returns together) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-x86_64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .x86_64, m4d_integration_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjX86_64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .x86_64, start_obj, main_obj, qemu);
}

test "riscv64: the struct-by-value integration (small/large/float/mixed structs, args and returns together) runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const qemu = (try findQemu(allocator, "qemu-riscv64")) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const main_obj = try vccMainObjForTarget(allocator, .riscv64, m4d_integration_src);
    defer allocator.free(main_obj);
    const start_obj = try startObjRiscv64(allocator);
    defer allocator.free(start_obj);

    try linkStaticAndRun(allocator, io, .riscv64, start_obj, main_obj, qemu);
}
