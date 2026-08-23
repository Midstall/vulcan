//! Dynamic-linking coverage for the shared linker (`vulcan-link`) on riscv64: the PLT/GOT
//! + eager JUMP_SLOT import path, proven by the REAL cross glibc
//! `ld-linux-riscv64-lp64d.so.1` under `qemu-riscv64`. Builds `libadd.so` (an ET_DYN
//! exporting a leaf `add(a, b) = a + b`) and a dynexe whose `_start` CALLS the imported
//! `add(1, 41)` (an undefined `R_RISCV_CALL` bound to the `.so`) then does the riscv64
//! `exit` syscall with the result. The kernel-substitute (qemu) reads PT_INTERP, runs the
//! cross ld.so, which loads `libadd.so`, eagerly binds the JUMP_SLOT (`DF_BIND_NOW`), and
//! enters `_start` -> `add(1, 41) = 42` -> `exit(42)`. The riscv64 PLT reuses riscv64's
//! own static extern stub (`auipc t0, %pcrel_hi(got)` / `ld t0, %pcrel_lo(got)(t0)` /
//! `jr t0`) to load the resolved address out of the GOT slot and tail-jump. Both qemu and
//! the cross loader are gated on (absent -> the test skips cleanly).

const std = @import("std");
const object = @import("../object.zig");
const encode = @import("../encode.zig");
const link = @import("../link.zig");
const ld = @import("vulcan-link");
const run_helper = @import("../../tests/run_helper.zig");

/// Build `libadd.so`'s object: a leaf `add(a, b) = a + b` (riscv64 lp64: `a` in `a0`/x10,
/// `b` in `a1`/x11, result in `a0`), `add` a global, defined function at `.text` offset 0.
///   add  a0, a0, a1
///   ret  (jalr x0, x1, 0)
fn buildAddObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.add(.x10, .x10, .x11), // a0 = a0 + a1
        encode.jalr(.x0, .x1, 0), // ret
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "add", .value = 0, .kind = .func, .defined = true, .section = .text },
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &.{} });
}

/// Build the dynexe's object: `_start` calls the imported `add(1, 41)` then `exit`s with the
/// result. `add` is UNDEFINED (an import, bound to `libadd.so`). The `auipc ra` + `jalr ra`
/// far-call pair carries an `R_RISCV_CALL` at the `auipc` (offset 8). Freestanding: the exit
/// is a raw `ecall`, no libc.
///   addi a0, x0, 1      @0   (add's first arg)
///   addi a1, x0, 41     @4   (add's second arg)
///   auipc ra, 0         @8   (R_RISCV_CALL reloc -> add, patched to reach the PLT)
///   jalr  ra, ra, 0     @12
///   addi a7, x0, 93     @16  (SYS_exit)
///   ecall               @20  -> exit(a0 = add's result)
fn buildDynStartObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.addi(.x10, .x0, 1), // a0 = 1
        encode.addi(.x11, .x0, 41), // a1 = 41
        encode.auipc(.x1, 0), // call add (hi) - R_RISCV_CALL @8
        encode.jalr(.x1, .x1, 0), // call add (lo)
        encode.addi(.x17, .x0, 93), // a7 = 93 (exit)
        encode.ecall(), // ecall -> exit(a0)
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "add", .kind = .func, .defined = false, .section = .text }, // import
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 8, .symbol = 1, .type = .call }, // auipc/jalr add -> symbol index 1
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// A `.o` exporting ONLY a writable `.data` global `counter = 42` (STT_OBJECT). A single
/// `ret` keeps `.text` non-empty (the emitter/placement expect a `.text` section). `counter`
/// is a defined, global data symbol at `.data` offset 0. This is the shared library's data
/// global the dynexe imports via the GOT.
fn buildCounterObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{encode.jalr(.x0, .x1, 0)}; // ret
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "counter", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data },
    };
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &.{} });
}

/// A `_start` `.o` that IMPORTS the data global `counter` via the GOT (GOT-indirect
/// addressing), loads the i32 it points at, and exits with that value. `counter` is an
/// UNDEFINED STT_OBJECT global (a data import bound to a `.so`). The `auipc` at `.text` offset
/// 0 carries the `R_RISCV_GOT_HI20` relocation and the adjacent `ld` (offset 4) loads
/// `counter`'s address out of the GOT slot (its low-12 half is patched by the emitter, paired
/// to the `auipc`). The `lw a0, 0(a0)` then dereferences the GOT-resolved address.
/// Freestanding: the exit is a raw `ecall`, no libc.
///   auipc a0, 0        @0   (R_RISCV_GOT_HI20 -> counter, patched to the GOT slot)
///   ld    a0, 0(a0)    @4   -> a0 = *(GOT slot) = &counter
///   lw    a0, 0(a0)    @8   -> a0 = *counter = 42
///   addi  a7, x0, 93   @12  (SYS_exit)
///   ecall              @16  -> exit(a0)
fn buildDataStartObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.auipc(.x10, 0), // a0 = pc + %got_pcrel_hi(counter)  (R_RISCV_GOT_HI20 @0)
        encode.ld(.x10, .x10, 0), // a0 = *(GOT slot)  (lo12 paired to the auipc, emitter-patched)
        encode.lw(.x10, .x10, 0), // a0 = *counter = 42
        encode.addi(.x17, .x0, 93), // a7 = 93 (exit)
        encode.ecall(), // ecall -> exit(a0)
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "counter", .kind = .object, .defined = false, .section = .data }, // data import
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 0, .symbol = 1, .type = .got_hi20 }, // auipc counter@GOT_HI20
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// Locate a real cross glibc `ld-linux-riscv64-lp64d.so.1` (the interpreter to embed in
/// the dynexe). Returns an allocated absolute path, or null if none is found.
///
/// Ask the C compiler for the target's loader instead of scanning `/nix/store`: fast,
/// cross-platform, and a target's test runs only where its toolchain exists. Native `cc`
/// answers for the host arch; a cross target needs its `<triple>-gcc`, absent here -> the
/// query fails or echoes the bare name -> skip. `cc -print-file-name=<name>` returns an
/// ABSOLUTE path (leading `/`) when the compiler has that glibc file, else it echoes the
/// bare `<name>` (no leading `/`), which means "not available" -> skip.
fn findCrossInterp(allocator: std.mem.Allocator) !?[]u8 {
    const host = @import("builtin").cpu.arch;
    const ccs: []const []const u8 = if (host == .riscv64)
        &.{ "cc", "gcc" }
    else
        &.{ "riscv64-unknown-linux-gnu-gcc", "riscv64-linux-gnu-gcc" };
    const arg = try std.fmt.allocPrint(allocator, "-print-file-name={s}", .{"ld-linux-riscv64-lp64d.so.1"});
    defer allocator.free(arg);
    for (ccs) |ccname| {
        const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ ccname, arg } }) catch continue;
        defer allocator.free(proc.stdout);
        defer allocator.free(proc.stderr);
        const t = std.mem.trim(u8, proc.stdout, " \t\r\n");
        if (t.len > 0 and t[0] == '/') return try allocator.dupe(u8, t);
    }
    return null;
}

/// Locate the `qemu-riscv64` user-mode emulator on PATH. Returns an allocated absolute path,
/// or null if absent (so the caller skips cleanly and needs no PATH in the child env).
fn findQemu(allocator: std.mem.Allocator) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "sh", "-c", "command -v qemu-riscv64" },
    }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// One parsed ELF64 program header (the fields this test asserts on).
const Phdr = struct { p_type: u32, p_offset: u64, p_vaddr: u64 };

fn parsePhdrs(allocator: std.mem.Allocator, exe: []const u8) ![]Phdr {
    const r = std.mem.readInt;
    const e_phoff = r(u64, exe[32..40], .little);
    const e_phentsize = r(u16, exe[54..56], .little);
    const e_phnum = r(u16, exe[56..58], .little);
    var out: std.ArrayList(Phdr) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < e_phnum) : (i += 1) {
        const p = exe[@intCast(e_phoff + i * e_phentsize)..];
        try out.append(allocator, .{
            .p_type = r(u32, p[0..4], .little),
            .p_offset = r(u64, p[8..16], .little),
            .p_vaddr = r(u64, p[16..24], .little),
        });
    }
    return out.toOwnedSlice(allocator);
}

/// Read one `.dynamic` tag's payload out of the PT_DYNAMIC segment (stops at DT_NULL), or
/// null if the tag is absent.
fn readDynTag(so: []const u8, phdrs: []const Phdr, tag: i64) !?u64 {
    const r = std.mem.readInt;
    var dyn_off: ?u64 = null;
    for (phdrs) |p| if (p.p_type == 2) { // PT_DYNAMIC
        dyn_off = p.p_offset;
    };
    const off = dyn_off orelse return error.NotFound;
    var i: u64 = 0;
    while (true) : (i += 1) {
        const e = off + i * 16;
        const t = r(i64, so[@intCast(e)..][0..8], .little);
        if (t == 0) return null; // DT_NULL
        if (t == tag) return r(u64, so[@intCast(e + 8)..][0..8], .little);
    }
}

/// True iff any parsed phdr has `p_type == want`.
fn hasPhdr(phdrs: []const Phdr, want: u32) bool {
    for (phdrs) |p| if (p.p_type == want) return true;
    return false;
}

/// A `_start` `.o` with a pointer-initialized data global: a `.data` int
/// `g = 42` (offset 0, padded to 8) and a `.data` u64 pointer `p = &g` (offset 8, an 8-byte
/// slot carrying an `R_RISCV_64` data reloc against `g`). `_start` computes `&p` via a
/// `PCREL_HI20`/`PCREL_LO12_I` `auipc`/`addi` pair (a local label `.Lpcrel_hi0` at the
/// `auipc`'s own offset, exactly as `object.writeModule` synthesizes for a real
/// `global_addr`), loads the 8-byte pointer `p` holds (`ld`, = `g`'s runtime address),
/// dereferences it to read the i32 (42) via `lw`, and exits with it. In a PIE the loader
/// must apply an `R_RISCV_RELATIVE` to `p` (writing `base + g_offset`). Without it, `p` holds
/// `g`'s base-0 address and the deref reads garbage at a nonzero base.
///   auipc a0, 0        @0   (R_RISCV_PCREL_HI20 -> p)
///   addi  a0, a0, 0    @4   (R_RISCV_PCREL_LO12_I -> .Lpcrel_hi0, paired to @0) -> a0 = &p
///   ld    a0, 0(a0)    @8   -> a0 = *p = &g
///   lw    a0, 0(a0)    @12  -> a0 = *g = 42
///   addi  a7, x0, 93   @16  (SYS_exit)
///   ecall              @20  -> exit(a0)
fn buildPointerDataObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.auipc(.x10, 0), // a0 = pc + %pcrel_hi(p)  (R_RISCV_PCREL_HI20 @0)
        encode.addi(.x10, .x10, 0), // a0 = a0 + %pcrel_lo(p)  (paired lo12 @4)  -> a0 = &p
        encode.ld(.x10, .x10, 0), // a0 = *p = &g
        encode.lw(.x10, .x10, 0), // a0 = *g = 42
        encode.addi(.x17, .x0, 93), // a7 = 93 (exit)
        encode.ecall(), // ecall -> exit(a0)
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    // `.data`: g (i32 42) at 0, padding to 8, then p (u64, placeholder 0) at 8.
    var data: [16]u8 = [_]u8{0} ** 16;
    std.mem.writeInt(u32, data[0..4], 42, .little); // g = 42

    // Locals must precede globals: index 0 is the local pcrel label, then _start/g/p.
    const symbols = [_]object.Symbol{
        .{ .name = ".Lpcrel_hi0", .value = 0, .size = 0, .binding = .local, .kind = .notype, .defined = true, .section = .text },
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "g", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data },
        .{ .name = "p", .value = 8, .size = 8, .kind = .object, .defined = true, .section = .data },
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 0, .symbol = 3, .type = .pcrel_hi20 }, // auipc p@PCREL_HI20
        .{ .offset = 4, .symbol = 0, .type = .pcrel_lo12_i }, // addi p@PCREL_LO12, paired to the auipc's local label
    };
    // The pointer slot p (`.data` offset 8) holds `&g`: an `R_RISCV_64` against `g` (symbol index 2).
    const data_relocs = [_]object.DataRelocEntry{
        .{ .section = .data, .offset = 8, .symbol = 2 },
    };
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &relocs, .data_relocs = &data_relocs });
}

test "riscv64 linkDynamic dynexe imports add() from a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-riscv64. Skip cleanly if either is absent.
    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // Build `libadd.so`: a freestanding ET_DYN exporting `add(a, b) = a + b`.
    const add_obj = try buildAddObj(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{
        .mode = .shared,
        .soname = "libadd.so",
    });
    defer allocator.free(libadd_so);

    // Build the dynexe: `_start` calls the imported `add(1, 41)` then exits with the result.
    const start_obj = try buildDynStartObj(allocator);
    defer allocator.free(start_obj);
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    // Structural: ET_EXEC with an imported call (DT_JMPREL present, i.e. a `.rela.plt`).
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_JMPREL: i64 = 23;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_JMPREL)) != null);

    // Write both to a tmp dir. The exe gets the executable bit. Run it under the real cross
    // ld.so (invoked by qemu-riscv64, which reads the embedded PT_INTERP).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libadd.so", .data = libadd_so });
    try tmp.dir.writeFile(io, .{
        .sub_path = "dynexe",
        .data = dynexe,
        .flags = .{ .permissions = .executable_file },
    });

    // LD_LIBRARY_PATH = "." (relative to the child's cwd = the tmp dir) so the guest ld.so
    // finds libadd.so by soname. qemu-user passes the environment through to the guest.
    // qemu is given by absolute path and the exe as `./dynexe`, so no PATH lookup is needed.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    run_helper.runExpectExit(allocator, io, .{
        .argv = &.{ qemu, "./dynexe" },
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}

test "riscv64 linkDynamic dynexe imports DATA global counter from a .so via GOT/R_RISCV_64 and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-riscv64. Skip cleanly if either is absent.
    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // Build `libcounter.so`: a freestanding ET_DYN exporting the data global `counter = 42`.
    const counter_obj = try buildCounterObj(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{
        .mode = .shared,
        .soname = "libcounter.so",
    });
    defer allocator.free(libcounter_so);

    // Sanity: the `.so` really exports `counter` (STT_OBJECT) so the dynexe can bind to it.
    var so_exports = try ld.readSharedExports(allocator, libcounter_so);
    defer so_exports.deinit(allocator);
    var so_has_counter = false;
    for (so_exports.symbols) |s| if (std.mem.eql(u8, s, "counter")) {
        so_has_counter = true;
    };
    try std.testing.expect(so_has_counter);

    // Build the dynexe: `_start` loads `counter`'s address from the GOT, reads the i32 (42),
    // and exits with it. The `.shared` input's `counter` export satisfies the undefined GOT
    // reference. DT_NEEDED is derived from libcounter.so's soname (passed explicitly too).
    const start_obj = try buildDataStartObj(allocator);
    defer allocator.free(start_obj);
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    // Structural: ET_EXEC with a data import (a `.rela.dyn` R_RISCV_64 must exist, i.e. DT_RELA
    // present). If the data-import path is unimplemented, DT_RELA is absent -> fail.
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_RELA: i64 = 7;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_RELA)) != null);

    // Write both to a tmp dir. The exe gets the executable bit. Run it under the real cross
    // ld.so (invoked by qemu-riscv64, which reads the embedded PT_INTERP).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libcounter.so", .data = libcounter_so });

    // Structural (host readelf, skipped if absent): the dynexe carries a R_RISCV_64 in
    // `.rela.dyn` for the imported `counter` (the reloc a real `ld.so` applies to fill the GOT).
    try tmp.dir.writeFile(io, .{ .sub_path = "dynexe.ro", .data = dynexe });
    relcheck: {
        const rels = std.process.run(allocator, io, .{
            .argv = &.{ "readelf", "-W", "--use-dynamic", "-r", "dynexe.ro" },
            .cwd = .{ .dir = tmp.dir },
        }) catch |e| switch (e) {
            error.FileNotFound => break :relcheck, // readelf absent: rely on the run below.
            else => return e,
        };
        defer allocator.free(rels.stdout);
        defer allocator.free(rels.stderr);
        if (rels.term != .exited or rels.term.exited != 0) break :relcheck;
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_RISCV_64") != null);
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "counter") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "dynexe",
        .data = dynexe,
        .flags = .{ .permissions = .executable_file },
    });

    // LD_LIBRARY_PATH = "." (relative to the child's cwd = the tmp dir) so the guest ld.so
    // finds libcounter.so by soname. The cross ld.so loads libcounter.so, applies the
    // R_RISCV_64 (writes counter's address into the exe's GOT slot), and enters `_start`, which
    // reads counter (42) through the GOT and exits with it.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    run_helper.runExpectExit(allocator, io, .{
        .argv = &.{ qemu, "./dynexe" },
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}

test "object.writeModule emits a .rela.data R_RISCV_64 for a pointer-initialized data global (closes the data-section-reloc gap on riscv64, the last arch)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A module with `g` (writable int) and `p` (writable pointer = &g, carrying a DataReloc).
    var module = link.Module{};
    defer module.deinit(allocator);
    const g_bytes = [_]u8{ 42, 0, 0, 0 };
    const p_bytes = [_]u8{0} ** 8;
    try module.addWritable(allocator, "g", &g_bytes);
    try module.addWritableRelocs(allocator, "p", &p_bytes, &.{.{ .off = 0, .symbol = "g" }});

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "ptr.o", .data = obj });

    // The object must carry an `R_RISCV_64` in `.rela.data` (the pointer-init reloc that was
    // previously dropped). readelf -rW shows both the reloc type and the section name.
    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "ptr.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_RISCV_64") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, ".rela.data") != null);
}

test "object.writeModuleWithDebug ALSO emits a .rela.data R_RISCV_64 (the DWARF variant is a second reconstruct site, must not drop the data reloc)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var module = link.Module{};
    defer module.deinit(allocator);
    const g_bytes = [_]u8{ 42, 0, 0, 0 };
    const p_bytes = [_]u8{0} ** 8;
    try module.addWritable(allocator, "g", &g_bytes);
    try module.addWritableRelocs(allocator, "p", &p_bytes, &.{.{ .off = 0, .symbol = "g" }});

    const obj = try object.writeModuleWithDebug(allocator, &module, "mod.c");
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "ptr_dbg.o", .data = obj });

    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "ptr_dbg.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_RISCV_64") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, ".rela.data") != null);
}

test "riscv64 linkDynamic PIE dynexe (ET_DYN + base 0) imports add() and the REAL cross ld.so loads it at a chosen base and runs it to exit 42 under qemu (lp64d e_flags preserved)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // Build `libadd.so`: a freestanding ET_DYN exporting `add(a, b) = a + b`.
    const add_obj = try buildAddObj(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{
        .mode = .shared,
        .soname = "libadd.so",
    });
    defer allocator.free(libadd_so);

    // Build the PIE dynexe: `_start` calls the imported `add(1, 41)` then exits with the
    // result. `.pie = true` asks the emitter for an ET_DYN executable laid out at base 0
    // (every DT_*/reloc r_offset/e_entry a base-0 vaddr the real ld.so biases by its chosen
    // load address).
    const start_obj = try buildDynStartObj(allocator);
    defer allocator.free(start_obj);
    const pie = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .pie = true,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(pie);

    // ELF header: a PIE is ET_DYN (3), NOT ET_EXEC (2).
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little)); // ET_DYN

    // The lp64d float-ABI e_flags (EF_RISCV_FLOAT_ABI_DOUBLE = 0x4) must survive on the PIE -
    // the strict real riscv64 ld.so refuses a float-ABI mismatch.
    try std.testing.expectEqual(@as(u32, 4), std.mem.readInt(u32, pie[48..52], .little)); // e_flags

    // It still carries PT_INTERP + PT_DYNAMIC + an imported call (DT_JMPREL present).
    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    try std.testing.expect(hasPhdr(phdrs, 3)); // PT_INTERP
    try std.testing.expect(hasPhdr(phdrs, 2)); // PT_DYNAMIC
    const DT_JMPREL: i64 = 23;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_JMPREL)) != null);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libadd.so", .data = libadd_so });

    // Structural (host readelf, skipped if absent): `readelf -h` reports Type: DYN AND the
    // lp64d float-ABI flag for the PIE.
    try tmp.dir.writeFile(io, .{ .sub_path = "pie.ro", .data = pie });
    typecheck: {
        const hdr = std.process.run(allocator, io, .{
            .argv = &.{ "readelf", "-h", "pie.ro" },
            .cwd = .{ .dir = tmp.dir },
        }) catch |e| switch (e) {
            error.FileNotFound => break :typecheck,
            else => return e,
        };
        defer allocator.free(hdr.stdout);
        defer allocator.free(hdr.stderr);
        if (hdr.term != .exited or hdr.term.exited != 0) break :typecheck;
        try std.testing.expect(std.mem.indexOf(u8, hdr.stdout, "DYN") != null);
        try std.testing.expect(std.mem.indexOf(u8, hdr.stdout, "0x4") != null or std.mem.indexOf(u8, hdr.stdout, "double-float") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    // The kernel-substitute (qemu) reads PT_INTERP, runs the cross ld.so, which, because
    // this is an ET_DYN executable, picks a load base, biases e_entry + the GOT/PLT
    // r_offsets, loads libadd.so, applies the eager JUMP_SLOT, and enters `_start` at the
    // biased address -> `add(1, 41) = 42` -> `exit(42)`.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    run_helper.runExpectExit(allocator, io, .{
        .argv = &.{ qemu, "./pie" },
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}

test "riscv64 linkDynamic PIE with a pointer-initialized data global: the REAL cross ld.so applies R_RISCV_RELATIVE to p at a chosen base under qemu and *p reads g = exit 42" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // A single object: `_start` derefs `p` (= &g) to read g (42). g and p are both internal,
    // so there is NO .so. The only fixup is p's internal pointer, which a PIE resolves via
    // an R_RISCV_RELATIVE the loader applies at its chosen base.
    const obj = try buildPointerDataObj(allocator);
    defer allocator.free(obj);
    const pie = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .pie = true,
        .interp = interp,
        .entry = "_start",
    });
    defer allocator.free(pie);

    // A PIE is ET_DYN, and it must now carry a `.rela.dyn` (DT_RELA present) for the RELATIVE.
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little)); // ET_DYN
    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    const DT_RELA: i64 = 7;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_RELA)) != null);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Structural (host readelf, skipped if absent): the PIE carries an R_RISCV_RELATIVE.
    try tmp.dir.writeFile(io, .{ .sub_path = "pie.ro", .data = pie });
    relcheck: {
        const rels = std.process.run(allocator, io, .{
            .argv = &.{ "readelf", "-W", "--use-dynamic", "-r", "pie.ro" },
            .cwd = .{ .dir = tmp.dir },
        }) catch |e| switch (e) {
            error.FileNotFound => break :relcheck,
            else => return e,
        };
        defer allocator.free(rels.stdout);
        defer allocator.free(rels.stderr);
        if (rels.term != .exited or rels.term.exited != 0) break :relcheck;
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_RISCV_RELATIVE") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    // Run twice (two independent loader-chosen bases). Without the RELATIVE, p holds g's
    // base-0 address and the deref faults / reads garbage at a nonzero base. With it, *p = g
    // = 42 both times.
    var runs: usize = 0;
    while (runs < 2) : (runs += 1) {
        run_helper.runExpectExit(allocator, io, .{
            .argv = &.{ qemu, "./pie" },
            .cwd = .{ .dir = tmp.dir },
        }, 42) catch |e| switch (e) {
            error.FileNotFound => return error.SkipZigTest,
            else => return e,
        };
    }
}

test "riscv64 linkDynamic NON-PIE ET_EXEC with a pointer-initialized data global resolves p directly at link time (control): *p reads g = exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const obj = try buildPointerDataObj(allocator);
    defer allocator.free(obj);
    // Non-PIE (default `.pie = false`): a fixed-base ET_EXEC. The linker writes g's absolute
    // vaddr straight into p at link time. NO dynamic RELATIVE is needed.
    const exe = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .interp = interp,
        .entry = "_start",
    });
    defer allocator.free(exe);

    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, exe[16..18], .little)); // ET_EXEC

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{
        .sub_path = "exe",
        .data = exe,
        .flags = .{ .permissions = .executable_file },
    });

    run_helper.runExpectExit(allocator, io, .{
        .argv = &.{ qemu, "./exe" },
        .cwd = .{ .dir = tmp.dir },
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}
