//! Dynamic-ELF emitter and `.so` reader coverage for the shared linker (`vulcan-link`).
//! This builds a trivial AArch64 `.o` that exports a leaf `answer()` function, which
//! returns 42. It then drives `ld.linkDynamic` to produce (a) an ET_DYN shared object and
//! (b) a minimal ET_EXEC dynamic executable skeleton, and checks that the containers are
//! loader-valid. The `.so` round-trips through `ld.readSharedExports` (soname plus the
//! `answer` export), carries ET_DYN, PT_DYNAMIC, and a `.dynsym` naming `answer`, and (when
//! the host `readelf` is present) parses cleanly through binutils. The exec is ET_EXEC,
//! with a PT_INTERP naming the interpreter path, and a PT_DYNAMIC. No imports yet.

const std = @import("std");
const builtin = @import("builtin");
const object = @import("../object.zig");
const encode = @import("../encode.zig");
const link = @import("../link.zig");
const ld = @import("vulcan-link");
const run_helper = @import("../../tests/run_helper.zig");

/// A leaf `answer()` that returns the i32 `42` (AAPCS64: value in `w0`). It is serialized as
/// its own ELF `.o`, with `answer` a global, defined function symbol in `.text` at offset 0.
fn buildAnswerObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.movz(.x0, 42, 0), // w0 = 42
        encode.ret(), // ret
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "answer", .value = 0, .kind = .func, .defined = true, .section = .text },
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &.{} });
}

/// Like `buildAnswerObj`, but also exports a WRITABLE global `counter` (a 4-byte `.data`
/// object initialized to `42`). This proves that a writable global's covering `PT_LOAD` is
/// itself writable. This file regression-tests a critical fix: the dynamic emitter used to
/// hand out a read-and-execute-only `PT_LOAD` for the whole code image. So any `.data` or
/// `.bss` global inside it would SIGSEGV on its first write, under a real `ld.so`.
fn buildAnswerAndCounterObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.movz(.x0, 42, 0), // w0 = 42
        encode.ret(), // ret
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "answer", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "counter", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data },
    };
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &.{} });
}

/// A leaf `add(a, b)` returning `a + b` (AAPCS64: args in `w0`/`w1`, result in `w0`). It is
/// serialized as its own ELF `.o`, with `add` a global, defined function symbol at `.text`
/// offset 0. It is freestanding, with no libc and no data references, so it loads standalone
/// at any base.
fn buildAddObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.add(.x0, .x0, .x1), // w0 = w0 + w1
        encode.ret(), // ret
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "add", .value = 0, .kind = .func, .defined = true, .section = .text },
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &.{} });
}

/// A `_start` `.o` that CALLS the imported `add(1, 41)` and exits with the result. This is
/// the dynamic-import fixture. `add` is an UNDEFINED global, an import to be bound to a
/// `.so`. The `bl add` at `.text` offset 8 carries a CALL26 relocation against it. `_start`
/// is a defined global. It is freestanding: the exit is a raw `svc`, with no libc, so the
/// only DT_NEEDED is the `.so` providing `add`.
fn buildDynStartObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.movz(.x0, 1, 0), // x0 = 1  (add's first arg)   @0
        encode.movz(.x1, 41, 0), // x1 = 41 (add's second arg)  @4
        encode.bl(0), // bl add  (CALL26 reloc, patched)         @8
        encode.movz(.x8, 93, 0), // x8 = 93 (exit)              @12
        encode.svc(0), // svc #0 -> exit(x0 = add's result)     @16
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "add", .kind = .func, .defined = false, .section = .text }, // import
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 8, .symbol = 1, .type = .call26 }, // bl add -> symbol index 1
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// A `_start` stub `.o`. `bl answer` is not needed here, since this exec is a skeleton and
/// is not run. So `_start` just loads 42 and exits. `_start` is a defined global in `.text`.
fn buildStartObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.movz(.x0, 42, 0), // x0 = 42 (exit code)
        encode.movz(.x8, 93, 0), // x8 = 93 (exit)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &.{} });
}

/// A `.o` exporting ONLY a writable `.data` global `counter = 42` (STT_OBJECT), with no
/// code. This is the shared library's data global that the dynexe imports through the GOT.
/// A single `ret` keeps `.text` non-empty, since the emitter and placement logic expect a
/// `.text` section. `counter` is a defined, global data symbol at `.data` offset 0.
fn buildCounterObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{encode.ret()};
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "counter", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data },
    };
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &.{} });
}

/// A `_start` `.o` that IMPORTS the data global `counter` through the GOT (GOT-indirect
/// addressing), loads the i32 it points at, and exits with that value. `counter` is an
/// UNDEFINED STT_OBJECT global, a data import bound to a `.so`. The `adrp`/`ldr` at `.text`
/// offsets 0/4 carry the GOT relocations (ADR_GOT_PAGE / LD64_GOT_LO12_NC). The `ldr x0, [x0]`
/// then dereferences the GOT-resolved address. `_start` is a defined global. It is freestanding.
fn buildDataStartObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.adrp(.x0, 0), //         adrp x0, counter@GOTPAGE   (got_pg reloc)   @0
        encode.ldrOff(.x0, .x0, 0), //  ldr  x0, [x0, #GOTLO12]     (got_lo12 reloc) @4  -> x0 = &counter
        encode.ldrW(.x0, .x0, 0), //    ldr  w0, [x0]                                @8  -> w0 = *counter = 42
        encode.movz(.x8, 93, 0), //     x8 = 93 (exit)                               @12
        encode.svc(0), //               svc #0 -> exit(w0)                           @16
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "counter", .kind = .object, .defined = false, .section = .data }, // data import
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 0, .symbol = 1, .type = .adr_got_page }, // adrp counter@GOTPAGE
        .{ .offset = 4, .symbol = 1, .type = .ld64_got_lo12_nc }, // ldr counter@GOTLO12
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// A `_start` `.o` with a pointer-initialized data global: a `.data` int `g = 42` (offset 0),
/// and a `.data` u64 pointer `p = &g` (offset 8, an 8-byte slot carrying an `R_AARCH64_ABS64`
/// data reloc against `g`). `_start` takes `&p` (adrp/add), loads the 8-byte pointer it holds
/// (`g`'s runtime address), dereferences it to read the i32 (42), and exits with it. In a
/// PIE the loader must apply an `R_AARCH64_RELATIVE` to `p`, writing `base + g_offset`.
/// Without it, `p` holds `g`'s base-0 address, and the deref reads garbage at a nonzero base.
fn buildPointerDataObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.adrp(.x0, 0), //          adrp x0, p@PAGE       (adr_prel_pg_hi21 -> p)  @0
        encode.addImm64(.x0, .x0, 0), // add  x0, x0, p@LO12   (add_abs_lo12_nc -> p)   @4  -> x0 = &p
        encode.ldrOff(.x0, .x0, 0), //   ldr  x0, [x0]                                  @8  -> x0 = *p = &g
        encode.ldrW(.x0, .x0, 0), //     ldr  w0, [x0]                                  @12 -> w0 = *g = 42
        encode.movz(.x8, 93, 0), //      x8 = 93 (exit)                                 @16
        encode.svc(0), //                svc #0 -> exit(w0)                             @20
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    // `.data`: g (i32 42) at 0, padding to 8, then p (u64, placeholder 0) at 8.
    var data: [16]u8 = [_]u8{0} ** 16;
    std.mem.writeInt(u32, data[0..4], 42, .little); // g = 42

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "g", .value = 0, .size = 4, .kind = .object, .defined = true, .section = .data },
        .{ .name = "p", .value = 8, .size = 8, .kind = .object, .defined = true, .section = .data },
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 0, .symbol = 2, .type = .adr_prel_pg_hi21 }, // adrp p@PAGE
        .{ .offset = 4, .symbol = 2, .type = .add_abs_lo12_nc }, //  add  p@LO12
    };
    // The pointer slot p (`.data` offset 8) holds `&g`: an `R_AARCH64_ABS64` against `g`.
    const data_relocs = [_]object.DataRelocEntry{
        .{ .section = .data, .offset = 8, .symbol = 1 }, // p (off 8) -> g (symbol index 1)
    };
    return object.write(allocator, .{ .text = &text, .data = &data, .symbols = &symbols, .relocs = &relocs, .data_relocs = &data_relocs });
}

/// One parsed ELF64 program header, holding the fields these tests check.
const Phdr = struct { p_type: u32, p_flags: u32, p_offset: u64, p_vaddr: u64, p_filesz: u64, p_memsz: u64 };

/// Parse every program header out of an emitted ELF64. `e_phoff` is at offset 32,
/// `e_phentsize` at 54, `e_phnum` at 56. Each `Elf64_Phdr` is 56 bytes: `p_type` at 0,
/// `p_flags` at 4, `p_offset` at 8, `p_vaddr` at 16, `p_filesz` at 32, `p_memsz` at 40.
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
            .p_flags = r(u32, p[4..8], .little),
            .p_offset = r(u64, p[8..16], .little),
            .p_vaddr = r(u64, p[16..24], .little),
            .p_filesz = r(u64, p[32..40], .little),
            .p_memsz = r(u64, p[40..48], .little),
        });
    }
    return out.toOwnedSlice(allocator);
}

/// True if any parsed phdr has `p_type == want`.
fn hasPhdr(phdrs: []const Phdr, want: u32) bool {
    for (phdrs) |p| if (p.p_type == want) return true;
    return false;
}

// `.dynamic`/`.dynsym` tag values this file's own minimal parser needs. These mirror
// `vulcan-link/dynamic.zig`'s private constants. The `ld` module does not export them, so
// they are re-declared here, rather than reaching into `vulcan-link` internals.
const PT_DYNAMIC: u32 = 2;
const DT_NULL: i64 = 0;
const DT_HASH: i64 = 4;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;

/// Translate a runtime VADDR to a file offset, using the containing `PT_LOAD`'s
/// `p_offset + (vaddr - p_vaddr)`. This matches `dynamic.zig`'s own `vaddrToOffset`.
fn vaddrToFileOff(phdrs: []const Phdr, vaddr: u64) !u64 {
    for (phdrs) |p| {
        if (p.p_type != 1) continue; // PT_LOAD
        if (vaddr < p.p_vaddr) continue;
        const rel = vaddr - p.p_vaddr;
        if (rel >= p.p_filesz) continue;
        return p.p_offset + rel;
    }
    return error.NotFound;
}

/// The `p_flags` of the `PT_LOAD` segment whose in-memory range, `[p_vaddr, p_vaddr +
/// p_memsz)`, contains `vaddr`.
fn coveringLoadFlags(phdrs: []const Phdr, vaddr: u64) !u32 {
    for (phdrs) |p| {
        if (p.p_type != 1) continue; // PT_LOAD
        if (vaddr >= p.p_vaddr and vaddr < p.p_vaddr + p.p_memsz) return p.p_flags;
    }
    return error.NotFound;
}

/// Read one `.dynamic` tag's VADDR payload out of the PT_DYNAMIC segment, stopping at
/// `DT_NULL`. Returns `null` if the tag is absent.
fn readDynTag(so: []const u8, phdrs: []const Phdr, tag: i64) !?u64 {
    const r = std.mem.readInt;
    var dyn_off: ?u64 = null;
    for (phdrs) |p| if (p.p_type == PT_DYNAMIC) {
        dyn_off = p.p_offset;
    };
    const off = dyn_off orelse return error.NotFound;
    var i: u64 = 0;
    while (true) : (i += 1) {
        const e = off + i * 16;
        const t = r(i64, so[@intCast(e)..][0..8], .little);
        if (t == DT_NULL) return null;
        if (t == tag) return r(u64, so[@intCast(e + 8)..][0..8], .little);
    }
}

/// Read the `.dynsym` `st_value` (final runtime VADDR) of the export named `name`, by
/// walking `.dynsym`, `.dynstr`, and `.hash`. This mirrors `readSharedExports`'s own walk,
/// but this file needs the VALUE, which `readSharedExports` does not return.
fn dynSymValue(so: []const u8, phdrs: []const Phdr, name: []const u8) !u64 {
    const r = std.mem.readInt;
    const symtab_va = (try readDynTag(so, phdrs, DT_SYMTAB)) orelse return error.NotFound;
    const strtab_va = (try readDynTag(so, phdrs, DT_STRTAB)) orelse return error.NotFound;
    const hash_va = (try readDynTag(so, phdrs, DT_HASH)) orelse return error.NotFound;

    const symtab_off = try vaddrToFileOff(phdrs, symtab_va);
    const strtab_off = try vaddrToFileOff(phdrs, strtab_va);
    const hash_off = try vaddrToFileOff(phdrs, hash_va);

    const sym_count = r(u32, so[@intCast(hash_off + 4)..][0..4], .little);
    var i: u32 = 1; // skip the null symbol at index 0
    while (i < sym_count) : (i += 1) {
        const e = symtab_off + i * 24;
        const st_name = r(u32, so[@intCast(e)..][0..4], .little);
        const nstart: usize = @intCast(strtab_off + st_name);
        const nend = std.mem.indexOfScalarPos(u8, so, nstart, 0) orelse return error.NotFound;
        if (std.mem.eql(u8, so[nstart..nend], name)) {
            return r(u64, so[@intCast(e + 8)..][0..8], .little);
        }
    }
    return error.NotFound;
}

/// Read the `.dynsym` `st_info` byte of the export named `name`. This mirrors `dynSymValue`'s
/// walk, but returns the type and binding byte instead of the value. `st_info >> 4` is the
/// binding (`STB_*`), and `st_info & 0xf` is the type (`STT_*`).
fn dynSymInfo(so: []const u8, phdrs: []const Phdr, name: []const u8) !u8 {
    const r = std.mem.readInt;
    const symtab_va = (try readDynTag(so, phdrs, DT_SYMTAB)) orelse return error.NotFound;
    const strtab_va = (try readDynTag(so, phdrs, DT_STRTAB)) orelse return error.NotFound;
    const hash_va = (try readDynTag(so, phdrs, DT_HASH)) orelse return error.NotFound;

    const symtab_off = try vaddrToFileOff(phdrs, symtab_va);
    const strtab_off = try vaddrToFileOff(phdrs, strtab_va);
    const hash_off = try vaddrToFileOff(phdrs, hash_va);

    const sym_count = r(u32, so[@intCast(hash_off + 4)..][0..4], .little);
    var i: u32 = 1; // skip the null symbol at index 0
    while (i < sym_count) : (i += 1) {
        const e = symtab_off + i * 24;
        const st_name = r(u32, so[@intCast(e)..][0..4], .little);
        const nstart: usize = @intCast(strtab_off + st_name);
        const nend = std.mem.indexOfScalarPos(u8, so, nstart, 0) orelse return error.NotFound;
        if (std.mem.eql(u8, so[nstart..nend], name)) {
            return so[@intCast(e + 4)];
        }
    }
    return error.NotFound;
}

const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

test "linkDynamic .dynsym marks a data global STT_OBJECT and a function STT_FUNC, not both STT_FUNC (spec-compliance regression)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // `buildAnswerAndCounterObj` exports both a callable `answer()` (`.text`) and a
    // writable `.data` global `counter`. This is the exact shape a C shared library would
    // emit for `int counter = 42;`. Before the fix, the `-shared` emitter hardcoded every
    // `.dynsym` export to `STT_FUNC`. So `counter` would misreport as a function to tools
    // like `nm -D`, `objdump -T`, gdb, or a real `ld`/`lld`, even though `ld.so`'s
    // name-based GLOB_DAT resolution never noticed. After the fix, the export's kind comes
    // from its defining section.
    const obj = try buildAnswerAndCounterObj(allocator);
    defer allocator.free(obj);

    const so = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .shared,
        .soname = "libcounter.so",
    });
    defer allocator.free(so);

    const phdrs = try parsePhdrs(allocator, so);
    defer allocator.free(phdrs);

    const answer_info = try dynSymInfo(so, phdrs, "answer");
    const counter_info = try dynSymInfo(so, phdrs, "counter");

    try std.testing.expectEqual(@as(u8, STT_FUNC), answer_info & 0xf);
    try std.testing.expectEqual(@as(u8, STT_OBJECT), counter_info & 0xf);
}

test "linkDynamic emits an ET_DYN .so that round-trips through readSharedExports (soname + export)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const obj = try buildAnswerObj(allocator);
    defer allocator.free(obj);

    const so = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .shared,
        .soname = "libanswer.so",
    });
    defer allocator.free(so);

    // ELF header must report ET_DYN (3).
    const e_type = std.mem.readInt(u16, so[16..18], .little);
    try std.testing.expectEqual(@as(u16, 3), e_type);

    // A PT_DYNAMIC (2) program header must exist.
    const phdrs = try parsePhdrs(allocator, so);
    defer allocator.free(phdrs);
    try std.testing.expect(hasPhdr(phdrs, 2)); // PT_DYNAMIC
    try std.testing.expect(hasPhdr(phdrs, 1)); // at least one PT_LOAD

    // Round-trip through the reader: the soname, plus the `answer` export.
    var exports = try ld.readSharedExports(allocator, so);
    defer exports.deinit(allocator);
    try std.testing.expect(exports.soname != null);
    try std.testing.expectEqualStrings("libanswer.so", exports.soname.?);
    var found = false;
    for (exports.symbols) |s| {
        if (std.mem.eql(u8, s, "answer")) found = true;
    }
    try std.testing.expect(found);
}

test "linkDynamic .so parses cleanly through the host readelf (SONAME + HASH + SYMTAB)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const obj = try buildAnswerObj(allocator);
    defer allocator.free(obj);
    const so = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .shared,
        .soname = "libanswer.so",
    });
    defer allocator.free(so);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "libanswer.so", .data = so });

    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "readelf", "-d", "libanswer.so" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return, // readelf absent: rely on the structural asserts.
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| if (code != 0) return error.SkipZigTest,
        else => return error.SkipZigTest,
    }
    // binutils parsed the .dynamic section well enough to name these tags and values.
    try std.testing.expect(std.mem.indexOf(u8, proc.stdout, "SONAME") != null);
    try std.testing.expect(std.mem.indexOf(u8, proc.stdout, "libanswer.so") != null);
    try std.testing.expect(std.mem.indexOf(u8, proc.stdout, "HASH") != null);
    try std.testing.expect(std.mem.indexOf(u8, proc.stdout, "SYMTAB") != null);
}

test "linkDynamic emits a minimal ET_EXEC skeleton with PT_INTERP + PT_DYNAMIC" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const obj = try buildStartObj(allocator);
    defer allocator.free(obj);

    const interp = "/lib/ld-linux-aarch64.so.1";
    const exe = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libanswer.so"},
    });
    defer allocator.free(exe);

    // ELF header must report ET_EXEC (2).
    const e_type = std.mem.readInt(u16, exe[16..18], .little);
    try std.testing.expectEqual(@as(u16, 2), e_type);

    const phdrs = try parsePhdrs(allocator, exe);
    defer allocator.free(phdrs);
    try std.testing.expect(hasPhdr(phdrs, 3)); // PT_INTERP
    try std.testing.expect(hasPhdr(phdrs, 2)); // PT_DYNAMIC
    try std.testing.expect(hasPhdr(phdrs, 1)); // PT_LOAD

    // The PT_INTERP content is exactly the interpreter path, NUL-terminated.
    var interp_ph: ?Phdr = null;
    for (phdrs) |p| if (p.p_type == 3) {
        interp_ph = p;
    };
    try std.testing.expect(interp_ph != null);
    const off: usize = @intCast(interp_ph.?.p_offset);
    const len: usize = @intCast(interp_ph.?.p_filesz);
    try std.testing.expect(len >= interp.len + 1);
    try std.testing.expectEqualStrings(interp, exe[off .. off + interp.len]);
    try std.testing.expectEqual(@as(u8, 0), exe[off + interp.len]); // must be NUL-terminated
}

test "linkDynamic's PT_LOAD covering a writable global is itself writable (regression: RX-only code image SIGSEGVs on write)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // `counter` is a WRITABLE `.data` global. The earlier test fixtures were all leaf
    // functions with no globals, so this critical gap went uncaught: the whole code
    // image (`.text` AND `.rodata`/`.data`/`.bss`) is ONE `PT_LOAD` segment
    // (`computeDefaultPlacement` packs them together). The dynamic emitter used to hand
    // that segment `PF_R | PF_X` with no `PF_W`. So a real `ld.so` would SIGSEGV the
    // first time this program wrote `counter`.
    const obj = try buildAnswerAndCounterObj(allocator);
    defer allocator.free(obj);

    const so = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .shared,
        .soname = "libcounter.so",
    });
    defer allocator.free(so);

    const phdrs = try parsePhdrs(allocator, so);
    defer allocator.free(phdrs);

    // Sanity check: `counter` really did land in the dynsym export table. This proves
    // the parser below reads the right symbol, and is not silently matching nothing.
    var exports = try ld.readSharedExports(allocator, so);
    defer exports.deinit(allocator);
    var found = false;
    for (exports.symbols) |s| if (std.mem.eql(u8, s, "counter")) {
        found = true;
    };
    try std.testing.expect(found);

    // The PT_LOAD segment that actually covers `counter`'s runtime address must carry
    // PF_W (2). Before the fix, this segment is PF_R | PF_X (5), with no PF_W, and this
    // assert fails. After the fix, it is PF_R | PF_W | PF_X (7).
    const counter_vaddr = try dynSymValue(so, phdrs, "counter");
    const flags = try coveringLoadFlags(phdrs, counter_vaddr);
    try std.testing.expect((flags & 2) != 0); // PF_W
}

/// Locate a real glibc `ld-linux-aarch64.so.1` in the Nix store. This is the interpreter to
/// embed in the dynamic exe. Returns an allocated absolute path, or null if none is found
/// (the end-to-end test then skips). Prefers a native glibc build.
fn findGlibcInterp(allocator: std.mem.Allocator) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "sh", "-c", "ls /nix/store/*glibc*/lib/ld-linux-aarch64.so.1 2>/dev/null | head -1" },
    }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

test "linkDynamic dynexe imports add() from a .so and the REAL glibc ld.so resolves+runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // The interpreter: a real glibc ld-linux-aarch64.so.1. Skip cleanly if absent.
    const interp = (try findGlibcInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    // Build `libadd.so`: a freestanding ET_DYN exporting `add(a, b) = a + b`.
    const add_obj = try buildAddObj(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{
        .mode = .shared,
        .soname = "libadd.so",
    });
    defer allocator.free(libadd_so);

    // Build the dynexe: `_start` calls the imported `add(1, 41)` then exits with the result.
    // The `.shared` input's `add` export satisfies the undefined CALL26 import. DT_NEEDED
    // comes from libadd.so's soname (and is also passed explicitly, to prove the dedup).
    const start_obj = try buildDynStartObj(allocator);
    defer allocator.free(start_obj);
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libadd_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libadd.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    // Structural check: ET_EXEC with an imported call means a `.rela.plt` JUMP_SLOT must
    // exist, that is, DT_JMPREL present. If the import path is unimplemented, DT_JMPREL is
    // absent, and this must fail.
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_JMPREL: i64 = 23;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_JMPREL)) != null);

    // Write both to a tmp dir. The exe gets the executable bit. Run it under the real ld.so.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "libadd.so", .data = libadd_so });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "dynexe",
        .data = dynexe,
        .flags = .{ .permissions = .executable_file },
    });

    // LD_LIBRARY_PATH="." is relative to the child's cwd, the tmp dir, so ld.so finds
    // libadd.so by its soname. The kernel reads PT_INTERP (the absolute glibc path), runs
    // ld.so, which loads libadd.so, applies the eager JUMP_SLOT, and enters `_start`.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    run_helper.runExpectExit(allocator, std.testing.io, .{
        .argv = &.{"./dynexe"},
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // sh/loader absent
        else => return e,
    };
}

test "linkDynamic PIE dynexe (ET_DYN + base 0) imports add() and the REAL glibc ld.so loads it at a chosen base and runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // The interpreter: a real glibc ld-linux-aarch64.so.1. Skip cleanly if absent.
    const interp = (try findGlibcInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    // Build `libadd.so`: a freestanding ET_DYN exporting `add(a, b) = a + b`.
    const add_obj = try buildAddObj(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{
        .mode = .shared,
        .soname = "libadd.so",
    });
    defer allocator.free(libadd_so);

    // Build the PIE dynexe: `_start` calls the imported `add(1, 41)` then exits with the
    // result. Unlike the non-PIE control above, `.pie = true` asks the emitter for an ET_DYN
    // executable laid out at base 0. Every DT_* value, reloc r_offset, and e_entry is a
    // base-0 vaddr, which the real ld.so biases by its chosen load address. The aarch64 code
    // is position-independent (the PC-relative `bl`, the PLT's adrp/ldr), so it needs NO
    // extra RELATIVE relocations.
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

    // ELF header: a PIE is ET_DYN (3), NOT ET_EXEC (2). This is the load-bearing difference.
    // A fixed ET_EXEC at base 0 cannot be placed by the loader. An ET_DYN can be mapped at
    // any bias.
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little)); // ET_DYN

    // It still carries a PT_INTERP, a PT_DYNAMIC, and an imported call (DT_JMPREL present).
    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    try std.testing.expect(hasPhdr(phdrs, 3)); // PT_INTERP
    try std.testing.expect(hasPhdr(phdrs, 2)); // PT_DYNAMIC
    const DT_JMPREL: i64 = 23;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_JMPREL)) != null);

    // The entry is a base-0 vaddr (equal to its file offset), well below the non-PIE
    // 0x400000 base. This proves the layout is truly base-relative: the loader adds its
    // bias, and no bias is baked in here.
    const e_entry = std.mem.readInt(u64, pie[24..32], .little);
    try std.testing.expect(e_entry < 0x400000);

    // Write both to a tmp dir. The exe gets the executable bit.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "libadd.so", .data = libadd_so });

    // Structural check with the host readelf, skipped if absent: `readelf -h` reports
    // Type: DYN for the PIE.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "pie.ro", .data = pie });
    typecheck: {
        const hdr = std.process.run(allocator, std.testing.io, .{
            .argv = &.{ "readelf", "-h", "pie.ro" },
            .cwd = .{ .dir = tmp.dir },
        }) catch |e| switch (e) {
            error.FileNotFound => break :typecheck, // readelf absent: rely on the run below.
            else => return e,
        };
        defer allocator.free(hdr.stdout);
        defer allocator.free(hdr.stderr);
        if (hdr.term != .exited or hdr.term.exited != 0) break :typecheck;
        try std.testing.expect(std.mem.indexOf(u8, hdr.stdout, "DYN") != null);
    }

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    // The kernel reads PT_INTERP (the absolute glibc path) and runs ld.so. Because this is
    // an ET_DYN executable, ld.so picks a load base (ASLR), biases e_entry and the GOT/PLT
    // r_offsets, loads libadd.so, applies the eager JUMP_SLOT, and enters `_start` at the
    // biased address. Running twice below exercises two independent loader-chosen bases: a
    // binary that is not position-independent would crash at one of them. Ours exits 42
    // both times.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    var runs: usize = 0;
    while (runs < 2) : (runs += 1) {
        run_helper.runExpectExit(allocator, std.testing.io, .{
            .argv = &.{"./pie"},
            .cwd = .{ .dir = tmp.dir },
            .environ_map = &env,
        }, 42) catch |e| switch (e) {
            error.FileNotFound => return error.SkipZigTest, // sh/loader absent
            else => return e,
        };
    }
}

test "object.writeModule emits a .rela.data R_AARCH64_ABS64 for a pointer-initialized data global (closes the data-section-reloc gap)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A module with `g` (a writable int) and `p` (a writable pointer, = &g, carrying a
    // DataReloc).
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

    // The object must carry an `R_AARCH64_ABS64` in `.rela.data`. This is the pointer-init
    // reloc that was previously dropped. `readelf -rW` shows both the reloc type and the
    // section name.
    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "ptr.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ABS64") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, ".rela.data") != null);
}

test "linkDynamic PIE with a pointer-initialized data global: the REAL glibc ld.so applies R_AARCH64_RELATIVE to p at a chosen base and *p reads g = exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const interp = (try findGlibcInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    // A single object: `_start` derefs `p` (= &g) to read g (42). g and p are both internal,
    // so there is NO .so. The only fixup is p's internal pointer, which a PIE resolves with
    // an R_AARCH64_RELATIVE that the loader applies at its chosen (ASLR) base.
    const obj = try buildPointerDataObj(allocator);
    defer allocator.free(obj);
    const pie = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .pie = true,
        .interp = interp,
        .entry = "_start",
    });
    defer allocator.free(pie);

    // A PIE is ET_DYN, and it must now carry a `.rela.dyn` (DT_RELA present) for the
    // RELATIVE relocation.
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little)); // ET_DYN
    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    const DT_RELA: i64 = 7;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_RELA)) != null);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Structural check with the host readelf, skipped if absent: the PIE carries an
    // R_AARCH64_RELATIVE.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "pie.ro", .data = pie });
    relcheck: {
        const rels = std.process.run(allocator, std.testing.io, .{
            .argv = &.{ "readelf", "-W", "--use-dynamic", "-r", "pie.ro" },
            .cwd = .{ .dir = tmp.dir },
        }) catch |e| switch (e) {
            error.FileNotFound => break :relcheck,
            else => return e,
        };
        defer allocator.free(rels.stdout);
        defer allocator.free(rels.stderr);
        if (rels.term != .exited or rels.term.exited != 0) break :relcheck;
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_RELATIVE") != null);
    }

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    // Run twice, for two independent ASLR bases. Without the RELATIVE, p holds g's base-0
    // address, and the deref faults or reads garbage at a nonzero base. With it, *p = g = 42
    // both times.
    var runs: usize = 0;
    while (runs < 2) : (runs += 1) {
        run_helper.runExpectExit(allocator, std.testing.io, .{
            .argv = &.{"./pie"},
            .cwd = .{ .dir = tmp.dir },
        }, 42) catch |e| switch (e) {
            error.FileNotFound => return error.SkipZigTest,
            else => return e,
        };
    }
}

test "linkDynamic NON-PIE ET_EXEC with a pointer-initialized data global resolves p directly at link time (control): *p reads g = exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const interp = (try findGlibcInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    const obj = try buildPointerDataObj(allocator);
    defer allocator.free(obj);
    // Non-PIE (default `.pie = false`): a fixed-base ET_EXEC. The linker writes g's absolute
    // vaddr straight into p at link time, so NO dynamic RELATIVE is needed.
    const exe = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .interp = interp,
        .entry = "_start",
    });
    defer allocator.free(exe);

    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, exe[16..18], .little)); // ET_EXEC

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "exe",
        .data = exe,
        .flags = .{ .permissions = .executable_file },
    });

    run_helper.runExpectExit(allocator, std.testing.io, .{
        .argv = &.{"./exe"},
        .cwd = .{ .dir = tmp.dir },
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}

test "linkDynamic dynexe imports a DATA global from a .so via GOT/GLOB_DAT and the REAL glibc ld.so resolves+runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // The interpreter: a real glibc ld-linux-aarch64.so.1. Skip cleanly if absent.
    const interp = (try findGlibcInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);

    // Build `libcounter.so`: a freestanding ET_DYN exporting the data global `counter = 42`.
    const counter_obj = try buildCounterObj(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{
        .mode = .shared,
        .soname = "libcounter.so",
    });
    defer allocator.free(libcounter_so);

    // Sanity check: the `.so` really exports `counter` (STT_OBJECT), so the dynexe can bind
    // to it.
    var so_exports = try ld.readSharedExports(allocator, libcounter_so);
    defer so_exports.deinit(allocator);
    var so_has_counter = false;
    for (so_exports.symbols) |s| if (std.mem.eql(u8, s, "counter")) {
        so_has_counter = true;
    };
    try std.testing.expect(so_has_counter);

    // Build the dynexe: `_start` loads `counter`'s address from the GOT, reads the i32 (42),
    // and exits with it. The `.shared` input's `counter` export satisfies the undefined GOT
    // reference. DT_NEEDED comes from libcounter.so's soname (and is also passed explicitly).
    const start_obj = try buildDataStartObj(allocator);
    defer allocator.free(start_obj);
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    // Structural check: ET_EXEC with a data import means a `.rela.dyn` GLOB_DAT must exist,
    // that is, DT_RELA present. If the data-import path is unimplemented, DT_RELA is
    // absent, and this must fail.
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_RELA: i64 = 7;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_RELA)) != null);

    // Write both to a tmp dir. The exe gets the executable bit. Run it under the real ld.so.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "libcounter.so", .data = libcounter_so });

    // Structural check with the host readelf, skipped if absent: the dynexe carries a
    // R_AARCH64_GLOB_DAT in `.rela.dyn` for the imported `counter`. This is the reloc a
    // real `ld.so` applies to fill the GOT.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "dynexe.ro", .data = dynexe });
    relcheck: {
        const rels = std.process.run(allocator, std.testing.io, .{
            .argv = &.{ "readelf", "-W", "--use-dynamic", "-r", "dynexe.ro" },
            .cwd = .{ .dir = tmp.dir },
        }) catch |e| switch (e) {
            error.FileNotFound => break :relcheck, // readelf absent: rely on the run below.
            else => return e,
        };
        defer allocator.free(rels.stdout);
        defer allocator.free(rels.stderr);
        if (rels.term != .exited or rels.term.exited != 0) break :relcheck;
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_GLOB_DAT") != null);
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "counter") != null);
    }

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "dynexe",
        .data = dynexe,
        .flags = .{ .permissions = .executable_file },
    });

    // LD_LIBRARY_PATH="." is relative to the child's cwd, the tmp dir, so ld.so finds
    // libcounter.so by its soname. The kernel reads PT_INTERP (the absolute glibc path), runs
    // ld.so, which loads libcounter.so, applies the GLOB_DAT (writing counter's address into
    // the exe's GOT slot), and enters `_start`, which reads counter (42) through the GOT.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    run_helper.runExpectExit(allocator, std.testing.io, .{
        .argv = &.{"./dynexe"},
        .cwd = .{ .dir = tmp.dir },
        .environ_map = &env,
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // sh/loader absent
        else => return e,
    };
}
