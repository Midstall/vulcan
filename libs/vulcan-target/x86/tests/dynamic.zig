//! Dynamic-linking coverage for the shared linker (`vulcan-link`) on i386 (32-bit x86): the
//! ELFCLASS32 plus `SHT_REL` PLT/GOT plus eager JUMP_SLOT import path, proven by
//! the real cross glibc `ld-linux.so.2` under `qemu-i386`. This builds `libadd.so` (an ELF32
//! ET_DYN exporting a leaf `add(a, b) = a + b`) and a dynexe whose `_start` calls the
//! imported `add(1, 41)` (an undefined `R_386_PC32` bound to the `.so`) then does the i386
//! `exit` syscall (`int 0x80`) with the result. The kernel-substitute (qemu) reads
//! PT_INTERP, runs the cross ld.so, which loads `libadd.so`, eagerly binds the JUMP_SLOT
//! (`DF_BIND_NOW`), and enters `_start` -> `add(1, 41) = 42` -> `exit(42)`. The host here is
//! aarch64, so the only way to run i386 code is under `qemu-i386`. Both qemu and the cross
//! loader are gated on (when absent, the test skips cleanly).
//!
//! This is the ELF32/REL counterpart of `x86_64/tests/dynamic.zig`: everything is 32-bit
//! width, and the raw object uses `Elf32_Sym`, `Elf32_Rel`, and `SHT_REL` (no addend field)
//! with `EM_386`, see `writeRawObject32`.

const std = @import("std");
const ld = @import("vulcan-link");
const object = @import("../object.zig");
const link = @import("../link.zig");
const run_helper = @import("../../tests/run_helper.zig");

// i386 relocation code a data-section pointer-init reloc carries (`R_386_32`, a plain
// absolute address, the same numeric code as a `.text` `global_addr` load, distinguished by
// which section its `SHT_REL` table targets).
const R_386_32: u32 = 1;

// i386 relocation code the `_start` object's `call add` carries (`R_386_PC32`, an in-field
// addend REL, the addend is pre-written into the rel32 field, `-4` here).
const R_386_PC32: u32 = 2;

/// One symbol to emit into a raw ELF32 relocatable object's `.symtab`. `shndx` is the
/// section index the symbol is defined in (0 = undefined/import, 1 = `.text` here). `info`
/// is `(binding << 4) | type` (for example `0x12` = global func, `0x10` = global notype).
const Sym = struct { name: []const u8, value: u32 = 0, shndx: u16, info: u8, size: u32 = 0 };

/// One `SHT_REL` relocation against `.text`: `offset` is the byte offset of the patched
/// field, `sym` the `.symtab` index, `typ` the `R_386_*` code. `Elf32_Rel` has no addend
/// field. The addend lives in the relocated field itself (pre-written by `text`).
const Rel = struct { offset: u32, sym: u32, typ: u32 };

/// Emit a minimal ELF32 i386 relocatable object (`ET_REL`, `EM_386`) with a single `.text`,
/// an optional writable `.data` (when `data` is non-empty), a `.symtab`/`.strtab`, and (when
/// `rels` is non-empty) a `.rel.text`. This mirrors what `x86/object.zig`'s `assemble`
/// produces, but takes raw `.text`/`.data` bytes, symbols, and relocs directly (the low-level
/// fixtures the leaf tests need. `x86/object.zig` only exposes the IR-driven `writeModule`).
/// `Elf32_*` structures are not narrowed `Elf64_*` layouts: the `Elf32_Sym` orders
/// st_value and st_size before st_info, and relocations are `SHT_REL` (`Elf32_Rel`, 8 bytes,
/// no addend). A symbol defined in `.data` sets its `shndx` to the `.data` section index (2).
/// The caller owns the returned bytes.
fn writeRawObject32(allocator: std.mem.Allocator, text: []const u8, data: []const u8, syms: []const Sym, rels: []const Rel) ![]u8 {
    const w = std.mem.writeInt;
    const has_data = data.len > 0;
    const has_rel = rels.len > 0;

    // Section indices: NULL(0), .text(1), [.data(2)], .symtab, .strtab, [.rel.text], .shstrtab(last).
    const text_ndx: u16 = 1;
    const data_ndx: u16 = 2; // meaningful only when `has_data`
    const symtab_ndx: u16 = if (has_data) 3 else 2;
    const strtab_ndx: u16 = symtab_ndx + 1;
    const rel_ndx: u16 = strtab_ndx + 1; // meaningful only when `has_rel`
    const shstrtab_ndx: u16 = if (has_rel) rel_ndx + 1 else strtab_ndx + 1;
    const nsections: u16 = shstrtab_ndx + 1;

    // `.strtab`: the symbol name string table (index 0 is the empty string).
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

    // `.symtab`: the null symbol (index 0), then each `Sym` (Elf32_Sym, 16 bytes).
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 16);
    for (syms, 0..) |s, i| {
        var e: [16]u8 = @splat(0);
        w(u32, e[0..4], name_offs[i], .little); // st_name
        w(u32, e[4..8], s.value, .little); // st_value
        w(u32, e[8..12], s.size, .little); // st_size
        e[12] = s.info; // st_info
        e[13] = 0; // st_other
        w(u16, e[14..16], s.shndx, .little); // st_shndx
        try symtab.appendSlice(allocator, &e);
    }

    // `.rel.text` (Elf32_Rel, 8 bytes: r_offset, r_info = (sym << 8) | type).
    var rel: std.ArrayList(u8) = .empty;
    defer rel.deinit(allocator);
    for (rels) |r| {
        var e: [8]u8 = @splat(0);
        w(u32, e[0..4], r.offset, .little); // r_offset
        w(u32, e[4..8], (r.sym << 8) | r.typ, .little); // r_info
        try rel.appendSlice(allocator, &e);
    }

    // `.shstrtab`: section-name strings, in section-index order.
    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const data_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".data\x00");
    const symtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const rel_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rel.text\x00");
    const shstrtab_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    // File layout: ehdr (52), then each section's data (aligned), then the section headers (40 each).
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
    const data_off = off;
    if (has_data) off += data.len;
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
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    w(u16, buf[16..18], 1, .little); // e_type = ET_REL
    w(u16, buf[18..20], 3, .little); // e_machine = EM_386
    w(u32, buf[20..24], 1, .little); // e_version
    w(u32, buf[32..36], @intCast(sh_off), .little); // e_shoff
    w(u16, buf[40..42], 52, .little); // e_ehsize
    w(u16, buf[46..48], 40, .little); // e_shentsize
    w(u16, buf[48..50], nsections, .little); // e_shnum
    w(u16, buf[50..52], shstrtab_ndx, .little); // e_shstrndx

    @memcpy(buf[text_off..][0..text.len], text);
    if (has_data) @memcpy(buf[data_off..][0..data.len], data);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
    if (has_rel) @memcpy(buf[rel_off..][0..rel.items.len], rel.items);
    @memcpy(buf[shstr_off..][0..shstr.items.len], shstr.items);

    // Section headers (Elf32_Shdr, 40 bytes). `first_global` (symtab sh_info) = 1: index 0
    // is the null symbol (local), and every emitted `Sym` here is global.
    const putShdr = struct {
        fn f(b: []u8, at: usize, name: u32, typ: u32, flags: u32, o: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u32, entsize: u32) void {
            const ww = std.mem.writeInt;
            ww(u32, b[at + 0 ..][0..4], name, .little); // sh_name
            ww(u32, b[at + 4 ..][0..4], typ, .little); // sh_type
            ww(u32, b[at + 8 ..][0..4], flags, .little); // sh_flags
            ww(u32, b[at + 16 ..][0..4], @intCast(o), .little); // sh_offset
            ww(u32, b[at + 20 ..][0..4], @intCast(size), .little); // sh_size
            ww(u32, b[at + 24 ..][0..4], sh_link, .little); // sh_link
            ww(u32, b[at + 28 ..][0..4], sh_info, .little); // sh_info
            ww(u32, b[at + 32 ..][0..4], addralign, .little); // sh_addralign
            ww(u32, b[at + 36 ..][0..4], entsize, .little); // sh_entsize
        }
    }.f;
    const SHT_PROGBITS: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_STRTAB: u32 = 3;
    const SHT_REL: u32 = 9;
    const SHF_WRITE: u32 = 0x1;
    const SHF_ALLOC: u32 = 0x2;
    const SHF_EXECINSTR: u32 = 0x4;
    const SHF_INFO_LINK: u32 = 0x40;

    // NULL section header (index 0) stays zero.
    putShdr(buf, sh_off + text_ndx * 40, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    if (has_data) {
        putShdr(buf, sh_off + data_ndx * 40, data_name, SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, data_off, data.len, 0, 0, 4, 0);
    }
    putShdr(buf, sh_off + symtab_ndx * 40, symtab_name, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 4, 16);
    putShdr(buf, sh_off + strtab_ndx * 40, strtab_name, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    if (has_rel) {
        putShdr(buf, sh_off + rel_ndx * 40, rel_name, SHT_REL, SHF_INFO_LINK, rel_off, rel.items.len, symtab_ndx, text_ndx, 4, 8);
    }
    putShdr(buf, sh_off + shstrtab_ndx * 40, shstrtab_name, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);

    return buf;
}

/// Build `libadd.so`'s object: a leaf `add(a, b) = a + b` (i386 cdecl: `a` at `[esp+4]`,
/// `b` at `[esp+8]`, result in `eax`), `add` a global, defined function at `.text` offset 0.
///   8b 44 24 04   mov eax, [esp+4]   ; a
///   03 44 24 08   add eax, [esp+8]   ; + b
///   c3            ret
fn buildAddObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{ 0x8b, 0x44, 0x24, 0x04, 0x03, 0x44, 0x24, 0x08, 0xc3 };
    const syms = [_]Sym{.{ .name = "add", .shndx = 1, .info = 0x12 }}; // global func in .text
    return writeRawObject32(allocator, &text, &.{}, &syms, &.{});
}

/// Build the dynexe's object: `_start` calls the imported `add(1, 41)` then exits with the
/// result. `add` is undefined (an import, bound to `libadd.so`). The `call add` (E8 rel32)
/// carries an `R_386_PC32` at the rel32 field (offset 5). This is freestanding: the exit is a
/// raw i386 `int 0x80` (eax = SYS_exit = 1, ebx = code), no libc.
///   6a 29            push 41           @0   (add's second arg, cdecl pushes right-to-left)
///   6a 01            push 1            @2   (add's first arg)
///   e8 00 00 00 00   call add          @4   (rel32 @5, PC32 reloc)
///   83 c4 08         add esp, 8        @9   (pop the two args)
///   89 c3            mov ebx, eax      @12  (result -> exit code)
///   b8 01 00 00 00   mov eax, 1        @14  (SYS_exit)
///   cd 80            int 0x80          @19
fn buildDynStartObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0x6a, 0x29, // push 41
        0x6a, 0x01, // push 1
        0xe8, 0x00, 0x00, 0x00, 0x00, // call add
        0x83, 0xc4, 0x08, // add esp, 8
        0x89, 0xc3, // mov ebx, eax
        0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0xcd, 0x80, // int 0x80
    };
    const syms = [_]Sym{
        .{ .name = "_start", .shndx = 1, .info = 0x12 }, // global func in .text
        .{ .name = "add", .shndx = 0, .info = 0x10 }, // undefined global (import)
    };
    const rels = [_]Rel{
        .{ .offset = 5, .sym = 2, .typ = R_386_PC32 }, // call add -> symtab index 2
    };
    return writeRawObject32(allocator, &text, &.{}, &syms, &rels);
}

// i386 GOT-indirect data-import relocation code (`R_386_GOT32`): the `mov eax, [abs32]`
// reading `counter`'s address from its GOT slot carries this at the abs32 field.
const R_386_GOT32: u32 = 3;

/// Build `libcounter.so`'s object: a `.data` global `counter = 42` (STT_OBJECT), plus a
/// single `ret` so `.text` is non-empty (the emitter/placement expect a `.text` section).
/// `counter` is a defined, global data symbol at `.data` offset 0 (shndx = 2, the `.data`
/// section index `writeRawObject32` assigns).
///   .text:  c3               ret
///   .data:  2a 00 00 00      counter = 42
fn buildCounterObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{0xc3}; // ret
    var data: [4]u8 = undefined;
    std.mem.writeInt(u32, &data, 42, .little);
    const syms = [_]Sym{.{ .name = "counter", .shndx = 2, .info = 0x11, .size = 4 }}; // global object in .data (0x11 = STB_GLOBAL|STT_OBJECT)
    return writeRawObject32(allocator, &text, &data, &syms, &.{});
}

/// Build the dynexe's object: `_start` imports the data global `counter` through the GOT
/// (GOT-indirect addressing), loads the i32 it points at, and exits with that value.
/// `counter` is undefined (a data import, bound to `libcounter.so`). The `mov eax, [abs32]`
/// (8B 05 <abs32>) carries an `R_386_GOT32` at the abs32 field (offset 2), which the linker
/// patches to the GOT slot's absolute vaddr (the real ld.so writes counter's address there
/// through R_386_GLOB_DAT). This is freestanding: the exit is a raw i386 `int 0x80`.
///   8b 05 00 00 00 00   mov eax, [counter@GOT]   @0  (abs32 @2, GOT32 reloc) -> eax = &counter
///   8b 00               mov eax, [eax]           @6  -> eax = *counter = 42
///   89 c3               mov ebx, eax             @8  (result -> exit code)
///   b8 01 00 00 00      mov eax, 1               @10 (SYS_exit)
///   cd 80               int 0x80                 @15
fn buildDataStartObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0x8b, 0x05, 0x00, 0x00, 0x00, 0x00, // mov eax, [counter@GOT]  (abs32 @2)
        0x8b, 0x00, //                          mov eax, [eax]
        0x89, 0xc3, //                          mov ebx, eax
        0xb8, 0x01, 0x00, 0x00, 0x00, //        mov eax, 1
        0xcd, 0x80, //                          int 0x80
    };
    const syms = [_]Sym{
        .{ .name = "_start", .shndx = 1, .info = 0x12 }, // global func in .text
        .{ .name = "counter", .shndx = 0, .info = 0x11 }, // undefined global object (data import)
    };
    const rels = [_]Rel{
        .{ .offset = 2, .sym = 2, .typ = R_386_GOT32 }, // mov eax, [counter@GOT] -> symtab index 2
    };
    return writeRawObject32(allocator, &text, &.{}, &syms, &rels);
}

/// Locate a real cross glibc i386 `ld-linux.so.2` (the interpreter to embed in the
/// dynexe). Returns an allocated absolute path, or null if none is found.
///
/// Ask the C compiler for the target's loader instead of scanning `/nix/store`: fast,
/// cross-platform, and a target's test runs only where its toolchain exists. Native `cc`
/// answers for the host arch; a cross target needs its `<triple>-gcc`, absent here -> the
/// query fails or echoes the bare name -> skip. `cc -print-file-name=<name>` returns an
/// ABSOLUTE path (leading `/`) when the compiler has that glibc file, else it echoes the
/// bare `<name>` (no leading `/`), which means "not available" -> skip.
fn findCrossInterp(allocator: std.mem.Allocator) !?[]u8 {
    const host = @import("builtin").cpu.arch;
    const ccs: []const []const u8 = if (host == .x86)
        &.{ "cc", "gcc" }
    else
        &.{ "i686-unknown-linux-gnu-gcc", "i686-linux-gnu-gcc" };
    const arg = try std.fmt.allocPrint(allocator, "-print-file-name={s}", .{"ld-linux.so.2"});
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

/// Locate the `qemu-i386` user-mode emulator on PATH. Returns an allocated absolute path, or
/// null if absent (so the caller skips cleanly and needs no PATH in the child env).
fn findQemu(allocator: std.mem.Allocator) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "sh", "-c", "command -v qemu-i386" },
    }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// One parsed ELF32 program header (the fields this test asserts on).
const Phdr = struct { p_type: u32, p_offset: u32, p_vaddr: u32 };

fn parsePhdrs(allocator: std.mem.Allocator, exe: []const u8) ![]Phdr {
    const r = std.mem.readInt;
    const e_phoff = r(u32, exe[28..32], .little);
    const e_phentsize = r(u16, exe[42..44], .little);
    const e_phnum = r(u16, exe[44..46], .little);
    var out: std.ArrayList(Phdr) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < e_phnum) : (i += 1) {
        const p = exe[@intCast(e_phoff + i * e_phentsize)..];
        try out.append(allocator, .{
            .p_type = r(u32, p[0..4], .little),
            .p_offset = r(u32, p[4..8], .little), // Elf32_Phdr p_offset@4
            .p_vaddr = r(u32, p[8..12], .little), // p_vaddr@8
        });
    }
    return out.toOwnedSlice(allocator);
}

/// Read one `.dynamic` tag's payload out of the PT_DYNAMIC segment (stops at DT_NULL), or
/// null if the tag is absent. Elf32_Dyn is 8 bytes (d_tag:i32@0, d_val:u32@4).
fn readDynTag(so: []const u8, phdrs: []const Phdr, tag: i32) !?u32 {
    const r = std.mem.readInt;
    var dyn_off: ?u32 = null;
    for (phdrs) |p| if (p.p_type == 2) { // PT_DYNAMIC
        dyn_off = p.p_offset;
    };
    const off = dyn_off orelse return error.NotFound;
    var i: u32 = 0;
    while (true) : (i += 1) {
        const e = off + i * 8;
        const t = r(i32, so[@intCast(e)..][0..4], .little);
        if (t == 0) return null; // DT_NULL
        if (t == tag) return r(u32, so[@intCast(e + 4)..][0..4], .little);
    }
}

test "i386 linkDynamic dynexe imports add() from an ELF32 .so and the REAL cross ld.so runs it to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-i386. Skip cleanly if either is absent.
    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // Build `libadd.so`: a freestanding ELF32 ET_DYN exporting `add(a, b) = a + b`.
    const add_obj = try buildAddObj(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{
        .mode = .shared,
        .soname = "libadd.so",
    });
    defer allocator.free(libadd_so);

    // Structural: an ELFCLASS32 ET_DYN.
    try std.testing.expectEqual(@as(u8, 1), libadd_so[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, libadd_so[16..18], .little)); // ET_DYN

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

    // Structural: ELFCLASS32 ET_EXEC with an imported call (DT_JMPREL present, i.e. a
    // `.rel.plt`) tagged `DT_PLTREL = DT_REL` (17), not ELF64's `DT_RELA`.
    try std.testing.expectEqual(@as(u8, 1), dynexe[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_PLTREL: i32 = 20;
    const DT_JMPREL: i32 = 23;
    const DT_REL: u32 = 17;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_JMPREL)) != null);
    try std.testing.expectEqual(@as(?u32, DT_REL), try readDynTag(dynexe, phdrs, DT_PLTREL));

    // Write both to a tmp dir. The exe gets the executable bit. Run it under the real cross
    // ld.so (invoked by qemu-i386, which reads the embedded PT_INTERP).
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

test "i386 linkDynamic dynexe imports a DATA global from an ELF32 .so via GOT/GLOB_DAT and the REAL cross ld.so resolves+runs it to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-i386. Skip cleanly if either is absent.
    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // Build `libcounter.so`: a freestanding ELF32 ET_DYN exporting the data global
    // `counter = 42` (STT_OBJECT, in `.data`).
    const counter_obj = try buildCounterObj(allocator);
    defer allocator.free(counter_obj);
    const libcounter_so = try ld.linkDynamic(allocator, &.{.{ .object = counter_obj }}, .{
        .mode = .shared,
        .soname = "libcounter.so",
    });
    defer allocator.free(libcounter_so);

    // Structural: an ELFCLASS32 ET_DYN that really exports `counter`.
    try std.testing.expectEqual(@as(u8, 1), libcounter_so[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, libcounter_so[16..18], .little)); // ET_DYN
    var so_exports = try ld.readSharedExports(allocator, libcounter_so);
    defer so_exports.deinit(allocator);
    var so_has_counter = false;
    for (so_exports.symbols) |s| if (std.mem.eql(u8, s, "counter")) {
        so_has_counter = true;
    };
    try std.testing.expect(so_has_counter);

    // Build the dynexe: `_start` loads `counter`'s address from the GOT (an undefined
    // R_386_GOT32 data import bound to `libcounter.so`), reads the i32 (42), and exits with
    // it. DT_NEEDED is derived from libcounter.so's soname (passed explicitly too).
    const start_obj = try buildDataStartObj(allocator);
    defer allocator.free(start_obj);
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    // Structural: ELFCLASS32 ET_EXEC with a data import. An ELF32 `.rel.dyn` (DT_REL) with a
    // GLOB_DAT must exist. i386 dynamic uses REL (DT_REL/DT_RELSZ/DT_RELENT), not ELF64's RELA.
    // If the data-import path is unimplemented, DT_REL is absent, so the test fails.
    try std.testing.expectEqual(@as(u8, 1), dynexe[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_REL: i32 = 17;
    const DT_RELENT: i32 = 19;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_REL)) != null);
    try std.testing.expectEqual(@as(?u32, 8), try readDynTag(dynexe, phdrs, DT_RELENT)); // Elf32_Rel = 8

    // Write both to a tmp dir. The exe gets the executable bit.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libcounter.so", .data = libcounter_so });

    // Structural (host readelf, skipped if absent): the dynexe carries an R_386_GLOB_DAT in
    // `.rel.dyn` for the imported `counter` (the reloc a real ld.so applies to fill the GOT).
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
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_386_GLOB_DAT") != null);
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "counter") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "dynexe",
        .data = dynexe,
        .flags = .{ .permissions = .executable_file },
    });

    // LD_LIBRARY_PATH = "." (relative to the child's cwd = the tmp dir) so the guest ld.so
    // finds libcounter.so by soname. The cross ld.so loads it, applies the GLOB_DAT (writes
    // counter's address into the exe's GOT slot), and enters `_start`, which loads counter's
    // address from the absolute GOT slot (mov eax, [abs32]), reads counter (42), and exits 42.
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

// --- i386 PIE (ET_DYN + R_386_RELATIVE + `.rel.data` object emission) ---

/// Build a raw ELF32 object: `_start` derefs an internal pointer global `p` (= &g) and exits
/// with `g`'s value. `g` and `p` are both defined in this same object (no `.so`), so the only
/// fixup is `p`'s own internal pointer init, the exact shape a PIE resolves through
/// `R_386_RELATIVE` (or, non-PIE, a direct absolute write at link time).
///
/// `_start` cannot load `p`'s address through a plain absolute `mov reg, imm32` (`R_386_32` in
/// `.text`): `linkDynamic`'s two-pass scheme (an initial link at nominal base 0, later
/// wrapped at `base+code_off` inside `emit32`) bakes an ABS32 value that would go stale once
/// the code image is copied unmodified into the final file at a different offset. This is safe
/// only for PC-relative relocations (invariant under a uniform shift). So `_start` gets `p`'s
/// address through the classic PIC "call-next; pop" idiom plus an `R_386_PC32` reloc on an `add`
/// immediate (mirrors how a real PIC compiler computes a local address without a GOT):
///   e8 00 00 00 00   call next          @0    (rel32=0, calls the very next byte, no reloc)
/// next:
///   58               pop eax            @5    ; eax = the CPU's actual runtime addr of `next`
///   05 02 00 00 00   add eax, K         @6    ; imm32 (@7) is an R_386_PC32 against `p`, with
///                                              ; in-field addend 2 (the 2-byte gap between
///                                              ; `next` and the imm32 field) so `eax` becomes
///                                              ; p's own runtime address after this adds
///   8b 00            mov eax, [eax]     @11   ; eax = *p (p's stored value: &g, RELATIVE-fixed)
///   8b 00            mov eax, [eax]     @13   ; eax = *eax = g's value (42)
///   89 c3            mov ebx, eax       @15
///   b8 01 00 00 00   mov eax, 1         @17   (SYS_exit)
///   cd 80            int 0x80           @22
/// `.data`: g (i32 42) at offset 0, p (u32 placeholder 0) at offset 4.
fn buildPointerDataObj32(allocator: std.mem.Allocator) ![]u8 {
    const w = std.mem.writeInt;
    const text = [_]u8{
        0xe8, 0x00, 0x00, 0x00, 0x00, // call next
        0x58, // pop eax
        0x05, 0x02, 0x00, 0x00, 0x00, // add eax, 2 (PC32 -> p, in-field addend 2)
        0x8b, 0x00, // mov eax, [eax]
        0x8b, 0x00, // mov eax, [eax]
        0x89, 0xc3, // mov ebx, eax
        0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0xcd, 0x80, // int 0x80
    };
    var data: [8]u8 = [_]u8{0} ** 8;
    w(u32, data[0..4], 42, .little); // g = 42
    // p's slot (offset 4) stays 0. The linker (not the object) fills it in.

    // Section indices: NULL(0), .text(1), .data(2), .rel.text(3), .rel.data(4), .symtab(5),
    // .strtab(6), .shstrtab(7).
    const text_ndx: u16 = 1;
    const data_ndx: u16 = 2;
    const reltext_ndx: u16 = 3;
    const reldata_ndx: u16 = 4;
    const symtab_ndx: u16 = 5;
    const strtab_ndx: u16 = 6;
    const shstrtab_ndx: u16 = 7;
    const nsections: u16 = 8;

    var strtab: std.ArrayList(u8) = .empty;
    defer strtab.deinit(allocator);
    try strtab.append(allocator, 0);
    const start_name: u32 = @intCast(strtab.items.len);
    try strtab.appendSlice(allocator, "_start\x00");
    const g_name: u32 = @intCast(strtab.items.len);
    try strtab.appendSlice(allocator, "g\x00");
    const p_name: u32 = @intCast(strtab.items.len);
    try strtab.appendSlice(allocator, "p\x00");

    // `.symtab` (Elf32_Sym, 16 bytes: st_name, st_value, st_size, st_info, st_other,
    // st_shndx): null(0), _start(1, global func, .text@0), g(2, global object, .data@0),
    // p(3, global object, .data@4).
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    const appendSym = struct {
        fn f(list: *std.ArrayList(u8), a: std.mem.Allocator, name: u32, value: u32, size: u32, info: u8, shndx: u16) !void {
            var e: [16]u8 = @splat(0);
            std.mem.writeInt(u32, e[0..4], name, .little); // st_name
            std.mem.writeInt(u32, e[4..8], value, .little); // st_value
            std.mem.writeInt(u32, e[8..12], size, .little); // st_size
            e[12] = info; // st_info
            std.mem.writeInt(u16, e[14..16], shndx, .little); // st_shndx
            try list.appendSlice(a, &e);
        }
    }.f;
    try symtab.appendNTimes(allocator, 0, 16); // null symbol
    try appendSym(&symtab, allocator, start_name, 0, text.len, 0x12, text_ndx); // global func
    try appendSym(&symtab, allocator, g_name, 0, 4, 0x11, data_ndx); // global object
    try appendSym(&symtab, allocator, p_name, 4, 4, 0x11, data_ndx); // global object

    // `.rel.text`: the `add eax, K`'s imm32 (offset 7), an R_386_PC32 against `p` (symtab
    // index 3).
    var rel_text: std.ArrayList(u8) = .empty;
    defer rel_text.deinit(allocator);
    {
        var e: [8]u8 = @splat(0);
        w(u32, e[0..4], 7, .little); // r_offset
        w(u32, e[4..8], (@as(u32, 3) << 8) | R_386_PC32, .little); // r_info (sym=p(3))
        try rel_text.appendSlice(allocator, &e);
    }

    // `.rel.data`: p's slot (offset 4 in .data) holds `&g` (R_386_32 against g, symtab index
    // 2). This is the exact shape `x86/object.zig`'s `.rel.data` emission produces
    // from a `link.Module`'s `DataReloc`.
    var rel_data: std.ArrayList(u8) = .empty;
    defer rel_data.deinit(allocator);
    {
        var e: [8]u8 = @splat(0);
        w(u32, e[0..4], 4, .little); // r_offset (p's slot within .data)
        w(u32, e[4..8], (@as(u32, 2) << 8) | R_386_32, .little); // r_info (sym=g(2))
        try rel_data.appendSlice(allocator, &e);
    }

    // `.shstrtab`.
    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const data_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".data\x00");
    const reltext_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rel.text\x00");
    const reldata_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rel.data\x00");
    const symtab_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const shstrtab_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    // File layout: Elf32_Ehdr (52), then each section's data (4-aligned), then the section
    // header table (Elf32_Shdr, 40 bytes each).
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
    const data_off = off;
    off += data.len;
    off = alignUp(off, 4);
    const reltext_off = off;
    off += rel_text.items.len;
    const reldata_off = off;
    off += rel_data.items.len;
    const symtab_off = off;
    off += symtab.items.len;
    const strtab_off = off;
    off += strtab.items.len;
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
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    w(u16, buf[16..18], 1, .little); // e_type = ET_REL
    w(u16, buf[18..20], 3, .little); // e_machine = EM_386
    w(u32, buf[20..24], 1, .little); // e_version
    w(u32, buf[32..36], @intCast(sh_off), .little); // e_shoff
    w(u16, buf[40..42], 52, .little); // e_ehsize
    w(u16, buf[46..48], 40, .little); // e_shentsize
    w(u16, buf[48..50], nsections, .little); // e_shnum
    w(u16, buf[50..52], shstrtab_ndx, .little); // e_shstrndx

    @memcpy(buf[text_off..][0..text.len], &text);
    @memcpy(buf[data_off..][0..data.len], &data);
    @memcpy(buf[reltext_off..][0..rel_text.items.len], rel_text.items);
    @memcpy(buf[reldata_off..][0..rel_data.items.len], rel_data.items);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
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
    const SHF_WRITE: u32 = 0x1;
    const SHF_EXECINSTR: u32 = 0x4;
    const SHF_INFO_LINK: u32 = 0x40;

    putShdr(buf, sh_off + text_ndx * 40, text_sname, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + data_ndx * 40, data_sname, SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, data_off, data.len, 0, 0, 4, 0);
    putShdr(buf, sh_off + reltext_ndx * 40, reltext_sname, SHT_REL, SHF_INFO_LINK, reltext_off, rel_text.items.len, symtab_ndx, text_ndx, 4, 8);
    putShdr(buf, sh_off + reldata_ndx * 40, reldata_sname, SHT_REL, SHF_INFO_LINK, reldata_off, rel_data.items.len, symtab_ndx, data_ndx, 4, 8);
    putShdr(buf, sh_off + symtab_ndx * 40, symtab_sname, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 4, 16);
    putShdr(buf, sh_off + strtab_ndx * 40, strtab_sname, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    putShdr(buf, sh_off + shstrtab_ndx * 40, shstrtab_sname, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);

    return buf;
}

test "i386 linkDynamic PIE dynexe (ET_DYN + base 0) imports add() and the REAL cross ld.so loads it at a chosen base and runs it to exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const add_obj = try buildAddObj(allocator);
    defer allocator.free(add_obj);
    const libadd_so = try ld.linkDynamic(allocator, &.{.{ .object = add_obj }}, .{
        .mode = .shared,
        .soname = "libadd.so",
    });
    defer allocator.free(libadd_so);

    // Build the PIE dynexe: `_start` calls the imported `add(1, 41)` then exits with the
    // result. `.pie = true` asks `emit32` for an ET_DYN executable laid out at base 0. i386
    // code is position-independent (the PC-relative `call`, the PLT's absolute `ff 25`
    // through a GOT slot the loader fills), so a function-only PIE needs no extra
    // relocations, proving the function-import path already works through `emit32`'s
    // `opts.pie` handling. The pointer-data test below adds the new RELATIVE path.
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

    // ELF header: ELFCLASS32, and a PIE is ET_DYN (3), NOT ET_EXEC (2).
    try std.testing.expectEqual(@as(u8, 1), pie[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little));

    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    const DT_JMPREL: i32 = 23;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_JMPREL)) != null);

    // The entry is a base-0 vaddr (== its file offset), well below the non-PIE 0x400000 base.
    const e_entry = std.mem.readInt(u32, pie[24..28], .little);
    try std.testing.expect(e_entry < 0x400000);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libadd.so", .data = libadd_so });

    // Structural (host readelf, skipped if absent): `readelf -h` reports Type: DYN.
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
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

    // Run twice to confirm the ET_DYN load is stable and repeatable under the real cross ld.so.
    var runs: usize = 0;
    while (runs < 2) : (runs += 1) {
        run_helper.runExpectExit(allocator, io, .{
            .argv = &.{ qemu, "./pie" },
            .cwd = .{ .dir = tmp.dir },
            .environ_map = &env,
        }, 42) catch |e| switch (e) {
            error.FileNotFound => return error.SkipZigTest,
            else => return e,
        };
    }
}

test "i386 linkDynamic PIE with a pointer-initialized data global: the REAL cross ld.so applies R_386_RELATIVE to p at a chosen base and *p reads g = exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // A single object: `_start` derefs `p` (= &g) to read g (42). g and p are both internal,
    // so there is no `.so`. The only fixup is p's internal pointer, which a PIE resolves through
    // an R_386_RELATIVE the loader applies at its chosen (ASLR-like) base.
    const obj = try buildPointerDataObj32(allocator);
    defer allocator.free(obj);
    const pie = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .pie = true,
        .interp = interp,
        .entry = "_start",
    });
    defer allocator.free(pie);

    // A PIE is ET_DYN (3), and it must carry a `.rel.dyn` (DT_REL present) for the RELATIVE.
    try std.testing.expectEqual(@as(u8, 1), pie[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little));
    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    const DT_REL: i32 = 17;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_REL)) != null);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Structural (host readelf, skipped if absent): the PIE carries an R_386_RELATIVE.
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
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_386_RELATIVE") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    // Run twice under qemu (a fixed nonzero load base, repeated). Without the RELATIVE, p's
    // slot holds g's base-0 address (the loader never touches it), which is wrong at the
    // nonzero base, so the deref crashes or garbles. With it, the loader's in-place add makes
    // *p = g = 42 both times.
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

test "i386 linkDynamic NON-PIE ET_EXEC with a pointer-initialized data global resolves p directly at link time (control): *p reads g = exit 42 under qemu-i386" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const obj = try buildPointerDataObj32(allocator);
    defer allocator.free(obj);
    // Non-PIE (default `.pie = false`): a fixed-base ET_EXEC. The linker writes g's absolute
    // vaddr straight into p at link time. No dynamic RELATIVE is needed.
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

test "i386 object.writeModule emits a .rel.data R_386_32 for a pointer-initialized data global (closes the data-section-reloc gap)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A module with `g` (writable int) and `p` (writable pointer = &g, carrying a DataReloc).
    var module: link.Module = .{};
    defer module.deinit(allocator);
    const g_bytes = [_]u8{ 42, 0, 0, 0 };
    const p_bytes = [_]u8{0} ** 4;
    try module.addWritable(allocator, "g", &g_bytes);
    try module.addWritableRelocs(allocator, "p", &p_bytes, &.{.{ .off = 0, .symbol = "g" }});

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "ptr.o", .data = obj });

    // The object must carry an `R_386_32` in `.rel.data` (the pointer-init reloc that was
    // previously dropped). readelf -rW shows both the reloc type and the section name.
    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "ptr.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_386_32") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, ".rel.data") != null);
}
