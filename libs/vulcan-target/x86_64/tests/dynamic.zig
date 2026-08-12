//! Dynamic-linking coverage for the shared linker (`vulcan-link`) on x86-64. This test
//! proves the PLT/GOT and eager JUMP_SLOT import path with the REAL cross glibc
//! `ld-linux-x86-64.so.2` under `qemu-x86_64`. The test builds `libadd.so` (an ET_DYN that
//! exports a leaf `add(a, b) = a + b`) and a dynexe whose `_start` CALLS the imported
//! `add(1, 41)` (an undefined `R_X86_64_PLT32` bound to the `.so`), then runs the x86-64
//! `exit` syscall with the result. The kernel-substitute (qemu) reads PT_INTERP and runs
//! the cross ld.so. The cross ld.so loads `libadd.so`, eagerly binds the JUMP_SLOT
//! (`DF_BIND_NOW`), and enters `_start` -> `add(1, 41) = 42` -> `exit(42)`. The host here
//! is aarch64, so the only way to run x86-64 code is under `qemu-x86_64`. Both qemu and the
//! cross loader are gated: if either is absent, the test skips cleanly.

const std = @import("std");
const ld = @import("vulcan-link");
const ir = @import("vulcan-ir");
const object = @import("../object.zig");
const object_emit = @import("../../object_emit.zig");
const link = @import("../link.zig");
const run_helper = @import("../../tests/run_helper.zig");

// x86-64 relocation code the `_start` object's `call add` carries.
const R_X86_64_PLT32: u32 = 4;
// x86-64 relocation code a GOT-indirect `mov rd, [rip+disp32]` data reference carries
// (R_X86_64_GOTPCREL). The linker resolves it into a `.got` slot plus `R_X86_64_GLOB_DAT`.
const R_X86_64_GOTPCREL: u32 = 9;
// x86-64 relocation code a RIP-relative `lea rd, [rip+disp32]` (a `global_addr`) carries.
const R_X86_64_PC32: u32 = 2;
// x86-64 relocation code a `.data`/`.rodata` pointer-init slot carries: a 64-bit absolute
// address (`S + A`) written into the slot. The linker turns an internal-target address into
// `R_X86_64_RELATIVE` (for a PIE or `.so`), or writes it directly (for a non-PIE exec).
const R_X86_64_64: u32 = 1;

/// One symbol to emit into a raw ELF64 relocatable object's `.symtab`. `shndx` is the
/// section index that defines the symbol (0 = undefined or import, 1 = `.text` here).
/// `info` is `(binding << 4) | type` (for example, `0x12` = global func, `0x10` = global notype).
const Sym = struct { name: []const u8, value: u64 = 0, shndx: u16, info: u8, size: u64 = 0 };

/// One RELA relocation against `.text`. `offset` is the byte offset of the patched field.
/// `sym` is the `.symtab` index. `typ` is the `R_X86_64_*` code. `addend` is the RELA addend.
const Rel = struct { offset: u64, sym: u32, typ: u32, addend: i64 };

/// Emit a minimal ELF64 x86-64 relocatable object (`ET_REL`, `EM_X86_64`) with a single
/// `.text` section plus a `.symtab`/`.strtab`, and a `.rela.text` when `rels` is not empty.
/// This mirrors what `x86_64/object.zig`'s `assemble` produces, but it takes raw `.text`
/// bytes, symbols, and relocs directly. The aarch64 dynamic test gets this low-level path
/// from its `object.write`. x86-64's `object.zig` only exposes the IR-driven `writeModule`,
/// so the leaf fixtures here hand-assemble the object. The caller owns the returned bytes.
fn writeRawObject(allocator: std.mem.Allocator, text: []const u8, syms: []const Sym, rels: []const Rel) ![]u8 {
    const w = std.mem.writeInt;
    const has_rela = rels.len > 0;

    // Section indices: NULL(0), .text(1), .symtab(2), .strtab(3), [.rela.text(4)], .shstrtab(last).
    const text_ndx: u16 = 1;
    const symtab_ndx: u16 = 2;
    const strtab_ndx: u16 = 3;
    const rela_ndx: u16 = 4;
    const shstrtab_ndx: u16 = if (has_rela) 5 else 4;
    const nsections: u16 = if (has_rela) 6 else 5;

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

    // `.symtab`: the null symbol (index 0), then each `Sym`.
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24);
    for (syms, 0..) |s, i| {
        var e: [24]u8 = @splat(0);
        w(u32, e[0..4], name_offs[i], .little); // st_name
        e[4] = s.info; // st_info
        e[5] = 0; // st_other
        w(u16, e[6..8], s.shndx, .little); // st_shndx
        w(u64, e[8..16], s.value, .little); // st_value
        w(u64, e[16..24], s.size, .little); // st_size
        try symtab.appendSlice(allocator, &e);
    }

    // `.rela.text`.
    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (rels) |r| {
        var e: [24]u8 = @splat(0);
        w(u64, e[0..8], r.offset, .little); // r_offset
        w(u64, e[8..16], (@as(u64, r.sym) << 32) | r.typ, .little); // r_info
        w(i64, e[16..24], r.addend, .little); // r_addend
        try rela.appendSlice(allocator, &e);
    }

    // `.shstrtab`: section-name strings, in section-index order.
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

    // File layout: ehdr, then each section's data (8-aligned), then the section headers.
    var off: usize = 64;
    const alignUp = struct {
        fn f(v: usize, a: usize) usize {
            return std.mem.alignForward(usize, v, a);
        }
    }.f;
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
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    w(u16, buf[16..18], 1, .little); // e_type = ET_REL
    w(u16, buf[18..20], 62, .little); // e_machine = EM_X86_64
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[40..48], sh_off, .little); // e_shoff
    w(u16, buf[52..54], 64, .little); // e_ehsize
    w(u16, buf[58..60], 64, .little); // e_shentsize
    w(u16, buf[60..62], nsections, .little); // e_shnum
    w(u16, buf[62..64], shstrtab_ndx, .little); // e_shstrndx

    @memcpy(buf[text_off..][0..text.len], text);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
    if (has_rela) @memcpy(buf[rela_off..][0..rela.items.len], rela.items);
    @memcpy(buf[shstr_off..][0..shstr.items.len], shstr.items);

    // Section headers. `first_global` (symtab sh_info) = 1: index 0 is the null symbol
    // (local), and every emitted `Sym` here is global.
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

    // NULL section header (index 0) stays zero.
    putShdr(buf, sh_off + text_ndx * 64, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + symtab_ndx * 64, symtab_name, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 8, 24);
    putShdr(buf, sh_off + strtab_ndx * 64, strtab_name, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    if (has_rela) {
        putShdr(buf, sh_off + rela_ndx * 64, rela_name, SHT_RELA, SHF_INFO_LINK, rela_off, rela.items.len, symtab_ndx, text_ndx, 8, 24);
    }
    putShdr(buf, sh_off + shstrtab_ndx * 64, shstrtab_name, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);

    return buf;
}

/// Build `libadd.so`'s object: a leaf `add(a, b) = a + b` (SysV ABI: `a` in `edi`, `b` in
/// `esi`, result in `eax`). `add` is a global, defined function at `.text` offset 0.
///   89 f8   mov eax, edi
///   01 f0   add eax, esi
///   c3      ret
fn buildAddObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{ 0x89, 0xf8, 0x01, 0xf0, 0xc3 };
    const syms = [_]Sym{.{ .name = "add", .shndx = 1, .info = 0x12 }}; // global func in .text
    return writeRawObject(allocator, &text, &syms, &.{});
}

/// Build the dynexe's object: `_start` calls the imported `add(1, 41)` then exits with the
/// result. `add` is UNDEFINED (an import bound to `libadd.so`). The `call add` (E8 rel32)
/// carries an `R_X86_64_PLT32` at the rel32 field (offset 11). This code is freestanding:
/// the exit uses a raw `syscall`, not libc.
///   bf 01 00 00 00   mov edi, 1        @0   (add's first arg)
///   be 29 00 00 00   mov esi, 41       @5   (add's second arg)
///   e8 00 00 00 00   call add          @10  (rel32 @11, PLT32 reloc)
///   89 c7            mov edi, eax      @15  (result -> exit code)
///   b8 3c 00 00 00   mov eax, 60       @17  (SYS_exit)
///   0f 05            syscall           @22
fn buildDynStartObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0xbf, 0x01, 0x00, 0x00, 0x00, // mov edi, 1
        0xbe, 0x29, 0x00, 0x00, 0x00, // mov esi, 41
        0xe8, 0x00, 0x00, 0x00, 0x00, // call add
        0x89, 0xc7, // mov edi, eax
        0xb8, 0x3c, 0x00, 0x00, 0x00, // mov eax, 60
        0x0f, 0x05, // syscall
    };
    const syms = [_]Sym{
        .{ .name = "_start", .shndx = 1, .info = 0x12 }, // global func in .text
        .{ .name = "add", .shndx = 0, .info = 0x10 }, // undefined global (import)
    };
    const rels = [_]Rel{
        .{ .offset = 11, .sym = 2, .typ = R_X86_64_PLT32, .addend = -4 }, // call add -> symtab index 2
    };
    return writeRawObject(allocator, &text, &syms, &rels);
}

/// Build `libcounter.so`'s object through the IR-driven `link.Module` data API: an EXPORTED
/// writable `.data` global `counter = 42` (STT_OBJECT), plus a never-called leaf function so
/// `.text` stays non-empty (the emitter and placement logic expect a `.text` section). This
/// is the shared library's data global that the dynexe imports through the GOT.
/// `object.writeModule` lays `counter` into `.data` as a defined, global STT_OBJECT symbol.
fn buildCounterObj(allocator: std.mem.Allocator) ![]u8 {
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

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "_lib_dummy", &dummy);
    try module.addWritable(allocator, "counter", &data); // .data global `counter = 42`
    return object.writeModule(allocator, &module);
}

/// Build the dynexe's object: `_start` IMPORTS the data global `counter` through the GOT
/// (GOT-indirect addressing), loads the i32 it points at (42), and exits with it. `counter`
/// is an UNDEFINED STT_OBJECT global, a data import bound to `libcounter.so`. The
/// `mov rax, [rip+disp32]` at `.text` offset 0 carries an `R_X86_64_GOTPCREL` at its disp32
/// field (offset 3). The linker resolves this into a `.got` slot that the real `ld.so` fills
/// through `R_X86_64_GLOB_DAT`. The `mov eax, [rax]` then reads the value at the
/// GOT-resolved address. This code is freestanding: the exit uses a raw `syscall`, not libc.
///   48 8b 05 00 00 00 00   mov rax, [rip+0]   @0   (disp32 @3, GOTPCREL reloc)  -> rax = &counter
///   8b 00                  mov eax, [rax]     @7   -> eax = *counter = 42
///   89 c7                  mov edi, eax       @9   (result -> exit code)
///   b8 3c 00 00 00         mov eax, 60        @11  (SYS_exit)
///   0f 05                  syscall            @16
fn buildDataStartObj(allocator: std.mem.Allocator) ![]u8 {
    const text = [_]u8{
        0x48, 0x8b, 0x05, 0x00, 0x00, 0x00, 0x00, // mov rax, [rip+0]  (GOTPCREL)
        0x8b, 0x00, // mov eax, [rax]
        0x89, 0xc7, // mov edi, eax
        0xb8, 0x3c, 0x00, 0x00, 0x00, // mov eax, 60
        0x0f, 0x05, // syscall
    };
    const syms = [_]Sym{
        .{ .name = "_start", .shndx = 1, .info = 0x12 }, // global func in .text
        .{ .name = "counter", .shndx = 0, .info = 0x11 }, // undefined global OBJECT (data import)
    };
    const rels = [_]Rel{
        .{ .offset = 3, .sym = 2, .typ = R_X86_64_GOTPCREL, .addend = -4 }, // mov counter@GOTPCREL
    };
    return writeRawObject(allocator, &text, &syms, &rels);
}

/// A `_start` fixture with a pointer-initialized data global: a `.data` int `g = 42`
/// (offset 0) and a `.data` u64 pointer `p = &g` (offset 8, an 8-byte slot carrying an
/// `R_X86_64_64` data reloc against `g`). `_start` computes `&p` through a RIP-relative
/// `lea` (an `R_X86_64_PC32` reloc against `p`). It dereferences the result twice: first to
/// read the pointer value (`g`'s runtime address), then to read the i32 at that address. It
/// then exits with the result. In a PIE, the loader must apply an `R_X86_64_RELATIVE` to `p`
/// (writing `base + g_offset`). Without it, `p` holds `g`'s base-0 address, and the deref
/// reads garbage or faults at a nonzero base. This is hand-assembled raw ELF64: unlike
/// `writeRawObject` (which only carries `.text`/`.rela.text`), this fixture also needs a
/// `.data` section and a `.rela.data` section (the exact shape that `x86_64/object.zig`'s
/// `.rela.data` emission produces from a `link.Module`'s `DataReloc`). So it is assembled
/// standalone rather than through the shared helper.
///   48 8d 05 00 00 00 00   lea rax, [rip+0]   @0   (disp32 @3, PC32 reloc -> p)  -> rax = &p
///   48 8b 00               mov rax, [rax]     @7   -> rax = *p = &g
///   8b 00                  mov eax, [rax]     @10  -> eax = *g = 42
///   89 c7                  mov edi, eax       @12  (result -> exit code)
///   b8 3c 00 00 00         mov eax, 60        @14  (SYS_exit)
///   0f 05                  syscall            @19
fn buildPointerDataObj(allocator: std.mem.Allocator) ![]u8 {
    const w = std.mem.writeInt;
    const text = [_]u8{
        0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00, // lea rax, [rip+0]  (PC32 -> p)
        0x48, 0x8b, 0x00, // mov rax, [rax]
        0x8b, 0x00, // mov eax, [rax]
        0x89, 0xc7, // mov edi, eax
        0xb8, 0x3c, 0x00, 0x00, 0x00, // mov eax, 60
        0x0f, 0x05, // syscall
    };
    // `.data`: g (i32 42) at 0, padding to 8, then p (u64, placeholder 0) at 8.
    var data: [16]u8 = [_]u8{0} ** 16;
    w(u32, data[0..4], 42, .little);

    // Section indices: NULL(0), .text(1), .data(2), .rela.text(3), .rela.data(4),
    // .symtab(5), .strtab(6), .shstrtab(7).
    const text_ndx: u16 = 1;
    const data_ndx: u16 = 2;
    const rela_text_ndx: u16 = 3;
    const rela_data_ndx: u16 = 4;
    const symtab_ndx: u16 = 5;
    const strtab_ndx: u16 = 6;
    const shstrtab_ndx: u16 = 7;
    const nsections: u16 = 8;

    // `.strtab`: index 0 is the empty string.
    var strtab: std.ArrayList(u8) = .empty;
    defer strtab.deinit(allocator);
    try strtab.append(allocator, 0);
    const start_name: u32 = @intCast(strtab.items.len);
    try strtab.appendSlice(allocator, "_start\x00");
    const g_name: u32 = @intCast(strtab.items.len);
    try strtab.appendSlice(allocator, "g\x00");
    const p_name: u32 = @intCast(strtab.items.len);
    try strtab.appendSlice(allocator, "p\x00");

    // `.symtab`: null(0), _start(1, global func, .text), g(2, global object, .data@0),
    // p(3, global object, .data@8).
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    const appendSym = struct {
        fn f(list: *std.ArrayList(u8), a: std.mem.Allocator, name: u32, info: u8, shndx: u16, value: u64, size: u64) !void {
            var e: [24]u8 = @splat(0);
            std.mem.writeInt(u32, e[0..4], name, .little);
            e[4] = info;
            std.mem.writeInt(u16, e[6..8], shndx, .little);
            std.mem.writeInt(u64, e[8..16], value, .little);
            std.mem.writeInt(u64, e[16..24], size, .little);
            try list.appendSlice(a, &e);
        }
    }.f;
    try symtab.appendNTimes(allocator, 0, 24); // null symbol
    try appendSym(&symtab, allocator, start_name, 0x12, text_ndx, 0, text.len); // global func
    try appendSym(&symtab, allocator, g_name, 0x11, data_ndx, 0, 4); // global object
    try appendSym(&symtab, allocator, p_name, 0x11, data_ndx, 8, 8); // global object

    // `.rela.text`: the `lea`'s disp32 (PC32 against p, symtab index 3).
    var rela_text: std.ArrayList(u8) = .empty;
    defer rela_text.deinit(allocator);
    {
        var e: [24]u8 = @splat(0);
        w(u64, e[0..8], 3, .little); // r_offset (disp32 field)
        w(u64, e[8..16], (@as(u64, 3) << 32) | R_X86_64_PC32, .little); // r_info (sym=p(3))
        w(i64, e[16..24], -4, .little); // r_addend
        try rela_text.appendSlice(allocator, &e);
    }

    // `.rela.data`: p's slot (offset 8 in .data) holds `&g` (R_X86_64_64 against g, symtab
    // index 2). This is the exact shape that `x86_64/object.zig`'s `.rela.data` emission
    // produces from a `link.Module`'s `DataReloc`.
    var rela_data: std.ArrayList(u8) = .empty;
    defer rela_data.deinit(allocator);
    {
        var e: [24]u8 = @splat(0);
        w(u64, e[0..8], 8, .little); // r_offset (p's slot within .data)
        w(u64, e[8..16], (@as(u64, 2) << 32) | R_X86_64_64, .little); // r_info (sym=g(2))
        w(i64, e[16..24], 0, .little); // r_addend
        try rela_data.appendSlice(allocator, &e);
    }

    // `.shstrtab`.
    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const data_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".data\x00");
    const relatext_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rela.text\x00");
    const reladata_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".rela.data\x00");
    const symtab_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".symtab\x00");
    const strtab_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".strtab\x00");
    const shstrtab_sname: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    // File layout: ehdr, then each section's data (8-aligned), then the section headers.
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
    const data_off = off;
    off += data.len;
    off = alignUp(off, 8);
    const rela_text_off = off;
    off += rela_text.items.len;
    const rela_data_off = off;
    off += rela_data.items.len;
    const symtab_off = off;
    off += symtab.items.len;
    const strtab_off = off;
    off += strtab.items.len;
    const shstr_off = off;
    off += shstr.items.len;
    off = alignUp(off, 8);
    const sh_off = off;
    const total = sh_off + @as(usize, nsections) * 64;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    w(u16, buf[16..18], 1, .little); // e_type = ET_REL
    w(u16, buf[18..20], 62, .little); // e_machine = EM_X86_64
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[40..48], sh_off, .little); // e_shoff
    w(u16, buf[52..54], 64, .little); // e_ehsize
    w(u16, buf[58..60], 64, .little); // e_shentsize
    w(u16, buf[60..62], nsections, .little); // e_shnum
    w(u16, buf[62..64], shstrtab_ndx, .little); // e_shstrndx

    @memcpy(buf[text_off..][0..text.len], &text);
    @memcpy(buf[data_off..][0..data.len], &data);
    @memcpy(buf[rela_text_off..][0..rela_text.items.len], rela_text.items);
    @memcpy(buf[rela_data_off..][0..rela_data.items.len], rela_data.items);
    @memcpy(buf[symtab_off..][0..symtab.items.len], symtab.items);
    @memcpy(buf[strtab_off..][0..strtab.items.len], strtab.items);
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
    const SHF_WRITE: u64 = 0x1;
    const SHF_EXECINSTR: u64 = 0x4;
    const SHF_INFO_LINK: u64 = 0x40;

    putShdr(buf, sh_off + text_ndx * 64, text_sname, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    putShdr(buf, sh_off + data_ndx * 64, data_sname, SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, data_off, data.len, 0, 0, 8, 0);
    putShdr(buf, sh_off + rela_text_ndx * 64, relatext_sname, SHT_RELA, SHF_INFO_LINK, rela_text_off, rela_text.items.len, symtab_ndx, text_ndx, 8, 24);
    putShdr(buf, sh_off + rela_data_ndx * 64, reladata_sname, SHT_RELA, SHF_INFO_LINK, rela_data_off, rela_data.items.len, symtab_ndx, data_ndx, 8, 24);
    putShdr(buf, sh_off + symtab_ndx * 64, symtab_sname, SHT_SYMTAB, 0, symtab_off, symtab.items.len, strtab_ndx, 1, 8, 24);
    putShdr(buf, sh_off + strtab_ndx * 64, strtab_sname, SHT_STRTAB, 0, strtab_off, strtab.items.len, 0, 0, 1, 0);
    putShdr(buf, sh_off + shstrtab_ndx * 64, shstrtab_sname, SHT_STRTAB, 0, shstr_off, shstr.items.len, 0, 0, 1, 0);

    return buf;
}

/// Locate a real cross glibc `ld-linux-x86-64.so.2` in the Nix store (the interpreter to
/// embed in the dynexe). Returns an allocated absolute path, or null if none is found.
fn findCrossInterp(allocator: std.mem.Allocator) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "sh", "-c", "find /nix/store -maxdepth 3 -name ld-linux-x86-64.so.2 2>/dev/null | head -1" },
    }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Locate the `qemu-x86_64` user-mode emulator on PATH. Returns an allocated absolute path,
/// or null if absent (so the caller skips cleanly and needs no PATH in the child env).
fn findQemu(allocator: std.mem.Allocator) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "sh", "-c", "command -v qemu-x86_64" },
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

test "x86-64 linkDynamic dynexe imports add() from a .so and the REAL cross ld.so runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-x86_64. Skip cleanly if either is absent.
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
    // ld.so (invoked by qemu-x86_64, which reads the embedded PT_INTERP).
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

test "x86-64 linkDynamic dynexe imports a DATA global from a .so via GOT/GLOB_DAT and the REAL cross ld.so resolves+runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-x86_64. Skip cleanly if either is absent.
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

    // Build the dynexe: `_start` loads `counter`'s address from the GOT (GOT-indirect
    // `mov rax, [rip+disp32]`), reads the i32 (42), and exits with it. The `.shared` input's
    // `counter` export satisfies the undefined GOTPCREL reference. DT_NEEDED is derived from
    // libcounter.so's soname (and is also passed explicitly).
    const start_obj = try buildDataStartObj(allocator);
    defer allocator.free(start_obj);
    const dynexe = try ld.linkDynamic(allocator, &.{ .{ .object = start_obj }, .{ .shared = libcounter_so } }, .{
        .mode = .exec,
        .interp = interp,
        .needed = &.{"libcounter.so"},
        .entry = "_start",
    });
    defer allocator.free(dynexe);

    // Structural: ET_EXEC with a data import (a `.rela.dyn` GLOB_DAT must exist, that is,
    // DT_RELA present). If the data-import path is unimplemented, DT_RELA is absent -> fail.
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, dynexe[16..18], .little)); // ET_EXEC
    const phdrs = try parsePhdrs(allocator, dynexe);
    defer allocator.free(phdrs);
    const DT_RELA: i64 = 7;
    try std.testing.expect((try readDynTag(dynexe, phdrs, DT_RELA)) != null);

    // Write both to a tmp dir. The exe gets the executable bit. Run it under the real cross
    // ld.so (invoked by qemu-x86_64, which reads the embedded PT_INTERP).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "libcounter.so", .data = libcounter_so });

    // Structural (host readelf, skipped if absent): the dynexe carries a R_X86_64_GLOB_DAT in
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
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_X86_64_GLOB_DAT") != null);
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "counter") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "dynexe",
        .data = dynexe,
        .flags = .{ .permissions = .executable_file },
    });

    // LD_LIBRARY_PATH = "." (relative to the child's cwd, the tmp dir), so the guest ld.so
    // finds libcounter.so by soname. qemu-user passes the environment through to the guest.
    // The kernel-substitute (qemu) reads PT_INTERP and runs the cross ld.so. The cross ld.so
    // loads libcounter.so, applies the GLOB_DAT (writes counter's address into the exe's GOT
    // slot), and enters `_start`, which reads counter (42) through the GOT.
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

test "x86_64 linkDynamic PIE dynexe (ET_DYN + base 0) imports add() and the REAL cross ld.so loads it at a chosen base and runs it to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Gate on the cross loader and qemu-x86_64. Skip cleanly if either is absent.
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
    // result. Unlike the non-PIE dynexe above, `.pie = true` asks the emitter for an ET_DYN
    // executable laid out at base 0 (every DT_*/reloc r_offset/e_entry is a base-0 vaddr that
    // the real ld.so biases by its chosen load address). The x86-64 code is
    // position-independent (the PC-relative `call`, the PLT's `jmp *disp32(%rip)`), so it
    // needs NO extra RELATIVE relocations. This proves the function-import path already
    // works through the shared `DynOptions.pie` machinery. The pointer-data test below adds
    // the new path.
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
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little));

    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    const DT_JMPREL: i64 = 23;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_JMPREL)) != null);

    // The entry is a base-0 vaddr (== its file offset), well below the non-PIE 0x400000 base:
    // proof the layout is truly base-relative (the loader adds its bias, we do not bake one in).
    const e_entry = std.mem.readInt(u64, pie[24..32], .little);
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

    // qemu-user reads the embedded PT_INTERP and runs the cross ld.so. Because this is an
    // ET_DYN executable, the loader picks a load base, biases e_entry and the GOT/PLT
    // r_offsets, loads libadd.so, applies the eager JUMP_SLOT, and enters `_start` at the
    // biased address. The base is nonzero (qemu-user pins it at a fixed high address rather
    // than re-randomizing per run), so a binary that assumed a base-0 layout would crash.
    // Ours exits 42. We run twice to confirm the ET_DYN load is stable and repeatable.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", ".");

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

test "x86_64 linkDynamic PIE with a pointer-initialized data global: the REAL cross ld.so applies R_X86_64_RELATIVE to p at a chosen base and *p reads g = exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    // A single object: `_start` derefs `p` (= &g) to read g (42). g and p are both internal,
    // so there is NO .so. The only fixup is p's internal pointer, which a PIE resolves
    // through an R_X86_64_RELATIVE that the loader applies at its chosen (ASLR) base.
    const obj = try buildPointerDataObj(allocator);
    defer allocator.free(obj);
    const pie = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .pie = true,
        .interp = interp,
        .entry = "_start",
    });
    defer allocator.free(pie);

    // A PIE is ET_DYN (3), and it must carry a `.rela.dyn` (DT_RELA present) for the RELATIVE.
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pie[16..18], .little));
    const phdrs = try parsePhdrs(allocator, pie);
    defer allocator.free(phdrs);
    const DT_RELA: i64 = 7;
    try std.testing.expect((try readDynTag(pie, phdrs, DT_RELA)) != null);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Structural (host readelf, skipped if absent): the PIE carries an R_X86_64_RELATIVE.
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
        try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_X86_64_RELATIVE") != null);
    }

    try tmp.dir.writeFile(io, .{
        .sub_path = "pie",
        .data = pie,
        .flags = .{ .permissions = .executable_file },
    });

    // Run twice under qemu (a fixed nonzero load base, repeated). Without the RELATIVE, p
    // holds g's base-0 address, and the deref faults or reads garbage at the nonzero base.
    // With the RELATIVE, the loader biases p, and *p = g = 42 both times.
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

test "x86_64 linkDynamic NON-PIE ET_EXEC with a pointer-initialized data global resolves p directly at link time (control): *p reads g = exit 42 under qemu" {
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

test "x86_64 object.writeModule emits a .rela.data R_X86_64_64 for a pointer-initialized data global (closes the data-section-reloc gap)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A module with `g` (writable int) and `p` (writable pointer = &g, carrying a DataReloc).
    var module: link.Module = .{};
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

    // The object must carry an `R_X86_64_64` in `.rela.data` (the pointer-init reloc that was
    // previously dropped). readelf -rW shows both the reloc type and the section name.
    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "ptr.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_X86_64_64") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, ".rela.data") != null);
}

/// Build one x86-64 relocatable object with THREE per-function `.text.<fn>` sections, the
/// `-ffunction-sections` shape the GC mark pass walks:
///   `.text._start` calls `used`, then exits with the returned value.
///   `.text.used`   returns 42 (the reachable callee).
///   `.text.unused` returns 7 (a leaf that NOTHING calls, so `--gc-sections` drops it).
/// `_start`/`used`/`unused` are global defined functions, one per section. The `call used`
/// (`E8 rel32`) at `.text._start` offset 0 carries an `R_X86_64_PLT32` at rel32 (offset 1,
/// addend -4) against `used` (symbol index 1).
fn buildGcProgramObj(allocator: std.mem.Allocator) ![]u8 {
    // `.text._start`: call used ; mov edi, eax ; mov eax, 60 (SYS_exit) ; syscall
    const start_text = [_]u8{
        0xe8, 0x00, 0x00, 0x00, 0x00, // call used   (rel32 @1)
        0x89, 0xc7, //                   mov edi, eax
        0xb8, 0x3c, 0x00, 0x00, 0x00, // mov eax, 60
        0x0f, 0x05, //                   syscall
    };
    // `.text.used`: mov eax, 42 ; ret
    const used_text = [_]u8{ 0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3 };
    // `.text.unused`: mov eax, 7 ; ret  (the immediate 7 is the drop marker)
    const unused_text = [_]u8{ 0xb8, 0x07, 0x00, 0x00, 0x00, 0xc3 };

    const alloc_exec: u64 = 0x2 | 0x4; // SHF_ALLOC | SHF_EXECINSTR
    const relocs = [_]object_emit.OutReloc{
        .{ .offset = 1, .symbol = 1, .r_type = R_X86_64_PLT32, .addend = -4 }, // call -> used
    };
    const sections = [_]object_emit.OutSection{
        .{ .name = ".text._start", .sh_type = 1, .flags = alloc_exec, .bytes = &start_text, .size = start_text.len, .addralign = 16, .relocs = &relocs },
        .{ .name = ".text.used", .sh_type = 1, .flags = alloc_exec, .bytes = &used_text, .size = used_text.len, .addralign = 16 },
        .{ .name = ".text.unused", .sh_type = 1, .flags = alloc_exec, .bytes = &unused_text, .size = unused_text.len, .addralign = 16 },
    };
    const symbols = [_]object_emit.OutSymbol{
        .{ .name = "_start", .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .func, .defined = true },
        .{ .name = "used", .section = 1, .value = 0, .size = 0, .binding = .global, .sym_type = .func, .defined = true },
        .{ .name = "unused", .section = 2, .value = 0, .size = 0, .binding = .global, .sym_type = .func, .defined = true },
    };
    return object_emit.emit(allocator, &sections, &symbols, .{ .class = .elf64, .machine = 62, .use_rela = true });
}

test "x86-64 linkDynamic with gc_sections drops an uncalled function's .text section, and keeps it with gc off" {
    const allocator = std.testing.allocator;

    const obj = try buildGcProgramObj(allocator);
    defer allocator.free(obj);

    // The distinctive bytes of `unused` (`mov eax, 7`). They exist ONLY in `.text.unused`, so
    // their presence tracks whether that section survived. Linking needs no loader or qemu,
    // only a PT_INTERP string, so this drop proof runs on every host (a placeholder path is
    // fine, only the RUN test below needs a real loader).
    const marker = [_]u8{ 0xb8, 0x07, 0x00, 0x00, 0x00 };
    const interp = "/lib64/ld-linux-x86-64.so.2";

    // GC OFF: `.text.unused` is kept, so its marker is present in the image.
    const exe_off = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .interp = interp,
        .entry = "_start",
        .gc_sections = false,
    });
    defer allocator.free(exe_off);
    try std.testing.expect(std.mem.indexOf(u8, exe_off, &marker) != null);

    // GC ON: nothing calls `unused`, so `.text.unused` is dropped. Its marker is gone.
    const exe_on = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .interp = interp,
        .entry = "_start",
        .gc_sections = true,
    });
    defer allocator.free(exe_on);
    try std.testing.expect(std.mem.indexOf(u8, exe_on, &marker) == null);
    // The dropped section's bytes are gone, the definitive proof. The overall file never
    // grows either (page/region alignment can absorb the freed bytes, so this is `<=`).
    try std.testing.expect(exe_on.len <= exe_off.len);
}

test "x86-64 linkDynamic gc_sections program (uncalled function dropped) still runs to exit 42 under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // The RUN needs a REAL cross loader and qemu-x86_64. Skip if either is absent.
    const interp = (try findCrossInterp(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(interp);
    const qemu = (try findQemu(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(qemu);

    const obj = try buildGcProgramObj(allocator);
    defer allocator.free(obj);

    // Link with GC on: `.text.unused` is dropped, `_start` + `used` are kept.
    const exe_on = try ld.linkDynamic(allocator, &.{.{ .object = obj }}, .{
        .mode = .exec,
        .interp = interp,
        .entry = "_start",
        .gc_sections = true,
    });
    defer allocator.free(exe_on);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{
        .sub_path = "gcexe",
        .data = exe_on,
        .flags = .{ .permissions = .executable_file },
    });
    // `_start` calls `used` (kept) and exits with 42.
    run_helper.runExpectExit(allocator, io, .{
        .argv = &.{ qemu, "./gcexe" },
        .cwd = .{ .dir = tmp.dir },
    }, 42) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}
