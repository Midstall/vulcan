//! This file tests linker-script layout for the shared linker (`vulcan-link`). It
//! builds two real ELF64 `.o` files: a `_start` stub that calls `main` and exits, and
//! a `main` that reads a writable `.data` global `g` (value 5) through a
//! `global_addr` and returns `g + 37`. It then drives `link.linkInputsScript` with a
//! representative GNU-ld-style script. The script has ENTRY, a custom
//! `. = 0x400000` base, ALIGN, output sections that gather
//! `*(.text*)`/`*(.rodata*)`/`*(.data*)`/`*(.bss*)`, and `__bss_start`/`__bss_end`
//! boundary symbols. The host is aarch64, so the script-laid-out image is wrapped in
//! a runnable ELF and executed directly. Exit code 42 proves the global resolves at
//! its script-assigned VMA, the CALL26 into `main` resolves across the two objects,
//! and `ENTRY(_start)` is honored as the entry.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const object = @import("../object.zig");
const link = @import("../link.zig");
const encode = @import("../encode.zig");
const ld = @import("vulcan-link");

const Function = ir.function.Function;

/// `main() i32 { return *(&g) + 37; }` compiled with a writable `.data` global `g = 5`,
/// serialized as its own ELF `.o`. The `global_addr` becomes an `adrp`/`add` pair with
/// two relocations against `g`. `main` reads it back and adds 37, so `main() == 42`.
fn buildMainObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var main = Function.init(allocator);
    defer main.deinit();
    const t = try main.types.intern(i32k);
    const ptr_t = try main.types.ptrGlobal();
    const b = try main.appendBlock();
    const g = try main.appendGlobalAddr(b, ptr_t, "g");
    const gv = try main.appendInst(b, t, .{ .load = .{ .ptr = g } });
    const c37 = try main.appendInst(b, t, .{ .iconst = 37 });
    const sum = try main.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = gv, .rhs = c37 } });
    main.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    // A writable `.data` global `g` holding the i32 `5` (little-endian).
    const g_bytes = [_]u8{ 5, 0, 0, 0 };
    try module.addWritable(allocator, "g", &g_bytes);
    return object.writeModule(allocator, &module);
}

/// A hand-assembled `_start` stub `.o`. It has `bl main`, a CALL26 relocation against
/// the external `main`, then `x8 = 93`, the exit syscall, and `svc #0`. It exits with
/// `main`'s return value in `x0` (AAPCS64). `_start` is a defined global in `.text` at
/// offset 0. `main` is left undefined for the linker to resolve against the other
/// object.
fn buildStartObj(allocator: std.mem.Allocator) ![]u8 {
    const words = [_]u32{
        encode.bl(0), // bl main, patched by the CALL26 relocation
        encode.movz(.x8, 93, 0), // x8 = 93, the exit syscall number
        encode.svc(0), // svc #0 calls exit(x0)
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .kind = .func, .defined = true, .section = .text },
        .{ .name = "main", .kind = .notype, .defined = false },
    };
    const relocs = [_]object.Reloc{
        .{ .offset = 0, .symbol = 1, .type = .call26 }, // bl targets main, symbol index 1
    };
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

const script_text =
    \\ENTRY(_start)
    \\SECTIONS {
    \\  . = 0x400000;
    \\  .text : { *(.text*) }
    \\  . = ALIGN(16);
    \\  .rodata : { *(.rodata*) }
    \\  .data : { *(.data*) }
    \\  __bss_start = .;
    \\  .bss : { *(.bss*) }
    \\  __bss_end = .;
    \\}
;

test "linkInputsScript lays out via a linker script; the .data global resolves at its script VMA and ENTRY(_start) runs to exit 42" {
    const allocator = std.testing.allocator;

    const start_o = try buildStartObj(allocator);
    defer allocator.free(start_o);
    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);

    var script = try ld.parseScript(allocator, script_text, null);
    defer script.deinit();

    var linked = try ld.linkInputsScript(allocator, &.{
        .{ .object = start_o },
        .{ .object = main_o },
    }, &script, null);
    defer linked.deinit(allocator);

    // The script places `.text` at 0x400000 with `_start` first, in input order, so
    // the entry is `_start` at exactly the base.
    try std.testing.expectEqual(@as(u64, 0x400000), linked.entry);
    try std.testing.expect(ld.elf.findSymbol(linked.placement.symbols, "_start") != null);
    try std.testing.expectEqual(@as(u64, 0x400000), ld.elf.findSymbol(linked.placement.symbols, "_start").?);
    // Script-defined boundary symbols were injected.
    try std.testing.expect(ld.elf.findSymbol(linked.placement.symbols, "__bss_start") != null);
    try std.testing.expect(ld.elf.findSymbol(linked.placement.symbols, "__bss_end") != null);
    // The data global resolved at an address >= the base VMA. Its exact address
    // depends on section sizes.
    const g_addr = ld.elf.findSymbol(linked.placement.symbols, "g").?;
    try std.testing.expect(g_addr >= 0x400000);

    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the AArch64 ELF directly

    const exe = try ld.writeElfSegments(linked.arch, allocator, &linked.placement, linked.entry);
    defer allocator.free(exe);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.out", .data = exe, .flags = .{ .permissions = .executable_file } });
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
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // *(&g) + 37 == 5 + 37
        else => return error.BackendFailed,
    }
}

// ---------------------------------------------------------------------------------------
// MEMORY regions, the LMA/AT split, and multiple PT_LOAD segments.
// ---------------------------------------------------------------------------------------

/// One parsed ELF64 program header (the fields this test asserts on).
const Phdr = struct { p_type: u32, p_flags: u32, p_vaddr: u64, p_paddr: u64, p_filesz: u64, p_memsz: u64 };

/// Parse the `PT_LOAD` program headers out of an emitted ELF64 executable. This
/// mirrors `resolve.emitElf64`'s header layout. `e_phoff` is at byte 32, `e_phnum` is
/// at byte 56, and each `Elf64_Phdr` is 56 bytes, with `p_type` at 0, `p_flags` at 4,
/// `p_vaddr` at 16, `p_paddr` at 24, `p_filesz` at 32, and `p_memsz` at 40.
fn parseLoadPhdrs(allocator: std.mem.Allocator, exe: []const u8) ![]Phdr {
    const r = std.mem.readInt;
    const e_phoff = r(u64, exe[32..40], .little);
    const e_phnum = r(u16, exe[56..58], .little);
    var out: std.ArrayList(Phdr) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < e_phnum) : (i += 1) {
        const p = exe[@intCast(e_phoff + i * 56)..];
        const t = r(u32, p[0..4], .little);
        if (t != 1) continue; // PT_LOAD only
        try out.append(allocator, .{
            .p_type = t,
            .p_flags = r(u32, p[4..8], .little),
            .p_vaddr = r(u64, p[16..24], .little),
            .p_paddr = r(u64, p[24..32], .little),
            .p_filesz = r(u64, p[32..40], .little),
            .p_memsz = r(u64, p[40..48], .little),
        });
    }
    return out.toOwnedSlice(allocator);
}

const script_split =
    \\ENTRY(_start)
    \\MEMORY {
    \\  rom (rx) : ORIGIN = 0x08000000, LENGTH = 256K
    \\  ram (rwx): ORIGIN = 0x20000000, LENGTH = 64K
    \\}
    \\SECTIONS {
    \\  .text : { *(.text*) } >rom
    \\  .data : { *(.data*) } >ram AT>rom
    \\}
;

test "linkInputsScript: a >ram AT>rom .data yields >=2 PT_LOADs with the data segment's p_vaddr (ram) != p_paddr (rom)" {
    const allocator = std.testing.allocator;

    const start_o = try buildStartObj(allocator);
    defer allocator.free(start_o);
    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);

    var script = try ld.parseScript(allocator, script_split, null);
    defer script.deinit();

    var linked = try ld.linkInputsScript(allocator, &.{
        .{ .object = start_o },
        .{ .object = main_o },
    }, &script, null);
    defer linked.deinit(allocator);

    // Two segments: `.text` in rom, where VMA equals LMA, and `.data` in ram loaded
    // from rom.
    try std.testing.expectEqual(@as(usize, 2), linked.placement.segments.len);
    // The `.data` global `g` resolves to its ram VMA (0x20000000), not its rom LMA.
    try std.testing.expectEqual(@as(u64, 0x20000000), ld.elf.findSymbol(linked.placement.symbols, "g").?);

    const exe = try ld.writeElfSegments(linked.arch, allocator, &linked.placement, linked.entry);
    defer allocator.free(exe);

    const phdrs = try parseLoadPhdrs(allocator, exe);
    defer allocator.free(phdrs);

    // At least 2 PT_LOADs. Exactly one of them, the `.data` segment, has
    // p_vaddr != p_paddr, with p_vaddr in ram and p_paddr in rom. This is the VMA/LMA
    // split, visible in the ELF.
    try std.testing.expect(phdrs.len >= 2);
    var split_found = false;
    for (phdrs) |ph| {
        if (ph.p_vaddr == ph.p_paddr) continue;
        split_found = true;
        try std.testing.expect(ph.p_vaddr >= 0x20000000 and ph.p_vaddr < 0x20000000 + 64 * 1024); // ram
        try std.testing.expect(ph.p_paddr >= 0x08000000 and ph.p_paddr < 0x08000000 + 256 * 1024); // rom
    }
    try std.testing.expect(split_found);
}

const script_two_region =
    \\ENTRY(_start)
    \\MEMORY {
    \\  rom (rx) : ORIGIN = 0x400000, LENGTH = 64K
    \\  ram (rw) : ORIGIN = 0x500000, LENGTH = 64K
    \\}
    \\SECTIONS {
    \\  .text : { *(.text*) } >rom
    \\  .data : { *(.data*) } >ram
    \\}
;

test "linkInputsScript: a two-region script (.text >rom, .data >ram, VMA == LMA) links to two PT_LOADs and runs to exit 42" {
    const allocator = std.testing.allocator;

    const start_o = try buildStartObj(allocator);
    defer allocator.free(start_o);
    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);

    var script = try ld.parseScript(allocator, script_two_region, null);
    defer script.deinit();

    var linked = try ld.linkInputsScript(allocator, &.{
        .{ .object = start_o },
        .{ .object = main_o },
    }, &script, null);
    defer linked.deinit(allocator);

    // `.text` sits at the rom base, giving the `_start` entry. `.data`'s `g` sits at
    // the ram base.
    try std.testing.expectEqual(@as(u64, 0x400000), linked.entry);
    try std.testing.expectEqual(@as(usize, 2), linked.placement.segments.len);
    try std.testing.expectEqual(@as(u64, 0x500000), ld.elf.findSymbol(linked.placement.symbols, "g").?);

    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the AArch64 ELF directly

    const exe = try ld.writeElfSegments(linked.arch, allocator, &linked.placement, linked.entry);
    defer allocator.free(exe);

    // Two PT_LOADs, each with p_vaddr == p_paddr, with no AT. This proves multi-region
    // and multi-PT_LOAD support.
    const phdrs = try parseLoadPhdrs(allocator, exe);
    defer allocator.free(phdrs);
    try std.testing.expect(phdrs.len >= 2);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.out", .data = exe, .flags = .{ .permissions = .executable_file } });
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
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // *(&g) + 37 == 5 + 37
        else => return error.BackendFailed,
    }
}
