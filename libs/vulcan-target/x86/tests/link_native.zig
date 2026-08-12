//! Coverage for the shared linker (`vulcan-link`) as used by the i386 backend: the
//! first real i386 object link (previously only hand-resolved in-test, see
//! `harness.zig`'s `runModuleData`). Builds a real ELF32 `.o` via `object.writeModule`,
//! links it with `vulcan-link`, wraps it in a runnable ELF32 executable, and executes
//! it under `qemu-i386`.
//!
//! Unlike x86-64/AArch64/RISC-V (all `SHT_RELA`), i386 objects use `SHT_REL`: the
//! addend for each relocation is pre-written into the relocated field itself (see
//! `object.zig`), not carried in the reloc entry. This test proves both i386 reloc
//! kinds resolve correctly through that path: `R_386_32` (a `global_addr` load's
//! `mov rd, imm32`, absolute) and `R_386_PC32` (a `call rel32`, PC-relative).

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("../encode.zig");
const object = @import("../object.zig");
const link = @import("../link.zig");
const ld = @import("vulcan-link");

const Function = ir.function.Function;

test "our i386 linker resolves an R_386_32 global load and an R_386_PC32 call, runs under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // int g = 5;
    // int helper(int v) { return v + 37; }
    // int main(void) { return helper(g); }
    // main() = helper(5) = 42. `g` is read back through a `global_addr` (the isel
    // `.abs32` arm: `mov rd, imm32`, an R_386_32 reloc), and `helper` is reached
    // through a `call rel32` (an R_386_PC32 reloc) - both relocs resolved by the new
    // `arch/x86.zig` linker, in the same object.
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const v = try helper.appendBlockParam(b, t);
        const r = try helper.appendArithImm(b, t, .add, v, 37);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(i32k);
        const ptr_t = try main.types.intern(.ptr);
        const b = try main.appendBlock();
        const p = try main.appendGlobalAddr(b, ptr_t, "g");
        const v = try main.appendInst(b, t, .{ .load = .{ .ptr = p } });
        const r = try main.appendCall(b, t, "helper", &.{v});
        main.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }

    const g_bytes = [_]u8{ 5, 0, 0, 0 }; // i32 5, little-endian
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);
    try module.addWritable(allocator, "g", &g_bytes); // .data (writable)

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);

    // The entry stub (below) is a raw prefix in front of the linked image, so the
    // image's real runtime load address is `base + stub_len`, not `base`: link at
    // that address so the PC32 call site and the R_386_32 absolute global load (both
    // resolved against true absolute runtime addresses) target the real runtime
    // locations of `g`/`helper`/`main`.
    //
    // exitseq: mov ebx, eax (2) ; mov eax, 1 (5) ; int 0x80 (2) = 9 bytes.
    // stub: call rel32 (5) ++ exitseq (9) = 14 bytes.
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.ebx, .eax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.eax, 1).slice());
    try exitseq.appendSlice(allocator, encode.int80().slice());
    const stub_len: u64 = 5 + exitseq.items.len;

    const base: u64 = 0x08048000; // the classic i386 ELF base
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    // `call main`: rel32 is relative to the byte right after the call instruction
    // (the stub sits at `base`, so the call ends at `base + 5`).
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

    const exe = try ld.writeElfExec(.x86, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(exe);
    try std.testing.expectEqualSlices(u8, "\x7fELF", exe[0..4]);
    try std.testing.expectEqual(@as(u8, 1), exe[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, exe[18..20], .little)); // EM_386

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.elf", .data = exe, .flags = .{ .permissions = .executable_file } });
    const proc = std.process.run(allocator, io, .{
        .argv = &.{ "qemu-i386", "./a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // helper(g) = 5 + 37
        else => return error.BackendFailed,
    }
}

test "object+ld+exec: i386 links through the Placement model, byte-identical (ELFCLASS32), and runs under qemu" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // Same program as the single-image path test above (g=5, helper(v)=v+37,
    // main()=helper(g)=42), but driven through computeDefaultPlacement/applyRelocs
    // directly instead of the `linkObjects` convenience wrapper.
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const v = try helper.appendBlockParam(b, t);
        const r = try helper.appendArithImm(b, t, .add, v, 37);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(i32k);
        const ptr_t = try main.types.intern(.ptr);
        const b = try main.appendBlock();
        const p = try main.appendGlobalAddr(b, ptr_t, "g");
        const v = try main.appendInst(b, t, .{ .load = .{ .ptr = p } });
        const r = try main.appendCall(b, t, "helper", &.{v});
        main.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }

    const g_bytes = [_]u8{ 5, 0, 0, 0 }; // i32 5, little-endian
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);
    try module.addWritable(allocator, "g", &g_bytes); // .data (writable)

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);

    // Same stub/entry-address reasoning as the test above: link at `base + stub_len`
    // so the stub can sit unrelocated at `base`.
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.ebx, .eax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.eax, 1).slice());
    try exitseq.appendSlice(allocator, encode.int80().slice());
    const stub_len: u64 = 5 + exitseq.items.len;

    const base: u64 = 0x08048000; // the classic i386 ELF base
    const link_base = base + stub_len;

    // Drive the Placement model directly: parse -> computeDefaultPlacement -> applyRelocs.
    var parsed = [_]ld.elf.ParsedObject{try ld.elf.parseObject(allocator, obj)};
    defer parsed[0].deinit(allocator);
    var placement = try ld.x86.computeDefaultPlacement(allocator, &parsed, link_base, null, false);
    defer placement.deinit(allocator);

    // The default placement is exactly one R|W|X segment mapping the whole image at
    // `link_base`.
    try std.testing.expectEqual(@as(usize, 1), placement.segments.len);
    try std.testing.expectEqual(link_base, placement.segments[0].vaddr);
    try std.testing.expectEqual(link_base, placement.segments[0].paddr);
    try std.testing.expectEqual(@as(u8, 7), placement.segments[0].flags);

    try ld.x86.applyRelocs(allocator, &placement, &parsed);

    // Byte-identity: a single-segment placement written by writeElfSegments must equal
    // the pre-refactor writeElfExec32 bytes for the same code/memsz/base/entry, and
    // land on the ELF32 branch (`emitElf32`, ELFCLASS32).
    {
        const code = placement.segments[0].bytes;
        const memsz = placement.segments[0].memsz;
        const via_exec = try ld.writeElfExec(.x86, allocator, code, memsz, link_base, link_base);
        defer allocator.free(via_exec);
        const via_seg = try ld.writeElfSegments(.x86, allocator, &placement, link_base);
        defer allocator.free(via_seg);
        try std.testing.expect(std.mem.eql(u8, via_exec, via_seg));
        try std.testing.expectEqual(@as(u8, 1), via_seg[4]); // ELFCLASS32
    }

    // `call main`: rel32 is relative to the byte right after the call instruction
    // (the stub sits at `base`, so the call ends at `base + 5`).
    const main_addr: i64 = @intCast(ld.elf.findSymbol(placement.symbols, "main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + 5)));
    var stub: std.ArrayList(u8) = .empty;
    defer stub.deinit(allocator);
    try stub.appendSlice(allocator, encode.callRel(rel).slice());
    try stub.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, stub.items.len);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, stub.items);
    try program.appendSlice(allocator, placement.segments[0].bytes);

    // Wrap the stub+image as a one-segment placement and emit via writeElfSegments.
    var run_segs = [_]ld.Segment{.{ .vaddr = base, .paddr = base, .bytes = program.items, .memsz = stub_len + placement.segments[0].memsz, .flags = 7 }};
    var run_pl: ld.Placement = .{ .segments = &run_segs, .places = &.{}, .symbols = &.{}, .entry = base };
    const exe = try ld.writeElfSegments(.x86, allocator, &run_pl, base);
    defer allocator.free(exe);
    try std.testing.expectEqualSlices(u8, "\x7fELF", exe[0..4]);
    try std.testing.expectEqual(@as(u8, 1), exe[4]); // ELFCLASS32
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, exe[18..20], .little)); // EM_386

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.elf", .data = exe, .flags = .{ .permissions = .executable_file } });
    const proc = std.process.run(allocator, io, .{
        .argv = &.{ "qemu-i386", "./a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // helper(g) = 5 + 37
        else => return error.BackendFailed,
    }
}
