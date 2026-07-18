//! Esperanto ET-SoC-1 (ETSOC-1) execution oracle. Vulcan's et-soc VPU packed-single
//! codegen (`isel.selectFunctionForModel` under the et-soc model) emits custom 8-lane
//! f32 opcodes that no general RISC-V emulator decodes. `sw-sysemu` (the functional
//! ETSOC-1 emulator, github.com/aifoundry-org/et-platform) is a genuine ISA interpreter
//! that computes real softfloat 8-lane f32, so it can execute that codegen.
//!
//! Unlike the qemu-riscv64 harness (`harness.zig`), sw-sysemu is a BAREMETAL machine
//! model: no Linux `ecall` ABI, no exit code. So this harness wraps the compiled
//! function in a tiny baremetal entry stub (enable the FP unit, set the stack pointer,
//! point the argument registers at input/output buffers laid out in the same image, call
//! the function, then `wfi` to halt) and reads the result back through `sw-sysemu`'s
//! `-dump_addr/-dump_size/-dump_file` memory dump of the output buffer.
//!
//! The whole test SKIPS when the `sys_emu` binary is absent (like the river/spike
//! runners skip when their emulator is not on PATH), so `zig build test` stays green in
//! CI while actually executing the codegen wherever sw-sysemu is built. To run it against
//! a local build, put that build's directory on `PATH`.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("vulcan-opt").microarch;
const encode = @import("../encode.zig");
const isel = @import("../isel.zig");

const Function = ir.function.Function;

/// File offset the loadable image is placed at (page-aligned so `p_vaddr` and `p_offset`
/// are congruent).
const code_offset: usize = 0x1000;

/// Write an ET-SoC-loadable ELF wrapping `image` (loaded at, and entered at, `base`).
/// sw-sysemu's loader walks each PT_LOAD segment's *sections* and copies each section's
/// bytes to memory. The production flat-image ELF writer emits no section headers, so it
/// would load nothing here; the image must instead be exposed as a `.text` section inside
/// the segment.
/// One PT_LOAD (R|W|X, so the kernel can store its result vector back), one `.text`
/// PROGBITS section spanning the whole image, and a `.shstrtab`. Caller owns the result.
fn writeSysemuElf(allocator: std.mem.Allocator, image: []const u8, base: u64) std.mem.Allocator.Error![]u8 {
    const strtab = "\x00.text\x00.shstrtab\x00"; // .text @1, .shstrtab @7
    const text_name: u32 = 1;
    const shstrtab_name: u32 = 7;
    const strtab_off = code_offset + image.len;
    const shoff = std.mem.alignForward(usize, strtab_off + strtab.len, 8);
    const total = shoff + 3 * 64;

    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    const w = std.mem.writeInt;

    buf[0] = 0x7f;
    buf[1] = 'E';
    buf[2] = 'L';
    buf[3] = 'F';
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    w(u16, buf[16..18], 2, .little); // e_type = ET_EXEC
    w(u16, buf[18..20], 243, .little); // e_machine = EM_RISCV
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[24..32], base, .little); // e_entry
    w(u64, buf[32..40], 64, .little); // e_phoff
    w(u64, buf[40..48], shoff, .little); // e_shoff
    w(u16, buf[52..54], 64, .little); // e_ehsize
    w(u16, buf[54..56], 56, .little); // e_phentsize
    w(u16, buf[56..58], 1, .little); // e_phnum
    w(u16, buf[58..60], 64, .little); // e_shentsize
    w(u16, buf[60..62], 3, .little); // e_shnum
    w(u16, buf[62..64], 2, .little); // e_shstrndx

    const p = buf[64..];
    w(u32, p[0..4], 1, .little); // p_type = PT_LOAD
    w(u32, p[4..8], 7, .little); // p_flags = R|W|X
    w(u64, p[8..16], code_offset, .little); // p_offset
    w(u64, p[16..24], base, .little); // p_vaddr
    w(u64, p[24..32], base, .little); // p_paddr
    w(u64, p[32..40], image.len, .little); // p_filesz
    w(u64, p[40..48], image.len, .little); // p_memsz
    w(u64, p[48..56], 0x1000, .little); // p_align

    @memcpy(buf[code_offset..][0..image.len], image);
    @memcpy(buf[strtab_off..][0..strtab.len], strtab);

    // Section 0 is the reserved null entry (already zeroed). Section 1 = .text, spanning
    // the whole loadable image so the loader copies it into memory. Section 2 = .shstrtab.
    const sh = buf[shoff..];
    w(u32, sh[64 + 0 ..][0..4], text_name, .little); // sh_name
    w(u32, sh[64 + 4 ..][0..4], 1, .little); // sh_type = SHT_PROGBITS
    w(u64, sh[64 + 8 ..][0..8], 0x7, .little); // sh_flags = ALLOC|WRITE|EXEC
    w(u64, sh[64 + 16 ..][0..8], base, .little); // sh_addr
    w(u64, sh[64 + 24 ..][0..8], code_offset, .little); // sh_offset
    w(u64, sh[64 + 32 ..][0..8], image.len, .little); // sh_size
    w(u64, sh[64 + 48 ..][0..8], 4, .little); // sh_addralign
    w(u32, sh[128 + 0 ..][0..4], shstrtab_name, .little); // sh_name
    w(u32, sh[128 + 4 ..][0..4], 3, .little); // sh_type = SHT_STRTAB
    w(u64, sh[128 + 24 ..][0..8], strtab_off, .little); // sh_offset
    w(u64, sh[128 + 32 ..][0..8], strtab.len, .little); // sh_size
    w(u64, sh[128 + 48 ..][0..8], 1, .little); // sh_addralign
    return buf;
}

/// The image loads into ETSOC-1 DRAM, which is executable and writable (the default
/// bootrom region at 0x8000001000 is read-execute, so storing the result vector there
/// would fault). The Minion resets straight into the entry stub at this address, so it
/// is both the load base and the entry point.
const load_base: u64 = 0x8000800000;
/// `wfi`: halts the Minion so sw-sysemu finishes emulation and performs its end-of-run
/// dump. encode.zig has no `wfi`, so it is spelled out as its fixed word here.
const wfi_word: u32 = 0x10500073;
/// A generous per-frame scratch stack reserved at the top of the image. The test kernel's
/// frame is a handful of words; 4 KiB is far more than any single frame needs.
const stack_reserve: u64 = 4096;

fn roundUp(x: u64, comptime n: u64) u64 {
    return (x + (n - 1)) & ~@as(u64, n - 1);
}

/// Append an `auipc rd, hi` + `addi rd, rd, lo` pair that materializes the absolute
/// address `load_base + target_off` into `rd`, PC-relative from the `auipc` sitting at
/// word index `from_idx`. Keeps the stub position-independent within the image, so no
/// 40-bit ETSOC-1 address ever has to be built as a literal.
fn appendPcRel(allocator: std.mem.Allocator, w: *std.ArrayList(u32), rd: encode.Reg, from_idx: usize, target_off: u64) std.mem.Allocator.Error!void {
    const pc_off: u64 = from_idx * 4;
    const delta: u32 = @intCast(target_off - pc_off);
    const hi: u20 = @truncate((delta +% 0x800) >> 12);
    const lo: i12 = @bitCast(@as(u12, @truncate(delta)));
    try w.append(allocator, encode.auipc(rd, hi));
    try w.append(allocator, encode.addi(rd, rd, lo));
}

/// The full baremetal image for one VPU run: the entry stub, the compiled function, and
/// the input/output buffers. `data_off` is the byte offset of the first input buffer;
/// `out_off` is the byte offset of the output buffer. Caller owns `bytes`.
const Image = struct {
    bytes: []u8,
    out_off: u64,

    fn deinit(self: Image, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
    }
};

/// Lay out and assemble the image from raw 32-bit lane words. `code` is the compiled
/// function (entry at word 0); `in_a` and `in_b` are the two 8-lane input vectors (as raw
/// little-endian u32 lanes, so this serves both f32 bit patterns and i32 values) the kernel
/// reads through its first two pointer arguments, writing 8 result words through its third.
fn buildImageWords(allocator: std.mem.Allocator, code: []const u32, in_a: [8]u32, in_b: [8]u32) std.mem.Allocator.Error!Image {
    // Fixed 13-word entry stub. Indices are load-bearing: the PC-relative pairs and the
    // call jal are patched against them, so the layout below must match one-for-one.
    //   0    lui   x6, 0x6           (x6 = 0x6000, mstatus.FS = Dirty)
    //   1    csrrs x0, mstatus, x6   (enable the FP/VPU unit)
    //   2-3  auipc/addi sp           (stack top)
    //   4-5  auipc/addi a0           (&in_a)
    //   6-7  auipc/addi a1           (&in_b)
    //   8-9  auipc/addi a2           (&out)
    //   10   jal   x1, <fn>          (call the compiled function)
    //   11   wfi                     (halt)
    //   12   jal   x0, 0             (safety self-loop, never reached)
    const stub_len: usize = 13;
    const total_code_words = stub_len + code.len;

    const data_off = roundUp(@as(u64, total_code_words) * 4, 32);
    const in_a_off = data_off;
    const in_b_off = data_off + 32;
    const out_off = data_off + 64;
    const stack_top_off = roundUp(out_off + 32 + stack_reserve, 16);

    var w: std.ArrayList(u32) = .empty;
    defer w.deinit(allocator);
    try w.append(allocator, encode.lui(.x6, 6)); // 0x6000
    try w.append(allocator, encode.csrrs(.x0, 0x300, .x6)); // mstatus.FS = Dirty
    try appendPcRel(allocator, &w, .x2, 2, stack_top_off); // sp
    try appendPcRel(allocator, &w, .x10, 4, in_a_off); // a0 = &in_a
    try appendPcRel(allocator, &w, .x11, 6, in_b_off); // a1 = &in_b
    try appendPcRel(allocator, &w, .x12, 8, out_off); // a2 = &out
    const fn_off: i21 = @intCast((stub_len - w.items.len) * 4);
    try w.append(allocator, encode.jal(.x1, fn_off)); // call the function
    try w.append(allocator, wfi_word);
    try w.append(allocator, encode.jal(.x0, 0));
    std.debug.assert(w.items.len == stub_len);
    try w.appendSlice(allocator, code);

    const bytes = try allocator.alloc(u8, stack_top_off);
    @memset(bytes, 0);
    for (w.items, 0..) |word, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    for (in_a, 0..) |v, i| std.mem.writeInt(u32, bytes[in_a_off + i * 4 ..][0..4], v, .little);
    for (in_b, 0..) |v, i| std.mem.writeInt(u32, bytes[in_b_off + i * 4 ..][0..4], v, .little);

    return .{ .bytes = bytes, .out_off = out_off };
}

/// f32 wrapper over `buildImageWords`: the two 8-lane f32 input vectors are laid out as their
/// little-endian bit patterns, exactly as before this file grew an integer path.
fn buildImage(allocator: std.mem.Allocator, code: []const u32, in_a: [8]f32, in_b: [8]f32) std.mem.Allocator.Error!Image {
    var a: [8]u32 = undefined;
    var b: [8]u32 = undefined;
    for (0..8) |i| {
        a[i] = @bitCast(in_a[i]);
        b[i] = @bitCast(in_b[i]);
    }
    return buildImageWords(allocator, code, a, b);
}

/// Run the assembled `image` under sw-sysemu and return the 32 raw output bytes (8 lanes) it
/// wrote to the output buffer. Returns `error.SkipZigTest` when the sw-sysemu binary is not
/// found. The f32 and integer readback wrappers below decode these bytes to their lane type.
fn runVpuImage(io: std.Io, allocator: std.mem.Allocator, image: Image) ![32]u8 {
    const elf = try writeSysemuElf(allocator, image.bytes, load_base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "kernel.elf", .data = elf });

    const out_addr = try std.fmt.allocPrint(allocator, "0x{x}", .{load_base + image.out_off});
    defer allocator.free(out_addr);
    const reset_pc = try std.fmt.allocPrint(allocator, "0x{x}", .{load_base});
    defer allocator.free(reset_pc);

    // Like the river/spike runners, the emulator is found by name on `PATH`; a custom
    // build is used by putting its directory on `PATH`. Absent, `process.run` returns
    // `error.FileNotFound` and the test skips.
    const argv = [_][]const u8{
        "sys_emu",        "-reset_pc",  reset_pc,
        "-single_thread", "-minions",   "0x1",
        "-shires",        "0x1",        "-elf_load",
        "kernel.elf",     "-dump_addr", out_addr,
        "-dump_size",     "32",         "-dump_file",
        "out.bin",
    };
    const result = std.process.run(allocator, io, .{ .argv = &argv, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // sw-sysemu not installed: skip, like river/spike do
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const dump = tmp.dir.readFileAlloc(io, "out.bin", allocator, .limited(64)) catch {
        std.debug.print("sw-sysemu produced no dump. term={any}\nstdout:\n{s}\nstderr:\n{s}\n", .{ result.term, result.stdout, result.stderr });
        return error.BackendFailed;
    };
    defer allocator.free(dump);
    if (dump.len < 32) return error.BackendFailed;

    var out: [32]u8 = undefined;
    @memcpy(&out, dump[0..32]);
    return out;
}

/// Run one 8-lane VPU kernel `code` (from `selectFunctionForModel`) under sw-sysemu with
/// the given f32 inputs, and return the 8 f32 lanes it wrote to the output buffer. Returns
/// `error.SkipZigTest` when the sw-sysemu binary is not found.
pub fn runVpuKernel(io: std.Io, allocator: std.mem.Allocator, code: []const u32, in_a: [8]f32, in_b: [8]f32) ![8]f32 {
    const image = try buildImage(allocator, code, in_a, in_b);
    defer image.deinit(allocator);
    const dump = try runVpuImage(io, allocator, image);
    var lanes: [8]f32 = undefined;
    for (0..8) |i| lanes[i] = @bitCast(std.mem.readInt(u32, dump[i * 4 ..][0..4], .little));
    return lanes;
}

/// Integer readback sibling of `runVpuKernel`: runs the kernel over two 8-lane i32 inputs
/// (passed as raw u32 lane words) and returns the 8 i32 lanes it wrote to the output buffer,
/// as the SAME little-endian bytes the f32 path reads. Returns `error.SkipZigTest` when the
/// sw-sysemu binary is not found.
pub fn runVpuKernelInt(io: std.Io, allocator: std.mem.Allocator, code: []const u32, in_a: [8]u32, in_b: [8]u32) ![8]i32 {
    const image = try buildImageWords(allocator, code, in_a, in_b);
    defer image.deinit(allocator);
    const dump = try runVpuImage(io, allocator, image);
    var lanes: [8]i32 = undefined;
    for (0..8) |i| lanes[i] = @bitCast(std.mem.readInt(u32, dump[i * 4 ..][0..4], .little));
    return lanes;
}

/// True if block 0 of `func` has a `load` producing a vector value (a coalesced wide vector load).
fn hasVectorLoad(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .load) continue;
        const r = func.instResult(inst).?;
        if (func.types.type_kind(func.valueType(r)) == .vector) return true;
    }
    return false;
}

/// True if block 0 of `func` has a `store` of a vector value (a coalesced wide vector store).
fn hasVectorStore(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .store) continue;
        if (func.types.type_kind(func.valueType(func.opcode(inst).store.value)) == .vector) return true;
    }
    return false;
}

/// Build the 8-lane elementwise f32 add kernel: `out[i] = a[i] + b[i]` for i in 0..8,
/// the exact shape `vectorize.runModel` produces under the et-soc model (scalar loads and
/// stores, one fused `<8 x f32>` add). Mirrors the structural isel test's builder.
fn buildAddKernel(func: *Function) !void {
    const V = ir.function.Value;
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = f32_t } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_a } });
    }
    const va = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&av) } });
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_b } });
    }
    const vb = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&bv) } });
    const vc = try func.appendInst(b, v8, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    for (0..8) |i| {
        const c = try func.appendInst(b, f32_t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, c, addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

/// Build a SCALAR 8-lane elementwise f32 kernel, `out[i] = a[i] * a[i] + b[i]` for i in 0..8:
/// 8 scalar loads of `a`, 8 scalar `a[i] * a[i]` muls, 8 scalar loads of `b`, 8 scalar
/// `mul_result + b[i]` adds, 8 scalar stores. Each arith op group is contiguous and shares one
/// `BinOp` (all 8 muls, then all 8 adds), the exact shape `vectorize.runModel` scans for, so
/// `microarch.optimize` SLP-fuses each group into one `<8 x f32>` op (chain reuse keeps the
/// mul's result in a vector register across the add, per vectorize.zig's `operandVector`).
/// Unlike `buildAddKernel` above (which builds the `<8 x f32>` arith directly), this stays
/// scalar so the SLP vectorizer itself is exercised, not just the VPU codegen for an
/// already-vectorized IR.
///
/// The operand each group first packs matters: `a` feeds BOTH sides of the mul group, so
/// packing it needs only the 8 already-live `a` scalars (the second pack of the same 8 values
/// extends their last use, it does not add new live ones). `a` is then fully dead (its
/// extracted, chain-reused vector took over) by the time `b` loads, so packing `b` for the add
/// group again needs only 8 live scalars, never both operands' 8 apiece at once. Mirroring the
/// `et-soc VPU: an 8-lane elementwise f32 add...` structural test's own note in isel.zig: the
/// vpu-mode scalar float pool is only f0..f9 (10 registers, no spill), so a group that packs
/// two DIFFERENT freshly-loaded 8-wide operands at once (e.g. a straight `a[i] * b[i]` as the
/// first group) would need 16 live scalars simultaneously and fail register allocation with
/// `error.Unsupported`.
fn buildSquareAddKernel(func: *Function) !void {
    const V = ir.function.Value;
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_a } });
    }
    var sqv: [8]V = undefined;
    for (0..8) |i| {
        sqv[i] = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = av[i], .rhs = av[i] } });
    }
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_b } });
    }
    var addv: [8]V = undefined;
    for (0..8) |i| {
        addv[i] = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = sqv[i], .rhs = bv[i] } });
    }
    for (0..8) |i| {
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, addv[i], addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

test "et-soc SLP: sw-sysemu executes an SLP-vectorized 8-lane f32 square-add and matches the scalar reference" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSquareAddKernel(&func);

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // SLP-vectorize the scalar kernel to 8-lane VPU arith. This (and the compile below) runs
    // unconditionally, before the sys_emu availability check, so a broken vectorizer or a
    // broken VPU isel path fails this test even when sw-sysemu itself is not on PATH.
    const changed = try mm.optimize(allocator, &func, model);
    try std.testing.expect(changed);
    // Memory coalescing must have fired: the scalar loads of `a`/`b` became wide `flw.ps` vector
    // loads and the scalar stores became one `fsw.ps` vector store. This is what makes the kernel
    // profitable without a scalar-float pack/unpack round trip; sw-sysemu executes those wide ops
    // below and the result must still match the scalar reference lane for lane.
    try std.testing.expect(hasVectorLoad(&func));
    try std.testing.expect(hasVectorStore(&func));
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const in_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const in_b = [8]f32{ 10.5, 20.25, -3.5, 0.0, 100.0, -0.5, 42.0, 1000.0 };

    const lanes = runVpuKernel(std.testing.io, allocator, code, in_a, in_b) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };

    // Scalar reference: plain host f32 mul then add, lane by lane. isel now FUSES this mul+add into
    // a single-rounding fmadd.ps, but the inputs here (a in 1..8 squared, b exact) make every product
    // and sum exactly representable, so single-rounding and the two-rounding host reference agree
    // bit-for-bit. If in_b is ever changed to inexact values, expect a 1-ulp fused-vs-unfused gap.
    for (0..8) |i| {
        const expected = in_a[i] * in_a[i] + in_b[i];
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(lanes[i])));
    }
}

test "et-soc VPU differential: sw-sysemu executes an 8-lane f32 add and matches the scalar reference" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildAddKernel(&func);

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const in_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const in_b = [8]f32{ 10.5, 20.25, -3.5, 0.0, 100.0, -0.5, 42.0, 1000.0 };

    const lanes = runVpuKernel(std.testing.io, allocator, code, in_a, in_b) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };

    // Scalar reference: plain host f32 add, lane by lane. sw-sysemu computes each lane
    // through a softfloat f32 add with the same round-to-nearest-even, so the results
    // must be bit-identical.
    for (0..8) |i| {
        const expected = in_a[i] + in_b[i];
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(lanes[i])));
    }
}

/// Build a direct-vector 8-lane f32 kernel `out[i] = a[i]*a[i] `second_op` b[i]`: pack `a` into
/// one `<8 x f32>`, one vector `mul(va, va)` (single-use), then one vector `second_op(mul, vb)`
/// consuming that product. With `second_op == .add` this is the exact `a*b + c` SSA shape the VPU
/// isel fuses into one `fmadd.ps`. With `second_op == .sub` it is the must-NOT-fuse shape (there is
/// no `fmsub.ps`, so it stays a separate `fmul.ps` + `fsub.ps`). Squaring `a` keeps this to two
/// input buffers so `runVpuKernel` drives it. Only `a` and `b` are ever packed at once (8 live
/// scalar floats), within the vpu-mode scalar pool f0..f9, so no pack overflows the pool.
fn buildVectorMulAddKernel(func: *Function, second_op: ir.function.BinOp) !void {
    const V = ir.function.Value;
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = f32_t } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    // Pack BOTH operands first, so the `mul` and its consuming add/sub are adjacent instructions:
    // the fusion predicate requires the add/sub to sit at exactly mul_index + 1 (nothing may run
    // between them). va is a live vector register while b packs, but that is only one register.
    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_a } });
    }
    const va = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&av) } });
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_b } });
    }
    const vb = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&bv) } });
    const prod = try func.appendInst(b, v8, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = va } });
    const res = try func.appendInst(b, v8, .{ .arith = .{ .op = second_op, .lhs = prod, .rhs = vb } });
    for (0..8) |i| {
        const c = try func.appendInst(b, f32_t, .{ .extract = .{ .aggregate = res, .index = @intCast(i) } });
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, c, addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

/// Count `fmadd.ps` words in `code`: opcode 0x5B (0b1011011) is the whole PS fused-multiply-add
/// family and this backend only ever emits the `fmadd` (sel = 00) member, so an opcode match is an
/// fmadd.ps (see encode.zig `vpuPsFma`).
fn countFmaddPs(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if ((w & 0x7F) == 0b1011011) n += 1;
    }
    return n;
}

/// Count `fmul.ps` words in `code`: PS R-type opcode 0x7B (0b1111011) with funct7 0b0001000 in
/// bits [31:25] (see encode.zig `fmul_ps`/`vpuPsRType`). A fused kernel must emit ZERO of these for
/// its product (the multiply folded into the fmadd.ps), an unfused one at least one.
fn countFmulPs(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if ((w & 0x7F) == 0b1111011 and (w >> 25) == 0b0001000) n += 1;
    }
    return n;
}

test "et-soc VPU differential: sw-sysemu executes a FUSED 8-lane f32 a*a+b as one fmadd.ps and matches the scalar reference" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildVectorMulAddKernel(&func, .add);

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    // Proof the fusion actually fired (not that the result is right by luck): the single-use
    // mul feeding the add folded into one fmadd.ps, so there is exactly one fmadd.ps and NO
    // separate fmul.ps for the product. This runs unconditionally, before the sys_emu
    // availability check, so it fails everywhere if the fusion regresses.
    try std.testing.expectEqual(@as(usize, 1), countFmaddPs(code));
    try std.testing.expectEqual(@as(usize, 0), countFmulPs(code));

    // Squares of small integers are exact and the sums below are exact too, so single-rounding
    // fmadd and the two-rounding host reference agree bit-for-bit (fp-contraction never changes
    // an exact result).
    const in_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const in_b = [8]f32{ 10.5, 20.25, -3.5, 0.0, 100.0, -0.5, 42.0, 1000.0 };

    const lanes = runVpuKernel(std.testing.io, allocator, code, in_a, in_b) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    for (0..8) |i| {
        const expected = in_a[i] * in_a[i] + in_b[i];
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(lanes[i])));
    }
}

test "et-soc VPU differential: an UNFUSED 8-lane f32 a*a-b keeps fmul.ps+fsub.ps (no fmsub.ps) and matches the scalar reference" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildVectorMulAddKernel(&func, .sub);

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    // Scope boundary: only the float ADD shape fuses. The et-soc ISA has no fmsub.ps, so a
    // sub-shape mul+sub must stay a standalone fmul.ps plus a fsub.ps and emit NO fmadd.ps.
    try std.testing.expectEqual(@as(usize, 0), countFmaddPs(code));
    try std.testing.expectEqual(@as(usize, 1), countFmulPs(code));

    const in_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const in_b = [8]f32{ 10.5, 20.25, -3.5, 0.0, 100.0, -0.5, 42.0, 1000.0 };

    const lanes = runVpuKernel(std.testing.io, allocator, code, in_a, in_b) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    for (0..8) |i| {
        const expected = in_a[i] * in_a[i] - in_b[i];
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(lanes[i])));
    }
}

/// Build the 8-lane `<8 x i32>` kernel `out[i] = op(a[i], b[i])`: 8 scalar i32 loads of `a`,
/// packed into one `<8 x i32>`, 8 scalar i32 loads of `b` packed likewise, one vector `arith`
/// lowered to the matching packed-integer (`pi`) op, then 8 extracts and 8 scalar i32 stores.
/// Mirrors `buildAddKernel` but every lane is an i32, so the pack/unpack rides the INT register
/// file (which spills freely), not the tight scalar-float pool - hence no register-pressure caveat.
fn buildIntKernel(func: *Function, op: ir.function.BinOp) !void {
    const V = ir.function.Value;
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = i32_t } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_a } });
    }
    const va = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&av) } });
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_b } });
    }
    const vb = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&bv) } });
    const vc = try func.appendInst(b, v8, .{ .arith = .{ .op = op, .lhs = va, .rhs = vb } });
    for (0..8) |i| {
        const c = try func.appendInst(b, i32_t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, c, addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

test "et-soc VPU differential: sw-sysemu executes 8-lane <8 x i32> pi ops and matches the scalar i32 reference" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Signed inputs (negatives included) so add/mul exercise two's-complement wraparound and the
    // sign bit, and xor exercises the bitwise path. The scalar reference is host i32 wrapping
    // arithmetic, which the pi ops must match lane-for-lane.
    const in_a = [8]i32{ 1, -2, 3, -4, 5, -6, 2147483647, -2147483648 };
    const in_b = [8]i32{ 10, 20, -30, 40, -50, 60, 1, -1 };
    var a_words: [8]u32 = undefined;
    var b_words: [8]u32 = undefined;
    for (0..8) |i| {
        a_words[i] = @bitCast(in_a[i]);
        b_words[i] = @bitCast(in_b[i]);
    }

    const Case = struct { op: ir.function.BinOp };
    const cases = [_]Case{ .{ .op = .add }, .{ .op = .mul }, .{ .op = .bit_xor } };
    for (cases) |c| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildIntKernel(&func, c.op);

        // The compile runs unconditionally, BEFORE the sys_emu availability check, so a broken pi
        // isel path fails this test even where sw-sysemu is not on PATH.
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const lanes = runVpuKernelInt(std.testing.io, allocator, code, a_words, b_words) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };

        for (0..8) |i| {
            const expected: i32 = switch (c.op) {
                .add => in_a[i] +% in_b[i],
                .mul => in_a[i] *% in_b[i],
                .bit_xor => in_a[i] ^ in_b[i],
                else => unreachable, // only the three cases above are built
            };
            try std.testing.expectEqual(expected, lanes[i]);
        }
    }
}

/// Build a SCALAR i32 multiply-add kernel, `out[i] = a[i] * b[i] + a[i]` for i in 0..8: 8 scalar
/// i32 loads of `a`, 8 scalar i32 loads of `b`, 8 contiguous `a[i] * b[i]` muls, 8 contiguous
/// `mul[i] + a[i]` adds, 8 scalar stores. Each arith group is contiguous and shares one BinOp (the
/// address math is `arith_imm`, a different opcode the SLP scan ignores), so under the et-soc model
/// `vectorize.runModel` SLP-fuses the mul group and the add group into two `<8 x i32>` pi ops (chain
/// reuse keeps the mul's result vector live into the add). Unlike `buildIntKernel` above, which
/// builds the `<8 x i32>` arith directly, this stays scalar so the SLP integer path itself is
/// exercised end to end. No register-pressure caveat like the f32 `buildSquareAddKernel`: the i32
/// lane scalars ride the INT file, which spills freely, so packing `a` and `b` both at once is fine.
fn buildIntMulAddKernel(func: *Function) !void {
    const V = ir.function.Value;
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_a } });
    }
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_b } });
    }
    var mulv: [8]V = undefined;
    for (0..8) |i| {
        mulv[i] = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = av[i], .rhs = bv[i] } });
    }
    var addv: [8]V = undefined;
    for (0..8) |i| {
        addv[i] = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = mulv[i], .rhs = av[i] } });
    }
    for (0..8) |i| {
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, addv[i], addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

/// Build a SCALAR i32 xor-then-shift kernel, `out[i] = (a[i] ^ b[i]) << 2` for i in 0..8: 8 scalar
/// i32 loads of each of `a` and `b`, 8 contiguous `a[i] ^ b[i]` xors, one shift-amount constant,
/// then 8 contiguous `xor[i] << 2` shifts, then 8 stores. Exercises the bitwise (`fxor.pi`) and the
/// left-shift (`fsll.pi`) pi lowerings that the SLP integer path produces. The shift amount is a
/// shared `iconst` (a non-arith op, so it does not break the contiguous shl run); the vectorizer
/// packs eight copies of it into the shift-amount vector.
fn buildIntXorShiftKernel(func: *Function) !void {
    const V = ir.function.Value;
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_a } });
    }
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_b } });
    }
    var xorv: [8]V = undefined;
    for (0..8) |i| {
        xorv[i] = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .bit_xor, .lhs = av[i], .rhs = bv[i] } });
    }
    const two = try func.appendInst(b, i32_t, .{ .iconst = 2 });
    var shv: [8]V = undefined;
    for (0..8) |i| {
        shv[i] = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .shl, .lhs = xorv[i], .rhs = two } });
    }
    for (0..8) |i| {
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, shv[i], addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

test "et-soc SLP: sw-sysemu executes SLP-vectorized 8-lane <8 x i32> pi kernels and matches the scalar i32 reference" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Signed inputs with negatives, and INT_MAX/INT_MIN, so the mul-add exercises two's-complement
    // wraparound and the sign bit, and the xor/shift exercises the bitwise and left-shift pi ops.
    // The scalar reference is host i32 wrapping arithmetic, which the SLP-produced pi ops must match
    // lane-for-lane.
    const in_a = [8]i32{ 1, -2, 3, -4, 5, -6, 2147483647, -2147483648 };
    const in_b = [8]i32{ 10, 20, -30, 40, -50, 60, 1, -1 };
    var a_words: [8]u32 = undefined;
    var b_words: [8]u32 = undefined;
    for (0..8) |i| {
        a_words[i] = @bitCast(in_a[i]);
        b_words[i] = @bitCast(in_b[i]);
    }

    const Kind = enum { mul_add, xor_shift };
    const cases = [_]Kind{ .mul_add, .xor_shift };
    for (cases) |kind| {
        var func = Function.init(allocator);
        defer func.deinit();
        switch (kind) {
            .mul_add => try buildIntMulAddKernel(&func),
            .xor_shift => try buildIntXorShiftKernel(&func),
        }

        // SLP-vectorize the SCALAR kernel to 8-lane <8 x i32> pi arith. This (and the compile below)
        // runs unconditionally, BEFORE the sys_emu availability check, so a broken vectorizer or a
        // broken pi isel path fails this test even when sw-sysemu is not on PATH.
        const changed = try mm.optimize(allocator, &func, model);
        try std.testing.expect(changed);
        // Coalescing fired: the scalar i32 loads/stores became wide vector loads/stores that
        // sw-sysemu executes below, matching the scalar i32 reference lane for lane.
        try std.testing.expect(hasVectorLoad(&func));
        try std.testing.expect(hasVectorStore(&func));
        var diags = try ir.verify.verify(allocator, &func, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());

        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const lanes = runVpuKernelInt(std.testing.io, allocator, code, a_words, b_words) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };

        for (0..8) |i| {
            const expected: i32 = switch (kind) {
                .mul_add => in_a[i] *% in_b[i] +% in_a[i],
                // Left shift is logical on the 32-bit pattern (zero-fill), identical for both
                // signednesses, so compute it in u32 space and reinterpret, matching `fsll.pi`.
                .xor_shift => @bitCast(@as(u32, @bitCast(in_a[i] ^ in_b[i])) << 2),
            };
            try std.testing.expectEqual(expected, lanes[i]);
        }
    }
}

test "buildImage lays out the entry stub, function, and buffers" {
    const allocator = std.testing.allocator;
    // A trivial one-word "function" (just `ret`) is enough to exercise the layout math.
    const code = [_]u32{encode.jalr(.x0, .x1, 0)};
    const in_a = [8]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const in_b = [8]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    const image = try buildImage(allocator, &code, in_a, in_b);
    defer image.deinit(allocator);

    // Entry stub is 13 words; the function follows; buffers are 32-byte aligned after.
    const data_off = roundUp((13 + code.len) * 4, 32);
    try std.testing.expectEqual(data_off + 64, image.out_off);
    // The first word is the FP-enable `lui x6, 0x6`, and the stub's call site is a jal
    // that reaches exactly the function (one word past the 13-word stub).
    try std.testing.expectEqual(encode.lui(.x6, 6), std.mem.readInt(u32, image.bytes[0..4], .little));
    try std.testing.expectEqual(encode.jal(.x1, 3 * 4), std.mem.readInt(u32, image.bytes[10 * 4 ..][0..4], .little));
    try std.testing.expectEqual(wfi_word, std.mem.readInt(u32, image.bytes[11 * 4 ..][0..4], .little));
    // in_a lands at the first buffer slot, little-endian f32 bits.
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1.0))), std.mem.readInt(u32, image.bytes[data_off..][0..4], .little));
}

// The matmul lowering (isel.zig `.matmul`) emits the tensor CSR-write protocol proven on
// sw-sysemu by /tmp/etsoc-build/matmul/mm.s, generalized to arbitrary m/n/k via a compile-time
// tile grid. A, B, and C are REAL row-major f32 matrices (A: m x k stride k*4, B: k x n stride
// n*4, C: m x n stride n*4 - no cache-line padding). Each buffer base is 64-byte aligned so the
// tensor_load unit (which addresses 64-byte lines) can reach the sub-tiles; rows whose real pitch
// is not a multiple of 64 are staged into a stack scratch by the lowering itself. This needs its
// own image layout (three pointer args at a0/a1/a2, three real row-major matrices) distinct from
// the two-8-lane-vector layout `buildImageWords` uses.

/// The matmul image: the same 13-word entry stub as `buildImageWords` (FP enable, sp, a0/a1/a2,
/// call, wfi), but a0=&A, a1=&B, a2=&C, where A/B/C are real row-major f32 matrices.
/// `c_off`/`c_size` locate the output for the dump. Caller owns `bytes`.
const MatmulImage = struct {
    bytes: []u8,
    c_off: u64,
    c_size: u64,

    fn deinit(self: MatmulImage, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
    }
};

/// The IR-level matmul input dtype (fp32 / fp16->fp32 / int8->int32 / uint8->int32).
const MMType = ir.function.MatMulType;

/// A/B element size in bytes for `dtype` (C is ALWAYS 32-bit, so its size is separate).
fn mmElemBytes(dtype: MMType) usize {
    return switch (dtype) {
        .fp32 => 4,
        .fp16 => 2,
        .int8, .uint8 => 1,
    };
}

/// Encode the integer value `v` as one A/B element of `dtype` into `dst` (little-endian). All the
/// differential tests use integer-valued inputs so the products/sums are EXACT for every dtype:
/// int8/uint8 are integers by nature; small integers are exactly representable in f16/f32.
fn writeMatmulElem(dst: []u8, dtype: MMType, v: i32) void {
    switch (dtype) {
        .fp32 => std.mem.writeInt(u32, dst[0..4], @bitCast(@as(f32, @floatFromInt(v))), .little),
        .fp16 => std.mem.writeInt(u16, dst[0..2], @bitCast(@as(f16, @floatFromInt(v))), .little),
        .int8 => dst[0] = @bitCast(@as(i8, @intCast(v))), // signed: two's-complement byte
        .uint8 => dst[0] = @intCast(v), // unsigned: 0..255
    }
}

/// Lay out and assemble a matmul image for `dtype`. `a` is row-major `m x k`, `b` is row-major
/// `k x n`, both as INTEGER values (encoded to the dtype's element type by `writeMatmulElem`),
/// written into memory as REAL row-major matrices (A stride k*elem, B stride n*elem). The tensor
/// unit does any dtype-specific SCP repacking itself (the isel lowering), so memory stays plain
/// row-major for every dtype. C is real row-major 32-bit output (stride n*4: fp32 for fp32/fp16,
/// int32 for int8/uint8), left zeroed for the kernel to fill. Each buffer base is 64-byte aligned.
/// A generous slack (the stack reserve) follows C so the tensor unit's 64-byte line over-reads at
/// the last A/B row never run off the end of the image.
fn buildMatmulImage(allocator: std.mem.Allocator, dtype: MMType, code: []const u32, a: []const i32, b: []const i32, m: u16, n: u16, k: u16) std.mem.Allocator.Error!MatmulImage {
    std.debug.assert(a.len == @as(usize, m) * k);
    std.debug.assert(b.len == @as(usize, k) * n);
    const elem = mmElemBytes(dtype);
    const stub_len: usize = 13;
    const total_code_words = stub_len + code.len;

    // Real row-major buffers, each base rounded to a 64-byte line.
    const a_off = roundUp(@as(u64, total_code_words) * 4, 64);
    const b_off = roundUp(a_off + @as(u64, m) * k * elem, 64);
    const c_off = roundUp(b_off + @as(u64, k) * n * elem, 64);
    const c_size: u64 = @as(u64, m) * n * 4; // C is always 32-bit accumulators
    const stack_top_off = roundUp(c_off + c_size + stack_reserve, 16);

    var w: std.ArrayList(u32) = .empty;
    defer w.deinit(allocator);
    try w.append(allocator, encode.lui(.x6, 6)); // 0x6000
    try w.append(allocator, encode.csrrs(.x0, 0x300, .x6)); // mstatus.FS = Dirty (enable FP/VPU)
    try appendPcRel(allocator, &w, .x2, 2, stack_top_off); // sp
    try appendPcRel(allocator, &w, .x10, 4, a_off); // a0 = &A
    try appendPcRel(allocator, &w, .x11, 6, b_off); // a1 = &B
    try appendPcRel(allocator, &w, .x12, 8, c_off); // a2 = &C
    const fn_off: i21 = @intCast((stub_len - w.items.len) * 4);
    try w.append(allocator, encode.jal(.x1, fn_off)); // call the matmul function
    try w.append(allocator, wfi_word); // halt
    try w.append(allocator, encode.jal(.x0, 0)); // safety self-loop
    std.debug.assert(w.items.len == stub_len);
    try w.appendSlice(allocator, code);

    const bytes = try allocator.alloc(u8, stack_top_off);
    @memset(bytes, 0);
    for (w.items, 0..) |word, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    // A: real row-major, element (i,j) at a_off + (i*k + j)*elem.
    for (0..m) |i| {
        for (0..k) |j| {
            writeMatmulElem(bytes[a_off + (i * k + j) * elem ..], dtype, a[i * k + j]);
        }
    }
    // B: real row-major, element (i,j) at b_off + (i*n + j)*elem.
    for (0..k) |i| {
        for (0..n) |j| {
            writeMatmulElem(bytes[b_off + (i * n + j) * elem ..], dtype, b[i * n + j]);
        }
    }
    return .{ .bytes = bytes, .c_off = c_off, .c_size = c_size };
}

/// Run a matmul image under sw-sysemu and return the `c_size` raw output bytes it wrote to C.
/// Returns `error.SkipZigTest` when the sw-sysemu binary is not on PATH (like the VPU runners).
fn runMatmulImage(io: std.Io, allocator: std.mem.Allocator, image: MatmulImage) ![]u8 {
    const elf = try writeSysemuElf(allocator, image.bytes, load_base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "kernel.elf", .data = elf });

    const out_addr = try std.fmt.allocPrint(allocator, "0x{x}", .{load_base + image.c_off});
    defer allocator.free(out_addr);
    const dump_size = try std.fmt.allocPrint(allocator, "{d}", .{image.c_size});
    defer allocator.free(dump_size);
    const reset_pc = try std.fmt.allocPrint(allocator, "0x{x}", .{load_base});
    defer allocator.free(reset_pc);

    const argv = [_][]const u8{
        "sys_emu",        "-reset_pc",  reset_pc,
        "-single_thread", "-minions",   "0x1",
        "-shires",        "0x1",        "-elf_load",
        "kernel.elf",     "-dump_addr", out_addr,
        "-dump_size",     dump_size,    "-dump_file",
        "out.bin",
    };
    const result = std.process.run(allocator, io, .{ .argv = &argv, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // sw-sysemu not installed: skip
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const dump = tmp.dir.readFileAlloc(io, "out.bin", allocator, .limited(4 << 20)) catch {
        std.debug.print("sw-sysemu produced no dump. term={any}\nstdout:\n{s}\nstderr:\n{s}\n", .{ result.term, result.stdout, result.stderr });
        return error.BackendFailed;
    };
    errdefer allocator.free(dump);
    if (dump.len < image.c_size) return error.BackendFailed;
    return dump;
}

/// Build a function containing exactly one `matmul c = a @ b` over three pointer parameters
/// (a0=&A, a1=&B, a2=&C), the shape the isel `.matmul` lowering handles. `accumulate` picks the
/// tensor_fma first_pass flag.
fn buildMatmulKernel(func: *Function, m: u16, n: u16, k: u16, dtype: MMType, accumulate: bool) !void {
    const ptr_t = try func.types.intern(.ptr);
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    try func.appendMatmul(blk, pa, pb, pc, m, n, k, dtype, accumulate);
    func.setTerminator(blk, .{ .ret = null });
}

/// Build an EMBEDDED matmul kernel: a value V is loaded into a float register BEFORE the matmul and
/// stored back AFTER it, so V is live ACROSS the matmul. The V slot lives just past C in memory (at
/// C + m*n*4, pre-initialized by the image); a >= 4-row matmul clobbers every scalar float temp
/// f0..f7 (TenC is f0..f(2*min(16,m)-1)), so V's register is definitely clobbered. Only the embedded
/// save/restore keeps V intact, so a mis-lowered (or non-self-contained) matmul makes the stored V
/// observably wrong while C stays correct. a0=&A, a1=&B, a2=&C, exactly like `buildMatmulKernel`.
fn buildEmbeddedMatmulKernel(func: *Function, m: u16, n: u16, k: u16, dtype: MMType) !void {
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    const vptr = try func.appendArithImm(blk, ptr_t, .add, pc, @as(i64, m) * n * 4);
    const vf = try func.appendInst(blk, f32_t, .{ .load = .{ .ptr = vptr } });
    try func.appendMatmulEmbedded(blk, pa, pb, pc, m, n, k, dtype, false, null);
    try func.appendStore(blk, vf, vptr);
    func.setTerminator(blk, .{ .ret = null });
}

/// The distinctive live-across value V the embedded differential checks survives the matmul: f32
/// 42.5 (a non-integer bit pattern so a clobber cannot coincidentally reproduce it).
const embedded_v_bits: u32 = @bitCast(@as(f32, 42.5));

/// The `csrw csr, rs1` word masked to the fields that identify the CSR write: opcode (SYSTEM,
/// 0x73), funct3 (001 = CSRRW), and the CSR number in bits [31:20]. rd/rs1 (the scratch registers)
/// are masked out so the structural check does not depend on register allocation.
fn csrwMasked(word: u32) u32 {
    return word & 0xFFF0_707F;
}

/// Whether `code` contains a `csrw <csr>, *` (any rs1/rd) somewhere.
fn hasCsrw(code: []const u32, csr: u12) bool {
    const want = csrwMasked(encode.csrw(csr, .x0));
    for (code) |word| {
        if (csrwMasked(word) == want) return true;
    }
    return false;
}

/// Reconstruct the 64-bit descriptor value written by each `csrw <csr>, rs1` in `code`, by running
/// a tiny interpreter over exactly the instruction forms `loadImm32`/`loadImm64` and the matmul
/// lowering emit (addi / lui / slli / ori / add / or / andi). Every descriptor is fully built into
/// its scratch register immediately before its `csrw`, so the register file value at the `csrw`
/// site is that descriptor. Used to inspect the tensor_fma descriptors' first_pass bit. Caller owns
/// the returned list.
fn collectCsrwDescriptors(allocator: std.mem.Allocator, code: []const u32, csr: u12) std.mem.Allocator.Error![]u64 {
    var regs = [_]u64{0} ** 32;
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(allocator);
    for (code) |word| {
        const opcode = word & 0x7F;
        const rd = (word >> 7) & 0x1F;
        const funct3 = (word >> 12) & 0x7;
        const rs1 = (word >> 15) & 0x1F;
        const rs2 = (word >> 20) & 0x1F;
        const imm_i: u64 = @bitCast(@as(i64, @as(i12, @bitCast(@as(u12, @intCast(word >> 20)))))); // sign-extended I-imm
        switch (opcode) {
            0x37 => regs[rd] = @bitCast(@as(i64, @as(i32, @bitCast(word & 0xFFFFF000)))), // lui (sign-extended)
            0x13 => switch (funct3) {
                0x0 => regs[rd] = regs[rs1] +% imm_i, // addi
                0x1 => regs[rd] = regs[rs1] << @intCast((word >> 20) & 0x3F), // slli
                0x6 => regs[rd] = regs[rs1] | imm_i, // ori
                0x7 => regs[rd] = regs[rs1] & imm_i, // andi
                else => {},
            },
            0x33 => switch (funct3) {
                0x0 => regs[rd] = regs[rs1] +% regs[rs2], // add
                0x6 => regs[rd] = regs[rs1] | regs[rs2], // or
                else => {},
            },
            0x73 => if (funct3 == 0x1 and (word >> 20) == csr) { // csrrw x0, csr, rs1
                try out.append(allocator, regs[rs1]);
            },
            else => {},
        }
        regs[0] = 0; // x0 stays zero
    }
    return try out.toOwnedSlice(allocator);
}

test "et-soc matmul: structural CSR sequence, fsw.ps store, and first_pass toggle" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // 16x16x32: one output tile (m,n <= 16), two K slices (fp32 K-tile 16, k=32 -> ki 0 and 1) so
    // first_pass toggles across the accumulation, and cols=16 so the store uses fsw.ps (two full
    // 8-lane groups per row).
    try buildMatmulKernel(&func, 16, 16, 32, .fp32, false);

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    // The lowering must emit the mcache_control enable (0x7e0), the tensor_loads (0x83f), the
    // tensor_fma (0x801), and the tensor_waits (0x830). Field-masked so register choices do not
    // matter.
    try std.testing.expect(hasCsrw(code, encode.CSR_MCACHE_CONTROL));
    try std.testing.expect(hasCsrw(code, encode.CSR_TENSOR_LOAD));
    try std.testing.expect(hasCsrw(code, encode.CSR_TENSOR_FMA));
    try std.testing.expect(hasCsrw(code, encode.CSR_TENSOR_WAIT));
    // The M0 mask preamble (needed for the fsw.ps readback) and at least two fsw.ps stores must be
    // present (cols=16 -> two full 8-lane groups per row).
    try std.testing.expect(std.mem.indexOfScalar(u32, code, encode.mov_m_x(0, .x0, 0xFF)) != null);
    var fsw_ps_count: usize = 0;
    for (code) |word| {
        // fsw.ps opcode 0x0B, funct3 110; mask out the register/immediate fields.
        if ((word & 0x707F) == (encode.fsw_ps(.f0, .x0, 0) & 0x707F)) fsw_ps_count += 1;
    }
    try std.testing.expect(fsw_ps_count >= 2);

    // first_pass toggle: the two K slices of the single output tile emit two tensor_fma descriptors,
    // the first with first_pass=1 (bit0, fresh TenC) and the second with first_pass=0 (accumulate).
    const fmas = try collectCsrwDescriptors(allocator, code, encode.CSR_TENSOR_FMA);
    defer allocator.free(fmas);
    try std.testing.expectEqual(@as(usize, 2), fmas.len);
    try std.testing.expectEqual(@as(u64, 1), fmas[0] & 1); // ki=0: first_pass=1
    try std.testing.expectEqual(@as(u64, 0), fmas[1] & 1); // ki=1: first_pass=0 (accumulate)
}

test "et-soc matmul: only the et-soc VPU model lowers matmul" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulKernel(&func, 2, 4, 2, .fp32, false);

    // A non-vpu riscv model must reject matmul cleanly (error.Unsupported), not mis-emit it.
    const river = mm.modelFor(.@"river-rc1.f");
    try std.testing.expect(!river.vpu());
    try std.testing.expectError(error.Unsupported, isel.selectFunctionForModel(allocator, &func, river));
}

test "et-soc matmul differential: sw-sysemu computes correct fp32 matrix products across tile shapes" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Shapes exercising every axis of the compile-time tile grid, all real row-major:
    //   2x4x2   - the original single tile (degenerate 1x1x1 grid), staged loads (k,n < 16).
    //   4x4x3   - single tile, K=3 staged.
    //   4x4x32  - two K slices (k=32) accumulated into one output tile; A aligned, B staged (n=4).
    //   8x32x8  - two N tiles (n=32); B aligned, A staged (k=8).
    //   32x8x8  - two M tiles (m=32); both A and B staged (k,n=8).
    //   32x32x32 - full 2x2 output grid, 2 K slices each; fully aligned (k,n mult of 16, no staging).
    // Integer-valued f32 inputs so the softfloat products/sums are exact and the scalar reference is
    // bit-identical to sw-sysemu.
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 2, .n = 4, .k = 2 },
        .{ .m = 4, .n = 4, .k = 3 },
        .{ .m = 4, .n = 4, .k = 32 },
        .{ .m = 8, .n = 32, .k = 8 },
        .{ .m = 32, .n = 8, .k = 8 },
        .{ .m = 32, .n = 32, .k = 32 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        // Deterministic small integer-valued matrices (kept small so the exact-fp32 window holds).
        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..m) |i| for (0..k) |j| {
            a[i * k + j] = @intCast((i * k + j) % 7);
        };
        for (0..k) |i| for (0..n) |j| {
            b[i * n + j] = @intCast((i * n + j) % 5);
        };

        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulKernel(&func, m, n, k, .fp32, false);
        // Compile runs unconditionally, before the sys_emu availability check, so a broken matmul
        // isel path fails here even where sw-sysemu is not on PATH.
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Reference: exact integer matmul in f32. C is real row-major (stride n*4): element (i,j) at
        // (i*n + j)*4 in the dump.
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const got: f32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got)));
            }
        }
    }
}

test "et-soc matmul differential: accumulate=true adds A*B onto a nonzero preloaded C (fp32)" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // accumulate=true must mean real `C += A*B`: the lowering PRELOADS the existing C tile into TenC
    // before the fma passes, so the (first_pass=0) fma computes `C_initial + A*B`. Seeding C with
    // NONZERO values before the run is the non-vacuity guard: an overwrite (accumulate=false) lowering
    // would drop the seed and produce plain A*B, FAILING the seed+A*B assert below. Shapes:
    //   4x4x32  - pure 4-col REMAINDER tile (cols=4, the staged-remainder preload path), two K slices.
    //   4x8x8   - one FULL 8-col group (cols=8, the direct flw.ps preload path), a single K slice.
    //   8x12x32 - one full group + one 4-col remainder (cols=12, BOTH preload paths), two K slices, m=8.
    // Integer-valued inputs and a 1..9 integer seed keep every f32 sum exact (well under 2^24), so the
    // scalar reference is bit-identical to sw-sysemu regardless of accumulation order.
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 4, .n = 4, .k = 32 },
        .{ .m = 4, .n = 8, .k = 8 },
        .{ .m = 8, .n = 12, .k = 32 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..m) |i| for (0..k) |j| {
            a[i * k + j] = @intCast((i * k + j) % 7);
        };
        for (0..k) |i| for (0..n) |j| {
            b[i * n + j] = @intCast((i * n + j) % 5);
        };

        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulKernel(&func, m, n, k, .fp32, true); // accumulate=true (this test's whole point)
        // Compile unconditionally (before the sys_emu availability check) so a preload-lowering
        // regression fails here even where sw-sysemu is not on PATH.
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        var image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
        defer image.deinit(allocator);

        // Pre-seed C with distinctive NONZERO integer-valued f32 (1..9) so seed + A*B stays exact and a
        // seed-dropping (overwrite) lowering would be caught. C is real row-major fp32, stride n*4.
        const c_base: usize = @intCast(image.c_off);
        const seed = try allocator.alloc(f32, @as(usize, m) * n);
        defer allocator.free(seed);
        for (0..m) |i| for (0..n) |j| {
            const s: f32 = @floatFromInt((i * n + j) % 9 + 1); // 1..9, never zero
            seed[i * n + j] = s;
            std.mem.writeInt(u32, image.bytes[c_base + (i * n + j) * 4 ..][0..4], @bitCast(s), .little);
        };

        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Reference: C_final[i][j] == seed[i][j] + sum_k A[i][k]*B[k][j], bit-exact fp32.
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = seed[i * n + j];
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const got: f32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got)));
            }
        }
    }
}

test "et-soc matmul differential: an EMBEDDED matmul preserves a live-across value and still computes C" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Every shape has m >= 4, so TenC (f0..f(2*min(16,m)-1)) covers all scalar float temps f0..f7 and
    // the loaded V is guaranteed to sit in a clobbered register. 16x16x16 is the worst case: it
    // clobbers the whole float file f0..f31. Both aligned (16x16x16, no staging) and staged
    // (4x4x4, 8x8x8: n%16!=0) tile paths are exercised so the save/restore wraps them all.
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 4, .n = 4, .k = 4 },
        .{ .m = 8, .n = 8, .k = 8 },
        .{ .m = 16, .n = 16, .k = 16 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..m) |i| for (0..k) |j| {
            a[i * k + j] = @intCast((i * k + j) % 7);
        };
        for (0..k) |i| for (0..n) |j| {
            b[i * n + j] = @intCast((i * n + j) % 5);
        };

        var func = Function.init(allocator);
        defer func.deinit();
        try buildEmbeddedMatmulKernel(&func, m, n, k, .fp32);
        // Compile unconditionally (before the sys_emu availability check) so an embedded-lowering
        // regression fails here even where sw-sysemu is not on PATH.
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        // Reuse the plain matmul image, then place the live-across value V just past C (in the stack
        // slack the builder already reserves) and widen the dump to include it. The matmul writes
        // exactly m*n*4 bytes of C (no overhang for these aligned column counts), so the V slot is
        // untouched by the matmul itself: only the kernel's post-matmul store writes it.
        var image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
        defer image.deinit(allocator);
        const v_slot: usize = @intCast(image.c_off + @as(u64, m) * n * 4);
        std.mem.writeInt(u32, image.bytes[v_slot..][0..4], embedded_v_bits, .little);
        image.c_size = @as(u64, m) * n * 4 + 4; // dump C plus the trailing V word

        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // (a) C is the exact integer matmul, byte-for-byte, despite the surrounding embedded code.
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const got: f32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got)));
            }
        }
        // (b) The live-across value V survived the matmul's full register clobber (the save/restore
        // is what makes this hold; without it V would read back as a TenC accumulator).
        const v_got = std.mem.readInt(u32, dump[@as(usize, m) * n * 4 ..][0..4], .little);
        try std.testing.expectEqual(embedded_v_bits, v_got);
    }
}

/// Signed-vs-unsigned for the int8 differential reference: the tensor unit sign-extends int8 (ua/ub
/// clear) or zero-extends uint8, so the host reference must do the same to stay bit-exact.
fn refElem(dtype: MMType, v: i32) i32 {
    return switch (dtype) {
        .int8 => @as(i32, @as(i8, @intCast(v))), // sign-extend the two's-complement byte
        .uint8 => @as(i32, @as(u8, @intCast(v))), // zero-extend
        else => unreachable, // integer reference is only used for the int8/uint8 paths
    };
}

test "et-soc matmul differential: sw-sysemu computes correct int8->int32 and uint8->int32 products" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // int8 K-tile is 64 (one 64-byte SCP line holds 64 int8). Shapes exercise: a sub-K tile
    // (4x4x4), a full single K-pass (4x4x64), two K passes (4x4x128), and a multi-output-tile grid
    // (32x32x32, one K pass since 32 <= 64). Both signed and unsigned int8. Integer inputs are exact
    // in int32, so the reference is bit-identical to sw-sysemu's tensor_ima8a32 path.
    const Case = struct { m: u16, n: u16, k: u16, dtype: MMType };
    const cases = [_]Case{
        .{ .m = 4, .n = 4, .k = 4, .dtype = .int8 },
        .{ .m = 4, .n = 4, .k = 64, .dtype = .int8 },
        .{ .m = 4, .n = 4, .k = 128, .dtype = .int8 },
        .{ .m = 32, .n = 32, .k = 32, .dtype = .int8 },
        .{ .m = 4, .n = 4, .k = 4, .dtype = .uint8 },
        .{ .m = 8, .n = 8, .k = 64, .dtype = .uint8 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;
        const dtype = cse.dtype;

        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        // Signed int8 data spans negatives (exercises sign-extension); uint8 data spans 0..255.
        for (0..a.len) |idx| a[idx] = if (dtype == .uint8) @intCast((idx * 3) % 200) else @as(i32, @intCast(idx % 15)) - 7;
        for (0..b.len) |idx| b[idx] = if (dtype == .uint8) @intCast((idx * 5) % 200) else @as(i32, @intCast(idx % 13)) - 6;

        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulKernel(&func, m, n, k, dtype, false);
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const image = try buildMatmulImage(allocator, dtype, code, a, b, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Reference: exact int32 matmul, element (i,j) at (i*n + j)*4 in the dump (int32 output).
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: i32 = 0;
                for (0..k) |kk| acc +%= refElem(dtype, a[i * k + kk]) *% refElem(dtype, b[kk * n + j]);
                const got: i32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(acc, got);
            }
        }
    }
}

test "et-soc matmul differential: sw-sysemu computes correct fp16->fp32 products" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // fp16 K-tile is 32 (one 64-byte SCP line holds 32 f16). 4x4x4 is a sub-K tile; 4x4x32 is one
    // full K pass; 8x16x32 exercises two output rows and a full-width column tile. Small integers are
    // exactly representable in f16 and their sums exact in the f32 accumulator, so the fp32 reference
    // is bit-identical to sw-sysemu's tensor_fma16a32 path.
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 4, .n = 4, .k = 4 },
        .{ .m = 4, .n = 4, .k = 32 },
        .{ .m = 8, .n = 16, .k = 32 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..a.len) |idx| a[idx] = @intCast(idx % 7);
        for (0..b.len) |idx| b[idx] = @intCast(idx % 5);

        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulKernel(&func, m, n, k, .fp16, false);
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const image = try buildMatmulImage(allocator, .fp16, code, a, b, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Reference: exact integer matmul accumulated in f32 (C is 32-bit fp32 output).
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const got: f32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got)));
            }
        }
    }
}

test "et-soc matmul: fma type field and unsigned bits per dtype" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // The fma descriptor's type field is bits 3:1 (fp32=0, fp16=1, int8=3); tena/tenb_unsigned are
    // bits 22/21, set only for uint8. Verify structurally for each dtype (no sysemu needed).
    const Case = struct { dtype: MMType, want_type: u64, want_unsigned: bool };
    const cases = [_]Case{
        .{ .dtype = .fp32, .want_type = 0, .want_unsigned = false },
        .{ .dtype = .fp16, .want_type = 1, .want_unsigned = false },
        .{ .dtype = .int8, .want_type = 3, .want_unsigned = false },
        .{ .dtype = .uint8, .want_type = 3, .want_unsigned = true },
    };
    for (cases) |cse| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulKernel(&func, 4, 4, 4, cse.dtype, false);
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const fmas = try collectCsrwDescriptors(allocator, code, encode.CSR_TENSOR_FMA);
        defer allocator.free(fmas);
        try std.testing.expect(fmas.len >= 1);
        for (fmas) |d| {
            try std.testing.expectEqual(cse.want_type, (d >> 1) & 0x7); // type field, bits 3:1
            try std.testing.expectEqual(cse.want_unsigned, ((d >> 22) & 1) != 0); // tena_unsigned, bit 22
            try std.testing.expectEqual(cse.want_unsigned, ((d >> 21) & 1) != 0); // tenb_unsigned, bit 21
        }
    }
}

// The isel `.matmul` lowering reads `mmv.input_signs`, when present, to set the fma descriptor's
// `ua`/`ub` bits INDEPENDENTLY per operand instead of both from the dtype-derived `di.uns` (see
// isel.zig, the `a_uns`/`b_uns` computed just above the tile grid). This is the asymmetric-inference
// shape: unsigned uint8 activations times signed int8 weights (or the mirror). Both operands still
// share the same 1-byte int8 HARDWARE dtype (verify requires `input_signs != null` to pair with
// `dtype == .int8`), so only the encoding/reference per operand differs, not the tile/K math.
const InputSigns = ir.function.InputSigns;

/// Encode one int8-hardware element as unsigned (raw byte, 0..255, zero-extended by the hardware)
/// or signed (two's-complement byte, sign-extended by the hardware) per `unsigned`. Sibling of
/// `writeMatmulElem` for the mixed-signedness matmul, where A and B may have DIFFERENT signedness
/// on the same `.int8` hardware dtype (so a single `dtype`-keyed encoder cannot express both).
fn writeElemSigned(dst: []u8, unsigned: bool, v: i32) void {
    dst[0] = if (unsigned) @intCast(v) else @bitCast(@as(i8, @intCast(v)));
}

/// Reference for one int8-hardware element under `unsigned`: zero-extend (the value is already
/// 0..255) or sign-extend the two's-complement byte. Sibling of `refElem` for mixed signedness.
fn refElemSigned(unsigned: bool, v: i32) i32 {
    return if (unsigned) v else @as(i32, @as(i8, @intCast(v)));
}

/// Build a function containing exactly one mixed-signedness `matmul c = a @ b` (int32 output, no
/// quant) via `appendMatmulSigned`. Sibling of `buildMatmulKernel`, always `dtype == .int8` (the
/// only dtype `input_signs` may pair with).
fn buildMatmulMixedKernel(func: *Function, m: u16, n: u16, k: u16, accumulate: bool, input_signs: InputSigns) !void {
    const ptr_t = try func.types.intern(.ptr);
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    try func.appendMatmulSigned(blk, pa, pb, pc, m, n, k, .int8, accumulate, input_signs);
    func.setTerminator(blk, .{ .ret = null });
}

/// Lay out and assemble a mixed-signedness matmul image: same layout as `buildMatmulImage` (A/B
/// real row-major, C is `m*n*4` int32 bytes), but A is encoded by `a_unsigned` and B by
/// `b_unsigned` independently via `writeElemSigned`, instead of both from one `dtype`. Sibling of
/// `buildMatmulImage`, kept separate so that builder's single-dtype encoding stays untouched.
fn buildMatmulMixedImage(allocator: std.mem.Allocator, code: []const u32, a: []const i32, b: []const i32, a_unsigned: bool, b_unsigned: bool, m: u16, n: u16, k: u16) std.mem.Allocator.Error!MatmulImage {
    std.debug.assert(a.len == @as(usize, m) * k);
    std.debug.assert(b.len == @as(usize, k) * n);
    const elem = 1; // int8 hardware dtype: always 1 byte, for both mixed operands
    const stub_len: usize = 13;
    const total_code_words = stub_len + code.len;

    const a_off = roundUp(@as(u64, total_code_words) * 4, 64);
    const b_off = roundUp(a_off + @as(u64, m) * k * elem, 64);
    const c_off = roundUp(b_off + @as(u64, k) * n * elem, 64);
    const c_size: u64 = @as(u64, m) * n * 4; // C is always 32-bit accumulators
    const stack_top_off = roundUp(c_off + c_size + stack_reserve, 16);

    var w: std.ArrayList(u32) = .empty;
    defer w.deinit(allocator);
    try w.append(allocator, encode.lui(.x6, 6)); // 0x6000
    try w.append(allocator, encode.csrrs(.x0, 0x300, .x6)); // mstatus.FS = Dirty (enable FP/VPU)
    try appendPcRel(allocator, &w, .x2, 2, stack_top_off); // sp
    try appendPcRel(allocator, &w, .x10, 4, a_off); // a0 = &A
    try appendPcRel(allocator, &w, .x11, 6, b_off); // a1 = &B
    try appendPcRel(allocator, &w, .x12, 8, c_off); // a2 = &C
    const fn_off: i21 = @intCast((stub_len - w.items.len) * 4);
    try w.append(allocator, encode.jal(.x1, fn_off)); // call the matmul function
    try w.append(allocator, wfi_word); // halt
    try w.append(allocator, encode.jal(.x0, 0)); // safety self-loop
    std.debug.assert(w.items.len == stub_len);
    try w.appendSlice(allocator, code);

    const bytes = try allocator.alloc(u8, stack_top_off);
    @memset(bytes, 0);
    for (w.items, 0..) |word, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    for (0..m) |i| {
        for (0..k) |j| {
            writeElemSigned(bytes[a_off + (i * k + j) * elem ..], a_unsigned, a[i * k + j]);
        }
    }
    for (0..k) |i| {
        for (0..n) |j| {
            writeElemSigned(bytes[b_off + (i * n + j) * elem ..], b_unsigned, b[i * n + j]);
        }
    }
    return .{ .bytes = bytes, .c_off = c_off, .c_size = c_size };
}

test "et-soc matmul differential: mixed-signedness sw-sysemu computes correct uint8-A x int8-B (and mirror) products" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // m=4,n=8,k=8 (sub-K-tile: k=8 < the int8 K-tile of 64, one fma pass). Case 1 is the real
    // asymmetric-inference shape: unsigned uint8 activations (A) times signed int8 weights (B).
    // Case 2 is the mirror (signed A x unsigned B), proving the override is genuinely
    // per-operand and not just "A always unsigned". Whichever operand is unsigned in a case gets
    // values > 127 (200, 255, 128) woven in: a sign-extension bug (isel wrongly using the
    // dtype-default signed `di.uns` for that operand instead of `mmv.input_signs`) would turn
    // 200/255/128 into -56/-1/-128, changing the accumulator, so the reference formula
    // (`refElemSigned`) catches it directly.
    const Case = struct { a_unsigned: bool, b_unsigned: bool };
    const cases = [_]Case{
        .{ .a_unsigned = true, .b_unsigned = false }, // uint8-A x int8-B: the asymmetric-inference shape
        .{ .a_unsigned = false, .b_unsigned = true }, // int8-A x uint8-B: the mirror
    };
    const m: u16 = 4;
    const n: u16 = 8;
    const k: u16 = 8;
    for (cases) |cse| {
        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..a.len) |idx| {
            a[idx] = if (cse.a_unsigned)
                (switch (idx % 4) {
                    0 => @as(i32, 200),
                    1 => 255,
                    2 => 128,
                    else => @intCast(idx % 200),
                })
            else
                @as(i32, @intCast(idx % 15)) - 7;
        }
        for (0..b.len) |idx| {
            b[idx] = if (cse.b_unsigned)
                (switch (idx % 4) {
                    0 => @as(i32, 200),
                    1 => 255,
                    2 => 128,
                    else => @intCast(idx % 200),
                })
            else
                @as(i32, @intCast(idx % 13)) - 6;
        }

        var func = Function.init(allocator);
        defer func.deinit();
        const input_signs: InputSigns = .{ .a_unsigned = cse.a_unsigned, .b_unsigned = cse.b_unsigned };
        try buildMatmulMixedKernel(&func, m, n, k, false, input_signs);
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const image = try buildMatmulMixedImage(allocator, code, a, b, cse.a_unsigned, cse.b_unsigned, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        for (0..m) |i| {
            for (0..n) |j| {
                var acc: i32 = 0;
                for (0..k) |kk| acc +%= refElemSigned(cse.a_unsigned, a[i * k + kk]) *% refElemSigned(cse.b_unsigned, b[kk * n + j]);
                const got: i32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(acc, got);
            }
        }
    }
}

// The quant epilogue (isel.zig `.matmul` case, `has_quant` branch) runs entirely on the vector
// register file after the last K-slice fma: (relu?) -> i32_to_f32 -> fp32_mul_row -> f32_to_i32 ->
// satint8 -> pack_128b. C then holds one requantized int8 byte per output element instead of the
// 32-bit accumulators the non-quant path stores. This needs its own kernel builder (dtype is
// always `.int8`, plus a `MatMulQuant`) and its own image builder (C is `m*n` bytes, not `m*n*4`).

const MMQuant = ir.function.MatMulQuant;
const MMQuantSpec = Function.MatMulQuantSpec;

/// Build a function containing exactly one requantizing `matmul c = quant(a @ b)` over three
/// pointer parameters (a0=&A, a1=&B, a2=&C). Sibling of `buildMatmulKernel`. `dtype` is the A/B
/// input dtype (verify rejects any dtype other than `.int8`/`.uint8` paired with a quant; the
/// symmetric plan-12/13 differentials all pass `.int8`, the asymmetric ones `.uint8`).
fn buildMatmulQuantKernel(func: *Function, m: u16, n: u16, k: u16, dtype: MMType, accumulate: bool, quant: MMQuant) !void {
    const ptr_t = try func.types.intern(.ptr);
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    try func.appendMatmulQuant(blk, pa, pb, pc, m, n, k, dtype, accumulate, quant);
    func.setTerminator(blk, .{ .ret = null });
}

/// Build a function containing exactly one requantizing `matmul c = quant(a @ b)` from a full
/// `MatMulQuantSpec` (bias + zero_point + scale + relu + out). Sibling of `buildMatmulQuantKernel`
/// / `buildMatmulQuantPerColumnKernel`, which only cover the symmetric (no-bias, zero_point == 0)
/// subset those two builders' call sites need; the asymmetric-uint8 differentials need bias and a
/// nonzero zero_point, so they go through `appendMatmulQuantSpec` directly.
fn buildMatmulQuantSpecKernel(func: *Function, m: u16, n: u16, k: u16, dtype: MMType, accumulate: bool, spec: MMQuantSpec) !void {
    const ptr_t = try func.types.intern(.ptr);
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    try func.appendMatmulQuantSpec(blk, pa, pb, pc, m, n, k, dtype, accumulate, spec);
    func.setTerminator(blk, .{ .ret = null });
}

/// Lay out and assemble a requantizing matmul image for `dtype` (`.int8` or `.uint8`, the only
/// dtypes a quant epilogue accepts). Same entry stub and A/B real row-major layout as
/// `buildMatmulImage`, but C is `m*n` BYTES (one requantized result per output element), not the
/// 32-bit accumulators the non-quant path writes. Kept as a sibling of `buildMatmulImage` rather
/// than folded into it so the frozen non-quant callers and their byte-for-byte expectations stay
/// untouched.
fn buildMatmulQuantImage(allocator: std.mem.Allocator, dtype: MMType, code: []const u32, a: []const i32, b: []const i32, m: u16, n: u16, k: u16) std.mem.Allocator.Error!MatmulImage {
    std.debug.assert(a.len == @as(usize, m) * k);
    std.debug.assert(b.len == @as(usize, k) * n);
    const elem = mmElemBytes(dtype);
    const stub_len: usize = 13;
    const total_code_words = stub_len + code.len;

    const a_off = roundUp(@as(u64, total_code_words) * 4, 64);
    const b_off = roundUp(a_off + @as(u64, m) * k * elem, 64);
    const c_off = roundUp(b_off + @as(u64, k) * n * elem, 64);
    const c_size: u64 = @as(u64, m) * n; // requantized output: one int8 byte per element
    const stack_top_off = roundUp(c_off + c_size + stack_reserve, 16);

    var w: std.ArrayList(u32) = .empty;
    defer w.deinit(allocator);
    try w.append(allocator, encode.lui(.x6, 6)); // 0x6000
    try w.append(allocator, encode.csrrs(.x0, 0x300, .x6)); // mstatus.FS = Dirty (enable FP/VPU)
    try appendPcRel(allocator, &w, .x2, 2, stack_top_off); // sp
    try appendPcRel(allocator, &w, .x10, 4, a_off); // a0 = &A
    try appendPcRel(allocator, &w, .x11, 6, b_off); // a1 = &B
    try appendPcRel(allocator, &w, .x12, 8, c_off); // a2 = &C
    const fn_off: i21 = @intCast((stub_len - w.items.len) * 4);
    try w.append(allocator, encode.jal(.x1, fn_off)); // call the matmul function
    try w.append(allocator, wfi_word); // halt
    try w.append(allocator, encode.jal(.x0, 0)); // safety self-loop
    std.debug.assert(w.items.len == stub_len);
    try w.appendSlice(allocator, code);

    const bytes = try allocator.alloc(u8, stack_top_off);
    @memset(bytes, 0);
    for (w.items, 0..) |word, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    for (0..m) |i| {
        for (0..k) |j| {
            writeMatmulElem(bytes[a_off + (i * k + j) * elem ..], dtype, a[i * k + j]);
        }
    }
    for (0..k) |i| {
        for (0..n) |j| {
            writeMatmulElem(bytes[b_off + (i * n + j) * elem ..], dtype, b[i * n + j]);
        }
    }
    return .{ .bytes = bytes, .c_off = c_off, .c_size = c_size };
}

/// Scalar host reference for the requantized int8 output: `acc = sum_k A[i][k]*B[k][j]` (exact
/// int32 for integer int8 inputs), `relu` clamps negative accs to 0, then `acc` scales by the fp32
/// `scale`, rounds to int32, and saturates to `-128..127`. Test inputs are chosen so `acc*scale`
/// is always an EXACT integer (never a `.5` fraction), so round-to-nearest-even (what
/// `f32_to_i32` uses in hardware) and plain truncation agree and the expected value is
/// unambiguous.
fn refQuantElem(acc_in: i32, relu: bool, scale: f32) i8 {
    const acc: i32 = if (relu) @max(0, acc_in) else acc_in;
    const scaled: f32 = @as(f32, @floatFromInt(acc)) * scale;
    const r: i32 = @intFromFloat(scaled); // exact by construction: no rounding-mode ambiguity
    const sat = std.math.clamp(r, -128, 127);
    return @intCast(sat);
}

/// Sibling of `refQuantElem` for `.satuint8` output: same `(relu?) -> *scale -> round` chain, but
/// saturates to `0..255` instead of `-128..127` (the `MatMulQuantOut.u8` requantize path).
fn refQuantElemU8(acc_in: i32, relu: bool, scale: f32) u8 {
    const acc: i32 = if (relu) @max(0, acc_in) else acc_in;
    const scaled: f32 = @as(f32, @floatFromInt(acc)) * scale;
    const r: i32 = @intFromFloat(scaled); // exact by construction: no rounding-mode ambiguity
    const sat = std.math.clamp(r, 0, 255);
    return @intCast(sat);
}

test "et-soc matmul quant differential: sw-sysemu computes bit-exact requantized int8 output" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    const Case = struct { m: u16, n: u16, k: u16, quant: MMQuant, a: []const i32, b: []const i32, expected: []const i8 };
    const cases = [_]Case{
        // GOLDEN proof shape (matches the hand-proven kernel): scale 0.5, no relu. B's columns are
        // constant across k (col0=10, col1=1, col2=-5, col3=0), so acc(i,j) = col_j * sum(A[i]).
        // Row1 col0: sum(A[1])=26 -> acc=260 -> *0.5=130, which SATURATES to 127.
        .{
            .m = 2,
            .n = 4,
            .k = 4,
            .quant = .{ .scale = .{ .scalar = 0x3F000000 }, .relu = false }, // 0.5
            .a = &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 },
            .b = &[_]i32{ 10, 1, -5, 0, 10, 1, -5, 0, 10, 1, -5, 0, 10, 1, -5, 0 },
            .expected = &[_]i8{ 50, 5, -25, 0, 127, 13, -65, 0 },
        },
        // Saturation both ways at scale 1.0: B's columns (constant across k) are 30/-30/6/-2, and
        // A's row sums are 10 and 14, so acc = col_j * row_sum lands both above +127 (col0) and
        // below -128 (col1), while col2/col3 stay within range (checking the unclamped math too).
        .{
            .m = 2,
            .n = 4,
            .k = 4,
            .quant = .{ .scale = .{ .scalar = 0x3F800000 }, .relu = false }, // 1.0
            .a = &[_]i32{ 1, 2, 3, 4, 2, 3, 4, 5 },
            .b = &[_]i32{ 30, -30, 6, -2, 30, -30, 6, -2, 30, -30, 6, -2, 30, -30, 6, -2 },
            .expected = &[_]i8{ 127, -128, 60, -20, 127, -128, 84, -28 },
        },
        // relu at scale 1.0: A's rows sum to +10 and -10; B's columns (constant across k) are
        // 5/-5/20/-3. Row0 accs are 50/-50/200/-30 -> relu keeps 50 and 200 (200 then saturates to
        // 127), zeroes -50 and -30. Row1 accs are -50/50/-200/30 -> relu keeps 50 and 30, zeroes
        // -50 and -200.
        .{
            .m = 2,
            .n = 4,
            .k = 4,
            .quant = .{ .scale = .{ .scalar = 0x3F800000 }, .relu = true }, // 1.0
            .a = &[_]i32{ 1, 2, 3, 4, -1, -2, -3, -4 },
            .b = &[_]i32{ 5, -5, 20, -3, 5, -5, 20, -3, 5, -5, 20, -3, 5, -5, 20, -3 },
            .expected = &[_]i8{ 50, 0, 127, 0, 0, 50, 0, 30 },
        },
    };

    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulQuantKernel(&func, m, n, k, .int8, false, cse.quant);
        // Compile runs unconditionally, before the sys_emu availability check, so a broken quant
        // isel path fails here even where sw-sysemu is not on PATH.
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        const image = try buildMatmulQuantImage(allocator, .int8, code, cse.a, cse.b, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        const scale: f32 = @bitCast(cse.quant.scale.scalar);
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: i32 = 0;
                for (0..k) |kk| acc +%= refElem(.int8, cse.a[i * k + kk]) *% refElem(.int8, cse.b[kk * n + j]);
                const want = refQuantElem(acc, cse.quant.relu, scale);
                try std.testing.expectEqual(want, cse.expected[i * n + j]); // sanity: hand-derived table matches the reference formula
                const got: i8 = @bitCast(dump[i * n + j]);
                try std.testing.expectEqual(want, got);
            }
        }
    }
}

test "et-soc matmul quant differential: multi-tile int8 store exercises tile-offset math" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // A 20x20 output over K=4 forms a 2x2 output-tile grid: tiles (0,0)=16x16, (0,1)=16x4,
    // (1,0)=4x16, (1,1)=4x4. This is the coverage the single-tile 2x4x4 cases miss: mi>0 and ni>0
    // (so the int8 store's `(mi*TILE+i)*n + ni*TILE` byte offset is actually nonzero on both axes)
    // and a full-width 16-column row (4 words stored per row from the even reg). One K-tile keeps
    // it to 2*2*1 = 4 tile-passes, well under the isel cap.
    const m: u16 = 20;
    const n: u16 = 20;
    const k: u16 = 4;
    const scale_bits: u32 = 0x3F800000; // 1.0
    const quant: MMQuant = .{ .scale = .{ .scalar = scale_bits }, .relu = false };

    // Small bounded inputs so every accumulator stays well within int8 range (no saturation): with
    // A in 0..3 and B in -1..1 over K=4, |acc| <= 3*1*4 = 12. The point of this case is the store
    // and descriptor field math across tiles, not saturation (the single-tile cases cover that).
    var a_buf: [@as(usize, m) * k]i32 = undefined;
    var b_buf: [@as(usize, k) * n]i32 = undefined;
    for (0..m) |i| {
        for (0..k) |kk| a_buf[i * k + kk] = @intCast((i + kk) % 4);
    }
    for (0..k) |kk| {
        for (0..n) |j| b_buf[kk * n + j] = @as(i32, @intCast(j % 3)) - 1;
    }

    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulQuantKernel(&func, m, n, k, .int8, false, quant);
    // Compile runs unconditionally so a broken quant isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .int8, code, &a_buf, &b_buf, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    const scale: f32 = @bitCast(scale_bits);
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.int8, a_buf[i * k + kk]) *% refElem(.int8, b_buf[kk * n + j]);
            // No hand-derived table here: the reference formula IS the oracle for all 400 elements.
            const want = refQuantElem(acc, quant.relu, scale);
            const got: i8 = @bitCast(dump[i * n + j]);
            try std.testing.expectEqual(want, got);
        }
    }
}

test "et-soc matmul quant: structural CSR sequence has tensor_quant, non-quant path does not" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // The GOLDEN quant shape must emit csrw 0x806 (tensor_quant) and a csrw 0x830 (tensor_wait)
    // carrying wait value 10 (TENSOR_WAIT_QUANT).
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulQuantKernel(&func, 2, 4, 4, .int8, false, .{ .scale = .{ .scalar = 0x3F000000 }, .relu = false });
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        try std.testing.expect(hasCsrw(code, encode.CSR_TENSOR_QUANT));
        const waits = try collectCsrwDescriptors(allocator, code, encode.CSR_TENSOR_WAIT);
        defer allocator.free(waits);
        try std.testing.expect(std.mem.indexOfScalar(u64, waits, encode.TENSOR_WAIT_QUANT) != null);
    }

    // The plain int8 matmul (no quant) must NOT emit csrw 0x806 - guards against always-emitting
    // the epilogue regardless of whether a quant is actually attached.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildMatmulKernel(&func, 2, 4, 4, .int8, false);
        const code = try isel.selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        try std.testing.expect(!hasCsrw(code, encode.CSR_TENSOR_QUANT));
    }
}

// `per_column` bakes one fp32 scale per output column as compile-time constant data (`func.
// scaleList`), not a guest-memory buffer, so the image layout is identical to the scalar quant
// harness above (`buildMatmulQuantImage`); only the kernel builder differs (it calls
// `appendMatmulQuantPerColumn` instead of `appendMatmulQuant`).

/// Build a function containing exactly one requantizing `matmul c = quant(a @ b)` with a
/// `per_column` scale. Sibling of `buildMatmulQuantKernel`. `out` picks the requantized output
/// element type (signed int8 or unsigned uint8), threaded straight to `appendMatmulQuantPerColumn`.
fn buildMatmulQuantPerColumnKernel(func: *Function, m: u16, n: u16, k: u16, accumulate: bool, relu: bool, out: ir.function.MatMulQuantOut, scales: []const u32) !void {
    const ptr_t = try func.types.intern(.ptr);
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    try func.appendMatmulQuantPerColumn(blk, pa, pb, pc, m, n, k, .int8, accumulate, relu, out, scales);
    func.setTerminator(blk, .{ .ret = null });
}

test "et-soc matmul quant differential: per_column scale, single tile computes bit-exact requantized int8 output" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // B's columns (constant across k) are 6/3/2/50; A's rows sum to 10 and 26 (same A as the
    // scalar GOLDEN case), so acc(i,j) = col_j * rowsum(i). Scales are powers of two so every
    // acc*scale is an exact integer: col1 (scale 0.5) needs an even acc (rowsum is even, any
    // col1 works); col2 (scale 0.25) needs acc%4==0 (col2=2 is even, and rowsum%4==2, so
    // col2*rowsum%4==0). col3=50 with scale 2.0 saturates both rows, proving saturation still
    // works per-column.
    const scales = [_]u32{ 0x3F800000, 0x3F000000, 0x3E800000, 0x40000000 }; // 1.0, 0.5, 0.25, 2.0
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i32{ 6, 3, 2, 50, 6, 3, 2, 50, 6, 3, 2, 50, 6, 3, 2, 50 };
    const m: u16 = 2;
    const n: u16 = 4;
    const k: u16 = 4;

    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulQuantPerColumnKernel(&func, m, n, k, false, false, .i8, &scales);
    // Compile runs unconditionally, before the sys_emu availability check, so a broken
    // per_column isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .int8, code, &a, &b, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    // Hand-derived sanity table: acc(i,j) = col_j * rowsum(i), rowsum = {10, 26}.
    const expected = [_]i8{ 60, 15, 5, 127, 127, 39, 13, 127 };
    // A wrong broadcast (every column scaled by col0's 1.0) would give column1 = 30 not 15 and
    // column2 = 20 not 5: the columns really are scaled differently, not just column0's value
    // spread across the row.
    try std.testing.expect(expected[1] != 30 and expected[2] != 20);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.int8, a[i * k + kk]) *% refElem(.int8, b[kk * n + j]);
            const scale: f32 = @bitCast(scales[j]);
            const want = refQuantElem(acc, false, scale);
            try std.testing.expectEqual(want, expected[i * n + j]);
            const got: i8 = @bitCast(dump[i * n + j]);
            try std.testing.expectEqual(want, got);
        }
    }
}

test "et-soc matmul quant differential: per_column scale, multi-tile proves scales[ni*TILE+g] indexing" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Same 20x20x4 shape as the scalar multi-tile test: a 2x2 output-tile grid on the column
    // axis (ni=0 covers columns 0..15, ni=1 covers columns 16..19). Scales cycle through FIVE
    // distinct values (cycle.len=5 deliberately does not divide TILE=16), so column g of the
    // ni=1 tile (absolute column 16+g) lands on a DIFFERENT cycle phase than column g of the
    // ni=0 tile ((16+g)%5 != g%5 in general, since 16%5=1). A `ni`-blind bug that always read
    // `scales[g]` instead of `scales[ni*TILE+g]` would therefore apply ni=0's column-g scale to
    // ni=1's column (16+g), which is a numerically different value here (unlike a cycle length
    // that divides 16, where the two would coincidentally agree) - so this test's PASS is by
    // itself proof the tile-relative-to-absolute column mapping is correct, not just a
    // by-inspection argument about the isel code.
    const m: u16 = 20;
    const n: u16 = 20;
    const k: u16 = 4;
    const cycle = [_]u32{ 0x3F800000, 0x3F000000, 0x40000000, 0x3E800000, 0x40800000 }; // 1.0, 0.5, 2.0, 0.25, 4.0
    var scales: [n]u32 = undefined;
    for (0..n) |j| scales[j] = cycle[j % cycle.len];

    // A cycles every row through a permutation of {0,1,2,3} over the k=4 window, so
    // sum_k A[i][k] is always exactly 6 regardless of i; with B constant across k, acc(i,j) is
    // always B[j]*6. B here is 2*((j%3)-1) (doubled vs the scalar multi-tile test) so acc is
    // always a multiple of 12 (-12, 0, or 12): that divides evenly by every scale in `cycle`
    // (including 0.25, giving -3/0/3; 4.0 only ever multiplies, so it can never introduce a
    // fraction), so no acc*scale is ever an exact `.5` tie - the hardware's round-to-nearest-even
    // and the host truncation in `refQuantElem` would disagree on a tie, and this test's job is
    // proving column indexing, not rounding mode. |acc*scale| stays well under 128 (48 at
    // worst), so nothing saturates either (the single-tile case above covers saturation).
    var a_buf: [@as(usize, m) * k]i32 = undefined;
    var b_buf: [@as(usize, k) * n]i32 = undefined;
    for (0..m) |i| {
        for (0..k) |kk| a_buf[i * k + kk] = @intCast((i + kk) % 4);
    }
    for (0..k) |kk| {
        for (0..n) |j| b_buf[kk * n + j] = 2 * (@as(i32, @intCast(j % 3)) - 1);
    }

    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulQuantPerColumnKernel(&func, m, n, k, false, false, .i8, &scales);
    // Compile runs unconditionally so a broken per_column isel path fails here even without
    // sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .int8, code, &a_buf, &b_buf, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.int8, a_buf[i * k + kk]) *% refElem(.int8, b_buf[kk * n + j]);
            // No hand-derived table here (400 elements): the reference formula IS the oracle,
            // with each column's OWN scale, exactly like `refQuantElem` computes on hardware.
            const scale: f32 = @bitCast(scales[j]);
            const want = refQuantElem(acc, false, scale);
            const got: i8 = @bitCast(dump[i * n + j]);
            try std.testing.expectEqual(want, got);
        }
    }
}

test "et-soc matmul quant differential: per_column scale with relu" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // A's rows sum to +10 and -10 (mirrors the scalar relu case); B's columns (constant across
    // k) are 5/-5/20/-4, giving acc(i,j) = col_j * rowsum(i) = col_j * (+-10). col3 is even (-4,
    // not the scalar test's -3) so col3*10 is a multiple of 20, and after the 0.25 scale (the
    // one fractional scale in the cycle) the result is still an exact integer, never a `.5` tie
    // that would disagree with hardware's round-to-nearest-even. Scales now differ per column
    // (1.0/0.5/2.0/0.25) instead of a single broadcast scale, so relu and per-column scaling
    // compose correctly together.
    const scales = [_]u32{ 0x3F800000, 0x3F000000, 0x40000000, 0x3E800000 }; // 1.0, 0.5, 2.0, 0.25
    const a = [_]i32{ 1, 2, 3, 4, -1, -2, -3, -4 };
    const b = [_]i32{ 5, -5, 20, -4, 5, -5, 20, -4, 5, -5, 20, -4, 5, -5, 20, -4 };
    const m: u16 = 2;
    const n: u16 = 4;
    const k: u16 = 4;

    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulQuantPerColumnKernel(&func, m, n, k, false, true, .i8, &scales);
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .int8, code, &a, &b, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.int8, a[i * k + kk]) *% refElem(.int8, b[kk * n + j]);
            const scale: f32 = @bitCast(scales[j]);
            const want = refQuantElem(acc, true, scale);
            const got: i8 = @bitCast(dump[i * n + j]);
            try std.testing.expectEqual(want, got);
        }
    }
}

// `MatMulQuant.out` picks the requantize chain's saturating transform: `.i8` (the default, all
// tests above) emits `.satint8` and clamps to `-128..127`; `.u8` emits `.satuint8` and clamps to
// `0..255`. The packed-byte store (isel.zig, the `has_quant` store loop) is byte-for-byte
// identical either way, so `buildMatmulQuantImage`/`buildMatmulQuantKernel` need no changes: only
// the kernel's `quant.out` (or `buildMatmulQuantPerColumnKernel`'s new `out` param) and the
// reference (`refQuantElemU8` instead of `refQuantElem`) differ.

test "et-soc matmul quant differential: sw-sysemu computes bit-exact requantized uint8 output, saturating both ends" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Same row-sum shape as the i8 "saturation both ways" case (A rows sum to 10 and 14), but B's
    // columns are chosen against the uint8 range 0..255: col0=30 pushes both rows' acc (300, 420)
    // above 255 (clamp to 255), col1=-30 pushes both rows negative (-300, -420; clamp to 0), col2=6
    // stays in range both rows (60, 84, proving no premature clamp), col3=-2 goes negative again
    // (-20, -28; clamp to 0). Scale 1.0 keeps every acc*scale an exact integer.
    const m: u16 = 2;
    const n: u16 = 4;
    const k: u16 = 4;
    const scale_bits: u32 = 0x3F800000; // 1.0
    const a = [_]i32{ 1, 2, 3, 4, 2, 3, 4, 5 };
    const b = [_]i32{ 30, -30, 6, -2, 30, -30, 6, -2, 30, -30, 6, -2, 30, -30, 6, -2 };
    const expected = [_]u8{ 255, 0, 60, 0, 255, 0, 84, 0 };

    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulQuantKernel(&func, m, n, k, .int8, false, .{ .scale = .{ .scalar = scale_bits }, .relu = false, .out = .u8 });
    // Compile runs unconditionally, before the sys_emu availability check, so a broken satuint8
    // isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .int8, code, &a, &b, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    const scale: f32 = @bitCast(scale_bits);
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.int8, a[i * k + kk]) *% refElem(.int8, b[kk * n + j]);
            const want = refQuantElemU8(acc, false, scale);
            try std.testing.expectEqual(want, expected[i * n + j]); // sanity: hand-derived table matches the reference formula
            const got: u8 = dump[i * n + j]; // uint8 output: raw byte IS the value, no bitcast
            try std.testing.expectEqual(want, got);
        }
    }
}

test "et-soc matmul quant differential: per_column scale, uint8 output with a saturating column" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Same A/B/scales as the i8 per_column single-tile case (rowsums 10/26; columns 6/3/2/50 at
    // scales 1.0/0.5/0.25/2.0), but read back as uint8: col3 (2.0 * 50 * rowsum) saturates to 255
    // on BOTH rows (500 and 1300, both far above 255) instead of i8's 127, while col0..col2 stay
    // in range and prove the per-column scale still composes correctly under the u8 clamp.
    const scales = [_]u32{ 0x3F800000, 0x3F000000, 0x3E800000, 0x40000000 }; // 1.0, 0.5, 0.25, 2.0
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i32{ 6, 3, 2, 50, 6, 3, 2, 50, 6, 3, 2, 50, 6, 3, 2, 50 };
    const m: u16 = 2;
    const n: u16 = 4;
    const k: u16 = 4;
    const expected = [_]u8{ 60, 15, 5, 255, 156, 39, 13, 255 };

    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulQuantPerColumnKernel(&func, m, n, k, false, false, .u8, &scales);
    // Compile runs unconditionally, before the sys_emu availability check, so a broken per_column
    // satuint8 isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .int8, code, &a, &b, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.int8, a[i * k + kk]) *% refElem(.int8, b[kk * n + j]);
            const scale: f32 = @bitCast(scales[j]);
            const want = refQuantElemU8(acc, false, scale);
            try std.testing.expectEqual(want, expected[i * n + j]); // sanity: hand-derived table matches the reference formula
            const got: u8 = dump[i * n + j]; // uint8 output: raw byte IS the value, no bitcast
            try std.testing.expectEqual(want, got);
        }
    }
}

// `MatMulQuant.bias` (optional per-column int32) and `.zero_point` (per-tensor int32, 0 =
// symmetric) round out the requantize epilogue for asymmetric quantization: unsigned (uint8)
// activations, a per-column bias folded into the int32 domain before scaling, and a nonzero
// output zero-point added after rounding. The oracle-verified hardware chain order is
// `(bias?) -> (relu?) -> *scale -> round -> (zero_point?) -> sat[u]int8 -> pack` (isel.zig's
// `.matmul` `has_quant` block), reading three consecutive SCP lines in that same order (bias,
// then scale, then zero_point). The reference below folds bias in BEFORE relu and zero_point in
// AFTER rounding, matching that exactly. `buildMatmulQuantSpecKernel`/`buildMatmulQuantImage`
// (both now dtype-parametrized) build the `.uint8`-input kernels these tests need; neither
// `buildMatmulQuantKernel` (fixed `MatMulQuant`, no bias knob reachable without a spec) nor
// `buildMatmulQuantPerColumnKernel` (no bias/zero_point knob at all) can express bias or
// zero_point, so these go through `appendMatmulQuantSpec` via `buildMatmulQuantSpecKernel`.

/// Asymmetric-uint8 reference: `acc = sum_k A[i,k]*B[k,j]` (both inputs read UNSIGNED via
/// `refElem(.uint8, ...)` by the caller), `v = acc + bias`, `relu` clamps `v` to `>= 0`, `v`
/// scales by the fp32 `scale` and rounds to int32 (test data is chosen so this is always exact,
/// never a `.5` tie), `zero_point` is added post-round, and the result saturates to `0..255` (the
/// only output type the asymmetric tests below use).
fn refAsymU8(acc_in: i32, bias: i32, relu: bool, scale: f32, zero_point: i32) u8 {
    const biased: i32 = acc_in +% bias;
    const v: i32 = if (relu) @max(0, biased) else biased;
    const scaled: f32 = @as(f32, @floatFromInt(v)) * scale;
    const r: i32 = @intFromFloat(scaled); // exact by construction: no rounding-mode ambiguity
    const zp_added: i32 = r +% zero_point;
    const sat = std.math.clamp(zp_added, 0, 255);
    return @intCast(sat);
}

test "et-soc matmul quant differential: asymmetric uint8, multi-tile, per-column bias + per-column scale + nonzero zero_point" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // LOAD-BEARING case: a 20x20 output over K=4 forms the same 2x2 output-tile grid as the
    // earlier multi-tile tests (mi in {0,1} covering 16/4 rows, ni in {0,1} covering 16/4 cols),
    // so this exercises `bias[ni*TILE+g]` AND `scale[ni*TILE+g]` indexing at ni>0, plus the
    // 3-line SCP load order (bias@40, scale@41, zero_point@42) end to end. A cycles a
    // permutation of {0,1,2,3} per row (uint8-safe, unsigned) so every row's K-sum is exactly 6
    // regardless of i; B is constant across k, one of 5 unsigned values per 5-column cycle
    // (0/5/10/15/20), so acc(i,j) = 6 * Bcol[j%5] is independent of i. The bias/scale cycles
    // (length 5, which does NOT divide TILE=16) put column (16+g) of the ni=1 tile on a
    // different cycle phase than column g of the ni=0 tile (16%5=1, so (16+g)%5 != g%5 in
    // general), so a `[g]`-instead-of-`[ni*TILE+g]` indexing bug would read the wrong bias/scale
    // for the second tile and produce numerically different (wrong) results - this test's pass
    // is proof of correct indexing, not just inspection. The chosen bias/scale/zero_point values
    // also drive some columns' `acc+bias` above the range that survives `*scale + zero_point`
    // (saturates to 255) and others below zero (clamps to 0 after the zero_point add), alongside
    // in-range columns.
    const m: u16 = 20;
    const n: u16 = 20;
    const k: u16 = 4;
    const scale_cycle = [_]u32{ 0x3F800000, 0x3F000000, 0x40000000, 0x3E800000, 0x40800000 }; // 1.0, 0.5, 2.0, 0.25, 4.0
    const bias_cycle = [_]i32{ -50, 200, -100, 50, -20 };
    const b_cols = [_]i32{ 0, 5, 10, 15, 20 }; // unsigned, uint8-safe
    const zero_point: i32 = 16;

    var a_buf: [@as(usize, m) * k]i32 = undefined;
    var b_buf: [@as(usize, k) * n]i32 = undefined;
    var scales: [n]u32 = undefined;
    var bias: [n]i32 = undefined;
    for (0..m) |i| {
        for (0..k) |kk| a_buf[i * k + kk] = @intCast((i + kk) % 4);
    }
    for (0..k) |kk| {
        for (0..n) |j| b_buf[kk * n + j] = b_cols[j % b_cols.len];
    }
    for (0..n) |j| {
        scales[j] = scale_cycle[j % scale_cycle.len];
        bias[j] = bias_cycle[j % bias_cycle.len];
    }

    var func = Function.init(allocator);
    defer func.deinit();
    const spec: MMQuantSpec = .{ .scale_per_column = &scales, .bias = &bias, .zero_point = zero_point, .relu = false, .out = .u8 };
    try buildMatmulQuantSpecKernel(&func, m, n, k, .uint8, false, spec);
    // Compile runs unconditionally, before the sys_emu availability check, so a broken
    // asymmetric isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .uint8, code, &a_buf, &b_buf, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.uint8, a_buf[i * k + kk]) *% refElem(.uint8, b_buf[kk * n + j]);
            const scale: f32 = @bitCast(scales[j]);
            // No hand-derived table here (400 elements): the reference formula IS the oracle,
            // with each column's OWN bias and scale, exactly like the hardware chain computes.
            const want = refAsymU8(acc, bias[j], false, scale, zero_point);
            const got: u8 = dump[i * n + j]; // uint8 output: raw byte IS the value, no bitcast
            try std.testing.expectEqual(want, got);
        }
    }
}

test "et-soc matmul quant differential: asymmetric uint8, scalar scale + per-column bias + zero_point, single tile" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Proves bias composes with the SCALAR (broadcast) scale path, not just per_column, and
    // exercises the 3-consecutive-SCP-line load order (bias@40, scale@41, zero_point@42) on the
    // smallest possible shape (single tile, no tile-offset indexing to worry about).
    const m: u16 = 2;
    const n: u16 = 4;
    const k: u16 = 4;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // uint8-safe (unsigned), rowsums 10 and 26
    const b = [_]i32{ 10, 1, 5, 0, 10, 1, 5, 0, 10, 1, 5, 0, 10, 1, 5, 0 }; // uint8-safe, constant across k
    const bias = [_]i32{ -100, 50, -30, 20 };
    const scale_bits: u32 = 0x3F000000; // 0.5
    const zero_point: i32 = 16;

    var func = Function.init(allocator);
    defer func.deinit();
    const spec: MMQuantSpec = .{ .scale_scalar = scale_bits, .bias = &bias, .zero_point = zero_point, .relu = false, .out = .u8 };
    try buildMatmulQuantSpecKernel(&func, m, n, k, .uint8, false, spec);
    // Compile runs unconditionally, before the sys_emu availability check, so a broken
    // asymmetric isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .uint8, code, &a, &b, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    const scale: f32 = @bitCast(scale_bits);
    // Hand-derived sanity table: acc(i,j) = Bcol[j] * rowsum(i), rowsums 10/26, then
    // (+bias)->(*0.5, exact)->(+16).
    const expected = [_]u8{ 16, 46, 26, 26, 96, 54, 66, 26 };
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.uint8, a[i * k + kk]) *% refElem(.uint8, b[kk * n + j]);
            const want = refAsymU8(acc, bias[j], false, scale, zero_point);
            try std.testing.expectEqual(want, expected[i * n + j]); // sanity: hand-derived table matches the reference formula
            const got: u8 = dump[i * n + j]; // uint8 output: raw byte IS the value, no bitcast
            try std.testing.expectEqual(want, got);
        }
    }
}

test "et-soc matmul quant differential: asymmetric uint8, per-column bias + zero_point with relu, proves relu sits after bias" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // A's rows sum to 10 and 14 (uint8-safe, unsigned); B is a uniform 5 across every column and
    // every k, so acc(i,j) = 50 / 70 for every column regardless of j. Bias then differentiates
    // the columns: bias[0]=-100 drives acc+bias negative on BOTH rows (-50 / -30), which relu
    // (applied AFTER bias, per the oracle chain order) zeroes; bias[2]=60 keeps both rows well
    // positive (110 / 130), which relu leaves untouched. If relu were (incorrectly) applied
    // BEFORE bias instead, row0 col0 would compute relu(50)=50, then +bias=-50 (never re-relu'd),
    // giving a different final byte than the correct order's zeroed 0 - so this test's pass is
    // proof of the chain ORDER, not just that bias and relu individually work.
    const m: u16 = 2;
    const n: u16 = 4;
    const k: u16 = 4;
    const a = [_]i32{ 1, 2, 3, 4, 2, 3, 4, 5 }; // rowsums 10 and 14
    const b = [_]i32{ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 }; // uniform 5, constant across k
    const bias = [_]i32{ -100, 0, 60, -80 };
    const scale_bits: u32 = 0x3F800000; // 1.0
    const zero_point: i32 = 8;

    var func = Function.init(allocator);
    defer func.deinit();
    const spec: MMQuantSpec = .{ .scale_scalar = scale_bits, .bias = &bias, .zero_point = zero_point, .relu = true, .out = .u8 };
    try buildMatmulQuantSpecKernel(&func, m, n, k, .uint8, false, spec);
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulQuantImage(allocator, .uint8, code, &a, &b, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    const scale: f32 = @bitCast(scale_bits);
    // Hand-derived sanity table: acc=50/70 for every column, then (+bias)->relu->(*1.0)->(+8).
    const expected = [_]u8{ 8, 58, 118, 8, 8, 78, 138, 8 };
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElem(.uint8, a[i * k + kk]) *% refElem(.uint8, b[kk * n + j]);
            const want = refAsymU8(acc, bias[j], true, scale, zero_point);
            try std.testing.expectEqual(want, expected[i * n + j]); // sanity: hand-derived table matches the reference formula
            const got: u8 = dump[i * n + j]; // uint8 output: raw byte IS the value, no bitcast
            try std.testing.expectEqual(want, got);
        }
    }
}

// The showcase: `input_signs` (unsigned uint8 activations x signed int8 weights) COMPOSED with the
// plan-15 asymmetric quant epilogue (per-column bias + per-column scale + nonzero output
// zero-point, unsigned uint8 output) - a real quantized MLP layer's inner loop end to end.

/// Sibling of `buildMatmulMixedImage` for a requantizing mixed-signedness matmul: same
/// per-operand-signed A/B encoding (`writeElemSigned`), but C is `m*n` BYTES (the quant epilogue's
/// requantized output), like `buildMatmulQuantImage` is to `buildMatmulImage`.
fn buildMatmulMixedQuantImage(allocator: std.mem.Allocator, code: []const u32, a: []const i32, b: []const i32, a_unsigned: bool, b_unsigned: bool, m: u16, n: u16, k: u16) std.mem.Allocator.Error!MatmulImage {
    std.debug.assert(a.len == @as(usize, m) * k);
    std.debug.assert(b.len == @as(usize, k) * n);
    const elem = 1; // int8 hardware dtype: always 1 byte, for both mixed operands
    const stub_len: usize = 13;
    const total_code_words = stub_len + code.len;

    const a_off = roundUp(@as(u64, total_code_words) * 4, 64);
    const b_off = roundUp(a_off + @as(u64, m) * k * elem, 64);
    const c_off = roundUp(b_off + @as(u64, k) * n * elem, 64);
    const c_size: u64 = @as(u64, m) * n; // requantized output: one byte per element
    const stack_top_off = roundUp(c_off + c_size + stack_reserve, 16);

    var w: std.ArrayList(u32) = .empty;
    defer w.deinit(allocator);
    try w.append(allocator, encode.lui(.x6, 6)); // 0x6000
    try w.append(allocator, encode.csrrs(.x0, 0x300, .x6)); // mstatus.FS = Dirty (enable FP/VPU)
    try appendPcRel(allocator, &w, .x2, 2, stack_top_off); // sp
    try appendPcRel(allocator, &w, .x10, 4, a_off); // a0 = &A
    try appendPcRel(allocator, &w, .x11, 6, b_off); // a1 = &B
    try appendPcRel(allocator, &w, .x12, 8, c_off); // a2 = &C
    const fn_off: i21 = @intCast((stub_len - w.items.len) * 4);
    try w.append(allocator, encode.jal(.x1, fn_off)); // call the matmul function
    try w.append(allocator, wfi_word); // halt
    try w.append(allocator, encode.jal(.x0, 0)); // safety self-loop
    std.debug.assert(w.items.len == stub_len);
    try w.appendSlice(allocator, code);

    const bytes = try allocator.alloc(u8, stack_top_off);
    @memset(bytes, 0);
    for (w.items, 0..) |word, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    for (0..m) |i| {
        for (0..k) |j| {
            writeElemSigned(bytes[a_off + (i * k + j) * elem ..], a_unsigned, a[i * k + j]);
        }
    }
    for (0..k) |i| {
        for (0..n) |j| {
            writeElemSigned(bytes[b_off + (i * n + j) * elem ..], b_unsigned, b[i * n + j]);
        }
    }
    return .{ .bytes = bytes, .c_off = c_off, .c_size = c_size };
}

test "et-soc matmul quant differential: mixed-signedness asymmetric inference, uint8 activations x int8 weights, multi-tile" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // The full asymmetric-inference shape: UNSIGNED uint8 activations (A) times SIGNED int8
    // weights (B), requantized through the plan-15 quant epilogue (per-column bias + per-column
    // scale + nonzero output zero-point) to an unsigned uint8 output - a real quantized MLP
    // layer. Same 20x20 output over K=4 (2x2 output-tile grid, mi/ni both > 0 on one tile) as the
    // symmetric asymmetric-uint8 showcase above, so this ALSO proves bias[ni*TILE+g] /
    // scale[ni*TILE+g] indexing at ni>0, now under a MIXED dtype. A's cycle includes 200 and 255
    // (both > 127): if isel emitted `tena_unsigned` from the hardware dtype's default (int8 ->
    // both signed) instead of `mmv.input_signs.a_unsigned`, those lanes would sign-extend to
    // -56/-1 instead of zero-extending to 200/255, producing a numerically different (wrong)
    // accumulator that this test's oracle (`refElemSigned`) catches.
    //
    // `a_cycle`'s length equals `k`, so every row's four A elements are the SAME multiset (just
    // rotated by `i`), making each row's sum an identical constant S regardless of i (a rotation
    // of k consecutive residues 0..k-1 through a length-k cycle visits every element exactly
    // once). B is constant across `kk` for a given column, so acc(i,j) = b_col[j] * S is exactly
    // reproducible by hand. S and every bias entry are multiples of 16 (the largest scale
    // denominator used, 1/0.0625), so `(acc+bias)*scale` is always an exact integer for every
    // (i,j): no `.5`-tie rounding-mode ambiguity between this f32 host reference and sw-sysemu's
    // hardware `f32_to_i32` (which rounds ties to even, unlike the plain truncation below).
    const m: u16 = 20;
    const n: u16 = 20;
    const k: u16 = 4;
    const a_cycle = [_]i32{ 200, 255, 41, 0 }; // unsigned uint8, sum S=496=16*31, includes >127 lanes
    const b_col = [_]i32{ -10, 5, -3, 2, 9 }; // signed int8, constant across k per column, spans both signs
    const scale_cycle = [_]u32{ 0x3F800000, 0x3F000000, 0x3E800000, 0x3D800000, 0x3E000000 }; // 1.0, 0.5, 0.25, 0.0625, 0.125
    const bias_cycle = [_]i32{ -160, 800, -256, 480, -720 }; // all multiples of 16
    const zero_point: i32 = 20;

    var a_buf: [@as(usize, m) * k]i32 = undefined;
    var b_buf: [@as(usize, k) * n]i32 = undefined;
    var scales: [n]u32 = undefined;
    var bias: [n]i32 = undefined;
    for (0..m) |i| {
        for (0..k) |kk| a_buf[i * k + kk] = a_cycle[(i + kk) % a_cycle.len];
    }
    for (0..k) |kk| {
        for (0..n) |j| b_buf[kk * n + j] = b_col[j % b_col.len];
    }
    for (0..n) |j| {
        scales[j] = scale_cycle[j % scale_cycle.len];
        bias[j] = bias_cycle[j % bias_cycle.len];
    }

    var func = Function.init(allocator);
    defer func.deinit();
    const input_signs: InputSigns = .{ .a_unsigned = true, .b_unsigned = false };
    const spec: MMQuantSpec = .{ .scale_per_column = &scales, .bias = &bias, .zero_point = zero_point, .relu = false, .out = .u8, .input_signs = input_signs };
    try buildMatmulQuantSpecKernel(&func, m, n, k, .int8, false, spec);
    // Compile runs unconditionally, before the sys_emu availability check, so a broken mixed+quant
    // isel path fails here even without sw-sysemu.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const image = try buildMatmulMixedQuantImage(allocator, code, &a_buf, &b_buf, true, false, m, n, k);
    defer image.deinit(allocator);
    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |kk| acc +%= refElemSigned(true, a_buf[i * k + kk]) *% refElemSigned(false, b_buf[kk * n + j]);
            const scale: f32 = @bitCast(scales[j]);
            // No hand-derived table here (400 elements): the reference formula IS the oracle,
            // with each column's OWN bias and scale, and the per-operand signedness override,
            // exactly like the hardware chain computes.
            const want = refAsymU8(acc, bias[j], false, scale, zero_point);
            const got: u8 = dump[i * n + j]; // uint8 output: raw byte IS the value, no bitcast
            try std.testing.expectEqual(want, got);
        }
    }
}

/// The number of `matmul` IR instructions anywhere in `func`. Structural check that recognition
/// really raised (or, for the rejected non-vacuity case, really did NOT raise) the nest.
fn countIrMatmuls(func: *const Function) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .matmul) count += 1;
        }
    }
    return count;
}

test "matmul_recog: both the raw nest and the recognized function compile, and recognition raises exactly one matmul" {
    // Structural pin for the recognition path. The RAW scalar loop nest raises NO matmul (0), the
    // RECOGNIZED form raises exactly one, and both compile through `selectFunctionForModel`.
    //
    // History: this test used to assert the raw nest FAILED to compile with `error.Unsupported`. That
    // was a symptom of the earlier spill-current integer allocator, which whole-spilled the value it
    // was placing at each exhaustion point, and so eventually spilled one of the nest's loop pointer
    // bases. Emission has no reload path for a spilled load/store base (riscv64/isel.zig: it returns
    // `error.Unsupported` there), so the whole compile failed. The integer allocator is now
    // eviction-based (Belady/furthest-next-use): at an exhaustion point it evicts the active value
    // whose NEXT use is furthest, keeping the hot values resident. The nest's pointer bases are read
    // every loop iteration, so their next use is always near and they are never chosen as the victim.
    // They stay in registers, no spilled base ever reaches emission, and the raw nest now compiles to
    // correct code (the values that DO spill are whole-interval spills emission reloads at every use).
    // The recognized path remains the intended fast path (`apply` orphans the nest, reachability-aware
    // isel lowers only the preheader + `matmul` + `ret`); it is exercised end to end, with sysemu
    // execution, by "matmul_recog differential" below.
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");

    // The raw form is the canonical nest: no recognition, edges split. Eviction keeps its pointer
    // bases resident, so it now compiles.
    var func_raw = Function.init(allocator);
    defer func_raw.deinit();
    try mm.matmul_recog.buildMatmulNest(&func_raw, .{ .m = 2, .n = 4, .k = 3 });
    try std.testing.expectEqual(@as(usize, 0), countIrMatmuls(&func_raw));
    try isel.splitCriticalEdges(allocator, &func_raw);
    const raw_code = try isel.selectFunctionForModel(allocator, &func_raw, model);
    allocator.free(raw_code);

    // The recognized form is the same nest raised to the matmul op (orphaning the nest), edges split. isel now
    // skips the unreachable scaffolding, so only the reachable preheader + `matmul` + `ret` are lowered.
    var func_recog = Function.init(allocator);
    defer func_recog.deinit();
    try mm.matmul_recog.buildMatmulNest(&func_recog, .{ .m = 2, .n = 4, .k = 3 });
    try std.testing.expect(try mm.matmul_recog.run(allocator, &func_recog, model));
    try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func_recog));
    try isel.splitCriticalEdges(allocator, &func_recog);
    const code = try isel.selectFunctionForModel(allocator, &func_recog, model);
    allocator.free(code);
}

test "matmul_recog differential: recognized matmul matches the host reference on sysemu" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Shapes: a small case, a mid case with distinct m/n/k, and the exact 16x16x16 single-tile
    // boundary (ceil(16/16) = 1 on every axis, per matmul_recog's own cap-gate test). N is a
    // multiple of 4 in every case (the isel fma b_cols encoding requirement matmul_recog mirrors).
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 2, .n = 4, .k = 3 },
        .{ .m = 4, .n = 8, .k = 4 },
        .{ .m = 16, .n = 16, .k = 16 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        // raises exactly one `matmul`, and that the matmul's operands are EXACTLY the function's own
        // A/B/C params (entry block params 0/1/2, in that order) with the exact input tile
        // dimensions, fp32, no accumulate - i.e. semantically identical to `matmul(a, b, c, m, n, k,
        // .fp32, false)` over a standalone 3-pointer-param function. ---
        var func_recog = Function.init(allocator);
        defer func_recog.deinit();
        try mm.matmul_recog.buildMatmulNest(&func_recog, .{ .m = m, .n = n, .k = k });
        const raised = try mm.matmul_recog.run(allocator, &func_recog, model);
        try std.testing.expect(raised);
        try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func_recog));
        var diags = try ir.verify.verify(allocator, &func_recog, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());

        var found: ?ir.function.MatMul = null;
        for (0..func_recog.blockCount()) |bi| {
            for (func_recog.blockInsts(@enumFromInt(bi))) |inst| {
                switch (func_recog.opcode(inst)) {
                    .matmul => |mm2| found = mm2,
                    else => {},
                }
            }
        }
        const raised_mm = found orelse return error.TestUnexpectedResult;
        const entry_params = func_recog.blockParams(@as(ir.function.Block, @enumFromInt(0)));
        try std.testing.expectEqual(entry_params[0], raised_mm.a);
        try std.testing.expectEqual(entry_params[1], raised_mm.b);
        try std.testing.expectEqual(entry_params[2], raised_mm.c);
        try std.testing.expectEqual(m, raised_mm.m);
        try std.testing.expectEqual(n, raised_mm.n);
        try std.testing.expectEqual(k, raised_mm.k);
        try std.testing.expectEqual(ir.function.MatMulType.fp32, raised_mm.dtype);
        try std.testing.expectEqual(false, raised_mm.accumulate);

        // preheader block still physically carries the now-orphaned loop nest's dead invariants
        // (zero/m_c/n_c/k_c/facc0, appended before `apply` overwrote the terminator - the IR has no
        // instruction- or block-deletion primitive, per `matmul_recog.zig`'s `apply` doc comment) and
        // the whole loop nest hangs off it, unreachable. `splitCriticalEdges` is the same caller step
        // every other loop-executing harness in this file runs before compiling (harness.zig:287,
        // compressed.zig:95, zicbop_differential.zig:100); this function's reachable entry ends in
        // `matmul` + `ret`, not an `if`, so it is a no-op here, kept only for parity with those
        // harnesses and in case a future recognized shape reintroduces a block-arg edge.
        // `selectFunctionForModel` is reachability-aware (Task 1: riscv64/isel.zig skips blocks
        // unreachable from the entry), so it lowers only the reachable preheader + matmul + ret; the
        // orphaned nest contributes neither register pressure nor code. This is the direct proof that
        // recognition's OWN output runs, not evidence-by-analogy from a separately built kernel. The
        // compile runs unconditionally, before the sys_emu availability check, so a broken matmul-op
        // lowering (or a regression in Task 1's reachability fix) fails this test even when sw-sysemu
        // is not on PATH.
        try isel.splitCriticalEdges(allocator, &func_recog);
        const code = try isel.selectFunctionForModel(allocator, &func_recog, model);
        defer allocator.free(code);

        // Deterministic small integer-valued matrices, so fp32 sums/products are EXACT and the host
        // reference is bit-identical to sw-sysemu's softfloat arithmetic.
        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..m) |i| for (0..k) |j| {
            a[i * k + j] = @intCast((i * k + j) % 7);
        };
        for (0..k) |i| for (0..n) |j| {
            b[i * n + j] = @intCast((i * n + j) % 5);
        };

        const image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Host reference: exact integer matmul in f32. C is real row-major (stride n*4).
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const want: u32 = @bitCast(acc);
                const got: u32 = std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little);
                try std.testing.expectEqual(want, got);
            }
        }
    }
}

test "matmul_recog differential: a MEMORY-accumulator nest raises to accumulate=true and adds A*B onto nonzero C" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Task 2 (memory accumulator): the canonical nest, but the k-loop accumulator is seeded with
    // `load(C[i][j])` (the SAME j-level C pointer the k_exit store writes back), so the nest computes
    // `C += A*B`. Recognition must raise it to matmul(accumulate=true), NOT accumulate=false: the
    // difference is observable ONLY when C starts nonzero, which is exactly the non-vacuity guard below.
    // Same shapes as the fresh-accumulator recognition differential (N a multiple of 4 throughout).
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 2, .n = 4, .k = 3 },
        .{ .m = 4, .n = 8, .k = 4 },
        .{ .m = 16, .n = 16, .k = 16 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        // exactly one matmul that is fp32, the right tile shape, over the function's own A/B/C params,
        // and crucially accumulate == TRUE (the whole point of this test). ---
        var func_recog = Function.init(allocator);
        defer func_recog.deinit();
        try mm.matmul_recog.buildMatmulNest(&func_recog, .{ .m = m, .n = n, .k = k, .mem_accumulate = true });
        const raised = try mm.matmul_recog.run(allocator, &func_recog, model);
        try std.testing.expect(raised);
        try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func_recog));
        var diags = try ir.verify.verify(allocator, &func_recog, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());

        var found: ?ir.function.MatMul = null;
        for (0..func_recog.blockCount()) |bi| {
            for (func_recog.blockInsts(@enumFromInt(bi))) |inst| {
                switch (func_recog.opcode(inst)) {
                    .matmul => |mm2| found = mm2,
                    else => {},
                }
            }
        }
        const raised_mm = found orelse return error.TestUnexpectedResult;
        const entry_params = func_recog.blockParams(@as(ir.function.Block, @enumFromInt(0)));
        try std.testing.expectEqual(entry_params[0], raised_mm.a);
        try std.testing.expectEqual(entry_params[1], raised_mm.b);
        try std.testing.expectEqual(entry_params[2], raised_mm.c);
        try std.testing.expectEqual(m, raised_mm.m);
        try std.testing.expectEqual(n, raised_mm.n);
        try std.testing.expectEqual(k, raised_mm.k);
        try std.testing.expectEqual(ir.function.MatMulType.fp32, raised_mm.dtype);
        try std.testing.expectEqual(true, raised_mm.accumulate); // memory accumulator: C += A*B

        // differential above; see its comment for why the orphaned nest costs nothing). ---
        try isel.splitCriticalEdges(allocator, &func_recog);
        const code = try isel.selectFunctionForModel(allocator, &func_recog, model);
        defer allocator.free(code);

        // Deterministic small integer-valued matrices, so fp32 sums/products stay EXACT.
        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..m) |i| for (0..k) |j| {
            a[i * k + j] = @intCast((i * k + j) % 7);
        };
        for (0..k) |i| for (0..n) |j| {
            b[i * n + j] = @intCast((i * n + j) % 5);
        };

        var image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
        defer image.deinit(allocator);

        // Pre-seed C with distinctive NONZERO integer-valued f32 (1..9). This is the NON-VACUITY guard:
        // a fresh (accumulate=false) raise would OVERWRITE C and drop the seed, failing the assert below.
        const c_base: usize = @intCast(image.c_off);
        const seed = try allocator.alloc(f32, @as(usize, m) * n);
        defer allocator.free(seed);
        for (0..m) |i| for (0..n) |j| {
            const s: f32 = @floatFromInt((i * n + j) % 9 + 1); // 1..9, never zero
            seed[i * n + j] = s;
            std.mem.writeInt(u32, image.bytes[c_base + (i * n + j) * 4 ..][0..4], @bitCast(s), .little);
        };

        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Reference: C_final[i][j] == seed[i][j] + sum_k A[i][k]*B[k][j], bit-exact fp32.
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = seed[i * n + j];
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const got: f32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(@as(u32, @bitCast(acc)), @as(u32, @bitCast(got)));
            }
        }
    }
}

/// Build a SURROUNDED fp32 matmul nest (`fn(A, B, C) void`) that is NOT the whole function: loop-free
/// setup runs before the nest and a live-value continuation runs after it, the shape Task 2 recognition
/// raises as an EMBEDDED matmul. A sentinel f32 V is loaded from `C + m*n*4` in the setup block (BEFORE
/// the preheader), threaded through the outer loop as a LOOP-INVARIANT header param, and stored back in
/// the continuation AFTER the nest, so V is live ACROSS the matmul (only the embedded save/restore keeps
/// it intact). The outer loop also threads its induction variable out, so the continuation exercises BOTH
/// exit-arg reconstruction paths: the iv (reconstructed to `iconst(m)`) and the invariant V (reconstructed
/// to the preheader's initial arg). The continuation stores V back (survival proof) and the reconstructed
/// final `i` (== m) to `C + m*n*4 + 4` (reconstruction proof). Row-major A(m x k)/B(k x n)/C(m x n), unit-
/// stepped element pointers: exactly the layout `buildMatmulImage` writes, so the plain fp32 image runs it.
fn buildSurroundedMatmulNest(func: *Function, m: u16, n: u16, k: u16, mem_accumulate: bool) !void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });

    const setup = try func.appendBlock(); // loop-free setup BEFORE the preheader (loads V)
    const preheader = try func.appendBlock();
    const i_header = try func.appendBlock();
    const i_body = try func.appendBlock();
    const j_header = try func.appendBlock();
    const j_body = try func.appendBlock();
    const k_header = try func.appendBlock();
    const k_body = try func.appendBlock();
    const k_exit = try func.appendBlock();
    const j_exit = try func.appendBlock();
    const cont = try func.appendBlock(); // outer.exit: the continuation AFTER the nest

    const a_base = try func.appendBlockParam(setup, ptr_t);
    const b_base = try func.appendBlockParam(setup, ptr_t);
    const c_base = try func.appendBlockParam(setup, ptr_t);

    const zero = try func.appendInst(setup, i32_t, .{ .iconst = 0 });
    const m_c = try func.appendInst(setup, i32_t, .{ .iconst = m });
    const n_c = try func.appendInst(setup, i32_t, .{ .iconst = n });
    const k_c = try func.appendInst(setup, i32_t, .{ .iconst = k });
    const facc0 = try func.appendInst(setup, f32_t, .{ .fconst = 0.0 });
    // The live-across sentinel V, loaded from the slot just past C, BEFORE the nest runs.
    const vptr = try func.appendArithImm(setup, ptr_t, .add, c_base, @as(i64, m) * n * 4);
    const v = try func.appendInst(setup, f32_t, .{ .load = .{ .ptr = vptr } });
    const b_row: i64 = @as(i64, n) * 4;
    const a_row_stride: i64 = @as(i64, k) * 4;
    try func.setJump(setup, preheader, &.{});

    // Preheader: enter the i-loop with i=0, a_row=A, c_ptr=C, and the invariant sentinel V.
    try func.setJump(preheader, i_header, &.{ zero, a_base, c_base, v });

    // i-loop: carries the A row pointer, the C write pointer, and the loop-invariant V.
    const i = try func.appendBlockParam(i_header, i32_t);
    const a_row = try func.appendBlockParam(i_header, ptr_t);
    const c_ptr = try func.appendBlockParam(i_header, ptr_t);
    const sv = try func.appendBlockParam(i_header, f32_t);
    const cmp_i = try func.appendInst(i_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = m_c } });
    // The exit edge threads out the iv `i` and the invariant `sv`: the two exit-arg reconstruction paths.
    try func.appendIf(i_header, cmp_i, .{ .target = i_body, .args = &.{ i, a_row, c_ptr, sv } }, .{ .target = cont, .args = &.{ i, sv } });

    const bi = try func.appendBlockParam(i_body, i32_t);
    const ib_a_row = try func.appendBlockParam(i_body, ptr_t);
    const ib_c_ptr = try func.appendBlockParam(i_body, ptr_t);
    _ = try func.appendBlockParam(i_body, f32_t); // ib_sv: threaded in by the straight-through in-edge, unused here
    try func.setJump(i_body, j_header, &.{ zero, ib_a_row, b_base, ib_c_ptr });

    const j = try func.appendBlockParam(j_header, i32_t);
    const ja_row = try func.appendBlockParam(j_header, ptr_t);
    const jb_col = try func.appendBlockParam(j_header, ptr_t);
    const jc_ptr = try func.appendBlockParam(j_header, ptr_t);
    const cmp_j = try func.appendInst(j_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n_c } });
    try func.appendIf(j_header, cmp_j, .{ .target = j_body, .args = &.{ j, ja_row, jb_col, jc_ptr } }, .{ .target = j_exit, .args = &.{jc_ptr} });

    const bj = try func.appendBlockParam(j_body, i32_t);
    const jba_row = try func.appendBlockParam(j_body, ptr_t);
    const jbb_col = try func.appendBlockParam(j_body, ptr_t);
    const jbc_ptr = try func.appendBlockParam(j_body, ptr_t);
    // `mem_accumulate`: seed the k-loop accumulator with `load(C[i][j])` (the SAME jbc_ptr k_exit stores
    // back to), so the nest computes C += A*B and the recognizer raises accumulate=true. This exercises
    // the embedded C-preload path (the preload runs inside the embedded save/restore). Off keeps the
    // fresh-accumulator (fconst 0) init, so the existing surrounded test stays byte-identical.
    const k_acc_init = if (mem_accumulate)
        try func.appendInst(j_body, f32_t, .{ .load = .{ .ptr = jbc_ptr } })
    else
        facc0;
    try func.setJump(j_body, k_header, &.{ zero, k_acc_init, jba_row, jbb_col });

    const kk = try func.appendBlockParam(k_header, i32_t);
    const acc = try func.appendBlockParam(k_header, f32_t);
    const a_k = try func.appendBlockParam(k_header, ptr_t);
    const b_k = try func.appendBlockParam(k_header, ptr_t);
    const cmp_k = try func.appendInst(k_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = kk, .rhs = k_c } });
    try func.appendIf(k_header, cmp_k, .{ .target = k_body, .args = &.{ kk, acc, a_k, b_k } }, .{ .target = k_exit, .args = &.{acc} });

    const bk = try func.appendBlockParam(k_body, i32_t);
    const bacc = try func.appendBlockParam(k_body, f32_t);
    const ba_k = try func.appendBlockParam(k_body, ptr_t);
    const bb_k = try func.appendBlockParam(k_body, ptr_t);
    const va = try func.appendInst(k_body, f32_t, .{ .load = .{ .ptr = ba_k } });
    const vb = try func.appendInst(k_body, f32_t, .{ .load = .{ .ptr = bb_k } });
    const prod = try func.appendInst(k_body, f32_t, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const nacc = try func.appendInst(k_body, f32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const nkk = try func.appendArithImm(k_body, i32_t, .add, bk, 1);
    const nba_k = try func.appendArithImm(k_body, ptr_t, .add, ba_k, 4);
    const nbb_k = try func.appendArithImm(k_body, ptr_t, .add, bb_k, b_row);
    try func.setJump(k_body, k_header, &.{ nkk, nacc, nba_k, nbb_k });

    const kacc = try func.appendBlockParam(k_exit, f32_t);
    try func.appendStore(k_exit, kacc, jbc_ptr);
    const nj = try func.appendArithImm(k_exit, i32_t, .add, bj, 1);
    const nb_col = try func.appendArithImm(k_exit, ptr_t, .add, jbb_col, 4);
    const nc_ptr = try func.appendArithImm(k_exit, ptr_t, .add, jbc_ptr, 4);
    try func.setJump(k_exit, j_header, &.{ nj, jba_row, nb_col, nc_ptr });

    const jx_c_ptr = try func.appendBlockParam(j_exit, ptr_t);
    const ni = try func.appendArithImm(j_exit, i32_t, .add, bi, 1);
    const na_row = try func.appendArithImm(j_exit, ptr_t, .add, ib_a_row, a_row_stride);
    // The outer back-edge threads `sv` (the i_header param itself, in scope here since i_header dominates
    // j_exit) straight back UNCHANGED, so V is provably loop-invariant. That is what makes the recognizer's
    // invariance check accept threading V out on the exit edge and reconstruct it to its initial value.
    try func.setJump(j_exit, i_header, &.{ ni, na_row, jx_c_ptr, sv });

    // The continuation AFTER the nest: prove V survived and the reconstructed final i (== m) is right.
    const fi = try func.appendBlockParam(cont, i32_t);
    const fv = try func.appendBlockParam(cont, f32_t);
    try func.appendStore(cont, fv, vptr); // V survived the matmul's clobber (the embedded save/restore)
    const fiptr = try func.appendArithImm(cont, ptr_t, .add, vptr, 4);
    try func.appendStore(cont, fi, fiptr); // the reconstructed outer induction variable, which must equal m
    func.setTerminator(cont, .{ .ret = null });
}

test "matmul_recog differential: a SURROUNDED nest is raised to an embedded matmul, computes C, and preserves a live-across value" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // m >= 4 so TenC (f0..f(2*min(16,m)-1)) covers f0..f7 and V's float register is definitely clobbered;
    // only the embedded save/restore keeps it intact. 4x4x4 is a single output tile (cap 1*1*1 <= 64),
    // N % 4 == 0, K % 1 == 0, so recognition's own gates accept it.
    const m: u16 = 4;
    const n: u16 = 4;
    const k: u16 = 4;

    // Build the surrounded nest, recognize it, and confirm it raised exactly one embedded
    // matmul of the right shape. Compile runs unconditionally (before the sys_emu availability check) so a
    // regression in recognition or the embedded lowering fails here even where sw-sysemu is not on PATH.
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSurroundedMatmulNest(&func, m, n, k, false);
    try std.testing.expect(try mm.matmul_recog.run(allocator, &func, model));
    try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func));
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var raised: ?ir.function.MatMul = null;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| switch (func.opcode(inst)) {
            .matmul => |mm2| raised = mm2,
            else => {},
        };
    }
    const raised_mm = raised orelse return error.TestUnexpectedResult;
    try std.testing.expect(raised_mm.embedded); // surrounded by a live value -> the self-contained lowering
    try std.testing.expect(!raised_mm.accumulate); // fresh (fconst 0) init -> overwrite, not C-accumulate
    try std.testing.expectEqual(m, raised_mm.m);
    try std.testing.expectEqual(n, raised_mm.n);
    try std.testing.expectEqual(k, raised_mm.k);
    try std.testing.expectEqual(ir.function.MatMulType.fp32, raised_mm.dtype);

    // Compile the recognized function itself and run it under sw-sysemu.
    try isel.splitCriticalEdges(allocator, &func);
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const a = try allocator.alloc(i32, @as(usize, m) * k);
    defer allocator.free(a);
    const b = try allocator.alloc(i32, @as(usize, k) * n);
    defer allocator.free(b);
    for (0..m) |i| for (0..k) |j| {
        a[i * k + j] = @intCast((i * k + j) % 7);
    };
    for (0..k) |i| for (0..n) |j| {
        b[i * n + j] = @intCast((i * n + j) % 5);
    };

    var image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
    defer image.deinit(allocator);
    // The live-across sentinel V sits at C + m*n*4 (the slot the kernel loads and stores back); the
    // reconstructed final i (== m) lands at C + m*n*4 + 4. Pre-seed V; widen the dump to cover both.
    const v_slot: usize = @intCast(image.c_off + @as(u64, m) * n * 4);
    std.mem.writeInt(u32, image.bytes[v_slot..][0..4], embedded_v_bits, .little);
    image.c_size = @as(u64, m) * n * 4 + 8; // C, then V, then the reconstructed i

    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    // (a) C is the exact fp32 matmul, byte-for-byte, despite the surrounding embedded code.
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
            const got: u32 = std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little);
            try std.testing.expectEqual(@as(u32, @bitCast(acc)), got);
        }
    }
    // (b) The live-across sentinel V survived the matmul's full register clobber, threaded through the
    // exit-arg reconstruction (V is the invariant outer-header param, reconstructed to its initial value).
    const v_got = std.mem.readInt(u32, dump[@as(usize, m) * n * 4 ..][0..4], .little);
    try std.testing.expectEqual(embedded_v_bits, v_got);
    // (c) The reconstructed outer induction variable equals the bound m (the iv -> iconst(m) path).
    const i_got = std.mem.readInt(u32, dump[@as(usize, m) * n * 4 + 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(u32, m), i_got);
}

test "matmul_recog differential: a SURROUNDED memory-accumulator nest raises an embedded accumulate=true matmul (C += A*B) and preserves a live-across value" {
    // Combines the two Plan-20 + non-whole-function paths that were only tested apart: a nest that is
    // BOTH surrounded (live V across it -> embedded lowering) AND a memory accumulator (acc seeded from
    // load(C) -> accumulate=true). This exercises the C-tile preload running INSIDE the embedded
    // save/restore, the one interaction the two separate differentials did not cover end-to-end.
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    const m: u16 = 4;
    const n: u16 = 4;
    const k: u16 = 4;

    var func = Function.init(allocator);
    defer func.deinit();
    try buildSurroundedMatmulNest(&func, m, n, k, true);
    try std.testing.expect(try mm.matmul_recog.run(allocator, &func, model));
    try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func));
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var raised: ?ir.function.MatMul = null;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| switch (func.opcode(inst)) {
            .matmul => |mm2| raised = mm2,
            else => {},
        };
    }
    const raised_mm = raised orelse return error.TestUnexpectedResult;
    try std.testing.expect(raised_mm.embedded); // surrounded by a live value -> self-contained lowering
    try std.testing.expect(raised_mm.accumulate); // load(C)-seeded accumulator -> C += A*B
    try std.testing.expectEqual(m, raised_mm.m);
    try std.testing.expectEqual(n, raised_mm.n);
    try std.testing.expectEqual(k, raised_mm.k);
    try std.testing.expectEqual(ir.function.MatMulType.fp32, raised_mm.dtype);

    try isel.splitCriticalEdges(allocator, &func);
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    const a = try allocator.alloc(i32, @as(usize, m) * k);
    defer allocator.free(a);
    const b = try allocator.alloc(i32, @as(usize, k) * n);
    defer allocator.free(b);
    for (0..m) |i| for (0..k) |j| {
        a[i * k + j] = @intCast((i * k + j) % 7);
    };
    for (0..k) |i| for (0..n) |j| {
        b[i * n + j] = @intCast((i * n + j) % 5);
    };

    var image = try buildMatmulImage(allocator, .fp32, code, a, b, m, n, k);
    defer image.deinit(allocator);
    // Pre-seed C with nonzero values: the nest computes C += A*B, so a nonzero seed distinguishes
    // accumulate (seed + A*B) from an overwrite lowering (A*B, seed lost). Seed (i*n+j)+1 = 1..16 and
    // the small-integer A/B keep every product/sum exact in f32, so bit-equality is order-independent.
    for (0..m) |i| for (0..n) |j| {
        const s: f32 = @floatFromInt((i * n + j) + 1);
        const slot: usize = @intCast(image.c_off + (i * n + j) * 4);
        std.mem.writeInt(u32, image.bytes[slot..][0..4], @bitCast(s), .little);
    };
    const v_slot: usize = @intCast(image.c_off + @as(u64, m) * n * 4);
    std.mem.writeInt(u32, image.bytes[v_slot..][0..4], embedded_v_bits, .little);
    image.c_size = @as(u64, m) * n * 4 + 8; // C, then V, then the reconstructed i

    const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(dump);

    // (a) C is the seed PLUS the exact fp32 matmul (the accumulate=true C-preload path, run inside the
    // embedded save/restore). An overwrite lowering would drop the seed and fail this.
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = @floatFromInt((i * n + j) + 1); // the C seed
            for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
            const got: u32 = std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little);
            try std.testing.expectEqual(@as(u32, @bitCast(acc)), got);
        }
    }
    // (b) The live-across sentinel V survived the matmul's clobber, and (c) the reconstructed final i == m.
    const v_got = std.mem.readInt(u32, dump[@as(usize, m) * n * 4 ..][0..4], .little);
    try std.testing.expectEqual(embedded_v_bits, v_got);
    const i_got = std.mem.readInt(u32, dump[@as(usize, m) * n * 4 + 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(u32, m), i_got);
}

test "matmul_recog differential: recognized int8/uint8/mixed matmul matches the host reference on sysemu" {
    // Sibling of the fp32 differential above, for the int8-family dtypes plan 19 Task A taught
    // `buildMatmulNest`/recognition to raise: `.int8` (both operands signed), `.uint8` (both
    // unsigned), and `.mixed` (A unsigned x B signed, raised as `.int8` plus the `input_signs`
    // override). Every int8-family case shares ONE image layout regardless of which of the three this
    // is: A/B are 1-byte elements, each operand's OWN signedness (not the nest's nominal dtype) drives
    // its encoding/reference, exactly what `buildMatmulMixedImage`/`refElemSigned` already do for the
    // OP-level mixed differential - so it is reused here unchanged for all three cases.
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    const ElemDtype = @FieldType(mm.matmul_recog.NestSpec, "elem_dtype");
    const Case = struct {
        elem: ElemDtype,
        m: u16,
        n: u16,
        k: u16,
        want_dtype: MMType,
        want_signs: ?InputSigns,
        a_unsigned: bool,
        b_unsigned: bool,
    };
    const cases = [_]Case{
        // int8 (both signed): a sub-K-tile shape, then K=68 spans two 64-int8 SCP lines
        // (k_tiles = ceil(68/64) = 2), per Task A's own K-tiling concern.
        .{ .elem = .int8, .m = 2, .n = 4, .k = 4, .want_dtype = .int8, .want_signs = null, .a_unsigned = false, .b_unsigned = false },
        .{ .elem = .int8, .m = 4, .n = 8, .k = 68, .want_dtype = .int8, .want_signs = null, .a_unsigned = false, .b_unsigned = false },
        // uint8 (both unsigned).
        .{ .elem = .uint8, .m = 4, .n = 4, .k = 4, .want_dtype = .uint8, .want_signs = null, .a_unsigned = true, .b_unsigned = true },
        // mixed: A unsigned uint8 x B signed int8, the asymmetric-inference shape. A's data (below)
        // weaves in values > 127 to prove the tensor unit zero-extends A rather than sign-extending it.
        .{ .elem = .mixed, .m = 4, .n = 8, .k = 4, .want_dtype = .int8, .want_signs = .{ .a_unsigned = true, .b_unsigned = false }, .a_unsigned = true, .b_unsigned = false },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        // raised op carries the right dtype/input_signs BEFORE compiling. This runs unconditionally
        // (no sys_emu dependency), so a dtype/input_signs regression in Task A fails here even without
        // the emulator. ---
        var func_recog = Function.init(allocator);
        defer func_recog.deinit();
        try mm.matmul_recog.buildMatmulNest(&func_recog, .{ .elem_dtype = cse.elem, .m = m, .n = n, .k = k });
        const raised = try mm.matmul_recog.run(allocator, &func_recog, model);
        try std.testing.expect(raised);
        try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func_recog));
        var diags = try ir.verify.verify(allocator, &func_recog, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());

        var found: ?ir.function.MatMul = null;
        for (0..func_recog.blockCount()) |bi| {
            for (func_recog.blockInsts(@enumFromInt(bi))) |inst| {
                switch (func_recog.opcode(inst)) {
                    .matmul => |mm2| found = mm2,
                    else => {},
                }
            }
        }
        const raised_mm = found orelse return error.TestUnexpectedResult;
        const entry_params = func_recog.blockParams(@as(ir.function.Block, @enumFromInt(0)));
        try std.testing.expectEqual(entry_params[0], raised_mm.a);
        try std.testing.expectEqual(entry_params[1], raised_mm.b);
        try std.testing.expectEqual(entry_params[2], raised_mm.c);
        try std.testing.expectEqual(m, raised_mm.m);
        try std.testing.expectEqual(n, raised_mm.n);
        try std.testing.expectEqual(k, raised_mm.k);
        try std.testing.expectEqual(cse.want_dtype, raised_mm.dtype);
        try std.testing.expectEqual(cse.want_signs, raised_mm.input_signs);
        try std.testing.expectEqual(false, raised_mm.accumulate);

        // hand-built stand-in), exactly as the fp32 differential above. `splitCriticalEdges` is a no-op
        // here (the reachable entry ends in `matmul` + `ret`, not an `if`) but kept for parity; the
        // orphaned dead nest below it lowers to nothing (unreachable from the entry). ---
        try isel.splitCriticalEdges(allocator, &func_recog);
        const code = try isel.selectFunctionForModel(allocator, &func_recog, model);
        defer allocator.free(code);

        // Deterministic integer-valued A/B. The unsigned operand of each case gets 200/255/128 woven
        // in (values > 127, exercising the top bit): a sign-extension bug (using the nest's nominal
        // dtype instead of `input_signs`, or vice versa) would corrupt exactly these lanes.
        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..a.len) |idx| {
            a[idx] = if (cse.a_unsigned)
                (switch (idx % 4) {
                    0 => @as(i32, 200),
                    1 => 255,
                    2 => 128,
                    else => @intCast(idx % 200),
                })
            else
                @as(i32, @intCast(idx % 15)) - 7;
        }
        for (0..b.len) |idx| {
            b[idx] = if (cse.b_unsigned)
                (switch (idx % 4) {
                    0 => @as(i32, 200),
                    1 => 255,
                    2 => 128,
                    else => @intCast(idx % 200),
                })
            else
                @as(i32, @intCast(idx % 13)) - 6;
        }

        const image = try buildMatmulMixedImage(allocator, code, a, b, cse.a_unsigned, cse.b_unsigned, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Host reference: exact int32 matmul, each operand sign/zero-extended per its OWN signedness
        // (`refElemSigned`), element (i,j) at (i*n + j)*4 in the dump (int32 output).
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: i32 = 0;
                for (0..k) |kk| acc +%= refElemSigned(cse.a_unsigned, a[i * k + kk]) *% refElemSigned(cse.b_unsigned, b[kk * n + j]);
                const got: i32 = @bitCast(std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little));
                try std.testing.expectEqual(acc, got);
            }
        }
    }
}

test "matmul_recog differential: recognized fp16 matmul matches the host reference on sysemu" {
    // Sibling of the fp32 differential above, for the fp16 dtype the fp16 follow-up taught
    // `buildMatmulNest`/recognition to raise: an f32 accumulator over `convert_f32(load_f16(pa)) *
    // convert_f32(load_f16(pb))`, FLOAT mul/add, 2-byte A/B elements, no `input_signs` (floats have no
    // signedness). Structure mirrors the fp32 and int8/uint8/mixed differentials exactly: structural
    // half (build the loop-nest form, recognize, confirm the raised op), then execution half (compile
    // and run `func_recog` ITSELF through sw-sysemu, checked against a host fp32 reference).
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Shapes: a small sub-K-tile case, the exact fp16 K-tile boundary (K_TILE = 16*factor = 32), and a
    // mid case with distinct m/n/k. N is a multiple of 4 (the isel fma b_cols encoding requirement) and
    // K is a multiple of 2 (the fp16 factor) in every case, matmul_recog's own gates.
    const Case = struct { m: u16, n: u16, k: u16 };
    const cases = [_]Case{
        .{ .m = 2, .n = 4, .k = 4 },
        .{ .m = 4, .n = 4, .k = 32 },
        .{ .m = 4, .n = 8, .k = 6 },
    };
    for (cases) |cse| {
        const m = cse.m;
        const n = cse.n;
        const k = cse.k;

        // raises exactly one `matmul` whose operands are EXACTLY the function's own A/B/C params, with
        // the exact input tile dimensions, dtype .fp16, null input_signs, no accumulate. ---
        var func_recog = Function.init(allocator);
        defer func_recog.deinit();
        try mm.matmul_recog.buildMatmulNest(&func_recog, .{ .elem_dtype = .fp16, .m = m, .n = n, .k = k });
        const raised = try mm.matmul_recog.run(allocator, &func_recog, model);
        try std.testing.expect(raised);
        try std.testing.expectEqual(@as(usize, 1), countIrMatmuls(&func_recog));
        var diags = try ir.verify.verify(allocator, &func_recog, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());

        var found: ?ir.function.MatMul = null;
        for (0..func_recog.blockCount()) |bi| {
            for (func_recog.blockInsts(@enumFromInt(bi))) |inst| {
                switch (func_recog.opcode(inst)) {
                    .matmul => |mm2| found = mm2,
                    else => {},
                }
            }
        }
        const raised_mm = found orelse return error.TestUnexpectedResult;
        const entry_params = func_recog.blockParams(@as(ir.function.Block, @enumFromInt(0)));
        try std.testing.expectEqual(entry_params[0], raised_mm.a);
        try std.testing.expectEqual(entry_params[1], raised_mm.b);
        try std.testing.expectEqual(entry_params[2], raised_mm.c);
        try std.testing.expectEqual(m, raised_mm.m);
        try std.testing.expectEqual(n, raised_mm.n);
        try std.testing.expectEqual(k, raised_mm.k);
        try std.testing.expectEqual(ir.function.MatMulType.fp16, raised_mm.dtype);
        try std.testing.expectEqual(@as(?InputSigns, null), raised_mm.input_signs);
        try std.testing.expectEqual(false, raised_mm.accumulate);

        // hand-built stand-in), exactly as the fp32/int8 differentials above. `splitCriticalEdges` is a
        // no-op here (the reachable entry ends in `matmul` + `ret`, not an `if`) but kept for parity;
        // the orphaned dead nest below it lowers to nothing (unreachable from the entry). ---
        try isel.splitCriticalEdges(allocator, &func_recog);
        const code = try isel.selectFunctionForModel(allocator, &func_recog, model);
        defer allocator.free(code);

        // Deterministic small integer-valued A/B: exactly representable in f16, so the products/sums
        // are exact and the host f32 reference is bit-identical to sw-sysemu's softfloat arithmetic.
        const a = try allocator.alloc(i32, @as(usize, m) * k);
        defer allocator.free(a);
        const b = try allocator.alloc(i32, @as(usize, k) * n);
        defer allocator.free(b);
        for (0..a.len) |idx| a[idx] = @intCast(idx % 7);
        for (0..b.len) |idx| b[idx] = @intCast(idx % 5);

        const image = try buildMatmulImage(allocator, .fp16, code, a, b, m, n, k);
        defer image.deinit(allocator);
        const dump = runMatmulImage(std.testing.io, allocator, image) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        defer allocator.free(dump);

        // Host reference: exact integer matmul accumulated in f32 (C is 32-bit fp32 output, stride n*4).
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0;
                for (0..k) |kk| acc += @as(f32, @floatFromInt(a[i * k + kk])) * @as(f32, @floatFromInt(b[kk * n + j]));
                const want: u32 = @bitCast(acc);
                const got: u32 = std.mem.readInt(u32, dump[(i * n + j) * 4 ..][0..4], .little);
                try std.testing.expectEqual(want, got);
            }
        }
    }
}

test "matmul_recog non-vacuity: a rejected nest is left as loops, unmutated and verify-clean" {
    // `no_product`: the k-loop accumulates a bare A load (`acc += A[i][kk]`), never multiplying by
    // B, so this is NOT a matmul (Task 2's "the accumulate's other operand is a product" gate must
    // reject it). It is otherwise the exact canonical scaffolding, so the rejection proves
    // recognition does not false-accept a same-shaped-but-differently-bodied nest.
    //
    // This does NOT additionally run the rejected nest on sw-sysemu: like the raw matmul nest, ANY
    // nest of this shape carries too many simultaneously-live pointers/counters for the riscv64
    // integer register file (see the register-pressure note above), so it cannot lower to executable
    // code today, independent of what recognition does. What IS proven, and is the actual content of
    // "recognition never corrupts a nest it does not fully prove": the function comes out of `run`
    // byte-for-byte identical (same block/instruction counts, no `matmul` anywhere) and still
    // verifies clean, i.e. recognition took the conservative "leave it as loops" exit rather than
    // mutating anything on the way to refusing.
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    var func = Function.init(allocator);
    defer func.deinit();
    try mm.matmul_recog.buildMatmulNest(&func, .{ .no_product = true });
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();

    const raised = try mm.matmul_recog.run(allocator, &func, model);
    try std.testing.expect(!raised);
    try std.testing.expectEqual(@as(usize, 0), countIrMatmuls(&func));
    // Recognition must leave a rejected nest byte-for-byte untouched: same block/instruction counts.
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}
