//! RISC-V 64-bit link backend for the shared static linker. Ported faithfully from
//! the old `libs/vulcan-target/riscv64/ld.zig`: it owns the RISC-V reloc math, the
//! external-call stub + GOT mechanism, the RVC (C-extension) compression pass, the
//! two-pass hi/lo relocation pairing, and the executable parameters. Depends only on
//! `std` and the generic `elf.zig`; the instruction-encoding and RVC-compression bit
//! math is re-homed here (previously borrowed from `encode.zig`/`compress.zig`) so the
//! linker library stays free of any dependency on `vulcan-target`.

const std = @import("std");
const elf = @import("../elf.zig");

const Error = elf.Error;
const SecKind = elf.SecKind;
const RelocType = elf.RelocType;
const Reloc = elf.Reloc;
const ObjSymbol = elf.ObjSymbol;
const ParsedObject = elf.ParsedObject;
const ResolvedSymbol = elf.ResolvedSymbol;
const Image = elf.Image;
const Resolver = elf.Resolver;
const Segment = elf.Segment;
const SecPlace = elf.SecPlace;
const Placement = elf.Placement;
const alignUp = elf.alignUp;
const findSymbol = elf.findSymbol;
const secIndex = elf.secIndex;
const places_per_object = elf.places_per_object;
const not_placed = elf.not_placed;

/// `EF_RISCV_FLOAT_ABI_DOUBLE`: the `e_flags` bit marking the `lp64d` (hard-double) float
/// ABI a real riscv64 glibc `ld.so` requires of anything it loads.
const EF_RISCV_FLOAT_ABI_DOUBLE: u32 = 0x0004;

/// Executable parameters for wrapping a RISC-V image in a static ELF64 (`ET_EXEC`). The
/// float-ABI `e_flags` is only consumed by the dynamic emitter (the strict `ld.so` path);
/// the static path ignores it (bare `qemu-user`), so this stays byte-identical there.
pub const exec_params: elf.ExecParams = .{ .e_machine = elf.EM_RISCV, .e_flags = EF_RISCV_FLOAT_ABI_DOUBLE };

// --- Minimal RISC-V instruction encoding (re-homed from encode.zig) ---------------

/// A RISC-V integer register, x0 through x31.
const Reg = enum(u5) {
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31,
};

fn num(reg: Reg) u32 {
    return @intFromEnum(reg);
}

fn iType(opcode: u7, funct3: u3, rd_: Reg, rs1_: Reg, imm: i12) u32 {
    return @as(u32, opcode) |
        (num(rd_) << 7) |
        (@as(u32, funct3) << 12) |
        (num(rs1_) << 15) |
        (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

/// `jalr rd, rs1, imm` (jump and link register, `ret` is `jalr x0, ra, 0`).
fn jalr(rd_: Reg, rs1_: Reg, imm: i12) u32 {
    return iType(0b1100111, 0b000, rd_, rs1_, imm);
}

/// `ld rd, imm(rs1)` (load 64-bit).
fn ld(rd_: Reg, rs1_: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b011, rd_, rs1_, imm);
}

/// `auipc rd, imm` (add upper immediate to PC: `rd = pc + (imm << 12)`).
fn auipc(rd_: Reg, imm: u20) u32 {
    return @as(u32, 0b0010111) | (num(rd_) << 7) | (@as(u32, imm) << 12);
}

/// `jal rd, offset` (jump and link). The 21-bit immediate is scattered.
fn jal(rd_: Reg, imm: i21) u32 {
    const u: u32 = @as(u21, @bitCast(imm));
    return @as(u32, 0b1101111) |
        (num(rd_) << 7) |
        (((u >> 12) & 0xff) << 12) | // imm[19:12]
        (((u >> 11) & 1) << 20) | // imm[11]
        (((u >> 1) & 0x3ff) << 21) | // imm[10:1]
        (((u >> 20) & 1) << 31); // imm[20]
}

// --- Relocation appliers (re-homed from ld.zig) -----------------------------------

/// Patch a single relocation into the image. These are all PC-relative, so only
/// image offsets matter. `site` and `target` are both offsets within `code`.
fn applyReloc(code: []u8, site: u64, target: i64, typ: RelocType) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const delta = target - @as(i64, @intCast(site));
    switch (typ) {
        .jal => try patchJal(code, s, delta),
        .pcrel_hi20 => {
            // The high 20 bits of the PC-relative delta go in an `auipc`/`lui`
            // U-immediate (bits 31:12). The +0x800 pre-rounds for the lo12 sign.
            const hi: u32 = @truncate(@as(u64, @bitCast(delta +% 0x800)) >> 12);
            const word = std.mem.readInt(u32, code[s..][0..4], .little);
            std.mem.writeInt(u32, code[s..][0..4], (word & 0x0000_0fff) | (hi << 12), .little);
        },
        .call => {
            // A standard far call: `auipc` at `site` plus `jalr` at `site + 4`,
            // patched together. Reaches a 32-bit (+/-2GiB) PC-relative target.
            try patchCallPair(code, s, delta);
        },
        else => return error.UnsupportedReloc,
    }
}

/// Patch a single `jal` at byte offset `s` in `code` so it reaches a target `delta` bytes
/// away (a signed PC-relative displacement, +/-1MiB range, always even). Keeps the
/// instruction's `rd` field (VCC emits `jal ra, name`, i.e. `rd = ra`, but this preserves
/// whatever `rd` the call used). Both the static `.jal` reloc path (`applyReloc`) and the
/// dynamic call->PLT redirect (`redirectCallNear`) route through here so the bit math is
/// single-sourced, mirroring `patchCallPair` for the `.call` auipc+jalr pair.
fn patchJal(code: []u8, s: usize, delta: i64) Error!void {
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    if (delta < -(1 << 20) or delta >= (1 << 20) or (delta & 1) != 0) return error.RelocationOutOfRange;
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const rd_: Reg = @enumFromInt(@as(u5, @truncate(word >> 7)));
    const patched = jal(rd_, @intCast(delta));
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

/// Patch an `R_RISCV_CALL` `auipc`+`jalr` pair at byte offset `s`: the high 20 bits of
/// `delta` (the PC-relative displacement from the `auipc`) go in the `auipc`'s U-immediate,
/// the low 12 in the `jalr`'s I-immediate (the +0x800 pre-rounds for the lo12 sign). Both
/// the static call-reloc path (`applyReloc`) and the dynamic call->PLT redirect
/// (`redirectCall`) route through here so the bit math is single-sourced.
fn patchCallPair(code: []u8, s: usize, delta: i64) Error!void {
    if (s > code.len or 8 > code.len - s) return error.MalformedObject;
    if (delta < -(1 << 31) or delta >= (1 << 31)) return error.RelocationOutOfRange;
    const hi: u32 = @truncate(@as(u64, @bitCast(delta +% 0x800)) >> 12);
    const lo: u32 = @as(u12, @truncate(@as(u64, @bitCast(delta))));
    const auipc_w = std.mem.readInt(u32, code[s..][0..4], .little);
    std.mem.writeInt(u32, code[s..][0..4], (auipc_w & 0x0000_0fff) | (hi << 12), .little);
    const jalr_w = std.mem.readInt(u32, code[s + 4 ..][0..4], .little);
    std.mem.writeInt(u32, code[s + 4 ..][0..4], (jalr_w & 0x000f_ffff) | (lo << 20), .little);
}

/// Patch a PCREL_LO12_I relocation: the low 12 bits of `pcrel` go in an I-type
/// immediate (bits 31:20). `pcrel` is the delta the paired `auipc` used.
fn patchLo12(code: []u8, site: u64, pcrel: i64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const lo: u32 = @as(u12, @truncate(@as(u64, @bitCast(pcrel))));
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    std.mem.writeInt(u32, code[s..][0..4], (word & 0x000f_ffff) | (lo << 20), .little);
}

/// Each external call gets a 3-instruction stub (`auipc`, `ld`, `jr`) that loads
/// the target's absolute address from a GOT slot and jumps, so a near `jal`/`call`
/// can reach any 64-bit address.
const stub_words = 3;
const stub_bytes = stub_words * 4;

/// An external symbol bound by the resolver: its name and absolute address.
const Extern = struct { name: []const u8, addr: u64 };

fn externIndex(externs: []const Extern, name: []const u8) ?usize {
    for (externs, 0..) |e, i| {
        if (std.mem.eql(u8, e.name, name)) return i;
    }
    return null;
}

/// Assign addresses for a set of parsed RISC-V relocatable objects (the default, no
/// linker-script layout), laid out in the order given with `.text` first at `base`, then
/// `.rodata`, `.data`, the synthesized extern stub/GOT region, and `.bss` (memory only).
/// Unlike the other three backends, riscv64 does real work here beyond region math:
///
///   * When `compress_text` is set, each object's `.text` is RVC-compressed *in place*
///     (mutating `parsed`'s reloc offsets and `.text` symbol values through the shrunk
///     layout map) before the region math sees the compressed lengths. That mutation
///     persists into `applyRelocs`, which reads the remapped offsets/values.
///   * When a `resolver` is given, every undefined symbol referenced by a `jal`/`call`
///     is bound to an absolute address through a synthesized 3-instruction stub
///     (`auipc t0, %pcrel_hi(got)` / `ld t0, %pcrel_lo(got)(t0)` / `jr t0`) plus a u64
///     GOT slot holding the resolver's address. The stubs + GOT bytes are written into
///     the segment's `bytes` at the stub/GOT region, and each stub's address is recorded
///     in `symbols` so a call to that name resolves to its stub (in `applyRelocs`).
///
/// Produces a single loadable `Segment` (the whole image at `base`, R|W|X), a `places`
/// map, and the resolved symbol table (including the stub symbols). No relocations are
/// applied here (that is `applyRelocs`, which pairs the two-pass hi/lo). The caller owns
/// the returned placement; it does not own `parsed`.
pub fn computeDefaultPlacement(allocator: std.mem.Allocator, parsed: []ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool) Error!Placement {
    const nobj = parsed.len;

    // Compress each object's `.text` before layout. Pin every reloc site (call/PC-relative) so the
    // linker's later patches land on intact 32-bit instructions, and remap this object's reloc
    // offsets and `.text` symbol values through the shrunk-layout offset map. The compressed
    // bytes are copied into the segment below; the `owned_texts` buffers are freed once the
    // copy is done (`parsed[oi].text` then dangles but is never read again - `applyRelocs`
    // patches the segment bytes, not `parsed`).
    var owned_texts = try allocator.alloc(?[]u8, nobj);
    for (owned_texts) |*t| t.* = null;
    defer {
        for (owned_texts) |t| if (t) |b| allocator.free(b);
        allocator.free(owned_texts);
    }
    if (compress_text) {
        for (0..nobj) |oi| {
            const p = &parsed[oi];
            const nwords = p.text.len / 4;
            if (nwords == 0) continue;
            const words = try allocator.alloc(u32, nwords);
            defer allocator.free(words);
            for (0..nwords) |k| words[k] = std.mem.readInt(u32, p.text[k * 4 ..][0..4], .little);
            const pins = try allocator.alloc(usize, p.relocs.len);
            defer allocator.free(pins);
            for (p.relocs, 0..) |r, ri| pins[ri] = @intCast(r.offset / 4);
            const offmap = try allocator.alloc(usize, nwords + 1);
            defer allocator.free(offmap);
            const cbytes = try compressPinned(allocator, words, pins, offmap);
            for (p.relocs) |*r| r.offset = offmap[@intCast(r.offset / 4)];
            for (p.symbols) |*s| {
                if (s.section == .text and s.defined) s.value = offmap[@intCast(s.value / 4)];
            }
            owned_texts[oi] = cbytes;
            p.text = cbytes;
        }
    }

    // Pack each object's sections within its region, recording per-object offsets.
    var text_at = try allocator.alloc(u64, nobj);
    defer allocator.free(text_at);
    var rodata_at = try allocator.alloc(u64, nobj);
    defer allocator.free(rodata_at);
    var data_at = try allocator.alloc(u64, nobj);
    defer allocator.free(data_at);
    var bss_at = try allocator.alloc(u64, nobj);
    defer allocator.free(bss_at);
    var text_total: u64 = 0;
    var rodata_total: u64 = 0;
    var data_total: u64 = 0;
    var bss_total: u64 = 0;
    for (0..nobj) |oi| {
        text_at[oi] = alignUp(text_total, 4);
        text_total = text_at[oi] + parsed[oi].text.len;
        rodata_at[oi] = alignUp(rodata_total, 8);
        rodata_total = rodata_at[oi] + parsed[oi].rodata.len;
        data_at[oi] = alignUp(data_total, 8);
        data_total = data_at[oi] + parsed[oi].data.len;
        bss_at[oi] = alignUp(bss_total, 8);
        bss_total = bss_at[oi] + parsed[oi].bss_size;
    }

    const rodata_region = alignUp(text_total, 8);
    const data_region = alignUp(rodata_region + rodata_total, 8);
    var data_end: u64 = text_total;
    if (rodata_total > 0) data_end = rodata_region + rodata_total;
    if (data_total > 0) data_end = data_region + data_total;

    const regionOffset = struct {
        fn f(section: SecKind, oi: usize, ro: u64, da: u64, bs: u64, t_at: []const u64, ro_at: []const u64, da_at: []const u64, bs_at: []const u64) u64 {
            return switch (section) {
                .text => t_at[oi],
                .rodata => ro + ro_at[oi],
                .data => da + da_at[oi],
                .bss => bs + bs_at[oi],
                .undef => unreachable,
            };
        }
    }.f;

    // Resolve each defined, non-local symbol to its final address.
    var symbols: std.ArrayList(ResolvedSymbol) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s.name);
        symbols.deinit(allocator);
    }
    for (0..nobj) |oi| {
        for (parsed[oi].symbols) |sym| {
            // A symbol can be "defined" (st_shndx != UNDEF) yet resolve to no
            // allocatable section: an ABS symbol (SHN_ABS = 0xfff1) or one naming a
            // reserved/non-allocatable section. Such symbols appear in real objects
            // and have no region offset, so skip them rather than hitting the
            // `.undef => unreachable` in regionOffset.
            if (!sym.defined or sym.local or sym.name.len == 0 or sym.section == .undef) continue;
            if (findSymbol(symbols.items, sym.name) != null) return error.DuplicateSymbol;
            const region = regionOffset(sym.section, oi, rodata_region, data_region, alignUp(data_end, 8), text_at, rodata_at, data_at, bss_at);
            const name = try allocator.dupe(u8, sym.name);
            errdefer allocator.free(name);
            try symbols.append(allocator, .{ .name = name, .address = base + region + sym.value, .section = sym.section });
        }
    }

    // Collect the external call targets the resolver can bind: undefined symbols
    // referenced by a `jal`/`call` relocation. Each needs a stub + GOT slot.
    var externs: std.ArrayList(Extern) = .empty;
    defer externs.deinit(allocator);
    if (resolver) |res| {
        for (0..nobj) |oi| {
            for (parsed[oi].relocs) |r| {
                if (r.type != .jal and r.type != .call) continue;
                if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
                const name = parsed[oi].symbols[r.symbol].name;
                if (findSymbol(symbols.items, name) != null) continue; // defined or already a stub
                if (externIndex(externs.items, name) != null) continue;
                const addr = res.resolve(name) orelse continue; // unresolved: surfaced later as UndefinedSymbol
                try externs.append(allocator, .{ .name = name, .addr = addr });
            }
        }
    }

    // Lay out the stub/GOT region after the data, then `.bss` after that.
    const ext_n = externs.items.len;
    const stub_region = alignUp(data_end, 4);
    const got_region = alignUp(stub_region + ext_n * stub_bytes, 8);
    const filesz = if (ext_n > 0) got_region + ext_n * 8 else data_end;
    const bss_region = alignUp(filesz, 8);
    const memsz = if (bss_total > 0) bss_region + bss_total else filesz;

    // The loadable file image: text, rodata, data laid into place (bss is zero, beyond
    // the file image entirely - it lives only in memsz).
    var code = try allocator.alloc(u8, @intCast(filesz));
    errdefer allocator.free(code);
    @memset(code, 0);
    for (0..nobj) |oi| {
        if (parsed[oi].text.len > 0) @memcpy(code[@intCast(text_at[oi])..][0..parsed[oi].text.len], parsed[oi].text);
        if (parsed[oi].rodata.len > 0) @memcpy(code[@intCast(rodata_region + rodata_at[oi])..][0..parsed[oi].rodata.len], parsed[oi].rodata);
        if (parsed[oi].data.len > 0) @memcpy(code[@intCast(data_region + data_at[oi])..][0..parsed[oi].data.len], parsed[oi].data);
    }

    // Synthesize each stub and its GOT slot directly into the segment bytes, and register
    // the stub as the symbol's address so calls route to it (resolved in `applyRelocs`).
    for (externs.items, 0..) |ext, i| {
        const stub_off = stub_region + i * stub_bytes;
        const got_off = got_region + i * 8;
        std.mem.writeInt(u64, code[@intCast(got_off)..][0..8], ext.addr, .little);
        // `auipc t0, %pcrel_hi(got)`, then `ld t0, %pcrel_lo(got)(t0)`, then `jr t0`
        const pcrel: i64 = @as(i64, @intCast(got_off)) - @as(i64, @intCast(stub_off));
        const hi: u20 = @truncate(@as(u64, @bitCast(pcrel +% 0x800)) >> 12);
        const lo: i12 = @bitCast(@as(u12, @truncate(@as(u64, @bitCast(pcrel)))));
        std.mem.writeInt(u32, code[@intCast(stub_off)..][0..4], auipc(.x5, hi), .little);
        std.mem.writeInt(u32, code[@intCast(stub_off + 4)..][0..4], ld(.x5, .x5, lo), .little);
        std.mem.writeInt(u32, code[@intCast(stub_off + 8)..][0..4], jalr(.x0, .x5, 0), .little);
        const name = try allocator.dupe(u8, ext.name);
        errdefer allocator.free(name);
        try symbols.append(allocator, .{ .name = name, .address = base + stub_off });
    }

    // One segment: the whole image at `base` (R|W|X), the bss tail living only in memsz.
    var segments = try allocator.alloc(Segment, 1);
    errdefer allocator.free(segments);
    segments[0] = .{ .vaddr = base, .paddr = base, .bytes = code, .memsz = memsz, .flags = 7 };

    // Record where each (object, section) landed: one segment (index 0), the region
    // offset being both the in-segment offset and (added to `base`) the runtime address.
    // A section with no bytes is not placed. `applyRelocs` reads only the `.text` place.
    var places = try allocator.alloc(SecPlace, nobj * places_per_object);
    errdefer allocator.free(places);
    for (0..nobj) |oi| {
        const present = [places_per_object]bool{
            parsed[oi].text.len > 0,
            parsed[oi].rodata.len > 0,
            parsed[oi].data.len > 0,
            parsed[oi].bss_size > 0,
        };
        inline for (.{ SecKind.text, SecKind.rodata, SecKind.data, SecKind.bss }) |kind| {
            const idx = oi * places_per_object + secIndex(kind);
            if (present[secIndex(kind)]) {
                const off = regionOffset(kind, oi, rodata_region, data_region, bss_region, text_at, rodata_at, data_at, bss_at);
                places[idx] = .{ .vaddr = base + off, .seg = 0, .seg_off = off };
            } else {
                places[idx] = .{ .vaddr = 0, .seg = not_placed, .seg_off = 0 };
            }
        }
    }

    return .{
        .segments = segments,
        .places = places,
        .symbols = try symbols.toOwnedSlice(allocator),
        .entry = 0,
    };
}

/// Apply every relocation into `placement`'s segment bytes, sourcing each site's address
/// and each target from the placement. RISC-V's relocations are all PC-relative, and the
/// PCREL_HI20/PCREL_LO12_I halves must be paired: the lo12's own reloc names the paired
/// `auipc` label (its symbol value is the auipc's `.text` offset), so a first pass resolves
/// every call/hi20 and records each hi20 site's resolved target, and a second pass patches
/// each lo12 against the delta its paired auipc used. The bit math is unchanged from the old
/// single-image path (only the address *source* moved to the placement): every site offset
/// derives from the `.text` place's `seg_off`, and every target delta from `seg.vaddr` (the
/// redundant `SecPlace.vaddr` field is never read - see the aarch64 backend for the
/// heap-layout miscompile that pins this). It does not own `parsed`.
pub fn applyRelocs(allocator: std.mem.Allocator, placement: *Placement, parsed: []ParsedObject) Error!void {
    const nobj = parsed.len;

    // Pass 2a: resolve calls and PCREL_HI20 (sites live in `.text`). Record each
    // hi20 site's resolved target so the paired lo12 recomputes the same delta.
    var hi_target: std.AutoHashMapUnmanaged(u64, i64) = .empty;
    defer hi_target.deinit(allocator);
    for (0..nobj) |oi| {
        const tp = placement.places[oi * places_per_object + secIndex(.text)];
        for (parsed[oi].relocs) |r| {
            if (r.type == .pcrel_lo12_i) continue;
            if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
            if (tp.seg == not_placed or tp.seg >= placement.segments.len) return error.MalformedObject;
            const seg = &placement.segments[tp.seg];
            const target_name = parsed[oi].symbols[r.symbol].name;
            const target_addr = findSymbol(placement.symbols, target_name) orelse return error.UndefinedSymbol;
            // Site is a segment-relative image offset; the target delta is taken against
            // the segment's own `vaddr` (so `delta = target_off - site = target_addr -
            // (seg.vaddr + site)`, the PC-relative displacement). Never read `tp.vaddr`.
            const site = std.math.add(u64, tp.seg_off, r.offset) catch return error.MalformedObject;
            const target_off: i64 = (@as(i64, @intCast(target_addr)) - @as(i64, @intCast(seg.vaddr))) + r.addend;
            try applyReloc(seg.bytes, site, target_off, r.type);
            if (r.type == .pcrel_hi20) try hi_target.put(allocator, site, target_off);
        }
    }

    // Pass 2b: resolve PCREL_LO12 against its paired `auipc` (keyed by the auipc's
    // segment-relative image offset, matching the hi20 site recorded above).
    for (0..nobj) |oi| {
        const tp = placement.places[oi * places_per_object + secIndex(.text)];
        for (parsed[oi].relocs) |r| {
            if (r.type != .pcrel_lo12_i) continue;
            if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
            if (tp.seg == not_placed or tp.seg >= placement.segments.len) return error.MalformedObject;
            const seg = &placement.segments[tp.seg];
            const auipc_site = std.math.add(u64, tp.seg_off, parsed[oi].symbols[r.symbol].value) catch return error.MalformedObject;
            const target_off = hi_target.get(auipc_site) orelse return error.MalformedObject;
            const pcrel = target_off - @as(i64, @intCast(auipc_site));
            const lo12_site = std.math.add(u64, tp.seg_off, r.offset) catch return error.MalformedObject;
            try patchLo12(seg.bytes, lo12_site, pcrel);
        }
    }
}

// --- Dynamic linking (PLT/GOT + eager JUMP_SLOT) ----------------------------------
// The dynamic emitter (`vulcan-link/dynamic.zig`) dispatches to these through its
// architecture-generic `DynArch` vtable. The riscv64 PLT entry REUSES the exact
// `auipc t0, %pcrel_hi(got)` / `ld t0, %pcrel_lo(got)(t0)` / `jr t0` stub the JIT extern
// path synthesizes in `computeDefaultPlacement` (loads a target address from a GOT slot and
// tail-jumps), and the call->PLT redirect reuses the `R_RISCV_CALL` `auipc`+`jalr` patch
// above. The static path never emits any of these.

/// `R_RISCV_JUMP_SLOT`: the dynamic relocation a `.rela.plt` entry carries so a real `ld.so`
/// binds an imported function's `.got.plt` slot to the resolved target address (eager under
/// `DF_BIND_NOW`). The dynamic emitter writes these; the static path never does.
pub const R_RISCV_JUMP_SLOT: u32 = 5;

/// `R_RISCV_64`: the 64-bit absolute dynamic relocation the `.rela.dyn` entry of a DATA import
/// carries so a real `ld.so` writes the imported symbol's runtime address (S + A) into its
/// `.got` slot at load. RISC-V has no distinct `GLOB_DAT` code (unlike aarch64/x86-64), so the
/// GOT `GLOB_DAT` role is filled by `R_RISCV_64` with a nonzero symbol index. The dynamic
/// emitter writes these; the static path never does.
pub const R_RISCV_64: u32 = 2;

/// `R_RISCV_RELATIVE`: the dynamic relocation a PIE/`.so`'s `.rela.dyn` entry carries for an
/// INTERNAL-target data-section pointer init (`int *p = &g;`): a real `ld.so` writes `load_bias
/// + r_addend` (the addend carries the target's base-0 vaddr) into the slot at `r_offset`,
/// with no symbol lookup (`r_info`'s symbol index is 0). A non-PIE `ET_EXEC` has a fixed load
/// address, so the slot is resolved directly at link time instead - this code is PIE/`.so`
/// only. Mirrors `aarch64.R_AARCH64_RELATIVE` / `x86_64.R_X86_64_RELATIVE`.
pub const R_RISCV_RELATIVE: u32 = 3;

/// Patch the `auipc` half (a `R_RISCV_GOT_HI20` site, at image offset `site`, runtime address
/// `site_vaddr`) of a GOT-indirect `auipc`/`ld` data-import pair so it computes the page of the
/// `.got` slot at `got_slot_vaddr`: the high 20 bits of `pcrel = got_slot_vaddr - site_vaddr`
/// go in the `auipc`'s U-immediate (the +0x800 pre-rounds for the paired `ld`'s lo12 sign).
/// This is the GOT analog of the `pcrel_hi20` patch in `applyReloc`, only its target is the GOT
/// slot rather than the symbol's direct address. Used only by the dynamic data-import path.
pub fn patchGotHi20(code: []u8, site: u64, site_vaddr: u64, got_slot_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const pcrel = @as(i64, @intCast(got_slot_vaddr)) - @as(i64, @intCast(site_vaddr));
    const hi: u32 = @truncate(@as(u64, @bitCast(pcrel +% 0x800)) >> 12);
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    std.mem.writeInt(u32, code[s..][0..4], (word & 0x0000_0fff) | (hi << 12), .little);
}

/// Patch the `ld` half (a `R_RISCV_PCREL_LO12_I` site, at image offset `site`, runtime address
/// `site_vaddr`) of a GOT-indirect `auipc`/`ld` data-import pair so it loads the `.got` slot at
/// `got_slot_vaddr`. The `ld` sits exactly one word after its paired `auipc` (isel emits them
/// adjacent), so the auipc's runtime address is `site_vaddr - 4`; the low 12 bits of the SAME
/// `pcrel = got_slot_vaddr - auipc_vaddr` the auipc used go in the `ld`'s I-immediate. This is
/// the `patchLo12` bit math, sourced against the GOT slot. Used only by the dynamic path.
pub fn patchGotLo12(code: []u8, site: u64, site_vaddr: u64, got_slot_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const auipc_vaddr = @as(i64, @intCast(site_vaddr)) - 4;
    const pcrel = @as(i64, @intCast(got_slot_vaddr)) - auipc_vaddr;
    const lo: u32 = @as(u12, @truncate(@as(u64, @bitCast(pcrel))));
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    std.mem.writeInt(u32, code[s..][0..4], (word & 0x000f_ffff) | (lo << 20), .little);
}

/// Encode a 16-byte riscv64 PLT entry that loads the resolved address out of the `.got.plt`
/// slot at `got_slot_vaddr` and tail-jumps to it, given the entry's own runtime address
/// `plt_vaddr`. This is the SAME three-instruction stub the JIT extern path emits (see
/// `computeDefaultPlacement`), only its PC-relative delta now addresses the GOT slot from the
/// PLT entry's own address rather than from a stub inside the code image:
///
///   auipc t0, %pcrel_hi(got)   ; t0 = plt_vaddr + (hi << 12)
///   ld    t0, %pcrel_lo(got)(t0) ; t0 = *(got_slot)  (ld.so wrote the target there)
///   jr    t0                    ; tail-jump (jalr x0, t0, 0)
///
/// `pcrel = got_slot_vaddr - plt_vaddr` (the +0x800 pre-rounds so the lo12's sign matches the
/// hi20). The stub is 12 bytes; bytes [12..16) stay zero as padding to the emitter's 16-byte
/// entry stride (never executed - `jr t0` transfers control away). The GOT slot (8 bytes,
/// initially 0) is filled by the loader under BIND_NOW before `_start` runs. Used only by the
/// dynamic import path; the returned bytes are copied into the `.plt` (an RX segment).
pub fn pltEntry(got_slot_vaddr: u64, plt_vaddr: u64) [16]u8 {
    var e: [16]u8 = @splat(0);
    const pcrel: i64 = @as(i64, @intCast(got_slot_vaddr)) - @as(i64, @intCast(plt_vaddr));
    const hi: u20 = @truncate(@as(u64, @bitCast(pcrel +% 0x800)) >> 12);
    const lo: i12 = @bitCast(@as(u12, @truncate(@as(u64, @bitCast(pcrel)))));
    std.mem.writeInt(u32, e[0..4], auipc(.x5, hi), .little);
    std.mem.writeInt(u32, e[4..8], ld(.x5, .x5, lo), .little);
    std.mem.writeInt(u32, e[8..12], jalr(.x0, .x5, 0), .little);
    return e;
}

/// Redirect the `R_RISCV_CALL` `auipc`+`jalr` pair at image offset `site` (the `auipc`'s
/// runtime address `site_vaddr`) to reach `target_plt_vaddr` (a synthesized PLT entry). The
/// pair is patched together (both the `auipc` hi and the `jalr` lo) against
/// `delta = target_plt_vaddr - site_vaddr`, exactly as the static `R_RISCV_CALL` reloc is
/// applied (`patchCallPair`). A public seam for the dynamic linker's import redirect.
/// `site`/`site_vaddr` name the `auipc` (the import call's reloc offset), as `applyRelocs`
/// sources a CALL site.
pub fn redirectCall(code: []u8, site: u64, site_vaddr: u64, target_plt_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    const delta = @as(i64, @intCast(target_plt_vaddr)) - @as(i64, @intCast(site_vaddr));
    return patchCallPair(code, s, delta);
}

/// Redirect a single `jal` (a near, single-instruction call - VCC's codegen, +/-1MiB reach)
/// at image offset `site` (runtime address `site_vaddr`) to reach `target_plt_vaddr` (a
/// synthesized PLT entry). `delta = target_plt_vaddr - site_vaddr`, patched via the same
/// `jal` bit math (and range check) the static `.jal` reloc uses (`patchJal`). A public seam
/// for the dynamic linker's import redirect, the near-call analog of `redirectCall`.
pub fn redirectCallNear(code: []u8, site: u64, site_vaddr: u64, target_plt_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    const delta = @as(i64, @intCast(target_plt_vaddr)) - @as(i64, @intCast(site_vaddr));
    return patchJal(code, s, delta);
}

// --- RVC compression (re-homed from compress.zig) ---------------------------------
// Only the code paths reachable from `compressPinned` are ported; the disassembly
// round-trip unit tests stay in compress.zig, which is not deleted.

// 32-bit instruction field accessors.
fn opc(w: u32) u32 {
    return w & 0x7f;
}
fn rdf(w: u32) u32 {
    return (w >> 7) & 0x1f;
}
fn f3(w: u32) u32 {
    return (w >> 12) & 7;
}
fn rs1f(w: u32) u32 {
    return (w >> 15) & 0x1f;
}
fn rs2f(w: u32) u32 {
    return (w >> 20) & 0x1f;
}
fn f7(w: u32) u32 {
    return (w >> 25) & 0x7f;
}
fn iimm(w: u32) i32 {
    return @as(i32, @bitCast(w)) >> 20;
}
fn simm(w: u32) i32 {
    const raw: u32 = (f7(w) << 5) | rdf(w);
    return @as(i32, @bitCast(raw << 20)) >> 20;
}

/// A compressed register field is 3 bits selecting x8..x15.
fn crp(r: u32) ?u3 {
    return if (r >= 8 and r <= 15) @intCast(r - 8) else null;
}
fn imm6(v: i32) u16 { // the 6-bit immediate common to c.addi/c.li: imm[5] at 12, imm[4:0] at 6:2
    const u: u16 = @intCast(@as(u32, @bitCast(v)) & 0x3f);
    return ((u >> 5) << 12) | ((u & 0x1f) << 2);
}
/// The base of a CI-format compressed instruction in quadrant 1: funct3 at [15:13], op=01.
fn ciBase(funct3: u16) u16 {
    return (funct3 << 13) | 0b01;
}

/// The RVC halfword for a compressible 32-bit instruction, or null if it cannot compress.
fn tryCompress(w: u32) ?u16 {
    switch (opc(w)) {
        0b0010011 => switch (f3(w)) { // OP-IMM
            0 => { // addi -> c.li / c.mv / c.addi / c.addi16sp
                const d = rdf(w);
                const s = rs1f(w);
                const imm = iimm(w);
                if (d == 0) return null;
                if (s == 0 and imm >= -32 and imm <= 31) // c.li rd, imm
                    return ciBase(0b010) | imm6(imm) | (@as(u16, @intCast(d)) << 7);
                if (imm == 0 and s != 0) // c.mv rd, rs (addi rd, rs, 0)
                    return 0b100_0_00000_00000_10 | (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(s)) << 2);
                if (d == 2 and s == 2 and imm != 0 and imm >= -512 and imm <= 511 and @rem(imm, 16) == 0) { // c.addi16sp
                    const u: u16 = @intCast(@as(u32, @bitCast(imm)) & 0x3ff);
                    return 0b011_0_00010_00000_01 |
                        (((u >> 9) & 1) << 12) | (((u >> 4) & 1) << 6) | (((u >> 6) & 1) << 5) |
                        (((u >> 7) & 3) << 3) | (((u >> 5) & 1) << 2);
                }
                if (d == s and imm != 0 and imm >= -32 and imm <= 31) // c.addi rd, rd, imm
                    return ciBase(0b000) | imm6(imm) | (@as(u16, @intCast(d)) << 7);
                if (s == 2 and imm > 0 and imm <= 1020 and @rem(imm, 4) == 0) { // c.addi4spn rd', sp, nzuimm
                    const dp = crp(d) orelse return null;
                    const u: u16 = @intCast(imm);
                    return (@as(u16, dp) << 2) |
                        (((u >> 4) & 3) << 11) | (((u >> 6) & 0xf) << 7) | (((u >> 2) & 1) << 6) | (((u >> 3) & 1) << 5);
                }
                return null;
            },
            1 => { // slli -> c.slli
                const d = rdf(w);
                const shamt = (w >> 20) & 0x3f;
                if (d == 0 or d != rs1f(w) or shamt == 0) return null;
                return 0b000_0000000000_10 | (@as(u16, @intCast(shamt >> 5)) << 12) |
                    (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(shamt & 0x1f)) << 2);
            },
            5 => { // srli / srai -> c.srli / c.srai (MISC-ALU, compressed regs)
                const d = rdf(w);
                if (d != rs1f(w)) return null;
                const dp = crp(d) orelse return null;
                const shamt = (w >> 20) & 0x3f;
                if (shamt == 0) return null;
                const sub: u16 = if (f7(w) == 0b0100000) 0b01 else 0b00; // srai vs srli
                return 0b100_0_00_000_00000_01 | (sub << 10) | (@as(u16, dp) << 7) |
                    (@as(u16, @intCast(shamt >> 5)) << 12) | (@as(u16, @intCast(shamt & 0x1f)) << 2);
            },
            7 => { // andi -> c.andi (MISC-ALU sub=10, compressed regs)
                const d = rdf(w);
                const imm = iimm(w);
                if (d != rs1f(w) or imm < -32 or imm > 31) return null;
                const dp = crp(d) orelse return null;
                return 0b100_0_10_000_00000_01 | (@as(u16, dp) << 7) | imm6(imm);
            },
            else => return null,
        },
        0b0011011 => { // addiw -> c.addiw
            if (f3(w) != 0) return null;
            const d = rdf(w);
            const imm = iimm(w);
            if (d == 0 or d != rs1f(w) or imm < -32 or imm > 31) return null;
            return ciBase(0b001) | imm6(imm) | (@as(u16, @intCast(d)) << 7);
        },
        0b0110111 => { // lui -> c.lui (rd not x0/x2, nonzero 6-bit signed field)
            const d = rdf(w);
            if (d == 0 or d == 2) return null;
            const imm20: i32 = @as(i32, @bitCast(w & 0xffff_f000)) >> 12; // sign-extended 20-bit field
            if (imm20 == 0 or imm20 < -32 or imm20 > 31) return null; // must fit the 6-bit c.lui field
            const field: u16 = @intCast(@as(u32, @bitCast(imm20)) & 0x3f);
            return (0b011 << 13) | 0b01 | (@as(u16, @intCast(d)) << 7) |
                (((field >> 5) & 1) << 12) | ((field & 0x1f) << 2);
        },
        0b0110011 => { // OP: add -> c.add / c.mv, and sub/xor/or/and -> MISC-ALU (compressed regs)
            const d = rdf(w);
            const s1 = rs1f(w);
            const s2 = rs2f(w);
            if (f3(w) == 0 and f7(w) == 0) { // add
                if (s1 == 0 and s2 != 0 and d != 0) // c.mv rd, rs2
                    return 0b100_0_00000_00000_10 | (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(s2)) << 2);
                if (d == s1 and s2 != 0) // c.add rd, rs2
                    return 0b100_1_00000_00000_10 | (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(s2)) << 2);
                return null;
            }
            // sub/xor/or/and rd', rd', rs2' with all three in x8..x15.
            if (d != s1) return null;
            const dp = crp(d) orelse return null;
            const sp = crp(s2) orelse return null;
            const sub2: u16 = switch (f3(w)) {
                0 => if (f7(w) == 0b0100000) 0b00 else return null, // sub
                4 => 0b01, // xor
                6 => 0b10, // or
                7 => 0b11, // and
                else => return null,
            };
            return 0b100_0_11_000_00_000_01 | (@as(u16, dp) << 7) | (sub2 << 5) | (@as(u16, sp) << 2);
        },
        0b1100111 => { // jalr -> c.jr / c.jalr (offset 0)
            if (iimm(w) != 0) return null;
            const d = rdf(w);
            const s = rs1f(w);
            if (s == 0) return null;
            if (d == 0) return 0b100_0_00000_00000_10 | (@as(u16, @intCast(s)) << 7); // c.jr rs
            if (d == 1) return 0b100_1_00000_00000_10 | (@as(u16, @intCast(s)) << 7); // c.jalr rs
            return null;
        },
        0b0000011 => return switch (f3(w)) { // loads
            2 => loadSp(w, false) orelse loadCrp(w, false), // lw
            3 => loadSp(w, true) orelse loadCrp(w, true), // ld
            else => null,
        },
        0b0100011 => return switch (f3(w)) { // stores
            2 => storeSp(w, false) orelse storeCrp(w, false), // sw
            3 => storeSp(w, true) orelse storeCrp(w, true), // sd
            else => null,
        },
        0b0000111 => return switch (f3(w)) { // load-fp: fld -> c.fldsp / c.fld (double only, c.flw is RV32)
            3 => fldSp(w) orelse fldCrp(w),
            else => null,
        },
        0b0100111 => return switch (f3(w)) { // store-fp: fsd -> c.fsdsp / c.fsd
            3 => fsdSp(w) orelse fsdCrp(w),
            else => null,
        },
        else => return null,
    }
}

fn fldSp(w: u32) ?u16 { // c.fldsp: any fp rd, sp base, uimm[5|4:3|8:6] scale 8
    if (rs1f(w) != 2) return null;
    const off = iimm(w);
    if (off < 0 or off > 511 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b001 << 13) | 0b10 | (@as(u16, @intCast(rdf(w))) << 7) |
        (((u >> 5) & 1) << 12) | (((u >> 3) & 3) << 5) | (((u >> 6) & 7) << 2);
}
fn fldCrp(w: u32) ?u16 { // c.fld: fp rd' + int base both x8..15, uimm[5:3|7:6] scale 8
    const dp = crp(rdf(w)) orelse return null;
    const bp = crp(rs1f(w)) orelse return null;
    const off = iimm(w);
    if (off < 0 or off > 255 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b001 << 13) | (@as(u16, dp) << 2) | (@as(u16, bp) << 7) |
        (((u >> 3) & 7) << 10) | (((u >> 6) & 3) << 5);
}
fn fsdSp(w: u32) ?u16 { // c.fsdsp: any fp rs2, sp base, uimm[5:3|8:6] scale 8
    if (rs1f(w) != 2) return null;
    const off = simm(w);
    if (off < 0 or off > 511 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b101 << 13) | 0b10 | (@as(u16, @intCast(rs2f(w))) << 2) |
        (((u >> 3) & 7) << 10) | (((u >> 6) & 7) << 7);
}
fn fsdCrp(w: u32) ?u16 { // c.fsd: fp rs2' + int base both x8..15, uimm[5:3|7:6] scale 8
    const sp2 = crp(rs2f(w)) orelse return null;
    const bp = crp(rs1f(w)) orelse return null;
    const off = simm(w);
    if (off < 0 or off > 255 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b101 << 13) | (@as(u16, sp2) << 2) | (@as(u16, bp) << 7) |
        (((u >> 3) & 7) << 10) | (((u >> 6) & 3) << 5);
}

fn loadSp(w: u32, dbl: bool) ?u16 {
    if (rs1f(w) != 2) return null; // sp base
    const d = rdf(w);
    if (d == 0) return null;
    const off = iimm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.ldsp: uimm[5|4:3|8:6], scale 8
        if (off > 511 or @rem(off, 8) != 0) return null;
        return 0b011_0_00000_00000_10 | (((u >> 5) & 1) << 12) | (@as(u16, @intCast(d)) << 7) |
            (((u >> 3) & 3) << 5) | (((u >> 6) & 7) << 2);
    } else { // c.lwsp: uimm[5|4:2|7:6], scale 4
        if (off > 255 or @rem(off, 4) != 0) return null;
        return 0b010_0_00000_00000_10 | (((u >> 5) & 1) << 12) | (@as(u16, @intCast(d)) << 7) |
            (((u >> 2) & 7) << 4) | (((u >> 6) & 3) << 2);
    }
}

fn storeSp(w: u32, dbl: bool) ?u16 {
    if (rs1f(w) != 2) return null;
    const s2 = rs2f(w);
    const off = simm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.sdsp: uimm[5:3|8:6]
        if (off > 511 or @rem(off, 8) != 0) return null;
        return 0b111_000000_00000_10 | (((u >> 3) & 7) << 10) | (((u >> 6) & 7) << 7) | (@as(u16, @intCast(s2)) << 2);
    } else { // c.swsp: uimm[5:2|7:6]
        if (off > 255 or @rem(off, 4) != 0) return null;
        return 0b110_000000_00000_10 | (((u >> 2) & 0xf) << 9) | (((u >> 6) & 3) << 7) | (@as(u16, @intCast(s2)) << 2);
    }
}

fn loadCrp(w: u32, dbl: bool) ?u16 {
    const dp = crp(rdf(w)) orelse return null;
    const bp = crp(rs1f(w)) orelse return null;
    const off = iimm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.ld: uimm[5:3|7:6], scale 8
        if (off > 255 or @rem(off, 8) != 0) return null;
        return 0b011_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 6) & 3) << 5) | (@as(u16, dp) << 2);
    } else { // c.lw: uimm[5:3|2|6], scale 4
        if (off > 127 or @rem(off, 4) != 0) return null;
        return 0b010_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 2) & 1) << 6) | (((u >> 6) & 1) << 5) | (@as(u16, dp) << 2);
    }
}

fn storeCrp(w: u32, dbl: bool) ?u16 {
    const sp2 = crp(rs2f(w)) orelse return null;
    const bp = crp(rs1f(w)) orelse return null;
    const off = simm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.sd
        if (off > 255 or @rem(off, 8) != 0) return null;
        return 0b111_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 6) & 3) << 5) | (@as(u16, sp2) << 2);
    } else { // c.sw
        if (off > 127 or @rem(off, 4) != 0) return null;
        return 0b110_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 2) & 1) << 6) | (((u >> 6) & 1) << 5) | (@as(u16, sp2) << 2);
    }
}

/// The compressed control-transfer form a 32-bit word can take (or `none`).
const CForm = enum { none, cj, beqz, bnez };

/// Classify a control transfer by form alone (registers, not displacement).
fn classifyBranch(w: u32) struct { form: CForm, reg: u3 } {
    if (isJal(w)) return .{ .form = if (rdf(w) == 0) .cj else .none, .reg = 0 }; // c.jal is RV32-only
    // B-type: c.beqz/c.bnez compare one x8..15 register against x0 (beq/bne are symmetric).
    const funct = f3(w);
    if (funct != 0 and funct != 1) return .{ .form = .none, .reg = 0 };
    const other: u32 = if (rs2f(w) == 0 and rs1f(w) != 0) rs1f(w) else if (rs1f(w) == 0 and rs2f(w) != 0) rs2f(w) else return .{ .form = .none, .reg = 0 };
    const cr = crp(other) orelse return .{ .form = .none, .reg = 0 };
    return .{ .form = if (funct == 0) .beqz else .bnez, .reg = cr };
}

/// Whether `disp` (bytes, always even here) fits a given compressed form's signed range.
fn branchFits(form: CForm, disp: i32) bool {
    return switch (form) {
        .none => false,
        .cj => disp >= -2048 and disp <= 2046, // CJ imm[11:1]
        .beqz, .bnez => disp >= -256 and disp <= 254, // CB imm[8:1]
    };
}

/// The c.j halfword (jal x0) for byte displacement `disp`.
fn encCJ(disp: i32) u16 {
    const u: u32 = @bitCast(disp);
    var h: u16 = (0b101 << 13) | 0b01;
    h |= @intCast(((u >> 11) & 1) << 12);
    h |= @intCast(((u >> 4) & 1) << 11);
    h |= @intCast(((u >> 8) & 3) << 9);
    h |= @intCast(((u >> 10) & 1) << 8);
    h |= @intCast(((u >> 6) & 1) << 7);
    h |= @intCast(((u >> 7) & 1) << 6);
    h |= @intCast(((u >> 1) & 7) << 3);
    h |= @intCast(((u >> 5) & 1) << 2);
    return h;
}

/// The c.beqz/c.bnez halfword for register `reg` (x8..15 as 0..7) and byte displacement `disp`.
fn encCB(form: CForm, reg: u3, disp: i32) u16 {
    const u: u32 = @bitCast(disp);
    var h: u16 = (@as(u16, if (form == .beqz) 0b110 else 0b111) << 13) | 0b01;
    h |= @as(u16, reg) << 7;
    h |= @intCast(((u >> 8) & 1) << 12);
    h |= @intCast(((u >> 3) & 3) << 10);
    h |= @intCast(((u >> 6) & 3) << 5);
    h |= @intCast(((u >> 1) & 3) << 3);
    h |= @intCast(((u >> 5) & 1) << 2);
    return h;
}

fn isJal(w: u32) bool {
    return opc(w) == 0b1101111;
}
fn isBranch(w: u32) bool {
    return opc(w) == 0b1100011;
}
fn jimm(w: u32) i32 {
    const u = (((w >> 31) & 1) << 20) | (((w >> 12) & 0xff) << 12) | (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3ff) << 1);
    return @as(i32, @bitCast(u << 11)) >> 11;
}
fn bimm(w: u32) i32 {
    const u = (((w >> 31) & 1) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 0x3f) << 5) | (((w >> 8) & 0xf) << 1);
    return @as(i32, @bitCast(u << 19)) >> 19;
}

/// Replace the J-type immediate of `w` with `disp` (bytes), keeping its rd field.
fn setJalImm(w: u32, disp: i32) u32 {
    const base = w & 0x00000fff;
    const u: u32 = @bitCast(disp);
    const imm = (((u >> 20) & 1) << 31) | (((u >> 1) & 0x3ff) << 21) | (((u >> 11) & 1) << 20) | (((u >> 12) & 0xff) << 12);
    return base | imm;
}

/// Replace the B-type immediate of `w` with `disp` (bytes), keeping rs1/rs2/funct3.
fn setBranchImm(w: u32, disp: i32) u32 {
    const base = w & 0x01ff_f07f;
    const u: u32 = @bitCast(disp);
    const imm = (((u >> 12) & 1) << 31) | (((u >> 5) & 0x3f) << 25) | (((u >> 1) & 0xf) << 8) | (((u >> 11) & 1) << 7);
    return base | imm;
}

/// Map an original byte offset to the compressed layout.
fn mapTarget(offs: []const usize, n: usize, target_old: usize) usize {
    const old_end = n * 4;
    if (target_old < old_end and target_old % 4 == 0) return offs[target_old / 4];
    return offs[n] + (target_old - old_end);
}

/// A resolved PC-relative `auipc` + companion pair (word indices `hi` and `lo`) whose
/// combined target is byte offset `target` in the original 32-bit layout.
const PcrelPair = struct { hi: usize, lo: usize, target: usize };

/// The linker primitive: compress `code`, keeping every word index in `pinned` verbatim and 32-bit,
/// while still compressing everything else and recomputing purely-internal branch displacements.
/// Writes the new byte offset of each old word index into `out_offsets` (length `code.len + 1`).
fn compressPinned(allocator: std.mem.Allocator, code: []const u32, pinned: []const usize, out_offsets: []usize) std.mem.Allocator.Error![]u8 {
    std.debug.assert(out_offsets.len == code.len + 1);
    return compressCore(allocator, code, &.{}, pinned, out_offsets);
}

fn compressCore(allocator: std.mem.Allocator, code: []const u32, pairs: []const PcrelPair, pinned: []const usize, out_offsets: ?[]usize) std.mem.Allocator.Error![]u8 {
    const n = code.len;

    const lo_pinned = try allocator.alloc(bool, n);
    defer allocator.free(lo_pinned);
    @memset(lo_pinned, false);
    for (pairs) |p| lo_pinned[p.lo] = true;
    const pin = try allocator.alloc(bool, n);
    defer allocator.free(pin);
    @memset(pin, false);
    for (pinned) |idx| pin[idx] = true;
    const half = try allocator.alloc(?u16, n);
    defer allocator.free(half);
    const form = try allocator.alloc(CForm, n);
    defer allocator.free(form);
    const creg_ = try allocator.alloc(u3, n);
    defer allocator.free(creg_);
    const target = try allocator.alloc(usize, n);
    defer allocator.free(target);
    const mappable = try allocator.alloc(bool, n);
    defer allocator.free(mappable);
    const small = try allocator.alloc(bool, n);
    defer allocator.free(small);

    for (code, 0..) |w, i| {
        form[i] = .none;
        mappable[i] = false;
        creg_[i] = 0;
        target[i] = i;
        if (pin[i]) {
            half[i] = null; // emitted verbatim, the external linker owns this site's immediate
            small[i] = false;
        } else if (isJal(w) or isBranch(w)) {
            half[i] = null;
            const disp0 = if (isJal(w)) jimm(w) else bimm(w);
            const ti: i64 = @as(i64, @intCast(i)) + @divTrunc(disp0, 4);
            if (ti >= 0 and ti <= @as(i64, @intCast(n))) {
                mappable[i] = true;
                target[i] = @intCast(ti);
                const c = classifyBranch(w);
                form[i] = c.form;
                creg_[i] = c.reg;
            }
            small[i] = form[i] != .none; // optimistic: assume it compresses
        } else if (lo_pinned[i]) {
            half[i] = null; // pcrel_lo12 site: keep 32-bit so its low immediate stays patchable
            small[i] = false;
        } else {
            half[i] = tryCompress(w);
            small[i] = half[i] != null;
        }
    }

    const offs = try allocator.alloc(usize, n + 1);
    defer allocator.free(offs);
    const sizeOf = struct {
        fn f(hlf: []const ?u16, sml: []const bool, i: usize) usize {
            return if (hlf[i] != null or sml[i]) 2 else 4;
        }
    }.f;

    // Relaxation: recompute offsets, expand any compressed transfer whose displacement no longer
    // fits, and repeat. Sizes only grow, so this converges.
    while (true) {
        var off: usize = 0;
        for (0..n) |i| {
            offs[i] = off;
            off += sizeOf(half, small, i);
        }
        offs[n] = off;
        var changed = false;
        for (0..n) |i| {
            if (form[i] != .none and small[i]) {
                const disp: i32 = @intCast(@as(i64, @intCast(offs[target[i]])) - @as(i64, @intCast(offs[i])));
                if (!branchFits(form[i], disp)) {
                    small[i] = false;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    if (out_offsets) |dst| @memcpy(dst, offs);

    const patched = try allocator.alloc(?u32, n);
    defer allocator.free(patched);
    @memset(patched, null);
    for (pairs) |p| {
        const new_hi = offs[p.hi];
        const new_target = mapTarget(offs, n, p.target);
        const pcrel: i64 = @as(i64, @intCast(new_target)) - @as(i64, @intCast(new_hi));
        const u: u64 = @bitCast(pcrel);
        const hi20: u20 = @truncate((u +% 0x800) >> 12);
        const lo12: u12 = @truncate(u);
        patched[p.hi] = (code[p.hi] & 0x0000_0fff) | (@as(u32, hi20) << 12); // auipc: keep rd, set imm20
        patched[p.lo] = (code[p.lo] & 0x000f_ffff) | (@as(u32, lo12) << 20); // I-type: keep rd/rs1/f3/op, set imm
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (code, 0..) |w, i| {
        if (half[i]) |h| {
            try out.append(allocator, @intCast(h & 0xff));
            try out.append(allocator, @intCast(h >> 8));
            continue;
        }
        if (patched[i]) |pw| { // a recomputed pcrel hi/lo site (always 32-bit)
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, pw, .little);
            try out.appendSlice(allocator, &buf);
            continue;
        }
        if ((isJal(w) or isBranch(w)) and mappable[i]) {
            const disp: i32 = @intCast(@as(i64, @intCast(offs[target[i]])) - @as(i64, @intCast(offs[i])));
            if (small[i]) {
                const h = switch (form[i]) {
                    .cj => encCJ(disp),
                    .beqz, .bnez => encCB(form[i], creg_[i], disp),
                    .none => unreachable,
                };
                try out.append(allocator, @intCast(h & 0xff));
                try out.append(allocator, @intCast(h >> 8));
                continue;
            }
            const word = if (isJal(w)) setJalImm(w, disp) else setBranchImm(w, disp);
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, word, .little);
            try out.appendSlice(allocator, &buf);
            continue;
        }
        // A non-branch that did not compress, or an unmappable transfer: emit unchanged.
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, w, .little);
        try out.appendSlice(allocator, &buf);
    }
    return out.toOwnedSlice(allocator);
}

test "compressPinned keeps pinned sites verbatim and reports the shrunk offset map" {
    const a = std.testing.allocator;
    // c.li-able + PINNED jal + PINNED addi + ret. Mirrors the compress.zig unit test, but here
    // proves the re-homed primitive behaves identically without any disassembler dependency.
    const code = [_]u32{
        0x00500513, // addi x10, x0, 5 -> c.li
        0x000000ef, // jal x1, 0 (PINNED call site)
        0x00358593, // addi x11, x11, 3 (PINNED)
        0x00008067, // jalr x0, x1, 0 (ret) -> 2 bytes
    };
    var offs: [5]usize = undefined;
    const bytes = try compressPinned(a, &code, &.{ 1, 2 }, &offs);
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 12), bytes.len);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 6, 10, 12 }, &offs);
    try std.testing.expectEqual(@as(u32, 0x000000ef), std.mem.readInt(u32, bytes[2..6], .little));
    try std.testing.expectEqual(@as(u32, 0x00358593), std.mem.readInt(u32, bytes[6..10], .little));
}

test "computeDefaultPlacement yields exactly one segment at base (vaddr==paddr==base, R|W|X)" {
    // The riscv64 default layout is always a single loadable segment holding the whole
    // image (text/rodata/data + any synthesized stub/GOT + bss tail) mapped at `base`.
    // `buildImageViaPlacement` relies on `segments[0]` being the whole image, so pin the
    // invariant here directly on a trivial one-object placement.
    const allocator = std.testing.allocator;
    const base: u64 = 0x80000000;

    var text = [_]u8{ 0x67, 0x80, 0x00, 0x00 }; // jalr x0, x1, 0 (ret)
    var syms = [_]ObjSymbol{.{ .name = "f", .value = 0, .defined = true, .local = false, .section = .text }};
    var relocs = [_]Reloc{};
    var parsed = [_]ParsedObject{.{
        .arch = .riscv64,
        .text = &text,
        .symbols = &syms,
        .relocs = &relocs,
    }};

    var placement = try computeDefaultPlacement(allocator, &parsed, base, null, false);
    defer placement.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), placement.segments.len);
    const seg = placement.segments[0];
    try std.testing.expectEqual(base, seg.vaddr);
    try std.testing.expectEqual(base, seg.paddr);
    try std.testing.expectEqual(@as(u8, 7), seg.flags);
    try std.testing.expectEqual(base, findSymbol(placement.symbols, "f").?);
}
