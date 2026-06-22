//! PE32+/COFF image emission for UEFI applications. A UEFI executable is a PE32+
//! image with the EFI_APPLICATION subsystem, entered at `efi_main(ImageHandle,
//! SystemTable)` (args arrive in the architecture's first two argument registers).
//! The container is architecture-independent. Only the COFF machine field and
//! relocation types differ per backend.
//!
//! Produces a single-section image: linked code (plus zero-init tail) in `.text`,
//! entered at a given offset. Generated code is position-independent (calls/branches
//! are PC-relative), so no base relocations are needed and UEFI can load it anywhere.
//! Freestanding-clean: byte generation only, no OS or allocator state beyond the
//! output buffer.

const std = @import("std");

pub const Error = std.mem.Allocator.Error;

/// COFF machine types for the targets Vulcan emits.
pub const Machine = enum(u16) {
    aarch64 = 0xAA64,
    riscv64 = 0x5064,
    x86_64 = 0x8664,
    i386 = 0x014C,
};

// PE/COFF constants.
const dos_size: usize = 64; // DOS header (no stub). PE header follows immediately
const pe_sig_size: usize = 4;
const coff_size: usize = 20;
const opt_size: usize = 240; // PE32+ optional header (112 fixed + 16 data dirs * 8)
const sect_hdr_size: usize = 40;
const file_align: u64 = 0x200;
const section_align: u64 = 0x1000;
const image_base: u64 = 0x400000;

const SUBSYSTEM_EFI_APPLICATION: u16 = 10;
const MAGIC_PE32_PLUS: u16 = 0x020B;
const FILE_EXECUTABLE: u16 = 0x0002;
const FILE_LARGE_ADDRESS_AWARE: u16 = 0x0020;
const FILE_LINE_NUMS_STRIPPED: u16 = 0x0004;
const SCN_CNT_CODE: u32 = 0x00000020;
const SCN_MEM_EXECUTE: u32 = 0x20000000;
const SCN_MEM_READ: u32 = 0x40000000;
const SCN_MEM_WRITE: u32 = 0x80000000;

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}

fn w(comptime T: type, buf: []u8, value: T) void {
    std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little);
}

/// Build a UEFI PE32+ application image: `code` placed in a single executable
/// `.text` section (with `mem_size - code.len` bytes of zero-init tail), entered at
/// byte `entry_offset` within it. Caller owns the bytes.
pub fn writeUefiImage(allocator: std.mem.Allocator, code: []const u8, mem_size: u64, entry_offset: u64, machine: Machine) Error![]u8 {
    const headers_end = dos_size + pe_sig_size + coff_size + opt_size + sect_hdr_size;
    const size_of_headers = alignUp(headers_end, file_align);
    const raw_size = alignUp(code.len, file_align);
    const text_rva = section_align; // first section at RVA 0x1000
    const virt_size = @max(mem_size, code.len);
    const size_of_image = text_rva + alignUp(virt_size, section_align);
    const total: usize = @intCast(size_of_headers + raw_size);

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    // DOS header: "MZ" and e_lfanew pointing at the PE signature.
    buf[0] = 'M';
    buf[1] = 'Z';
    w(u32, buf[0x3c..], @intCast(dos_size)); // e_lfanew

    // PE signature.
    @memcpy(buf[dos_size..][0..4], "PE\x00\x00");

    // COFF file header.
    const coff = buf[dos_size + pe_sig_size ..];
    w(u16, coff[0..], @intFromEnum(machine));
    w(u16, coff[2..], 1); // NumberOfSections
    w(u16, coff[16..], @intCast(opt_size)); // SizeOfOptionalHeader
    w(u16, coff[18..], FILE_EXECUTABLE | FILE_LARGE_ADDRESS_AWARE | FILE_LINE_NUMS_STRIPPED);

    // PE32+ optional header.
    const opt = coff[coff_size..];
    w(u16, opt[0..], MAGIC_PE32_PLUS);
    w(u32, opt[4..], @intCast(raw_size)); // SizeOfCode
    w(u32, opt[16..], @intCast(text_rva + entry_offset)); // AddressOfEntryPoint
    w(u32, opt[20..], @intCast(text_rva)); // BaseOfCode
    w(u64, opt[24..], image_base);
    w(u32, opt[32..], @intCast(section_align));
    w(u32, opt[36..], @intCast(file_align));
    w(u16, opt[48..], 1); // MajorSubsystemVersion (UEFI wants >= 1)
    w(u32, opt[56..], @intCast(size_of_image));
    w(u32, opt[60..], @intCast(size_of_headers));
    w(u16, opt[68..], SUBSYSTEM_EFI_APPLICATION);
    w(u32, opt[108..], 16); // NumberOfRvaAndSizes (the 16 data directories follow, all zero)

    // The single `.text` section header.
    const sect = coff[coff_size + opt_size ..];
    @memcpy(sect[0..5], ".text");
    w(u32, sect[8..], @intCast(virt_size)); // VirtualSize
    w(u32, sect[12..], @intCast(text_rva)); // VirtualAddress
    w(u32, sect[16..], @intCast(raw_size)); // SizeOfRawData
    w(u32, sect[20..], @intCast(size_of_headers)); // PointerToRawData
    w(u32, sect[36..], SCN_CNT_CODE | SCN_MEM_EXECUTE | SCN_MEM_READ | SCN_MEM_WRITE);

    // Section contents.
    @memcpy(buf[@intCast(size_of_headers)..][0..code.len], code);
    return buf;
}

test "emits a well-formed UEFI PE32+ header" {
    const allocator = std.testing.allocator;
    const code = [_]u8{ 0x00, 0x00, 0x80, 0xd2, 0xc0, 0x03, 0x5f, 0xd6 }; // mov x0,#0, ret
    const img = try writeUefiImage(allocator, &code, code.len, 0, .aarch64);
    defer allocator.free(img);

    try std.testing.expectEqual(@as(u8, 'M'), img[0]);
    try std.testing.expectEqual(@as(u8, 'Z'), img[1]);
    const lfanew = std.mem.readInt(u32, img[0x3c..0x40], .little);
    try std.testing.expectEqualSlices(u8, "PE\x00\x00", img[lfanew..][0..4]);

    const coff = img[lfanew + 4 ..];
    try std.testing.expectEqual(@as(u16, 0xAA64), std.mem.readInt(u16, coff[0..2], .little)); // AArch64
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, coff[2..4], .little)); // 1 section
    try std.testing.expectEqual(@as(u16, 240), std.mem.readInt(u16, coff[16..18], .little)); // opt hdr size

    const opt = coff[coff_size..];
    try std.testing.expectEqual(@as(u16, 0x020B), std.mem.readInt(u16, opt[0..2], .little)); // PE32+
    try std.testing.expectEqual(@as(u16, 10), std.mem.readInt(u16, opt[68..70], .little)); // EFI_APPLICATION
    try std.testing.expectEqual(@as(u32, 0x1000), std.mem.readInt(u32, opt[16..20], .little)); // entry RVA

    // The code lands at PointerToRawData and the entry RVA points at its start.
    const sect = coff[coff_size + opt_size ..];
    const praw = std.mem.readInt(u32, sect[20..24], .little);
    try std.testing.expectEqualSlices(u8, &code, img[praw..][0..code.len]);
    try std.testing.expectEqual(@as(u16, 0x5064), @intFromEnum(Machine.riscv64));
}
