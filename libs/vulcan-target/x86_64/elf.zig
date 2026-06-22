//! Static x86-64 Linux ELF executable writer (ET_EXEC, EM_X86_64). One R+X PT_LOAD
//! segment maps the whole file at a fixed base, entered at a byte offset into the code.
//! Generated code is position-independent (PC-relative branches), so the base does not
//! affect the bytes.

const std = @import("std");

pub const Error = std.mem.Allocator.Error;

pub const load_addr: u64 = 0x10000000;
const ehsize: usize = 64;
const phsize: usize = 56;
pub const code_offset: usize = ehsize + phsize; // file offset of the code

/// Wrap `code` into a static ELF executable entered at `entry_offset`. Caller owns the bytes.
pub fn writeExec(allocator: std.mem.Allocator, code: []const u8, entry_offset: usize) Error![]u8 {
    const total = code_offset + code.len;
    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    const put = struct {
        fn i(b: []u8, comptime T: type, off: usize, v: T) void {
            std.mem.writeInt(T, b[off..][0..@sizeOf(T)], v, .little);
        }
    }.i;

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    put(buf, u16, 16, 2); // e_type = ET_EXEC
    put(buf, u16, 18, 62); // e_machine = EM_X86_64
    put(buf, u32, 20, 1); // e_version
    put(buf, u64, 24, load_addr + code_offset + entry_offset); // e_entry
    put(buf, u64, 32, ehsize); // e_phoff
    put(buf, u16, 52, ehsize); // e_ehsize
    put(buf, u16, 54, phsize); // e_phentsize
    put(buf, u16, 56, 1); // e_phnum

    const ph = ehsize;
    put(buf, u32, ph + 0, 1); // PT_LOAD
    put(buf, u32, ph + 4, 5); // PF_R | PF_X
    put(buf, u64, ph + 8, 0); // p_offset
    put(buf, u64, ph + 16, load_addr); // p_vaddr
    put(buf, u64, ph + 24, load_addr); // p_paddr
    put(buf, u64, ph + 32, total); // p_filesz
    put(buf, u64, ph + 40, total); // p_memsz
    put(buf, u64, ph + 48, 0x1000); // p_align

    @memcpy(buf[code_offset..], code);
    return buf;
}

test "emits a well-formed x86-64 ELF executable" {
    const allocator = std.testing.allocator;
    const code = [_]u8{ 0xC3, 0x90 }; // ret, nop
    const elf = try writeExec(allocator, &code, 0);
    defer allocator.free(elf);
    try std.testing.expectEqualSlices(u8, "\x7fELF", elf[0..4]);
    try std.testing.expectEqual(@as(u16, 62), std.mem.readInt(u16, elf[18..20], .little)); // EM_X86_64
    try std.testing.expectEqual(load_addr + code_offset, std.mem.readInt(u64, elf[24..32], .little));
    try std.testing.expectEqualSlices(u8, &code, elf[code_offset..]);
}
