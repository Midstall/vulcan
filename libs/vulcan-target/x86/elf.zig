//! Static i386 Linux ELF executable writer (ET_EXEC, EM_386). One R+X PT_LOAD segment
//! maps the whole file at a fixed base, entered at a byte offset into the code.

const std = @import("std");

pub const Error = std.mem.Allocator.Error;

pub const load_addr: u32 = 0x08048000; // the classic i386 ELF base
const ehsize: usize = 52;
const phsize: usize = 32;
pub const code_offset: usize = ehsize + phsize;

/// Wrap `code` into a static i386 ELF executable entered at `entry_offset`. Caller owns
/// the bytes.
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
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    put(buf, u16, 16, 2); // e_type = ET_EXEC
    put(buf, u16, 18, 3); // e_machine = EM_386
    put(buf, u32, 20, 1); // e_version
    put(buf, u32, 24, load_addr + @as(u32, @intCast(code_offset + entry_offset))); // e_entry
    put(buf, u32, 28, ehsize); // e_phoff
    put(buf, u16, 40, ehsize); // e_ehsize
    put(buf, u16, 42, phsize); // e_phentsize
    put(buf, u16, 44, 1); // e_phnum

    const ph = ehsize;
    put(buf, u32, ph + 0, 1); // PT_LOAD
    put(buf, u32, ph + 4, 0); // p_offset
    put(buf, u32, ph + 8, load_addr); // p_vaddr
    put(buf, u32, ph + 12, load_addr); // p_paddr
    put(buf, u32, ph + 16, @intCast(total)); // p_filesz
    put(buf, u32, ph + 20, @intCast(total)); // p_memsz
    put(buf, u32, ph + 24, 5); // PF_R | PF_X
    put(buf, u32, ph + 28, 0x1000); // p_align

    @memcpy(buf[code_offset..], code);
    return buf;
}

test "emits a well-formed i386 ELF executable" {
    const allocator = std.testing.allocator;
    const code = [_]u8{ 0xC3, 0x90 };
    const elf = try writeExec(allocator, &code, 0);
    defer allocator.free(elf);
    try std.testing.expectEqualSlices(u8, "\x7fELF", elf[0..4]);
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, elf[18..20], .little)); // EM_386
    try std.testing.expectEqualSlices(u8, &code, elf[code_offset..]);
}
