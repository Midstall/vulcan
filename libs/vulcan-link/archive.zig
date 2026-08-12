//! Minimal `ar` archive (`.a`) parsing for the shared static linker, `std`-only. This
//! reads the common-format archive shell (the `!<arch>\n` magic, 60-byte member
//! headers, GNU long-name (`//`) member, SysV/GNU (`/`) and BSD (`__.SYMDEF`) symbol
//! index members) and hands back the real object members as raw byte slices. It does
//! NOT read any archive symbol index: the link driver (`resolve.zig`) decides which
//! members to pull by scanning each candidate member's own ELF symbol table via
//! `elf.zig`, so a member's symtab is the single source of truth either way.

const std = @import("std");
const elf = @import("elf.zig");

pub const Error = elf.Error;

const magic = "!<arch>\n";
const header_size: usize = 60;

/// One real object member pulled out of an archive: its resolved name (after GNU
/// long-name resolution, trailing `/`/space padding trimmed) and its raw bytes
/// (a slice into the archive buffer passed to `parseArchive`).
pub const Member = struct {
    name: []const u8,
    bytes: []const u8,
};

/// True iff `bytes` starts with the common-format archive magic `!<arch>\n`.
pub fn isArchive(bytes: []const u8) bool {
    return bytes.len >= magic.len and std.mem.eql(u8, bytes[0..magic.len], magic);
}

/// One raw member header + its data, before name interpretation (a long-name
/// reference cannot be resolved until every header has been walked, since the GNU
/// long-name table member can appear anywhere in the archive).
const RawEntry = struct {
    name_field: []const u8,
    data: []const u8,
};

/// Parse a common-format `ar` archive into its real object members. Skips the GNU
/// long-name table (`//`) and the SysV/GNU (`/`) or BSD (`__.SYMDEF*`) symbol index
/// members; resolves GNU long names (`/N` referencing an offset into `//`). Every
/// header field is bounds- and format-checked; a truncated or malformed header is
/// `error.MalformedObject`. The returned slice (and its `Member`s, whose `name`/`bytes`
/// borrow from `bytes`) is owned by the caller and freed with `allocator.free`.
pub fn parseArchive(allocator: std.mem.Allocator, bytes: []const u8) Error![]Member {
    if (!isArchive(bytes)) return error.MalformedObject;

    var raw: std.ArrayList(RawEntry) = .empty;
    defer raw.deinit(allocator);

    var offset: usize = magic.len;
    while (offset < bytes.len) {
        if (bytes.len - offset < header_size) return error.MalformedObject;
        const header = bytes[offset..][0..header_size];
        if (header[58] != '`' or header[59] != '\n') return error.MalformedObject;

        const size_field = std.mem.trimEnd(u8, header[48..58], " ");
        if (size_field.len == 0) return error.MalformedObject;
        const size = std.fmt.parseInt(usize, size_field, 10) catch return error.MalformedObject;

        const data_start = offset + header_size;
        if (size > bytes.len - data_start) return error.MalformedObject;
        const data = bytes[data_start..][0..size];

        try raw.append(allocator, .{ .name_field = header[0..16], .data = data });

        // Member data is padded to a 2-byte boundary; the pad byte is only present
        // when there is room for one (tolerate a final member with no trailing pad).
        var next = data_start + size;
        if (size % 2 == 1 and next < bytes.len) next += 1;
        offset = next;
    }

    // The GNU long-name table, if present, can appear at any position: find it
    // before resolving any `/N` reference.
    var longnames: ?[]const u8 = null;
    for (raw.items) |e| {
        if (e.name_field.len >= 2 and e.name_field[0] == '/' and e.name_field[1] == '/') {
            longnames = e.data;
            break;
        }
    }

    var members: std.ArrayList(Member) = .empty;
    errdefer members.deinit(allocator);

    for (raw.items) |e| {
        const nf = e.name_field;
        if (nf[0] == '/') {
            if (nf.len >= 2 and nf[1] == '/') continue; // the long-name table itself
            if (nf.len >= 2 and std.ascii.isDigit(nf[1])) {
                const off_field = std.mem.trimEnd(u8, nf[1..], " ");
                const off = std.fmt.parseInt(usize, off_field, 10) catch return error.MalformedObject;
                const table = longnames orelse return error.MalformedObject;
                if (off >= table.len) return error.MalformedObject;
                const end = std.mem.indexOfScalarPos(u8, table, off, '\n') orelse table.len;
                var name = table[off..end];
                if (name.len > 0 and name[name.len - 1] == '/') name = name[0 .. name.len - 1];
                try members.append(allocator, .{ .name = name, .bytes = e.data });
                continue;
            }
            // A lone "/" (blank-padded): the SysV/GNU archive symbol index. Skip it -
            // the link driver scans each member's own ELF symtab instead.
            continue;
        }
        if (std.mem.startsWith(u8, nf, "__.SYMDEF")) continue; // the BSD symbol index

        try members.append(allocator, .{ .name = trimShortName(nf), .bytes = e.data });
    }

    return members.toOwnedSlice(allocator);
}

/// Trim a short (non-long-name) member name: a GNU archive terminates the name with
/// `/` before the space padding; a BSD archive just space-pads.
fn trimShortName(field: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, field, '/')) |slash| return field[0..slash];
    return std.mem.trimEnd(u8, field, " ");
}

test "isArchive checks the magic" {
    try std.testing.expect(isArchive("!<arch>\n" ++ "anything"));
    try std.testing.expect(!isArchive("not an archive"));
    try std.testing.expect(!isArchive("!<arch>")); // too short, no trailing newline
}

fn writeHeader(buf: []u8, name: []const u8, size: usize) void {
    @memset(buf[0..header_size], ' ');
    @memcpy(buf[0..name.len], name);
    var size_buf: [10]u8 = undefined;
    const s = std.fmt.bufPrint(&size_buf, "{d}", .{size}) catch unreachable;
    @memcpy(buf[48..][0..s.len], s);
    buf[58] = '`';
    buf[59] = '\n';
}

test "parseArchive reads a short-named GNU member and skips the symbol index" {
    const allocator = std.testing.allocator;
    const payload = "hello";

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, magic);

    // A SysV symbol-index member ("/") with no data: must be skipped entirely.
    {
        var hdr: [header_size]u8 = undefined;
        writeHeader(&hdr, "/", 0);
        try buf.appendSlice(allocator, &hdr);
    }
    // The real member.
    {
        var hdr: [header_size]u8 = undefined;
        writeHeader(&hdr, "thing.o/", payload.len);
        try buf.appendSlice(allocator, &hdr);
        try buf.appendSlice(allocator, payload);
        if (payload.len % 2 == 1) try buf.append(allocator, '\n');
    }

    const members = try parseArchive(allocator, buf.items);
    defer allocator.free(members);
    try std.testing.expectEqual(@as(usize, 1), members.len);
    try std.testing.expectEqualStrings("thing.o", members[0].name);
    try std.testing.expectEqualStrings(payload, members[0].bytes);
}

test "parseArchive resolves a GNU long name via the // table" {
    const allocator = std.testing.allocator;
    const long_name = "a_very_long_member_name_that_does_not_fit_in_16_bytes.o";
    const payload = "xy";

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, magic);

    var names_blob: std.ArrayList(u8) = .empty;
    defer names_blob.deinit(allocator);
    try names_blob.appendSlice(allocator, long_name);
    try names_blob.appendSlice(allocator, "/\n");

    {
        var hdr: [header_size]u8 = undefined;
        writeHeader(&hdr, "//", names_blob.items.len);
        try buf.appendSlice(allocator, &hdr);
        try buf.appendSlice(allocator, names_blob.items);
        if (names_blob.items.len % 2 == 1) try buf.append(allocator, '\n');
    }
    {
        var hdr: [header_size]u8 = undefined;
        writeHeader(&hdr, "/0", payload.len);
        try buf.appendSlice(allocator, &hdr);
        try buf.appendSlice(allocator, payload);
        if (payload.len % 2 == 1) try buf.append(allocator, '\n');
    }

    const members = try parseArchive(allocator, buf.items);
    defer allocator.free(members);
    try std.testing.expectEqual(@as(usize, 1), members.len);
    try std.testing.expectEqualStrings(long_name, members[0].name);
    try std.testing.expectEqualStrings(payload, members[0].bytes);
}

test "parseArchive rejects a truncated header" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, magic);
    try buf.appendSlice(allocator, "short");
    try std.testing.expectError(error.MalformedObject, parseArchive(allocator, buf.items));
}
