//! DWARF debug-info emitter (v4, 32-bit DWARF format). This first slice emits the
//! `.debug_abbrev` and `.debug_info` sections describing a compilation unit and one
//! subprogram (function) DIE per function, each carrying its name and PC range
//! (DW_AT_low_pc / DW_AT_high_pc). That is enough for a debugger to name a function and
//! locate it in a backtrace. Line-number info (`.debug_line`), which needs source-location
//! tracking through the IR, is a later slice.
//!
//! Arch-independent (a container format like pe.zig / image.zig). Validated structurally
//! and, where `readelf` is present, by wrapping the sections in a minimal ELF and dumping.

const std = @import("std");

// DWARF constants (the subset this slice emits).
const DW_TAG_compile_unit: u8 = 0x11;
const DW_TAG_subprogram: u8 = 0x2e;
const DW_TAG_base_type: u8 = 0x24;
const DW_CHILDREN_no: u8 = 0x00;
const DW_CHILDREN_yes: u8 = 0x01;
const DW_AT_name: u8 = 0x03;
const DW_AT_byte_size: u8 = 0x0b;
const DW_AT_encoding: u8 = 0x3e;
const DW_AT_stmt_list: u8 = 0x10;
const DW_AT_low_pc: u8 = 0x11;
const DW_AT_high_pc: u8 = 0x12;
const DW_AT_language: u8 = 0x13;
const DW_AT_comp_dir: u8 = 0x1b;
const DW_AT_type: u8 = 0x49;
const DW_AT_producer: u8 = 0x25;
const DW_FORM_addr: u8 = 0x01;
const DW_FORM_data1: u8 = 0x0b;
const DW_FORM_data2: u8 = 0x05;
const DW_FORM_ref4: u8 = 0x13;
const DW_FORM_string: u8 = 0x08;
const DW_FORM_sec_offset: u8 = 0x17; // a 4-byte offset into another debug section (32-bit DWARF)
const DW_LANG_C99: u16 = 0x000c; // GLSL is C-like, and C99 is the closest standard code

// Base-type encodings (DW_ATE_*).
pub const Encoding = enum(u8) { boolean = 0x02, float = 0x04, signed = 0x05, unsigned = 0x07 };

/// A primitive type for a DWARF `DW_TAG_base_type` DIE (a function's return type, etc.).
pub const BaseType = struct { name: []const u8, encoding: Encoding, byte_size: u8 };

// Abbreviation codes used in `.debug_info`, indexing the table in `.debug_abbrev`.
const ABBREV_CU: u8 = 1;
const ABBREV_SUBPROGRAM: u8 = 2; // with a return type (DW_AT_type)
const ABBREV_BASE_TYPE: u8 = 3;
const ABBREV_SUBPROGRAM_VOID: u8 = 4; // no return type
const ABBREV_CU_LINES: u8 = 5; // compile unit with a DW_AT_stmt_list pointing at .debug_line

/// A function's debug description: its name, PC range, and optional return type.
pub const Subprogram = struct { name: []const u8, low_pc: u64, high_pc: u64, ret_type: ?BaseType = null };

/// A compilation unit: the producer/name/dir plus its overall PC range and functions.
pub const Unit = struct {
    name: []const u8,
    comp_dir: []const u8 = ".",
    producer: []const u8 = "vulcan",
    low_pc: u64,
    high_pc: u64,
    subprograms: []const Subprogram,
    /// When set, the CU carries `DW_AT_stmt_list` = this byte offset into `.debug_line`, linking the
    /// unit to its line-number program so a debugger can map its functions to source lines.
    stmt_list: ?u32 = null,
};

fn appendUleb(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: u64) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(v & 0x7f);
        v >>= 7;
        if (v != 0) byte |= 0x80;
        try out.append(allocator, byte);
        if (v == 0) break;
    }
}

fn appendStr(allocator: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.appendSlice(allocator, s);
    try out.append(allocator, 0); // DW_FORM_string is NUL-terminated
}

fn appendAddr(allocator: std.mem.Allocator, out: *std.ArrayList(u8), a: u64) !void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, a, .little);
    try out.appendSlice(allocator, &buf);
}

/// One abbreviation table entry: `code TAG children (AT FORM)* 0 0`.
fn appendAbbrevDecl(allocator: std.mem.Allocator, out: *std.ArrayList(u8), code: u8, tag: u8, children: u8, attrs: []const [2]u8) !void {
    try appendUleb(allocator, out, code);
    try appendUleb(allocator, out, tag);
    try out.append(allocator, children);
    for (attrs) |at| {
        try appendUleb(allocator, out, at[0]);
        try appendUleb(allocator, out, at[1]);
    }
    try appendUleb(allocator, out, 0); // attr list terminator
    try appendUleb(allocator, out, 0);
}

/// Emit the `.debug_abbrev` section: a compile-unit abbrev and a subprogram abbrev. Fixed
/// for this slice. Caller owns the result.
pub fn emitAbbrev(allocator: std.mem.Allocator) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendAbbrevDecl(allocator, &out, ABBREV_CU, DW_TAG_compile_unit, DW_CHILDREN_yes, &.{
        .{ DW_AT_producer, DW_FORM_string },
        .{ DW_AT_name, DW_FORM_string },
        .{ DW_AT_comp_dir, DW_FORM_string },
        .{ DW_AT_language, DW_FORM_data2 },
        .{ DW_AT_low_pc, DW_FORM_addr },
        .{ DW_AT_high_pc, DW_FORM_addr },
    });
    // A subprogram WITH a return type: DW_AT_type refers to its base-type DIE.
    try appendAbbrevDecl(allocator, &out, ABBREV_SUBPROGRAM, DW_TAG_subprogram, DW_CHILDREN_no, &.{
        .{ DW_AT_name, DW_FORM_string },
        .{ DW_AT_low_pc, DW_FORM_addr },
        .{ DW_AT_high_pc, DW_FORM_addr },
        .{ DW_AT_type, DW_FORM_ref4 },
    });
    // A base type (int/float/bool/...): name + encoding + byte size, no children.
    try appendAbbrevDecl(allocator, &out, ABBREV_BASE_TYPE, DW_TAG_base_type, DW_CHILDREN_no, &.{
        .{ DW_AT_name, DW_FORM_string },
        .{ DW_AT_encoding, DW_FORM_data1 },
        .{ DW_AT_byte_size, DW_FORM_data1 },
    });
    // A subprogram with NO return type (a void function): same as above minus DW_AT_type.
    try appendAbbrevDecl(allocator, &out, ABBREV_SUBPROGRAM_VOID, DW_TAG_subprogram, DW_CHILDREN_no, &.{
        .{ DW_AT_name, DW_FORM_string },
        .{ DW_AT_low_pc, DW_FORM_addr },
        .{ DW_AT_high_pc, DW_FORM_addr },
    });
    // A compile unit that also links a line-number program via DW_AT_stmt_list (sec_offset).
    try appendAbbrevDecl(allocator, &out, ABBREV_CU_LINES, DW_TAG_compile_unit, DW_CHILDREN_yes, &.{
        .{ DW_AT_producer, DW_FORM_string },
        .{ DW_AT_name, DW_FORM_string },
        .{ DW_AT_comp_dir, DW_FORM_string },
        .{ DW_AT_language, DW_FORM_data2 },
        .{ DW_AT_stmt_list, DW_FORM_sec_offset },
        .{ DW_AT_low_pc, DW_FORM_addr },
        .{ DW_AT_high_pc, DW_FORM_addr },
    });
    try appendUleb(allocator, &out, 0); // end of abbreviation table
    return out.toOwnedSlice(allocator);
}

/// Emit the `.debug_info` section for `unit`: a 32-bit-DWARF v4 unit header, the compile-unit
/// DIE, and a subprogram DIE per function, ending with a null DIE. Attribute order matches
/// `emitAbbrev`. Caller owns the result.
pub fn emitInfo(allocator: std.mem.Allocator, unit: Unit) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // Unit header. `unit_length` (the first u32) is back-patched once the body is known.
    try out.appendSlice(allocator, &.{ 0, 0, 0, 0 }); // unit_length placeholder
    try out.appendSlice(allocator, &.{ 4, 0 }); // version = 4 (u16 LE)
    try out.appendSlice(allocator, &.{ 0, 0, 0, 0 }); // debug_abbrev_offset = 0 (u32 LE)
    try out.append(allocator, 8); // address_size = 8

    // Compile-unit DIE: producer, name, comp_dir, language, [stmt_list], low_pc, high_pc. When the
    // unit links a line program (abbrev 5), DW_AT_stmt_list (sec_offset) is emitted after language.
    try appendUleb(allocator, &out, if (unit.stmt_list != null) ABBREV_CU_LINES else ABBREV_CU);
    try appendStr(allocator, &out, unit.producer);
    try appendStr(allocator, &out, unit.name);
    try appendStr(allocator, &out, unit.comp_dir);
    try out.appendSlice(allocator, &.{ @intCast(DW_LANG_C99 & 0xff), @intCast(DW_LANG_C99 >> 8) }); // DW_AT_language (data2 LE)
    if (unit.stmt_list) |so| try appendU32(allocator, &out, so); // DW_AT_stmt_list (sec_offset)
    try appendAddr(allocator, &out, unit.low_pc);
    try appendAddr(allocator, &out, unit.high_pc);

    // Emit a base_type DIE (abbrev 3) for each distinct return type, keyed by name, recording
    // each one's byte offset (from the CU start, which is byte 0 of this section) so the typed
    // subprogram DIEs can point at them via DW_FORM_ref4.
    var type_offsets: std.StringHashMapUnmanaged(u32) = .empty;
    defer type_offsets.deinit(allocator);
    for (unit.subprograms) |sub| {
        const ty = sub.ret_type orelse continue;
        if (type_offsets.contains(ty.name)) continue;
        try type_offsets.put(allocator, ty.name, @intCast(out.items.len));
        try appendUleb(allocator, &out, ABBREV_BASE_TYPE);
        try appendStr(allocator, &out, ty.name);
        try out.append(allocator, @intFromEnum(ty.encoding));
        try out.append(allocator, ty.byte_size);
    }

    // One subprogram DIE per function: abbrev 2 (typed) with a DW_FORM_ref4 to its base type,
    // else abbrev 4 (void) with no type reference.
    for (unit.subprograms) |sub| {
        if (sub.ret_type) |ty| {
            try appendUleb(allocator, &out, ABBREV_SUBPROGRAM);
            try appendStr(allocator, &out, sub.name);
            try appendAddr(allocator, &out, sub.low_pc);
            try appendAddr(allocator, &out, sub.high_pc);
            try appendU32(allocator, &out, type_offsets.get(ty.name).?); // DW_AT_type (ref4)
        } else {
            try appendUleb(allocator, &out, ABBREV_SUBPROGRAM_VOID);
            try appendStr(allocator, &out, sub.name);
            try appendAddr(allocator, &out, sub.low_pc);
            try appendAddr(allocator, &out, sub.high_pc);
        }
    }
    try appendUleb(allocator, &out, 0); // null DIE: end of the CU's children

    // Back-patch unit_length = total bytes after the length field itself.
    const unit_length: u32 = @intCast(out.items.len - 4);
    std.mem.writeInt(u32, out.items[0..4], unit_length, .little);
    return out.toOwnedSlice(allocator);
}

/// A function symbol as the linkers report it: a name and a byte offset into the code image.
pub const SymIn = struct { name: []const u8, offset: u64 };

/// Build the subprogram list for a linked code image from its function symbols. Each
/// function spans `[base+offset, base+next_offset)` (the last runs to `base+code_size`), so
/// the symbols are sorted by offset first. Caller owns the returned slice (the names are
/// borrowed from `syms`).
pub fn subprogramsFromSymbols(allocator: std.mem.Allocator, base: u64, code_size: u64, syms: []const SymIn) std.mem.Allocator.Error![]Subprogram {
    const sorted = try allocator.dupe(SymIn, syms);
    defer allocator.free(sorted);
    std.mem.sort(SymIn, sorted, {}, struct {
        fn lt(_: void, a: SymIn, b: SymIn) bool {
            return a.offset < b.offset;
        }
    }.lt);

    const subs = try allocator.alloc(Subprogram, sorted.len);
    errdefer allocator.free(subs);
    for (sorted, 0..) |s, i| {
        const end = if (i + 1 < sorted.len) sorted[i + 1].offset else code_size;
        subs[i] = .{ .name = s.name, .low_pc = base + s.offset, .high_pc = base + end };
    }
    return subs;
}

/// Emit a standalone minimal ELF containing just the DWARF for `unit` (`.debug_abbrev` +
/// `.debug_info`), readable by any DWARF consumer (readelf, gdb, objdump). Caller owns it.
pub fn emitDebugElf(allocator: std.mem.Allocator, unit: Unit) std.mem.Allocator.Error![]u8 {
    const abbrev = try emitAbbrev(allocator);
    defer allocator.free(abbrev);
    const info = try emitInfo(allocator, unit);
    defer allocator.free(info);
    return wrapSectionsElf(allocator, &.{ .{ .name = ".debug_abbrev", .data = abbrev }, .{ .name = ".debug_info", .data = info } });
}

const DW_LNS_copy: u8 = 1;
const DW_LNS_advance_pc: u8 = 2;
const DW_LNS_advance_line: u8 = 3;
const DW_LNE_end_sequence: u8 = 1;
const DW_LNE_set_address: u8 = 2;

/// One row of the line-number matrix: a source `line` starting at machine `address`.
pub const LineRow = struct { address: u64, line: u32 };

fn appendSleb(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: i64) !void {
    var v = value;
    while (true) {
        var byte: u8 = @intCast(@as(u64, @bitCast(v)) & 0x7f);
        v >>= 7; // arithmetic shift (sign-propagating)
        const done = (v == 0 and (byte & 0x40) == 0) or (v == -1 and (byte & 0x40) != 0);
        if (!done) byte |= 0x80;
        try out.append(allocator, byte);
        if (done) break;
    }
}

/// Emit a `.debug_line` section: a DWARF v4 line-number program mapping the `rows` (each a
/// machine address and the source line active there) for one source `file`, ending the
/// sequence at `end_address`. Uses explicit advance_pc / advance_line / copy opcodes (always
/// correct, if not the most compact). Caller owns the result.
pub fn emitLine(allocator: std.mem.Allocator, file: []const u8, rows: []const LineRow, end_address: u64) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, &.{ 0, 0, 0, 0 }); // unit_length (patched)
    try out.appendSlice(allocator, &.{ 4, 0 }); // version = 4
    try out.appendSlice(allocator, &.{ 0, 0, 0, 0 }); // header_length (patched)
    const after_header_len = out.items.len; // header_length is measured from here

    try out.append(allocator, 1); // minimum_instruction_length
    try out.append(allocator, 1); // maximum_operations_per_instruction (v4)
    try out.append(allocator, 1); // default_is_stmt
    try out.append(allocator, @bitCast(@as(i8, -5))); // line_base
    try out.append(allocator, 14); // line_range
    try out.append(allocator, 13); // opcode_base
    // standard_opcode_lengths for opcodes 1..12.
    try out.appendSlice(allocator, &.{ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 });
    try out.append(allocator, 0); // include_directories: empty list
    // file_names: one entry {name, dir_index=0, mtime=0, size=0}, then the terminator.
    try appendStr(allocator, &out, file);
    try appendUleb(allocator, &out, 0);
    try appendUleb(allocator, &out, 0);
    try appendUleb(allocator, &out, 0);
    try out.append(allocator, 0); // end of file list

    const header_length: u32 = @intCast(out.items.len - after_header_len);
    std.mem.writeInt(u32, out.items[6..10], header_length, .little);

    // Program. Start with the first row's address. The line register starts at 1.
    var cur_addr: u64 = if (rows.len > 0) rows[0].address else end_address;
    var cur_line: i64 = 1;
    // DW_LNE_set_address (extended): 0x00, len=9, subopcode, 8-byte address.
    try out.append(allocator, 0);
    try appendUleb(allocator, &out, 9);
    try out.append(allocator, DW_LNE_set_address);
    try appendAddr(allocator, &out, cur_addr);

    for (rows) |row| {
        if (row.address != cur_addr) {
            try out.append(allocator, DW_LNS_advance_pc);
            try appendUleb(allocator, &out, row.address - cur_addr);
            cur_addr = row.address;
        }
        if (@as(i64, row.line) != cur_line) {
            try out.append(allocator, DW_LNS_advance_line);
            try appendSleb(allocator, &out, @as(i64, row.line) - cur_line);
            cur_line = row.line;
        }
        try out.append(allocator, DW_LNS_copy); // append the row
    }
    if (end_address != cur_addr) {
        try out.append(allocator, DW_LNS_advance_pc);
        try appendUleb(allocator, &out, end_address - cur_addr);
    }
    // DW_LNE_end_sequence (extended): 0x00, len=1, subopcode.
    try out.append(allocator, 0);
    try appendUleb(allocator, &out, 1);
    try out.append(allocator, DW_LNE_end_sequence);

    const unit_length: u32 = @intCast(out.items.len - 4);
    std.mem.writeInt(u32, out.items[0..4], unit_length, .little);
    return out.toOwnedSlice(allocator);
}

/// A decoded line-matrix row. `end_sequence` marks the row terminating a sequence, where `address`
/// is one past the last instruction of the sequence and `line` is not meaningful.
pub const DecodedRow = struct { address: u64, line: u32, file: u32, end_sequence: bool };

const DW_LNS_set_file: u8 = 4;
const DW_LNS_set_column: u8 = 5;
const DW_LNS_negate_stmt: u8 = 6;
const DW_LNS_set_basic_block: u8 = 7;
const DW_LNS_const_add_pc: u8 = 8;
const DW_LNS_fixed_advance_pc: u8 = 9;

/// A little cursor over the section bytes (bounds-guarded reads, saturating at the end).
const Cursor = struct {
    b: []const u8,
    i: usize = 0,
    fn done(c: *const Cursor, end: usize) bool {
        return c.i >= end or c.i >= c.b.len;
    }
    fn byte(c: *Cursor) u8 {
        if (c.i >= c.b.len) return 0;
        defer c.i += 1;
        return c.b[c.i];
    }
    fn u16le(c: *Cursor) u16 {
        return @as(u16, c.byte()) | (@as(u16, c.byte()) << 8);
    }
    fn u32le(c: *Cursor) u32 {
        return @as(u32, c.u16le()) | (@as(u32, c.u16le()) << 16);
    }
    fn fixed(c: *Cursor, n: usize) u64 { // up to 8 little-endian bytes; the caller repositions the cursor
        var v: u64 = 0;
        // Clamp to 8: a u64 holds no more, and a hostile DW_LNE_set_address length
        // would otherwise drive `k * 8` past the u6 shift width and panic.
        for (0..@min(n, 8)) |k| v |= @as(u64, c.byte()) << @intCast(k * 8);
        return v;
    }
    fn uleb(c: *Cursor) u64 {
        var result: u64 = 0;
        // u7 so the += 7 cannot overflow the way a u6 counter does on an over-long
        // (malformed) encoding; stop once the shift would exceed the u64 width.
        var shift: u7 = 0;
        while (shift < 64) {
            const b = c.byte();
            result |= @as(u64, b & 0x7f) << @intCast(shift);
            if (b & 0x80 == 0) break;
            shift += 7;
        }
        return result;
    }
    fn sleb(c: *Cursor) i64 {
        var result: i64 = 0;
        var shift: u7 = 0;
        var b: u8 = 0;
        while (shift < 64) {
            b = c.byte();
            result |= @as(i64, @intCast(b & 0x7f)) << @intCast(shift);
            shift += 7;
            if (b & 0x80 == 0) break;
        }
        if (shift < 64 and (b & 0x40) != 0) result |= @as(i64, -1) << @intCast(shift); // sign-extend
        return result;
    }
};

/// Decode a `.debug_line` section into its address->line rows, running the DWARF line-number state
/// machine (handles DWARF v4/v5 unit headers, standard + extended + special opcodes). Directory and
/// file tables are skipped (we report the file *index*, not its name). Caller owns the result.
pub fn decodeLine(allocator: std.mem.Allocator, data: []const u8) std.mem.Allocator.Error![]DecodedRow {
    var rows: std.ArrayList(DecodedRow) = .empty;
    errdefer rows.deinit(allocator);
    var c: Cursor = .{ .b = data };

    while (c.i + 4 <= data.len) {
        const unit_len = c.u32le();
        if (unit_len == 0) break;
        const unit_end = @min(c.i + unit_len, data.len);
        const version = c.u16le();
        if (version >= 5) {
            _ = c.byte(); // address_size
            _ = c.byte(); // segment_selector_size
        }
        const header_length = c.u32le();
        const program_start = c.i + header_length;
        const min_inst_len = c.byte();
        if (version >= 4) _ = c.byte(); // maximum_operations_per_instruction
        _ = c.byte(); // default_is_stmt
        const line_base: i8 = @bitCast(c.byte());
        const line_range = c.byte();
        const opcode_base = c.byte();
        const nstd: usize = if (opcode_base > 0) opcode_base - 1 else 0;
        const std_lengths = try allocator.alloc(u8, nstd);
        defer allocator.free(std_lengths);
        for (std_lengths) |*sl| sl.* = c.byte();
        c.i = program_start; // skip the include_directories + file_names tables

        // Line-number registers, reset at each sequence start / end.
        var address: u64 = 0;
        var line: i64 = 1;
        var file: u32 = 1;
        const emit = struct {
            fn row(list: *std.ArrayList(DecodedRow), al: std.mem.Allocator, addr: u64, ln: i64, f: u32, es: bool) !void {
                try list.append(al, .{ .address = addr, .line = if (ln > 0) @intCast(ln) else 0, .file = f, .end_sequence = es });
            }
        }.row;

        while (!c.done(unit_end)) {
            const op = c.byte();
            if (op == 0) { // extended opcode: len, sub-opcode, operands
                const len = c.uleb();
                const after = c.i + len;
                const sub = c.byte();
                switch (sub) {
                    DW_LNE_end_sequence => {
                        try emit(&rows, allocator, address, line, file, true);
                        address = 0;
                        line = 1;
                        file = 1;
                    },
                    DW_LNE_set_address => address = c.fixed(if (len >= 1) len - 1 else 8),
                    else => {}, // define_file / vendor: skip
                }
                c.i = @min(after, data.len);
            } else if (op >= opcode_base) { // special opcode: bump address + line, emit a row
                const adj: u32 = op - opcode_base;
                if (line_range != 0) {
                    // Wrapping arithmetic on values derived from untrusted input: a
                    // malformed program must produce a wrong row, never panic.
                    address +%= @as(u64, min_inst_len) *% (adj / line_range);
                    line +%= @as(i64, line_base) + @as(i64, adj % line_range);
                }
                try emit(&rows, allocator, address, line, file, false);
            } else switch (op) {
                DW_LNS_copy => try emit(&rows, allocator, address, line, file, false),
                DW_LNS_advance_pc => address +%= c.uleb() *% min_inst_len,
                DW_LNS_advance_line => line +%= c.sleb(),
                DW_LNS_set_file => file = std.math.cast(u32, c.uleb()) orelse std.math.maxInt(u32),
                DW_LNS_set_column => _ = c.uleb(),
                DW_LNS_negate_stmt, DW_LNS_set_basic_block => {},
                DW_LNS_const_add_pc => if (line_range != 0) {
                    address +%= @as(u64, min_inst_len) *% ((255 - opcode_base) / line_range);
                },
                DW_LNS_fixed_advance_pc => address +%= c.u16le(), // raw (not scaled by min_inst_len)
                else => { // unknown standard opcode: skip its uleb operands per the header
                    const nargs = if (op - 1 < std_lengths.len) std_lengths[op - 1] else 0;
                    for (0..nargs) |_| _ = c.uleb();
                },
            }
        }
        c.i = unit_end;
    }
    return rows.toOwnedSlice(allocator);
}

/// Emit a standalone ELF containing just a `.debug_line` for the given rows. Caller owns it.
pub fn emitLineElf(allocator: std.mem.Allocator, file: []const u8, rows: []const LineRow, end_address: u64) std.mem.Allocator.Error![]u8 {
    const line = try emitLine(allocator, file, rows, end_address);
    defer allocator.free(line);
    return wrapSectionsElf(allocator, &.{.{ .name = ".debug_line", .data = line }});
}

/// Emit an ELF with `.debug_abbrev` + `.debug_info` + `.debug_line`, where the compile unit is linked
/// to the line program via `DW_AT_stmt_list` (offset 0 into `.debug_line`). This is the full,
/// debugger-consumable bundle: a consumer can go from a function DIE to its source lines. The `unit`'s
/// own `stmt_list` is overridden to 0 (the single line program starts at the section start). Caller
/// owns the result.
pub fn emitDebugElfWithLines(allocator: std.mem.Allocator, unit: Unit, file: []const u8, rows: []const LineRow, end_address: u64) std.mem.Allocator.Error![]u8 {
    var linked_unit = unit;
    linked_unit.stmt_list = 0;
    const abbrev = try emitAbbrev(allocator);
    defer allocator.free(abbrev);
    const info = try emitInfo(allocator, linked_unit);
    defer allocator.free(info);
    const line = try emitLine(allocator, file, rows, end_address);
    defer allocator.free(line);
    return wrapSectionsElf(allocator, &.{
        .{ .name = ".debug_abbrev", .data = abbrev },
        .{ .name = ".debug_info", .data = info },
        .{ .name = ".debug_line", .data = line },
    });
}

test "abbrev table has a compile-unit and a subprogram declaration" {
    const a = std.testing.allocator;
    const abbrev = try emitAbbrev(a);
    defer a.free(abbrev);

    // First decl: code 1, DW_TAG_compile_unit, DW_CHILDREN_yes.
    try std.testing.expectEqual(@as(u8, ABBREV_CU), abbrev[0]);
    try std.testing.expectEqual(DW_TAG_compile_unit, abbrev[1]);
    try std.testing.expectEqual(DW_CHILDREN_yes, abbrev[2]);
    // It ends with a 0 (table terminator).
    try std.testing.expectEqual(@as(u8, 0), abbrev[abbrev.len - 1]);
}

test "debug_info header is a well-formed 32-bit DWARF v4 unit" {
    const a = std.testing.allocator;
    const info = try emitInfo(a, .{
        .name = "shader.glsl",
        .low_pc = 0x1000,
        .high_pc = 0x1010,
        .subprograms = &.{.{ .name = "f", .low_pc = 0x1000, .high_pc = 0x1010 }},
    });
    defer a.free(info);

    // unit_length matches the actual remaining bytes.
    const unit_length = std.mem.readInt(u32, info[0..4], .little);
    try std.testing.expectEqual(@as(u32, @intCast(info.len - 4)), unit_length);
    // version = 4, address_size = 8.
    try std.testing.expectEqual(@as(u16, 4), std.mem.readInt(u16, info[4..6], .little));
    try std.testing.expectEqual(@as(u8, 8), info[10]);
    // First DIE is the compile unit (abbrev code 1).
    try std.testing.expectEqual(@as(u8, ABBREV_CU), info[11]);
    // The function name appears verbatim, and the whole thing ends in the null DIE.
    try std.testing.expect(std.mem.indexOf(u8, info, "f\x00") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "shader.glsl\x00") != null);
    try std.testing.expectEqual(@as(u8, 0), info[info.len - 1]);
}

/// A named PROGBITS section to place in the wrapper ELF.
const Section = struct { name: []const u8, data: []const u8 };

/// Package `sections` (plus an auto-generated `.shstrtab`) into a minimal ELF64 (ET_REL) so a
/// DWARF consumer (readelf/gdb/objdump) can read them. Test-and-tooling helper.
fn wrapSectionsElf(allocator: std.mem.Allocator, sections: []const Section) ![]u8 {
    // Build the section-header string table: an initial NUL (the null section's empty name),
    // then each section's name, then ".shstrtab".
    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const name_offs = try allocator.alloc(u32, sections.len);
    defer allocator.free(name_offs);
    for (sections, 0..) |s, i| {
        name_offs[i] = @intCast(shstr.items.len);
        try shstr.appendSlice(allocator, s.name);
        try shstr.append(allocator, 0);
    }
    const shstr_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab");
    try shstr.append(allocator, 0);

    // Byte layout: ELF header (64) | section datas | shstrtab | section-header table.
    var off: u64 = 64;
    const data_offs = try allocator.alloc(u64, sections.len);
    defer allocator.free(data_offs);
    for (sections, 0..) |s, i| {
        data_offs[i] = off;
        off += s.data.len;
    }
    const off_shstr = off;
    off += shstr.items.len;
    const off_shdr = off;
    const shnum: u16 = @intCast(sections.len + 2); // null + sections + shstrtab

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, &.{ 0x7f, 'E', 'L', 'F', 2, 1, 1, 0 }); // magic, 64-bit, LE, v1
    try out.appendSlice(allocator, &(.{0} ** 8)); // e_ident padding
    try appendU16(allocator, &out, 1); // e_type = ET_REL
    try appendU16(allocator, &out, 0xB7); // e_machine = AArch64 (arbitrary valid)
    try appendU32(allocator, &out, 1); // e_version
    try appendU64(allocator, &out, 0); // e_entry
    try appendU64(allocator, &out, 0); // e_phoff
    try appendU64(allocator, &out, off_shdr); // e_shoff
    try appendU32(allocator, &out, 0); // e_flags
    try appendU16(allocator, &out, 64); // e_ehsize
    try appendU16(allocator, &out, 0); // e_phentsize
    try appendU16(allocator, &out, 0); // e_phnum
    try appendU16(allocator, &out, 64); // e_shentsize
    try appendU16(allocator, &out, shnum); // e_shnum
    try appendU16(allocator, &out, shnum - 1); // e_shstrndx = the last section (shstrtab)

    for (sections) |s| try out.appendSlice(allocator, s.data);
    try out.appendSlice(allocator, shstr.items);

    try appendShdr(allocator, &out, 0, 0, 0, 0, 0); // null section
    for (sections, 0..) |s, i| try appendShdr(allocator, &out, name_offs[i], 1, data_offs[i], s.data.len, 1); // PROGBITS
    try appendShdr(allocator, &out, shstr_name, 3, off_shstr, shstr.items.len, 1); // .shstrtab (STRTAB)
    return out.toOwnedSlice(allocator);
}

fn appendU16(a: std.mem.Allocator, out: *std.ArrayList(u8), v: u16) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    try out.appendSlice(a, &b);
}
fn appendU32(a: std.mem.Allocator, out: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(a, &b);
}
fn appendU64(a: std.mem.Allocator, out: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try out.appendSlice(a, &b);
}
fn appendShdr(a: std.mem.Allocator, out: *std.ArrayList(u8), name: u32, sh_type: u32, offset: u64, size: u64, addralign: u64) !void {
    try appendU32(a, out, name); // sh_name
    try appendU32(a, out, sh_type); // sh_type
    try appendU64(a, out, 0); // sh_flags
    try appendU64(a, out, 0); // sh_addr
    try appendU64(a, out, offset); // sh_offset
    try appendU64(a, out, size); // sh_size
    try appendU32(a, out, 0); // sh_link
    try appendU32(a, out, 0); // sh_info
    try appendU64(a, out, addralign); // sh_addralign
    try appendU64(a, out, 0); // sh_entsize
}

test "readelf parses the emitted DWARF (compile unit + subprograms)" {
    const a = std.testing.allocator;
    const io = std.testing.io;

    const abbrev = try emitAbbrev(a);
    defer a.free(abbrev);
    const info = try emitInfo(a, .{
        .name = "shader.glsl",
        .low_pc = 0x1000,
        .high_pc = 0x1030,
        .subprograms = &.{
            .{ .name = "helper", .low_pc = 0x1000, .high_pc = 0x1010 },
            .{ .name = "main", .low_pc = 0x1010, .high_pc = 0x1030 },
        },
    });
    defer a.free(info);
    const elf = try wrapSectionsElf(a, &.{ .{ .name = ".debug_abbrev", .data = abbrev }, .{ .name = ".debug_info", .data = info } });
    defer a.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "dbg.o", .data = elf });

    const res = std.process.run(a, io, .{
        .argv = &.{ "readelf", "--debug-dump=info", "dbg.o" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest, // readelf unavailable
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) {
        std.debug.print("readelf failed:\n{s}\n", .{res.stderr});
        return error.ReadelfFailed;
    }
    // Binutils' DWARF reader must recognize the CU, both subprograms, and their names.
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_compile_unit") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_subprogram") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "shader.glsl") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "helper") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "main") != null);
}

test "subprogramsFromSymbols computes PC ranges from sorted offsets" {
    const a = std.testing.allocator;
    // Deliberately out of offset order. Base 0x2000, image 0x50 bytes.
    const subs = try subprogramsFromSymbols(a, 0x2000, 0x50, &.{
        .{ .name = "main", .offset = 0x20 },
        .{ .name = "helper", .offset = 0x00 },
    });
    defer a.free(subs);
    // Sorted: helper [0x2000, 0x2020), main [0x2020, 0x2050).
    try std.testing.expectEqualStrings("helper", subs[0].name);
    try std.testing.expectEqual(@as(u64, 0x2000), subs[0].low_pc);
    try std.testing.expectEqual(@as(u64, 0x2020), subs[0].high_pc);
    try std.testing.expectEqualStrings("main", subs[1].name);
    try std.testing.expectEqual(@as(u64, 0x2020), subs[1].low_pc);
    try std.testing.expectEqual(@as(u64, 0x2050), subs[1].high_pc);
}

test "readelf decodes the emitted .debug_line (address -> line table)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    // Lines 10, 11, 13 at successive 4-byte aarch64 instruction addresses from 0x1000.
    const elf = try emitLineElf(a, "shader.glsl", &.{
        .{ .address = 0x1000, .line = 10 },
        .{ .address = 0x1004, .line = 11 },
        .{ .address = 0x1008, .line = 13 },
    }, 0x100c);
    defer a.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "line.o", .data = elf });
    const res = std.process.run(a, io, .{ .argv = &.{ "readelf", "--debug-dump=decodedline", "line.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) {
        std.debug.print("readelf failed:\n{s}\n", .{res.stderr});
        return error.ReadelfFailed;
    }
    // The decoded table names the file and shows the line numbers at their addresses.
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "shader.glsl") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "10") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "13") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "0x1008") != null);
}

test "decodeLine round-trips the emitted line program" {
    const a = std.testing.allocator;
    const rows = [_]LineRow{
        .{ .address = 0x1000, .line = 10 },
        .{ .address = 0x1004, .line = 11 },
        .{ .address = 0x1010, .line = 42 }, // a jump in both address and line
        .{ .address = 0x1014, .line = 7 }, // line goes backwards
    };
    const line = try emitLine(a, "x.glsl", &rows, 0x1020);
    defer a.free(line);

    const decoded = try decodeLine(a, line);
    defer a.free(decoded);

    // Every input row comes back verbatim (in order), followed by a terminating end_sequence row.
    try std.testing.expectEqual(rows.len + 1, decoded.len);
    for (rows, 0..) |r, i| {
        try std.testing.expectEqual(r.address, decoded[i].address);
        try std.testing.expectEqual(r.line, decoded[i].line);
        try std.testing.expect(!decoded[i].end_sequence);
    }
    try std.testing.expect(decoded[rows.len].end_sequence);
    try std.testing.expectEqual(@as(u64, 0x1020), decoded[rows.len].address);
}

test "decodeLine handles special opcodes (compact clang-style encoding)" {
    const a = std.testing.allocator;
    // Hand-build a minimal v4 line program that uses a special opcode (the compact form real
    // producers emit): set_address(0x2000), a special opcode advancing line by +1 with no address
    // move, then end_sequence. With line_base=-5, line_range=14, opcode_base=13: to advance line by
    // +1 and address by 0, adjusted = (1 - line_base) + line_range*0 = 6, special = 6 + opcode_base.
    var prog: std.ArrayList(u8) = .empty;
    defer prog.deinit(a);
    try prog.appendSlice(a, &.{ 0, 0, 0, 0 }); // unit_length (patched)
    try prog.appendSlice(a, &.{ 4, 0 }); // version 4
    try prog.appendSlice(a, &.{ 0, 0, 0, 0 }); // header_length (patched)
    const hdr_start = prog.items.len;
    try prog.appendSlice(a, &.{ 1, 1, 1, @bitCast(@as(i8, -5)), 14, 13 }); // min_inst,max_ops,is_stmt,line_base,line_range,opcode_base
    try prog.appendSlice(a, &.{ 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 }); // standard_opcode_lengths[12]
    try prog.append(a, 0); // no include_directories
    try prog.appendSlice(a, "f.c\x00"); // one file
    try prog.appendSlice(a, &.{ 0, 0, 0 }); // dir_index, mtime, size
    try prog.append(a, 0); // end of file list
    const header_length: u32 = @intCast(prog.items.len - hdr_start);
    std.mem.writeInt(u32, prog.items[6..10], header_length, .little);
    // program: set_address 0x2000
    try prog.appendSlice(a, &.{ 0, 9, DW_LNE_set_address });
    try prog.appendSlice(a, &.{ 0x00, 0x20, 0, 0, 0, 0, 0, 0 });
    try prog.append(a, @intCast(6 + 13)); // special: line += 1, address += 0
    try prog.appendSlice(a, &.{ 0, 1, DW_LNE_end_sequence });
    const unit_length: u32 = @intCast(prog.items.len - 4);
    std.mem.writeInt(u32, prog.items[0..4], unit_length, .little);

    const decoded = try decodeLine(a, prog.items);
    defer a.free(decoded);
    try std.testing.expectEqual(@as(usize, 2), decoded.len);
    try std.testing.expectEqual(@as(u64, 0x2000), decoded[0].address);
    try std.testing.expectEqual(@as(u32, 2), decoded[0].line); // started at 1, +1
    try std.testing.expect(decoded[1].end_sequence);
}

test "decodeLine reads a real compiler's .debug_line (gcc/cc -g)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const elf_read = @import("elf_read.zig");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // A tiny function whose body lines (2, 3) must appear in the decoded line table.
    try tmp.dir.writeFile(io, .{ .sub_path = "add.c", .data = "int add(int a, int b) {\n  int c = a + b;\n  return c;\n}\n" });

    const cc = std.process.run(a, io, .{
        .argv = &.{ "cc", "-gdwarf-4", "-O0", "-c", "add.c", "-o", "add.o" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // no C compiler
        else => return e,
    };
    defer a.free(cc.stdout);
    defer a.free(cc.stderr);
    if (cc.term != .exited or cc.term.exited != 0) return error.SkipZigTest; // toolchain refused

    const obj = try tmp.dir.readFileAlloc(io, "add.o", a, .limited(4 << 20));
    defer a.free(obj);

    const dl = (try elf_read.sectionByName(obj, ".debug_line")) orelse return error.SkipZigTest;
    const rows = try decodeLine(a, dl);
    defer a.free(rows);

    // The decoder must recover the source lines a real producer emitted (2 and 3 from the body),
    // and terminate the sequence.
    var saw2 = false;
    var saw3 = false;
    var saw_end = false;
    for (rows) |r| {
        if (r.end_sequence) {
            saw_end = true;
            continue;
        }
        if (r.line == 2) saw2 = true;
        if (r.line == 3) saw3 = true;
    }
    try std.testing.expect(saw2);
    try std.testing.expect(saw3);
    try std.testing.expect(saw_end);
}

test "readelf links the compile unit to its line program via DW_AT_stmt_list" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    // One CU (two functions) bundled with a line program. The CU must reference the lines.
    const elf = try emitDebugElfWithLines(a, .{
        .name = "shader.glsl",
        .low_pc = 0x1000,
        .high_pc = 0x1010,
        .subprograms = &.{
            .{ .name = "helper", .low_pc = 0x1000, .high_pc = 0x1008 },
            .{ .name = "main", .low_pc = 0x1008, .high_pc = 0x1010 },
        },
    }, "shader.glsl", &.{
        .{ .address = 0x1000, .line = 10 },
        .{ .address = 0x1008, .line = 20 },
    }, 0x1010);
    defer a.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "bundle.o", .data = elf });

    const info = std.process.run(a, io, .{ .argv = &.{ "readelf", "--debug-dump=info", "bundle.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(info.stdout);
    defer a.free(info.stderr);
    if (info.term != .exited or info.term.exited != 0) {
        std.debug.print("readelf info failed:\n{s}\n", .{info.stderr});
        return error.ReadelfFailed;
    }
    // The CU carries a DW_AT_stmt_list attribute pointing at the line program.
    try std.testing.expect(std.mem.indexOf(u8, info.stdout, "DW_AT_stmt_list") != null);

    // And the line program itself decodes: with the CU linked, readelf attributes the rows to it.
    const dl = std.process.run(a, io, .{ .argv = &.{ "readelf", "--debug-dump=decodedline", "bundle.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(dl.stdout);
    defer a.free(dl.stderr);
    if (dl.term != .exited or dl.term.exited != 0) {
        std.debug.print("readelf decodedline failed:\n{s}\n", .{dl.stderr});
        return error.ReadelfFailed;
    }
    try std.testing.expect(std.mem.indexOf(u8, dl.stdout, "shader.glsl") != null);
    try std.testing.expect(std.mem.indexOf(u8, dl.stdout, "20") != null);
    try std.testing.expect(std.mem.indexOf(u8, dl.stdout, "0x1008") != null);
}

test "readelf shows base types and typed subprogram return types" {
    const a = std.testing.allocator;
    const io = std.testing.io;

    const i32_ty: BaseType = .{ .name = "int", .encoding = .signed, .byte_size = 4 };
    const f32_ty: BaseType = .{ .name = "float", .encoding = .float, .byte_size = 4 };
    const abbrev = try emitAbbrev(a);
    defer a.free(abbrev);
    const info = try emitInfo(a, .{
        .name = "shader.glsl",
        .low_pc = 0x1000,
        .high_pc = 0x1030,
        .subprograms = &.{
            .{ .name = "geti", .low_pc = 0x1000, .high_pc = 0x1010, .ret_type = i32_ty },
            .{ .name = "getf", .low_pc = 0x1010, .high_pc = 0x1020, .ret_type = f32_ty },
            .{ .name = "doit", .low_pc = 0x1020, .high_pc = 0x1030 }, // void: no ret_type
        },
    });
    defer a.free(info);
    const elf = try wrapSectionsElf(a, &.{ .{ .name = ".debug_abbrev", .data = abbrev }, .{ .name = ".debug_info", .data = info } });
    defer a.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "typed.o", .data = elf });
    const res = std.process.run(a, io, .{ .argv = &.{ "readelf", "--debug-dump=info", "typed.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) {
        std.debug.print("readelf failed:\n{s}\n", .{res.stderr});
        return error.ReadelfFailed;
    }
    // Base type DIEs are present with their names, and the CU declares its language.
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_base_type") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "int") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "float") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_AT_language") != null);
    // A typed subprogram carries a DW_AT_type reference (readelf prints it on the subprogram).
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_AT_type") != null);
}

test "multiple subprograms each get a DIE" {
    const a = std.testing.allocator;
    const info = try emitInfo(a, .{
        .name = "m",
        .low_pc = 0,
        .high_pc = 0x40,
        .subprograms = &.{
            .{ .name = "helper", .low_pc = 0, .high_pc = 0x10 },
            .{ .name = "main", .low_pc = 0x10, .high_pc = 0x40 },
        },
    });
    defer a.free(info);
    try std.testing.expect(std.mem.indexOf(u8, info, "helper\x00") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "main\x00") != null);
}
