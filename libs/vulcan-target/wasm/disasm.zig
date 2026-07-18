//! WebAssembly bytecode disassembler: decodes a function body byte stream (the output of
//! `isel.zig` / `encode.zig`) into a readable, block-indented instruction listing, so
//! debugging the Wasm backend does not mean reading raw hex. Opcode byte assignments follow
//! the WebAssembly spec (the correct reading of a real Wasm binary). Immediates (LEB128
//! indices, memarg align/offset, const values) are decoded inline. An unknown byte prints as
//! `.byte 0x<hex>` and decoding continues.
//!
//! `format` renders a whole code buffer; `disasmInst` decodes one instruction and reports
//! its length, for programmatic stepping. Validated by round-tripping encoder output.

const std = @import("std");

/// Decode a whole function-body byte stream into a listing (`offset: indent  text`). Caller
/// owns the result.
pub fn format(allocator: std.mem.Allocator, code: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var pos: usize = 0;
    var depth: usize = 0;
    while (pos < code.len) {
        const op = code[pos];
        // `end` and `else` close the current block, so they dedent before printing.
        if (op == 0x0B or op == 0x05) depth -|= 1;
        try out.print(allocator, "{x:0>4}: ", .{pos});
        for (0..depth) |_| try out.appendSlice(allocator, "  ");
        const len = try decodeOne(allocator, &out, code, pos);
        try out.append(allocator, '\n');
        if (op == 0x02 or op == 0x03 or op == 0x04 or op == 0x05) depth += 1; // block/loop/if/else open
        pos += len;
    }
    return out.toOwnedSlice(allocator);
}

/// Decode one instruction at `code[pos]`, appending its text to `out`, and return its total
/// byte length (opcode + immediates).
pub fn disasmInst(allocator: std.mem.Allocator, out: *std.ArrayList(u8), code: []const u8, pos: usize) std.mem.Allocator.Error!usize {
    return decodeOne(allocator, out, code, pos);
}

/// Decode a whole function body: the leading local-declaration vector (which `format`
/// alone would misread as opcodes) then the instruction expression. This is what
/// `isel.selectFunction(...).code` holds. Caller owns the result.
pub fn formatBody(allocator: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var r = Reader{ .code = body, .pos = 0 };
    const groups = r.u32Leb();
    try out.appendSlice(allocator, "locals:");
    if (groups == 0) {
        try out.appendSlice(allocator, " (none)");
    } else
    // Cap by body length: each group consumes at least two bytes, so a count larger
    // than the whole body is malformed. Prevents a tiny input with a ~4G count from
    // driving a multi-gigabyte output loop.
    for (0..@min(groups, body.len)) |_| {
        const count = r.u32Leb();
        try out.print(allocator, " {d} x {s}", .{ count, valType(r.byte()) });
    }
    try out.append(allocator, '\n');
    const listing = try format(allocator, body[r.pos..]);
    defer allocator.free(listing);
    try out.appendSlice(allocator, listing);
    return out.toOwnedSlice(allocator);
}

/// A cursor that reads LEB128 and raw values, advancing an index.
const Reader = struct {
    code: []const u8,
    pos: usize,

    fn byte(r: *Reader) u8 {
        if (r.pos >= r.code.len) return 0;
        const b = r.code[r.pos];
        r.pos += 1;
        return b;
    }
    fn u32Leb(r: *Reader) u32 {
        var result: u32 = 0;
        var shift: u5 = 0;
        while (true) {
            const b = r.byte();
            result |= @as(u32, b & 0x7F) << shift;
            if (b & 0x80 == 0) break;
            shift +%= 7;
        }
        return result;
    }
    fn s64Leb(r: *Reader) i64 {
        var result: i64 = 0;
        var shift: u7 = 0;
        var b: u8 = 0;
        // Bound the shift: an over-long (malformed) LEB must not drive `shift`
        // past the u64 width, where `@intCast(shift)` to the u6 shift type panics.
        while (shift < 64) {
            b = r.byte();
            result |= @as(i64, b & 0x7F) << @intCast(shift);
            shift += 7;
            if (b & 0x80 == 0) break;
        }
        if (shift < 64 and (b & 0x40) != 0) result |= @as(i64, -1) << @intCast(shift);
        return result;
    }
    fn raw(r: *Reader, comptime T: type) T {
        const n = @sizeOf(T);
        var bytes: [n]u8 = undefined;
        for (0..n) |i| bytes[i] = r.byte();
        // wasm immediates are little-endian on the wire; assemble in that order so
        // f32/f64 constants decode correctly on a big-endian host too.
        const Bits = std.meta.Int(.unsigned, n * 8);
        return @bitCast(std.mem.readInt(Bits, &bytes, .little));
    }
};

fn decodeOne(a: std.mem.Allocator, out: *std.ArrayList(u8), code: []const u8, pos: usize) !usize {
    var r = Reader{ .code = code, .pos = pos };
    const op = r.byte();
    switch (op) {
        0x02, 0x03, 0x04 => { // block / loop / if <blocktype>
            const mnem: []const u8 = switch (op) {
                0x02 => "block",
                0x03 => "loop",
                else => "if",
            };
            const bt = r.byte();
            try out.appendSlice(a, mnem);
            if (bt != 0x40) try out.print(a, " (result {s})", .{valType(bt)});
        },
        0x0C => try out.print(a, "br {d}", .{r.u32Leb()}),
        0x0D => try out.print(a, "br_if {d}", .{r.u32Leb()}),
        0x0E => { // br_table vec(labelidx) default
            const n = r.u32Leb();
            try out.appendSlice(a, "br_table");
            // Cap by code length: each label consumes at least one byte, so a count
            // exceeding the buffer is malformed. Bounds the loop over untrusted input.
            for (0..@min(n, code.len)) |_| try out.print(a, " {d}", .{r.u32Leb()});
            try out.print(a, " (default {d})", .{r.u32Leb()});
        },
        0x10 => try out.print(a, "call {d}", .{r.u32Leb()}),
        0x11 => { // call_indirect typeidx tableidx
            const ty = r.u32Leb();
            try out.print(a, "call_indirect {d} {d}", .{ ty, r.u32Leb() });
        },
        0x20 => try out.print(a, "local.get {d}", .{r.u32Leb()}),
        0x21 => try out.print(a, "local.set {d}", .{r.u32Leb()}),
        0x22 => try out.print(a, "local.tee {d}", .{r.u32Leb()}),
        0x23 => try out.print(a, "global.get {d}", .{r.u32Leb()}),
        0x24 => try out.print(a, "global.set {d}", .{r.u32Leb()}),
        0x28...0x3E => { // memory load/store: align, offset
            const al = r.u32Leb();
            const off = r.u32Leb();
            // `al` is a log2 alignment from untrusted bytes; a value >= 64 would make
            // the shift undefined, so clamp it (0 flags the malformed alignment).
            const align_bytes: u64 = if (al < 64) @as(u64, 1) << @intCast(al) else 0;
            try out.print(a, "{s} offset={d} align={d}", .{ memName(op), off, align_bytes });
        },
        0x3F => {
            _ = r.byte(); // memidx (0x00)
            try out.appendSlice(a, "memory.size");
        },
        0x40 => {
            _ = r.byte();
            try out.appendSlice(a, "memory.grow");
        },
        0x41 => try out.print(a, "i32.const {d}", .{r.s64Leb()}),
        0x42 => try out.print(a, "i64.const {d}", .{r.s64Leb()}),
        0x43 => try out.print(a, "f32.const {e}", .{r.raw(f32)}),
        0x44 => try out.print(a, "f64.const {e}", .{r.raw(f64)}),
        else => if (simple(op)) |m| {
            try out.appendSlice(a, m);
        } else {
            try out.print(a, ".byte 0x{x:0>2}", .{op});
        },
    }
    return r.pos - pos;
}

fn valType(b: u8) []const u8 {
    return switch (b) {
        0x7F => "i32",
        0x7E => "i64",
        0x7D => "f32",
        0x7C => "f64",
        else => "?",
    };
}

fn memName(op: u8) []const u8 {
    return switch (op) {
        0x28 => "i32.load",
        0x29 => "i64.load",
        0x2A => "f32.load",
        0x2B => "f64.load",
        0x2C => "i32.load8_s",
        0x2D => "i32.load8_u",
        0x2E => "i32.load16_s",
        0x2F => "i32.load16_u",
        0x30 => "i64.load8_s",
        0x31 => "i64.load8_u",
        0x32 => "i64.load16_s",
        0x33 => "i64.load16_u",
        0x34 => "i64.load32_s",
        0x35 => "i64.load32_u",
        0x36 => "i32.store",
        0x37 => "i64.store",
        0x38 => "f32.store",
        0x39 => "f64.store",
        0x3A => "i32.store8",
        0x3B => "i32.store16",
        0x3C => "i64.store8",
        0x3D => "i64.store16",
        0x3E => "i64.store32",
        else => "??",
    };
}

/// The mnemonic of a no-immediate opcode (control, comparisons, numeric, conversions), or
/// null if `op` carries an immediate or is unknown.
fn simple(op: u8) ?[]const u8 {
    return switch (op) {
        0x00 => "unreachable",
        0x01 => "nop",
        0x05 => "else",
        0x0B => "end",
        0x0F => "return",
        0x1A => "drop",
        0x1B => "select",
        // i32 comparisons
        0x45 => "i32.eqz",
        0x46 => "i32.eq",
        0x47 => "i32.ne",
        0x48 => "i32.lt_s",
        0x49 => "i32.lt_u",
        0x4A => "i32.gt_s",
        0x4B => "i32.gt_u",
        0x4C => "i32.le_s",
        0x4D => "i32.le_u",
        0x4E => "i32.ge_s",
        0x4F => "i32.ge_u",
        // i64 comparisons
        0x50 => "i64.eqz",
        0x51 => "i64.eq",
        0x52 => "i64.ne",
        0x53 => "i64.lt_s",
        0x54 => "i64.lt_u",
        0x55 => "i64.gt_s",
        0x56 => "i64.gt_u",
        0x57 => "i64.le_s",
        0x58 => "i64.le_u",
        0x59 => "i64.ge_s",
        0x5A => "i64.ge_u",
        // f32 / f64 comparisons
        0x5B => "f32.eq",
        0x5C => "f32.ne",
        0x5D => "f32.lt",
        0x5E => "f32.gt",
        0x5F => "f32.le",
        0x60 => "f32.ge",
        0x61 => "f64.eq",
        0x62 => "f64.ne",
        0x63 => "f64.lt",
        0x64 => "f64.gt",
        0x65 => "f64.le",
        0x66 => "f64.ge",
        // i32 numeric
        0x67 => "i32.clz",
        0x68 => "i32.ctz",
        0x69 => "i32.popcnt",
        0x6A => "i32.add",
        0x6B => "i32.sub",
        0x6C => "i32.mul",
        0x6D => "i32.div_s",
        0x6E => "i32.div_u",
        0x6F => "i32.rem_s",
        0x70 => "i32.rem_u",
        0x71 => "i32.and",
        0x72 => "i32.or",
        0x73 => "i32.xor",
        0x74 => "i32.shl",
        0x75 => "i32.shr_s",
        0x76 => "i32.shr_u",
        0x77 => "i32.rotl",
        0x78 => "i32.rotr",
        // i64 numeric
        0x79 => "i64.clz",
        0x7A => "i64.ctz",
        0x7B => "i64.popcnt",
        0x7C => "i64.add",
        0x7D => "i64.sub",
        0x7E => "i64.mul",
        0x7F => "i64.div_s",
        0x80 => "i64.div_u",
        0x81 => "i64.rem_s",
        0x82 => "i64.rem_u",
        0x83 => "i64.and",
        0x84 => "i64.or",
        0x85 => "i64.xor",
        0x86 => "i64.shl",
        0x87 => "i64.shr_s",
        0x88 => "i64.shr_u",
        0x89 => "i64.rotl",
        0x8A => "i64.rotr",
        // f32 numeric
        0x8B => "f32.abs",
        0x8C => "f32.neg",
        0x8D => "f32.ceil",
        0x8E => "f32.floor",
        0x8F => "f32.trunc",
        0x90 => "f32.nearest",
        0x91 => "f32.sqrt",
        0x92 => "f32.add",
        0x93 => "f32.sub",
        0x94 => "f32.mul",
        0x95 => "f32.div",
        0x96 => "f32.min",
        0x97 => "f32.max",
        0x98 => "f32.copysign",
        // f64 numeric
        0x99 => "f64.abs",
        0x9A => "f64.neg",
        0x9B => "f64.ceil",
        0x9C => "f64.floor",
        0x9D => "f64.trunc",
        0x9E => "f64.nearest",
        0x9F => "f64.sqrt",
        0xA0 => "f64.add",
        0xA1 => "f64.sub",
        0xA2 => "f64.mul",
        0xA3 => "f64.div",
        0xA4 => "f64.min",
        0xA5 => "f64.max",
        0xA6 => "f64.copysign",
        // conversions
        0xA7 => "i32.wrap_i64",
        0xA8 => "i32.trunc_f32_s",
        0xA9 => "i32.trunc_f32_u",
        0xAA => "i32.trunc_f64_s",
        0xAB => "i32.trunc_f64_u",
        0xAC => "i64.extend_i32_s",
        0xAD => "i64.extend_i32_u",
        0xAE => "i64.trunc_f32_s",
        0xAF => "i64.trunc_f32_u",
        0xB0 => "i64.trunc_f64_s",
        0xB1 => "i64.trunc_f64_u",
        0xB2 => "f32.convert_i32_s",
        0xB3 => "f32.convert_i32_u",
        0xB4 => "f32.convert_i64_s",
        0xB5 => "f32.convert_i64_u",
        0xB6 => "f32.demote_f64",
        0xB7 => "f64.convert_i32_s",
        0xB8 => "f64.convert_i32_u",
        0xB9 => "f64.convert_i64_s",
        0xBA => "f64.convert_i64_u",
        0xBB => "f64.promote_f32",
        0xBC => "i32.reinterpret_f32",
        0xBD => "i64.reinterpret_f64",
        0xBE => "f32.reinterpret_i32",
        0xBF => "f64.reinterpret_i64",
        0xC0 => "i32.extend8_s",
        0xC1 => "i32.extend16_s",
        0xC2 => "i64.extend8_s",
        0xC3 => "i64.extend16_s",
        0xC4 => "i64.extend32_s",
        else => null,
    };
}

test "disassembles a small function body" {
    const a = std.testing.allocator;
    // local.get 0; local.get 1; i32.add; end
    const body = [_]u8{ 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B };
    const text = try format(a, &body);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\0000: local.get 0
        \\0002: local.get 1
        \\0004: i32.add
        \\0005: end
        \\
    , text);
}

test "decodes immediates: consts, calls, memargs, br_table" {
    const a = std.testing.allocator;
    // i32.const -3 (LEB 0x7D); i32.const 200 (LEB 0xC8 0x01); call 5; i32.load offset=8 align=4(2^2)
    const body = [_]u8{ 0x41, 0x7D, 0x41, 0xC8, 0x01, 0x10, 0x05, 0x28, 0x02, 0x08 };
    const text = try format(a, &body);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\0000: i32.const -3
        \\0002: i32.const 200
        \\0005: call 5
        \\0007: i32.load offset=8 align=4
        \\
    , text);
}

test "block nesting indents the listing" {
    const a = std.testing.allocator;
    // block (result i32); i32.const 1; if; i32.const 2; else; i32.const 3; end; end
    const body = [_]u8{ 0x02, 0x7F, 0x41, 0x01, 0x04, 0x40, 0x41, 0x02, 0x05, 0x41, 0x03, 0x0B, 0x0B };
    const text = try format(a, &body);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\0000: block (result i32)
        \\0002:   i32.const 1
        \\0004:   if
        \\0006:     i32.const 2
        \\0008:   else
        \\0009:     i32.const 3
        \\000b:   end
        \\000c: end
        \\
    , text);
}

test "formatBody parses the locals header then the expression" {
    const a = std.testing.allocator;
    // locals: one i32 (01 01 7F); local.get 0; local.get 1; i32.add; end
    const body = [_]u8{ 0x01, 0x01, 0x7F, 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B };
    const text = try formatBody(a, &body);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\locals: 1 x i32
        \\0000: local.get 0
        \\0002: local.get 1
        \\0004: i32.add
        \\0005: end
        \\
    , text);
}

test "br_table lists its labels and default" {
    const a = std.testing.allocator;
    // br_table 0 1 (default 2)
    const body = [_]u8{ 0x0E, 0x02, 0x00, 0x01, 0x02 };
    const text = try format(a, &body);
    defer a.free(text);
    try std.testing.expectEqualStrings("0000: br_table 0 1 (default 2)\n", text);
}
