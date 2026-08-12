//! ABI struct/union classification. `classify` maps a struct/union `CType` plus a
//! target `layout.TargetLayout` to how that aggregate is passed as an argument and
//! returned, per the real per-arch ABI: x86_64 SysV eightbyte classification, AAPCS64,
//! including the Homogeneous Float Aggregate rule, RISC-V lp64d's FP-struct
//! flattening, and plain i386 SysV, which is always memory. This module is pure and
//! total. It does no IR emission and no allocation, since every list is an inline
//! fixed array, mirroring `ir.function.Ret`'s style.

const std = @import("std");
const ctype = @import("ctype.zig");
const layout = @import("layout.zig");

/// A register-sized piece's class: which register file it goes in.
pub const Class = enum { integer, sse };

/// One eightbyte-sized, or smaller for a trailing partial piece, chunk of an aggregate,
/// classified for one argument or return register. `bytes` is at most 8 for `.integer`
/// and either 4 or 8 for `.sse`: a lone `float`'s eightbyte, or one packed with a second
/// `float`, or a `double`'s. `offset` is this piece's byte offset from the start of the
/// aggregate. x86_64 windows and RISC-V's flattened fields can start anywhere, while an
/// integer eightbyte's offset is always `index * 8`.
pub const Eightbyte = struct { class: Class, bytes: u8, offset: u8 };

/// An inline, allocation-free list of up to 4 `Eightbyte`s, mirroring
/// `ir.function.Ret`'s `values` and `count` shape. 4 covers every target here: the
/// largest register-passed aggregate is 2 eightbytes on x86_64 and riscv64, and a
/// 4-`float` AArch64 HFA needs 4 slots.
pub const EightbyteList = struct {
    items: [4]Eightbyte = undefined,
    count: u8 = 0,

    pub fn slice(self: *const EightbyteList) []const Eightbyte {
        return self.items[0..self.count];
    }
};

/// Build an `EightbyteList` from up to 4 pieces, in order. `items` must hold at most 4.
fn eightbytes(items: []const Eightbyte) EightbyteList {
    std.debug.assert(items.len <= 4);
    var r: EightbyteList = .{ .count = @intCast(items.len) };
    for (items, 0..) |it, i| r.items[i] = it;
    return r;
}

/// How a struct/union argument reaches the callee.
pub const ArgPlan = union(enum) {
    /// Decomposed into these scalars, each in the next GP or FP argument register.
    registers: EightbyteList,
    /// A pointer to a caller-made copy, passed as an ordinary integer argument (AAPCS64/
    /// RISC-V's indirect path for an oversized, non-HFA aggregate).
    memory_ref,
    /// The struct's raw bytes, passed by value on the stack (SysV's `MEMORY` class, and
    /// i386's only aggregate-passing mode).
    memory_stack,
};

/// How a struct/union return value reaches the caller.
pub const ReturnPlan = union(enum) {
    /// The callee places these into the return registers. The caller reads them back.
    registers: EightbyteList,
    /// A hidden pointer: the caller allocates the return slot and passes its address.
    /// The callee copies the result there and also returns that same address.
    sret,
};

pub const ClassResult = struct { arg: ArgPlan, ret: ReturnPlan };

/// Write `class` into `out[base..base + size)`, clipped to `out`'s 16-byte span. A
/// struct past 16 bytes is already routed to memory before this is ever called, so
/// bytes beyond the clip are never read back.
fn markRange(out: *[16]?Class, base: u64, size: u64, class: Class) void {
    var i: u64 = 0;
    while (i < size) : (i += 1) {
        const idx = base + i;
        if (idx < 16) out[idx] = class;
    }
}

/// Flatten `ty`, recursing through nested structs and arrays by absolute byte offset,
/// into `out`'s per-byte class map for x86_64 SysV merging. A leaf `float` marks its
/// bytes `.sse`. Every other leaf, such as int, pointer, function, or void, marks its
/// bytes `.integer`. A byte no field covers stays `null`, which the merge in
/// `classifyX86_64` reads as SysV NO_CLASS: padding that merges as the identity, so it
/// never forces an eightbyte to `.integer`. A nested union's bytes are marked
/// `.integer` without recursing into its overlapping members, the same conservative
/// "classify by size" stance `classifyX86_64` takes for a top-level union.
fn collectLeavesX86_64(ty: ctype.CType, base_offset: u64, l: layout.TargetLayout, out: *[16]?Class) void {
    switch (ty) {
        .@"struct" => |s| {
            if (s.is_union) {
                markRange(out, base_offset, ty.sizeInBytes(l) catch 0, .integer);
                return;
            }
            for (s.fields) |f| collectLeavesX86_64(f.ty, base_offset + f.offset, l, out);
        },
        .array => |a| {
            const stride = a.elem.storageSize(l) catch (a.elem.sizeInBytes(l) catch 0);
            var i: u64 = 0;
            while (i < a.len) : (i += 1) collectLeavesX86_64(a.elem.*, base_offset + i * stride, l, out);
        },
        .float => markRange(out, base_offset, ty.sizeInBytes(l) catch 0, .sse),
        else => markRange(out, base_offset, ty.sizeInBytes(l) catch 0, .integer),
    }
}

/// True if any of `def`'s own fields sits at an offset that is not a multiple of its
/// own alignment. This is the `#pragma pack` case that forces SysV `MEMORY` even under
/// 16 bytes, since an unaligned field cannot land cleanly in a register.
fn hasUnalignedField(def: *const ctype.StructDef, l: layout.TargetLayout) bool {
    for (def.fields) |f| {
        const a = f.ty.alignOf(l) catch return true;
        if (f.offset % a != 0) return true;
    }
    return false;
}

fn classifyX86_64(ty: ctype.CType, l: layout.TargetLayout) ClassResult {
    const def = ty.@"struct";
    const size = ty.sizeInBytes(l) catch 0;
    if (size > 16 or hasUnalignedField(def, l)) {
        return .{ .arg = .memory_stack, .ret = .sret };
    }
    var byte_class: [16]?Class = .{null} ** 16;
    if (!def.is_union) collectLeavesX86_64(ty, 0, l, &byte_class);

    var pieces: [4]Eightbyte = undefined;
    var count: u8 = 0;
    var offset: u64 = 0;
    while (offset < size) : (offset += 8) {
        const window: u8 = @intCast(@min(8, size - offset));
        // A union's members overlap, so its bytes are never individually classified above.
        // The map stays all-`null`, and the eightbyte is treated as plain `.integer`. This
        // matches the conservative "classify by size" stance for unions.
        var class: Class = .integer;
        if (!def.is_union) {
            // Merge the mapped byte classes per the SysV rule. A real integer byte forces
            // `.integer` (INTEGER wins over SSE). An unmapped byte is padding, which is
            // NO_CLASS and merges as the identity, so it is skipped. An eightbyte holding a
            // float plus trailing padding stays `.sse`. An all-padding eightbyte cannot occur
            // within a real struct's size, so it defaults to `.integer` defensively.
            var has_sse = false;
            var has_int = false;
            var i: u64 = 0;
            while (i < window) : (i += 1) {
                if (byte_class[offset + i]) |c| {
                    if (c == .sse) has_sse = true else has_int = true;
                }
            }
            class = if (has_int) .integer else if (has_sse) .sse else .integer;
        }
        pieces[count] = .{ .class = class, .bytes = window, .offset = @intCast(offset) };
        count += 1;
    }
    const list = eightbytes(pieces[0..count]);
    return .{ .arg = .{ .registers = list }, .ret = .{ .registers = list } };
}

/// A struct or union `ty`'s flattened float leaves for the AArch64 Homogeneous Float
/// Aggregate test: `ty` qualifies if every recursively flattened member is the same
/// `FloatKind` and there are 1 to 4 of them. Returns that shared kind and count, or
/// `null` if `ty` does not qualify.
fn hfaKind(ty: ctype.CType) ?struct { kind: ctype.FloatKind, count: u8 } {
    var count: u32 = 0;
    var kind: ?ctype.FloatKind = null;
    if (!hfaFlatten(ty, &count, &kind) or count == 0 or count > 4) return null;
    return .{ .kind = kind.?, .count = @intCast(count) };
}

fn hfaFlatten(ty: ctype.CType, count: *u32, kind: *?ctype.FloatKind) bool {
    switch (ty) {
        .float => |fk| {
            if (kind.*) |k| {
                if (k != fk) return false;
            } else {
                kind.* = fk;
            }
            count.* += 1;
            return count.* <= 4;
        },
        .@"struct" => |s| {
            // A union's members overlap rather than concatenate, so it is never an HFA under
            // this flatten (matches the conservative union stance elsewhere in this file).
            if (s.is_union or s.fields.len == 0) return false;
            for (s.fields) |f| {
                if (!hfaFlatten(f.ty, count, kind)) return false;
            }
            return true;
        },
        .array => |a| {
            if (a.len == 0) return false;
            var i: u64 = 0;
            while (i < a.len) : (i += 1) {
                if (!hfaFlatten(a.elem.*, count, kind)) return false;
            }
            return true;
        },
        else => return false,
    }
}

/// `ceil(size / 8)` plain `.integer` eightbytes, the shared "small aggregate, no float
/// special case" path for AArch64, when not an HFA, and RISC-V, when not flattened.
fn integerEightbytes(size: u64) EightbyteList {
    const n: u8 = @intCast((size + 7) / 8);
    var pieces: [4]Eightbyte = undefined;
    var i: u8 = 0;
    while (i < n) : (i += 1) {
        const remaining = size - @as(u64, i) * 8;
        pieces[i] = .{ .class = .integer, .bytes = @intCast(@min(8, remaining)), .offset = i * 8 };
    }
    return eightbytes(pieces[0..n]);
}

fn classifyAArch64(ty: ctype.CType, l: layout.TargetLayout) ClassResult {
    const size = ty.sizeInBytes(l) catch 0;
    if (hfaKind(ty)) |hfa| {
        const bytes: u8 = if (hfa.kind == .f64) 8 else 4;
        var pieces: [4]Eightbyte = undefined;
        var i: u8 = 0;
        while (i < hfa.count) : (i += 1) pieces[i] = .{ .class = .sse, .bytes = bytes, .offset = i * bytes };
        const list = eightbytes(pieces[0..hfa.count]);
        return .{ .arg = .{ .registers = list }, .ret = .{ .registers = list } };
    }
    if (size <= 16) {
        const list = integerEightbytes(size);
        return .{ .arg = .{ .registers = list }, .ret = .{ .registers = list } };
    }
    return .{ .arg = .memory_ref, .ret = .sret };
}

/// `def`'s fields for RISC-V's FP-struct flatten test: `def`'s own fields, unless it
/// has exactly one field that is itself a non-union struct, in which case that inner
/// struct's fields are used instead. A single-field wrapper struct flattens one level
/// before the at-most-2-fields test applies.
fn riscvFlattenFields(def: *const ctype.StructDef) []const ctype.Field {
    if (def.fields.len == 1 and def.fields[0].ty == .@"struct" and !def.fields[0].ty.@"struct".is_union) {
        return def.fields[0].ty.@"struct".fields;
    }
    return def.fields;
}

fn classifyRiscv64(ty: ctype.CType, l: layout.TargetLayout) ClassResult {
    const def = ty.@"struct";
    const size = ty.sizeInBytes(l) catch 0;
    if (!def.is_union) {
        const flat = riscvFlattenFields(def);
        // lp64d's hardware FP calling convention: at most 2 fields, each at most XLEN,
        // 8 bytes, with at least one floating field and at most one integer field. This
        // is what makes `{float; float;}` pass in two FPRs and `{int; double;}` pass in
        // one GPR plus one FPR.
        if (flat.len >= 1 and flat.len <= 2) {
            var float_count: u8 = 0;
            var int_count: u8 = 0;
            var fits = true;
            for (flat) |f| {
                const fsize = f.ty.sizeInBytes(l) catch 0;
                if (fsize == 0 or fsize > 8) {
                    fits = false;
                    break;
                }
                if (f.ty.isFloat()) float_count += 1 else int_count += 1;
            }
            if (fits and float_count >= 1 and int_count <= 1) {
                var pieces: [4]Eightbyte = undefined;
                for (flat, 0..) |f, i| {
                    const fsize: u8 = @intCast(f.ty.sizeInBytes(l) catch 0);
                    pieces[i] = .{ .class = if (f.ty.isFloat()) .sse else .integer, .bytes = fsize, .offset = @intCast(f.offset) };
                }
                const list = eightbytes(pieces[0..flat.len]);
                return .{ .arg = .{ .registers = list }, .ret = .{ .registers = list } };
            }
        }
    }
    if (size <= 16) {
        const list = integerEightbytes(size);
        return .{ .arg = .{ .registers = list }, .ret = .{ .registers = list } };
    }
    return .{ .arg = .memory_ref, .ret = .sret };
}

/// i386 SysV passes every struct or union by value on the stack and returns every one
/// through a hidden pointer. There is no register classification and no FP registers
/// for aggregates, regardless of size or field types.
fn classifyX86(ty: ctype.CType, l: layout.TargetLayout) ClassResult {
    _ = ty;
    _ = l;
    return .{ .arg = .memory_stack, .ret = .sret };
}

/// The ABI passing and return plan for struct or union `ty` under target `l`. `ty`
/// must be a `.@"struct"` `CType`, either a struct or a union, and `StructDef.is_union`
/// tells them apart. Every per-arch rule below handles both, treating a union
/// conservatively (see `classifyX86_64`'s and `hfaFlatten`'s doc comments).
pub fn classify(ty: ctype.CType, l: layout.TargetLayout) ClassResult {
    std.debug.assert(ty == .@"struct");
    return switch (l.arch) {
        .x86_64 => classifyX86_64(ty, l),
        .aarch64 => classifyAArch64(ty, l),
        .riscv64 => classifyRiscv64(ty, l),
        .x86 => classifyX86(ty, l),
    };
}

const testing = std.testing;

const float_t: ctype.CType = .{ .float = .f32 };
const double_t: ctype.CType = .{ .float = .f64 };

fn expectEightbytes(expected: []const Eightbyte, actual: EightbyteList) !void {
    try testing.expectEqual(expected.len, actual.count);
    for (expected, actual.slice()) |e, a| {
        try testing.expectEqual(e.class, a.class);
        try testing.expectEqual(e.bytes, a.bytes);
        try testing.expectEqual(e.offset, a.offset);
    }
}

test "x86_64: struct{int;int;} is one integer eightbyte" {
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.int_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.int_t, .offset = 4 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 8, .alignment = 4 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{.{ .class = .integer, .bytes = 8, .offset = 0 }}, r.arg.registers);
    try expectEightbytes(&.{.{ .class = .integer, .bytes = 8, .offset = 0 }}, r.ret.registers);
}

test "x86_64: struct{double;double;} is two sse eightbytes" {
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = double_t, .offset = 0 },
        .{ .name = "b", .ty = double_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .sse, .bytes = 8, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "x86_64: struct{int;double;} is integer then sse" {
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.int_t, .offset = 0 },
        .{ .name = "b", .ty = double_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .integer, .bytes = 8, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "x86_64: struct{float;double;} first eightbyte is sse (trailing padding is NO_CLASS)" {
    // float at 0 (4B), 4B alignment padding at 4-7, double at 8 (8B), whole struct 16B align 8.
    // The padding after the float is SysV NO_CLASS, so eightbyte 0 stays SSE, NOT integer. gcc
    // passes both eightbytes in xmm0/xmm1.
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = float_t, .offset = 0 },
        .{ .name = "b", .ty = double_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .sse, .bytes = 8, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "x86_64: struct{double;float;} second eightbyte is sse (trailing padding is NO_CLASS)" {
    // double at 0 (8B), float at 8 (4B), 4B trailing padding at 12-15, whole struct 16B.
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = double_t, .offset = 0 },
        .{ .name = "b", .ty = float_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .sse, .bytes = 8, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "x86_64: struct{char;double;} first eightbyte is integer (a real int byte forces it)" {
    // char at 0 (1B), 7B padding at 1-7, double at 8. Eightbyte 0 has a real integer byte, so
    // it is INTEGER (the padding does not change that). Eightbyte 1 is SSE.
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.char_t, .offset = 0 },
        .{ .name = "b", .ty = double_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .integer, .bytes = 8, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "x86_64: struct{long;long;long;} (24B) is memory" {
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.long_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.long_t, .offset = 8 },
        .{ .name = "c", .ty = ctype.long_t, .offset = 16 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 24, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try testing.expect(r.arg == .memory_stack);
    try testing.expect(r.ret == .sret);
}

test "x86_64: struct{float;float;} packs into one sse eightbyte" {
    const l = layout.forArch(.x86_64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = float_t, .offset = 0 },
        .{ .name = "b", .ty = float_t, .offset = 4 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 8, .alignment = 4 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{.{ .class = .sse, .bytes = 8, .offset = 0 }}, r.arg.registers);
}

test "aarch64: struct{double;double;} is an HFA of two sse eightbytes" {
    const l = layout.forArch(.aarch64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = double_t, .offset = 0 },
        .{ .name = "b", .ty = double_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .sse, .bytes = 8, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "aarch64: struct{float x4;} is an HFA of four 4-byte sse eightbytes" {
    const l = layout.forArch(.aarch64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = float_t, .offset = 0 },
        .{ .name = "b", .ty = float_t, .offset = 4 },
        .{ .name = "c", .ty = float_t, .offset = 8 },
        .{ .name = "d", .ty = float_t, .offset = 12 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 4 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{
        .{ .class = .sse, .bytes = 4, .offset = 0 },
        .{ .class = .sse, .bytes = 4, .offset = 4 },
        .{ .class = .sse, .bytes = 4, .offset = 8 },
        .{ .class = .sse, .bytes = 4, .offset = 12 },
    }, r.arg.registers);
}

test "aarch64: struct{long;long;} (not HFA) is two integer eightbytes" {
    const l = layout.forArch(.aarch64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.long_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.long_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .integer, .bytes = 8, .offset = 0 }, .{ .class = .integer, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "aarch64: struct{long;long;long;} (24B, not HFA) is memory_ref/sret" {
    const l = layout.forArch(.aarch64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.long_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.long_t, .offset = 8 },
        .{ .name = "c", .ty = ctype.long_t, .offset = 16 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 24, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try testing.expect(r.arg == .memory_ref);
    try testing.expect(r.ret == .sret);
}

test "riscv64: struct{float;float;} flattens into two sse pieces" {
    const l = layout.forArch(.riscv64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = float_t, .offset = 0 },
        .{ .name = "b", .ty = float_t, .offset = 4 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 8, .alignment = 4 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .sse, .bytes = 4, .offset = 0 }, .{ .class = .sse, .bytes = 4, .offset = 4 } }, r.arg.registers);
}

test "riscv64: struct{int;double;} flattens into integer then sse" {
    const l = layout.forArch(.riscv64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.int_t, .offset = 0 },
        .{ .name = "b", .ty = double_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 16, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try expectEightbytes(&.{ .{ .class = .integer, .bytes = 4, .offset = 0 }, .{ .class = .sse, .bytes = 8, .offset = 8 } }, r.arg.registers);
}

test "riscv64: struct{long;long;long;} (24B) is memory_ref/sret" {
    const l = layout.forArch(.riscv64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.long_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.long_t, .offset = 8 },
        .{ .name = "c", .ty = ctype.long_t, .offset = 16 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 24, .alignment = 8 };
    const r = classify(.{ .@"struct" = &def }, l);
    try testing.expect(r.arg == .memory_ref);
    try testing.expect(r.ret == .sret);
}

test "riscv64: struct{int;int;int;} (12B, no float, 3 fields) skips flatten, two integer eightbytes" {
    const l = layout.forArch(.riscv64);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.int_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.int_t, .offset = 4 },
        .{ .name = "c", .ty = ctype.int_t, .offset = 8 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 12, .alignment = 4 };
    const r = classify(.{ .@"struct" = &def }, l);
    try testing.expectEqual(@as(u8, 2), r.arg.registers.count);
    for (r.arg.registers.slice()) |piece| try testing.expectEqual(Class.integer, piece.class);
}

test "x86: any struct is memory_stack/sret" {
    const l = layout.forArch(.x86);
    const fields = [_]ctype.Field{
        .{ .name = "a", .ty = ctype.int_t, .offset = 0 },
        .{ .name = "b", .ty = ctype.int_t, .offset = 4 },
    };
    const def: ctype.StructDef = .{ .name = "S", .is_union = false, .fields = &fields, .size = 8, .alignment = 4 };
    const r = classify(.{ .@"struct" = &def }, l);
    try testing.expect(r.arg == .memory_stack);
    try testing.expect(r.ret == .sret);
}
