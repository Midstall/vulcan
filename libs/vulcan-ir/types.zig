//! The IR type system. Types are interned: structurally identical types share
//! one `Type` handle, so equality is an index comparison. Types carry no layout
//! (size, alignment, packing). They are semantic only, with bytes decided later
//! behind the target seam.

const std = @import("std");

/// A handle to an interned type. Equality is index equality.
pub const Type = enum(u32) { _ };

/// An integer type: a signedness and a bit width.
pub const Int = struct {
    signedness: std.builtin.Signedness,
    bits: u16,
};

/// The floating-point formats in the primitive core.
pub const FloatKind = enum { f32, f64 };

/// A fixed-length SIMD vector over a primitive scalar element.
pub const Vector = struct {
    len: u32,
    elem: Type,
};

/// A fixed-length array of an element type. High profile only.
pub const Array = struct {
    len: u64,
    elem: Type,
};

/// A fat pointer: an address plus a runtime length over an element type. High
/// profile only. Legalizes to a `{ ptr, i64 }` pair. The length is runtime, so
/// it is not part of the type.
pub const Slice = struct {
    elem: Type,
};

/// The structural description of a type. Identical descriptions intern to the
/// same `Type` handle.
pub const TypeKind = union(enum) {
    /// A native boolean. Its physical width is a codegen decision, not fixed here.
    bool,
    /// An integer with a signedness and bit width.
    int: Int,
    /// A floating-point value.
    float: FloatKind,
    /// An opaque, typeless pointer. Address math is explicit.
    ptr,
    /// A fixed-length SIMD vector over a primitive scalar.
    vector: Vector,
    /// An aggregate of ordered field types. High profile only. The slice is
    /// owned by the `TypeTable` once interned.
    @"struct": []const Type,
    /// A fixed-length array. High profile only.
    array: Array,
    /// A fat pointer (address plus runtime length) over an element. High profile only.
    slice: Slice,
};

/// Hash/equality over `TypeKind` by *content*, so structs intern by their field
/// types rather than by slice identity.
const TypeContext = struct {
    pub fn hash(_: TypeContext, key: TypeKind) u64 {
        var hasher = std.hash.Wyhash.init(0);
        std.hash.autoHashStrat(&hasher, key, .Deep);
        return hasher.final();
    }

    pub fn eql(_: TypeContext, a: TypeKind, b: TypeKind) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .bool, .ptr => true,
            .int => a.int.signedness == b.int.signedness and a.int.bits == b.int.bits,
            .float => a.float == b.float,
            .vector => a.vector.len == b.vector.len and a.vector.elem == b.vector.elem,
            .array => a.array.len == b.array.len and a.array.elem == b.array.elem,
            .slice => a.slice.elem == b.slice.elem,
            .@"struct" => std.mem.eql(Type, a.@"struct", b.@"struct"),
        };
    }
};

/// Interns types so structurally identical types share one `Type` handle.
pub const TypeTable = struct {
    const DedupMap = std.HashMapUnmanaged(TypeKind, Type, TypeContext, std.hash_map.default_max_load_percentage);

    allocator: std.mem.Allocator,
    kinds: std.ArrayList(TypeKind),
    dedup: DedupMap,

    pub fn init(allocator: std.mem.Allocator) TypeTable {
        return .{ .allocator = allocator, .kinds = .empty, .dedup = .empty };
    }

    pub fn deinit(self: *TypeTable) void {
        for (self.kinds.items) |kind| self.freeKind(kind);
        self.kinds.deinit(self.allocator);
        self.dedup.deinit(self.allocator);
    }

    /// Intern a type, returning its handle. Structurally identical kinds return
    /// the same handle. Slice-bearing kinds (structs) are copied into storage the
    /// table owns, so the handle stays valid regardless of the caller's memory.
    pub fn intern(self: *TypeTable, kind: TypeKind) std.mem.Allocator.Error!Type {
        if (self.dedup.getContext(kind, .{})) |existing| return existing;

        const owned = try self.ownKind(kind);
        errdefer self.freeKind(owned);

        const handle: Type = @enumFromInt(@as(u32, @intCast(self.kinds.items.len)));
        try self.kinds.append(self.allocator, owned);
        errdefer _ = self.kinds.pop();

        try self.dedup.putContext(self.allocator, owned, handle, .{});
        return handle;
    }

    /// Borrow the structural kind backing a handle.
    pub fn type_kind(self: *const TypeTable, handle: Type) TypeKind {
        return self.kinds.items[@intFromEnum(handle)];
    }

    /// The number of interned types (handles are `0..count`).
    pub fn count(self: *const TypeTable) usize {
        return self.kinds.items.len;
    }

    /// Wrap a type handle so it renders itself via `{f}`.
    pub fn fmt(self: *const TypeTable, ty: Type) TypeFormatter {
        return .{ .table = self, .ty = ty };
    }

    /// Parse a textual type, interning it. The whole text must be one type.
    pub fn parseType(self: *TypeTable, text: []const u8) (ParseError || std.mem.Allocator.Error)!Type {
        const parsed = try self.parseTypePrefix(text);
        if (parsed.len != text.len) return error.InvalidType;
        return parsed.ty;
    }

    /// Parse a type from the start of `text`, returning it and how many bytes it
    /// consumed. Lets callers parse a type embedded in a larger string.
    pub fn parseTypePrefix(self: *TypeTable, text: []const u8) (ParseError || std.mem.Allocator.Error)!struct { ty: Type, len: usize } {
        var p = TypeParser{ .table = self, .src = text };
        const ty = try p.parse();
        return .{ .ty = ty, .len = p.pos };
    }

    /// Copy any slice payload into table-owned memory.
    fn ownKind(self: *TypeTable, kind: TypeKind) std.mem.Allocator.Error!TypeKind {
        return switch (kind) {
            .@"struct" => |fields| .{ .@"struct" = try self.allocator.dupe(Type, fields) },
            else => kind,
        };
    }

    /// Free any table-owned slice payload.
    fn freeKind(self: *TypeTable, kind: TypeKind) void {
        switch (kind) {
            .@"struct" => |fields| self.allocator.free(fields),
            else => {},
        }
    }
};

/// Errors the type parser can produce (beyond allocation failure).
pub const ParseError = error{InvalidType};

fn isLetter(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isWordChar(c: u8) bool {
    return isLetter(c) or isDigit(c);
}

/// A cursor-based recursive-descent parser over a single type's text.
const TypeParser = struct {
    table: *TypeTable,
    src: []const u8,
    pos: usize = 0,

    const Error = ParseError || std.mem.Allocator.Error;

    fn peek(self: *TypeParser) ?u8 {
        return if (self.pos < self.src.len) self.src[self.pos] else null;
    }

    fn skipWs(self: *TypeParser) void {
        while (self.pos < self.src.len and self.src[self.pos] == ' ') : (self.pos += 1) {}
    }

    /// Consume `c`, erroring if it is not next.
    fn eat(self: *TypeParser, c: u8) Error!void {
        if (self.peek() == c) {
            self.pos += 1;
        } else {
            return error.InvalidType;
        }
    }

    /// Consume `c` if it is next, reporting whether it was.
    fn tryEat(self: *TypeParser, c: u8) bool {
        if (self.peek() == c) {
            self.pos += 1;
            return true;
        }
        return false;
    }

    /// Consume and return a run of word characters (letters and digits).
    fn readWord(self: *TypeParser) []const u8 {
        const start = self.pos;
        while (self.pos < self.src.len and isWordChar(self.src[self.pos])) : (self.pos += 1) {}
        return self.src[start..self.pos];
    }

    fn readNumber(self: *TypeParser) Error!u64 {
        const start = self.pos;
        while (self.pos < self.src.len and isDigit(self.src[self.pos])) : (self.pos += 1) {}
        if (self.pos == start) return error.InvalidType;
        return std.fmt.parseInt(u64, self.src[start..self.pos], 10) catch error.InvalidType;
    }

    fn parse(self: *TypeParser) Error!Type {
        self.skipWs();
        const c = self.peek() orelse return error.InvalidType;
        return switch (c) {
            '<' => self.parseVector(),
            '[' => self.parseArrayOrSlice(),
            '{' => self.parseStruct(),
            else => if (isLetter(c)) self.parseScalar() else error.InvalidType,
        };
    }

    fn parseVector(self: *TypeParser) Error!Type {
        try self.eat('<');
        self.skipWs();
        const len = try self.readNumber();
        self.skipWs();
        try self.eat('x');
        const elem = try self.parse();
        self.skipWs();
        try self.eat('>');
        return self.table.intern(.{ .vector = .{ .len = @intCast(len), .elem = elem } });
    }

    fn parseArrayOrSlice(self: *TypeParser) Error!Type {
        try self.eat('[');
        if (self.tryEat(']')) {
            const elem = try self.parse();
            return self.table.intern(.{ .slice = .{ .elem = elem } });
        }
        self.skipWs();
        const len = try self.readNumber();
        self.skipWs();
        try self.eat('x');
        const elem = try self.parse();
        self.skipWs();
        try self.eat(']');
        return self.table.intern(.{ .array = .{ .len = len, .elem = elem } });
    }

    fn parseStruct(self: *TypeParser) Error!Type {
        try self.eat('{');
        self.skipWs();

        var fields: std.ArrayList(Type) = .empty;
        defer fields.deinit(self.table.allocator);

        if (self.peek() != '}') {
            while (true) {
                const field = try self.parse();
                try fields.append(self.table.allocator, field);
                self.skipWs();
                if (self.tryEat(',')) {
                    self.skipWs();
                    continue;
                }
                break;
            }
        }
        self.skipWs();
        try self.eat('}');
        return self.table.intern(.{ .@"struct" = fields.items });
    }

    fn parseScalar(self: *TypeParser) Error!Type {
        const word = self.readWord();
        if (std.mem.eql(u8, word, "bool")) return self.table.intern(.bool);
        if (std.mem.eql(u8, word, "ptr")) return self.table.intern(.ptr);
        if (std.mem.eql(u8, word, "f32")) return self.table.intern(.{ .float = .f32 });
        if (std.mem.eql(u8, word, "f64")) return self.table.intern(.{ .float = .f64 });
        if (word.len >= 2 and (word[0] == 'i' or word[0] == 'u')) {
            const signedness: std.builtin.Signedness = if (word[0] == 'i') .signed else .unsigned;
            const bits = std.fmt.parseInt(u16, word[1..], 10) catch return error.InvalidType;
            return self.table.intern(.{ .int = .{ .signedness = signedness, .bits = bits } });
        }
        return error.InvalidType;
    }
};

/// A type handle bundled with its table, so it can render itself via `{f}`.
pub const TypeFormatter = struct {
    table: *const TypeTable,
    ty: Type,

    pub fn format(self: TypeFormatter, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self.table.type_kind(self.ty)) {
            .bool => try w.writeAll("bool"),
            .int => |i| {
                const prefix: []const u8 = switch (i.signedness) {
                    .signed => "i",
                    .unsigned => "u",
                };
                try w.print("{s}{d}", .{ prefix, i.bits });
            },
            .float => |f| try w.writeAll(@tagName(f)),
            .ptr => try w.writeAll("ptr"),
            .vector => |v| try w.print("<{d} x {f}>", .{ v.len, self.table.fmt(v.elem) }),
            .array => |a| try w.print("[{d} x {f}]", .{ a.len, self.table.fmt(a.elem) }),
            .slice => |s| try w.print("[]{f}", .{self.table.fmt(s.elem)}),
            .@"struct" => |fields| {
                try w.writeAll("{ ");
                for (fields, 0..) |fld, idx| {
                    if (idx != 0) try w.writeAll(", ");
                    try w.print("{f}", .{self.table.fmt(fld)});
                }
                try w.writeAll(" }");
            },
        }
    }
};

test "parsing scalar types" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    try std.testing.expectEqual(
        try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
        try table.parseType("i32"),
    );
    try std.testing.expectEqual(
        try table.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } }),
        try table.parseType("u8"),
    );
    try std.testing.expectEqual(try table.intern(.bool), try table.parseType("bool"));
    try std.testing.expectEqual(try table.intern(.{ .float = .f64 }), try table.parseType("f64"));
    try std.testing.expectEqual(try table.intern(.ptr), try table.parseType("ptr"));
}

test "parsing composite types round-trips with printing" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try table.intern(.{ .float = .f32 });
    const v4 = try table.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const arr = try table.intern(.{ .array = .{ .len = 8, .elem = i32_t } });
    const sl = try table.intern(.{ .slice = .{ .elem = i32_t } });
    const st = try table.intern(.{ .@"struct" = &.{ i32_t, f32_t } });

    try std.testing.expectEqual(v4, try table.parseType("<4 x i32>"));
    try std.testing.expectEqual(arr, try table.parseType("[8 x i32]"));
    try std.testing.expectEqual(sl, try table.parseType("[]i32"));
    try std.testing.expectEqual(st, try table.parseType("{ i32, f32 }"));
}

test "printing scalar types" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const s32 = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u8_t = try table.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });

    try std.testing.expectFmt("i32", "{f}", .{table.fmt(s32)});
    try std.testing.expectFmt("u8", "{f}", .{table.fmt(u8_t)});
    try std.testing.expectFmt("bool", "{f}", .{table.fmt(try table.intern(.bool))});
    try std.testing.expectFmt("f64", "{f}", .{table.fmt(try table.intern(.{ .float = .f64 }))});
    try std.testing.expectFmt("ptr", "{f}", .{table.fmt(try table.intern(.ptr))});
}

test "printing composite types" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try table.intern(.{ .float = .f32 });

    const v4 = try table.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const arr = try table.intern(.{ .array = .{ .len = 8, .elem = i32_t } });
    const sl = try table.intern(.{ .slice = .{ .elem = i32_t } });
    const st = try table.intern(.{ .@"struct" = &.{ i32_t, f32_t } });

    try std.testing.expectFmt("<4 x i32>", "{f}", .{table.fmt(v4)});
    try std.testing.expectFmt("[8 x i32]", "{f}", .{table.fmt(arr)});
    try std.testing.expectFmt("[]i32", "{f}", .{table.fmt(sl)});
    try std.testing.expectFmt("{ i32, f32 }", "{f}", .{table.fmt(st)});
}

test "interning dedups identical primitive types" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const bool_a = try table.intern(.bool);
    const bool_b = try table.intern(.bool);
    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    try std.testing.expectEqual(bool_a, bool_b);
    try std.testing.expect(bool_a != i32_t);
}

test "integers distinguish signedness" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const s32 = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try table.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const s32_again = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    try std.testing.expectEqual(s32, s32_again);
    try std.testing.expect(s32 != u32_t);
}

test "float and pointer primitives intern distinctly" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const f32_a = try table.intern(.{ .float = .f32 });
    const f32_b = try table.intern(.{ .float = .f32 });
    const f64_t = try table.intern(.{ .float = .f64 });
    const ptr_a = try table.intern(.ptr);
    const ptr_b = try table.intern(.ptr);

    try std.testing.expectEqual(f32_a, f32_b);
    try std.testing.expectEqual(ptr_a, ptr_b);
    try std.testing.expect(f32_a != f64_t);
    try std.testing.expect(f32_a != ptr_a);
}

test "type_kind reads back the interned kind" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i64_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    try std.testing.expectEqual(TypeKind{ .int = .{ .signedness = .signed, .bits = 64 } }, table.type_kind(i64_t));
}

test "vectors intern by length and element" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v4_a = try table.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const v4_b = try table.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const v8 = try table.intern(.{ .vector = .{ .len = 8, .elem = i32_t } });

    try std.testing.expectEqual(v4_a, v4_b);
    try std.testing.expect(v4_a != v8);
}

test "structs intern by field contents, not slice identity" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try table.intern(.{ .float = .f32 });

    // Two independent slices holding identical field types.
    const fields_a = [_]Type{ i32_t, f32_t };
    const fields_b = [_]Type{ i32_t, f32_t };
    const s_a = try table.intern(.{ .@"struct" = &fields_a });
    const s_b = try table.intern(.{ .@"struct" = &fields_b });
    const s_reordered = try table.intern(.{ .@"struct" = &[_]Type{ f32_t, i32_t } });

    // Identical contents dedup despite coming from different slices.
    try std.testing.expectEqual(s_a, s_b);
    // Different field order is a different type.
    try std.testing.expect(s_a != s_reordered);
    // The interned fields survive: the table owns its own copy.
    try std.testing.expectEqualSlices(Type, &fields_a, table.type_kind(s_a).@"struct");
}

test "arrays intern by length and element, distinct from vectors" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const a4_a = try table.intern(.{ .array = .{ .len = 4, .elem = i32_t } });
    const a4_b = try table.intern(.{ .array = .{ .len = 4, .elem = i32_t } });
    const a8 = try table.intern(.{ .array = .{ .len = 8, .elem = i32_t } });
    const v4 = try table.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });

    try std.testing.expectEqual(a4_a, a4_b);
    try std.testing.expect(a4_a != a8);
    try std.testing.expect(a4_a != v4);
}

test "slice is distinct from array and pointer" {
    var table = TypeTable.init(std.testing.allocator);
    defer table.deinit();

    const i32_t = try table.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const sl_a = try table.intern(.{ .slice = .{ .elem = i32_t } });
    const sl_b = try table.intern(.{ .slice = .{ .elem = i32_t } });
    const arr = try table.intern(.{ .array = .{ .len = 4, .elem = i32_t } });
    const ptr = try table.intern(.ptr);

    try std.testing.expectEqual(sl_a, sl_b);
    try std.testing.expect(sl_a != arr);
    try std.testing.expect(sl_a != ptr);
}
