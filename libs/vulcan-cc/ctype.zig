//! C types. `IntType` is a C integer type: a rank plus signedness, with a width that
//! depends on the target (see `bits`). It also holds the promotion rules and the usual
//! arithmetic conversion rules. `CType` extends this to the full set this frontend
//! understands: an integer, a pointer to some `CType`, or a fixed-length,
//! single-dimension array of some `CType`.

const std = @import("std");
const ir = @import("vulcan-ir");
const layout = @import("layout.zig");

/// C integer rank, from narrowest to widest. `long` sits between `int` and `long long`
/// in rank, even though on LP64 targets it shares `long long`'s 64-bit width. `bool`
/// (C99 `_Bool`) sits below `char`. C11 6.3.1.1p1 requires that `_Bool`'s rank be less
/// than every other standard integer type's rank. This is what makes it promote to
/// `int` like every other sub-`int` rank, with no special case (see `IntType.promote`).
pub const Rank = enum(u3) { bool, char, short, int, long, longlong };

/// A C integer type: its rank plus signedness. The width depends on the target
/// (see `bits`).
pub const IntType = struct {
    rank: Rank,
    signed: bool,

    /// This type's width in bits under target layout `l`.
    pub fn bits(self: IntType, l: layout.TargetLayout) u16 {
        return switch (self.rank) {
            .bool => 8,
            .char => 8,
            .short => 16,
            .int => 32,
            .long => l.long_bits,
            .longlong => 64,
        };
    }
    pub fn isUnsigned(self: IntType) bool {
        return !self.signed;
    }

    /// Integer promotion: types of rank < int promote to (signed) int; else unchanged.
    pub fn promote(self: IntType) IntType {
        return if (@intFromEnum(self.rank) < @intFromEnum(Rank.int)) .{ .rank = .int, .signed = true } else self;
    }

    /// The usual arithmetic conversions over two already-promoted operand types (C89 6.2.1.5).
    pub fn commonType(a0: IntType, b0: IntType, l: layout.TargetLayout) IntType {
        const a = a0.promote();
        const b = b0.promote();
        if (a.rank == b.rank and a.signed == b.signed) return a;
        // Same signedness: the higher rank wins.
        if (a.signed == b.signed) {
            return if (@intFromEnum(a.rank) >= @intFromEnum(b.rank)) a else b;
        }
        // Mixed signedness. Find the unsigned operand and the signed operand.
        const u = if (!a.signed) a else b;
        const s = if (a.signed) a else b;
        // If the unsigned operand's rank is at least the signed operand's rank, use the unsigned type.
        if (@intFromEnum(u.rank) >= @intFromEnum(s.rank)) return u;
        // The signed operand has strictly higher rank. If it is strictly wider, it can
        // represent every value of the unsigned type, so keep it signed. Otherwise use
        // the unsigned version of the signed operand's higher rank.
        if (s.bits(l) > u.bits(l)) return s;
        return .{ .rank = s.rank, .signed = false };
    }

    /// The IR integer type (signedness + width) this `IntType` lowers to under layout `l`.
    pub fn irInt(self: IntType, l: layout.TargetLayout) ir.types.Int {
        return .{ .signedness = if (self.signed) .signed else .unsigned, .bits = self.bits(l) };
    }
};

/// A C floating-point type's width: `float` (`f32`) or `double`/`long double` (`f64`).
/// This frontend has no extended-precision `long double`, so it collapses to plain `double`.
pub const FloatKind = enum { f32, f64 };

/// A `const`/`volatile` qualifier pair. It is carried on `CType.ptr` (the pointee's
/// qualifiers), `CType.array` (the element's), `Field` (the field's own), and threaded
/// separately onto `Param`, `GlobalDecl`, `Stmt.decl`, `lower.Binding`, and
/// `lower.GlobalInfo` (the declared object's own qualifiers). See
/// `parser.parseDeclarator`. This struct is purely representational: every consumer
/// (`eql`, layout, codegen) ignores it. Const-correctness checking and the volatile IR
/// flag are what give it effect.
pub const Quals = struct { is_const: bool = false, is_volatile: bool = false };

/// One field of a `StructDef`: its name, C type, byte offset from the start of the
/// struct or union (C layout, computed by `parser.parseStructOrUnion`), and its own
/// qualifiers (for example `struct { const int x; }`).
///
/// `bit_width` is the declared width of a bitfield member (`int a:5;` gives
/// `bit_width = 5`). It is `null` for every ordinary, non-bitfield field, which keeps
/// every existing struct unchanged. When it is set, `ty` is the field's declaring
/// integer type (the storage unit `int`, `unsigned`, `short`, and so on), `offset` is
/// that storage unit's byte offset, and `bit_offset` is the field's first bit within the
/// unit (packed from the least significant bit). A bitfield has no address of its own
/// (you cannot take `&s.f`), so its read and write both go through the bitfield path in
/// `lower.lowerAddr` and `lower.lowerExpr`, not a plain load or store. See those sites.
/// `bit_offset` is meaningless, and reads `0`, for a non-bitfield field.
pub const Field = struct { name: []const u8, ty: CType, offset: u64, quals: Quals = .{}, bit_width: ?u32 = null, bit_offset: u32 = 0 };

/// A struct or union's definition: its tag name, whether it is a union (all fields at
/// offset 0, size equal to the widest field) or a struct (fields laid out in order with
/// C alignment and padding), its fields with their computed offsets, and the whole
/// aggregate's overall `size` and `alignment`. It is allocated once per tag in the
/// parser's arena, at the first sight of that tag: a forward declaration, a bare
/// reference, or a body. See `parser.parseStructOrUnion`. If a body later completes it,
/// the def is mutated in place at the same address. `CType.@"struct"` points at it, and
/// `CType.eql` compares structs by pointer identity to this def, so the same tag means
/// the same def means the same type, not a structural comparison.
///
/// `complete` is false for a tag that has been declared or referenced (`struct Tag;`,
/// `struct Tag *p;`) but has no body yet. `fields`, `size`, and `alignment` then read
/// `&.{}`, `0`, and `0`, which are meaningless until completion. A pointer to an
/// incomplete tag is always valid, because a pointer's own size never depends on what it
/// points to (see `CType.sizeInBytes`). A by-value use, such as a local, a parameter, a
/// field of this type, a `sizeof` of it, or an array of it, is invalid while `complete`
/// is false. `sizeInBytes`, `alignOf`, `storageSize`, and `irType` all report
/// `error.IncompleteType` for it. It defaults to `true` so every existing struct or
/// union literal, such as the `va_list` synthetic defs below, stays exactly as it was: a
/// normal `struct Tag { ... }` with no forward declaration is complete the moment its
/// def is built, same as before.
pub const StructDef = struct { name: []const u8, is_union: bool, fields: []const Field, size: u64, alignment: u64, complete: bool = true };

/// Errors this file's own type-shape queries can report: a by-value use of a struct or
/// union tag that has been declared or referenced but never completed with a body (see
/// `StructDef.complete`'s doc comment for the full rule), or an attempt to lower a bare
/// `void` value to an IR type. `void` itself has a well-defined size and alignment (GNU
/// gives `sizeof(void) == 1`, see `CType.sizeInBytes`), so it is a legal type name, in
/// `sizeof(void)` and as `void*`'s pointee. Only lowering one as a value, such as a
/// `void` local, field, parameter, or return, is meaningless and reports `VoidValue`.
/// `void*` itself never reaches this: `CType.irType`'s `.ptr` arm interns a bare IR `ptr`
/// regardless of pointee.
pub const Error = error{ IncompleteType, VoidValue };

/// A function's type: its return type, where `null` means `void`, matching the existing
/// void-return convention (see `parser.zig`'s file-scope `void name(...)` special case),
/// and its parameter types. It holds bare types only, no names, because a function type
/// has no notion of parameter names. A definition or prototype needs names for its body
/// or diagnostics; see `parser.Param` and `parser.FuncDecl`, which carry names alongside
/// a `FuncType`-shaped ret/params pair built independently. `ret` and `params` are
/// arena-allocated by whoever builds a `FuncType` (the parser's declarator code), the
/// same convention as `CType.ptr`'s `pointee` and `CType.array`'s `elem`.
pub const FuncType = struct {
    ret: ?*const CType,
    params: []const CType,
    /// True for a variadic function (`int printf(const char *fmt, ...)`). `params` then
    /// holds only the fixed, named parameters, and a call may pass more arguments than
    /// `params.len`. See `lower.zig`'s variadic call-lowering path, which checks this
    /// flag instead of requiring exact arity. Defaults to false, so every existing
    /// `FuncType` literal built before this field existed stays a plain, fixed-arity
    /// function type.
    is_variadic: bool = false,

    /// Structural equality, the same rule as `CType.eql`: both `ret`s null, meaning both
    /// void, or both non-null and themselves `eql`; the same param count with each pair
    /// `eql`; and the same `is_variadic`. A variadic and a fixed-arity function with
    /// otherwise identical signatures are still different types, because a call through
    /// one is checked and lowered differently than through the other.
    pub fn eql(a: FuncType, b: FuncType) bool {
        if ((a.ret == null) != (b.ret == null)) return false;
        if (a.ret) |ar| {
            if (!ar.eql(b.ret.?.*)) return false;
        }
        if (a.params.len != b.params.len) return false;
        for (a.params, b.params) |pa, pb| {
            if (!pa.eql(pb)) return false;
        }
        return a.is_variadic == b.is_variadic;
    }
};

/// A full C type: an integer, a pointer to some `CType`, a fixed-length array of some
/// `CType`, or a struct or union, or a function type.
///
/// `elem` may itself be an `.array`, giving multi-dimensional arrays: `int a[2][3]` is
/// `array{len=2, elem=array{len=3, elem=int}}`.
///
/// A struct or union carries its `StructDef`, with fields and layout computed once at
/// definition time by the parser. Pointee and element types are allocated by whoever
/// builds them, the parser's arena, since a union cannot directly contain itself.
///
/// A function type carries its `FuncType`. A function pointer is a `ptr` whose
/// `pointee` is a `CType` of tag `.func`. This frontend has no bare "function object"
/// storage, and neither does C: a function only ever appears as a call target or decays
/// to a pointer. So `.func` shows up in exactly two places: as `ptr.pointee`, for a
/// function-pointer-typed variable using the grouped `(*name)(params)` declarator, or
/// standing alone as the type a bodyless prototype's declarator resolves to before
/// `parser.parse` unpacks it into a `FuncDecl`. It is never actually stored as an
/// object's own `CType`. See `parser.FuncDecl`, which carries `ret` and `params` flat,
/// not a `.func` `CType`.
pub const CType = union(enum) {
    int: IntType,
    /// A pointer to `pointee`. `quals` is the pointee's own qualifiers: `const int *p`
    /// is `ptr{ .pointee = int, .quals = .{ .is_const = true } }`. It is not whether `p`
    /// itself is const, which is the declared object's own quals, tracked separately on
    /// `Param`, `GlobalDecl`, `Stmt.decl`, and `lower.Binding`. See
    /// `parser.parseDeclarator`'s doc comment for how `int *const p` differs. For a
    /// chain of pointers this nests naturally: the qualifier right after a given `*`
    /// becomes the next-outer `ptr`'s `.quals`. It describes what that pointer points
    /// to, this pointer, as qualified, so `int * const * p` recurses correctly with no
    /// special case.
    ptr: struct { pointee: *const CType, quals: Quals = .{} },
    /// An array of `len` `elem`s. `quals` is the element's own qualifiers, mirroring
    /// `ptr`: `const int a[3]` is `array{ .elem = int, .len = 3, .quals = .{ .is_const = true } }`.
    array: struct { elem: *const CType, len: u64, quals: Quals = .{} },
    @"struct": *const StructDef,
    /// A `float` or `double`. No pointee or element allocation is needed. Unlike `ptr`,
    /// `array`, and `@"struct"`, a `FloatKind` is a plain value, not arena-owned.
    float: FloatKind,
    /// A function type. See this union's doc comment for where `.func` shows up: always
    /// as `ptr.pointee`, or transiently as a bodyless-prototype declarator's resolved
    /// type before `parser.parse` unpacks it into a `FuncDecl`.
    func: *const FuncType,
    /// `void`, a type name with no values. It is legal only as `sizeof(void)`, a GNU
    /// extension where `sizeInBytes` and `alignOf` both report `1`, matching gcc, or as
    /// a pointer's pointee (`void*`, `void**`, and so on). A `.ptr` never reads through
    /// to its pointee's own `irType`, so `void*` lowers exactly like any other pointer.
    /// Lowering a bare `void_` itself as a value, such as a `void` local, field,
    /// parameter, or return, is a use error. `irType` reports `error.VoidValue` rather
    /// than fabricating some IR type for it.
    void_: void,

    pub fn isInt(self: CType) bool {
        return self == .int;
    }
    pub fn isFloat(self: CType) bool {
        return self == .float;
    }
    /// True for the `_Bool` type specifically, `Rank.bool`, not just any integer.
    /// `lower.convertTo` reads this to pick the compare-nonzero conversion path instead
    /// of the ordinary int-int truncate/extend one.
    pub fn isBool(self: CType) bool {
        return switch (self) {
            .int => |i| i.rank == .bool,
            else => false,
        };
    }
    pub fn asInt(self: CType) ?IntType {
        return switch (self) {
            .int => |i| i,
            else => null,
        };
    }
    pub fn asFloat(self: CType) ?FloatKind {
        return switch (self) {
            .float => |fk| fk,
            else => null,
        };
    }
    pub fn pointee(self: CType) ?*const CType {
        return switch (self) {
            .ptr => |p| p.pointee,
            else => null,
        };
    }

    /// This type's size in bytes under target layout `l`: an int's `bits/8`, a pointer's
    /// `ptr_bits/8`, an array's `len * elem`'s own size, computed recursively, or a
    /// struct or union's exact size, including padding, from its `StructDef`.
    ///
    /// A pointer's own size never depends on its pointee's completeness. A pointer to an
    /// incomplete struct is always a valid, fully-sized object. An array of an
    /// incomplete struct is itself `error.IncompleteType`, since its own size cannot be
    /// computed either.
    ///
    /// The struct or union size here is the exact, padding-included size, not the
    /// rounded-up storage-blob size `irType` reserves for it. Those can differ: `irType`
    /// always rounds up to a whole number of i64 words, `sizeInBytes` never does.
    ///
    /// Reports `error.IncompleteType` for a struct or union tag with no body yet
    /// (`StructDef.complete == false`), because its size is genuinely unknown, not `0`.
    pub fn sizeInBytes(self: CType, l: layout.TargetLayout) Error!u64 {
        return switch (self) {
            .int => |i| i.bits(l) / 8,
            .ptr => l.ptr_bits / 8,
            .array => |a| a.len * (try a.elem.sizeInBytes(l)),
            .@"struct" => |s| blk: {
                if (!s.complete) break :blk error.IncompleteType;
                break :blk s.size;
            },
            .float => |fk| switch (fk) {
                .f32 => 4,
                .f64 => 8,
            },
            // A bare function type has no size of its own. C forbids `sizeof` on one; it
            // is never an object. This arm is only ever reached, if at all, through a
            // switch-exhaustiveness requirement, since every real use goes through
            // `ptr`'s own `sizeInBytes`, which is pointer-sized regardless of pointee.
            // `0` is the least-surprising sentinel, matching this method's doc comment's
            // "no size" framing for the analogous case.
            .func => 0,
            // `sizeof(void)` is a GNU extension, not standard C, but gcc and clang both
            // give `1`. Treating `void` as a 1-byte, 1-aligned type is what makes
            // `void*` pointer arithmetic, where `p + 1` steps one byte like `char*`,
            // fall out of the existing pointer-arithmetic machinery with no dedicated
            // `void*` case.
            .void_ => 1,
        };
    }

    /// This type's storage stride in bytes: the space one instance actually occupies in
    /// an alloca or as an array element. For a struct or union this is the rounded-up
    /// i64-blob size `irType` reserves (`ceil(size/8)*8`), not the exact C `size` that
    /// `sizeInBytes` reports. They agree for every non-aggregate type. They differ for a
    /// struct whose exact size is not a multiple of 8. Use this for address striding,
    /// such as array indexing and pointer arithmetic. Use `sizeInBytes` for `sizeof`.
    /// Reports `error.IncompleteType` for an incomplete struct or union tag, because a
    /// by-value object of it needs storage. A pointer to it never calls this. See
    /// `sizeInBytes`'s doc comment for the same pointer versus by-value split.
    pub fn storageSize(self: CType, l: layout.TargetLayout) Error!u64 {
        return switch (self) {
            .int, .ptr, .float, .void_ => try self.sizeInBytes(l),
            .array => |a| a.len * (try a.elem.storageSize(l)),
            .@"struct" => |s| blk: {
                if (!s.complete) break :blk error.IncompleteType;
                break :blk ((s.size + 7) / 8) * 8;
            },
            .func => 0, // Mirrors `sizeInBytes`: no object, no storage stride.
        };
    }

    /// This type's required alignment in bytes under target layout `l`: an int's or
    /// pointer's own size, because C aligns scalars to their width, an array's element
    /// alignment, since arrays add no alignment beyond their element's, or a struct or
    /// union's own computed `alignment`, the widest field alignment. `parser.parseStructOrUnion`
    /// uses this to lay out a struct's own fields. Reports `error.IncompleteType` for an
    /// incomplete struct or union tag, the same rule as `sizeInBytes`, because a
    /// by-value field of it cannot be laid out without knowing its alignment.
    pub fn alignOf(self: CType, l: layout.TargetLayout) Error!u64 {
        return switch (self) {
            .int => |i| i.bits(l) / 8,
            .ptr => l.ptr_bits / 8,
            .array => |a| try a.elem.alignOf(l),
            .@"struct" => |s| blk: {
                if (!s.complete) break :blk error.IncompleteType;
                break :blk s.alignment;
            },
            .float => try self.sizeInBytes(l), // C aligns a float or double to its own width.
            .func => 1, // No object, so no alignment requirement of its own.
            .void_ => 1, // GNU gives sizeof(void) == 1, so alignOf follows suit.
        };
    }

    /// The IR type this `CType` lowers to, interned into `func`'s type table under
    /// target layout `l`: an int's `Int`, a bare `ptr` for any pointer, because Vulcan
    /// IR does not carry pointee types, an IR `array` of the element's own `irType`,
    /// computed recursively, or, for a struct or union, the 8-aligned storage blob
    /// `array{ .len = ceil(size/8), .elem = i64 }`. The backend has no native aggregate
    /// type, so a struct variable is stored as a blob of whole `i64` words wide enough
    /// to hold it. `sizeInBytes` above still reports the exact, unrounded C size for
    /// `sizeof`.
    pub fn irType(self: CType, func: *ir.function.Function, l: layout.TargetLayout) (Error || std.mem.Allocator.Error)!ir.types.Type {
        return switch (self) {
            .int => |i| func.types.intern(.{ .int = i.irInt(l) }),
            .ptr => func.types.intern(.ptr),
            .array => |a| func.types.intern(.{ .array = .{ .len = a.len, .elem = try a.elem.irType(func, l) } }),
            // A by-value object of an incomplete struct or union has no storage width to
            // reserve. This reports `error.IncompleteType`, checked before reading
            // `s.size`, which is meaningless and reads `0` while incomplete, rather than
            // silently allocating a 0-word blob. A pointer to an incomplete tag never
            // reaches this arm; the `.ptr` arm above interns a bare IR `ptr` regardless
            // of pointee.
            .@"struct" => |s| blk: {
                if (!s.complete) break :blk error.IncompleteType;
                break :blk func.types.intern(.{ .array = .{ .len = (s.size + 7) / 8, .elem = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } }) } });
            },
            .float => |fk| func.types.intern(.{ .float = switch (fk) {
                .f32 => .f32,
                .f64 => .f64,
            } }),
            // A pointer to a function, the only way `.func` is meaningfully lowered, for
            // a function-pointer variable, already goes through the `.ptr` arm above,
            // which interns a bare IR `ptr` regardless of pointee. It never recurses
            // into `.func` here. This arm exists for switch-exhaustiveness and
            // defensive callers. If a bare `.func` CType is ever asked for its own IR
            // type directly, which no lowering path does yet, it lowers the same way a
            // function decays to a pointer everywhere else in C. An array parameter's
            // own decay is the closest existing precedent; see `parser.parseParams`.
            .func => func.types.intern(.ptr),
            // A bare `void` has no IR representation as a value. Only `void*`, a `.ptr`
            // handled above with no pointee lookup at all, ever actually lowers.
            // Reaching this arm means something tried to give a `void`-typed local,
            // field, parameter, or return an actual storage slot, which is meaningless.
            // This fails closed rather than fabricating a 0-width or 1-byte IR type for it.
            .void_ => error.VoidValue,
        };
    }

    /// Structural equality: the same case, and, for ptr and array, the pointee or
    /// element types are themselves equal, checked recursively. This is not pointer
    /// identity, since pointee and elem `CType`s are separately arena-allocated per
    /// declarator. A struct or union instead compares by pointer identity of its
    /// `StructDef`: two struct types are the same type only if they name the same tag
    /// definition. C has no structural struct typing, so two separately-defined structs
    /// with identical fields are still distinct types. `quals` is ignored here by
    /// design: `const int*` and `int*` are the same `CType` for every purpose `eql`
    /// serves, such as conversion checks and param or arg matching. Only the dedicated
    /// const-check reads `quals` directly, never through `eql`.
    pub fn eql(a: CType, b: CType) bool {
        return switch (a) {
            .int => |ai| b == .int and std.meta.eql(ai, b.int),
            .ptr => |ap| b == .ptr and ap.pointee.eql(b.ptr.pointee.*),
            .array => |aa| b == .array and aa.len == b.array.len and aa.elem.eql(b.array.elem.*),
            .@"struct" => |as| b == .@"struct" and as == b.@"struct",
            .float => |af| b == .float and af == b.float,
            .func => |af| b == .func and af.eql(b.func.*),
            .void_ => b == .void_,
        };
    }

    /// Integer promotion, lifted to `CType`. Meaningful for `.int`, because C
    /// promotions apply only to arithmetic operands. Pointer arithmetic is handled in
    /// lowering, not here. A float operand promotes to itself. Past K&R C, "float
    /// promotion" under the usual arithmetic conversions is a no-op. There is no
    /// implicit `float` to `double` conversion here.
    pub fn promote(self: CType) CType {
        if (self.isFloat()) return self;
        std.debug.assert(self.isInt());
        return .{ .int = self.int.promote() };
    }

    /// The usual arithmetic conversions, lifted to `CType`. Meaningful for two `.int`
    /// operands (see `promote`), or when either operand is `.float`. The result is then
    /// the wider float, `f64` if either operand is `f64`, else `f32`. An int operand
    /// converts straight to that float, never through an intermediate int-int conversion.
    pub fn commonType(a: CType, b: CType, l: layout.TargetLayout) CType {
        if (a.isFloat() or b.isFloat()) {
            const a_f64 = (a.asFloat() orelse .f32) == .f64;
            const b_f64 = (b.asFloat() orelse .f32) == .f64;
            return .{ .float = if (a_f64 or b_f64) .f64 else .f32 };
        }
        std.debug.assert(a.isInt() and b.isInt());
        return .{ .int = IntType.commonType(a.int, b.int, l) };
    }
};

/// Construct an integer `CType` for the given rank and signedness. This is a small
/// convenience so callers do not need to spell out the `.{ .int = .{ ... } }` wrapping.
pub fn mkInt(rank: Rank, signed: bool) CType {
    return .{ .int = .{ .rank = rank, .signed = signed } };
}

/// Synthetic `StructDef`s backing `__builtin_va_list` (see `builtinVaList`). These are
/// opaque as far as this frontend is concerned: no field this frontend ever reads
/// through, only the size and alignment the target ABI needs. The backend owns what the
/// bytes inside actually mean.
const va_list_x86_64_def: StructDef = .{ .name = "__builtin_va_list", .is_union = false, .fields = &.{}, .size = 24, .alignment = 8 };
const va_list_x86_64_struct: CType = .{ .@"struct" = &va_list_x86_64_def };
const va_list_aarch64_def: StructDef = .{ .name = "__builtin_va_list", .is_union = false, .fields = &.{}, .size = 32, .alignment = 8 };
const va_list_aarch64_struct: CType = .{ .@"struct" = &va_list_aarch64_def };

/// The `__builtin_va_list` type: `<stdarg.h>`'s `va_list` aliases this (see
/// `stdarg.zig`). Its representation is pure per-target ABI, not a rule this frontend
/// enforces itself. It reuses the existing struct, array, and pointer machinery
/// (`sizeInBytes`, `alignOf`, array-parameter decay) instead of a dedicated `CType`
/// tag, so every existing exhaustive switch over `CType` needs zero new arms.
pub fn builtinVaList(l: layout.TargetLayout) CType {
    return switch (l.arch) {
        // SysV x86-64: a 24-byte, 8-aligned struct, passed and decayed as `array[1]`,
        // matching glibc, so a bare `va_list` parameter or argument decays to a
        // pointer to it, the same way any other array does in this frontend.
        .x86_64 => .{ .array = .{ .elem = &va_list_x86_64_struct, .len = 1 } },
        // AAPCS64: a 32-byte, 8-aligned struct, passed and decayed as `array[1]`,
        // mirroring x86_64. AAPCS64 passes a composite larger than 16 bytes
        // indirectly, as a pointer to it, so a bare `va_list` argument must reach the
        // callee as a pointer to the 32-byte struct. This frontend does not pass
        // structs by value, but it does decay an array to a pointer to its first
        // element, which is exactly that pointer. `va_start`, `va_arg`, and `va_end`
        // still operate on the object's address, since element 0 sits at offset 0, so
        // the define side is unchanged. `va_list ap;` still allocates the same 32
        // bytes at the same address.
        .aarch64 => .{ .array = .{ .elem = &va_list_aarch64_struct, .len = 1 } },
        // riscv64 and x86 (i386): a plain pointer, the simplest ABI shape, stepped by
        // the backend's own variadic lowering. The pointee type is never read
        // through in this frontend. `char_t` is only a stand-in so `sizeInBytes` and
        // `alignOf` resolve to the target's pointer width.
        .riscv64, .x86 => .{ .ptr = .{ .pointee = &char_t } },
    };
}

/// `_Bool` (C99): 1 byte, treated as unsigned. Its only representable values are 0 and
/// 1, so signedness never actually matters for its own storage, but `false` keeps it
/// consistent with "an unsigned 1-byte int" for `IntType.bits` and `irInt` purposes.
pub const bool_t: CType = .{ .int = .{ .rank = .bool, .signed = false } };
/// `void`. See `CType.void_`'s doc comment. This is a shared constant, the same
/// convention as `int_t` and `char_t` below, since `void` carries no data of its own to
/// distinguish instances.
pub const void_t: CType = .{ .void_ = {} };
pub const char_t: CType = .{ .int = .{ .rank = .char, .signed = true } };
pub const int_t: CType = .{ .int = .{ .rank = .int, .signed = true } };
pub const uint_t: CType = .{ .int = .{ .rank = .int, .signed = false } };
pub const long_t: CType = .{ .int = .{ .rank = .long, .signed = true } };
pub const ulong_t: CType = .{ .int = .{ .rank = .long, .signed = false } };

test "promotion lifts char/short to int" {
    try std.testing.expectEqual(int_t, char_t.promote());
    try std.testing.expectEqual(int_t, (CType{ .int = .{ .rank = .short, .signed = false } }).promote());
    try std.testing.expectEqual(long_t, long_t.promote());
}
test "usual arithmetic conversions" {
    const l = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    try std.testing.expectEqual(int_t, CType.commonType(int_t, int_t, l));
    try std.testing.expectEqual(uint_t, CType.commonType(int_t, uint_t, l)); // same width, unsigned wins
    try std.testing.expectEqual(long_t, CType.commonType(int_t, long_t, l)); // wider wins
    try std.testing.expectEqual(int_t, CType.commonType(char_t, char_t, l)); // both promote to int
}
test "commonType keeps signedness for same-sign different-rank" {
    const lp64 = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    const long_long_signed = CType{ .int = .{ .rank = .longlong, .signed = true } };
    // long + long long (both signed, both 64-bit on LP64) -> signed long long, NOT unsigned.
    try std.testing.expectEqual(long_long_signed, CType.commonType(long_t, long_long_signed, lp64));
    // int + long on ILP32 (both signed, both 32-bit) -> signed long, NOT unsigned.
    const ilp32 = layout.TargetLayout{ .long_bits = 32, .ptr_bits = 32, .char_signed = true };
    try std.testing.expectEqual(long_t, CType.commonType(int_t, long_t, ilp32));
}
test "commonType mixed sign: unsigned wins at >= rank, signed wins when strictly wider" {
    const lp64 = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    // int + unsigned int (same rank, mixed sign) -> unsigned int.
    try std.testing.expectEqual(uint_t, CType.commonType(int_t, uint_t, lp64));
    // signed long (64) + unsigned int (32): signed strictly wider -> signed long.
    try std.testing.expectEqual(long_t, CType.commonType(long_t, uint_t, lp64));
    // unsigned long (64) + signed int (32): unsigned higher rank -> unsigned long.
    try std.testing.expectEqual(ulong_t, CType.commonType(ulong_t, int_t, lp64));
}

test "sizeInBytes: int/ptr/array" {
    const l = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    try std.testing.expectEqual(@as(u64, 4), try int_t.sizeInBytes(l));
    try std.testing.expectEqual(@as(u64, 8), try long_t.sizeInBytes(l));
    const ptr_to_int: CType = .{ .ptr = .{ .pointee = &int_t } };
    try std.testing.expectEqual(@as(u64, 8), try ptr_to_int.sizeInBytes(l));
    const arr: CType = .{ .array = .{ .elem = &int_t, .len = 3 } };
    try std.testing.expectEqual(@as(u64, 12), try arr.sizeInBytes(l));
}

test "builtinVaList sizes per target" {
    const x86_64_layout = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = true, .arch = .x86_64 };
    try std.testing.expectEqual(@as(u64, 24), try builtinVaList(x86_64_layout).sizeInBytes(x86_64_layout));
    const aarch64_layout = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .aarch64 };
    try std.testing.expectEqual(@as(u64, 32), try builtinVaList(aarch64_layout).sizeInBytes(aarch64_layout));
    const riscv64_layout = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .riscv64 };
    try std.testing.expectEqual(@as(u64, riscv64_layout.ptr_bits / 8), try builtinVaList(riscv64_layout).sizeInBytes(riscv64_layout));
    const i386_layout = layout.TargetLayout{ .long_bits = 32, .ptr_bits = 32, .char_signed = true, .arch = .x86 };
    try std.testing.expectEqual(@as(u64, i386_layout.ptr_bits / 8), try builtinVaList(i386_layout).sizeInBytes(i386_layout));
}

// A struct or union tag with no body yet (`complete = false`) has no known size,
// alignment, or storage width. Every by-value query on it fails closed with
// `error.IncompleteType` rather than reporting a misleading `0`. A pointer to it,
// though, stays a plain, fully-sized pointer. `sizeInBytes`'s `.ptr` arm never
// consults the pointee.
test "incomplete struct fails closed on every by-value size/align query" {
    const l = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    const incomplete_def: StructDef = .{ .name = "Opaque", .is_union = false, .fields = &.{}, .size = 0, .alignment = 0, .complete = false };
    const incomplete: CType = .{ .@"struct" = &incomplete_def };
    try std.testing.expectError(error.IncompleteType, incomplete.sizeInBytes(l));
    try std.testing.expectError(error.IncompleteType, incomplete.alignOf(l));
    try std.testing.expectError(error.IncompleteType, incomplete.storageSize(l));
    // A pointer to the same incomplete def is unaffected. Its own size is just `ptr_bits/8`.
    const ptr_to_incomplete: CType = .{ .ptr = .{ .pointee = &incomplete } };
    try std.testing.expectEqual(@as(u64, 8), try ptr_to_incomplete.sizeInBytes(l));
    // An array of an incomplete struct is equally unresolvable.
    const arr_of_incomplete: CType = .{ .array = .{ .elem = &incomplete, .len = 3 } };
    try std.testing.expectError(error.IncompleteType, arr_of_incomplete.sizeInBytes(l));
}

test "CType.eql compares structurally, not by pointer identity" {
    const p1: CType = .{ .ptr = .{ .pointee = &int_t } };
    const p2: CType = .{ .ptr = .{ .pointee = &int_t } };
    try std.testing.expect(p1.eql(p2));
    try std.testing.expect(!p1.eql(long_t));
    const a1: CType = .{ .array = .{ .elem = &int_t, .len = 3 } };
    const a2: CType = .{ .array = .{ .elem = &int_t, .len = 3 } };
    const a3: CType = .{ .array = .{ .elem = &int_t, .len = 4 } };
    try std.testing.expect(a1.eql(a2));
    try std.testing.expect(!a1.eql(a3));
}

// `Quals` on `ptr` and `array` is ignored by `eql`. `const int*` and `int*`, likewise
// `const int[3]` and `int[3]`, are the same `CType` for every conversion or
// param-matching purpose. Only the dedicated const-check ever reads `.quals` directly.
test "CType.eql ignores quals" {
    const const_ptr: CType = .{ .ptr = .{ .pointee = &int_t, .quals = .{ .is_const = true } } };
    const plain_ptr: CType = .{ .ptr = .{ .pointee = &int_t } };
    try std.testing.expect(const_ptr.eql(plain_ptr));
    const const_arr: CType = .{ .array = .{ .elem = &int_t, .len = 3, .quals = .{ .is_volatile = true } } };
    const plain_arr: CType = .{ .array = .{ .elem = &int_t, .len = 3 } };
    try std.testing.expect(const_arr.eql(plain_arr));
}

// `void` has a well-defined size and alignment, GNU gives `sizeof(void) == 1`, even
// though it can never lower to an IR value type. `void*`, a `.ptr`, is unaffected,
// since a pointer's own size and IR type never consult its pointee.
test "void: sizeof/alignof are 1, void* is pointer-sized, eql/irType behave" {
    const l = layout.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    try std.testing.expectEqual(@as(u64, 1), try void_t.sizeInBytes(l));
    try std.testing.expectEqual(@as(u64, 1), try void_t.alignOf(l));
    try std.testing.expectEqual(@as(u64, 1), try void_t.storageSize(l));
    try std.testing.expect(void_t.eql(void_t));
    try std.testing.expect(!void_t.eql(int_t));
    const void_ptr: CType = .{ .ptr = .{ .pointee = &void_t } };
    try std.testing.expectEqual(@as(u64, 8), try void_ptr.sizeInBytes(l));
    const int_ptr: CType = .{ .ptr = .{ .pointee = &int_t } };
    // `void*` and `int*` are different `CType`s, since their pointees differ. `eql`
    // still distinguishes pointee types. It is `lower.convertTo`'s pointer-pointer arm,
    // which is pointee-agnostic for every `T* <-> U*` pair, not just `void*`, that
    // allows the implicit conversion between them, not `eql`.
    try std.testing.expect(!void_ptr.eql(int_ptr));
}
