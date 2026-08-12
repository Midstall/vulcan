//! The C parser converts tokens to an AST (Abstract Syntax Tree).
//! It supports multiple functions with typed parameters and return types. Any integer
//! type-spec that `parseTypeSpec` resolves is valid, for example `int`, `long`, or
//! `unsigned short`. A function body is a statement list. It can hold typed declarations
//! with initializers, assignment and compound-assignment expression statements, and a
//! terminating `return`. Expressions cover integer constants, names, the unary operators
//! -/~/!, precedence-climbed binary arithmetic, bitwise, and shift operators, comparisons,
//! assignment and compound-assignment, and calls. The returned `Unit` owns all AST nodes
//! in its arena.

const std = @import("std");
const lexer = @import("lexer.zig");
const layout_mod = @import("layout.zig");
const ctype = @import("ctype.zig");
const preproc = @import("preproc.zig");
const consteval = @import("consteval.zig");

test "parse a function with an int parameter" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator, "int id(int a){ return a; }", layout_mod.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 1), unit.funcs.len);
    try std.testing.expectEqualStrings("id", unit.funcs[0].name);
    try std.testing.expectEqual(@as(usize, 1), unit.funcs[0].params.len);
    try std.testing.expectEqualStrings("a", unit.funcs[0].params[0].name);
    try std.testing.expectEqualStrings("a", unit.funcs[0].body[0].ret.?.name);
}
test "parse multiple functions" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator, "int a(void){ return 1; } int b(void){ return 2; }", layout_mod.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 2), unit.funcs.len);
    try std.testing.expectEqualStrings("a", unit.funcs[0].name);
    try std.testing.expectEqualStrings("b", unit.funcs[1].name);
}
test "parse still handles void + constant" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator, "int f(void){ return -7; }", layout_mod.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(i64, 7), unit.funcs[0].body[0].ret.?.negate.int_lit.value);
}
// `__builtin_va_arg(ap, type-name)` parses to `Expr.va_arg`. The parser resolves its
// second argument as a TYPE, not an expression, at parse time.
test "parses __builtin_va_arg(ap, int) into Expr.va_arg" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator, "int f(int ap){ return __builtin_va_arg(ap, int); }", layout_mod.host(), .{});
    defer unit.deinit();
    const ret_expr = unit.funcs[0].body[0].ret.?;
    try std.testing.expectEqualStrings("ap", ret_expr.va_arg.ap.name);
    try std.testing.expect(ret_expr.va_arg.ty.eql(ctype.int_t));
}
// `__builtin_va_list` is a recognized type-specifier, not just a `CType`/size fact, so a
// bare local declaration of it parses.
test "parses __builtin_va_list as a recognized type-specifier" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    var unit = try parse(allocator, "int f(void){ __builtin_va_list ap; return 0; }", l, .{});
    defer unit.deinit();
    try std.testing.expect(unit.funcs[0].body[0].decl.ty.eql(ctype.builtinVaList(l)));
}
// `parseTypeSpec` is wired into decl/param/func parsing below, so `long`, `unsigned`, and
// similar keywords parse there too. This test still calls the (file-private) `Parser`
// directly, since that is the narrowest way to exercise the multi-word type-spec mix.
fn testParseTypeSpec(allocator: std.mem.Allocator, source: []const u8, l: layout_mod.TargetLayout) !ctype.CType {
    const toks = try lexer.tokenize(allocator, source);
    defer lexer.freeTokens(allocator, toks);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
    return (try p.parseTypeSpec()).ty;
}

test "parseTypeSpec resolves the multi-word soup, order-independent" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    try std.testing.expectEqual(ctype.ulong_t, try testParseTypeSpec(allocator, "unsigned long int", l));
    try std.testing.expectEqual(ctype.ulong_t, try testParseTypeSpec(allocator, "long unsigned int", l));
    try std.testing.expectEqual(ctype.int_t, try testParseTypeSpec(allocator, "int", l));
    try std.testing.expectEqual(ctype.mkInt(.longlong, true), try testParseTypeSpec(allocator, "long long", l));
    try std.testing.expectEqual(ctype.mkInt(.char, false), try testParseTypeSpec(allocator, "unsigned char", l));
    // Plain `char` follows the target's char signedness.
    const arm_layout = layout_mod.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = false };
    const x86_layout = layout_mod.TargetLayout{ .long_bits = 64, .ptr_bits = 64, .char_signed = true };
    try std.testing.expectEqual(ctype.mkInt(.char, false), try testParseTypeSpec(allocator, "char", arm_layout));
    try std.testing.expectEqual(ctype.mkInt(.char, true), try testParseTypeSpec(allocator, "char", x86_layout));
    // No type keyword at all is rejected.
    try std.testing.expectError(error.UnexpectedToken, testParseTypeSpec(allocator, "x", l));
}

// `parseDeclarator` parses `*` characters (each wraps `ptr`) and array suffixes (each
// wraps `array`) around a name, after a base type-spec.
test "parseDeclarator parses pointer and array declarators" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    {
        const toks = try lexer.tokenize(allocator, "int *p");
        defer lexer.freeTokens(allocator, toks);
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
        const base = try p.parseTypeSpec();
        const d = try p.parseDeclarator(base.ty, base.quals);
        try std.testing.expectEqualStrings("p", d.name);
        try std.testing.expect(d.ty == .ptr);
        try std.testing.expect(d.ty.ptr.pointee.eql(ctype.int_t));
    }
    {
        const toks = try lexer.tokenize(allocator, "int **pp");
        defer lexer.freeTokens(allocator, toks);
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
        const base = try p.parseTypeSpec();
        const d = try p.parseDeclarator(base.ty, base.quals);
        try std.testing.expectEqualStrings("pp", d.name);
        try std.testing.expect(d.ty == .ptr);
        try std.testing.expect(d.ty.ptr.pointee.* == .ptr);
        try std.testing.expect(d.ty.ptr.pointee.ptr.pointee.eql(ctype.int_t));
    }
    {
        const toks = try lexer.tokenize(allocator, "int arr[4]");
        defer lexer.freeTokens(allocator, toks);
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
        const base = try p.parseTypeSpec();
        const d = try p.parseDeclarator(base.ty, base.quals);
        try std.testing.expectEqualStrings("arr", d.name);
        try std.testing.expect(d.ty == .array);
        try std.testing.expectEqual(@as(u64, 4), d.ty.array.len);
        try std.testing.expect(d.ty.array.elem.eql(ctype.int_t));
    }
}

// Qualifier attribution rule for `const`/`volatile`. A leading qualifier before a plain
// declarator, one with no `*`, lands on the declared OBJECT's own `quals`. A leading
// qualifier before a POINTER declarator (`const int *p`) instead lands on the pointer's
// `.quals`, the POINTEE's qualifiers. See `ctype.CType.ptr`'s doc comment. A qualifier
// right after the `*` (`int *const p`) attributes to the pointer OBJECT itself, not the
// pointee. See `parseDeclarator`'s doc comment for the general rule, at any depth, that
// this follows.
test "parser attributes const/volatile qualifiers correctly" {
    const allocator = std.testing.allocator;
    {
        // `const int x`: no `*` in the declarator, so the leading qualifier is the OBJECT's
        // own quals. The type itself stays plain `int`.
        var unit = try parse(allocator, "int f(void){ const int x = 1; return x; }", layout_mod.host(), .{});
        defer unit.deinit();
        const d = unit.funcs[0].body[0].decl;
        try std.testing.expect(d.ty == .int);
        try std.testing.expect(d.quals.is_const);
        try std.testing.expect(!d.quals.is_volatile);
    }
    {
        // `volatile int x`: same shape, the OTHER qualifier.
        var unit = try parse(allocator, "int f(void){ volatile int x = 1; return x; }", layout_mod.host(), .{});
        defer unit.deinit();
        const d = unit.funcs[0].body[0].decl;
        try std.testing.expect(d.quals.is_volatile);
        try std.testing.expect(!d.quals.is_const);
    }
    {
        // `const int *p`: the leading qualifier, before the base type, attaches to the
        // FIRST `*`'s pointee-quals. The pointer object `p` itself is NOT const, so it can
        // be reassigned.
        var unit = try parse(allocator, "int f(int a){ int x = a; const int *p = &x; return *p; }", layout_mod.host(), .{});
        defer unit.deinit();
        const d = unit.funcs[0].body[1].decl;
        try std.testing.expect(d.ty == .ptr);
        try std.testing.expect(d.ty.ptr.quals.is_const); // pointee (int) is const
        try std.testing.expect(!d.quals.is_const); // p itself is not
    }
    {
        // `int *const p`: the qualifier right after `*` attaches to the declared OBJECT. `p`
        // itself is const, so it cannot be reassigned. The pointee (`int`) stays unqualified.
        var unit = try parse(allocator, "int f(int a){ int x = a; int *const p = &x; return *p; }", layout_mod.host(), .{});
        defer unit.deinit();
        const d = unit.funcs[0].body[1].decl;
        try std.testing.expect(d.ty == .ptr);
        try std.testing.expect(!d.ty.ptr.quals.is_const); // pointee (int) is NOT const
        try std.testing.expect(d.quals.is_const); // p itself is
    }
}
// Multiple `[N]` suffixes wrap from the INNERMOST (last) dimension outward. `int a[2][3]`
// becomes `array{len=2, elem=array{len=3, elem=int}}`. This is an array of 2 arrays of 3
// ints, not the other way around. This matches `a[i][j]`'s expected row-major addressing.
test "parseDeclarator wraps multi-dim array dimensions innermost-first" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    const toks = try lexer.tokenize(allocator, "int a[2][3]");
    defer lexer.freeTokens(allocator, toks);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
    const base = try p.parseTypeSpec();
    const d = try p.parseDeclarator(base.ty, base.quals);
    try std.testing.expectEqualStrings("a", d.name);
    try std.testing.expect(d.ty == .array);
    try std.testing.expectEqual(@as(u64, 2), d.ty.array.len); // outer: 2 elements
    try std.testing.expect(d.ty.array.elem.* == .array);
    try std.testing.expectEqual(@as(u64, 3), d.ty.array.elem.array.len); // inner: 3 elements
    try std.testing.expect(d.ty.array.elem.array.elem.eql(ctype.int_t));
}

// An array dimension is a full CONSTANT-EXPRESSION, including `sizeof`, not just a bare
// integer literal.
test "parseArrayDims evaluates a constant-expression length, including sizeof" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    // `void*` is not a modeled type in this frontend at all. This is a pre-existing gap:
    // `ctype.zig` has no `void` case, see `parseAtom`'s `(void)e` special case doc. So
    // `char*` stands in for it here. It has the same size on an LP64 host (8 bytes) and
    // gives the same arithmetic result this test checks.
    const toks = try lexer.tokenize(allocator, "char x[12*sizeof(int) - 5*sizeof(char*)]");
    defer lexer.freeTokens(allocator, toks);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
    const base = try p.parseTypeSpec();
    const d = try p.parseDeclarator(base.ty, base.quals);
    try std.testing.expectEqualStrings("x", d.name);
    try std.testing.expect(d.ty == .array);
    try std.testing.expectEqual(@as(u64, 8), d.ty.array.len); // 12*4 - 5*8 = 8, on an LP64 host
}

test "parseArrayDims still parses a plain integer-literal dimension (byte-identical)" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    const toks = try lexer.tokenize(allocator, "int a[3]");
    defer lexer.freeTokens(allocator, toks);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
    const base = try p.parseTypeSpec();
    const d = try p.parseDeclarator(base.ty, base.quals);
    try std.testing.expectEqual(@as(u64, 3), d.ty.array.len);
}

test "parseArrayDims rejects a non-constant dimension (VLA is out of scope)" {
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    const toks = try lexer.tokenize(allocator, "int a[n]");
    defer lexer.freeTokens(allocator, toks);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
    const base = try p.parseTypeSpec();
    try std.testing.expectError(error.Unsupported, p.parseDeclarator(base.ty, base.quals));
}

test "parseArrayDims accepts a zero-length dimension (GNU flexible member)" {
    // `T a[0]` is the GNU zero-length-array extension. glibc's `struct file_handle` ends
    // with `unsigned char f_handle[0];`. It parses as an array of length 0, the same layout
    // as the empty `[]` form. A NEGATIVE dimension is still rejected.
    const allocator = std.testing.allocator;
    const l = layout_mod.host();
    const toks = try lexer.tokenize(allocator, "int a[0]");
    defer lexer.freeTokens(allocator, toks);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = l };
    const base = try p.parseTypeSpec();
    const d = try p.parseDeclarator(base.ty, base.quals);
    try std.testing.expect(d.ty == .array);
    try std.testing.expectEqual(@as(u64, 0), d.ty.array.len);
}

// This test covers a bodyless prototype (`int puts(const char *s);`), an honored `extern`
// variable declaration (`extern int counter;`, no storage or definition), and a
// function-pointer-typed variable declarator (`int (*fp)(int, int);`, the grouped
// pointer-to-function form). It checks the parse and AST level only, not lowering.
test "parse: bodyless prototype, extern variable, and function-pointer declarator" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator, "int puts(const char *s); extern int counter; int (*fp)(int, int);", layout_mod.host(), .{});
    defer unit.deinit();

    // `puts`: a FuncDecl (no body), one param `const char*`, ret `int`.
    try std.testing.expectEqual(@as(usize, 1), unit.func_decls.len);
    const fd = unit.func_decls[0];
    try std.testing.expectEqualStrings("puts", fd.name);
    try std.testing.expect(fd.ret != null);
    try std.testing.expect(fd.ret.?.eql(ctype.int_t));
    try std.testing.expectEqual(StorageClass.none, fd.storage);
    try std.testing.expectEqual(@as(usize, 1), fd.params.len);
    try std.testing.expectEqualStrings("s", fd.params[0].name);
    try std.testing.expect(fd.params[0].ty == .ptr);
    try std.testing.expect(fd.params[0].ty.ptr.pointee.asInt().?.rank == .char);
    try std.testing.expect(fd.params[0].ty.ptr.quals.is_const);

    // `counter`/`fp` are both plain globals (declarations, not function definitions).
    try std.testing.expectEqual(@as(usize, 2), unit.globals.len);

    const counter = unit.globals[0];
    try std.testing.expectEqualStrings("counter", counter.name);
    try std.testing.expectEqual(StorageClass.extern_, counter.storage);
    try std.testing.expect(counter.init == null);
    try std.testing.expect(counter.ty.eql(ctype.int_t));

    // `fp`: `ptr -> func{ ret = int, params = [int, int] }`.
    const fp = unit.globals[1];
    try std.testing.expectEqualStrings("fp", fp.name);
    try std.testing.expect(fp.ty == .ptr);
    try std.testing.expect(fp.ty.ptr.pointee.* == .func);
    const ft = fp.ty.ptr.pointee.func;
    try std.testing.expect(ft.ret != null);
    try std.testing.expect(ft.ret.?.eql(ctype.int_t));
    try std.testing.expectEqual(@as(usize, 2), ft.params.len);
    try std.testing.expect(ft.params[0].eql(ctype.int_t));
    try std.testing.expect(ft.params[1].eql(ctype.int_t));
}

// `__attribute__((...))` parses, and records nothing, at every position. It always balances
// NESTED parens inside its own body, not just stopping at the first `)`. This test exercises
// the depth-tracking directly. `__format__(__printf__, 1, 2)` opens a THIRD level of parens
// inside the attribute's own `((...))` wrapper. The test also covers every attribute
// POSITION at once: leading the decl, trailing the prototype's `)`, stacked, on a struct
// field, and on a typedef. A gcc-differential run-time check is not needed here, unlike
// `native.zig`'s tests, because `__format__` requires a genuinely variadic printf-shaped
// function to satisfy gcc's OWN semantic check. That concern is unrelated to whether VCC's
// PARSER balances the paren nesting.
test "parses __attribute__ with a nested-paren argument list at every position, ignoring it" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator,
        \\__attribute__((__pure__)) int f(int x) __attribute__((__nothrow__)) __attribute__((__format__(__printf__, 1, 2)));
        \\int f(int x){ return x; }
        \\struct S { int a __attribute__((__deprecated__)); };
        \\typedef int myint __attribute__((__aligned__(4)));
    , layout_mod.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 1), unit.funcs.len);
    try std.testing.expectEqualStrings("f", unit.funcs[0].name);
    try std.testing.expect(unit.funcs[0].ret.eql(ctype.int_t));
}

test "parse still works end to end with a full C source" {
    const allocator = std.testing.allocator;
    var unit = try parse(allocator, "int f(void){ int x = 0; return x; }", layout_mod.host(), .{});
    defer unit.deinit();
    try std.testing.expectEqual(@as(usize, 1), unit.funcs.len);
}

/// A binary arithmetic, bitwise, or shift operator: `add`..`rem` for arithmetic, plus the
/// bitwise and shift operators.
pub const BinOp = enum { add, sub, mul, div, rem, bit_and, bit_or, bit_xor, shl, shr };

/// A relational or equality operator.
pub const CmpOp = enum { eq, ne, lt, le, gt, ge };

/// An expression: an integer literal, its negation, a name reference, a binary operation,
/// a comparison, a bitwise complement, or a logical not.
pub const Expr = union(enum) {
    int_lit: struct { value: i64, ty: ctype.CType },
    /// A `float`/`double` literal: its decoded `f64` value, parsed via
    /// `std.fmt.parseFloat` since every C float literal fits an `f64` (narrowing to `f32`
    /// happens on lowering's `.fconst`, not here), and its `CType` (`.float{.f32}` for an
    /// `f`/`F` suffix, `.float{.f64}` otherwise, so `1.5f` differs from `1.5`/`1.5l`).
    float_lit: struct { value: f64, ty: ctype.CType },
    negate: *Expr,
    complement: *Expr,
    lognot: *Expr,
    name: []const u8,
    /// `&e`: the address of the lvalue `e`. Lowering routes this through `lowerAddr`, not
    /// `lowerExpr`. No load is ever emitted for the operand.
    addrof: *Expr,
    /// `*e`: dereference the pointer value `e`. Also an lvalue in its own right. It is one
    /// of `lowerAddr`'s cases (the other being `.name`), so `*p` on the LEFT of `=` stores
    /// through it instead of loading.
    deref: *Expr,
    /// `base[idx]`, with multi-dim chaining: `base` is an array name, which decays to a
    /// pointer in `lowerExpr`, another `.index` (`a[i][j]` parses as `index{base=index{a,i},
    /// idx=j}`, see `parsePrimary`), or any other pointer-typed expression. `idx` indexes
    /// it, scaled by the pointee's size. Lowering routes this through `lowerAddr`'s `.index`
    /// arm. As an rvalue it loads through the resolved address, or, if the element itself is
    /// an array, decays to a pointer with no load (see `lowerExpr`'s `.index` arm). As an
    /// assignment target it stores through it, same as `.deref`.
    index: struct { base: *Expr, idx: *Expr },
    /// `base.field` (`arrow = false`) or `base->field` (`arrow = true`). For `.`, `base` is
    /// a struct LVALUE (its address is what `lowerAddr` needs, same as `.index`'s array
    /// base). For `->`, `base` is a POINTER rvalue whose VALUE is the struct's address.
    /// Chains with `[]` and with itself in `parsePrimary`'s postfix loop, with the same
    /// left-to-right associativity as `.index`: `arr[i].field`, `p->a.b`, and so on.
    member: struct { base: *Expr, field: []const u8, arrow: bool },
    binary: struct { op: BinOp, lhs: *Expr, rhs: *Expr },
    compare: struct { op: CmpOp, lhs: *Expr, rhs: *Expr },
    logand: struct { lhs: *Expr, rhs: *Expr },
    logor: struct { lhs: *Expr, rhs: *Expr },
    /// `target op= value` (`op` null = plain `=`). `target` is any lvalue expression: a bare
    /// name, `*p`, or `a[i]`, since `target` is a full `*Expr` that lowering resolves via
    /// `lowerAddr`.
    assign: struct { target: *Expr, op: ?BinOp, value: *Expr },
    /// `++e`/`--e` (prefix, `prefix = true`) or `e++`/`e--` (postfix, `prefix = false`).
    /// `inc` selects `++` vs `--`. `target` is any lvalue (`.name`/`.deref`/`.index`/
    /// `.member`), resolved through `lowerAddr` exactly like `.assign`'s target. Lowering
    /// loads the old value through it, stores old +/- 1 (int/float) or old scaled by
    /// `sizeof(pointee)` (pointer, C11 6.5.6) back through it, and yields the NEW value
    /// (prefix) or the OLD one (postfix).
    incdec: struct { target: *Expr, inc: bool, prefix: bool },
    /// A function call. `callee` is a full expression, not just a bare identifier captured
    /// at the call site, so `foo(...)`, `(foo)(...)`, and `tbl[i](...)` all parse the same
    /// way, chained as a postfix operator in `parsePrimary` alongside `[]`/`.`/`->`.
    /// Lowering (`lower.zig`'s `.call` arm) only resolves a `.name` callee today, the
    /// same-TU `findFunc` path. Anything else is `error.Unsupported` until indirect calls
    /// through function pointers are added.
    call: struct { callee: *Expr, args: []*Expr },
    ternary: struct { cond: *Expr, then: *Expr, els: *Expr },
    /// `sizeof(type-name)`: the type is resolved at parse time, by `parseTypeSpec`, so there
    /// is nothing left to lower but the constant.
    sizeof_type: ctype.CType,
    /// `sizeof expr`: the operand is unevaluated. There are no side effects and no IR is
    /// emitted. Only its `CType` matters, computed statically by lowering's `typeOf`.
    sizeof_expr: *Expr,
    /// `"..."`: the DECODED bytes, with no quotes, escapes already resolved by the lexer's
    /// `str_lit` token. It is duped into the parser arena so it outlives the token, which is
    /// freed right after parsing (see `parse`'s `defer lexer.freeTokens`). Lowering mints a
    /// fresh anonymous `.rodata` `DataObject` per occurrence (NUL-terminated) and treats the
    /// expression as an array `char[bytes.len + 1]` that decays to `char*` as an rvalue,
    /// exactly like an array name (`lower.lowerExpr`'s `.name` decay).
    str_lit: []const u8,
    /// `L"..."`: a WIDE string literal, the DECODED bytes (no quotes, no `L` prefix, escapes
    /// already resolved by the lexer's `wstr_lit` token, one byte per character, the same
    /// decode shape as `.str_lit`), duped into the parser arena. `wchar_t` is modeled as a
    /// signed 32-bit int on every target this frontend supports, so lowering widens each
    /// byte to 4 bytes when it mints the `.rodata` object (`lower.lowerWStrLit`). This node
    /// stores only the 1-byte-per-char decoded text, same as `.str_lit`. It is typed
    /// `int[bytes.len + 1]`, decaying to `int*` as an rvalue: the `wchar_t[]`/`wchar_t*`
    /// shape, mirroring `.str_lit`'s `char[]`/`char*`.
    wstr_lit: []const u8,
    /// `(type)e`: a cast to `target`, an UNQUALIFIED type for now (no `const`/`volatile`
    /// cast targets yet. See the parser's `.lparen` disambiguation, which only fires when a
    /// type-name clearly follows the `(`). `operand` is a unary expression, parsed via
    /// `parseAtom`, so `(int)-x`, `(int)(y)`, and nested `(int)(char)x` all parse correctly.
    /// Lowering routes it through `L.convertTo`: int-to-int and float-to-float, plus
    /// int-to-pointer.
    cast: struct { target: ctype.CType, operand: *Expr },
    /// `(void)e`: a statement-expr discard. It evaluates `e` for its side effects only. The
    /// result is unused. Kept as its own variant, rather than `.cast` with some placeholder
    /// target, because there is no genuine `void` `CType` this frontend can put in
    /// `.cast.target`. See `ctype.zig`, which has no `void` case at all.
    cast_void: *Expr,
    /// `__builtin_va_arg(ap, type-name)`: `<stdarg.h>`'s `va_arg(ap, ty)` macro expands to
    /// this, see `stdarg.zig`. It fetches the next variadic argument of type `ty` from the
    /// `va_list` lvalue `ap`. `ty` is resolved at PARSE TIME (`parseTypeSpec` plus
    /// `parseAbstractDeclarator`, the exact same type-name parse `sizeof`/a cast use). Unlike
    /// `sizeof`'s `type-name`, this is a value-producing expression, so `ap` stays a full
    /// `*Expr` (the `va_list` lvalue, lowered to its address).
    va_arg: struct { ap: *Expr, ty: ctype.CType },
    /// `({ stmt* })` (a GNU statement expression): a compound statement used as an
    /// EXPRESSION. Its value is its LAST statement's value if that statement is an
    /// expression-statement, else `void` (mirrors `Expr.cast_void`'s convention: no genuine
    /// void CType, reports as `int`, never actually observed. See `lower.zig`'s `.stmt_expr`
    /// arm). Parsed by `parseAtom`'s `.lparen` arm, recognized by a `{` right after the `(`,
    /// and lowered by running its statements in a fresh scope, exactly like an ordinary
    /// `{ }` block.
    stmt_expr: []Stmt,
    /// `(type-name){ initializer-list }` (a C99 compound literal): an UNNAMED object of type
    /// `ty`, initialized by `init` exactly like a local declaration's brace-list
    /// (`lower.lowerAggregateInit`, the same engine used for local aggregate initializers).
    /// Per C99, it is also an LVALUE, so `&(int){5}`, `(struct S){1,2}.x`, and
    /// `(int[3]){1,2,3}[1]` all work through the ordinary postfix loop
    /// (`.addrof`/`.member`/`.index`), for free, once `parseAtom` returns this node. Parsed
    /// by `parseAtom`'s `.lparen` cast arm: a `{` right after the type-name's closing `)`,
    /// checked BEFORE the plain-cast operand parse, makes this a compound literal instead of
    /// `Expr.cast`. Anything else keeps the ordinary cast behavior, unchanged. Lowered to a
    /// fresh AUTOMATIC (stack) temp, see `lower.zig`'s `materializeCompoundLiteral`.
    /// File-scope/`static` use is OUT OF SCOPE and fails closed: `consteval.evalInit` has no
    /// case for this variant.
    compound_literal: struct { ty: ctype.CType, init: *Initializer },
    /// `lhs, rhs` (the C comma OPERATOR, C 6.5.17): evaluate `lhs`, discard its value, then
    /// evaluate `rhs`. The whole expression's value AND type are `rhs`'s. It is
    /// left-associative (`a, b, c` parses as `(a, b), c`, see `parseComma`). It is NOT an
    /// lvalue: `lower.lvalueTypeOf`/`lowerAddr` both fail closed on this node. Parsed only by
    /// `parseComma`, the `expression` grammar production. `parseExpr` (`assignment-
    /// expression`, no top-level comma) never builds one, so a comma inside call arguments,
    /// an initializer list, a `_Generic` association, or a declarator list stays a plain
    /// SEPARATOR, unaffected.
    comma: struct { lhs: *Expr, rhs: *Expr },
    // `_Generic(controlling-expr, type-name: expr, ..., default: expr)` (C11 6.5.1.1) is
    // resolved ENTIRELY AT PARSE TIME, in `parseAtom`, not here. The controlling expression's
    // STATIC type (`typeOfExpr`, unevaluated, see there) picks exactly ONE association's
    // `expr`, which becomes the whole `_Generic` expression. So there is no dedicated `Expr`
    // case for it at all, only whichever arm's `*Expr` won. This also guarantees the
    // standard's "the other associations are not evaluated" rule for free: their `*Expr`
    // nodes are parsed, so the syntax is checked, but never linked into the returned tree,
    // so `lower.zig` never visits them.
};

/// One `case <int> :` or `default :` arm of a `switch`: `label` null = `default`. `body`
/// runs from this label up to the next label, or the closing `}`. C fallthrough means the
/// lowering does NOT implicitly break between arms.
pub const Case = struct { label: ?i64, body: []Stmt };

/// A statement: an `int` declaration with initializer, an expression statement
/// (assignment / compound assignment), a `return`, a `{ }` compound block (its own
/// lexical scope), an `if`/`else` (`els` empty = no else), a `while` loop, a `for` loop
/// (`init`/`cond`/`incr` each optional, the loop opens a scope for a declared `init`
/// variable), a `switch` (`cases` in source order, at most meaningfully one `default`, but
/// parsing does not enforce that), or a `break`/`continue` (`break` valid inside a loop or
/// a switch. `continue` only inside a loop, skipping any enclosing switch. Lowering rejects
/// them otherwise).
pub const Stmt = union(enum) {
    /// `init` is a full `Initializer`, not a bare `Expr`, so a local (like a global) can
    /// carry a brace-list/string aggregate initializer. Only a `static` local's is ever
    /// compile-time-folded (`lower.lowerStmt`'s `.decl` arm). An ordinary (automatic) local
    /// only accepts the `.expr` shape, lowered as a runtime store, and fails closed
    /// (`error.Unsupported`) on a `.list`, which is out of scope for stack storage.
    decl: struct { name: []const u8, init: ?*Initializer, ty: ctype.CType, storage: StorageClass = .none, is_const: bool = false, quals: ctype.Quals = .{} },
    expr: *Expr,
    /// `null` is a valueless `return;`, valid C for any return type, not just `void`. A
    /// non-`void` function's `return;` yields an indeterminate value. gcc accepts it with a
    /// warning, suppressed here by `-w`.
    ret: ?*Expr,
    block: []Stmt,
    /// A multi-declarator declaration `T a, b, c;`. Each element is a `.decl` sharing the
    /// base type. Unlike `.block`, it opens NO new scope: the declared names belong to the
    /// ENCLOSING block, so they stay visible in the statements that follow.
    decl_group: []Stmt,
    if_: struct { cond: *Expr, then: []Stmt, els: []Stmt },
    while_: struct { cond: *Expr, body: []Stmt },
    for_: struct { init: ?*Stmt, cond: ?*Expr, incr: ?*Expr, body: []Stmt },
    do_: struct { body: []Stmt, cond: *Expr },
    switch_: struct { value: *Expr, cases: []Case },
    break_,
    continue_,
    /// `goto name;`. `name` is duped into the parser arena. Computed goto (`goto *expr;`) is
    /// out of scope and fails closed at parse time.
    goto_: []const u8,
    /// `name: stmt`, a label prefixing exactly one statement. `name` is duped into the
    /// parser arena. A label has FUNCTION scope in C, so `lower.zig` resolves it against the
    /// whole function body, not this statement's lexical scope.
    label: struct { name: []const u8, body: *Stmt },
};

/// Fold an expression to a compile-time integer constant, for `case` labels only:
/// literals and negation (for example `case -1:`). Anything else (names, arithmetic, ...) is not
/// a valid case label here, so returns null and the caller reports `error.UnexpectedToken`.
fn constInt(e: *const Expr) ?i64 {
    return switch (e.*) {
        .int_lit => |v| v.value,
        .negate => |x| if (constInt(x)) |n| -n else null,
        else => null,
    };
}

/// The literal's type from its suffix and value (C89 6.1.3.2, common-case subset). The
/// suffix sets a MINIMUM rank and signedness (`u`/`U` gives unsigned, one `l`/`L` gives
/// long, two gives long long). The parser then picks the smallest rank at or above that
/// minimum whose type, unsigned if `u`, else signed, holds `value`. `value` is always >= 0.
/// A lexed literal never carries its own minus, since `Expr.negate` wraps it instead, so
/// "fits" is just an upper-bound check. Deferred edge case: a literal too large even for
/// `unsigned long long` is out of scope. The digit run must already fit an `i64` to parse
/// at all.
fn intLitType(l: layout_mod.TargetLayout, value: u64, has_u: bool, long_count: u8) ctype.CType {
    const suffix_rank: ctype.Rank = if (long_count >= 2) .longlong else if (long_count == 1) .long else .int;
    const ranks = [_]ctype.Rank{ .int, .long, .longlong };
    for (ranks) |r| {
        if (@intFromEnum(r) < @intFromEnum(suffix_rank)) continue;
        const it: ctype.IntType = .{ .rank = r, .signed = !has_u };
        const bits = it.bits(l);
        // `int`/`long long` are fixed at 32/64. Only `long` varies (32 on ILP32, 64 on
        // LP64), so `bits` is always 32 or 64 here. The 64-bit ranks always hold a value that
        // parsed into a `u64` for the unsigned case. A signed 64-bit rank holds it only when
        // the top bit is clear, else it needs the unsigned last resort below.
        const fits = if (bits >= 64)
            (has_u or value <= 0x7FFF_FFFF_FFFF_FFFF)
        else
            (if (has_u) value < 0x1_0000_0000 else value <= 0x7FFF_FFFF);
        if (fits) return .{ .int = it };
    }
    // Too large for a signed long long with no `u`. A hex/octal literal then takes the
    // unsigned long long type (C11 6.4.4.1), so a full 64-bit value never overflows the type.
    return ctype.mkInt(.longlong, true);
}

/// The numeric value of an `int_lit` token's digit text. Any trailing `[uUlL]` suffix is
/// already peeled off by the caller. This function detects the C radix from its prefix:
/// `0x`/`0X` gives hex, `0b`/`0B` gives binary, a leading `0` followed by more digits gives
/// octal (C89's leading-zero form, no letter marker), else decimal. A bare `0` stays
/// decimal, since there are no digits left to reinterpret as octal.
fn parseIntLitValue(comptime T: type, text: []const u8) Error!T {
    var base: u8 = 10;
    var digits = text;
    if (text.len > 2 and text[0] == '0' and (text[1] == 'x' or text[1] == 'X')) {
        base = 16;
        digits = text[2..];
    } else if (text.len > 2 and text[0] == '0' and (text[1] == 'b' or text[1] == 'B')) {
        base = 2;
        digits = text[2..];
    } else if (text.len > 1 and text[0] == '0') {
        base = 8;
        digits = text[1..];
    }
    return std.fmt.parseInt(T, digits, base) catch error.Overflow;
}

/// A parsed function parameter: its declared type and name, plus its own qualifiers.
/// `const int p` reads as a param-level `const`, though that is rare. This differs from the
/// pointee-const of `const int *p`, which lives on `ty.ptr.quals` instead. See
/// `ctype.CType.ptr`'s doc comment.
pub const Param = struct { name: []const u8, ty: ctype.CType, quals: ctype.Quals = .{} };

/// A parsed function: its name, parameters, its return type, and its statement-list body.
/// `is_variadic` is true for `int f(int a, ...) { ... }`. `params` then holds only the
/// fixed (named) ones, mirroring `ctype.FuncType.is_variadic`.
pub const Func = struct { name: []const u8, params: []Param, body: []Stmt, ret: ctype.CType, is_variadic: bool = false, is_static: bool = false };

/// A global's storage-class specifier. `.none` is an ordinary definition, backed by `.bss`,
/// `.data`, or `.rodata` per its initializer, see `lower.compile`. `.static` (internal
/// linkage) and `.extern_` (a declaration with no storage of its own) are recorded, but
/// linkage does not change single-TU behavior here. A `static` global is lowered exactly
/// like a plain one.
pub const StorageClass = enum { none, static, extern_ };

/// A bodyless function DECLARATION: `int f(int);`/`extern int f(int);`. This is a
/// prototype, not a definition, so there is no `body` (unlike `Func`). `ret == null` means
/// a `void`-returning prototype, matching `Func`'s and `FuncType`'s convention. See
/// `ctype.FuncType`'s doc comment. `params` carries NAMES, unlike `ctype.FuncType.params`,
/// which is bare types only, since a prototype's params can be named for documentation
/// (`int f(int count);`) even with no body to bind them to. `storage` records whether this
/// came from a plain `int f(int);` (`.none`) or an `extern int f(int);` (`.extern_`). Either
/// shape is a declaration, never a definition, here. It is not lowered yet. External calls
/// are added later. `is_variadic` is true for `int f(int a, ...);`. `params` then holds
/// only the fixed (named) ones, mirroring `ctype.FuncType.is_variadic`.
pub const FuncDecl = struct { name: []const u8, params: []Param, ret: ?ctype.CType, storage: StorageClass, is_variadic: bool = false };

/// One step of a C99 DESIGNATOR chain on a brace-list element: a named field
/// (`.field = ...`) or a constant array index (`[index] = ...`). `parseDesignatedInitializer`
/// parses a whole chain (`.a.b`, `[i].f`, `[i][j]`) into a slice, outermost step first. Each
/// step picks the position within the aggregate that the NEXT step, or, on the last step,
/// the initializer's value, applies to.
pub const Designator = union(enum) { field: []const u8, index: u64 };

/// An initializer: either a single expression (`int g = 1;`), or a brace-list of them for
/// an aggregate (`int a[3] = {1,2,3};`), recursively, so nested braces give nested
/// aggregates, for example `int a[2][2] = {{1,2},{3,4}};`. `designators` is non-empty ONLY
/// on a brace-list ELEMENT that used a C99 designated form (`.a[3].b`). Every other
/// `Initializer`, the outermost one handed to `evalInit`/`lowerAggregateInit`, or a plain
/// positional list element, leaves it at the `&.{}` default. `consteval.evalInit` walks
/// this: a scalar target accepts `.expr` (or a single-element `.list` wrapping one). An
/// `.array`/`.@"struct"` target requires `.list` (or, for a `char` array, a `.expr` that is
/// a `.str_lit`) and zero-fills anything the list leaves short. It is used for both
/// file-scope globals (`GlobalDecl.init`) and locals (`Stmt.decl.init`). A `static` local's
/// is compile-time-folded exactly like a global's. An ordinary (automatic) local only
/// supports the bare `.expr` shape or a `.list`, lowered as a runtime store sequence. See
/// `lower.lowerStmt`'s `.decl` arm.
pub const Initializer = struct {
    designators: []Designator = &.{},
    value: union(enum) { expr: *Expr, list: []Initializer },
};

/// A parsed file-scope variable declaration: its name, declared type, storage class,
/// const-ness, and optional initializer. `init = null` means a bare `int g;`, zero-
/// initialized `.bss`, see `lower.compile`. `is_const` (a leading `const`) routes a non-null
/// initializer to `.rodata` instead of `.data`.
pub const GlobalDecl = struct { name: []const u8, ty: ctype.CType, storage: StorageClass, is_const: bool, init: ?*Initializer, quals: ctype.Quals = .{} };

/// If `ty` is an array with an UNSIZED (`len == 0`) OUTER dimension and `init` is present,
/// resolve that dimension's length from the initializer: a brace-list's element count
/// (`int a[] = {1,2,3,4}` gives `len = 4`), or, for a `char` array initialized by a bare
/// string literal, not a brace list, the decoded string's length plus one for the NUL
/// (`char s[] = "hi"` gives `len = 3`). Returns `ty` unchanged in every other case: no
/// initializer, an inner dimension left unsized, or an initializer shape that cannot fix a
/// length here. `hasUnsizedDim`/lowering fail those closed downstream rather than this
/// function guessing. Shared by `parse`'s file-scope global loop and `parseDeclStmt`'s local
/// declarations, since a `static` local's aggregate/string initializer needs the same fixup.
/// A list's HIGHEST reached position, not merely its element COUNT, sizes the array, since
/// an INDEX designator (`[4] = 9`) can reach past the element count. `int a[] = {[4] = 9}`
/// is a 5-element array in C99, matching this same position-tracking (a designator SETS the
/// current position, a positional element uses-then-advances it) that the runtime/const-fold
/// paths use to place each element.
fn resolveUnsizedArrayLen(ty: ctype.CType, init: ?*const Initializer) ctype.CType {
    if (ty != .array or ty.array.len != 0) return ty;
    const iz = init orelse return ty;
    const len: u64 = switch (iz.value) {
        .list => |list| blk: {
            var pos: u64 = 0;
            var high: u64 = 0;
            for (list) |item| {
                if (item.designators.len > 0 and item.designators[0] == .index) pos = item.designators[0].index;
                if (pos + 1 > high) high = pos + 1;
                pos += 1;
            }
            break :blk high;
        },
        .expr => |e| if (e.* == .str_lit) e.str_lit.len + 1 else return ty,
    };
    return .{ .array = .{ .elem = ty.array.elem, .len = len, .quals = ty.array.quals } };
}

/// A parsed translation unit. Owns all AST nodes through `arena`.
pub const Unit = struct {
    arena: std.heap.ArenaAllocator,
    funcs: []Func,
    /// Every file-scope variable declaration, in source order.
    globals: []GlobalDecl,
    /// Every bodyless function declaration, a plain prototype (`int f(int);`) or an
    /// `extern` one (`extern int f(int);`), in source order. A function DEFINITION
    /// (`int f(int){...}`) is never listed here, only in `funcs`.
    func_decls: []FuncDecl,

    pub fn deinit(self: *Unit) void {
        self.arena.deinit();
    }
};

pub const Error = preproc.Error || error{ UnexpectedToken, Overflow, Unsupported, StructRedefinition, BitfieldTooWide, BitfieldNonInteger, BitfieldZeroNamed, GenericNoMatch } || ctype.Error;

/// Which kind of binary node an operator token builds: plain arithmetic/bitwise/shift, a
/// relational/equality comparison, or a short-circuiting logical `&&`/`||`. Each is a
/// different `Expr` variant.
const OpInfo = union(enum) { arith: BinOp, cmp: CmpOp, logand, logor };

/// Binding power of a binary operator token, higher binds tighter, or null. Relational
/// (`lp = 10`) and equality (`lp = 9`) sit below bitwise-or (`lp = 12`). `&&` (`lp = 6`) and
/// `||` (`lp = 5`) sit below all comparisons, matching C's precedence ladder. They
/// short-circuit, so they lower to branches (see `lowerExpr`), not a bitwise op.
fn binPrec(kind: lexer.Kind) ?struct { info: OpInfo, lp: u8 } {
    return switch (kind) {
        .plus => .{ .info = .{ .arith = .add }, .lp = 20 },
        .minus => .{ .info = .{ .arith = .sub }, .lp = 20 },
        .star => .{ .info = .{ .arith = .mul }, .lp = 30 },
        .slash => .{ .info = .{ .arith = .div }, .lp = 30 },
        .percent => .{ .info = .{ .arith = .rem }, .lp = 30 },
        .lshift => .{ .info = .{ .arith = .shl }, .lp = 18 },
        .rshift => .{ .info = .{ .arith = .shr }, .lp = 18 },
        .amp => .{ .info = .{ .arith = .bit_and }, .lp = 14 },
        .caret => .{ .info = .{ .arith = .bit_xor }, .lp = 13 },
        .pipe => .{ .info = .{ .arith = .bit_or }, .lp = 12 },
        .lt => .{ .info = .{ .cmp = .lt }, .lp = 10 },
        .le => .{ .info = .{ .cmp = .le }, .lp = 10 },
        .gt => .{ .info = .{ .cmp = .gt }, .lp = 10 },
        .ge => .{ .info = .{ .cmp = .ge }, .lp = 10 },
        .eq_eq => .{ .info = .{ .cmp = .eq }, .lp = 9 },
        .bang_eq => .{ .info = .{ .cmp = .ne }, .lp = 9 },
        .amp_amp => .{ .info = .{ .logand = {} }, .lp = 6 },
        .pipe_pipe => .{ .info = .{ .logor = {} }, .lp = 5 },
        else => null,
    };
}

/// The compound operator behind an assignment token, or null if not an assignment token.
/// The outer optional means "is an assignment operator". The inner optional means "is
/// compound". A null inner value means plain `=`.
fn assignOp(kind: lexer.Kind) ??BinOp {
    return switch (kind) {
        .assign => @as(?BinOp, null),
        .plus_eq => .add,
        .minus_eq => .sub,
        .star_eq => .mul,
        .slash_eq => .div,
        .percent_eq => .rem,
        .amp_eq => .bit_and,
        .pipe_eq => .bit_or,
        .caret_eq => .bit_xor,
        .lshift_eq => .shl,
        .rshift_eq => .shr,
        else => null,
    };
}

/// Whether `kind` is a type-specifier keyword. This is used only to disambiguate
/// `sizeof (`: a `(` followed by one of these starts a type-name (`sizeof(int)`), anything
/// else starts a parenthesized expression (`sizeof(x)`). This is a keyword-only check. It
/// does not recognize a typedef-name token, and cannot, being kind-only with no `Parser` to
/// consult `typedefs`. So `sizeof(u64)`, with a typedef'd `u64`, still parses as `sizeof
/// expr`, looking up `u64` as a variable, not `sizeof(type-name)`. This is a known, untested
/// gap, not a miscompile: it fails to resolve `u64` as a name and errors, rather than
/// silently doing the wrong thing. Otherwise this set is exactly the tokens `parseTypeSpec`
/// accepts, plus `void` (not a `parseTypeSpec` type, but still unambiguously a type-name
/// token). For example `sizeof(void)` exists in C even though this frontend does not lower
/// it to anything.
fn startsTypeSpec(kind: lexer.Kind) bool {
    return switch (kind) {
        // `const`/`volatile` lead a qualified type-name, so a cast or `sizeof` can begin
        // with one: `(const unsigned char *) p`, `sizeof (volatile int)`. `parseTypeSpec`
        // consumes the leading qualifier run before the base type.
        .kw_int, .kw_char, .kw_short, .kw_long, .kw_unsigned, .kw_signed, .kw_void, .kw_struct, .kw_union, .kw_enum, .kw_float, .kw_double, .kw_bool, .kw_const, .kw_volatile => true,
        else => false,
    };
}

/// `x` rounded up to the next multiple of `a`. `a` is a power of two, as every C
/// alignment/size here is. Shared by `parseStructOrUnion`'s field-offset and
/// overall-size computation.
fn alignUp(x: u64, a: u64) u64 {
    return (x + a - 1) / a * a;
}

/// `text` is one of the no-op function-specifier/prefix idents this parser recognizes and
/// discards: `inline` and its GNU underscored spellings (a function marked `inline` still
/// compiles as an ordinary definition, since VCC does no inlining), `_Noreturn` (VCC does
/// no noreturn-driven codegen), and `__extension__` (GNU's "suppress -pedantic" no-op
/// prefix). None of these change a type, storage class, or value.
fn isNoOpSpecifierIdent(text: []const u8) bool {
    return std.mem.eql(u8, text, "inline") or
        std.mem.eql(u8, text, "__inline") or
        std.mem.eql(u8, text, "__inline__") or
        std.mem.eql(u8, text, "_Noreturn") or
        std.mem.eql(u8, text, "__extension__") or
        // `register` and `auto` are storage-class keywords VCC does not model as lexer
        // keywords. They affect no codegen here, so they are ignored like the specifiers above.
        // Real code writes `register const unsigned char *p1;` in hot loops.
        std.mem.eql(u8, text, "register") or
        std.mem.eql(u8, text, "auto");
}

const Parser = struct {
    toks: []const lexer.Token,
    pos: usize,
    arena: std.mem.Allocator,
    /// The target layout, for resolving plain-`char` signedness in `parseTypeSpec`.
    layout: layout_mod.TargetLayout,
    /// Every `struct`/`union` tag defined so far in this translation unit, arena-owned, the
    /// same arena as everything else. `findStruct` looks it up when `struct Tag` appears as
    /// a bare reference, no `{ ... }` body, to an already-defined tag. Defaults to `.empty`
    /// so every existing test literal that constructs a bare `Parser{ ... }`, with no
    /// `.structs` field, still compiles unchanged.
    structs: std.ArrayList(*ctype.StructDef) = .empty,
    /// Every file-scope `typedef` registered so far, name mapped to aliased CType,
    /// arena-owned same as `structs`. Consulted by `parseTypeSpec`: an `.ident` here in
    /// type-specifier position, with no integer specifier seen yet, is that alias's type.
    /// This is the classic typedef-name-vs-identifier ambiguity, resolved by having the
    /// parser consult this table. Also consulted by `isTypeName` for callers that need a
    /// plain bool. Defaults to `.empty` so every existing bare `Parser{ ... }` test literal
    /// still compiles unchanged.
    typedefs: std.StringHashMapUnmanaged(ctype.CType) = .empty,
    /// Every `enum` constant registered so far, name mapped to its `i64` value, arena-owned
    /// same as `structs`/`typedefs`. Populated by `parseEnum`, auto-incrementing from 0, or
    /// from an explicit `= const-int`, and consulted by `parseAtom`'s ident case, which
    /// substitutes a registered name for its `.int_lit` value at PARSE TIME (see there for
    /// why this is simplest: it avoids threading a constant table into `lower.zig`) UNLESS
    /// the name is shadowed by an in-scope variable/param (see `locals`). Defaults to
    /// `.empty` so every existing bare `Parser{ ... }` test literal still compiles unchanged.
    enum_consts: std.StringHashMapUnmanaged(i64) = .empty,
    /// Every param/local declared so far in THE FUNCTION CURRENTLY BEING PARSED, oldest
    /// first. Each binding carries its TYPE too, and the whole thing has genuine BLOCK
    /// SCOPING. Consulted two ways: `parseAtom` checks whether a name is bound at all, so a
    /// local/param shadows an `enum` constant of the same name (`enum{A,B}; int f(int A){...
    /// A...}` must use the parameter, not the constant). `typeOfExpr`, the PARSE-TIME,
    /// unevaluated static-type resolver `typeof`/`_Generic` need, see there, asks a bound
    /// name's TYPE. Both go through `lookupLocalType`, which searches NEWEST entry first, so
    /// a shadowing declaration in a nested block is found ahead of an outer one of the same
    /// name, mirroring `lower.zig`'s `env`/`lookup`. Block scoping mirrors `lower.zig`'s
    /// `scopeMark`/`popScope` too: `pushScope`/`popScope` bracket every scope-opening
    /// construct, a `{ ... }` compound body, a `switch` body, a `for` loop's own init-scope,
    /// see each call site, so a scope's declarations are simply TRUNCATED off the end when
    /// the scope closes, restoring whatever binding, or absence, preceded the shadow. This
    /// is what fixes a bug found in review: an OLDER flat map kept a shadow's type visible
    /// past its own block's closing `}`, so `typeof(x)`/`_Generic(x, ...)` after a shadowing
    /// block silently resolved `x` to the INNER (out-of-scope) type instead of the outer
    /// one. A function's own parameters are pushed at FUNCTION scope, before any nested
    /// `pushScope`, so they stay visible for the whole body, same as before this fix. It is
    /// cleared wholesale, not merely popped, at the start of each new function/top-level
    /// item, see the two clear sites below, since nothing needs to survive across functions.
    /// Defaults to `.empty` so every existing bare `Parser{ ... }` test literal still
    /// compiles unchanged.
    locals: std.ArrayList(LocalBinding) = .empty,

    /// The name of the function whose body is currently being parsed, for the predefined
    /// `__func__`/`__FUNCTION__`/`__PRETTY_FUNCTION__` identifiers. Each expands to a string
    /// literal of this name. It is empty outside any function body. It is set before a
    /// body's statements are parsed by both definition paths (`parseFuncAfterType` and
    /// `finishFuncOrDecl`).
    cur_func_name: []const u8 = "",

    /// Whether an `inline`/`__inline`/`__inline__` specifier led the top-level item
    /// currently being parsed. VCC does no inlining, so it emits every inline function out
    /// of line, but as a LOCAL object symbol. Each translation unit that includes the header
    /// gets its own private copy, so the copies never collide at link time (gnulib's
    /// `_GL_INLINE` helpers such as `mb_width_aux` are defined in a header and pulled into
    /// many objects). Reset per top-level item, set by the leading no-op-specifier skip.
    saw_inline: bool = false,

    /// Whether `name` is a registered typedef-name. This is a thin wrapper over `typedefs`
    /// for callers (declarator/param/decl parsers) that just need a yes/no, without
    /// themselves reaching into the map. `__builtin_va_list` is ALWAYS a type name, even
    /// before any `typedef` registers it. It is a compiler built-in, resolved directly by
    /// `parseTypeSpecBase` rather than through the `typedefs` table. See there for why: a
    /// table entry would need seeding into every `Parser{...}` construction site, including
    /// the many bare test literals that build one directly.
    fn isTypeName(self: *Parser, name: []const u8) bool {
        return std.mem.eql(u8, name, "__builtin_va_list") or self.typedefs.contains(name);
    }

    /// One declared param/local's name and TYPE, in the order it was pushed onto `locals`.
    const LocalBinding = struct { name: []const u8, ty: ctype.CType };

    /// Mark the current scope depth in `locals`. Restore it with `popScope` to drop every
    /// binding a nested scope pushed. See `locals`'s doc comment for the whole mechanism,
    /// mirroring `lower.zig`'s `scopeMark`.
    fn pushScope(self: *Parser) usize {
        return self.locals.items.len;
    }
    /// Close the scope opened by the matching `pushScope`: truncate `locals` back to `mark`,
    /// discarding every binding declared since (mirrors `lower.zig`'s `popScope`). This
    /// never fails, so every call site uses `defer`.
    fn popScope(self: *Parser, mark: usize) void {
        self.locals.shrinkRetainingCapacity(mark);
    }
    /// Bind `name` to `ty` as a new param/local of the CURRENT (innermost) scope. A
    /// redeclaration of the same name in the SAME scope simply appends another entry.
    /// `lookupLocalType` finds the newest one, so the later declaration still wins.
    fn declareLocal(self: *Parser, name: []const u8, ty: ctype.CType) Error!void {
        try self.locals.append(self.arena, .{ .name = name, .ty = ty });
    }
    /// `name`'s TYPE if it is a currently-in-scope param/local, else `null`. This searches
    /// `locals` NEWEST first, so a shadowing declaration in an enclosing (still-open) scope
    /// is found before an outer one of the same name (see `locals`'s doc comment).
    fn lookupLocalType(self: *Parser, name: []const u8) ?ctype.CType {
        var i = self.locals.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.locals.items[i].name, name)) return self.locals.items[i].ty;
        }
        return null;
    }

    fn peek(self: *Parser) lexer.Token {
        return self.toks[self.pos];
    }

    fn expect(self: *Parser, kind: lexer.Kind) Error!lexer.Token {
        const t = self.toks[self.pos];
        if (t.kind != kind) return error.UnexpectedToken;
        self.pos += 1;
        return t;
    }

    fn node(self: *Parser, value: Expr) Error!*Expr {
        const p = try self.arena.create(Expr);
        p.* = value;
        return p;
    }

    /// primary := atom postfix*
    /// postfix := '[' expr ']' | '.' ident | '->' ident | '(' args ')'
    /// Every postfix token right after an atom, or a previous postfix, applies to the
    /// result, chaining left-to-right: `a[i][j]` parses as `index{ base = index{a, i}, idx =
    /// j }`, `arr[i].field` as `member{ base = index{arr, i}, field }`, `p->a.b` as
    /// `member{ base = member{base=p, field=a, arrow=true}, field=b, arrow=false }`. Each
    /// postfix token nests one level deeper, matching C's left-to-right postfix
    /// associativity (this is also what multi-dim arrays already relied on). Wrapping the
    /// atom here, rather than inside `parseAtom` itself, is what gives postfix its correct C
    /// precedence relative to the unary operators below. Each of those recurses into
    /// `parsePrimary`, not `parseAtom`, for its operand, so `-a[i]` parses as `-(a[i])` and
    /// `&a[i]` as `&(a[i])`, matching C: postfix binds tighter than prefix unary.
    fn parsePrimary(self: *Parser) Error!*Expr {
        var result = try self.parseAtom();
        while (true) {
            switch (self.peek().kind) {
                .lbracket => {
                    self.pos += 1;
                    const idx = try self.parseComma();
                    _ = try self.expect(.rbracket);
                    result = try self.node(.{ .index = .{ .base = result, .idx = idx } });
                },
                .dot => {
                    self.pos += 1;
                    const nm = try self.expect(.ident);
                    result = try self.node(.{ .member = .{ .base = result, .field = try self.arena.dupe(u8, nm.text), .arrow = false } });
                },
                .arrow => {
                    self.pos += 1;
                    const nm = try self.expect(.ident);
                    result = try self.node(.{ .member = .{ .base = result, .field = try self.arena.dupe(u8, nm.text), .arrow = true } });
                },
                // Postfix `++`/`--` binds exactly as tight as `[]`/`.`/`->`. `arr[i]++`,
                // `p->field++`, and similar expressions all resolve their lvalue first,
                // same as those.
                .plus_plus => {
                    self.pos += 1;
                    result = try self.node(.{ .incdec = .{ .target = result, .inc = true, .prefix = false } });
                },
                .minus_minus => {
                    self.pos += 1;
                    result = try self.node(.{ .incdec = .{ .target = result, .inc = false, .prefix = false } });
                },
                // A call: `( args )` applies to ANY already-parsed postfix expression, not
                // just a bare identifier. `foo(...)`, `(foo)(...)` (the grouped-expr
                // `(foo)` from `parseAtom`'s `.lparen` case becomes `result` here), and
                // `tbl[i](...)` (chained off `.index`) all parse the same way.
                .lparen => {
                    self.pos += 1;
                    var args: std.ArrayList(*Expr) = .empty;
                    if (self.peek().kind != .rparen) {
                        while (true) {
                            try args.append(self.arena, try self.parseExpr());
                            if (self.peek().kind == .comma) {
                                self.pos += 1;
                                continue;
                            }
                            break;
                        }
                    }
                    _ = try self.expect(.rparen);
                    result = try self.node(.{ .call = .{ .callee = result, .args = try args.toOwnedSlice(self.arena) } });
                },
                else => break,
            }
        }
        return result;
    }

    /// atom := int_lit | ident | '(' expr ')' | '-' primary | '~' primary | '!' primary
    ///       | '&' primary | '*' primary | 'sizeof' (...)
    fn parseAtom(self: *Parser) Error!*Expr {
        // `__extension__` may also prefix a bare EXPRESSION (glibc's macro-body idiom, for
        // example `__extension__ (2 + 40)`), not just a declaration (see the file-scope and
        // `parseStmt` leading skips above). It is a no-op prefix, recursing straight back
        // into `parseAtom` for whatever follows it.
        if (self.peek().kind == .ident and std.mem.eql(u8, self.peek().text, "__extension__")) {
            self.pos += 1;
            return self.parseAtom();
        }
        const t = self.peek();
        switch (t.kind) {
            .minus => {
                self.pos += 1;
                return self.node(.{ .negate = try self.parsePrimary() });
            },
            .tilde => {
                self.pos += 1;
                return self.node(.{ .complement = try self.parsePrimary() });
            },
            .bang => {
                self.pos += 1;
                return self.node(.{ .lognot = try self.parsePrimary() });
            },
            // Prefix `&`/`*` (address-of / dereference). Both bind like the other unary
            // ops above, recursing into `parsePrimary`, not `parseBin`. The token position,
            // start of a primary, is what disambiguates these from the INFIX `&`/`*`
            // (bitwise-and / multiply) that `binPrec` handles once a primary is already
            // parsed.
            .amp => {
                self.pos += 1;
                return self.node(.{ .addrof = try self.parsePrimary() });
            },
            .star => {
                self.pos += 1;
                return self.node(.{ .deref = try self.parsePrimary() });
            },
            // Prefix `++`/`--`: same binding as the other unary ops above. It recurses into
            // `parsePrimary`, not `parseAtom`, for the operand, so `++a[i]` parses as
            // `++(a[i])`, matching C: postfix binds tighter than prefix unary.
            .plus_plus => {
                self.pos += 1;
                return self.node(.{ .incdec = .{ .target = try self.parsePrimary(), .inc = true, .prefix = true } });
            },
            .minus_minus => {
                self.pos += 1;
                return self.node(.{ .incdec = .{ .target = try self.parsePrimary(), .inc = false, .prefix = true } });
            },
            .kw_sizeof => {
                self.pos += 1;
                // `sizeof (` plus a type-name means `sizeof(type-name)`. Anything else means
                // `sizeof expr`, parenthesized or not, since `parsePrimary` itself handles a
                // bare `(expr)`. A type-name is a type-specifier keyword OR a registered
                // typedef-name, for example `sizeof(size_t)`, the same recognition the cast
                // branch below already does.
                const after_lparen = self.toks[self.pos + 1];
                if (self.peek().kind == .lparen and
                    (startsTypeSpec(after_lparen.kind) or (after_lparen.kind == .ident and self.isTypeName(after_lparen.text))))
                {
                    self.pos += 1; // consume '('
                    const base = try self.parseTypeSpec();
                    const ct = try self.parseAbstractDeclarator(base.ty, base.quals); // for example `sizeof(int*)`, no name
                    _ = try self.expect(.rparen);
                    return self.node(.{ .sizeof_type = ct });
                }
                return self.node(.{ .sizeof_expr = try self.parsePrimary() });
            },
            .lparen => {
                // Cast-vs-parenthesized-expr disambiguation: peek the token right after `(`,
                // like `sizeof (` above, generalized to also recognize a typedef-name, not
                // just a type-specifier keyword. Only when THAT clearly starts a type-name
                // is this a cast. Otherwise it is the existing parenthesized-expr path,
                // unchanged, so a plain `(a + b)` or `(x)` over a variable `x`, never a
                // registered typedef, still parses as before.
                const next = self.toks[self.pos + 1];
                if (startsTypeSpec(next.kind) or (next.kind == .ident and self.isTypeName(next.text))) {
                    self.pos += 1; // consume '('
                    // `(void)e`, a statement-expr discard idiom, versus `(void*)e`/`(void**)e`,
                    // a real cast to a generic object pointer: disambiguated by peeking the
                    // token right after `void`. `)` immediately closes the parenthesis, so
                    // this is the bare discard form (`Expr.cast_void`). Anything else, a `*`
                    // starting a declarator, falls through to the general cast path just
                    // below, where `void` is a first-class `parseTypeSpec` result and
                    // `parseAbstractDeclarator` adds the `*`-run, producing a real
                    // `Expr.cast` targeting `void*`/`void**`.
                    if (next.kind == .kw_void and self.toks[self.pos + 1].kind == .rparen) {
                        self.pos += 1; // consume 'void'
                        _ = try self.expect(.rparen);
                        return self.node(.{ .cast_void = try self.parseAtom() });
                    }
                    const base = try self.parseTypeSpec();
                    // An ABSTRACT declarator, no name, the same `*`-wrapping `sizeof(type-name)`
                    // already uses, so `(int*)`, `(unsigned char *)`, `(char**)` all resolve.
                    // An array/function abstract declarator is not attempted here, out of
                    // scope, same as `sizeof`'s.
                    const target = try self.parseAbstractDeclarator(base.ty, base.quals);
                    _ = try self.expect(.rparen);
                    // `(type-name){ ... }` (a C99 compound literal): a `{` right after the
                    // type-name's closing `)` means this is NOT a cast at all. The brace-list
                    // initializes a fresh unnamed object of `target`'s type (see
                    // `Expr.compound_literal`'s doc comment). Checked BEFORE the plain-cast
                    // operand parse below, so an ordinary `(int)-x`/`(int)(y)`, with no `{`
                    // here, still takes the cast path.
                    if (self.peek().kind == .lbrace) {
                        const iz = try self.arena.create(Initializer);
                        iz.* = try self.parseInitializerValue();
                        return self.node(.{ .compound_literal = .{ .ty = target, .init = iz } });
                    }
                    // The operand is a cast-expression, which includes POSTFIX (a call, index,
                    // or member access binds INSIDE the cast): `(T) f(x)` is `(T)(f(x))`, not
                    // `((T)f)(x)`. `parsePrimary` supplies that postfix on top of `parseAtom`,
                    // so `(int)-x`, `(int)(y)`, and nested `(int)(char)x` still work, each
                    // recursing through `parseAtom`, AND `(wchar_t) towlower(wc)` parses as
                    // one call.
                    const operand = try self.parsePrimary();
                    return self.node(.{ .cast = .{ .target = target, .operand = operand } });
                }
                // `({ stmt* })` (a GNU statement expression): `next` is `{` right after the
                // `(`, neither a type-name (the branch above) nor an ordinary parenthesized
                // expression (the fallback below, which would choke trying to `parseExpr` a
                // `{`). `parseBlockOrStmt` parses the WHOLE `{ ... }`. It is already sitting
                // right at the `{`, since only `(` was consumed here. See `Expr.stmt_expr`'s
                // doc comment for what its VALUE means.
                if (next.kind == .lbrace) {
                    self.pos += 1; // consume '('
                    const stmts = try self.parseBlockOrStmt();
                    _ = try self.expect(.rparen);
                    return self.node(.{ .stmt_expr = stmts });
                }
                self.pos += 1;
                const inner = try self.parseComma();
                _ = try self.expect(.rparen);
                return inner;
            },
            .ident => {
                self.pos += 1;
                // `__func__` (C99), `__FUNCTION__`, and `__PRETTY_FUNCTION__` (GCC) are
                // predefined identifiers, each a string literal of the enclosing function's
                // name. Real code uses them in `assert`/`abort` diagnostic messages.
                if (std.mem.eql(u8, t.text, "__func__") or std.mem.eql(u8, t.text, "__FUNCTION__") or std.mem.eql(u8, t.text, "__PRETTY_FUNCTION__")) {
                    return self.node(.{ .str_lit = try self.arena.dupe(u8, self.cur_func_name) });
                }
                // `__alignof__(type)` / `__alignof` / `_Alignof` yield a type's alignment as
                // an integer constant (glibc `malloca.c` folds `__alignof__(long)` into an enum
                // value). Only the TYPE-NAME form is handled. The operand-expression form
                // (`__alignof__ expr`) needs type inference and is out of scope. Checked before
                // the enum/variable handling, since these names are never variables.
                if ((std.mem.eql(u8, t.text, "__alignof__") or std.mem.eql(u8, t.text, "__alignof") or std.mem.eql(u8, t.text, "_Alignof")) and
                    self.peek().kind == .lparen and
                    (startsTypeSpec(self.toks[self.pos + 1].kind) or (self.toks[self.pos + 1].kind == .ident and self.isTypeName(self.toks[self.pos + 1].text))))
                {
                    self.pos += 1; // consume '('
                    const base = try self.parseTypeSpec();
                    const at = try self.parseAbstractDeclarator(base.ty, base.quals);
                    _ = try self.expect(.rparen);
                    const a: i64 = @intCast(try at.alignOf(self.layout));
                    return self.node(.{ .int_lit = .{ .value = a, .ty = ctype.ulong_t } });
                }
                // An `enum` constant resolves to its `i64` value at PARSE TIME. This is
                // checked BEFORE the call/variable-name handling below, so a registered
                // constant name never reaches `lowerExpr` as a `.name` at all. No constant
                // table needs threading into lowering. An ordinary ident that ISN'T a
                // registered enum constant, the common case, falls straight through
                // unaffected. It substitutes only when `t.text` isn't ALSO a declared
                // variable/param currently in scope (`locals`), so a shadowing local/param
                // must win.
                if (self.enum_consts.get(t.text)) |v| {
                    if (self.lookupLocalType(t.text) == null) {
                        return self.node(.{ .int_lit = .{ .value = v, .ty = ctype.int_t } });
                    }
                    // else: shadowed by an in-scope variable of the same name. Fall through
                    // and treat it as an ordinary variable reference, resolved by lowering's
                    // `.name`.
                }
                // `__builtin_va_arg(ap, type-name)`: its SECOND argument is a TYPE NAME, not
                // an expression (`sizeof`/a cast's own shape), so it cannot parse through the
                // ordinary call-argument-list path (`parsePrimary`'s postfix `(` loop, which
                // parses every arg as a full expression). Intercept it here, ahead of that,
                // the same way `sizeof`/a cast intercept `(` before falling into ordinary
                // expression parsing.
                if (std.mem.eql(u8, t.text, "__builtin_va_arg") and self.peek().kind == .lparen) {
                    self.pos += 1; // '('
                    const ap = try self.parseExpr();
                    _ = try self.expect(.comma);
                    const base = try self.parseTypeSpec();
                    const ty = try self.parseAbstractDeclarator(base.ty, base.quals);
                    _ = try self.expect(.rparen);
                    return self.node(.{ .va_arg = .{ .ap = ap, .ty = ty } });
                }
                // `_Generic(controlling-expr, type-name: expr, ..., default: expr)` (C11
                // 6.5.1.1) is resolved ENTIRELY HERE, at parse time, to whichever
                // association's `expr` matches. See `Expr`'s doc comment for why there is no
                // dedicated node for it. The controlling expression's STATIC type
                // (`typeOfExpr`, unevaluated: `_Generic` never evaluates it either, only asks
                // its type) picks the association whose type-name is COMPATIBLE (`CType.eql`,
                // structural equality, exactly the rule this frontend already uses for
                // conversion/param-matching checks) with it. `default:` catches no match.
                // EVERY association's `expr` is still PARSED, the syntax must be valid, just
                // not RETURNED unless selected. The ones that lose are simply unreachable
                // AST, never visited by `lower.zig`, which is what keeps them unevaluated
                // (no side effects, no undefined-name check) per the standard.
                if (std.mem.eql(u8, t.text, "_Generic") and self.peek().kind == .lparen) {
                    self.pos += 1; // '('
                    const controlling = try self.parseExpr();
                    const controlling_ty = try self.typeOfExpr(controlling);
                    _ = try self.expect(.comma);
                    var selected: ?*Expr = null;
                    var default_expr: ?*Expr = null;
                    while (true) {
                        if (self.peek().kind == .kw_default) {
                            self.pos += 1;
                            _ = try self.expect(.colon);
                            default_expr = try self.parseExpr();
                        } else {
                            const base = try self.parseTypeSpec();
                            const ct = try self.parseAbstractDeclarator(base.ty, base.quals);
                            _ = try self.expect(.colon);
                            const assoc_expr = try self.parseExpr();
                            if (selected == null and ct.eql(controlling_ty)) selected = assoc_expr;
                        }
                        if (self.peek().kind == .comma) {
                            self.pos += 1;
                            continue;
                        }
                        break;
                    }
                    _ = try self.expect(.rparen);
                    if (selected) |s| return s;
                    if (default_expr) |d| return d;
                    return error.GenericNoMatch; // no compatible association and no default: ill-formed
                }
                // A trailing `(` is not special-cased here: a bare identifier always parses
                // as `.name`, and a following call is picked up by `parsePrimary`'s postfix
                // loop instead, uniformly with `(foo)(...)` and `tbl[i](...)`.
                return self.node(.{ .name = try self.arena.dupe(u8, t.text) });
            },
            .int_lit => {
                self.pos += 1;
                // Split the trailing `[uUlL]+` suffix (lexed into the token text) back out
                // from the digits: `u`/`U` -> unsigned, one `l`/`L` -> long, two -> long long.
                var end = t.text.len;
                var has_u = false;
                var long_count: u8 = 0;
                while (end > 0) : (end -= 1) {
                    switch (t.text[end - 1]) {
                        'u', 'U' => has_u = true,
                        'l', 'L' => long_count += 1,
                        else => break,
                    }
                }
                // Parse the magnitude as `u64` so a full-width unsigned literal, for example
                // `0xFFFFFFFFFFFFFFFF`, `UINT64_MAX` in real headers, does not overflow. The
                // stored `value` is the two's-complement BIT PATTERN as `i64`. Its
                // unsigned-ness lives in `ty`, so a consumer reads the bits back through the
                // type.
                const uvalue = try parseIntLitValue(u64, t.text[0..end]);
                const ty = intLitType(self.layout, uvalue, has_u, long_count);
                return self.node(.{ .int_lit = .{ .value = @bitCast(uvalue), .ty = ty } });
            },
            .float_lit => {
                self.pos += 1;
                // Split the trailing float suffix (`f`/`F`/`l`/`L`, lexed into the token
                // text) back out: `f`/`F` gives `float` (f32). No suffix, or `l`/`L` (no
                // extended `long double` here), gives `double` (f64).
                var end = t.text.len;
                var is_f32 = false;
                while (end > 0) : (end -= 1) {
                    switch (t.text[end - 1]) {
                        'f', 'F' => is_f32 = true,
                        'l', 'L' => {},
                        else => break,
                    }
                }
                const value = std.fmt.parseFloat(f64, t.text[0..end]) catch return error.Overflow;
                const ty: ctype.CType = .{ .float = if (is_f32) .f32 else .f64 };
                return self.node(.{ .float_lit = .{ .value = value, .ty = ty } });
            },
            .str_lit => {
                self.pos += 1;
                // `t.text` is the lexer's DECODED bytes, owned by the token, freed by
                // `freeTokens` right after parsing. Dupe it into the parser arena so the AST
                // node outlives it.
                // Adjacent string literals concatenate (C11 5.1.1.2 phase 6): `"ab" "cd"` is
                // one `"abcd"`. Real code writes `_("A") "B"`-style split messages. Only the
                // narrow-narrow case is folded here. A mixed wide/narrow run is out of scope.
                if (self.peek().kind == .str_lit) {
                    var buf: std.ArrayList(u8) = .empty;
                    try buf.appendSlice(self.arena, t.text);
                    while (self.peek().kind == .str_lit) {
                        try buf.appendSlice(self.arena, self.peek().text);
                        self.pos += 1;
                    }
                    return self.node(.{ .str_lit = try buf.toOwnedSlice(self.arena) });
                }
                return self.node(.{ .str_lit = try self.arena.dupe(u8, t.text) });
            },
            .wstr_lit => {
                self.pos += 1;
                // Same decode/dupe shape as `.str_lit` above. `t.text` is 1 byte per
                // character. The 4-byte `wchar_t` widening happens at lowering time.
                return self.node(.{ .wstr_lit = try self.arena.dupe(u8, t.text) });
            },
            else => return error.UnexpectedToken,
        }
    }

    /// Precedence-climbing binary parse: consume operators whose left binding power meets
    /// `min_bp`, recursing at `lp + 1` on the right so equal-precedence operators
    /// associate left.
    fn parseBin(self: *Parser, min_bp: u8) Error!*Expr {
        var lhs = try self.parsePrimary();
        while (binPrec(self.peek().kind)) |e| {
            if (e.lp < min_bp) break;
            self.pos += 1;
            const rhs = try self.parseBin(e.lp + 1); // +1 = left-associative
            lhs = switch (e.info) {
                .arith => |op| try self.node(.{ .binary = .{ .op = op, .lhs = lhs, .rhs = rhs } }),
                .cmp => |op| try self.node(.{ .compare = .{ .op = op, .lhs = lhs, .rhs = rhs } }),
                .logand => try self.node(.{ .logand = .{ .lhs = lhs, .rhs = rhs } }),
                .logor => try self.node(.{ .logor = .{ .lhs = lhs, .rhs = rhs } }),
            };
        }
        return lhs;
    }

    /// expression := assignment-expression (',' assignment-expression)* (C 6.5.17). This is
    /// the comma OPERATOR, a distinct, WIDER production than `parseExpr` (`assignment-
    /// expression`, no top-level comma). It is left-associative: `a, b, c` parses as
    /// `(a, b), c`, folded left-to-right in the loop below (matching `parseBin`'s left-
    /// associative shape, not `parseExpr`'s own right-associative assignment chain). Only the
    /// handful of call sites that are genuinely a C `expression` (a parenthesized group, a
    /// `for`/`if`/`while`/`switch` controlling expression, a `return` value, an expression-
    /// statement, an array subscript, `typeof`'s operand, the `?:` middle) route through this.
    /// Every comma-SEPARATED-LIST context (call arguments, an initializer list, a declarator
    /// list, a `_Generic` association) stays on `parseExpr` so a comma there keeps meaning
    /// "next item", not "comma operator" (see each call site's own comment).
    fn parseComma(self: *Parser) Error!*Expr {
        var e = try self.parseExpr();
        while (self.peek().kind == .comma) {
            self.pos += 1;
            const rhs = try self.parseExpr();
            e = try self.node(.{ .comma = .{ .lhs = e, .rhs = rhs } });
        }
        return e;
    }

    /// expr := lvalue assign-op expr | conditional
    /// Assignment is the lowest-precedence, right-associative production. Parse the LHS as
    /// an ordinary conditional-expression first, then check for an assignment operator. This
    /// is safe because `=`/`+=`/... are never `binPrec` operators, so
    /// `parseBin`/`parseConditional` already stop cleanly right before one without consuming
    /// it. This also generalizes the LHS beyond a bare name for free: `*p = 5` parses `*p`
    /// as a normal prefix-`*` primary (see `parsePrimary`) and lands here the same as a name
    /// would. Whether the LHS is actually an lvalue (name, `*expr`, and so on) is
    /// `lowerAddr`'s job, not the parser's. An LHS like `(a + b) = 5` parses fine and fails
    /// later with `error.Unsupported`. This is `assignment-expression` (C's grammar), NOT
    /// `expression`. A top-level comma is the comma OPERATOR (`parseComma`), a different,
    /// wider production. See `parseComma`'s doc comment for exactly which call sites use
    /// which.
    fn parseExpr(self: *Parser) Error!*Expr {
        const lhs = try self.parseConditional();
        if (assignOp(self.peek().kind)) |compound| {
            self.pos += 1;
            const value = try self.parseExpr(); // right-associative
            return self.node(.{ .assign = .{ .target = lhs, .op = compound, .value = value } });
        }
        return lhs;
    }

    /// conditional := binary ('?' expr ':' conditional)?
    /// Sits just above assignment and below `||`/the binary ladder. `parseBin` never
    /// consumes `?`, it is not in `binPrec`, so it stops cleanly at the operator and this
    /// wraps it. It is right-associative (`a ? b : c ? d : e` == `a ? b : (c ? d : e)`),
    /// matching C. The middle operand is a full expression.
    fn parseConditional(self: *Parser) Error!*Expr {
        const cond = try self.parseBin(0);
        if (self.peek().kind != .question) return cond;
        self.pos += 1;
        const then_e = try self.parseComma(); // the middle can be a full expression
        _ = try self.expect(.colon);
        const else_e = try self.parseConditional(); // right-associative
        return self.node(.{ .ternary = .{ .cond = cond, .then = then_e, .els = else_e } });
    }

    /// Look up an already-defined `struct`/`union` tag by name, newest-first. Shadowing is
    /// not really a concern here, since VCC has no nested tag scopes yet, but scanning
    /// back-to-front costs nothing and matches `L.lookup`'s convention.
    fn findStruct(self: *Parser, name: []const u8) ?*ctype.StructDef {
        var i = self.structs.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.structs.items[i].name, name)) return self.structs.items[i];
        }
        return null;
    }

    /// Arena-allocate a fresh, INCOMPLETE `StructDef` for `name`. This handles a forward
    /// declaration, `struct Tag;`, or a bare reference to a tag never seen before, `struct
    /// Tag *p;`, and registers it in `self.structs`. Its fields/size/alignment are all
    /// placeholder (`&.{}`/`0`/`0`) until, if ever, a body (`struct Tag { ... }`) completes
    /// it in place. See `parseStructOrUnion`, the only caller.
    fn registerIncompleteStruct(self: *Parser, name: []const u8, is_union: bool) Error!*ctype.StructDef {
        const def = try self.arena.create(ctype.StructDef);
        def.* = .{ .name = try self.arena.dupe(u8, name), .is_union = is_union, .fields = &.{}, .size = 0, .alignment = 0, .complete = false };
        try self.structs.append(self.arena, def);
        return def;
    }

    /// Parse `struct Tag { field-decls }`, a definition completing `Tag`, a fresh tag or one
    /// only forward-declared so far, `struct Tag;`/a bare `struct Tag` reference (this
    /// registers `Tag` INCOMPLETE if this is the first time it has been seen. A pointer to
    /// it is valid even if it is never completed in this translation unit), a bare reference
    /// to an already-COMPLETE tag, or a tagless `struct { field-decls }`, always a
    /// definition, since there is no name to defer completion of. Returns the tag's CType.
    /// The caller has already consumed the `struct`/`union` keyword itself.
    fn parseStructOrUnion(self: *Parser, is_union: bool) Error!ctype.CType {
        if (self.peek().kind != .ident) {
            // No tag at all: this can only be a definition (`struct { ... }` right after the
            // keyword), so a body must follow immediately. The def is anonymous, an empty
            // name, NOT appended to `self.structs`, so `findStruct` can never resolve it by
            // name. The only way to reach this exact `StructDef` again is through the
            // `CType` this call returns, usually bound straight to a typedef name by the
            // caller.
            _ = try self.expect(.lbrace);
            const def = try self.arena.create(ctype.StructDef);
            def.* = .{ .name = "", .is_union = is_union, .fields = &.{}, .size = 0, .alignment = 0, .complete = false };
            try self.parseStructBody(def, is_union);
            return .{ .@"struct" = def };
        }
        const tag = try self.expect(.ident);
        if (self.peek().kind != .lbrace) { // a bare reference, not a definition
            const def = self.findStruct(tag.text) orelse try self.registerIncompleteStruct(tag.text, is_union);
            return .{ .@"struct" = def };
        }
        self.pos += 1; // '{'
        // Register (or reuse) `Tag` INCOMPLETE before parsing the body, so a self-reference
        // inside it (`struct Tag *next;`) resolves via `findStruct` to THIS same def rather
        // than failing, and so every pointer/alias minted from an EARLIER forward
        // declaration sees the completion below through that same shared pointer.
        // Re-defining an already-COMPLETE tag (a second body for the same name) is invalid C.
        const def = if (self.findStruct(tag.text)) |existing| blk: {
            if (existing.complete) return error.StructRedefinition;
            break :blk existing;
        } else try self.registerIncompleteStruct(tag.text, is_union);
        try self.parseStructBody(def, is_union);
        return .{ .@"struct" = def };
    }

    /// Parse a `{ field-decls }` body, the opening `{` already consumed, and complete `def`
    /// IN PLACE with the result. Shared by `parseStructOrUnion`'s tagged- and tagless-body
    /// paths, both of which have already found-or-created an incomplete `def` for the caller
    /// to complete here. Completing in place, not a fresh `arena.create`, matters for the
    /// tagged path: every earlier pointer/alias minted from a forward declaration holds this
    /// same address, and `CType.@"struct"`/`CType.eql` key off that pointer's identity, so
    /// they all observe the completion automatically.
    fn parseStructBody(self: *Parser, def: *ctype.StructDef, is_union: bool) Error!void {
        var raw: std.ArrayList(struct { name: []const u8, ty: ctype.CType, quals: ctype.Quals, bit_width: ?u32 }) = .empty;
        while (self.peek().kind != .rbrace) {
            const base = try self.parseTypeSpec();
            // A BITFIELD member is `type-spec [declarator] : const-expr`. An ANONYMOUS
            // bitfield (`int :3;` or the zero-width `int :0;`) has NO declarator at all, so
            // the colon follows the type-spec directly. Detect that first, before
            // `parseDeclarator`, which demands a name. Otherwise parse the declarator, then a
            // trailing `:` turns a normal member into a NAMED bitfield.
            if (self.peek().kind == .colon) {
                const w = try self.parseBitfieldWidth(base.ty);
                // A `__attribute__` run may trail an anonymous bitfield's width, before its
                // `;` (for example `int :3 __attribute__((__packed__));`).
                try self.skipAttributeSpecifiers();
                _ = try self.expect(.semicolon);
                try raw.append(self.arena, .{ .name = "", .ty = base.ty, .quals = base.quals, .bit_width = w });
                continue;
            }
            const d = try self.parseDeclarator(base.ty, base.quals);
            var bit_width: ?u32 = null;
            if (self.peek().kind == .colon) bit_width = try self.parseBitfieldWidth(d.ty);
            // A zero-width bitfield must be UNNAMED (C99 6.7.2.1p3). It is purely an
            // alignment break for the NEXT field, so giving it a name is meaningless and gcc
            // rejects it ("named bit-field 'a' has zero width"). Fail closed instead of
            // silently treating it as an anonymous alignment break and dropping the name.
            if (bit_width == 0 and d.name.len != 0) return error.BitfieldZeroNamed;
            // A `__attribute__` run may trail a struct field's declarator, or its bitfield
            // width, before its `;` (for example `int a __attribute__((__deprecated__));`).
            try self.skipAttributeSpecifiers();
            _ = try self.expect(.semicolon);
            try raw.append(self.arena, .{ .name = d.name, .ty = d.ty, .quals = d.quals, .bit_width = bit_width });
        }
        _ = try self.expect(.rbrace);
        // C layout: a struct lays fields out in declaration order, each at the next offset
        // aligned up to its own alignment, introducing padding as needed, with the whole
        // struct's size rounded up to its overall (widest-field) alignment at the end. A
        // union instead puts every field at offset 0, sized to the widest field, with no
        // padding between fields since there is only ever one "current" member. A field of
        // an incomplete type (for example `struct Inc { struct Inc x; }`, not a self-
        // reference through a pointer, a genuinely infinite-size field) fails closed here
        // via `alignOf`/`sizeInBytes`'s own `error.IncompleteType`.
        var fields: std.ArrayList(ctype.Field) = .empty;
        var off: u64 = 0;
        var al: u64 = 1;
        // `bit_cursor` is the next free bit, from the struct start, WHILE a run of bitfields
        // is open. It is `null` for a struct with no bitfields, so the ordinary-field path
        // below stays byte-identical.
        var bit_cursor: ?u64 = null;
        for (raw.items) |f| {
            if (f.bit_width) |w| {
                // LSB-first bitfield packing (System V / AAPCS, the 4 targets agree for
                // these cases). The storage unit is one whole object of the declaring type.
                // `tb` is that type's bit capacity, `t` its byte size, `fa` its alignment.
                const fa = try f.ty.alignOf(self.layout);
                const t = try f.ty.sizeInBytes(self.layout);
                const tb = t * 8;
                al = @max(al, fa);
                if (is_union) {
                    // A union bitfield sits at offset 0 like every other union member and
                    // reserves one whole storage unit. A `:0` in a union has nothing to align.
                    if (w != 0 and f.name.len != 0) try fields.append(self.arena, .{ .name = f.name, .ty = f.ty, .offset = 0, .quals = f.quals, .bit_width = w, .bit_offset = 0 });
                    off = @max(off, t);
                    continue;
                }
                var bc = bit_cursor orelse off * 8;
                if (w == 0) {
                    // Zero-width bitfield: no storage of its own. It forces the NEXT field to
                    // start at the next storage-unit boundary of this declaring type.
                    bc = alignUp(bc, tb);
                    bit_cursor = bc;
                    off = bc / 8;
                    continue;
                }
                // Place the field at `bc`. If it would straddle a storage-unit boundary of its
                // own type, bump it up to the next such boundary first.
                if (bc / tb != (bc + w - 1) / tb) bc = alignUp(bc, tb);
                const unit_byte = (bc / tb) * t;
                // An anonymous (unnamed) bitfield reserves its bits but is never accessed, so
                // it gets no `Field` entry. Only the cursor advances.
                if (f.name.len != 0) try fields.append(self.arena, .{ .name = f.name, .ty = f.ty, .offset = unit_byte, .quals = f.quals, .bit_width = w, .bit_offset = @intCast(bc - unit_byte * 8) });
                bc += w;
                bit_cursor = bc;
                off = (bc + 7) / 8;
                continue;
            }
            // Ordinary (non-bitfield) field. A non-bitfield after a bitfield run closes it,
            // the cursor clears, and resumes byte layout at the next aligned offset. `off`
            // already tracks the next free byte past the last bit.
            bit_cursor = null;
            const fa = try f.ty.alignOf(self.layout);
            const fo = if (is_union) 0 else alignUp(off, fa);
            try fields.append(self.arena, .{ .name = f.name, .ty = f.ty, .offset = fo, .quals = f.quals });
            const fsz = try f.ty.sizeInBytes(self.layout);
            if (is_union) {
                off = @max(off, fsz);
            } else {
                off = fo + fsz;
            }
            al = @max(al, fa);
        }
        const size = alignUp(off, al);
        def.fields = try fields.toOwnedSlice(self.arena);
        def.size = size;
        def.alignment = al;
        def.complete = true;
    }

    /// Parse a bitfield width `: const-expr`, the leading `:` still unconsumed, against its
    /// declaring type `ty`. The width is a full CONSTANT-EXPRESSION, parsed at the
    /// conditional-expression level and folded by `consteval.eval`, exactly like an array
    /// dimension. A bitfield must have an INTEGER declaring type (`error.BitfieldNonInteger`
    /// otherwise, since a float/pointer/struct bitfield is nonsense), and its width must not
    /// exceed that type's own bit count (`error.BitfieldTooWide`). Width `0` is the
    /// alignment-break form, only meaningful for an anonymous bitfield, but this helper just
    /// returns it. The layout loop is what gives `:0` its no-storage/next-unit meaning.
    fn parseBitfieldWidth(self: *Parser, ty: ctype.CType) Error!u32 {
        _ = try self.expect(.colon);
        if (!ty.isInt()) return error.BitfieldNonInteger;
        const expr = try self.parseConditional();
        const v = try consteval.eval(self.arena, expr, self.layout, null);
        if (v < 0) return error.BitfieldTooWide; // a negative width is nonsense, fail closed
        const max_bits = (try ty.sizeInBytes(self.layout)) * 8;
        if (@as(u64, @intCast(v)) > max_bits) return error.BitfieldTooWide;
        return @intCast(v);
    }

    /// Parse `enum Tag { A, B = val, C }`, a definition registering each constant in
    /// `enum_consts`, or a bare `enum Tag` reference. Unlike `struct`/`union`, an `enum` is
    /// never itself a distinct type at runtime here. It is always `int` (`ctype.int_t`), so
    /// there is no tag table to consult/validate for a bare reference, no `findStruct`
    /// equivalent needed. `enum Tag` alone is simply `int`, whether or not `Tag` was ever
    /// defined. The tag name itself, if present, is consumed and otherwise unused. Only the
    /// constants a `{ ... }` body declares matter. The caller has already consumed the
    /// `enum` keyword itself.
    fn parseEnum(self: *Parser) Error!ctype.CType {
        if (self.peek().kind == .ident) self.pos += 1; // optional tag, consumed but unused
        if (self.peek().kind != .lbrace) return ctype.int_t; // bare `enum Tag` reference
        self.pos += 1; // '{'
        // Auto-increment: the first constant with no `= val` starts at 0. Each subsequent
        // constant, with no `= val` of its own, is one more than the PREVIOUS constant's
        // value, explicit or auto, matching C (`A=5, B, C=20` -> A=5, B=6, C=20).
        var next_value: i64 = 0;
        while (true) {
            const name = try self.expect(.ident);
            if (self.peek().kind == .assign) {
                self.pos += 1;
                // An enumerator value is a full CONSTANT-EXPRESSION (C11 6.7.2.2), not just an
                // integer literal. Real headers write `A = 1 << 3`, `B = A + 1`, `(int) X`,
                // and similar expressions. A prior enumerator name folds to its value at
                // parse time (see `parseAtom`), so `consteval` sees only literals and
                // operators.
                const e = try self.parseConditional();
                next_value = try consteval.eval(self.arena, e, self.layout, null);
            }
            try self.enum_consts.put(self.arena, try self.arena.dupe(u8, name.text), next_value);
            next_value += 1;
            if (self.peek().kind == .comma) {
                self.pos += 1;
                if (self.peek().kind == .rbrace) break; // trailing comma before '}'
                continue;
            }
            break;
        }
        _ = try self.expect(.rbrace);
        return ctype.int_t;
    }

    /// Consume a run of `const`/`volatile`/`restrict` tokens, in any order or repetition,
    /// OR-ing each into `quals`. Shared by `parseTypeSpec` (leading/trailing qualifiers
    /// around a base type-spec) and `parseDeclarator`'s `*`-loop (a qualifier right after a
    /// `*`, for example `int *const`).
    ///
    /// `restrict` (and the GNU spellings `__restrict`/`__restrict__`) has no lexer keyword.
    /// It is matched here by identifier TEXT instead, because glibc headers put it on pointer
    /// params (`const char *__restrict __filename`) and VCC does no restrict-based
    /// optimization: the qualifier is consumed and its meaning is dropped. Matching by text
    /// keeps the lexer's keyword table free of a keyword that carries no semantics in VCC.
    fn parseQualRun(self: *Parser, quals: *ctype.Quals) void {
        while (true) {
            switch (self.peek().kind) {
                .kw_const => {
                    quals.is_const = true;
                    self.pos += 1;
                },
                .kw_volatile => {
                    quals.is_volatile = true;
                    self.pos += 1;
                },
                .ident => {
                    const text = self.peek().text;
                    if (std.mem.eql(u8, text, "restrict") or
                        std.mem.eql(u8, text, "__restrict") or
                        std.mem.eql(u8, text, "__restrict__"))
                    {
                        self.pos += 1;
                        continue;
                    }
                    return;
                },
                else => return,
            }
        }
    }

    /// Consume a run of zero or more GNU `__attribute__((...))` specifiers, recording
    /// nothing. VCC does no attribute-driven codegen, so an attribute is parsed only far
    /// enough to skip its body. It never changes the type, storage, or value the
    /// surrounding declaration builds. Both GNU spellings (`__attribute__` and the shorter
    /// `__attribute`) are accepted. Each must be followed by a DOUBLE parenthesis (`((`). A
    /// single `(` is a clear syntax error, not a shape this parses at all. The body between
    /// the two opening parens and their matching close may itself hold balanced parens
    /// (`__format__(__printf__, 1, 2)`), so every token is consumed by tracking paren DEPTH,
    /// starting at 2 for the two `(` already consumed, rather than stopping at the first
    /// `)`. Attributes STACK (`__attribute__((a)) __attribute__((b))`), so this loops until
    /// no `__attribute__`/`__attribute` token remains.
    fn skipAttributeSpecifiers(self: *Parser) Error!void {
        while (self.peek().kind == .ident and
            (std.mem.eql(u8, self.peek().text, "__attribute__") or std.mem.eql(u8, self.peek().text, "__attribute")))
        {
            self.pos += 1; // the __attribute__/__attribute identifier itself
            _ = try self.expect(.lparen);
            _ = try self.expect(.lparen);
            var depth: u32 = 2; // the two '(' just consumed above
            while (depth > 0) {
                switch (self.peek().kind) {
                    .lparen => depth += 1,
                    .rparen => depth -= 1,
                    .eof => return error.UnexpectedToken, // unterminated attribute body
                    else => {},
                }
                self.pos += 1;
            }
        }
    }

    /// Consume a run of zero or more trailing declarator decorations glibc emits, in ANY
    /// order and INTERLEAVED: a GNU `__asm__`/`asm` SYMBOL-RENAME label (`__asm__("name")`,
    /// for example `int p(void) __asm__("myp");`) and `__attribute__` runs
    /// (`skipAttributeSpecifiers`, for example `... __asm__("qq")
    /// __attribute__((__nothrow__))`). Both are parsed and their content DISCARDED. VCC
    /// never links a declarator's C name to a DIFFERENT ELF symbol, so an asm label's
    /// rename is silently IGNORED, not honored. The symbol keeps its plain C name, so a
    /// caller's ordinary call to that name still resolves. LIMITATION: a header that relies
    /// on `__asm__` to redirect a call to a DIFFERENTLY spelled symbol, glibc's `__REDIRECT`
    /// machinery, resolves to the ORIGINAL C name instead of the renamed target. glibc DOES
    /// emit these redirects at __GNUC__=4.2 (`__REDIRECT` is defined for `__GNUC__ >= 2`), so
    /// stdio.h's `fscanf`/`scanf`/`sscanf` isoc99 redirect labels reach this code. Ignoring
    /// the rename is harmless for a caller of a NON-redirected function such as `printf`. A
    /// caller of a redirected function resolves to the plain C name, the C89 variant, still
    /// an exported glibc symbol, rather than the isoc99 target. Honoring `__REDIRECT`
    /// renames is future work.
    fn skipAsmLabelAndAttributes(self: *Parser) Error!void {
        while (true) {
            try self.skipAttributeSpecifiers();
            if (self.peek().kind == .ident and
                (std.mem.eql(u8, self.peek().text, "__asm__") or std.mem.eql(u8, self.peek().text, "asm")))
            {
                self.pos += 1; // the asm/__asm__ identifier itself
                _ = try self.expect(.lparen);
                // The rename target is a STRING LITERAL. C adjacent-string concatenation
                // means it may be SEVERAL `str_lit` tokens in a row, not just one. glibc's
                // `__REDIRECT` spells it `__asm__(__ASMNAME(#alias))`, which expands through
                // `__STRING(__USER_LABEL_PREFIX__) #alias` to TWO adjacent literals (for
                // example `"" "__isoc99_fscanf"`). Consume the whole run, at least one, all
                // discarded. See the LIMITATION above on why the rename is ignored.
                _ = try self.expect(.str_lit);
                while (self.peek().kind == .str_lit) self.pos += 1;
                _ = try self.expect(.rparen);
                continue;
            }
            break;
        }
    }

    /// `parseTypeSpecBase` plus a leading AND trailing run of `const`/`volatile` qualifiers.
    /// `const int`, `int const`, `volatile unsigned long` all resolve the same base type
    /// with `quals` recording what was seen. Every caller below that needs a full
    /// declarator (param/struct-field/typedef/decl/global/sizeof/cast) uses THIS wrapper,
    /// not `parseTypeSpecBase` directly, so a leading qualifier is never silently dropped.
    /// Some callers (`parseDeclStmt`, the file-scope loop in `parse`) ALSO run their own
    /// leading `const`/`volatile`-accepting prefix loop, interleaved with `static`/
    /// `extern`, which are not part of a type-spec at all. By the time THEY call this, any
    /// qualifier already consumed there is gone from the token stream, so this wrapper's own
    /// leading-run finds nothing there and is a harmless no-op. A TRAILING qualifier (`int
    /// const x;`, rare but legal) is still caught here either way.
    fn parseTypeSpec(self: *Parser) Error!struct { ty: ctype.CType, quals: ctype.Quals } {
        // A `__attribute__` run may lead the whole decl-specifier sequence
        // (`__attribute__((__pure__)) int g(void)`). Skip it before anything else.
        try self.skipAttributeSpecifiers();
        var quals: ctype.Quals = .{};
        self.parseQualRun(&quals);
        const ty = try self.parseTypeSpecBase();
        self.parseQualRun(&quals);
        // A `__attribute__` run may also trail the type-spec, before the declarator. This is
        // what glibc uses right after a `struct`/`union` closing `}` (for example `struct S
        // { ... } __attribute__((packed)) v;`) and is the single funnel every caller of
        // `parseTypeSpec` (global/local/param/struct-field/typedef/sizeof/cast) gets this
        // position through for free.
        try self.skipAttributeSpecifiers();
        return .{ .ty = ty, .quals = quals };
    }

    /// The STATIC type of `e`, computed at PARSE TIME with NO lowering context. `e` is never
    /// evaluated, no IR, no side effects, only its TYPE is asked for. This is the parse-time
    /// counterpart of `lower.zig`'s `typeOf` (the exact rule `sizeof expr`'s unevaluated
    /// operand already follows there, see `Expr.sizeof_expr`'s doc comment), needed here
    /// because `typeof(expr)`/`_Generic`'s controlling expression must resolve to a concrete
    /// `CType` right here in the PARSER. A type-specifier composes with the declarator that
    /// follows it, for example `typeof(x) *p`, so it cannot wait for lowering the way
    /// `sizeof_expr` does. This is narrower than `lower.zig`'s `typeOf`: it only knows what
    /// THIS parser already tracks (`locals`, the currently in-scope params/locals,
    /// `structs`, `typedefs`), with no visibility into another function's return type or a
    /// global's declared type, since no such tables are threaded into the parser, or into a
    /// statement-expression's tail value. `.call`/`.stmt_expr`/an unresolvable `.name` all
    /// fail closed with `error.Unsupported` rather than guessing. These are untested,
    /// undocumented-in-tests gaps, not miscompiles.
    fn typeOfExpr(self: *Parser, e: *const Expr) Error!ctype.CType {
        return switch (e.*) {
            .int_lit => |v| v.ty,
            .float_lit => |v| v.ty,
            .name => |n| self.lookupLocalType(n) orelse error.Unsupported,
            .str_lit => |s| try self.wrapArray(ctype.char_t, s.len + 1, .{}),
            // `wchar_t` is a signed 32-bit `int` on every target. The element type is
            // `ctype.int_t`, not `char_t`, unlike `.str_lit` just above.
            .wstr_lit => |s| try self.wrapArray(ctype.int_t, s.len + 1, .{}),
            .negate => |inner| negt: {
                const t = try self.typeOfExpr(inner);
                if (t.isFloat()) break :negt t; // `-d` keeps the operand's float type
                if (!t.isInt()) break :negt error.Unsupported;
                break :negt t.promote();
            },
            .complement => |inner| compt: {
                const t = try self.typeOfExpr(inner);
                if (!t.isInt()) break :compt error.Unsupported;
                break :compt t.promote();
            },
            .lognot, .compare, .logand, .logor => ctype.int_t,
            .addrof => |inner| try self.wrapPtr(try self.lvalueTypeOfExpr(inner), .{}),
            .deref, .index, .member => try self.lvalueTypeOfExpr(e),
            .binary => |b| bin: {
                const lt = try self.typeOfExpr(b.lhs);
                const rt = try self.typeOfExpr(b.rhs);
                if (!(lt.isInt() or lt.isFloat()) or !(rt.isInt() or rt.isFloat())) break :bin error.Unsupported;
                break :bin if (b.op == .shl or b.op == .shr) lt.promote() else ctype.CType.commonType(lt, rt, self.layout);
            },
            .assign => |a| try self.lvalueTypeOfExpr(a.target),
            .incdec => |ie| try self.lvalueTypeOfExpr(ie.target),
            .ternary => |t| tern: {
                const tt = try self.typeOfExpr(t.then);
                const et = try self.typeOfExpr(t.els);
                if (!(tt.isInt() or tt.isFloat()) or !(et.isInt() or et.isFloat())) break :tern error.Unsupported;
                break :tern ctype.CType.commonType(tt, et, self.layout);
            },
            .sizeof_type, .sizeof_expr => ctype.ulong_t,
            // A cast's type is always its own target. See `Expr.cast`'s doc comment.
            .cast => |c| c.target,
            .cast_void => ctype.int_t,
            .va_arg => |va| va.ty,
            // A compound literal is an lvalue of its own declared type. No lowering is
            // needed to know that, same as a cast's target just above.
            .compound_literal => |cl| cl.ty,
            // A call's return type needs a whole-program function table this parser does
            // not thread (see this function's own doc comment). A statement-expression's
            // value needs full statement analysis, also not attempted here. Both fail closed.
            .call, .stmt_expr => error.Unsupported,
            // `lhs, rhs`: the comma operator's type is `rhs`'s. `lhs` is only evaluated for
            // its side effects, which this unevaluated (parse-time) context never performs
            // anyway.
            .comma => |c| try self.typeOfExpr(c.rhs),
        };
    }

    /// The parse-time counterpart of `lower.zig`'s `lvalueTypeOf`, `typeOfExpr`'s helper for
    /// the lvalue-shaped nodes (`.name`/`.str_lit`/`.deref`/`.index`/`.member`). Each is
    /// resolved the exact same way, pointee/element/field lookup, just sourced from this
    /// parser's own tables instead of a lowering environment.
    fn lvalueTypeOfExpr(self: *Parser, e: *const Expr) Error!ctype.CType {
        return switch (e.*) {
            .name => |n| self.lookupLocalType(n) orelse error.Unsupported,
            .str_lit => |s| try self.wrapArray(ctype.char_t, s.len + 1, .{}),
            .deref => |inner| blk: {
                const t = try self.typeOfExpr(inner);
                const pointee = t.pointee() orelse break :blk error.Unsupported;
                break :blk pointee.*;
            },
            .index => |ix| blk: {
                const bt = try self.typeOfExpr(ix.base);
                const elem = if (bt == .array) bt.array.elem else bt.pointee() orelse break :blk error.Unsupported;
                break :blk elem.*;
            },
            .member => |m| blk: {
                const base_ty = if (m.arrow) arrow_blk: {
                    const t = try self.typeOfExpr(m.base);
                    break :arrow_blk (t.pointee() orelse break :blk error.Unsupported).*;
                } else try self.lvalueTypeOfExpr(m.base);
                if (base_ty != .@"struct") break :blk error.Unsupported;
                for (base_ty.@"struct".fields) |f| {
                    if (std.mem.eql(u8, f.name, m.field)) break :blk f.ty;
                }
                break :blk error.Unsupported;
            },
            else => error.Unsupported,
        };
    }

    /// Parse a run of integer type-specifier keywords, order-independent, for example
    /// `unsigned long int`, `long long`, `signed char`, into a CType. Errors if no type
    /// keyword is present. Wired into decl/param/func parsing below. `struct`/`union` are
    /// handled FIRST and return directly. They bypass the integer-specifier soup below
    /// entirely, since a struct/union type-spec is never combined with `int`/`long`/and
    /// similar keywords.
    fn parseTypeSpecBase(self: *Parser) Error!ctype.CType {
        if (self.peek().kind == .kw_struct) {
            self.pos += 1;
            return self.parseStructOrUnion(false);
        }
        if (self.peek().kind == .kw_union) {
            self.pos += 1;
            return self.parseStructOrUnion(true);
        }
        if (self.peek().kind == .kw_enum) {
            self.pos += 1;
            return self.parseEnum();
        }
        // `float` (`f32`) and `double`/`long double` (both collapse to `f64`, no extended
        // precision here) are never combined with the integer-specifier soup below, so,
        // like `struct`/`union`/`enum` above, they are handled first and return directly. A
        // leading `long` before `double` (`long double`) is simply consumed and ignored. It
        // changes nothing since both spellings already resolve to `f64`.
        if (self.peek().kind == .kw_long and self.toks[self.pos + 1].kind == .kw_double) {
            self.pos += 2;
            return .{ .float = .f64 };
        }
        if (self.peek().kind == .kw_float) {
            self.pos += 1;
            return .{ .float = .f32 };
        }
        if (self.peek().kind == .kw_double) {
            self.pos += 1;
            return .{ .float = .f64 };
        }
        // C99 `_Bool`: like `float`/`double` above, never combined with the integer-specifier
        // soup below, since C forbids `unsigned _Bool` and similar forms, handled first and
        // returned directly.
        if (self.peek().kind == .kw_bool) {
            self.pos += 1;
            return ctype.bool_t;
        }
        // `void`: like `float`/`double`/`_Bool` above, never combined with the integer-
        // specifier soup below, handled first and returned directly. This is what makes
        // `void *p`, `void **pp`, and `sizeof(void)` parse as ordinary type-specifiers.
        // `void` as an EMPTY PARAMETER LIST (`f(void)`) and the `(void)e` discard idiom are
        // both special-cased ahead of any call into this function (see `parseParams` and
        // `parseAtom`'s `.lparen` arm), so they never reach here. A bare `void` object, a
        // local/field/param/return actually typed `void`, not `void*`, still parses fine as
        // a type-spec. It only fails later, in `lower.zig`, when something tries to give it
        // storage (`ctype.CType.irType` reports `error.VoidValue`).
        if (self.peek().kind == .kw_void) {
            self.pos += 1;
            return ctype.void_t;
        }
        // `typeof`/`__typeof__` (a GNU extension): a type-specifier of its own, like
        // `void`/`_Bool`/`float`/`double` above, never combined with the int-specifier soup
        // below. `typeof ( operand )` yields the OPERAND's type: either a TYPE-NAME
        // (`typeof(int)`, disambiguated the same way `sizeof`/a cast peek past their `(`,
        // via `startsTypeSpec`/a registered typedef-name), resolved through `parseTypeSpec`
        // directly, or an EXPRESSION (`typeof(x)`), resolved through `typeOfExpr`. See its
        // doc comment for why that must happen HERE, at parse time, rather than deferred like
        // `sizeof_expr`. This is an `.ident`-text check, there is no dedicated token kind for
        // it, so `parseStmt`'s own local-declaration dispatch also recognizes it by name.
        // See that site.
        if (self.peek().kind == .ident and
            (std.mem.eql(u8, self.peek().text, "typeof") or std.mem.eql(u8, self.peek().text, "__typeof__")))
        {
            self.pos += 1;
            _ = try self.expect(.lparen);
            const next = self.peek();
            if (startsTypeSpec(next.kind) or (next.kind == .ident and self.isTypeName(next.text))) {
                const base = try self.parseTypeSpec();
                _ = try self.expect(.rparen);
                return base.ty;
            }
            const operand = try self.parseComma();
            _ = try self.expect(.rparen);
            return try self.typeOfExpr(operand);
        }
        var saw_int = false;
        var saw_char = false;
        var saw_short = false;
        var longs: u8 = 0;
        var sign: ?bool = null;
        var any = false;
        while (true) {
            // `__extension__` (a GNU marker that silences pedantic warnings) may lead or sit
            // inside a type-specifier run, for example glibc's `__extension__ unsigned long
            // long int __value64;`. It contributes no type, so skip it and keep collecting.
            if (self.peek().kind == .ident and std.mem.eql(u8, self.peek().text, "__extension__")) {
                self.pos += 1;
                continue;
            }
            // `__signed__`/`__signed` are GNU aliases for `signed` (glibc `bits/types.h`
            // spells `__signed__ char __s8;`). `__unsigned__` is not a GNU keyword, so only
            // the signed aliases are handled here.
            if (self.peek().kind == .ident and
                (std.mem.eql(u8, self.peek().text, "__signed__") or std.mem.eql(u8, self.peek().text, "__signed")))
            {
                sign = true;
                any = true;
                self.pos += 1;
                continue;
            }
            switch (self.peek().kind) {
                .kw_int => {
                    saw_int = true;
                    any = true;
                    self.pos += 1;
                },
                .kw_char => {
                    saw_char = true;
                    any = true;
                    self.pos += 1;
                },
                .kw_short => {
                    saw_short = true;
                    any = true;
                    self.pos += 1;
                },
                .kw_long => {
                    longs += 1;
                    any = true;
                    self.pos += 1;
                },
                .kw_signed => {
                    sign = true;
                    any = true;
                    self.pos += 1;
                },
                .kw_unsigned => {
                    sign = false;
                    any = true;
                    self.pos += 1;
                },
                else => break,
            }
        }
        // `__builtin_va_list`: a compiler built-in type name, checked BEFORE the
        // registered-typedef lookup below. It is never itself registered in `typedefs` (see
        // `isTypeName`'s doc). `<stdarg.h>`'s `typedef __builtin_va_list va_list;` reaches
        // this same arm while parsing ITS OWN right-hand side, which is what makes `va_list`
        // resolve to the correct per-target shape once registered.
        if (!any and self.peek().kind == .ident and std.mem.eql(u8, self.peek().text, "__builtin_va_list")) {
            self.pos += 1;
            return ctype.builtinVaList(self.layout);
        }
        // The typedef-name-vs-identifier ambiguity. An `.ident` in type-specifier position,
        // with NO integer specifier seen yet (`!any`, since a typedef-name is never combined
        // with `int`/`long`/etc, so this cannot misfire mid-soup), that is a registered
        // typedef-name is that alias's type. Consume it and return directly, same as
        // `struct`/`union` above. An ordinary ident that ISN'T a registered typedef-name
        // falls straight through to the `error.UnexpectedToken` below unconsumed. This is
        // what keeps ordinary variable idents, which are never in `typedefs`, parsing as
        // expressions, not types.
        if (!any and self.peek().kind == .ident) {
            if (self.typedefs.get(self.peek().text)) |ty| {
                self.pos += 1;
                return ty;
            }
        }
        if (!any) return error.UnexpectedToken;
        const rank: ctype.Rank = if (saw_char) .char else if (saw_short) .short else if (longs >= 2) .longlong else if (longs == 1) .long else .int;
        // Default signedness: signed, except plain `char` (no `int`/`short`/`long` alongside
        // it and no explicit `signed`/`unsigned`) which follows the target's char signedness.
        const signed = sign orelse (if (saw_char and !saw_int and !saw_short and longs == 0) self.layout.char_signed else true);
        return ctype.mkInt(rank, signed);
    }

    /// The result of parsing one declarator. This was previously an anonymous struct
    /// literal, but `parseDeclarator` and `parseGroupedFuncPtrDeclarator` are now mutually
    /// recursive and must agree on the exact SAME return type. Zig gives every anonymous
    /// struct literal its own distinct nominal type, even when structurally identical, so
    /// `return self.parseGroupedFuncPtrDeclarator(...)` needs a shared name to type-check.
    /// `func_params` is non-null only for a BARE function-declarator suffix (for example
    /// `puts` in `int puts(const char *s)`), the NAMED params alongside the `.func` `ty`'s
    /// own bare-type params (see `parseFuncDeclaratorTail`). It is `null` for every other
    /// declarator.
    const DeclaratorResult = struct { name: []const u8, ty: ctype.CType, quals: ctype.Quals, func_params: ?[]Param = null };

    /// Parse a declarator after a base type: `* ... name [N] [M] ...`. Pointers wrap left.
    /// Array suffixes wrap around the name from the INNERMOST (last) dimension outward, so
    /// `int a[2][3]` becomes `array{len=2, elem=array{len=3, elem=int}}`, the same shape
    /// `a[i][j]` addressing (`lowerAddr`'s `.index` arm, scaled by the recursive
    /// `sizeInBytes`) expects: indexing the outer `[2]` first steps by a whole `int[3]`
    /// (12 bytes), then the inner `[3]` steps by one `int` (4 bytes). Returns the full CType,
    /// the name, and the declared OBJECT's own `quals`.
    ///
    /// Qualifier attribution (`base_quals` is whatever qualified the base type-spec, for
    /// example the `const` in `const int *p`): a qualifier always describes "the type of
    /// whatever is immediately to its right at that point in the declarator". Threading
    /// `pending` through the `*`-loop gives exactly that. It starts as `base_quals`,
    /// qualifying the base type, that is, what the FIRST `*` points to. Each `wrapPtr` call
    /// consumes it as THAT pointer's `.quals`, the pointee-quals, see `ctype.CType.ptr`'s
    /// doc comment, then a fresh `parseQualRun` right after the `*` becomes the quals for
    /// whatever comes next: either the NEXT `*`'s pointer, if another follows, or, once the
    /// loop ends, the declared object itself. This is why `int *const p` (pointer-object
    /// const) and `const int *p` (pointee const) attribute correctly, and why deeper chains
    /// (`int * const * p`) recurse right with no special-casing. For `const int *p`, ONE
    /// star consumes `base_quals={const}` as its `.quals` (pointee const), leaving
    /// `pending={}` for the object. For `int *const p`, the star consumes `base_quals={}`
    /// (pointee plain), then `const` is parsed into `pending={const}`, which, with no
    /// further star, becomes the object's own quals. A `[` may be immediately followed by
    /// `]`, empty, no `int_lit`, for an unsized dimension (`len == 0`). This is only
    /// meaningful as a param declarator, which decays regardless of `len` in `parseParams`.
    /// A LOCAL with any dimension `len == 0` is rejected in `lowerStmt`.
    ///
    /// This adds two RECURSIVE declarator forms, both scoped narrowly. This frontend does
    /// not implement the general abstract-declarator grammar. Anything else involving `(`
    /// in declarator position fails closed with a clear diagnostic rather than silently
    /// mis-parsing:
    ///   - a function-declarator SUFFIX, `name ( params )` (for example `int puts(const
    ///     char *s)`): detected right after the name, before array-dims, since a function
    ///     cannot return an array directly, nor be one. It delegates to
    ///     `parseFuncDeclaratorTail`, which builds the `.func` `CType` (ret is whatever `ty`
    ///     had accumulated so far, that is, any leading `*`s already applied: `int *f(int)`
    ///     returns `int*`) AND returns the NAMED params alongside it, stashed on
    ///     `func_params`, since `ctype.FuncType.params` itself is bare types only. A
    ///     definition/prototype needs the names for its body/`FuncDecl`.
    ///   - a GROUPED pointer-to-function declarator, `( * name ) ( params )` (for example
    ///     `int (*fp)(int, int)`): the ONLY parenthesized-declarator shape understood here,
    ///     checked FIRST, before the `*`-loop, since this form itself starts with `(`, not a
    ///     bare `*`. It delegates to `parseGroupedFuncPtrDeclarator`, which yields `ptr ->
    ///     func`. `func_params` stays `null` for this form: a function-POINTER variable is
    ///     never a definition, so there is no body to bind param names to.
    fn parseDeclarator(self: *Parser, base: ctype.CType, base_quals: ctype.Quals) Error!DeclaratorResult {
        if (self.peek().kind == .lparen) return self.parseGroupedFuncPtrDeclarator(base, base_quals, true);
        var ty = base;
        var pending = base_quals;
        while (self.peek().kind == .star) {
            self.pos += 1;
            ty = try self.wrapPtr(ty, pending);
            pending = .{};
            self.parseQualRun(&pending);
        }
        const name = try self.expect(.ident);
        const name_owned = try self.arena.dupe(u8, name.text);
        if (self.peek().kind == .lparen) {
            const tail = try self.parseFuncDeclaratorTail(ty);
            return .{ .name = name_owned, .ty = tail.ty, .quals = pending, .func_params = tail.named };
        }
        const had_array = self.peek().kind == .lbracket;
        ty = try self.parseArrayDims(ty, pending);
        if (had_array) pending = .{}; // fully absorbed into the array's elem quals
        return .{ .name = name_owned, .ty = ty, .quals = pending };
    }

    /// The `( params )` tail of a function declarator, given its already-resolved return type
    /// `ret`. Shared by the bare function-declarator suffix (`name(params)`,
    /// `parseDeclarator`'s main path) and the grouped pointer-to-function form
    /// (`(*name)(params)`, `parseGroupedFuncPtrDeclarator` below). Builds BOTH the `.func`
    /// `CType` (`ctype.FuncType.params` is bare types, no names. This also copies
    /// `parseParams`'s `is_variadic` onto it) and the NAMED `Param` list, which `parseParams`
    /// already parses/returns, that a definition/prototype's body/`FuncDecl` actually needs,
    /// reusing ONE parse of the param list for both. `self.pos` must be sitting AT the `(`
    /// on entry. This consumes it, same convention `parseArrayDims` follows for `[`. Always
    /// a non-void `ret`: nothing routes a `void` return type through here. `parse`'s
    /// file-scope `void name(...)` special case builds its `Func`/`FuncDecl` directly, with
    /// no `.func` `CType`/`ctype.FuncType` involved at all.
    /// A param may come back with `name.len == 0`. `parseParams` allows an ANONYMOUS
    /// parameter, for example `int, int` in `int (*fp)(int, int)`, since a function TYPE's
    /// signature binds no names at all. `ctype.FuncType.params` does not care, bare types
    /// only. A caller that DOES need every param named, an actual definition's body, is
    /// responsible for checking that itself. Nothing currently does.
    ///
    /// Does NOT clear `locals` itself, unlike an older version of this code, which cleared
    /// right before parsing params. This helper can now be reached RECURSIVELY (a callback
    /// PARAMETER that is itself a grouped pointer-to-function, for example `int reg(int
    /// (*cb) (int))`, parses `cb`'s own `(int)` tail through here too), and clearing here
    /// would wipe the OUTER param list's already-registered names mid-parse. `parse`'s
    /// top-level loop clears once per top-level item instead (see there), which is the only
    /// place a genuinely NEW function's own scope begins.
    fn parseFuncDeclaratorTail(self: *Parser, ret: ctype.CType) Error!struct { ty: ctype.CType, named: []Param } {
        self.pos += 1; // '('
        const pr = try self.parseParams();
        _ = try self.expect(.rparen);
        const ret_box = try self.arena.create(ctype.CType);
        ret_box.* = ret;
        const param_types = try self.arena.alloc(ctype.CType, pr.params.len);
        for (pr.params, 0..) |pm, i| param_types[i] = pm.ty;
        const ft = try self.arena.create(ctype.FuncType);
        ft.* = .{ .ret = ret_box, .params = param_types, .is_variadic = pr.is_variadic };
        return .{ .ty = .{ .func = ft }, .named = pr.params };
    }

    /// Grouped pointer-to-function declarator: `( * name ) ( params )`, for example
    /// `int (*fp)(int, int)`. The ONLY parenthesized/grouped declarator form this frontend
    /// understands. Anything else starting with `(` here (a plain grouped declarator
    /// `int (a);`, pointer-to-array `int (*a)[5]`, a grouped declarator with further `*`s
    /// around it, and so on) is future work and fails closed (`error.Unsupported`) rather
    /// than silently mis-parsing. `quals`, whatever qualified the base type before this
    /// declarator, passes straight through as the declared object's own quals. This narrow
    /// form has no qualifier-attribution rule of its own to apply (no test exercises
    /// `const` here). `name` itself is optional ONLY when `require_name` is false (a
    /// callback PARAMETER's own grouped-pointer-to-function form, for example `int
    /// (*)(int)` with no name at all, mirrors `parseParams`'s anonymous-parameter allowance
    /// below). The top-level declarator path (`parseDeclarator`) always passes
    /// `require_name = true`.
    fn parseGroupedFuncPtrDeclarator(self: *Parser, ret: ctype.CType, quals: ctype.Quals, require_name: bool) Error!DeclaratorResult {
        self.pos += 1; // '('
        if (self.peek().kind != .star) return error.Unsupported;
        self.pos += 1;
        var name: []const u8 = "";
        if (self.peek().kind == .ident) {
            const t = try self.expect(.ident);
            name = try self.arena.dupe(u8, t.text);
        } else if (require_name) {
            return error.Unsupported;
        }
        _ = try self.expect(.rparen);
        if (self.peek().kind != .lparen) return error.Unsupported;
        const tail = try self.parseFuncDeclaratorTail(ret);
        const ptr_ty = try self.wrapPtr(tail.ty, .{});
        return .{ .name = name, .ty = ptr_ty, .quals = quals };
    }

    /// Parse an ABSTRACT declarator, no name, after a base type. Only `sizeof(type-name)`
    /// and a cast target need this, since a type-name never binds an identifier. Same
    /// wrapping/attribution rules as `parseDeclarator`. The leftover "object" quals, since
    /// there is no object to attach them to, are simply discarded.
    fn parseAbstractDeclarator(self: *Parser, base: ctype.CType, base_quals: ctype.Quals) Error!ctype.CType {
        var ty = base;
        var pending = base_quals;
        while (self.peek().kind == .star) {
            self.pos += 1;
            ty = try self.wrapPtr(ty, pending);
            pending = .{};
            self.parseQualRun(&pending);
        }
        return self.parseArrayDims(ty, pending);
    }

    /// Collect zero or more `[N]`/`[]` array suffixes left-to-right, then wrap `ty` from the
    /// innermost (last) dimension outward. Shared by `parseDeclarator` and
    /// `parseAbstractDeclarator`. See `parseDeclarator`'s doc comment for why innermost-first
    /// is the wrap order multi-dim indexing needs. `elem_quals` qualifies the innermost
    /// element (`base`). It only ever applies to the FIRST wrap built, the one whose `.elem`
    /// is `base` itself. Outer dimensions (`int a[2][3]`'s outer `[2]`) get no qualifier of
    /// their own in this grammar.
    ///
    /// `N` is a full CONSTANT-EXPRESSION, not just a bare integer literal. It is parsed at
    /// the conditional-expression grammar level (`parseConditional`, matching C's own
    /// `constant-expression` production, which excludes assignment/comma but allows `?:`),
    /// then folded to its `i64` value by `consteval.eval`. This is exactly how `sizeof`,
    /// arithmetic, and shifts inside a dimension become available for free. A dimension
    /// that is not a compile-time constant, or one that folds to zero or negative, fails
    /// closed with `error.Unsupported`. A VARIABLE-length array, a runtime-sized dimension,
    /// is out of scope for this frontend, so it is rejected outright rather than silently
    /// accepted as some fixed size. An empty `[]`, no expression before `]`, still parses as
    /// length 0, the existing decay/incomplete-array case.
    fn parseArrayDims(self: *Parser, base: ctype.CType, elem_quals: ctype.Quals) Error!ctype.CType {
        var ty = base;
        var dims: std.ArrayList(u64) = .empty;
        while (self.peek().kind == .lbracket) {
            self.pos += 1;
            var len: u64 = 0;
            if (self.peek().kind != .rbracket) {
                const dim_expr = try self.parseConditional();
                const v = try consteval.eval(self.arena, dim_expr, self.layout, null);
                // A NEGATIVE dimension is nonsensical, and a failed static-assert idiom, so
                // fail closed. A non-constant dimension already errored inside
                // `consteval.eval` above. A runtime-length VLA is out of scope. ZERO is
                // allowed: `T a[0];` is the GNU zero-length-array / old-style flexible
                // member, for example glibc's `struct file_handle { ...; unsigned char
                // f_handle[0]; }`. It lays out like the empty `[]` form (length 0).
                if (v < 0) return error.Unsupported;
                len = @intCast(v);
            }
            _ = try self.expect(.rbracket);
            try dims.append(self.arena, len);
        }
        var d = dims.items.len;
        var innermost = true;
        while (d > 0) {
            d -= 1;
            ty = try self.wrapArray(ty, dims.items[d], if (innermost) elem_quals else .{});
            innermost = false;
        }
        return ty;
    }

    fn wrapPtr(self: *Parser, pointee: ctype.CType, quals: ctype.Quals) Error!ctype.CType {
        const p = try self.arena.create(ctype.CType);
        p.* = pointee;
        return .{ .ptr = .{ .pointee = p, .quals = quals } };
    }
    fn wrapArray(self: *Parser, elem: ctype.CType, len: u64, quals: ctype.Quals) Error!ctype.CType {
        const e = try self.arena.create(ctype.CType);
        e.* = elem;
        return .{ .array = .{ .elem = e, .len = len, .quals = quals } };
    }

    /// A PARAMETER declarator: identical grammar to `parseDeclarator` (pointers, array dims,
    /// the function-declarator suffix, the grouped pointer-to-function form), but the name
    /// is OPTIONAL. A param list is the one place C allows an anonymous declarator wherever
    /// only a TYPE matters: a bodyless prototype (`int puts(const char*);`) or a
    /// function-pointer's signature (`int (*fp)(int, int)`'s `int`s bind no name at all). A
    /// missing name comes back as `""`. An actual function DEFINITION's body referencing an
    /// unnamed param fails the same way referencing any other undeclared name would. Lowering,
    /// not parsing, catches that. It is out of scope here, and no test constructs it.
    fn parseParamDeclarator(self: *Parser, base: ctype.CType, base_quals: ctype.Quals) Error!struct { name: []const u8, ty: ctype.CType, quals: ctype.Quals } {
        if (self.peek().kind == .lparen) {
            const d = try self.parseGroupedFuncPtrDeclarator(base, base_quals, false);
            return .{ .name = d.name, .ty = d.ty, .quals = d.quals };
        }
        var ty = base;
        var pending = base_quals;
        while (self.peek().kind == .star) {
            self.pos += 1;
            ty = try self.wrapPtr(ty, pending);
            pending = .{};
            self.parseQualRun(&pending);
        }
        if (self.peek().kind != .ident) {
            // Anonymous: no name to bind. An abstract ARRAY suffix still attaches, though. A
            // real prototype can spell an unnamed array parameter (`tmpnam(char[20])`), which
            // decays to a pointer exactly like the named `char s[20]` case does (see
            // `parseParams`'s array-to-pointer rewrite). A function-declarator suffix DOES
            // need a name to bind and never appears abstractly here, so only array dims are
            // consumed. `parseArrayDims` is a no-op when the next token is not `[`, so the
            // plain anonymous `int`/`int *` case is unchanged.
            const had_array = self.peek().kind == .lbracket;
            ty = try self.parseArrayDims(ty, pending);
            if (had_array) pending = .{};
            return .{ .name = "", .ty = ty, .quals = pending };
        }
        const name = try self.expect(.ident);
        const name_owned = try self.arena.dupe(u8, name.text);
        if (self.peek().kind == .lparen) {
            const tail = try self.parseFuncDeclaratorTail(ty);
            return .{ .name = name_owned, .ty = tail.ty, .quals = pending };
        }
        const had_array = self.peek().kind == .lbracket;
        ty = try self.parseArrayDims(ty, pending);
        if (had_array) pending = .{};
        return .{ .name = name_owned, .ty = ty, .quals = pending };
    }

    /// params := 'void' | (type-spec declarator) (',' type-spec declarator)*, each declarator
    /// per `parseParamDeclarator`, name optional. C's array-parameter decay: `int p[N]`,
    /// parsed as an array declarator, same as a local, is rewritten here to `int *p`, a
    /// pointer to the array's element type. A parameter never actually receives an array,
    /// only a pointer to its first element, so the array `CType` would be meaningless, and
    /// `sizeInBytes` wrong, if left as-is on the binding. Reuses the element pointer the
    /// declarator's `wrapArray` already allocated, no fresh allocation needed. A bare
    /// function-typed parameter (`d.ty == .func`, no pointer, for example `void reg(int
    /// f(int))`, which real C decays to a function pointer) is OUT OF SCOPE and fails closed
    /// (`error.Unsupported`) rather than silently mis-typing the param. The pointer FORM
    /// (`void reg(int (*f)(int))`) already works, since that is the grouped-pointer-to-
    /// function declarator `parseParamDeclarator` delegates to above.
    /// `...` may appear only right after at least one named parameter and only as the last
    /// thing in the list (`int f(int a, ...)`, never `int h(...)` or `int g(int a, ..., int
    /// b)`). Both fail closed with `error.Unsupported` rather than silently mis-parsing.
    fn parseParams(self: *Parser) Error!struct { params: []Param, is_variadic: bool } {
        // `(void)`, an EMPTY parameter list, the C idiom for "takes no arguments", is only
        // this when `void` is the WHOLE list, immediately followed by `)`. Peeking one token
        // further is what stops a leading `void *p`/`void x` (legal type-specifiers, see
        // `parseTypeSpecBase`) from being misread as this empty-list marker. `void` could
        // never start a real parameter type before, so checking `kw_void` alone was
        // unambiguous. Now it isn't.
        if (self.peek().kind == .kw_void and self.toks[self.pos + 1].kind == .rparen) {
            self.pos += 1;
            return .{ .params = &.{}, .is_variadic = false };
        }
        // `f()` with EMPTY parentheses is a K&R unprototyped declarator (C11 6.7.6.3p14).
        // Autoconf's conftest uses `int main () { }`. VCC treats it as "takes no arguments",
        // the same as `(void)`, which is correct for a definition and adequate for the hello
        // build. A real argument mismatch through such a prototype is not diagnosed.
        if (self.peek().kind == .rparen) {
            return .{ .params = &.{}, .is_variadic = false };
        }
        if (self.peek().kind == .ellipsis) return error.Unsupported; // `...` needs >= 1 named param first
        var list: std.ArrayList(Param) = .empty;
        var is_variadic = false;
        while (true) {
            const base = try self.parseTypeSpec();
            const d = try self.parseParamDeclarator(base.ty, base.quals);
            // A `__attribute__` run may trail a parameter's declarator, before its `,`/`)`
            // (for example `int f(int x __attribute__((__unused__)))`).
            try self.skipAttributeSpecifiers();
            if (d.ty == .func) return error.Unsupported;
            const ty: ctype.CType = if (d.ty == .array) .{ .ptr = .{ .pointee = d.ty.array.elem, .quals = d.ty.array.quals } } else d.ty;
            try list.append(self.arena, .{ .name = d.name, .ty = ty, .quals = d.quals });
            // Register the param name so `parseAtom` knows it shadows any `enum` constant of
            // the same name for the rest of this function. Skipped for an anonymous param,
            // since there is nothing to shadow with an empty name. Bound at FUNCTION scope,
            // no `pushScope` yet open here, so it stays visible for the whole body (see
            // `locals`'s doc comment).
            if (d.name.len != 0) {
                try self.declareLocal(d.name, ty);
            }
            if (self.peek().kind == .comma) {
                self.pos += 1;
                if (self.peek().kind == .ellipsis) {
                    self.pos += 1;
                    is_variadic = true;
                    if (self.peek().kind != .rparen) return error.Unsupported; // `...` must be last
                    break;
                }
                continue;
            }
            break;
        }
        return .{ .params = try list.toOwnedSlice(self.arena), .is_variadic = is_variadic };
    }

    /// A `{ ... }` compound body, or a single statement, as a statement list. The `{ ... }`
    /// case opens its own scope, see `locals`'s doc comment: every caller of this function
    /// (`if`/`while`/`do`/`for`'s body, and a GNU statement-expression `({ ... })`) gets
    /// correct block scoping for free. A bare single statement, no `{ }`, needs no scope of
    /// its own, since C does not allow a declaration there anyway.
    fn parseBlockOrStmt(self: *Parser) Error![]Stmt {
        if (self.peek().kind == .lbrace) {
            self.pos += 1;
            const mark = self.pushScope();
            defer self.popScope(mark);
            var stmts: std.ArrayList(Stmt) = .empty;
            while (self.peek().kind != .rbrace) try stmts.append(self.arena, try self.parseStmt());
            _ = try self.expect(.rbrace);
            return stmts.toOwnedSlice(self.arena);
        }
        const one = try self.arena.alloc(Stmt, 1);
        one[0] = try self.parseStmt();
        return one;
    }

    /// The body of a typed local declaration (`('static'|'extern'|'const')* type-spec
    /// declarator ('=' expr)? ';'`), shared by `parseStmt`'s keyword-led case below and its
    /// typedef-name-led case (an `.ident` that is a registered typedef, for example `u64 x =
    /// a;` / `Point p;`, see `parseStmt`). A leading run of storage-class/qualifier
    /// keywords, order-independent, mirrors the file-scope loop in `parse` below: `static`
    /// (lowering binds the local to a persistent data object instead of a stack alloca),
    /// `extern` (parsed but not specially lowered, deferred, no test needs it), and `const`
    /// (drives `.rodata` routing for a `static` local's initializer, same rule as a
    /// file-scope global).
    fn parseDeclStmt(self: *Parser) Error!Stmt {
        var storage: StorageClass = .none;
        var is_const = false;
        // `lead_quals` mirrors `is_const` (`const`) but also tracks `volatile`, and is
        // threaded into `parseDeclarator` as its `base_quals`. So, unlike `is_const` (kept
        // exactly as before, purely for `.rodata` routing), a leading qualifier here
        // correctly attributes to the POINTEE when the declarator turns out to have a `*`
        // (`const int *p`), not the declared object itself. See `parseDeclarator`'s doc
        // comment.
        var lead_quals: ctype.Quals = .{};
        while (true) {
            switch (self.peek().kind) {
                .kw_static => {
                    storage = .static;
                    self.pos += 1;
                },
                .kw_extern => {
                    storage = .extern_;
                    self.pos += 1;
                },
                .kw_const => {
                    is_const = true;
                    lead_quals.is_const = true;
                    self.pos += 1;
                },
                .kw_volatile => {
                    lead_quals.is_volatile = true;
                    self.pos += 1;
                },
                // `inline`/`_Noreturn`/`__extension__` compose with the other specifiers here
                // too, mirroring the file-scope run, see its own comment. A local declaration
                // has no genuine use for `inline`/`_Noreturn` (a LOCAL function declarator is
                // already rejected below), but recognizing them here costs nothing and keeps
                // this loop's shape identical to the file-scope one.
                .ident => if (isNoOpSpecifierIdent(self.peek().text)) {
                    self.pos += 1;
                } else break,
                else => break,
            }
        }
        const base = try self.parseTypeSpec();
        // A TYPE-ONLY declaration with no declarator: `enum { K = ... };` (its constants are
        // the point) or `struct T { ... };` (a tag definition). `parseTypeSpec` already
        // registered the enum constants / struct tag, so a `;` right here means the statement
        // is complete and introduces no object. gnulib writes a local `enum { DEFAULT_MXFAST =
        // ... };` inside a function body.
        if (self.peek().kind == .semicolon) {
            self.pos += 1;
            return .{ .block = &.{} };
        }
        // A declaration may list several declarators sharing the base type, separated by
        // `,`: `unsigned char c1, c2;`, `int *p, q;` (only `p` is a pointer, each declarator
        // applies its own `*`/`[]` to `base`). Each becomes its own `.decl`. One declarator
        // returns that `.decl` directly, more than one returns a `.decl_group`, which opens
        // no scope, so every name stays visible afterward.
        var decls: std.ArrayList(Stmt) = .empty;
        while (true) {
            const d = try self.parseDeclarator(base.ty, .{
                .is_const = lead_quals.is_const or base.quals.is_const,
                .is_volatile = lead_quals.is_volatile or base.quals.is_volatile,
            });
            // A LOCAL function declarator (`int f(int);` inside a function body, a
            // block-scope prototype, real but obscure C) is out of scope. `Stmt.decl` has no
            // shape for a bodyless declaration. Fails closed rather than building a
            // nonsensical `.decl` a later lowering pass would choke on.
            if (d.ty == .func) return error.Unsupported;
            // An `__attribute__` run or an `__asm__`/`asm` rename label may trail a
            // declarator, before its `=`/`,`/`;`. Skip it here.
            try self.skipAsmLabelAndAttributes();
            var init_iz: ?*Initializer = null;
            if (self.peek().kind == .assign) {
                self.pos += 1;
                const iz = try self.arena.create(Initializer);
                iz.* = try self.parseInitializerValue();
                init_iz = iz;
            }
            // Resolve an unsized array's length from its initializer BEFORE `lowerStmt`'s
            // `hasUnsizedDim` check ever sees it.
            const ty = resolveUnsizedArrayLen(d.ty, init_iz);
            // Register the local's name and type so `parseAtom` knows it shadows an `enum`
            // constant of the same name, and `typeof`/`_Generic` can resolve on it, for the
            // rest of its scope. After `resolveUnsizedArrayLen`, so a later
            // `typeof(this_local)` sees the fixed-up length.
            try self.declareLocal(d.name, ty);
            try decls.append(self.arena, .{ .decl = .{ .name = d.name, .init = init_iz, .ty = ty, .storage = storage, .is_const = is_const, .quals = d.quals } });
            if (self.peek().kind == .comma) {
                self.pos += 1;
                continue;
            }
            break;
        }
        _ = try self.expect(.semicolon);
        if (decls.items.len == 1) return decls.items[0];
        return .{ .decl_group = try decls.toOwnedSlice(self.arena) };
    }

    /// stmt := 'int' ident '=' expr ';' | 'return' expr ';' | '{' stmt* '}'
    ///       | 'if' '(' expr ')' block-or-stmt ('else' block-or-stmt)?
    ///       | 'while' '(' expr ')' block-or-stmt
    ///       | 'for' '(' (decl | expr ';' | ';') expr? ';' expr? ')' block-or-stmt
    ///       | 'switch' '(' expr ')' '{' ('case' int-const ':' | 'default' ':') stmt* '}'
    ///       | 'break' ';' | 'continue' ';' | expr ';'
    /// `_Static_assert ( constant-expression [ , string-literal ] ) ;` (C11 6.7.10). Returns
    /// true when it consumed one, the current token was `_Static_assert` followed by `(`,
    /// false otherwise so the caller falls through unchanged. The condition is folded now. A
    /// zero condition is a compile error, the assertion failed. The optional diagnostic
    /// message is skipped by paren depth, since VCC has no use for its text. Shared by the
    /// file-scope loop and `parseStmt`.
    fn tryStaticAssert(self: *Parser) Error!bool {
        if (!(self.peek().kind == .ident and std.mem.eql(u8, self.peek().text, "_Static_assert") and
            self.toks[self.pos + 1].kind == .lparen)) return false;
        self.pos += 2; // `_Static_assert` and its `(`
        const cond = try self.parseConditional();
        const v = try consteval.eval(self.arena, cond, self.layout, null);
        if (self.peek().kind == .comma) {
            self.pos += 1;
            // Skip the message argument up to the matching `)`, without consuming that `)`.
            var depth: u32 = 0;
            while (!(depth == 0 and self.peek().kind == .rparen)) {
                switch (self.peek().kind) {
                    .lparen => depth += 1,
                    .rparen => depth -= 1,
                    .eof => return error.UnexpectedToken, // unterminated _Static_assert
                    else => {},
                }
                self.pos += 1;
            }
        }
        _ = try self.expect(.rparen);
        _ = try self.expect(.semicolon);
        if (v == 0) return error.Unsupported; // static assertion failed
        return true;
    }

    /// Skip a run of `_Pragma ( "..." )` operators (C99 6.10.9). `_Pragma` is the operator
    /// form of `#pragma`. VCC honors no pragmas, so each is consumed and dropped, the same as
    /// a `#pragma` line. gcc's own headers wrap code in `_Pragma("GCC diagnostic push")` /
    /// `_Pragma("GCC diagnostic pop")` pairs at declaration and statement boundaries.
    fn skipPragmas(self: *Parser) void {
        while (self.peek().kind == .ident and std.mem.eql(u8, self.peek().text, "_Pragma") and
            self.toks[self.pos + 1].kind == .lparen)
        {
            self.pos += 2; // `_Pragma` and its `(`
            var depth: u32 = 1;
            while (depth > 0) {
                switch (self.peek().kind) {
                    .lparen => depth += 1,
                    .rparen => depth -= 1,
                    .eof => return,
                    else => {},
                }
                self.pos += 1;
            }
        }
    }

    fn parseStmt(self: *Parser) Error!Stmt {
        // `__extension__` may lead a LOCAL declaration statement too, mirroring the
        // file-scope leading skip (see that site's comment), ahead of everything below,
        // including the typedef-name lookahead just below (a typedef-name is still whatever
        // token follows `__extension__`, not `__extension__` itself).
        while (self.peek().kind == .ident and std.mem.eql(u8, self.peek().text, "__extension__")) self.pos += 1;
        self.skipPragmas(); // `_Pragma("...")` between statements is a no-op.
        // `_Static_assert (expr, "msg");` as a block-scope statement (C11 6.7.10). Folded
        // and discarded. It produces no code, so an empty block is its whole effect.
        if (try self.tryStaticAssert()) return .{ .block = &.{} };
        // A statement-level `asm(...)`/`__asm__(...)` (GNU inline assembly) is parsed and its
        // ENTIRE body discarded. Checked here, ahead of the typedef-name lookahead just
        // below (neither `asm` nor `__asm__` is ever a registered typedef-name, but checking
        // first keeps this independent of that heuristic). VCC emits no inline assembly at
        // all, so this is a pure "consume the balanced group, do nothing" skip. An empty
        // block is this statement's whole observable effect, none. Every token inside the
        // outer `(...)` is skipped by PAREN DEPTH, same technique as
        // `skipAttributeSpecifiers`, so nested parens inside the asm text (rare, but legal
        // once operand constraints like `"r"(x)` are involved) do not stop it early.
        if (self.peek().kind == .ident and
            (std.mem.eql(u8, self.peek().text, "asm") or std.mem.eql(u8, self.peek().text, "__asm__")) and
            self.toks[self.pos + 1].kind == .lparen)
        {
            self.pos += 2; // the asm/__asm__ identifier and its opening '('
            var depth: u32 = 1;
            while (depth > 0) {
                switch (self.peek().kind) {
                    .lparen => depth += 1,
                    .rparen => depth -= 1,
                    .eof => return error.UnexpectedToken, // unterminated asm statement
                    else => {},
                }
                self.pos += 1;
            }
            _ = try self.expect(.semicolon);
            return .{ .block = &.{} };
        }
        // A typedef-name in declaration position (`u64 x = a;`, `Point p;`) never shows up as
        // one of the keyword kinds the switch below matches. It is an `.ident` token, so it
        // is checked here FIRST, ahead of the switch. Only fires when the ident is a
        // REGISTERED typedef-name (`isTypeName`). An ordinary variable ident, never
        // registered, falls through to the switch's `else`, a plain expression statement.
        // This is what keeps `x = 1;`-style statements over ordinary names parsing
        // unchanged.
        //
        // A REGISTERED typedef-name is not enough by itself. C lets a parameter/local shadow
        // a file-scope typedef in its own scope (`int f(int myint) { myint = 5; ... }`), and
        // this parser has no scope stack (flat file-scope table) to tell "typedef-name" from
        // "shadowing variable of the same name" apart. Instead of a full scope stack, use a
        // one-token-further lookahead heuristic: a typedef-name only STARTS a declaration if
        // the token after it also looks like the start of a declarator, `.ident` (the
        // declared name, `myint x;`) or `.star` (a pointer declarator, `myint *p;`).
        // Anything else after it (`=`, `;`, an operator, `(` as a call, and so on) means the
        // typedef-name is being used as a shadowing variable in an expression statement, so
        // fall through to the switch's expression-statement path. `self.toks` is
        // eof-terminated, so `self.pos + 1` is always in bounds here. A statement-leading
        // token is never the last one in the stream.
        //
        // KNOWN LIMITATION (like the `sizeof(u64)` gap above): `myint * p;` where `myint` is
        // shadowed by a variable is the classic C typedef/expression ambiguity. `.star` next
        // still reads as "pointer declarator" here, so it is parsed as a declaration
        // (`myint` declaring a pointer named `p`) rather than the multiplication expression
        // `myint * p`. Resolving this genuinely needs a scope stack, to know whether `myint`
        // is shadowed at this point, which is future work. It is untested and undocumented
        // in tests.
        // A LABEL (`name:`) is an `.ident` followed directly by `:`. Checked ahead of the
        // typedef-name lookahead just below, since a label name is never itself a
        // declarator. `case`/`default` labels use `.kw_case`/`.kw_default`, not a bare
        // `.ident`, so they never reach this check.
        if (self.peek().kind == .ident and self.toks[self.pos + 1].kind == .colon) {
            const name = try self.arena.dupe(u8, self.peek().text);
            self.pos += 2; // the ident and its ':'
            const body = try self.arena.create(Stmt);
            body.* = try self.parseStmt();
            return .{ .label = .{ .name = name, .body = body } };
        }
        if (self.peek().kind == .ident and self.isTypeName(self.peek().text) and
            (self.toks[self.pos + 1].kind == .ident or self.toks[self.pos + 1].kind == .star))
        {
            return self.parseDeclStmt();
        }
        // A statement led by `register`/`auto` (storage-class specifiers, lexed as plain
        // identifiers here) is a local declaration. Route it to `parseDeclStmt`, whose storage
        // loop then skips the specifier (see `isNoOpSpecifierIdent`). The label form
        // (`register:`) is already handled just above, so a following `:` never reaches here.
        if (self.peek().kind == .ident and
            (std.mem.eql(u8, self.peek().text, "register") or std.mem.eql(u8, self.peek().text, "auto")))
        {
            return self.parseDeclStmt();
        }
        // A LOCAL `typedef <type> <name>;` inside a function body (gnulib's memchr.c writes
        // `typedef unsigned long int longword;`). Mirrors the file-scope typedef handling.
        // The alias is registered in the shared `typedefs` table. VCC keeps no block-scoped
        // typedef stack, so a local typedef stays visible past its block, which real code
        // never relies on. It emits no code, so its statement is an empty block.
        if (self.peek().kind == .kw_typedef) {
            self.pos += 1;
            const base = try self.parseTypeSpec();
            const d = try self.parseDeclarator(base.ty, base.quals);
            try self.skipAsmLabelAndAttributes();
            _ = try self.expect(.semicolon);
            try self.typedefs.put(self.arena, d.name, d.ty);
            return .{ .block = &.{} };
        }
        // `typeof(...)`/`__typeof__(...)` may ALSO lead a local declaration (`typeof(x) y =
        // x;`). Like a typedef-name just above, it is an `.ident` token, so it never reaches
        // the KEYWORD-kind switch below. Unlike a typedef-name, `typeof`'s syntax is always
        // `typeof (`. There is no bare-name form to disambiguate from a shadowing variable,
        // so a `(` right after is enough to recognize it.
        if (self.peek().kind == .ident and
            (std.mem.eql(u8, self.peek().text, "typeof") or std.mem.eql(u8, self.peek().text, "__typeof__")) and
            self.toks[self.pos + 1].kind == .lparen)
        {
            return self.parseDeclStmt();
        }
        switch (self.peek().kind) {
            // `.kw_void` joins this list so `void *p = &x;` dispatches to `parseDeclStmt`
            // like every other type-specifier keyword. `void` could never start a LOCAL
            // declaration before, no CType modeled it, so it had no reason to be here. Now
            // it can (`void*`, `void**`).
            .kw_int, .kw_char, .kw_short, .kw_long, .kw_signed, .kw_unsigned, .kw_struct, .kw_union, .kw_enum, .kw_static, .kw_extern, .kw_const, .kw_volatile, .kw_float, .kw_double, .kw_bool, .kw_void => return self.parseDeclStmt(),
            .kw_return => {
                self.pos += 1;
                // `return;` with no operand: valid C, distinct from `return <expr>;`. A `;`
                // right after `return` means "no value".
                if (self.peek().kind == .semicolon) {
                    self.pos += 1;
                    return .{ .ret = null };
                }
                const e = try self.parseComma();
                _ = try self.expect(.semicolon);
                return .{ .ret = e };
            },
            .lbrace => {
                self.pos += 1;
                // A nested `{ ... }` used directly as a statement, not through
                // `parseBlockOrStmt`, for example a bare block inside a function body, opens
                // its own scope too. See `locals`'s doc comment.
                const mark = self.pushScope();
                defer self.popScope(mark);
                var stmts: std.ArrayList(Stmt) = .empty;
                while (self.peek().kind != .rbrace) try stmts.append(self.arena, try self.parseStmt());
                _ = try self.expect(.rbrace);
                return .{ .block = try stmts.toOwnedSlice(self.arena) };
            },
            .kw_if => {
                self.pos += 1;
                _ = try self.expect(.lparen);
                const cond = try self.parseComma();
                _ = try self.expect(.rparen);
                const then_body = try self.parseBlockOrStmt();
                var els: []Stmt = &.{};
                if (self.peek().kind == .kw_else) {
                    self.pos += 1;
                    els = try self.parseBlockOrStmt();
                }
                return .{ .if_ = .{ .cond = cond, .then = then_body, .els = els } };
            },
            .kw_while => {
                self.pos += 1;
                _ = try self.expect(.lparen);
                const cond = try self.parseComma();
                _ = try self.expect(.rparen);
                const body = try self.parseBlockOrStmt();
                return .{ .while_ = .{ .cond = cond, .body = body } };
            },
            .kw_do => {
                self.pos += 1;
                const body = try self.parseBlockOrStmt();
                _ = try self.expect(.kw_while);
                _ = try self.expect(.lparen);
                const cond = try self.parseComma();
                _ = try self.expect(.rparen);
                _ = try self.expect(.semicolon);
                return .{ .do_ = .{ .body = body, .cond = cond } };
            },
            .kw_for => {
                self.pos += 1;
                _ = try self.expect(.lparen);
                // A `for` loop opens its OWN scope for its init declaration (`for (int i =
                // 0; ...)`), enclosing the condition/increment/body. See `locals`'s doc
                // comment. The body, if braced, gets its own NESTED scope from
                // `parseBlockOrStmt` below, same as any other `{ ... }`.
                const mark = self.pushScope();
                defer self.popScope(mark);
                var init_s: ?*Stmt = null;
                if (self.peek().kind != .semicolon) {
                    const s = try self.arena.create(Stmt);
                    s.* = try self.parseStmt(); // parseStmt consumes the trailing ';' for decl/expr
                    init_s = s;
                } else {
                    self.pos += 1; // empty init ';'
                }
                var cond: ?*Expr = null;
                if (self.peek().kind != .semicolon) cond = try self.parseComma();
                _ = try self.expect(.semicolon);
                var incr: ?*Expr = null;
                if (self.peek().kind != .rparen) incr = try self.parseComma();
                _ = try self.expect(.rparen);
                const body = try self.parseBlockOrStmt();
                return .{ .for_ = .{ .init = init_s, .cond = cond, .incr = incr, .body = body } };
            },
            .kw_switch => {
                self.pos += 1;
                _ = try self.expect(.lparen);
                const value = try self.parseComma();
                _ = try self.expect(.rparen);
                _ = try self.expect(.lbrace);
                // A `switch` body is a single block scope shared by every `case`/`default`
                // arm, like an ordinary `{ ... }`. See `locals`'s doc comment.
                const mark = self.pushScope();
                defer self.popScope(mark);
                var cases: std.ArrayList(Case) = .empty;
                while (self.peek().kind != .rbrace) {
                    var label: ?i64 = null;
                    if (self.peek().kind == .kw_case) {
                        self.pos += 1;
                        // A `case` label is a constant-EXPRESSION (C11 6.8.4.2), folded the same
                        // way as an enumerator value: `case 'a':`, `case '0' + 1:`, `case
                        // (int) X:` all reduce to a compile-time integer.
                        const e = try self.parseConditional();
                        label = consteval.eval(self.arena, e, self.layout, null) catch return error.UnexpectedToken;
                        _ = try self.expect(.colon);
                    } else {
                        _ = try self.expect(.kw_default);
                        _ = try self.expect(.colon);
                    }
                    var body: std.ArrayList(Stmt) = .empty;
                    while (self.peek().kind != .kw_case and self.peek().kind != .kw_default and self.peek().kind != .rbrace)
                        try body.append(self.arena, try self.parseStmt());
                    try cases.append(self.arena, .{ .label = label, .body = try body.toOwnedSlice(self.arena) });
                }
                _ = try self.expect(.rbrace);
                return .{ .switch_ = .{ .value = value, .cases = try cases.toOwnedSlice(self.arena) } };
            },
            // A bare `;` is a NULL STATEMENT (C11 6.8.3p3). It does nothing. Autoconf's
            // conftest bodies and many real programs use it. An empty block `{ }` lowers to
            // no code, the same effect.
            .semicolon => {
                self.pos += 1;
                return .{ .block = &.{} };
            },
            .kw_break => {
                self.pos += 1;
                _ = try self.expect(.semicolon);
                return .break_;
            },
            .kw_continue => {
                self.pos += 1;
                _ = try self.expect(.semicolon);
                return .continue_;
            },
            .kw_goto => {
                self.pos += 1;
                // Computed goto (`goto *expr;`) is out of scope. Fail closed rather than
                // mis-parse `*expr` as a label name.
                if (self.peek().kind == .star) return error.Unsupported;
                const name = try self.expect(.ident);
                _ = try self.expect(.semicolon);
                return .{ .goto_ = try self.arena.dupe(u8, name.text) };
            },
            else => {
                const e = try self.parseComma();
                _ = try self.expect(.semicolon);
                return .{ .expr = e };
            },
        }
    }

    /// func := type-spec ident '(' params ')' '{' stmt* '}'
    fn parseFunc(self: *Parser) Error!Func {
        const ret = try self.parseTypeSpec();
        return self.parseFuncAfterType(ret.ty);
    }

    /// A global declarator's optional `= <initializer>`, both scalar and aggregate:
    /// delegates to `parseInitializerValue` for the actual shape. Returns `null`, no `=`
    /// present, for a plain `int g;`.
    fn parseGlobalInit(self: *Parser) Error!?*Initializer {
        // A `__attribute__` run may trail a global declarator, before `=`/`;`/`,` (for
        // example `int y __attribute__((__unused__)) = 7;`). Skip it here, ahead of the `=`
        // check. This is the single funnel BOTH the first declarator and every comma-list
        // declarator (`parse`'s top-level loop calls this for each) route through. This also
        // skips an `__asm__`/`asm` label, interleaved, either order.
        try self.skipAsmLabelAndAttributes();
        if (self.peek().kind != .assign) return null;
        self.pos += 1;
        const iz = try self.arena.create(Initializer);
        iz.* = try self.parseInitializerValue();
        return iz;
    }

    /// initializer := '{' designated-init (',' designated-init)* ','? '}' | assignment-expr
    /// Parse one initializer: a brace-enclosed, comma-separated (trailing comma allowed)
    /// list of nested initializers, recursing for nested aggregates, for example
    /// `{{1,2},{3,4}}`, or, with no `{` at all, a bare assignment-expression. There is no
    /// comma-as-operator confusion: a top-level `,` inside braces separates elements, and
    /// outside braces separates declarators, neither is the comma operator here. Each
    /// element of a brace-list may lead with a C99 DESIGNATOR chain.
    /// `parseDesignatedInitializer` handles that. A plain element, no `.`/`[` leading it,
    /// falls straight through to this same function.
    fn parseInitializerValue(self: *Parser) Error!Initializer {
        if (self.peek().kind == .lbrace) {
            self.pos += 1;
            var list: std.ArrayList(Initializer) = .empty;
            while (self.peek().kind != .rbrace) {
                try list.append(self.arena, try self.parseDesignatedInitializer());
                if (self.peek().kind == .comma) {
                    self.pos += 1;
                    if (self.peek().kind == .rbrace) break; // trailing comma before '}'
                    continue;
                }
                break;
            }
            _ = try self.expect(.rbrace);
            return .{ .value = .{ .list = try list.toOwnedSlice(self.arena) } };
        }
        return .{ .value = .{ .expr = try self.parseExpr() } };
    }

    /// designated-init := designator+ '=' initializer | initializer
    /// designator := '.' ident | '[' const-expr ']'
    /// One element of a brace-list: an optional CHAIN of designators (`.a.b`, `[i].f`,
    /// `[i][j]`, most-outer step first), then `=`, then the value. A chain SETS the
    /// position, field or element, that value targets, honored by `consteval.evalInit` and
    /// `lower.lowerAggregateInit`. With no leading `.`/`[` at all, this is a plain positional
    /// element, `parseInitializerValue`. An array-index designator's bracketed expression is
    /// a compile-time constant, folded here (`consteval.eval`) exactly like an array
    /// declarator's own dimension (`parseArrayDims`), never a runtime-computed index.
    fn parseDesignatedInitializer(self: *Parser) Error!Initializer {
        var chain: std.ArrayList(Designator) = .empty;
        while (true) {
            if (self.peek().kind == .dot) {
                self.pos += 1;
                const nm = try self.expect(.ident);
                try chain.append(self.arena, .{ .field = try self.arena.dupe(u8, nm.text) });
            } else if (self.peek().kind == .lbracket) {
                self.pos += 1;
                const idx_expr = try self.parseConditional();
                const v = try consteval.eval(self.arena, idx_expr, self.layout, null);
                if (v < 0) return error.Unsupported; // a negative index isn't valid C
                _ = try self.expect(.rbracket);
                try chain.append(self.arena, .{ .index = @intCast(v) });
            } else break;
        }
        if (chain.items.len == 0) return self.parseInitializerValue();
        _ = try self.expect(.assign);
        var iz = try self.parseInitializerValue();
        iz.designators = try chain.toOwnedSlice(self.arena);
        return iz;
    }

    /// The rest of `parseFunc`, given an already-parsed return type. It parses the name
    /// itself, no declarator, so no pointer-returning functions, then delegates to
    /// `parseFuncAfterDeclarator` for the params/body. Kept for `parseFunc`'s narrower
    /// interface. The top-level `parse` loop instead reuses `parseDeclarator` for the name,
    /// which also gives it pointer-return support for free, and calls
    /// `parseFuncAfterDeclarator` directly.
    fn parseFuncAfterType(self: *Parser, ret: ctype.CType) Error!Func {
        const name = try self.expect(.ident);
        return self.parseFuncAfterDeclarator(ret, try self.arena.dupe(u8, name.text));
    }

    /// A function definition's params/body, given its already-resolved return type and name
    /// (from a declarator or a bare `ident`, see the two callers above/below). Split out so
    /// the top-level `parse` loop can parse a top-level item's type-spec + declarator ONCE
    /// and only THEN decide whether what follows (`(`) is a function, or a global variable
    /// declaration with no `(` at all.
    fn parseFuncAfterDeclarator(self: *Parser, ret: ctype.CType, name: []const u8) Error!Func {
        // `locals` tracks declared vars/params for THIS function only, so an enum constant
        // can be shadowed. Clear it before parsing this function's params/body so a
        // previous function's locals do not leak into this one.
        self.locals.clearRetainingCapacity();
        self.cur_func_name = name; // for `__func__` inside the body.
        _ = try self.expect(.lparen);
        const pr = try self.parseParams();
        _ = try self.expect(.rparen);
        _ = try self.expect(.lbrace);
        var body: std.ArrayList(Stmt) = .empty;
        while (self.peek().kind != .rbrace) try body.append(self.arena, try self.parseStmt());
        _ = try self.expect(.rbrace);
        return .{ .name = name, .params = pr.params, .body = try body.toOwnedSlice(self.arena), .ret = ret, .is_variadic = pr.is_variadic, .is_static = self.saw_inline };
    }

    /// Finish a function declarator once its name and already-parsed params are known. A
    /// full `{ ... }` body makes it a DEFINITION (`Func`, appended to `funcs`). A bare `;`
    /// makes it a bodyless DECLARATION (`FuncDecl`, appended to `func_decls`, carrying
    /// `storage` so `extern int f(int);` and `int f(int);` are told apart). Shared by
    /// `parse`'s file-scope `void name(...)` special case (`ret = null`, `storage` always
    /// `.none`, see that branch's own comment for why) and its general (typed-return) case
    /// below. `ret == null` for a definition falls back to `ctype.int_t`, the exact same
    /// invisible placeholder the OLD `void`-function code path used directly. A void
    /// function's body never reads its own "return type".
    fn finishFuncOrDecl(self: *Parser, name: []const u8, params: []Param, ret: ?ctype.CType, storage: StorageClass, is_variadic: bool, funcs: *std.ArrayList(Func), func_decls: *std.ArrayList(FuncDecl)) Error!void {
        // A `__attribute__` run may trail a function prototype/definition's `)`, before the
        // `{`/`;`, glibc's dominant `__THROW`/`__nonnull`/`__wur` position (for example `int
        // f(int x) __attribute__((__nothrow__));`). Skip it here, shared by both the
        // bodyless-prototype and the full-definition path below, and by both callers of this
        // function (the `void`-returning special case and the general typed-return case).
        // This also skips a trailing `__asm__`/`asm` SYMBOL-RENAME label at this same
        // position, glibc's other dominant spot for it (for example `int p(void)
        // __asm__("myp");`), interleaved with `__attribute__` in either order (`int q(void)
        // __asm__("qq") __attribute__((__nothrow__));`). See `skipAsmLabelAndAttributes`'s
        // doc comment for why the rename itself is ignored, not honored.
        try self.skipAsmLabelAndAttributes();
        if (self.peek().kind == .lbrace) {
            self.pos += 1;
            self.cur_func_name = name; // for `__func__` inside the body.
            var body: std.ArrayList(Stmt) = .empty;
            while (self.peek().kind != .rbrace) try body.append(self.arena, try self.parseStmt());
            _ = try self.expect(.rbrace);
            try funcs.append(self.arena, .{ .name = name, .params = params, .body = try body.toOwnedSlice(self.arena), .ret = ret orelse ctype.int_t, .is_variadic = is_variadic, .is_static = storage == .static or self.saw_inline });
            return;
        }
        _ = try self.expect(.semicolon);
        try func_decls.append(self.arena, .{ .name = name, .params = params, .ret = ret, .storage = storage, .is_variadic = is_variadic });
    }
};

/// Parse `source` into a `Unit`: one or more functions, until `eof`. The token `text`
/// slices (names) are copied into the unit's arena so the caller need not keep `source`
/// alive. `layout` resolves target-dependent type details (plain-`char` signedness, and
/// `long`'s width downstream in lowering) in `parseTypeSpec`, used by decl/param/func
/// parsing. `pp` configures the preprocessor pass that runs ahead of parsing. Use `.{}`
/// for plain C89 source with no directives/macros/includes.
pub fn parse(allocator: std.mem.Allocator, source: []const u8, layout: layout_mod.TargetLayout, pp: preproc.Options) Error!Unit {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();

    const toks = try preproc.preprocess(allocator, source, pp);
    defer lexer.freeTokens(allocator, toks);

    var p = Parser{ .toks = toks, .pos = 0, .arena = arena.allocator(), .layout = layout };
    var funcs: std.ArrayList(Func) = .empty;
    var globals: std.ArrayList(GlobalDecl) = .empty;
    var func_decls: std.ArrayList(FuncDecl) = .empty;
    while (p.peek().kind != .eof) {
        // This clear moved HERE, once per top-level item, from inside
        // `parseFuncAfterDeclarator`. `locals` tracks declared vars/params for THE FUNCTION
        // CURRENTLY BEING PARSED only, so an `enum` constant can be shadowed. Clear it
        // before EVERY top-level item, not just before a function's own param list, since
        // `parseFuncDeclaratorTail` can now be reached recursively (a callback PARAMETER
        // that is itself a function-pointer type parses its own `(params)` tail through the
        // very same helper) and clearing THERE would wipe an outer function's
        // already-registered param names mid-parse.
        p.locals.clearRetainingCapacity();
        p.saw_inline = false; // reset the inline marker for this top-level item.
        // A `__attribute__` run may lead a WHOLE top-level item, ahead of even
        // `typedef`/`void`/the storage-class run below. `parseTypeSpec`'s own leading skip
        // only fires for the typed-return/global path, which calls it. The `typedef` and
        // `void`-returning-function special cases just below do not, so it is caught here too.
        try p.skipAttributeSpecifiers();
        p.skipPragmas(); // `_Pragma("...")` around a top-level item is a no-op.
        // A leading run of `inline`/`_Noreturn`/`__extension__` (no-op function
        // specifiers/prefixes) may ALSO lead a whole top-level item, ahead of even
        // `typedef`/`void`/the storage-class run below. Critically, this is ahead of the
        // bare-`void` special case just below, so `_Noreturn void n(void){ }` still reaches
        // THAT case (its `ret = null` placeholder, not a real `void` return type threaded
        // through the general typed-return path, which a body with no `return` cannot fall
        // off the end of, see that case's own comment for why). It records nothing either
        // way. See `isNoOpSpecifierIdent`'s doc comment for what each spelling means and why
        // VCC ignores it.
        while (p.peek().kind == .ident and isNoOpSpecifierIdent(p.peek().text)) {
            const s = p.peek().text;
            if (std.mem.eql(u8, s, "inline") or std.mem.eql(u8, s, "__inline") or std.mem.eql(u8, s, "__inline__")) p.saw_inline = true;
            p.pos += 1;
        }
        // The leading skips above (attributes, `_Pragma`, no-op specifiers) may consume the
        // last tokens in the file. A trailing `_Pragma("GCC diagnostic pop")` is common. The
        // `while` guard only re-checks at the top, so stop here if nothing real is left.
        if (p.peek().kind == .eof) break;
        // `_Static_assert (expr, "msg");` at file scope (C11 6.7.10). gnulib's `verify`
        // emits it once VCC advertises a new enough `__GNUC__`. Evaluated and discarded here.
        if (try p.tryStaticAssert()) continue;
        // `typedef <type> <name>;` at file scope. Checked FIRST, ahead of the general
        // type-spec parse below. `parseTypeSpec` itself has no notion of the `typedef`
        // keyword, so this is a distinct top-level item shape. Consume `typedef`, then
        // reuse `parseTypeSpec` + `parseDeclarator`, exactly like a struct field or a local
        // decl, to get the aliased type and the alias name, register it, and move on. A
        // typedef declares no function, so it is skipped in the `funcs` loop entirely.
        if (p.peek().kind == .kw_typedef) {
            p.pos += 1;
            const base = try p.parseTypeSpec();
            const d = try p.parseDeclarator(base.ty, base.quals);
            // A `__attribute__` run may trail a typedef's declarator, before its `;` (for
            // example `typedef int myint __attribute__((__aligned__(4)));`).
            try p.skipAttributeSpecifiers();
            _ = try p.expect(.semicolon);
            try p.typedefs.put(p.arena, d.name, d.ty);
            continue;
        }
        // A `void`-returning function definition, needed to test a global written by one
        // function and read by another with a natural "setter" shape. `void` is NOT a
        // `parseTypeSpec` result. This frontend has no genuine void `CType`. There is
        // nothing for a bare `void x;` global/local/param to mean, and those stay rejected
        // exactly as before. So it is special-cased here, narrowly, as a function-
        // definition-only shape: consume `void`, then a name and `(` MUST follow. A bare
        // `void;`/`void x;` is not this case and falls through to `parseTypeSpec` below,
        // which still rejects it. The function's declared return type is `ctype.int_t` as a
        // placeholder, invisible to any caller, since a "void" function's body has no
        // `return` (the fall-off-the-end seals an unread `return 0`) and every call site
        // uses it as a bare statement, discarding whatever value the call "returns".
        // `void` prototypes (`void f(int);`) fork the same way the typed case below does.
        // Params are parsed directly here, mirroring an older version's own clear+params
        // order, then `finishFuncOrDecl` picks `{` (definition) vs `;` (declaration, always
        // `storage = .none`, no leading `static`/`extern` reaches this branch, see this
        // branch's ORIGINAL comment above: those are consumed by the storage-loop below,
        // which does not understand `void`).
        if (p.peek().kind == .kw_void and p.toks[p.pos + 1].kind == .ident) {
            p.pos += 1;
            const name = try p.expect(.ident);
            const name_owned = try p.arena.dupe(u8, name.text);
            _ = try p.expect(.lparen);
            const pr = try p.parseParams();
            _ = try p.expect(.rparen);
            try p.finishFuncOrDecl(name_owned, pr.params, null, .none, pr.is_variadic, &funcs, &func_decls);
            continue;
        }
        // A leading run of storage-class/qualifier keywords, order-independent: `static`/
        // `extern` (recorded on the `GlobalDecl`, though linkage does not change single-TU
        // behavior here) and `const`/`volatile` (`const` DOES drive `.rodata` routing in
        // `lower.compile` via `is_const`, unchanged; `lead_quals` additionally threads into
        // `parseDeclarator` as `base_quals` so a pointer declarator attributes correctly, see
        // `parseDeclarator`'s doc comment and `parseDeclStmt`, which mirrors this exact shape
        // for locals). For example `static const int k = 5;` or `const static int k = 5;`.
        var storage: StorageClass = .none;
        var is_const = false;
        var lead_quals: ctype.Quals = .{};
        while (true) {
            switch (p.peek().kind) {
                .kw_static => {
                    storage = .static;
                    p.pos += 1;
                },
                .kw_extern => {
                    storage = .extern_;
                    p.pos += 1;
                },
                .kw_const => {
                    is_const = true;
                    lead_quals.is_const = true;
                    p.pos += 1;
                },
                .kw_volatile => {
                    lead_quals.is_volatile = true;
                    p.pos += 1;
                },
                // `inline`/`_Noreturn`/`__extension__` may also compose WITHIN this run,
                // interleaved with `static`/`extern`/`const`/`volatile` in any order (for
                // example `static inline int f(void)`). This records nothing, same as the
                // leading run above this loop handles for the case where one of these starts
                // the item.
                .ident => if (isNoOpSpecifierIdent(p.peek().text)) {
                    // Catch `inline` interleaved with a storage class (`extern inline`,
                    // gnulib's `_GL_INLINE`), so this function is emitted local like a
                    // leading `inline` (see `saw_inline`).
                    const s = p.peek().text;
                    if (std.mem.eql(u8, s, "inline") or std.mem.eql(u8, s, "__inline") or std.mem.eql(u8, s, "__inline__")) p.saw_inline = true;
                    p.pos += 1;
                } else break,
                else => break,
            }
        }
        // Every other top-level item starts with a type-spec: a function's return type or a
        // global's declared type, or a bare `struct`/`union` TAG DEFINITION with no
        // declarator at all. Parse the type-spec once, this is also where `struct Tag { ...
        // }`'s body gets parsed and registered in `p.structs`, then a lone `;` right after
        // means this item WAS the bare definition, nothing more to do.
        const base = try p.parseTypeSpec();
        if (p.peek().kind == .semicolon) {
            p.pos += 1;
            continue;
        }
        const base_quals: ctype.Quals = .{
            .is_const = lead_quals.is_const or base.quals.is_const,
            .is_volatile = lead_quals.is_volatile or base.quals.is_volatile,
        };
        // Otherwise a declarator follows, parsed ONCE here, shared with locals/params, see
        // `parseDeclarator`, and what it resolved to disambiguates a function (`d.ty ==
        // .func`, the function-declarator suffix already consumed its own `(params)`, see
        // `parseDeclarator`'s doc comment) from a global variable declaration (anything
        // else: `;` for a bare `int g;`, `=` for an initializer, or `,` starting another
        // comma-separated declarator).
        const d = try p.parseDeclarator(base.ty, base_quals);
        if (d.ty == .func) {
            // A definition (`{`) or a bodyless declaration (`;`), told apart by
            // `finishFuncOrDecl`. `storage` (`extern`/`static`, parsed above) carries onto a
            // `FuncDecl` for the `;` case, and is silently dropped for a `{`-bodied
            // definition. `Func` has no storage field. Linkage does not change single-TU
            // lowering here regardless.
            const ret: ctype.CType = d.ty.func.ret.?.*;
            try p.finishFuncOrDecl(d.name, d.func_params.?, ret, storage, d.ty.func.is_variadic, &funcs, &func_decls);
            continue;
        }
        // A global variable declaration: `int g;`, `int g = <expr>;`, or a comma-separated
        // list (`int a = 1, b;`), each name/declarator wrapping fresh over the SAME base type
        // (`int *a, b;` declares `a` a pointer and `b` a plain `int`, same as a local/param
        // declarator list would) with its OWN optional initializer.
        const init0 = try p.parseGlobalInit();
        try globals.append(arena.allocator(), .{ .name = d.name, .ty = resolveUnsizedArrayLen(d.ty, init0), .storage = storage, .is_const = is_const, .init = init0, .quals = d.quals });
        while (p.peek().kind == .comma) {
            p.pos += 1;
            const d2 = try p.parseDeclarator(base.ty, base_quals);
            // A function declarator cannot continue a global's comma-list (`int a, f(int);`,
            // mixing a variable and a prototype in one declaration). This is out of scope
            // and fails closed rather than silently dropping `f`'s prototype semantics.
            if (d2.ty == .func) return error.Unsupported;
            const init2 = try p.parseGlobalInit();
            try globals.append(arena.allocator(), .{ .name = d2.name, .ty = resolveUnsizedArrayLen(d2.ty, init2), .storage = storage, .is_const = is_const, .init = init2, .quals = d2.quals });
        }
        _ = try p.expect(.semicolon);
    }
    return .{
        .arena = arena,
        .funcs = try funcs.toOwnedSlice(arena.allocator()),
        .globals = try globals.toOwnedSlice(arena.allocator()),
        .func_decls = try func_decls.toOwnedSlice(arena.allocator()),
    };
}
