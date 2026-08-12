//! The C lexer: turns source bytes into a stream of tokens. It tokenizes identifiers,
//! int literals, and the full operator and punctuator table.

const std = @import("std");

/// A lexical token kind. Covers identifiers, int literals, and the full operator and
/// punctuator table: arithmetic, bitwise, shift, comparison, assignment and
/// compound-assignment, and punctuation.
pub const Kind = enum {
    kw_int,
    kw_char,
    kw_short,
    kw_long,
    kw_unsigned,
    kw_signed,
    kw_void,
    kw_return,
    kw_if,
    kw_else,
    kw_while,
    kw_for,
    kw_do,
    kw_switch,
    kw_case,
    kw_default,
    kw_break,
    kw_continue,
    kw_goto,
    kw_sizeof,
    kw_struct,
    kw_union,
    kw_enum,
    kw_typedef,
    kw_static,
    kw_extern,
    kw_const,
    kw_volatile,
    kw_float,
    kw_double,
    kw_bool,
    ident,
    int_lit,
    float_lit,
    str_lit,
    /// A wide string literal `L"..."`. Its `wchar_t` element is a signed 32-bit int on
    /// every target this frontend supports, so the parser and lowering widen each
    /// decoded byte to 4 bytes. The lexer itself stores the same 1-byte-per-char
    /// decoded text a narrow `str_lit` does.
    wstr_lit,
    lparen,
    rparen,
    lbrace,
    rbrace,
    lbracket,
    rbracket,
    semicolon,
    comma,
    minus,
    minus_minus,
    plus,
    plus_plus,
    star,
    slash,
    percent,
    amp,
    pipe,
    caret,
    tilde,
    bang,
    lt,
    gt,
    assign,
    lshift,
    rshift,
    le,
    ge,
    eq_eq,
    bang_eq,
    plus_eq,
    minus_eq,
    star_eq,
    slash_eq,
    percent_eq,
    amp_eq,
    pipe_eq,
    caret_eq,
    lshift_eq,
    rshift_eq,
    amp_amp,
    pipe_pipe,
    question,
    colon,
    dot,
    /// `...`: a variadic function declarator's trailing parameter (`int f(int, ...)`)
    /// and a variadic function-like macro's parameter list (`#define F(...)`, checked
    /// by `preproc.zig`'s own soft-fail check). Matched as one token, ahead of `.dot`
    /// in `operators` below, so maximal munch never splits it into three `.dot`s.
    ellipsis,
    arrow,
    eof,
    // Preprocessor-only kinds: `preproc.scan` emits these for `#`, `##`, and physical
    // line ends. `preproc.emit` consumes or rejects them before the token stream
    // reaches the parser, so a `Token` handed to `parser.parse` never carries one of
    // these three.
    hash,
    hash_hash,
    newline,
};

/// A token: its kind, its text, and its 1-based line and column for diagnostics.
/// `.text` is usually a slice borrowed from the input, but string literals, decoded
/// bytes, and char-literal `int_lit`s, a synthesized decimal string, heap-allocate it
/// instead. Those set `owns_text` so `freeTokens` knows to free it. Free a token slice
/// with `freeTokens`, never a bare `allocator.free(toks)`, or the owned text leaks.
pub const Token = struct { kind: Kind, text: []const u8, line: u32, col: u32, owns_text: bool = false };

pub const Error = error{UnexpectedChar} || std.mem.Allocator.Error;

/// Frees a token slice returned by `tokenize`: first the heap-allocated `.text` of every
/// token that owns it (string/char literals), then the slice itself. Mirrors how `tokenize`
/// allocates; use this instead of `allocator.free(toks)`.
pub fn freeTokens(allocator: std.mem.Allocator, toks: []Token) void {
    for (toks) |t| if (t.owns_text) allocator.free(t.text);
    allocator.free(toks);
}

pub fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_';
}

pub fn isIdentCont(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// Maps identifier text to its keyword `Kind`, or null if it is a plain identifier.
/// `preproc.emit` applies this to `.ident` tokens at final emission time, after macro
/// expansion, so a macro-expanded name that happens to spell a keyword is classified
/// the same way source-written text is.
pub fn keyword(text: []const u8) ?Kind {
    if (std.mem.eql(u8, text, "int")) return .kw_int;
    if (std.mem.eql(u8, text, "char")) return .kw_char;
    if (std.mem.eql(u8, text, "short")) return .kw_short;
    if (std.mem.eql(u8, text, "long")) return .kw_long;
    if (std.mem.eql(u8, text, "unsigned")) return .kw_unsigned;
    if (std.mem.eql(u8, text, "signed")) return .kw_signed;
    if (std.mem.eql(u8, text, "void")) return .kw_void;
    if (std.mem.eql(u8, text, "return")) return .kw_return;
    if (std.mem.eql(u8, text, "if")) return .kw_if;
    if (std.mem.eql(u8, text, "else")) return .kw_else;
    if (std.mem.eql(u8, text, "while")) return .kw_while;
    if (std.mem.eql(u8, text, "for")) return .kw_for;
    if (std.mem.eql(u8, text, "do")) return .kw_do;
    if (std.mem.eql(u8, text, "switch")) return .kw_switch;
    if (std.mem.eql(u8, text, "case")) return .kw_case;
    if (std.mem.eql(u8, text, "default")) return .kw_default;
    if (std.mem.eql(u8, text, "break")) return .kw_break;
    if (std.mem.eql(u8, text, "continue")) return .kw_continue;
    if (std.mem.eql(u8, text, "goto")) return .kw_goto;
    if (std.mem.eql(u8, text, "sizeof")) return .kw_sizeof;
    if (std.mem.eql(u8, text, "struct")) return .kw_struct;
    if (std.mem.eql(u8, text, "union")) return .kw_union;
    if (std.mem.eql(u8, text, "enum")) return .kw_enum;
    if (std.mem.eql(u8, text, "typedef")) return .kw_typedef;
    if (std.mem.eql(u8, text, "static")) return .kw_static;
    if (std.mem.eql(u8, text, "extern")) return .kw_extern;
    if (std.mem.eql(u8, text, "const")) return .kw_const;
    if (std.mem.eql(u8, text, "volatile")) return .kw_volatile;
    if (std.mem.eql(u8, text, "float")) return .kw_float;
    if (std.mem.eql(u8, text, "double")) return .kw_double;
    // C99 `_Bool`: a 1-byte integer type whose only values are 0 and 1. `isIdentStart`
    // already accepts a leading `_`, so `_Bool` scans as a plain identifier first. This
    // keyword check is what reclassifies it, the same as every other type keyword above.
    if (std.mem.eql(u8, text, "_Bool")) return .kw_bool;
    return null;
}

pub const Operator = struct { text: []const u8, kind: Kind };

/// Operators and punctuators, ordered LONGEST FIRST so a linear first-match scan performs
/// maximal munch (`<<=` matched before `<<` before `<`).
pub const operators = [_]Operator{
    .{ .text = "<<=", .kind = .lshift_eq },  .{ .text = ">>=", .kind = .rshift_eq },
    .{ .text = "<<", .kind = .lshift },      .{ .text = ">>", .kind = .rshift },
    .{ .text = "<=", .kind = .le },          .{ .text = ">=", .kind = .ge },
    .{ .text = "==", .kind = .eq_eq },       .{ .text = "!=", .kind = .bang_eq },
    .{ .text = "+=", .kind = .plus_eq },     .{ .text = "-=", .kind = .minus_eq },
    .{ .text = "*=", .kind = .star_eq },     .{ .text = "/=", .kind = .slash_eq },
    .{ .text = "%=", .kind = .percent_eq },  .{ .text = "&=", .kind = .amp_eq },
    .{ .text = "|=", .kind = .pipe_eq },     .{ .text = "^=", .kind = .caret_eq },
    .{ .text = "&&", .kind = .amp_amp },     .{ .text = "||", .kind = .pipe_pipe },
    .{ .text = "->", .kind = .arrow },       .{ .text = "++", .kind = .plus_plus },
    .{ .text = "--", .kind = .minus_minus }, .{ .text = "+", .kind = .plus },
    .{ .text = "-", .kind = .minus },        .{ .text = "*", .kind = .star },
    .{ .text = "/", .kind = .slash },        .{ .text = "%", .kind = .percent },
    .{ .text = "&", .kind = .amp },          .{ .text = "|", .kind = .pipe },
    .{ .text = "^", .kind = .caret },        .{ .text = "~", .kind = .tilde },
    .{ .text = "!", .kind = .bang },         .{ .text = "<", .kind = .lt },
    .{ .text = ">", .kind = .gt },           .{ .text = "=", .kind = .assign },
    .{ .text = "(", .kind = .lparen },       .{ .text = ")", .kind = .rparen },
    .{ .text = "{", .kind = .lbrace },       .{ .text = "}", .kind = .rbrace },
    .{ .text = ";", .kind = .semicolon },    .{ .text = ",", .kind = .comma },
    .{ .text = "?", .kind = .question },     .{ .text = ":", .kind = .colon },
    .{ .text = "[", .kind = .lbracket },     .{ .text = "]", .kind = .rbracket },
    .{ .text = "...", .kind = .ellipsis },   .{ .text = ".", .kind = .dot },
};

/// Longest operator whose text is a prefix of `rest`, or null. `operators` is ordered
/// longest-first, so the first match is the maximal munch.
pub fn matchOperator(rest: []const u8) ?Operator {
    for (operators) |op| if (std.mem.startsWith(u8, rest, op.text)) return op;
    return null;
}

/// Decodes a single escape sequence starting right after the backslash (`source[i.*]` is
/// the character following `\`). Advances `i`/`col` past the consumed escape characters
/// and returns the decoded byte. Supports `\n \t \r \\ \" \'`, up-to-3-digit octal
/// (`\0`, `\123`, ...), and `\xHH` (1-2 hex digits).
pub fn decodeEscape(source: []const u8, i: *usize, col: *u32) Error!u8 {
    const c = source[i.*];
    switch (c) {
        'n' => {
            i.* += 1;
            col.* += 1;
            return '\n';
        },
        't' => {
            i.* += 1;
            col.* += 1;
            return '\t';
        },
        'r' => {
            i.* += 1;
            col.* += 1;
            return '\r';
        },
        '\\' => {
            i.* += 1;
            col.* += 1;
            return '\\';
        },
        '"' => {
            i.* += 1;
            col.* += 1;
            return '"';
        },
        '\'' => {
            i.* += 1;
            col.* += 1;
            return '\'';
        },
        // The remaining C simple escapes (C11 6.4.4.4): `\a` alert, `\b` backspace, `\f` form
        // feed, `\v` vertical tab, and `\?` (a literal question mark, allowed in an escape so a
        // source can avoid a trigraph). Real system headers use `\v`/`\f` in `iswspace`-style
        // character tests.
        'a' => {
            i.* += 1;
            col.* += 1;
            return 7;
        },
        'b' => {
            i.* += 1;
            col.* += 1;
            return 8;
        },
        'f' => {
            i.* += 1;
            col.* += 1;
            return 12;
        },
        'v' => {
            i.* += 1;
            col.* += 1;
            return 11;
        },
        '?' => {
            i.* += 1;
            col.* += 1;
            return '?';
        },
        '0'...'7' => {
            // Octal `\ddd`, 1-3 digits. Accumulate with wrapping arithmetic and keep only
            // the low 8 bits: these are char/string bytes, so a value > 255 truncates to a
            // byte rather than overflowing.
            var val: u8 = 0;
            var n: usize = 0;
            while (i.* < source.len and source[i.*] >= '0' and source[i.*] <= '7' and n < 3) : (n += 1) {
                val = val *% 8 +% (source[i.*] - '0');
                i.* += 1;
                col.* += 1;
            }
            return val;
        },
        'x' => {
            // Hex `\xH...`, 1+ digits (no C-mandated cap). Accumulate with wrapping
            // arithmetic in a u8 so an over-long escape like `\x123456` truncates to its
            // low byte instead of panicking on integer overflow.
            i.* += 1;
            col.* += 1;
            var val: u8 = 0;
            var n: usize = 0;
            while (i.* < source.len) {
                const digit = std.fmt.charToDigit(source[i.*], 16) catch break;
                val = val *% 16 +% digit;
                i.* += 1;
                col.* += 1;
                n += 1;
            }
            if (n == 0) return error.UnexpectedChar;
            return val;
        },
        else => return error.UnexpectedChar,
    }
}

/// Consumes a `.` at `i.*` and the (possibly empty, as in `2.`) run of decimal digits
/// following it. Shared by both digit-led floats (`2.5`, `2.`, where the integer part is
/// already consumed) and dot-led floats (`.5`, where the `.` is the very first character).
fn scanFloatFraction(source: []const u8, i: *usize, col: *u32) void {
    i.* += 1; // the '.'
    col.* += 1;
    while (i.* < source.len and std.ascii.isDigit(source[i.*])) {
        i.* += 1;
        col.* += 1;
    }
}

/// Consumes an optional exponent (`e`/`E`, optional `+`/`-` sign, then digits) and then an
/// optional float suffix (`f`/`F`/`l`/`L`), advancing past whichever are present. Called
/// once a literal is already known to be a float, so the suffix is always checked even
/// when there's no exponent (e.g. `1.0f`).
fn scanFloatExponentAndSuffix(source: []const u8, i: *usize, col: *u32) void {
    if (i.* < source.len and (source[i.*] == 'e' or source[i.*] == 'E')) {
        i.* += 1;
        col.* += 1;
        if (i.* < source.len and (source[i.*] == '+' or source[i.*] == '-')) {
            i.* += 1;
            col.* += 1;
        }
        while (i.* < source.len and std.ascii.isDigit(source[i.*])) {
            i.* += 1;
            col.* += 1;
        }
    }
    while (i.* < source.len and switch (source[i.*]) {
        'f', 'F', 'l', 'L' => true,
        else => false,
    }) {
        i.* += 1;
        col.* += 1;
    }
}

/// Scans one numeric literal starting at `source[i.*]`. This is either a decimal digit,
/// an int_lit or digit-led float_lit such as `42`, `0x1F`, `1.5`, or `2.`, or a `.`
/// immediately followed by a digit, a dot-led float_lit like `.5`, where the caller
/// confirms the digit before calling. It advances `i` and `col` past the literal and
/// returns whether it is a float literal. It handles the `0x` or `0X` hex and `0b` or
/// `0B` binary integer prefixes, never floats, a decimal digit run optionally followed
/// by a fractional part and/or exponent (`.`, `e`, or `E`, which make it a float_lit),
/// and the trailing suffix: integer (`u`, `U`, `l`, or `L`), or, once known to be a
/// float, float (`f`, `F`, `l`, or `L`, via `scanFloatExponentAndSuffix`). Shared by
/// `lexer.tokenize` and `preproc.scan`'s number scanners, so the radix and float
/// detection logic lives in exactly one place instead of being duplicated per caller.
pub fn scanNumber(source: []const u8, i: *usize, col: *u32) Error!bool {
    if (source[i.*] == '.') {
        scanFloatFraction(source, i, col);
        scanFloatExponentAndSuffix(source, i, col);
        return true;
    }
    var is_float = false;
    if (source[i.*] == '0' and i.* + 1 < source.len and (source[i.* + 1] == 'x' or source[i.* + 1] == 'X')) {
        // Hex integer literal: `0x` or `0X` plus hex digits. Hex floats (`0x1p3`) are
        // out of scope, so a trailing `p` or `P` is simply left unconsumed. Hex stays int-only.
        i.* += 2;
        col.* += 2;
        // The prefix MUST be followed by at least one hex digit; `0x`, `0xG`, `0x` at EOF
        // are malformed literals (gcc rejects them), not `int_lit("0x")` + a stray token.
        if (i.* >= source.len or !std.ascii.isHex(source[i.*])) return error.UnexpectedChar;
        while (i.* < source.len and std.ascii.isHex(source[i.*])) {
            i.* += 1;
            col.* += 1;
        }
    } else if (source[i.*] == '0' and i.* + 1 < source.len and (source[i.* + 1] == 'b' or source[i.* + 1] == 'B')) {
        // Binary integer literal: `0b`/`0B` + binary digits.
        i.* += 2;
        col.* += 2;
        // Same rule as hex: at least one binary digit must follow. `0b`, `0b2` (2 is not a
        // binary digit), `0b` at EOF are malformed.
        if (i.* >= source.len or (source[i.*] != '0' and source[i.*] != '1')) return error.UnexpectedChar;
        while (i.* < source.len and (source[i.*] == '0' or source[i.*] == '1')) {
            i.* += 1;
            col.* += 1;
        }
    } else {
        // Decimal digit run. A leading-zero run like `0755` is kept as plain decimal
        // digit text here. Octal interpretation is left to the parser or sema layer.
        while (i.* < source.len and std.ascii.isDigit(source[i.*])) {
            i.* += 1;
            col.* += 1;
        }
        // Float detection: a `.` (fractional part, possibly empty as in `2.`) or an
        // exponent `e`/`E` turns this into a float_lit instead of an int_lit. The hex/binary
        // branches above never reach here, so this can't misfire on something like `0x1e3`.
        if (i.* < source.len and source[i.*] == '.') {
            is_float = true;
            scanFloatFraction(source, i, col);
        } else if (i.* < source.len and (source[i.*] == 'e' or source[i.*] == 'E')) {
            is_float = true;
        }
        if (is_float) scanFloatExponentAndSuffix(source, i, col);
    }
    if (!is_float) {
        // Trailing integer-literal suffix (`u`, `U`, `l`, or `L`, in any combination or
        // order, such as `u`, `L`, `UL`, `LL`, or `ull`), consumed into the token's
        // text. The parser splits it back out and resolves it to a `CType`. Hex digits
        // never include these letters, so this cannot eat part of a hex literal's digit run.
        while (i.* < source.len and switch (source[i.*]) {
            'u', 'U', 'l', 'L' => true,
            else => false,
        }) {
            i.* += 1;
            col.* += 1;
        }
    }
    return is_float;
}

/// Scan a `"`-delimited string body, the opening quote already consumed by the caller:
/// decode escapes into a fresh owned buffer, up to and including the closing `"`.
/// Shared by narrow `str_lit` and wide `wstr_lit`. The two differ only in which `Kind`
/// the caller wraps this text in, not in how the bytes are decoded.
fn scanStringBody(allocator: std.mem.Allocator, source: []const u8, i: *usize, col: *u32, line: *u32) Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    while (i.* < source.len and source[i.*] != '"') {
        if (source[i.*] == '\\') {
            i.* += 1;
            col.* += 1;
            if (i.* >= source.len) return error.UnexpectedChar;
            try buf.append(allocator, try decodeEscape(source, i, col));
        } else {
            if (source[i.*] == '\n') {
                line.* += 1;
                col.* = 1;
            } else {
                col.* += 1;
            }
            try buf.append(allocator, source[i.*]);
            i.* += 1;
        }
    }
    if (i.* >= source.len) return error.UnexpectedChar;
    i.* += 1;
    col.* += 1;
    return try buf.toOwnedSlice(allocator);
}

/// Scan a `'`-delimited char body, the opening quote already consumed by the caller:
/// decode one, possibly escaped, byte, up to and including the closing `'`. Shared by
/// narrow `'c'` and wide `L'c'`, since a wide char literal is functionally identical to
/// a narrow one for a single ASCII char (see `tokenize`'s `L`-prefix branch). Both
/// funnel through here and both emit the same `.int_lit`. This is `pub` because
/// `preproc.zig`'s own standalone scanner (see that file's `scan`) reuses this
/// directly, so its `L'c'` handling cannot drift from this one.
pub fn scanCharBody(source: []const u8, i: *usize, col: *u32) Error!u8 {
    if (i.* >= source.len) return error.UnexpectedChar;
    var value: u8 = undefined;
    if (source[i.*] == '\\') {
        i.* += 1;
        col.* += 1;
        if (i.* >= source.len) return error.UnexpectedChar;
        value = try decodeEscape(source, i, col);
    } else {
        value = source[i.*];
        i.* += 1;
        col.* += 1;
    }
    if (i.* >= source.len or source[i.*] != '\'') return error.UnexpectedChar;
    i.* += 1;
    col.* += 1;
    return value;
}

/// Tokenize `source`. The result always ends with an `eof` token. Whitespace (space,
/// tab, newline, carriage return) separates tokens and is otherwise discarded. Caller
/// owns the returned slice.
pub fn tokenize(allocator: std.mem.Allocator, source: []const u8) Error![]Token {
    var out: std.ArrayList(Token) = .empty;
    // On a mid-scan failure, free any heap-allocated token text already produced (string /
    // char literals) before dropping the list, so nothing leaks.
    errdefer {
        for (out.items) |t| if (t.owns_text) allocator.free(t.text);
        out.deinit(allocator);
    }

    var i: usize = 0;
    var line: u32 = 1;
    var col: u32 = 1;
    while (i < source.len) {
        const c = source[i];
        switch (c) {
            ' ', '\t', '\r' => {
                i += 1;
                col += 1;
            },
            '\n' => {
                i += 1;
                line += 1;
                col = 1;
            },
            else => {
                // A wide-literal prefix `L'...'` or `L"..."`: only fires when the `L` is
                // immediately followed by `'` or `"`, with no whitespace, so `Label`,
                // `L1`, and a variable named `L` all stay plain identifiers. They fall
                // through to the `isIdentStart` branch below untouched.
                if (c == 'L' and i + 1 < source.len and (source[i + 1] == '\'' or source[i + 1] == '"')) {
                    const start_col = col;
                    i += 1; // the `L` itself
                    col += 1;
                    const quote = source[i];
                    i += 1; // the opening quote
                    col += 1;
                    if (quote == '"') {
                        const text = try scanStringBody(allocator, source, &i, &col, &line);
                        errdefer allocator.free(text);
                        try out.append(allocator, .{ .kind = .wstr_lit, .text = text, .line = line, .col = start_col, .owns_text = true });
                    } else {
                        // `L'c'` is functionally identical to `'c'`, both `int` with the
                        // same code point, so emit the exact same `.int_lit` the narrow
                        // path does.
                        const value = try scanCharBody(source, &i, &col);
                        const text = try std.fmt.allocPrint(allocator, "{d}", .{value});
                        errdefer allocator.free(text);
                        try out.append(allocator, .{ .kind = .int_lit, .text = text, .line = line, .col = start_col, .owns_text = true });
                    }
                } else if (isIdentStart(c)) {
                    const start = i;
                    const start_col = col;
                    while (i < source.len and isIdentCont(source[i])) {
                        i += 1;
                        col += 1;
                    }
                    const text = source[start..i];
                    try out.append(allocator, .{ .kind = keyword(text) orelse .ident, .text = text, .line = line, .col = start_col });
                } else if (std.ascii.isDigit(c)) {
                    const start = i;
                    const start_col = col;
                    const is_float = try scanNumber(source, &i, &col);
                    try out.append(allocator, .{ .kind = if (is_float) .float_lit else .int_lit, .text = source[start..i], .line = line, .col = start_col });
                } else if (c == '.' and i + 1 < source.len and std.ascii.isDigit(source[i + 1])) {
                    // A `.` immediately followed by a digit starts a float literal (`.5`),
                    // not the `.` member-access operator.
                    const start = i;
                    const start_col = col;
                    _ = try scanNumber(source, &i, &col);
                    try out.append(allocator, .{ .kind = .float_lit, .text = source[start..i], .line = line, .col = start_col });
                } else if (c == '"') {
                    const start_col = col;
                    i += 1;
                    col += 1;
                    const text = try scanStringBody(allocator, source, &i, &col, &line);
                    errdefer allocator.free(text);
                    try out.append(allocator, .{ .kind = .str_lit, .text = text, .line = line, .col = start_col, .owns_text = true });
                } else if (c == '\'') {
                    const start_col = col;
                    i += 1;
                    col += 1;
                    const value = try scanCharBody(source, &i, &col);
                    const text = try std.fmt.allocPrint(allocator, "{d}", .{value});
                    errdefer allocator.free(text);
                    try out.append(allocator, .{ .kind = .int_lit, .text = text, .line = line, .col = start_col, .owns_text = true });
                } else if (matchOperator(source[i..])) |op| {
                    try out.append(allocator, .{ .kind = op.kind, .text = source[i .. i + op.text.len], .line = line, .col = col });
                    i += op.text.len;
                    col += @intCast(op.text.len);
                } else {
                    return error.UnexpectedChar;
                }
            },
        }
    }
    try out.append(allocator, .{ .kind = .eof, .text = source[source.len..source.len], .line = line, .col = col });
    return out.toOwnedSlice(allocator);
}

test "tokenize a trivial function" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "int f(void){ return 42; }");
    defer freeTokens(allocator, toks);

    const kinds = [_]Kind{
        .kw_int, .ident,     .lparen,  .kw_void,   .rparen,
        .lbrace, .kw_return, .int_lit, .semicolon, .rbrace,
        .eof,
    };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("f", toks[1].text);
    try std.testing.expectEqualStrings("42", toks[7].text);
}

test "tokenize reports line and column" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "int\n  f");
    defer freeTokens(allocator, toks);
    try std.testing.expectEqual(@as(u32, 2), toks[1].line);
    try std.testing.expectEqual(@as(u32, 3), toks[1].col);
}

test "an unknown character is rejected" {
    // '@' is not part of the token set. tokenize errors before toOwnedSlice, so the
    // errdefer frees the in-progress token list and std.testing.allocator stays happy.
    try std.testing.expectError(error.UnexpectedChar, tokenize(std.testing.allocator, "int f(void){ return @; }"));
}

test "empty input yields a single eof token at line 1, col 1" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "");
    defer freeTokens(allocator, toks);
    try std.testing.expectEqual(@as(usize, 1), toks.len);
    try std.testing.expectEqual(Kind.eof, toks[0].kind);
    try std.testing.expectEqual(@as(u32, 1), toks[0].line);
    try std.testing.expectEqual(@as(u32, 1), toks[0].col);
}

test "maximal munch of multi-char operators" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "a<<=b>>c==d!=e<=f>=g");
    defer freeTokens(allocator, toks);
    const kinds = [_]Kind{ .ident, .lshift_eq, .ident, .rshift, .ident, .eq_eq, .ident, .bang_eq, .ident, .le, .ident, .ge, .ident, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

test "single-char operators" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "+ * / % & | ^ ~ ! < > =");
    defer freeTokens(allocator, toks);
    const kinds = [_]Kind{ .plus, .star, .slash, .percent, .amp, .pipe, .caret, .tilde, .bang, .lt, .gt, .assign, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

test "control-flow keywords and && || ? :" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "if while && || ? :");
    defer freeTokens(allocator, toks);
    const kinds = [_]Kind{ .kw_if, .kw_while, .amp_amp, .pipe_pipe, .question, .colon, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

test "int-literal suffixes are captured into the token text" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "1u 1L 1UL 1LL 1ull 1");
    defer freeTokens(allocator, toks);
    const texts = [_][]const u8{ "1u", "1L", "1UL", "1LL", "1ull", "1" };
    try std.testing.expectEqual(texts.len + 1, toks.len); // + eof
    for (texts, toks[0..texts.len]) |want, got| {
        try std.testing.expectEqual(Kind.int_lit, got.kind);
        try std.testing.expectEqualStrings(want, got.text);
    }
}

test "array brackets tokenize as lbracket/rbracket" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "a[3]");
    defer freeTokens(allocator, toks);
    const kinds = [_]Kind{ .ident, .lbracket, .int_lit, .rbracket, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

// struct, union, enum, and typedef keywords, plus `.` and `->` for member access, not
// used yet. They are lexed now so the parser can reject them cleanly later rather than
// choking on an unknown character.
test "struct/union/enum/typedef keywords and dot/arrow tokens" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "struct union enum typedef a.b p->q");
    defer freeTokens(allocator, toks);
    const kinds = [_]Kind{ .kw_struct, .kw_union, .kw_enum, .kw_typedef, .ident, .dot, .ident, .ident, .arrow, .ident, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

test "lexer: storage-class keywords" {
    const toks = try tokenize(std.testing.allocator, "static extern const");
    defer freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(Kind.kw_static, toks[0].kind);
    try std.testing.expectEqual(Kind.kw_extern, toks[1].kind);
    try std.testing.expectEqual(Kind.kw_const, toks[2].kind);
}

// The `volatile` qualifier keyword.
test "lexer: volatile keyword" {
    const toks = try tokenize(std.testing.allocator, "volatile");
    defer freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(Kind.kw_volatile, toks[0].kind);
}

// The `float` and `double` type keywords.
test "lexer: float/double keywords" {
    const toks = try tokenize(std.testing.allocator, "float double");
    defer freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(Kind.kw_float, toks[0].kind);
    try std.testing.expectEqual(Kind.kw_double, toks[1].kind);
}

test "lexer: string literal decodes escapes" {
    // str_lit's `.text` is freshly allocated (decoding escapes means it can't be a source
    // slice), so `freeTokens` reclaims it via the token's `owns_text` flag.
    const toks = try tokenize(std.testing.allocator, "\"a\\n\\0b\"");
    defer freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(Kind.str_lit, toks[0].kind);
    try std.testing.expect(toks[0].owns_text);
    try std.testing.expectEqualSlices(u8, &.{ 'a', '\n', 0, 'b' }, toks[0].text);
}

test "lexer: char literal is an int_lit" {
    // int_lit carries its value as raw source-style decimal text (see the int-literal
    // suffix test above, which checks `.text` directly against digit strings), so a char
    // literal synthesizes the decimal string of its integer value to match. That string is
    // freshly allocated (not a slice of source), so it sets `owns_text` and `freeTokens`
    // reclaims it.
    const toks = try tokenize(std.testing.allocator, "'A' '\\n'");
    defer freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(Kind.int_lit, toks[0].kind);
    try std.testing.expect(toks[0].owns_text);
    try std.testing.expectEqualSlices(u8, "65", toks[0].text);
    try std.testing.expectEqual(Kind.int_lit, toks[1].kind);
    try std.testing.expectEqualSlices(u8, "10", toks[1].text);
}

test "lexer: hex/octal escapes truncate to a byte and never panic" {
    // `\xff` -> 0xFF, and an over-long `\x123456` accumulates with wrapping arithmetic and
    // yields its low byte (0x56) instead of overflow-panicking.
    const toks = try tokenize(std.testing.allocator, "\"\\xff\\x123456\"");
    defer freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(Kind.str_lit, toks[0].kind);
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0x56 }, toks[0].text);
}

test "lexer hex/octal/binary int literals" {
    const t = try tokenize(std.testing.allocator, "0x1F 0755 0b101 42");
    defer freeTokens(std.testing.allocator, t);
    try std.testing.expectEqual(Kind.int_lit, t[0].kind);
    try std.testing.expectEqualStrings("0x1F", t[0].text);
    try std.testing.expectEqualStrings("0755", t[1].text);
    try std.testing.expectEqualStrings("0b101", t[2].text);
    try std.testing.expectEqualStrings("42", t[3].text);
}

test "lexer rejects a radix prefix with no digits" {
    // A `0x`/`0b` prefix must be followed by at least one valid digit; gcc rejects these
    // as malformed numeric literals rather than splitting them into `int_lit("0x")` + a
    // stray token.
    try std.testing.expectError(error.UnexpectedChar, tokenize(std.testing.allocator, "0x"));
    try std.testing.expectError(error.UnexpectedChar, tokenize(std.testing.allocator, "0xG"));
    try std.testing.expectError(error.UnexpectedChar, tokenize(std.testing.allocator, "0b"));
    try std.testing.expectError(error.UnexpectedChar, tokenize(std.testing.allocator, "0b2"));
}

test "lexer float literals" {
    const t = try tokenize(std.testing.allocator, "1.5 1e3 .5 2. 1.0f");
    defer freeTokens(std.testing.allocator, t);
    for (t[0..5]) |tok| try std.testing.expectEqual(Kind.float_lit, tok.kind);
    try std.testing.expectEqualStrings("1.0f", t[4].text);
}

test "lexer ++ and --" {
    const t = try tokenize(std.testing.allocator, "a++ --b");
    defer freeTokens(std.testing.allocator, t);
    try std.testing.expectEqual(Kind.plus_plus, t[1].kind);
    try std.testing.expectEqual(Kind.minus_minus, t[2].kind);
}

test "compound assignment operators" {
    const allocator = std.testing.allocator;
    const toks = try tokenize(allocator, "+= -= *= /= %= &= |= ^=");
    defer freeTokens(allocator, toks);
    const kinds = [_]Kind{ .plus_eq, .minus_eq, .star_eq, .slash_eq, .percent_eq, .amp_eq, .pipe_eq, .caret_eq, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}
