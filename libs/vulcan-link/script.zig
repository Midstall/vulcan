//! The linker-script parser: turns GNU-ld-style linker-script text (a subset covering
//! ENTRY, MEMORY, SECTIONS, the location counter, output sections and input wildcards,
//! symbol assignments, ALIGN/ORIGIN/LENGTH, LMA/AT, and region assignment) into a typed,
//! arena-backed `Script` AST. Pure text to AST: no layout, no linking, no dependency on
//! `elf.zig`/`resolve.zig`/`arch/*` (this file imports only `std`, keeping `vulcan-link`
//! std-only and this parser independently testable).
//!
//! The file builds the tokenizer, the `Expr` AST, a precedence-climbing `parseExpr`,
//! every AST type the grammar needs, and the top-level/MEMORY/SECTIONS statement
//! parser. `parse` is the full-script entry point.

const std = @import("std");

/// One parse diagnostic: a 1-based line/col and a STATIC message string (never owned
/// text - every call site passes a string literal).
pub const Diagnostic = struct { line: usize, col: usize, msg: []const u8 };

pub const ParseError = std.mem.Allocator.Error || error{ScriptSyntax};

pub const BinOp = enum { add, sub, mul, div, mod, band, bor, shl, shr };

/// An expression node. Every recursive field is a `*const Expr` allocated in the owning
/// `Script`'s arena; every `[]const u8` is likewise arena-duped (never a slice of the
/// original source), so a `Script` outlives the source text it was parsed from.
pub const Expr = union(enum) {
    number: u64,
    /// The location counter `.`.
    dot,
    /// A bare identifier reference (a symbol or, inside `ORIGIN`/`LENGTH`, a region name -
    /// those two are their own variants below since they parse as builtin calls).
    symbol: []const u8,
    binary: struct { op: BinOp, lhs: *const Expr, rhs: *const Expr },
    /// `ALIGN(boundary)` => value = dot; `ALIGN(value, boundary)` => both given.
    align_: struct { value: *const Expr, boundary: *const Expr },
    /// `ORIGIN(region)`.
    origin: []const u8,
    /// `LENGTH(region)`.
    length: []const u8,
};

pub const RegionFlags = struct { r: bool = false, w: bool = false, x: bool = false, a: bool = false, il: bool = false };
pub const Region = struct { name: []const u8, flags: RegionFlags, origin: Expr, length: Expr };
/// `name = value;` or `PROVIDE(name = value);`.
pub const Assign = struct { name: []const u8, value: Expr, provide: bool };
/// `*(.text .text.*)` => file_glob = "*", sections = [".text", ".text.*"]. `KEEP(...)`
/// unwraps to its inner spec. This parser has no garbage collection.
pub const InputSpec = struct { file_glob: []const u8 = "*", sections: []const []const u8 };
/// `AT(expr)` vs `AT>region`.
pub const Lma = union(enum) { addr: Expr, region: []const u8 };
/// One statement inside an output section's `{ }` body.
pub const SectionCmd = union(enum) { input: InputSpec, assign: Assign, set_dot: Expr };
pub const OutputSection = struct {
    name: []const u8,
    /// The optional address between the name and the `:`.
    vma: ?Expr,
    lma: ?Lma,
    /// The `>region` after the closing `}`.
    region: ?[]const u8,
    body: []const SectionCmd,
};
/// One statement inside the top-level `SECTIONS { }` body.
pub const Command = union(enum) { set_dot: Expr, assign: Assign, output: OutputSection };

/// The parsed script. Owns every AST node and every duped string through `arena`; call
/// `deinit` exactly once to free it all.
pub const Script = struct {
    arena: std.heap.ArenaAllocator,
    /// `ENTRY(sym)`, if present.
    entry: ?[]const u8,
    memory: []const Region,
    /// The `SECTIONS` body, in source order.
    commands: []const Command,

    pub fn deinit(self: *Script) void {
        self.arena.deinit();
    }
};

// ---------------------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------------------

const TokKind = enum {
    ident,
    number,
    lbrace,
    rbrace,
    lparen,
    rparen,
    colon,
    semicolon,
    comma,
    assign,
    plus,
    minus,
    star,
    slash,
    percent,
    amp,
    pipe,
    lshift,
    rshift,
    lt,
    gt,
    dot,
    /// `!`, only meaningful inside a MEMORY region's `(flags)` run (GNU-ld attribute
    /// negation). This parser tokenizes it but does not model negation semantics.
    bang,
    eof,
};

const Token = struct {
    kind: TokKind,
    text: []const u8,
    /// Only meaningful when `kind == .number`: the literal's already-computed value
    /// (K/M/G suffix already folded in).
    num: u64 = 0,
    line: usize,
    col: usize,
};

/// True for the first byte of an identifier (or the leading `.` of a dot-led identifier
/// like `.text`/`__bss_start` - see `Tokenizer.next`'s dot-disambiguation).
fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_' or c == '.' or c == '$';
}

/// True for a byte that continues an already-started identifier. Wider than
/// `isIdentStart`: digits and `-` may continue (but not start) an identifier, matching
/// GNU-ld section-name spellings like `.rodata.str1.1` or `foo-bar`.
fn isIdentCont(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '$';
}

fn hexDigit(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

const Tokenizer = struct {
    source: []const u8,
    i: usize = 0,
    line: usize = 1,
    col: usize = 1,
    diag: ?*Diagnostic,

    fn peekByte(self: *const Tokenizer) ?u8 {
        if (self.i >= self.source.len) return null;
        return self.source[self.i];
    }

    fn peekAt(self: *const Tokenizer, n: usize) ?u8 {
        if (self.i + n >= self.source.len) return null;
        return self.source[self.i + n];
    }

    /// Advances one byte, tracking 1-based line/col (a `\n` rolls the line and resets
    /// col; anything else just bumps col).
    fn advance(self: *Tokenizer) void {
        if (self.i >= self.source.len) return;
        if (self.source[self.i] == '\n') {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        self.i += 1;
    }

    fn err(self: *Tokenizer, line: usize, col: usize, msg: []const u8) ParseError {
        if (self.diag) |d| d.* = .{ .line = line, .col = col, .msg = msg };
        return error.ScriptSyntax;
    }

    /// Skips whitespace, `/* ... */` block comments, and `//` line comments, in any
    /// interleaving, until real token content or EOF.
    fn skipTrivia(self: *Tokenizer) ParseError!void {
        while (true) {
            const c = self.peekByte() orelse return;
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                self.advance();
                continue;
            }
            if (c == '/' and self.peekAt(1) == '*') {
                const line = self.line;
                const col = self.col;
                self.advance();
                self.advance();
                while (true) {
                    const cc = self.peekByte() orelse return self.err(line, col, "unterminated block comment");
                    if (cc == '*' and self.peekAt(1) == '/') {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
                continue;
            }
            if (c == '/' and self.peekAt(1) == '/') {
                while (true) {
                    const cc = self.peekByte() orelse return;
                    if (cc == '\n') break;
                    self.advance();
                }
                continue;
            }
            return;
        }
    }

    /// Scans a decimal or `0x`/`0X` hex integer literal, followed by an optional
    /// `K`/`M`/`G` (case-insensitive) binary-multiplier suffix (*1024 / *1024^2 /
    /// *1024^3). The suffix is only consumed when it is NOT itself followed by another
    /// identifier-continuing byte, so `4Kilo` tokenizes as the number `4` (no suffix)
    /// followed by the identifier `Kilo` rather than silently mis-scanning. Malformed
    /// hex (`0x` with no digits after it) is a syntax error, and so is a literal (with
    /// or without a K/M/G suffix applied) that overflows `u64` - checked arithmetic
    /// throughout, never silent wraparound.
    fn scanNumber(self: *Tokenizer) ParseError!Token {
        const line = self.line;
        const col = self.col;
        const start = self.i;
        var value: u64 = 0;
        if (self.peekByte() == '0' and (self.peekAt(1) == 'x' or self.peekAt(1) == 'X')) {
            self.advance();
            self.advance();
            var digits: usize = 0;
            while (self.peekByte()) |c| {
                const d = hexDigit(c) orelse break;
                value = std.math.mul(u64, value, 16) catch return self.err(line, col, "number literal out of range");
                value = std.math.add(u64, value, d) catch return self.err(line, col, "number literal out of range");
                digits += 1;
                self.advance();
            }
            if (digits == 0) return self.err(line, col, "malformed hex number: no digits after 0x");
        } else {
            while (self.peekByte()) |c| {
                if (!std.ascii.isDigit(c)) break;
                value = std.math.mul(u64, value, 10) catch return self.err(line, col, "number literal out of range");
                value = std.math.add(u64, value, (c - '0')) catch return self.err(line, col, "number literal out of range");
                self.advance();
            }
        }
        if (self.peekByte()) |c| {
            const mult: ?u64 = switch (c) {
                'K', 'k' => 1024,
                'M', 'm' => 1024 * 1024,
                'G', 'g' => 1024 * 1024 * 1024,
                else => null,
            };
            if (mult) |m| {
                const continues = if (self.peekAt(1)) |after| isIdentCont(after) else false;
                if (!continues) {
                    value = std.math.mul(u64, value, m) catch return self.err(line, col, "number literal out of range");
                    self.advance();
                }
            }
        }
        return Token{ .kind = .number, .text = self.source[start..self.i], .num = value, .line = line, .col = col };
    }

    /// Scans an identifier, including one that starts with the location-counter-looking
    /// `.` (e.g. `.text`, `.text.*`, `__bss_start`). The caller (`next`) has already
    /// confirmed `source[i]` is an identifier-start byte.
    fn scanIdent(self: *Tokenizer) Token {
        const line = self.line;
        const col = self.col;
        const start = self.i;
        self.advance();
        while (self.peekByte()) |c| {
            if (!isIdentCont(c)) break;
            self.advance();
        }
        return Token{ .kind = .ident, .text = self.source[start..self.i], .line = line, .col = col };
    }

    /// Scans the next token. Dot-disambiguation: a `.` whose next byte continues an
    /// identifier (e.g. the `t` in `.text`) is scanned as a whole identifier; a `.` that
    /// stands alone (end of input, or followed by whitespace/punctuation) is the
    /// location-counter `dot` token.
    fn next(self: *Tokenizer) ParseError!Token {
        try self.skipTrivia();
        const line = self.line;
        const col = self.col;
        const c = self.peekByte() orelse return Token{ .kind = .eof, .text = "", .line = line, .col = col };
        if (std.ascii.isDigit(c)) return self.scanNumber();
        if (c == '.') {
            const continues = if (self.peekAt(1)) |after| isIdentCont(after) else false;
            if (continues) return self.scanIdent();
            self.advance();
            return Token{ .kind = .dot, .text = ".", .line = line, .col = col };
        }
        if (isIdentStart(c)) return self.scanIdent();
        if (c == '<' and self.peekAt(1) == '<') {
            self.advance();
            self.advance();
            return Token{ .kind = .lshift, .text = "<<", .line = line, .col = col };
        }
        if (c == '>' and self.peekAt(1) == '>') {
            self.advance();
            self.advance();
            return Token{ .kind = .rshift, .text = ">>", .line = line, .col = col };
        }
        const single: ?TokKind = switch (c) {
            '{' => .lbrace,
            '}' => .rbrace,
            '(' => .lparen,
            ')' => .rparen,
            ':' => .colon,
            ';' => .semicolon,
            ',' => .comma,
            '=' => .assign,
            '+' => .plus,
            '-' => .minus,
            '*' => .star,
            '/' => .slash,
            '%' => .percent,
            '&' => .amp,
            '|' => .pipe,
            '<' => .lt,
            '>' => .gt,
            '!' => .bang,
            else => null,
        };
        if (single) |k| {
            self.advance();
            return Token{ .kind = k, .text = self.source[self.i - 1 .. self.i], .line = line, .col = col };
        }
        return self.err(line, col, "unexpected character");
    }
};

// ---------------------------------------------------------------------------------------
// Expression parser (precedence-climbing, low->high: | , & , << >> , + - , * / % , unary -)
// ---------------------------------------------------------------------------------------

const Parser = struct {
    tok: Tokenizer,
    /// One token of lookahead.
    cur: Token,
    /// The `Script` arena's allocator - every node and every duped string comes from
    /// here, so the whole AST is freed by one `arena.deinit()`.
    alloc: std.mem.Allocator,
    diag: ?*Diagnostic,

    fn init(source: []const u8, alloc: std.mem.Allocator, diag: ?*Diagnostic) ParseError!Parser {
        var tok = Tokenizer{ .source = source, .diag = diag };
        const first = try tok.next();
        return Parser{ .tok = tok, .cur = first, .alloc = alloc, .diag = diag };
    }

    fn bump(self: *Parser) ParseError!void {
        self.cur = try self.tok.next();
    }

    fn fail(self: *Parser, msg: []const u8) ParseError {
        if (self.diag) |d| d.* = .{ .line = self.cur.line, .col = self.cur.col, .msg = msg };
        return error.ScriptSyntax;
    }

    fn expect(self: *Parser, kind: TokKind, msg: []const u8) ParseError!Token {
        if (self.cur.kind != kind) return self.fail(msg);
        const t = self.cur;
        try self.bump();
        return t;
    }

    fn box(self: *Parser, e: Expr) ParseError!*const Expr {
        const p = try self.alloc.create(Expr);
        p.* = e;
        return p;
    }

    fn mkBinary(self: *Parser, op: BinOp, lhs: Expr, rhs: Expr) ParseError!Expr {
        return Expr{ .binary = .{ .op = op, .lhs = try self.box(lhs), .rhs = try self.box(rhs) } };
    }

    /// The expression grammar's entry point (lowest precedence: `|`).
    pub fn parseExpr(self: *Parser) ParseError!Expr {
        return self.parseBitOr();
    }

    fn parseBitOr(self: *Parser) ParseError!Expr {
        var lhs = try self.parseBitAnd();
        while (self.cur.kind == .pipe) {
            try self.bump();
            const rhs = try self.parseBitAnd();
            lhs = try self.mkBinary(.bor, lhs, rhs);
        }
        return lhs;
    }

    fn parseBitAnd(self: *Parser) ParseError!Expr {
        var lhs = try self.parseShift();
        while (self.cur.kind == .amp) {
            try self.bump();
            const rhs = try self.parseShift();
            lhs = try self.mkBinary(.band, lhs, rhs);
        }
        return lhs;
    }

    fn parseShift(self: *Parser) ParseError!Expr {
        var lhs = try self.parseAddSub();
        while (self.cur.kind == .lshift or self.cur.kind == .rshift) {
            const op: BinOp = if (self.cur.kind == .lshift) .shl else .shr;
            try self.bump();
            const rhs = try self.parseAddSub();
            lhs = try self.mkBinary(op, lhs, rhs);
        }
        return lhs;
    }

    fn parseAddSub(self: *Parser) ParseError!Expr {
        var lhs = try self.parseMulDiv();
        while (self.cur.kind == .plus or self.cur.kind == .minus) {
            const op: BinOp = if (self.cur.kind == .plus) .add else .sub;
            try self.bump();
            const rhs = try self.parseMulDiv();
            lhs = try self.mkBinary(op, lhs, rhs);
        }
        return lhs;
    }

    fn parseMulDiv(self: *Parser) ParseError!Expr {
        var lhs = try self.parseUnary();
        while (self.cur.kind == .star or self.cur.kind == .slash or self.cur.kind == .percent) {
            const op: BinOp = switch (self.cur.kind) {
                .star => .mul,
                .slash => .div,
                .percent => .mod,
                else => unreachable,
            };
            try self.bump();
            const rhs = try self.parseUnary();
            lhs = try self.mkBinary(op, lhs, rhs);
        }
        return lhs;
    }

    /// Unary `-`. `Expr` has no dedicated unary-negate variant (the AST contract in the
    /// plan lists exactly seven `Expr` shapes), so `-x` lowers to `binary(sub, 0, x)` -
    /// arithmetically identical and stays within that contract.
    fn parseUnary(self: *Parser) ParseError!Expr {
        if (self.cur.kind == .minus) {
            try self.bump();
            const operand = try self.parseUnary();
            return self.mkBinary(.sub, Expr{ .number = 0 }, operand);
        }
        return self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) ParseError!Expr {
        switch (self.cur.kind) {
            .number => {
                const v = self.cur.num;
                try self.bump();
                return Expr{ .number = v };
            },
            .dot => {
                try self.bump();
                return .dot;
            },
            .ident => {
                const text = self.cur.text;
                if (std.mem.eql(u8, text, "ALIGN")) return self.parseAlignCall();
                if (std.mem.eql(u8, text, "ORIGIN")) return self.parseRegionFnCall(.origin);
                if (std.mem.eql(u8, text, "LENGTH")) return self.parseRegionFnCall(.length);
                try self.bump();
                const name = try self.alloc.dupe(u8, text);
                return Expr{ .symbol = name };
            },
            .lparen => {
                try self.bump();
                const inner = try self.parseExpr();
                _ = try self.expect(.rparen, "expected ')' to close parenthesized expression");
                return inner;
            },
            else => return self.fail("expected an expression"),
        }
    }

    /// `ALIGN( expr )` (one arg: value defaults to the location counter, `.`) or
    /// `ALIGN( expr , expr )` (two args: value, then boundary). `self.cur` is the `ALIGN`
    /// identifier token on entry.
    fn parseAlignCall(self: *Parser) ParseError!Expr {
        try self.bump();
        _ = try self.expect(.lparen, "expected '(' after ALIGN");
        const first = try self.parseExpr();
        if (self.cur.kind == .comma) {
            try self.bump();
            const second = try self.parseExpr();
            _ = try self.expect(.rparen, "expected ')' to close ALIGN(...)");
            return Expr{ .align_ = .{ .value = try self.box(first), .boundary = try self.box(second) } };
        }
        _ = try self.expect(.rparen, "expected ')' to close ALIGN(...)");
        return Expr{ .align_ = .{ .value = try self.box(.dot), .boundary = try self.box(first) } };
    }

    /// `ORIGIN( ident )` / `LENGTH( ident )`. `self.cur` is the `ORIGIN`/`LENGTH`
    /// identifier token on entry.
    fn parseRegionFnCall(self: *Parser, comptime which: enum { origin, length }) ParseError!Expr {
        try self.bump();
        _ = try self.expect(.lparen, "expected '(' after ORIGIN/LENGTH");
        const name_tok = try self.expect(.ident, "expected a region name");
        const name = try self.alloc.dupe(u8, name_tok.text);
        _ = try self.expect(.rparen, "expected ')' to close ORIGIN(...)/LENGTH(...)");
        return switch (which) {
            .origin => Expr{ .origin = name },
            .length => Expr{ .length = name },
        };
    }

    // -----------------------------------------------------------------------------------
    // Statement grammar: top-level, MEMORY, SECTIONS, output sections, input specs.
    // -----------------------------------------------------------------------------------

    /// True when `self.cur` is an identifier token spelled exactly `text` (used to
    /// recognize keywords - `ENTRY`, `MEMORY`, `SECTIONS`, `PROVIDE`, `KEEP`, `AT` - which
    /// this grammar has no reserved-word set for; they're just identifiers in keyword
    /// position).
    fn curIsIdent(self: *const Parser, text: []const u8) bool {
        return self.cur.kind == .ident and std.mem.eql(u8, self.cur.text, text);
    }

    /// `ENTRY( ident )`. `self.cur` is the `ENTRY` identifier token on entry. No trailing
    /// `;` in GNU-ld's own grammar, so none is required here.
    fn parseEntry(self: *Parser) ParseError![]const u8 {
        try self.bump();
        _ = try self.expect(.lparen, "expected '(' after ENTRY");
        const name_tok = try self.expect(.ident, "expected a symbol name in ENTRY(...)");
        _ = try self.expect(.rparen, "expected ')' to close ENTRY(...)");
        return try self.alloc.dupe(u8, name_tok.text);
    }

    /// `PROVIDE ( ident = expr ) ;`. `self.cur` is the `PROVIDE` identifier token on
    /// entry. Used both at top level and inside a SECTIONS/output-section body.
    fn parseProvide(self: *Parser) ParseError!Assign {
        try self.bump();
        _ = try self.expect(.lparen, "expected '(' after PROVIDE");
        const name_tok = try self.expect(.ident, "expected a symbol name in PROVIDE(...)");
        const name = try self.alloc.dupe(u8, name_tok.text);
        _ = try self.expect(.assign, "expected '=' in PROVIDE(...)");
        const value = try self.parseExpr();
        _ = try self.expect(.rparen, "expected ')' to close PROVIDE(...)");
        _ = try self.expect(.semicolon, "expected ';' after PROVIDE(...)");
        return Assign{ .name = name, .value = value, .provide = true };
    }

    /// `MEMORY { region* }`. `self.cur` is the `MEMORY` identifier token on entry.
    fn parseMemory(self: *Parser, memory: *std.ArrayList(Region)) ParseError!void {
        try self.bump();
        _ = try self.expect(.lbrace, "expected '{' after MEMORY");
        while (self.cur.kind != .rbrace) {
            if (self.cur.kind == .eof) return self.fail("unexpected end of input inside MEMORY block (missing '}')");
            const region = try self.parseRegion();
            try memory.append(self.alloc, region);
        }
        try self.bump();
    }

    /// Applies one flag-run identifier's characters (e.g. `"rwx"`) to `flags`. Unknown
    /// letters are ignored (MVP: tolerant of GNU-ld flag letters this parser doesn't
    /// otherwise model, e.g. `o`/`d`); `i` and `l` are synonyms mapped to the single
    /// `il` field.
    fn applyRegionFlags(flags: *RegionFlags, text: []const u8) void {
        for (text) |c| {
            switch (c) {
                'r', 'R' => flags.r = true,
                'w', 'W' => flags.w = true,
                'x', 'X' => flags.x = true,
                'a', 'A' => flags.a = true,
                'i', 'I', 'l', 'L' => flags.il = true,
                else => {},
            }
        }
    }

    const RegionField = union(enum) { origin: Expr, length: Expr };

    /// One `ORIGIN = expr` / `LENGTH = expr` field (accepting the `org`/`o`/`len`/`l`
    /// aliases). `self.cur` is the key identifier on entry.
    fn parseRegionField(self: *Parser) ParseError!RegionField {
        if (self.cur.kind != .ident) return self.fail("expected ORIGIN or LENGTH in memory region");
        const text = self.cur.text;
        const is_origin = std.mem.eql(u8, text, "ORIGIN") or std.mem.eql(u8, text, "org") or std.mem.eql(u8, text, "o");
        const is_length = std.mem.eql(u8, text, "LENGTH") or std.mem.eql(u8, text, "len") or std.mem.eql(u8, text, "l");
        if (!is_origin and !is_length) return self.fail("expected ORIGIN or LENGTH in memory region");
        try self.bump();
        _ = try self.expect(.assign, "expected '=' after ORIGIN/LENGTH");
        const e = try self.parseExpr();
        return if (is_origin) RegionField{ .origin = e } else RegionField{ .length = e };
    }

    /// `name (flags)? : ORIGIN = expr , LENGTH = expr`. `self.cur` is the region-name
    /// identifier on entry. The comma between the two fields is optional (GNU ld also
    /// accepts a bare newline there, and this tokenizer treats newlines as trivia
    /// already, so "optional comma" is exactly the right model).
    fn parseRegion(self: *Parser) ParseError!Region {
        const name_tok = try self.expect(.ident, "expected a memory region name");
        const name = try self.alloc.dupe(u8, name_tok.text);
        var flags = RegionFlags{};
        if (self.cur.kind == .lparen) {
            try self.bump();
            while (self.cur.kind != .rparen) {
                switch (self.cur.kind) {
                    .ident => {
                        applyRegionFlags(&flags, self.cur.text);
                        try self.bump();
                    },
                    .bang => try self.bump(),
                    .eof => return self.fail("unexpected end of input in memory region flags (missing ')')"),
                    else => return self.fail("unexpected token in memory region flags"),
                }
            }
            try self.bump();
        }
        _ = try self.expect(.colon, "expected ':' after memory region name/flags");

        var origin: ?Expr = null;
        var length: ?Expr = null;
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            const field = try self.parseRegionField();
            switch (field) {
                .origin => |e| origin = e,
                .length => |e| length = e,
            }
            if (self.cur.kind == .comma) try self.bump();
        }
        if (origin == null) return self.fail("memory region missing ORIGIN");
        if (length == null) return self.fail("memory region missing LENGTH");
        return Region{ .name = name, .flags = flags, .origin = origin.?, .length = length.? };
    }

    /// `SECTIONS { command* }`. `self.cur` is the `SECTIONS` identifier token on entry.
    fn parseSections(self: *Parser, commands: *std.ArrayList(Command)) ParseError!void {
        try self.bump();
        _ = try self.expect(.lbrace, "expected '{' after SECTIONS");
        while (self.cur.kind != .rbrace) {
            if (self.cur.kind == .eof) return self.fail("unexpected end of input inside SECTIONS block (missing '}')");
            const cmd = try self.parseCommand();
            try commands.append(self.alloc, cmd);
        }
        try self.bump();
    }

    /// One statement inside `SECTIONS { }`: `. = expr ;` | `PROVIDE(...)` |
    /// `ident = expr ;` | an output section (`ident` not followed by `=`).
    fn parseCommand(self: *Parser) ParseError!Command {
        if (self.cur.kind == .dot) {
            try self.bump();
            _ = try self.expect(.assign, "expected '=' after '.'");
            const value = try self.parseExpr();
            _ = try self.expect(.semicolon, "expected ';' after location-counter assignment");
            return Command{ .set_dot = value };
        }
        if (self.curIsIdent("PROVIDE")) {
            return Command{ .assign = try self.parseProvide() };
        }
        if (self.cur.kind == .ident) {
            const name_tok = self.cur;
            try self.bump();
            if (self.cur.kind == .assign) {
                try self.bump();
                const value = try self.parseExpr();
                _ = try self.expect(.semicolon, "expected ';' after assignment");
                return Command{ .assign = Assign{ .name = try self.alloc.dupe(u8, name_tok.text), .value = value, .provide = false } };
            }
            return Command{ .output = try self.parseOutputSectionBody(name_tok.text) };
        }
        return self.fail("expected a SECTIONS command");
    }

    /// The rest of an output section after its name has already been consumed:
    /// `[vma_expr] : [AT( expr )] { section_cmd* } [> region] [AT> region]`.
    fn parseOutputSectionBody(self: *Parser, name_text: []const u8) ParseError!OutputSection {
        const name = try self.alloc.dupe(u8, name_text);
        var vma: ?Expr = null;
        if (self.cur.kind != .colon) {
            vma = try self.parseExpr();
        }
        _ = try self.expect(.colon, "expected ':' in output section");

        var lma: ?Lma = null;
        if (self.curIsIdent("AT")) {
            try self.bump();
            _ = try self.expect(.lparen, "expected '(' after AT");
            const addr = try self.parseExpr();
            _ = try self.expect(.rparen, "expected ')' to close AT(...)");
            lma = Lma{ .addr = addr };
        }

        _ = try self.expect(.lbrace, "expected '{' to open output section body");
        var body: std.ArrayList(SectionCmd) = .empty;
        while (self.cur.kind != .rbrace) {
            if (self.cur.kind == .eof) return self.fail("unexpected end of input inside output section body (missing '}')");
            try body.append(self.alloc, try self.parseSectionCmd());
        }
        try self.bump();

        var region: ?[]const u8 = null;
        if (self.cur.kind == .gt) {
            try self.bump();
            const region_tok = try self.expect(.ident, "expected a region name after '>'");
            region = try self.alloc.dupe(u8, region_tok.text);
        }
        if (self.curIsIdent("AT")) {
            try self.bump();
            _ = try self.expect(.gt, "expected '>' after AT (region LMA)");
            const region_tok = try self.expect(.ident, "expected a region name after 'AT>'");
            lma = Lma{ .region = try self.alloc.dupe(u8, region_tok.text) };
        }

        return OutputSection{ .name = name, .vma = vma, .lma = lma, .region = region, .body = try body.toOwnedSlice(self.alloc) };
    }

    /// One statement inside an output section's `{ }`: `. = expr ;` | `PROVIDE(...)` |
    /// `ident = expr ;` | an input spec (`*( ... )` or `KEEP( *( ... ) )`).
    fn parseSectionCmd(self: *Parser) ParseError!SectionCmd {
        if (self.cur.kind == .dot) {
            try self.bump();
            _ = try self.expect(.assign, "expected '=' after '.'");
            const value = try self.parseExpr();
            _ = try self.expect(.semicolon, "expected ';' after location-counter assignment");
            return SectionCmd{ .set_dot = value };
        }
        if (self.curIsIdent("PROVIDE")) {
            return SectionCmd{ .assign = try self.parseProvide() };
        }
        if (self.curIsIdent("KEEP")) {
            try self.bump();
            _ = try self.expect(.lparen, "expected '(' after KEEP");
            const spec = try self.parseInputSpec();
            _ = try self.expect(.rparen, "expected ')' to close KEEP(...)");
            return SectionCmd{ .input = spec };
        }
        if (self.cur.kind == .star) {
            return SectionCmd{ .input = try self.parseInputSpec() };
        }
        if (self.cur.kind == .ident) {
            const name_tok = self.cur;
            try self.bump();
            _ = try self.expect(.assign, "expected '=' in output-section-body assignment");
            const value = try self.parseExpr();
            _ = try self.expect(.semicolon, "expected ';' after assignment");
            return SectionCmd{ .assign = Assign{ .name = try self.alloc.dupe(u8, name_tok.text), .value = value, .provide = false } };
        }
        return self.fail("expected an input-section spec or assignment in output section body");
    }

    /// `*( pattern+ )`. `self.cur` is the `*` token on entry.
    fn parseInputSpec(self: *Parser) ParseError!InputSpec {
        _ = try self.expect(.star, "expected '*' to start an input-section spec");
        _ = try self.expect(.lparen, "expected '(' after '*' in an input-section spec");
        var sections: std.ArrayList([]const u8) = .empty;
        while (self.cur.kind != .rparen) {
            if (self.cur.kind == .eof) return self.fail("unexpected end of input inside an input-section spec (missing ')')");
            try sections.append(self.alloc, try self.parsePattern());
        }
        try self.bump();
        return InputSpec{ .file_glob = "*", .sections = try sections.toOwnedSlice(self.alloc) };
    }

    /// One section-name glob pattern inside `*( ... )`, e.g. `.text`, `COMMON`, or
    /// `.text.*`. The tokenizer scans `.text.*` as `ident(".text.")` + `star("*")` (the
    /// trailing `.` is identifier-continuing but `*` is not - see `Tokenizer.next`), so
    /// this glues any run of ident/dot/star tokens that are byte-adjacent in the source
    /// (no whitespace between them, checked via pointer arithmetic on the token texts,
    /// both slices of the same source buffer) into one pattern string.
    fn parsePattern(self: *Parser) ParseError![]const u8 {
        if (self.cur.kind != .ident and self.cur.kind != .dot and self.cur.kind != .star) {
            return self.fail("expected a section-name pattern");
        }
        var buf: std.ArrayList(u8) = .empty;
        try buf.appendSlice(self.alloc, self.cur.text);
        var prev = self.cur;
        try self.bump();
        while (self.cur.kind == .ident or self.cur.kind == .dot or self.cur.kind == .star) {
            const prev_end = @intFromPtr(prev.text.ptr) + prev.text.len;
            const cur_start = @intFromPtr(self.cur.text.ptr);
            if (prev_end != cur_start) break;
            try buf.appendSlice(self.alloc, self.cur.text);
            prev = self.cur;
            try self.bump();
        }
        return buf.toOwnedSlice(self.alloc);
    }
};

// ---------------------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------------------

/// Parses a full linker script into a `Script` AST: a sequence, in any order, of
/// `ENTRY(ident)`, `MEMORY { region* }`, `SECTIONS { command* }`, a bare
/// `ident = expr ;`, or `PROVIDE ( ident = expr ) ;` - the last two accumulate into
/// `Script.commands` alongside `SECTIONS`'s own commands. There is no separate top-level
/// slot for them; `Command.assign` already fits. An unrecognized top-level token is
/// `error.ScriptSyntax`. PHDRS/OVERLAY/SORT/data-commands (`LONG`/`BYTE`/...) and other
/// GNU-ld constructs outside this supported subset are not recognized keywords here, so
/// they fail closed the same way (the bare-assignment/output-section fallback expects a
/// `(` or `=` next and errors otherwise).
pub fn parse(child_allocator: std.mem.Allocator, source: []const u8, diag: ?*Diagnostic) ParseError!Script {
    var arena = std.heap.ArenaAllocator.init(child_allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var parser = try Parser.init(source, alloc, diag);

    var entry: ?[]const u8 = null;
    var memory: std.ArrayList(Region) = .empty;
    var commands: std.ArrayList(Command) = .empty;

    while (parser.cur.kind != .eof) {
        if (parser.curIsIdent("ENTRY")) {
            entry = try parser.parseEntry();
        } else if (parser.curIsIdent("MEMORY")) {
            try parser.parseMemory(&memory);
        } else if (parser.curIsIdent("SECTIONS")) {
            try parser.parseSections(&commands);
        } else if (parser.curIsIdent("PROVIDE")) {
            const a = try parser.parseProvide();
            try commands.append(alloc, Command{ .assign = a });
        } else if (parser.cur.kind == .ident) {
            const name_tok = parser.cur;
            try parser.bump();
            if (parser.cur.kind != .assign) return parser.fail("unknown top-level directive");
            try parser.bump();
            const value = try parser.parseExpr();
            _ = try parser.expect(.semicolon, "expected ';' after top-level assignment");
            try commands.append(alloc, Command{ .assign = Assign{ .name = try alloc.dupe(u8, name_tok.text), .value = value, .provide = false } });
        } else {
            return parser.fail("unknown top-level directive");
        }
    }

    return Script{
        .arena = arena,
        .entry = entry,
        .memory = try memory.toOwnedSlice(alloc),
        .commands = try commands.toOwnedSlice(alloc),
    };
}

/// Task-1 test entry point: parses a single standalone expression (consuming the whole
/// `source`, trailing tokens other than EOF are a syntax error) and hands it back as a
/// one-command `Script` (`commands[0] = .{ .set_dot = <the expression> }`) so tests can
/// exercise the real arena-owning parse path without needing the full statement grammar.
pub fn parseExprForTest(child_allocator: std.mem.Allocator, source: []const u8, diag: ?*Diagnostic) ParseError!Script {
    var arena = std.heap.ArenaAllocator.init(child_allocator);
    errdefer arena.deinit();
    var parser = try Parser.init(source, arena.allocator(), diag);
    const e = try parser.parseExpr();
    if (parser.cur.kind != .eof) return parser.fail("unexpected trailing tokens after expression");
    const commands = try arena.allocator().alloc(Command, 1);
    commands[0] = Command{ .set_dot = e };
    return Script{ .arena = arena, .entry = null, .memory = &.{}, .commands = commands };
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

/// Structural equality over `Expr` trees (dereferencing the `binary`/`align_` pointers),
/// since the AST can't derive `==` (it holds `*const Expr` fields).
fn exprEql(a: Expr, b: Expr) bool {
    return switch (a) {
        .number => |av| b == .number and av == b.number,
        .dot => b == .dot,
        .symbol => |as| b == .symbol and std.mem.eql(u8, as, b.symbol),
        .binary => |ab| b == .binary and ab.op == b.binary.op and
            exprEql(ab.lhs.*, b.binary.lhs.*) and exprEql(ab.rhs.*, b.binary.rhs.*),
        .align_ => |aa| b == .align_ and
            exprEql(aa.value.*, b.align_.value.*) and exprEql(aa.boundary.*, b.align_.boundary.*),
        .origin => |ao| b == .origin and std.mem.eql(u8, ao, b.origin),
        .length => |al| b == .length and std.mem.eql(u8, al, b.length),
    };
}

fn numberExpr(v: u64) Expr {
    return Expr{ .number = v };
}

test "parseExprForTest: hex plus K-suffixed decimal" {
    var script = try parseExprForTest(std.testing.allocator, "0x1000 + 4K", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.add, e.binary.op);
    try std.testing.expect(exprEql(e.binary.lhs.*, numberExpr(0x1000)));
    try std.testing.expect(exprEql(e.binary.rhs.*, numberExpr(4096)));
}

test "parseExprForTest: M suffix" {
    var script = try parseExprForTest(std.testing.allocator, "1M", null);
    defer script.deinit();
    try std.testing.expect(exprEql(script.commands[0].set_dot, numberExpr(1024 * 1024)));
}

test "parseExprForTest: G suffix" {
    var script = try parseExprForTest(std.testing.allocator, "1G", null);
    defer script.deinit();
    try std.testing.expect(exprEql(script.commands[0].set_dot, numberExpr(1024 * 1024 * 1024)));
}

test "parseExprForTest: lowercase k suffix not swallowed by a longer identifier" {
    var script = try parseExprForTest(std.testing.allocator, "4k", null);
    defer script.deinit();
    try std.testing.expect(exprEql(script.commands[0].set_dot, numberExpr(4096)));
}

test "parseExprForTest: ALIGN with one arg defaults value to dot" {
    var script = try parseExprForTest(std.testing.allocator, "ALIGN(16)", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .align_);
    try std.testing.expect(exprEql(e.align_.value.*, .dot));
    try std.testing.expect(exprEql(e.align_.boundary.*, numberExpr(16)));
}

test "parseExprForTest: ALIGN with two args" {
    var script = try parseExprForTest(std.testing.allocator, "ALIGN(., 16)", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .align_);
    try std.testing.expect(exprEql(e.align_.value.*, .dot));
    try std.testing.expect(exprEql(e.align_.boundary.*, numberExpr(16)));
}

test "parseExprForTest: ORIGIN(region) + LENGTH(region)" {
    var script = try parseExprForTest(std.testing.allocator, "ORIGIN(rom) + LENGTH(rom)", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.add, e.binary.op);
    try std.testing.expect(e.binary.lhs.* == .origin);
    try std.testing.expectEqualStrings("rom", e.binary.lhs.origin);
    try std.testing.expect(e.binary.rhs.* == .length);
    try std.testing.expectEqualStrings("rom", e.binary.rhs.length);
}

test "parseExprForTest: lone dot is the location counter" {
    var script = try parseExprForTest(std.testing.allocator, ".", null);
    defer script.deinit();
    try std.testing.expect(script.commands[0].set_dot == .dot);
}

test "parseExprForTest: dot with trailing space is still the location counter" {
    var script = try parseExprForTest(std.testing.allocator, ". ", null);
    defer script.deinit();
    try std.testing.expect(script.commands[0].set_dot == .dot);
}

test "parseExprForTest: a dot-led identifier is a symbol, not the location counter" {
    var script = try parseExprForTest(std.testing.allocator, ".text.foo", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .symbol);
    try std.testing.expectEqualStrings(".text.foo", e.symbol);
}

test "parseExprForTest: precedence - shift binds tighter than bitor, looser than nothing else here" {
    var script = try parseExprForTest(std.testing.allocator, "a << 2 | 1", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.bor, e.binary.op);
    try std.testing.expect(e.binary.lhs.* == .binary);
    try std.testing.expectEqual(BinOp.shl, e.binary.lhs.binary.op);
    try std.testing.expect(exprEql(e.binary.lhs.binary.lhs.*, Expr{ .symbol = "a" }));
    try std.testing.expect(exprEql(e.binary.lhs.binary.rhs.*, numberExpr(2)));
    try std.testing.expect(exprEql(e.binary.rhs.*, numberExpr(1)));
}

test "parseExprForTest: mul/div binds tighter than add/sub" {
    var script = try parseExprForTest(std.testing.allocator, "1 + 2 * 3", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.add, e.binary.op);
    try std.testing.expect(exprEql(e.binary.lhs.*, numberExpr(1)));
    try std.testing.expect(e.binary.rhs.* == .binary);
    try std.testing.expectEqual(BinOp.mul, e.binary.rhs.binary.op);
}

test "parseExprForTest: parentheses override precedence" {
    var script = try parseExprForTest(std.testing.allocator, "(1 + 2) * 3", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.mul, e.binary.op);
    try std.testing.expect(e.binary.lhs.* == .binary);
    try std.testing.expectEqual(BinOp.add, e.binary.lhs.binary.op);
}

test "parseExprForTest: unary minus lowers to sub-from-zero" {
    var script = try parseExprForTest(std.testing.allocator, "-5", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.sub, e.binary.op);
    try std.testing.expect(exprEql(e.binary.lhs.*, numberExpr(0)));
    try std.testing.expect(exprEql(e.binary.rhs.*, numberExpr(5)));
}

test "parseExprForTest: block and line comments are skipped" {
    var script = try parseExprForTest(std.testing.allocator, "/* c */ 5 // t", null);
    defer script.deinit();
    try std.testing.expect(exprEql(script.commands[0].set_dot, numberExpr(5)));
}

test "parseExprForTest: malformed expression sets diag line/col and errors" {
    var diag: Diagnostic = undefined;
    try std.testing.expectError(error.ScriptSyntax, parseExprForTest(std.testing.allocator, "1 +", &diag));
    try std.testing.expectEqual(@as(usize, 1), diag.line);
    try std.testing.expectEqual(@as(usize, 4), diag.col);
}

test "parseExprForTest: malformed hex literal sets diag and errors" {
    var diag: Diagnostic = undefined;
    try std.testing.expectError(error.ScriptSyntax, parseExprForTest(std.testing.allocator, "0x", &diag));
    try std.testing.expectEqual(@as(usize, 1), diag.line);
    try std.testing.expectEqual(@as(usize, 1), diag.col);
}

test "parseExprForTest: diag line tracks newlines" {
    var diag: Diagnostic = undefined;
    try std.testing.expectError(error.ScriptSyntax, parseExprForTest(std.testing.allocator, "1\n+\n", &diag));
    try std.testing.expectEqual(@as(usize, 3), diag.line);
}

test "parseExprForTest: '&' (band) operator is covered" {
    var script = try parseExprForTest(std.testing.allocator, "0xFF & 0x0F", null);
    defer script.deinit();
    const e = script.commands[0].set_dot;
    try std.testing.expect(e == .binary);
    try std.testing.expectEqual(BinOp.band, e.binary.op);
    try std.testing.expect(exprEql(e.binary.lhs.*, numberExpr(0xFF)));
    try std.testing.expect(exprEql(e.binary.rhs.*, numberExpr(0x0F)));
}

test "parseExprForTest: decimal literal overflowing u64 errors instead of wrapping" {
    var diag: Diagnostic = undefined;
    // 2^64 = 18446744073709551616, one past u64's max.
    try std.testing.expectError(error.ScriptSyntax, parseExprForTest(std.testing.allocator, "18446744073709551616", &diag));
    try std.testing.expectEqualStrings("number literal out of range", diag.msg);
}

test "parseExprForTest: hex literal overflowing u64 errors instead of wrapping" {
    var diag: Diagnostic = undefined;
    try std.testing.expectError(error.ScriptSyntax, parseExprForTest(std.testing.allocator, "0xFFFFFFFFFFFFFFFFF", &diag));
    try std.testing.expectEqualStrings("number literal out of range", diag.msg);
}

test "parseExprForTest: a K-suffixed literal that overflows on the multiply errors" {
    var diag: Diagnostic = undefined;
    // 0xFFFFFFFFFFFFFFFF * 1024 overflows even though the bare digits fit in u64.
    try std.testing.expectError(error.ScriptSyntax, parseExprForTest(std.testing.allocator, "0xFFFFFFFFFFFFFFFFK", &diag));
}

// -----------------------------------------------------------------------------------
// Tests for the top-level + MEMORY + SECTIONS statement grammar.
// -----------------------------------------------------------------------------------

const script_a =
    \\ENTRY(_start)
    \\SECTIONS {
    \\  . = 0x80000000;
    \\  .text : { *(.text .text.*) }
    \\  . = ALIGN(16);
    \\  .rodata : { *(.rodata*) }
    \\  .data : { *(.data*) }
    \\  __bss_start = .;
    \\  .bss : { *(.bss*) *(COMMON) }
    \\  __bss_end = .;
    \\}
;

const script_b =
    \\MEMORY {
    \\  rom (rx) : ORIGIN = 0x08000000, LENGTH = 256K
    \\  ram (rwx): ORIGIN = 0x20000000, LENGTH = 64K
    \\}
    \\SECTIONS {
    \\  .text : { *(.text*) } >rom
    \\  .data : { *(.data*) } >ram AT>rom
    \\}
;

fn expectInputSections(spec: InputSpec, expected: []const []const u8) !void {
    try std.testing.expectEqualStrings("*", spec.file_glob);
    try std.testing.expectEqual(expected.len, spec.sections.len);
    for (expected, spec.sections) |want, got| try std.testing.expectEqualStrings(want, got);
}

test "parse: representative script A - ENTRY + custom VMA + boundary symbols + wildcards" {
    var s = try parse(std.testing.allocator, script_a, null);
    defer s.deinit();

    try std.testing.expect(s.entry != null);
    try std.testing.expectEqualStrings("_start", s.entry.?);
    try std.testing.expectEqual(@as(usize, 0), s.memory.len);
    try std.testing.expectEqual(@as(usize, 8), s.commands.len);

    // . = 0x80000000;
    try std.testing.expect(s.commands[0] == .set_dot);
    try std.testing.expect(exprEql(s.commands[0].set_dot, numberExpr(0x80000000)));

    // .text : { *(.text .text.*) }
    try std.testing.expect(s.commands[1] == .output);
    const text_sec = s.commands[1].output;
    try std.testing.expectEqualStrings(".text", text_sec.name);
    try std.testing.expect(text_sec.vma == null);
    try std.testing.expect(text_sec.lma == null);
    try std.testing.expect(text_sec.region == null);
    try std.testing.expectEqual(@as(usize, 1), text_sec.body.len);
    try std.testing.expect(text_sec.body[0] == .input);
    try expectInputSections(text_sec.body[0].input, &.{ ".text", ".text.*" });

    // . = ALIGN(16);
    try std.testing.expect(s.commands[2] == .set_dot);
    const align_e = s.commands[2].set_dot;
    try std.testing.expect(align_e == .align_);
    try std.testing.expect(exprEql(align_e.align_.value.*, .dot));
    try std.testing.expect(exprEql(align_e.align_.boundary.*, numberExpr(16)));

    // .rodata : { *(.rodata*) }
    try std.testing.expect(s.commands[3] == .output);
    try std.testing.expectEqualStrings(".rodata", s.commands[3].output.name);
    try expectInputSections(s.commands[3].output.body[0].input, &.{".rodata*"});

    // .data : { *(.data*) }
    try std.testing.expect(s.commands[4] == .output);
    try std.testing.expectEqualStrings(".data", s.commands[4].output.name);
    try expectInputSections(s.commands[4].output.body[0].input, &.{".data*"});

    // __bss_start = .;
    try std.testing.expect(s.commands[5] == .assign);
    try std.testing.expectEqualStrings("__bss_start", s.commands[5].assign.name);
    try std.testing.expect(!s.commands[5].assign.provide);
    try std.testing.expect(s.commands[5].assign.value == .dot);

    // .bss : { *(.bss*) *(COMMON) }
    try std.testing.expect(s.commands[6] == .output);
    const bss_sec = s.commands[6].output;
    try std.testing.expectEqualStrings(".bss", bss_sec.name);
    try std.testing.expectEqual(@as(usize, 2), bss_sec.body.len);
    try expectInputSections(bss_sec.body[0].input, &.{".bss*"});
    try expectInputSections(bss_sec.body[1].input, &.{"COMMON"});

    // __bss_end = .;
    try std.testing.expect(s.commands[7] == .assign);
    try std.testing.expectEqualStrings("__bss_end", s.commands[7].assign.name);
    try std.testing.expect(s.commands[7].assign.value == .dot);
}

test "parse: representative script B - MEMORY regions + LMA via AT>" {
    var s = try parse(std.testing.allocator, script_b, null);
    defer s.deinit();

    try std.testing.expect(s.entry == null);
    try std.testing.expectEqual(@as(usize, 2), s.memory.len);

    const rom = s.memory[0];
    try std.testing.expectEqualStrings("rom", rom.name);
    try std.testing.expect(rom.flags.r);
    try std.testing.expect(rom.flags.x);
    try std.testing.expect(!rom.flags.w);
    try std.testing.expect(exprEql(rom.origin, numberExpr(0x08000000)));
    try std.testing.expect(exprEql(rom.length, numberExpr(256 * 1024)));

    const ram = s.memory[1];
    try std.testing.expectEqualStrings("ram", ram.name);
    try std.testing.expect(ram.flags.r);
    try std.testing.expect(ram.flags.w);
    try std.testing.expect(ram.flags.x);
    try std.testing.expect(exprEql(ram.origin, numberExpr(0x20000000)));
    try std.testing.expect(exprEql(ram.length, numberExpr(64 * 1024)));

    try std.testing.expectEqual(@as(usize, 2), s.commands.len);

    try std.testing.expect(s.commands[0] == .output);
    const text_sec = s.commands[0].output;
    try std.testing.expectEqualStrings(".text", text_sec.name);
    try std.testing.expect(text_sec.region != null);
    try std.testing.expectEqualStrings("rom", text_sec.region.?);
    try std.testing.expect(text_sec.lma == null);

    try std.testing.expect(s.commands[1] == .output);
    const data_sec = s.commands[1].output;
    try std.testing.expectEqualStrings(".data", data_sec.name);
    try std.testing.expect(data_sec.region != null);
    try std.testing.expectEqualStrings("ram", data_sec.region.?);
    try std.testing.expect(data_sec.lma != null);
    try std.testing.expect(data_sec.lma.? == .region);
    try std.testing.expectEqualStrings("rom", data_sec.lma.?.region);
}

test "parse: comments interleaved throughout a script are skipped" {
    const src =
        \\// entry point
        \\ENTRY(_start) /* the start symbol */
        \\SECTIONS {
        \\  // set the load address
        \\  . = 0x1000;
        \\  .text : { *(.text) } // code
        \\}
    ;
    var s = try parse(std.testing.allocator, src, null);
    defer s.deinit();
    try std.testing.expectEqualStrings("_start", s.entry.?);
    try std.testing.expectEqual(@as(usize, 2), s.commands.len);
    try std.testing.expect(s.commands[1] == .output);
}

test "parse: top-level PROVIDE accumulates into commands" {
    var s = try parse(std.testing.allocator, "PROVIDE(heap_size = 0x1000);", null);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 1), s.commands.len);
    try std.testing.expect(s.commands[0] == .assign);
    try std.testing.expectEqualStrings("heap_size", s.commands[0].assign.name);
    try std.testing.expect(s.commands[0].assign.provide);
    try std.testing.expect(exprEql(s.commands[0].assign.value, numberExpr(0x1000)));
}

test "parse: top-level bare assignment accumulates into commands" {
    var s = try parse(std.testing.allocator, "HEAP_SIZE = 0x2000;", null);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 1), s.commands.len);
    try std.testing.expect(s.commands[0] == .assign);
    try std.testing.expectEqualStrings("HEAP_SIZE", s.commands[0].assign.name);
    try std.testing.expect(!s.commands[0].assign.provide);
}

test "parse: output section with an explicit VMA and AT(...) LMA" {
    const src = "SECTIONS { .text 0x1000 : AT(0x8000) { *(.text) } }";
    var s = try parse(std.testing.allocator, src, null);
    defer s.deinit();
    const sec = s.commands[0].output;
    try std.testing.expect(sec.vma != null);
    try std.testing.expect(exprEql(sec.vma.?, numberExpr(0x1000)));
    try std.testing.expect(sec.lma != null);
    try std.testing.expect(sec.lma.? == .addr);
    try std.testing.expect(exprEql(sec.lma.?.addr, numberExpr(0x8000)));
}

test "parse: KEEP() unwraps to its inner input spec" {
    const src = "SECTIONS { .init_array : { KEEP(*(.init_array)) } }";
    var s = try parse(std.testing.allocator, src, null);
    defer s.deinit();
    const sec = s.commands[0].output;
    try std.testing.expectEqual(@as(usize, 1), sec.body.len);
    try expectInputSections(sec.body[0].input, &.{".init_array"});
}

test "parse: malformed script - missing closing brace errors with line/col" {
    var diag: Diagnostic = undefined;
    const src = "SECTIONS {\n  .text : { *(.text) }\n";
    try std.testing.expectError(error.ScriptSyntax, parse(std.testing.allocator, src, &diag));
    try std.testing.expect(diag.line >= 1);
}

test "parse: malformed script - bad MEMORY field (missing LENGTH) errors with line/col" {
    var diag: Diagnostic = undefined;
    const src = "MEMORY { rom (rx) : ORIGIN = 0x1000 }";
    try std.testing.expectError(error.ScriptSyntax, parse(std.testing.allocator, src, &diag));
    try std.testing.expect(diag.line >= 1);
    try std.testing.expect(diag.col >= 1);
}

test "parse: malformed script - unknown top-level keyword errors" {
    var diag: Diagnostic = undefined;
    const src = "FOO(bar)";
    try std.testing.expectError(error.ScriptSyntax, parse(std.testing.allocator, src, &diag));
    try std.testing.expect(diag.line >= 1);
}

test "parse: malformed script - unterminated input spec errors" {
    var diag: Diagnostic = undefined;
    const src = "SECTIONS { .text : { *(.text } }";
    try std.testing.expectError(error.ScriptSyntax, parse(std.testing.allocator, src, &diag));
    try std.testing.expect(diag.line >= 1);
}
