//! GLSL lexer: shader source text to a token stream. Identifiers and keywords, integer
//! and floating literals, operators/punctuation, skipping whitespace and `//` and
//! `/* */` comments.

const std = @import("std");

pub const Error = error{ UnexpectedChar, UnterminatedComment } || std.mem.Allocator.Error;

pub const Tag = enum {
    ident,
    int_lit,
    float_lit,
    // keywords
    kw_void,
    kw_float,
    kw_int,
    kw_uint,
    kw_bool,
    kw_return,
    kw_if,
    kw_else,
    kw_for,
    kw_while,
    kw_break,
    kw_continue,
    kw_discard,
    kw_switch,
    kw_case,
    kw_default,
    kw_true,
    kw_false,
    kw_const,
    kw_in,
    kw_out,
    kw_uniform,
    kw_layout,
    // GLSL ES 1.00 storage qualifiers (attribute = a vertex `in`, varying = a
    // VS->FS varying, i.e. an `out` in the VS and an `in` in the FS).
    kw_attribute,
    kw_varying,
    // GLSL ES precision: the `precision` statement + the precision qualifiers
    // (parsed and ignored for codegen).
    kw_precision,
    kw_lowp,
    kw_mediump,
    kw_highp,
    kw_vec2,
    kw_vec3,
    kw_vec4,
    kw_ivec2,
    kw_ivec3,
    kw_ivec4,
    kw_uvec2,
    kw_uvec3,
    kw_uvec4,
    kw_bvec2,
    kw_bvec3,
    kw_bvec4,
    kw_mat2,
    kw_mat3,
    kw_mat4,
    kw_sampler2d,
    kw_samplercube,
    kw_sampler3d,
    kw_sampler2darray,
    kw_sampler2dshadow,
    kw_samplercubeshadow,
    kw_sampler2darrayshadow,
    kw_struct,
    // punctuation / operators
    lparen,
    rparen,
    lbrace,
    rbrace,
    lbracket,
    rbracket,
    semicolon,
    comma,
    dot,
    plus,
    minus,
    star,
    slash,
    percent,
    assign,
    eq,
    neq,
    lt,
    gt,
    le,
    ge,
    bang,
    tilde,
    amp_amp,
    pipe_pipe,
    amp,
    pipe,
    caret,
    lt_lt,
    gt_gt,
    question,
    colon,
    plus_eq,
    minus_eq,
    star_eq,
    slash_eq,
    percent_eq,
    amp_eq,
    pipe_eq,
    caret_eq,
    lt_lt_eq,
    gt_gt_eq,
    plus_plus,
    minus_minus,
    eof,
};

pub const Token = struct {
    tag: Tag,
    text: []const u8, // the source slice (for idents/literals)
    line: u32 = 1, // 1-based source line where the token starts (for debug info)
};

const keywords = std.StaticStringMap(Tag).initComptime(.{
    .{ "void", .kw_void },                       .{ "float", .kw_float },                         .{ "int", .kw_int },
    .{ "uint", .kw_uint },                       .{ "bool", .kw_bool },                           .{ "return", .kw_return },
    .{ "if", .kw_if },                           .{ "else", .kw_else },                           .{ "for", .kw_for },
    .{ "while", .kw_while },                     .{ "true", .kw_true },                           .{ "false", .kw_false },
    .{ "break", .kw_break },                     .{ "continue", .kw_continue },                   .{ "discard", .kw_discard },
    .{ "const", .kw_const },                     .{ "in", .kw_in },                               .{ "out", .kw_out },
    .{ "uniform", .kw_uniform },                 .{ "layout", .kw_layout },                       .{ "vec2", .kw_vec2 },
    .{ "vec3", .kw_vec3 },                       .{ "vec4", .kw_vec4 },                           .{ "mat2", .kw_mat2 },
    .{ "mat3", .kw_mat3 },                       .{ "mat4", .kw_mat4 },                           .{ "sampler2D", .kw_sampler2d },
    // GLSL ES 1.00 additions.
    .{ "attribute", .kw_attribute },             .{ "varying", .kw_varying },                     .{ "precision", .kw_precision },
    .{ "lowp", .kw_lowp },                       .{ "mediump", .kw_mediump },                     .{ "highp", .kw_highp },
    .{ "ivec2", .kw_ivec2 },                     .{ "ivec3", .kw_ivec3 },                         .{ "ivec4", .kw_ivec4 },
    .{ "uvec2", .kw_uvec2 },                     .{ "uvec3", .kw_uvec3 },                         .{ "uvec4", .kw_uvec4 },
    .{ "bvec2", .kw_bvec2 },                     .{ "bvec3", .kw_bvec3 },                         .{ "bvec4", .kw_bvec4 },
    .{ "samplerCube", .kw_samplercube },         .{ "sampler3D", .kw_sampler3d },                 .{ "sampler2DArray", .kw_sampler2darray },
    .{ "sampler2DShadow", .kw_sampler2dshadow }, .{ "samplerCubeShadow", .kw_samplercubeshadow }, .{ "sampler2DArrayShadow", .kw_sampler2darrayshadow },
    .{ "struct", .kw_struct },                   .{ "switch", .kw_switch },                       .{ "case", .kw_case },
    .{ "default", .kw_default },
});

pub const Lexer = struct {
    src: []const u8,
    pos: usize = 0,
    line: u32 = 1, // current 1-based line, advanced as newlines are consumed

    fn peek(self: *const Lexer) u8 {
        return if (self.pos < self.src.len) self.src[self.pos] else 0;
    }
    fn peek2(self: *const Lexer) u8 {
        return if (self.pos + 1 < self.src.len) self.src[self.pos + 1] else 0;
    }

    /// Skip whitespace and comments.
    fn skipTrivia(self: *Lexer) Error!void {
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                if (c == '\n') self.line += 1;
                self.pos += 1;
            } else if (c == '#') {
                // A preprocessor directive (#version / #extension / #ifdef / #define ...):
                // skip to end of line. The supported GLSL ES subset does not act on
                // these (no macro expansion). They are accepted and ignored.
                while (self.pos < self.src.len and self.src[self.pos] != '\n') self.pos += 1;
            } else if (c == '/' and self.peek2() == '/') {
                while (self.pos < self.src.len and self.src[self.pos] != '\n') self.pos += 1;
            } else if (c == '/' and self.peek2() == '*') {
                self.pos += 2;
                while (true) {
                    if (self.pos + 1 >= self.src.len) return error.UnterminatedComment;
                    if (self.src[self.pos] == '*' and self.src[self.pos + 1] == '/') {
                        self.pos += 2;
                        break;
                    }
                    if (self.src[self.pos] == '\n') self.line += 1; // a block comment may span lines
                    self.pos += 1;
                }
            } else break;
        }
    }

    /// Produce the next token, stamped with the 1-based source line where it starts. Tokens
    /// never span a newline, so the line is fixed once trivia is skipped.
    pub fn next(self: *Lexer) Error!Token {
        try self.skipTrivia();
        const line = self.line;
        var tok = try self.nextRaw();
        tok.line = line;
        return tok;
    }

    fn nextRaw(self: *Lexer) Error!Token {
        if (self.pos >= self.src.len) return .{ .tag = .eof, .text = "" };
        const start = self.pos;
        const c = self.src[self.pos];

        if (isIdentStart(c)) {
            while (self.pos < self.src.len and isIdentCont(self.src[self.pos])) self.pos += 1;
            const text = self.src[start..self.pos];
            return .{ .tag = keywords.get(text) orelse .ident, .text = text };
        }
        if (isDigit(c) or (c == '.' and isDigit(self.peek2()))) return self.number();

        self.pos += 1;
        // Three-char shift-assign operators, checked before the two-char forms.
        const three: ?Tag = switch (c) {
            '<' => if (self.peek() == '<' and self.peek2() == '=') .lt_lt_eq else null,
            '>' => if (self.peek() == '>' and self.peek2() == '=') .gt_gt_eq else null,
            else => null,
        };
        if (three) |t| {
            self.pos += 2;
            return .{ .tag = t, .text = self.src[start..self.pos] };
        }
        const two: ?Tag = switch (c) {
            '=' => if (self.peek() == '=') .eq else null,
            '!' => if (self.peek() == '=') .neq else null,
            '<' => if (self.peek() == '<') .lt_lt else if (self.peek() == '=') .le else null,
            '>' => if (self.peek() == '>') .gt_gt else if (self.peek() == '=') .ge else null,
            '&' => if (self.peek() == '&') .amp_amp else if (self.peek() == '=') .amp_eq else null,
            '|' => if (self.peek() == '|') .pipe_pipe else if (self.peek() == '=') .pipe_eq else null,
            '^' => if (self.peek() == '=') .caret_eq else null,
            '%' => if (self.peek() == '=') .percent_eq else null,
            '+' => if (self.peek() == '=') .plus_eq else if (self.peek() == '+') .plus_plus else null,
            '-' => if (self.peek() == '=') .minus_eq else if (self.peek() == '-') .minus_minus else null,
            '*' => if (self.peek() == '=') .star_eq else null,
            '/' => if (self.peek() == '=') .slash_eq else null,
            else => null,
        };
        if (two) |t| {
            self.pos += 1;
            return .{ .tag = t, .text = self.src[start..self.pos] };
        }
        const tag: Tag = switch (c) {
            '(' => .lparen,
            ')' => .rparen,
            '{' => .lbrace,
            '}' => .rbrace,
            '[' => .lbracket,
            ']' => .rbracket,
            ';' => .semicolon,
            ',' => .comma,
            '.' => .dot,
            '+' => .plus,
            '-' => .minus,
            '*' => .star,
            '/' => .slash,
            '%' => .percent,
            '&' => .amp,
            '|' => .pipe,
            '^' => .caret,
            '~' => .tilde,
            '=' => .assign,
            '<' => .lt,
            '>' => .gt,
            '!' => .bang,
            '?' => .question,
            ':' => .colon,
            else => return error.UnexpectedChar,
        };
        return .{ .tag = tag, .text = self.src[start..self.pos] };
    }

    /// A numeric literal: integer, or float if it has a '.' or 'f'/'e' suffix. Integers
    /// may carry a trailing `u`/`U` unsigned suffix, kept in the token text.
    fn number(self: *Lexer) Token {
        const start = self.pos;
        // Hexadecimal integer literal (0x...).
        if (self.peek() == '0' and (self.peek2() == 'x' or self.peek2() == 'X')) {
            self.pos += 2;
            while (self.pos < self.src.len and isHexDigit(self.src[self.pos])) self.pos += 1;
            if (self.peek() == 'u' or self.peek() == 'U') self.pos += 1;
            return .{ .tag = .int_lit, .text = self.src[start..self.pos] };
        }
        var is_float = false;
        while (self.pos < self.src.len and isDigit(self.src[self.pos])) self.pos += 1;
        if (self.peek() == '.') {
            is_float = true;
            self.pos += 1;
            while (self.pos < self.src.len and isDigit(self.src[self.pos])) self.pos += 1;
        }
        if (self.peek() == 'e' or self.peek() == 'E') {
            is_float = true;
            self.pos += 1;
            if (self.peek() == '+' or self.peek() == '-') self.pos += 1;
            while (self.pos < self.src.len and isDigit(self.src[self.pos])) self.pos += 1;
        }
        if (self.peek() == 'f' or self.peek() == 'F') {
            is_float = true;
            self.pos += 1;
        } else if (self.peek() == 'u' or self.peek() == 'U') {
            // Unsigned integer suffix (only valid on integers, never floats).
            self.pos += 1;
        }
        return .{ .tag = if (is_float) .float_lit else .int_lit, .text = self.src[start..self.pos] };
    }
};

fn isDigit(c: u8) bool {
    return std.ascii.isDigit(c);
}
fn isHexDigit(c: u8) bool {
    return std.ascii.isHex(c);
}
fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_';
}
fn isIdentCont(c: u8) bool {
    return isIdentStart(c) or std.ascii.isDigit(c);
}

test "lexes a tiny function" {
    var lx = Lexer{ .src = "float f(float x) { return x * 2.0; }" };
    const expect = [_]Tag{ .kw_float, .ident, .lparen, .kw_float, .ident, .rparen, .lbrace, .kw_return, .ident, .star, .float_lit, .semicolon, .rbrace, .eof };
    for (expect) |tag| try std.testing.expectEqual(tag, (try lx.next()).tag);
}

test "tracks source line numbers across newlines and comments" {
    var lx = Lexer{ .src = "int a;\nfloat b;\n\n/* multi\n line */ bool c;" };
    const a = try lx.next(); // int
    try std.testing.expectEqual(@as(u32, 1), a.line);
    _ = try lx.next(); // a
    _ = try lx.next(); // ;
    const f = try lx.next(); // float, on line 2
    try std.testing.expectEqual(@as(u32, 2), f.line);
    _ = try lx.next(); // b
    _ = try lx.next(); // ;
    // The block comment spans lines 4-5; `bool` follows it on line 5.
    const b = try lx.next(); // bool
    try std.testing.expectEqual(@as(u32, 5), b.line);
}

test "skips comments and lexes operators" {
    var lx = Lexer{ .src = "a + b // line\n/* block */ <= c" };
    const expect = [_]Tag{ .ident, .plus, .ident, .le, .ident, .eof };
    for (expect) |tag| try std.testing.expectEqual(tag, (try lx.next()).tag);
}
