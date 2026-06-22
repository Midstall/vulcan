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
    kw_true,
    kw_false,
    kw_const,
    kw_in,
    kw_out,
    kw_uniform,
    kw_layout,
    kw_vec2,
    kw_vec3,
    kw_vec4,
    kw_mat2,
    kw_mat3,
    kw_mat4,
    kw_sampler2d,
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
    amp_amp,
    pipe_pipe,
    question,
    colon,
    plus_eq,
    minus_eq,
    star_eq,
    slash_eq,
    plus_plus,
    minus_minus,
    eof,
};

pub const Token = struct {
    tag: Tag,
    text: []const u8, // the source slice (for idents/literals)
};

const keywords = std.StaticStringMap(Tag).initComptime(.{
    .{ "void", .kw_void },       .{ "float", .kw_float },       .{ "int", .kw_int },
    .{ "uint", .kw_uint },       .{ "bool", .kw_bool },         .{ "return", .kw_return },
    .{ "if", .kw_if },           .{ "else", .kw_else },         .{ "for", .kw_for },
    .{ "while", .kw_while },     .{ "true", .kw_true },         .{ "false", .kw_false },
    .{ "break", .kw_break },     .{ "continue", .kw_continue }, .{ "discard", .kw_discard },
    .{ "const", .kw_const },     .{ "in", .kw_in },             .{ "out", .kw_out },
    .{ "uniform", .kw_uniform }, .{ "layout", .kw_layout },     .{ "vec2", .kw_vec2 },
    .{ "vec3", .kw_vec3 },       .{ "vec4", .kw_vec4 },         .{ "mat2", .kw_mat2 },
    .{ "mat3", .kw_mat3 },       .{ "mat4", .kw_mat4 },         .{ "sampler2D", .kw_sampler2d },
});

pub const Lexer = struct {
    src: []const u8,
    pos: usize = 0,

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
                self.pos += 1;
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
                    self.pos += 1;
                }
            } else break;
        }
    }

    pub fn next(self: *Lexer) Error!Token {
        try self.skipTrivia();
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
        const two: ?Tag = switch (c) {
            '=' => if (self.peek() == '=') .eq else null,
            '!' => if (self.peek() == '=') .neq else null,
            '<' => if (self.peek() == '=') .le else null,
            '>' => if (self.peek() == '=') .ge else null,
            '&' => if (self.peek() == '&') .amp_amp else null,
            '|' => if (self.peek() == '|') .pipe_pipe else null,
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

    /// A numeric literal: integer, or float if it has a '.' or 'f'/'e' suffix.
    fn number(self: *Lexer) Token {
        const start = self.pos;
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
        }
        return .{ .tag = if (is_float) .float_lit else .int_lit, .text = self.src[start..self.pos] };
    }
};

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}
fn isIdentStart(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}
fn isIdentCont(c: u8) bool {
    return isIdentStart(c) or isDigit(c);
}

test "lexes a tiny function" {
    var lx = Lexer{ .src = "float f(float x) { return x * 2.0; }" };
    const expect = [_]Tag{ .kw_float, .ident, .lparen, .kw_float, .ident, .rparen, .lbrace, .kw_return, .ident, .star, .float_lit, .semicolon, .rbrace, .eof };
    for (expect) |tag| try std.testing.expectEqual(tag, (try lx.next()).tag);
}

test "skips comments and lexes operators" {
    var lx = Lexer{ .src = "a + b // line\n/* block */ <= c" };
    const expect = [_]Tag{ .ident, .plus, .ident, .le, .ident, .eof };
    for (expect) |tag| try std.testing.expectEqual(tag, (try lx.next()).tag);
}
