//! GLSL parser: build an AST from the token stream. Function definitions, global
//! interface variables, declarations, assignments, control flow, and
//! precedence-climbing expressions. AST nodes are arena-allocated by the caller.

const std = @import("std");
const lexer = @import("lexer.zig");

const Lexer = lexer.Lexer;
const Tag = lexer.Tag;

pub const Error = lexer.Error || error{ParseError};

/// A GLSL type: scalars, float vectors, and float (square) matrices.
pub const Type = enum { void, float, int, uint, bool, vec2, vec3, vec4, mat2, mat3, mat4, sampler2d };

pub const BinOp = enum { add, sub, mul, div, mod, eq, ne, lt, gt, le, ge, logical_and, logical_or };
pub const UnOp = enum { neg, not };

pub const Expr = union(enum) {
    float_lit: f64,
    int_lit: i64,
    bool_lit: bool,
    ident: []const u8,
    unary: struct { op: UnOp, operand: *Expr },
    binary: struct { op: BinOp, lhs: *Expr, rhs: *Expr },
    call: struct { name: []const u8, args: []const *Expr },
    swizzle: struct { value: *Expr, field: []const u8 },
    ternary: struct { cond: *Expr, then: *Expr, @"else": *Expr },
};

pub const Stmt = union(enum) {
    ret: ?*Expr,
    decl: struct { ty: Type, name: []const u8, value: ?*Expr },
    assign: struct { name: []const u8, value: *Expr },
    expr: *Expr,
    break_,
    continue_,
    discard_,
    swizzle_assign: struct { name: []const u8, field: []const u8, value: *Expr },
    if_: struct { cond: *Expr, then: []const Stmt, @"else": []const Stmt },
    for_: struct { init: []const Stmt, cond: ?*Expr, incr: []const Stmt, body: []const Stmt },
};

pub const Param = struct { ty: Type, name: []const u8 };
pub const Function = struct { ret: Type, name: []const u8, params: []const Param, body: []const Stmt };

/// A storage qualifier on a global variable.
pub const Qualifier = enum { none, in_, out_, uniform };

/// A parsed `layout(...)` qualifier (keys this frontend cares about).
pub const Layout = struct { location: ?u32 = null, local_size: ?[3]u32 = null };

/// A module-scope variable: a shader interface (`in`/`out`/`uniform`) or constant.
pub const GlobalVar = struct { qualifier: Qualifier, location: ?u32, ty: Type, name: []const u8 };

pub const Module = struct {
    globals: []const GlobalVar = &.{},
    functions: []const Function = &.{},
    local_size: ?[3]u32 = null,
};

pub const Parser = struct {
    lx: Lexer,
    cur: lexer.Token,
    arena: std.mem.Allocator,

    pub fn init(arena: std.mem.Allocator, src: []const u8) Error!Parser {
        var lx = Lexer{ .src = src };
        const cur = try lx.next();
        return .{ .lx = lx, .cur = cur, .arena = arena };
    }

    fn advance(self: *Parser) Error!void {
        self.cur = try self.lx.next();
    }
    fn eat(self: *Parser, tag: Tag) Error!lexer.Token {
        if (self.cur.tag != tag) return error.ParseError;
        const t = self.cur;
        try self.advance();
        return t;
    }
    fn accept(self: *Parser, tag: Tag) Error!bool {
        if (self.cur.tag != tag) return false;
        try self.advance();
        return true;
    }

    fn node(self: *Parser, e: Expr) Error!*Expr {
        const p = try self.arena.create(Expr);
        p.* = e;
        return p;
    }

    /// Parse a whole translation unit: global declarations and function definitions.
    pub fn parseModule(self: *Parser) Error!Module {
        var globals: std.ArrayList(GlobalVar) = .empty;
        var funcs: std.ArrayList(Function) = .empty;
        var local_size: ?[3]u32 = null;
        while (self.cur.tag != .eof) {
            const layout = try self.parseOptLayout();
            const qualifier = try self.parseOptQualifier();
            if (layout.local_size) |ls| local_size = ls;
            // A layout-only declaration, e.g. `layout(local_size_x = 16) in`.
            if (qualifier != .none and self.cur.tag == .semicolon) {
                try self.advance();
                continue;
            }
            const ty = try self.parseType();
            const name = (try self.eat(.ident)).text;
            if (self.cur.tag == .lparen and qualifier == .none) {
                try funcs.append(self.arena, try self.parseFunctionRest(ty, name));
            } else {
                _ = try self.eat(.semicolon);
                try globals.append(self.arena, .{ .qualifier = qualifier, .location = layout.location, .ty = ty, .name = name });
            }
        }
        return .{
            .globals = try globals.toOwnedSlice(self.arena),
            .functions = try funcs.toOwnedSlice(self.arena),
            .local_size = local_size,
        };
    }

    fn parseOptQualifier(self: *Parser) Error!Qualifier {
        const q: Qualifier = switch (self.cur.tag) {
            .kw_in => .in_,
            .kw_out => .out_,
            .kw_uniform => .uniform,
            else => return .none,
        };
        try self.advance();
        return q;
    }

    /// Parse an optional `layout(key = value, ...)` qualifier, picking out `location` and
    /// `local_size_x/y/z`. Other keys (binding, std430, ...) are accepted and ignored.
    fn parseOptLayout(self: *Parser) Error!Layout {
        if (self.cur.tag != .kw_layout) return .{};
        try self.advance();
        _ = try self.eat(.lparen);
        var layout: Layout = .{};
        var ls = [3]u32{ 1, 1, 1 };
        var has_ls = false;
        while (self.cur.tag != .rparen) {
            const key = (try self.eat(.ident)).text;
            if (try self.accept(.assign)) {
                const v = std.fmt.parseInt(u32, (try self.eat(.int_lit)).text, 10) catch return error.ParseError;
                if (std.mem.eql(u8, key, "location")) {
                    layout.location = v;
                } else if (std.mem.eql(u8, key, "local_size_x")) {
                    ls[0] = v;
                    has_ls = true;
                } else if (std.mem.eql(u8, key, "local_size_y")) {
                    ls[1] = v;
                    has_ls = true;
                } else if (std.mem.eql(u8, key, "local_size_z")) {
                    ls[2] = v;
                    has_ls = true;
                }
            }
            if (!try self.accept(.comma)) break;
        }
        _ = try self.eat(.rparen);
        if (has_ls) layout.local_size = ls;
        return layout;
    }

    fn parseType(self: *Parser) Error!Type {
        const ty: Type = switch (self.cur.tag) {
            .kw_void => .void,
            .kw_float => .float,
            .kw_int => .int,
            .kw_uint => .uint,
            .kw_bool => .bool,
            .kw_vec2 => .vec2,
            .kw_vec3 => .vec3,
            .kw_vec4 => .vec4,
            .kw_mat2 => .mat2,
            .kw_mat3 => .mat3,
            .kw_mat4 => .mat4,
            .kw_sampler2d => .sampler2d,
            else => return error.ParseError,
        };
        try self.advance();
        return ty;
    }

    /// Parse a function given its already-consumed return type and name (read first to
    /// tell a function from a global variable).
    fn parseFunctionRest(self: *Parser, ret: Type, name: []const u8) Error!Function {
        _ = try self.eat(.lparen);
        var params: std.ArrayList(Param) = .empty;
        if (self.cur.tag != .rparen and self.cur.tag != .kw_void) {
            while (true) {
                _ = try self.accept(.kw_in); // qualifier, ignored for now
                const pty = try self.parseType();
                const pname = (try self.eat(.ident)).text;
                try params.append(self.arena, .{ .ty = pty, .name = pname });
                if (!try self.accept(.comma)) break;
            }
        } else _ = try self.accept(.kw_void);
        _ = try self.eat(.rparen);

        _ = try self.eat(.lbrace);
        var body: std.ArrayList(Stmt) = .empty;
        while (self.cur.tag != .rbrace) try body.append(self.arena, try self.parseStmt());
        _ = try self.eat(.rbrace);
        return .{ .ret = ret, .name = name, .params = try params.toOwnedSlice(self.arena), .body = try body.toOwnedSlice(self.arena) };
    }

    fn parseStmt(self: *Parser) Error!Stmt {
        if (self.cur.tag == .kw_if) return self.parseIf();
        if (self.cur.tag == .kw_for) return self.parseFor();
        if (self.cur.tag == .kw_while) return self.parseWhile();
        if (self.cur.tag == .kw_break) {
            try self.advance();
            _ = try self.eat(.semicolon);
            return .break_;
        }
        if (self.cur.tag == .kw_continue) {
            try self.advance();
            _ = try self.eat(.semicolon);
            return .continue_;
        }
        if (self.cur.tag == .kw_discard) {
            try self.advance();
            _ = try self.eat(.semicolon);
            return .discard_;
        }
        if (self.cur.tag == .kw_return) {
            try self.advance();
            if (try self.accept(.semicolon)) return .{ .ret = null };
            const e = try self.parseExpr();
            _ = try self.eat(.semicolon);
            return .{ .ret = e };
        }
        if (isTypeTag(self.cur.tag)) {
            _ = try self.accept(.kw_const);
            const ty = try self.parseType();
            const name = (try self.eat(.ident)).text;
            var value: ?*Expr = null;
            if (try self.accept(.assign)) value = try self.parseExpr();
            _ = try self.eat(.semicolon);
            return .{ .decl = .{ .ty = ty, .name = name, .value = value } };
        }
        // Assignment (`name = expr`, `name += expr`, `name++`, `name.xy = expr`, ...) or a
        // bare expression.
        if (self.cur.tag == .ident) {
            const save = self.lx;
            const name = self.cur.text;
            try self.advance();
            if (isAssignOp(self.cur.tag)) {
                const stmt = try self.parseAssignRest(name);
                _ = try self.eat(.semicolon);
                return stmt;
            }
            // A swizzle write: `name.field = expr`.
            if (self.cur.tag == .dot) {
                const after_name = self.lx;
                const after_cur = self.cur;
                try self.advance();
                if (self.cur.tag == .ident) {
                    const field = self.cur.text;
                    try self.advance();
                    if (self.cur.tag == .assign) {
                        try self.advance();
                        const value = try self.parseExpr();
                        _ = try self.eat(.semicolon);
                        return .{ .swizzle_assign = .{ .name = name, .field = field, .value = value } };
                    }
                }
                self.lx = after_name;
                self.cur = after_cur;
            }
            self.lx = save;
            self.cur = .{ .tag = .ident, .text = name };
        }
        const e = try self.parseExpr();
        _ = try self.eat(.semicolon);
        return .{ .expr = e };
    }

    /// Parse the rest of an assignment whose target `name` is already consumed: a plain
    /// `= expr`, a compound `+= expr` (desugared to `name = name + expr`), or `++`/`--`
    /// (desugared to `name = name + 1`).
    fn parseAssignRest(self: *Parser, name: []const u8) Error!Stmt {
        const tag = self.cur.tag;
        try self.advance();
        const value = switch (tag) {
            .assign => try self.parseExpr(),
            .plus_eq, .minus_eq, .star_eq, .slash_eq => blk: {
                const op: BinOp = switch (tag) {
                    .plus_eq => .add,
                    .minus_eq => .sub,
                    .star_eq => .mul,
                    else => .div,
                };
                const rhs = try self.parseExpr();
                break :blk try self.node(.{ .binary = .{ .op = op, .lhs = try self.node(.{ .ident = name }), .rhs = rhs } });
            },
            .plus_plus, .minus_minus => blk: {
                const op: BinOp = if (tag == .plus_plus) .add else .sub;
                break :blk try self.node(.{ .binary = .{ .op = op, .lhs = try self.node(.{ .ident = name }), .rhs = try self.node(.{ .int_lit = 1 }) } });
            },
            else => return error.ParseError,
        };
        return .{ .assign = .{ .name = name, .value = value } };
    }

    fn parseIf(self: *Parser) Error!Stmt {
        _ = try self.eat(.kw_if);
        _ = try self.eat(.lparen);
        const cond = try self.parseExpr();
        _ = try self.eat(.rparen);
        const then = try self.parseBlockOrStmt();
        const els: []const Stmt = if (try self.accept(.kw_else)) try self.parseBlockOrStmt() else &.{};
        return .{ .if_ = .{ .cond = cond, .then = then, .@"else" = els } };
    }

    fn parseFor(self: *Parser) Error!Stmt {
        _ = try self.eat(.kw_for);
        _ = try self.eat(.lparen);
        const init_stmts: []const Stmt = if (try self.accept(.semicolon)) &.{} else try self.arena.dupe(Stmt, &.{try self.parseStmt()});
        const cond: ?*Expr = if (self.cur.tag == .semicolon) null else try self.parseExpr();
        _ = try self.eat(.semicolon);
        const incr: []const Stmt = if (self.cur.tag == .rparen) &.{} else try self.arena.dupe(Stmt, &.{try self.parseSimpleStmt()});
        _ = try self.eat(.rparen);
        const body = try self.parseBlockOrStmt();
        return .{ .for_ = .{ .init = init_stmts, .cond = cond, .incr = incr, .body = body } };
    }

    fn parseWhile(self: *Parser) Error!Stmt {
        _ = try self.eat(.kw_while);
        _ = try self.eat(.lparen);
        const cond = try self.parseExpr();
        _ = try self.eat(.rparen);
        const body = try self.parseBlockOrStmt();
        return .{ .for_ = .{ .init = &.{}, .cond = cond, .incr = &.{}, .body = body } };
    }

    /// A statement without a trailing semicolon (the increment of a `for`): an assignment
    /// (`name = expr`, `name += expr`, `name++`, ...) or a bare expression.
    fn parseSimpleStmt(self: *Parser) Error!Stmt {
        if (self.cur.tag == .ident) {
            const save = self.lx;
            const name = self.cur.text;
            try self.advance();
            if (isAssignOp(self.cur.tag)) return self.parseAssignRest(name);
            self.lx = save;
            self.cur = .{ .tag = .ident, .text = name };
        }
        return .{ .expr = try self.parseExpr() };
    }

    /// A `{ ... }` block, or a single statement (returned as a one-element list).
    fn parseBlockOrStmt(self: *Parser) Error![]const Stmt {
        if (try self.accept(.lbrace)) {
            var list: std.ArrayList(Stmt) = .empty;
            while (self.cur.tag != .rbrace) try list.append(self.arena, try self.parseStmt());
            _ = try self.eat(.rbrace);
            return list.toOwnedSlice(self.arena);
        }
        return self.arena.dupe(Stmt, &.{try self.parseStmt()});
    }

    // Precedence-climbing: ternary < logical-or < compare < additive < multiplicative
    // < unary. The ternary `c ? a : b` sits above the binary operators and is
    // right-associative.
    fn parseExpr(self: *Parser) Error!*Expr {
        const cond = try self.parseBinary(0);
        if (self.cur.tag != .question) return cond;
        try self.advance();
        const then = try self.parseExpr();
        _ = try self.eat(.colon);
        const els = try self.parseExpr();
        return self.node(.{ .ternary = .{ .cond = cond, .then = then, .@"else" = els } });
    }

    fn parseBinary(self: *Parser, min_prec: u8) Error!*Expr {
        var lhs = try self.parseUnary();
        while (binPrec(self.cur.tag)) |info| {
            if (info.prec < min_prec) break;
            const op = info.op;
            try self.advance();
            const rhs = try self.parseBinary(info.prec + 1);
            lhs = try self.node(.{ .binary = .{ .op = op, .lhs = lhs, .rhs = rhs } });
        }
        return lhs;
    }

    fn parseUnary(self: *Parser) Error!*Expr {
        if (self.cur.tag == .minus) {
            try self.advance();
            return self.node(.{ .unary = .{ .op = .neg, .operand = try self.parseUnary() } });
        }
        if (self.cur.tag == .bang) {
            try self.advance();
            return self.node(.{ .unary = .{ .op = .not, .operand = try self.parseUnary() } });
        }
        return self.parsePostfix();
    }

    /// A primary followed by zero or more `.field` swizzles (`v.xyz`, `p.x`).
    fn parsePostfix(self: *Parser) Error!*Expr {
        var e = try self.parsePrimary();
        while (self.cur.tag == .dot) {
            try self.advance();
            const field = (try self.eat(.ident)).text;
            e = try self.node(.{ .swizzle = .{ .value = e, .field = field } });
        }
        return e;
    }

    fn parsePrimary(self: *Parser) Error!*Expr {
        switch (self.cur.tag) {
            .float_lit => {
                const v = std.fmt.parseFloat(f64, trimFloatSuffix(self.cur.text)) catch return error.ParseError;
                try self.advance();
                return self.node(.{ .float_lit = v });
            },
            .int_lit => {
                const v = std.fmt.parseInt(i64, self.cur.text, 10) catch return error.ParseError;
                try self.advance();
                return self.node(.{ .int_lit = v });
            },
            .kw_true, .kw_false => {
                const v = (self.cur.tag == .kw_true);
                try self.advance();
                return self.node(.{ .bool_lit = v });
            },
            .lparen => {
                try self.advance();
                const e = try self.parseExpr();
                _ = try self.eat(.rparen);
                return e;
            },
            .ident, .kw_float, .kw_int, .kw_uint, .kw_bool, .kw_vec2, .kw_vec3, .kw_vec4, .kw_mat2, .kw_mat3, .kw_mat4 => {
                // An identifier, or a call / type-constructor `name(args...)`.
                const name = self.cur.text;
                try self.advance();
                if (try self.accept(.lparen)) {
                    var args: std.ArrayList(*Expr) = .empty;
                    if (self.cur.tag != .rparen) {
                        while (true) {
                            try args.append(self.arena, try self.parseExpr());
                            if (!try self.accept(.comma)) break;
                        }
                    }
                    _ = try self.eat(.rparen);
                    return self.node(.{ .call = .{ .name = name, .args = try args.toOwnedSlice(self.arena) } });
                }
                return self.node(.{ .ident = name });
            },
            else => return error.ParseError,
        }
    }
};

const PrecInfo = struct { op: BinOp, prec: u8 };
fn binPrec(tag: Tag) ?PrecInfo {
    return switch (tag) {
        .pipe_pipe => .{ .op = .logical_or, .prec = 1 },
        .amp_amp => .{ .op = .logical_and, .prec = 2 },
        .eq => .{ .op = .eq, .prec = 3 },
        .neq => .{ .op = .ne, .prec = 3 },
        .lt => .{ .op = .lt, .prec = 4 },
        .gt => .{ .op = .gt, .prec = 4 },
        .le => .{ .op = .le, .prec = 4 },
        .ge => .{ .op = .ge, .prec = 4 },
        .plus => .{ .op = .add, .prec = 5 },
        .minus => .{ .op = .sub, .prec = 5 },
        .star => .{ .op = .mul, .prec = 6 },
        .slash => .{ .op = .div, .prec = 6 },
        .percent => .{ .op = .mod, .prec = 6 },
        else => null,
    };
}

fn isAssignOp(tag: Tag) bool {
    return switch (tag) {
        .assign, .plus_eq, .minus_eq, .star_eq, .slash_eq, .plus_plus, .minus_minus => true,
        else => false,
    };
}

fn isTypeTag(tag: Tag) bool {
    return switch (tag) {
        .kw_float, .kw_int, .kw_uint, .kw_bool, .kw_const, .kw_vec2, .kw_vec3, .kw_vec4, .kw_mat2, .kw_mat3, .kw_mat4, .kw_sampler2d => true,
        else => false,
    };
}

fn trimFloatSuffix(s: []const u8) []const u8 {
    return if (s.len > 0 and (s[s.len - 1] == 'f' or s[s.len - 1] == 'F')) s[0 .. s.len - 1] else s;
}

test "parses a scalar function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p = try Parser.init(arena.allocator(), "float f(float x) { return x * 2.0 + 1.0; }");
    const m = try p.parseModule();
    try std.testing.expectEqual(@as(usize, 1), m.functions.len);
    const f = m.functions[0];
    try std.testing.expectEqualStrings("f", f.name);
    try std.testing.expectEqual(Type.float, f.ret);
    try std.testing.expectEqual(@as(usize, 1), f.params.len);
    try std.testing.expect(f.body[0] == .ret);
    // x * 2.0 + 1.0 parses as (x * 2.0) + 1.0
    try std.testing.expect(f.body[0].ret.?.binary.op == .add);
}
