//! GLSL parser: build an AST from the token stream. Function definitions, global
//! interface variables, declarations, assignments, control flow, and
//! precedence-climbing expressions. AST nodes are arena-allocated by the caller.

const std = @import("std");
const lexer = @import("lexer.zig");

const Lexer = lexer.Lexer;
const Tag = lexer.Tag;

pub const Error = lexer.Error || error{ ParseError, Unsupported };

/// A GLSL type: scalars, float/int/bool vectors, and float (square) matrices.
pub const Type = enum { void, float, int, uint, bool, vec2, vec3, vec4, ivec2, ivec3, ivec4, uvec2, uvec3, uvec4, bvec2, bvec3, bvec4, mat2, mat3, mat4, sampler2d, sampler_cube, sampler3d, sampler2darray, sampler2dshadow, samplercubeshadow, sampler2darrayshadow };

pub const BinOp = enum { add, sub, mul, div, mod, eq, ne, lt, gt, le, ge, logical_and, logical_or, bit_and, bit_or, bit_xor, shl, shr };
pub const UnOp = enum { neg, not, bit_not };

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
    /// `value[index]` array subscript (lowering resolves a constant-foldable index).
    index: struct { value: *Expr, index: *Expr },
    /// A user-struct constructor `TypeName(args...)` (the type name is a declared struct).
    struct_ctor: struct { name: []const u8, args: []const *Expr },
};

/// A statement plus the 1-based source line it starts on (threaded to IR for debug info).
pub const Stmt = struct { kind: StmtKind, line: u32 = 0 };

/// Wrap a statement kind with the source line it starts on.
fn stmt(line: u32, kind: StmtKind) Stmt {
    return .{ .kind = kind, .line = line };
}

pub const StmtKind = union(enum) {
    ret: ?*Expr,
    decl: struct { ty: Type, name: []const u8, value: ?*Expr, array_len: ?u32 = null, struct_name: ?[]const u8 = null },
    assign: struct { name: []const u8, value: *Expr },
    expr: *Expr,
    break_,
    continue_,
    discard_,
    swizzle_assign: struct { name: []const u8, field: []const u8, value: *Expr },
    /// A store to a complex lvalue: `a[i] = v`, `a[i].field = v`, `s.field = v` (struct
    /// member), `a[i].field.xy = v`. The target is an lvalue Expr (ident / index / swizzle
    /// chain) resolved during lowering.
    store: struct { target: *Expr, value: *Expr },
    if_: struct { cond: *Expr, then: []const Stmt, @"else": []const Stmt },
    for_: struct { init: []const Stmt, cond: ?*Expr, incr: []const Stmt, body: []const Stmt },
};

pub const Param = struct { ty: Type, name: []const u8 };
pub const Function = struct { ret: Type, name: []const u8, params: []const Param, body: []const Stmt };

/// A storage qualifier on a global variable. `varying` is stage-dependent (an `out` in
/// a vertex shader, an `in` in a fragment shader). The lowering resolves it from the
/// compile stage. `attribute` is a GLSL ES 1.00 vertex input (== `in`).
pub const Qualifier = enum { none, in_, out_, uniform, attribute, varying, const_ };

/// A parsed `layout(...)` qualifier (keys this frontend cares about).
pub const Layout = struct { location: ?u32 = null, local_size: ?[3]u32 = null, binding: ?u32 = null };

/// A named uniform interface block (`layout(std140, binding=N) uniform BlockName { ... };`).
/// Its members are appended to `Module.globals` as ordinary `uniform` globals (so the
/// lowering lays them out in the default block exactly like loose uniforms), and this record
/// captures the block's name, its (optional) explicit binding, and its member names in order.
/// A named block is a shared UBO the app fills with glBufferData + binds with glBindBufferBase,
/// so the GLES layer routes the app's buffer to the block instead of glUniform* storage.
pub const UniformBlockDef = struct {
    name: []const u8,
    instance_name: ?[]const u8 = null,
    binding: ?u32 = null,
    member_names: []const []const u8,
};

/// One field of a user-declared `struct`. `struct_name` is set when the field's type is
/// itself a named struct (nested structs), else `ty` is the scalar/vector/matrix type.
pub const StructField = struct { ty: Type, name: []const u8, struct_name: ?[]const u8 = null, array_len: ?u32 = null };

/// A user-declared `struct Name { ... };`.
pub const StructDef = struct { name: []const u8, fields: []const StructField };

/// A module-scope variable: a shader interface (`in`/`out`/`uniform`) or constant.
/// `array_len` is set for an array global (`uniform vec3 c[3];`). `struct_name` for a
/// global whose type is a user struct.
pub const GlobalVar = struct { qualifier: Qualifier, location: ?u32, ty: Type, name: []const u8, init: ?*Expr = null, array_len: ?u32 = null, struct_name: ?[]const u8 = null };

pub const Module = struct {
    globals: []const GlobalVar = &.{},
    functions: []const Function = &.{},
    structs: []const StructDef = &.{},
    uniform_blocks: []const UniformBlockDef = &.{},
    local_size: ?[3]u32 = null,
};

pub const Parser = struct {
    lx: Lexer,
    cur: lexer.Token,
    arena: std.mem.Allocator,
    /// Names of structs declared so far, so the parser can recognize `TypeName x;` and
    /// `TypeName(...)` constructor calls (an identifier that names a declared struct).
    struct_names: std.ArrayList([]const u8) = .empty,

    pub fn init(arena: std.mem.Allocator, src: []const u8) Error!Parser {
        var lx = Lexer{ .src = src };
        const cur = try lx.next();
        return .{ .lx = lx, .cur = cur, .arena = arena };
    }

    fn isStructName(self: *const Parser, name: []const u8) bool {
        for (self.struct_names.items) |s| if (std.mem.eql(u8, s, name)) return true;
        return false;
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
        var structs: std.ArrayList(StructDef) = .empty;
        var uniform_blocks: std.ArrayList(UniformBlockDef) = .empty;
        var local_size: ?[3]u32 = null;
        while (self.cur.tag != .eof) {
            // `precision <prec> <type>;` - a default-precision declaration (GLSL ES).
            // Parsed and ignored for codegen.
            if (self.cur.tag == .kw_precision) {
                try self.advance();
                _ = try self.skipPrecisionQualifier();
                _ = try self.parseType();
                _ = try self.eat(.semicolon);
                continue;
            }
            // A struct definition `struct Name { fields... };` (the trailing `;` may instead
            // begin a variable, e.g. `struct S {..} s;`, which we also accept).
            if (self.cur.tag == .kw_struct) {
                const def = try self.parseStructDef();
                try self.struct_names.append(self.arena, def.name);
                try structs.append(self.arena, def);
                // An optional variable declared right after the struct body.
                if (self.cur.tag == .ident) {
                    const vname = (try self.eat(.ident)).text;
                    const alen = try self.parseOptArraySize();
                    _ = try self.eat(.semicolon);
                    try globals.append(self.arena, .{ .qualifier = .none, .location = null, .ty = .void, .name = vname, .array_len = alen, .struct_name = def.name });
                } else {
                    _ = try self.eat(.semicolon);
                }
                continue;
            }
            const layout = try self.parseOptLayout();
            const qualifier = try self.parseOptQualifier();
            // A bare global may carry a leading precision qualifier (`mediump float u;`).
            _ = try self.skipPrecisionQualifier();
            if (layout.local_size) |ls| local_size = ls;
            // A layout-only declaration, e.g. `layout(local_size_x = 16) in`.
            if (qualifier != .none and self.cur.tag == .semicolon) {
                try self.advance();
                continue;
            }
            // A global whose type is a user-declared struct (`LightSource sources[3];`).
            if (self.cur.tag == .ident and self.isStructName(self.cur.text)) {
                const sname = (try self.eat(.ident)).text;
                const name = (try self.eat(.ident)).text;
                const alen = try self.parseOptArraySize();
                _ = try self.eat(.semicolon);
                try globals.append(self.arena, .{ .qualifier = qualifier, .location = layout.location, .ty = .void, .name = name, .array_len = alen, .struct_name = sname });
                continue;
            }
            // A named uniform interface block: `uniform BlockName { members... } [inst];`. The
            // block name is a fresh identifier (not a type keyword, not a struct name) followed
            // by `{`. Its members are emitted as ordinary `uniform` globals (default-block
            // layout), and a UniformBlockDef records the block for the GLES layer's UBO routing.
            if (qualifier == .uniform and self.cur.tag == .ident) {
                const saved_lx = self.lx;
                const saved_cur = self.cur;
                const bname = (try self.eat(.ident)).text;
                if (self.cur.tag == .lbrace) {
                    try self.parseUniformBlock(bname, layout, &globals, &uniform_blocks);
                    continue;
                }
                // Not a block (e.g. a stray identifier): rewind and let normal parsing error.
                self.lx = saved_lx;
                self.cur = saved_cur;
            }
            const ty = try self.parseType();
            const name = (try self.eat(.ident)).text;
            if (self.cur.tag == .lparen and qualifier == .none) {
                try funcs.append(self.arena, try self.parseFunctionRest(ty, name));
            } else {
                // An optional array size after the name (`uniform vec3 c[3];`).
                const alen = try self.parseOptArraySize();
                // A `const` (or any) global may carry a `= <expr>` initializer.
                var init_expr: ?*Expr = null;
                if (try self.accept(.assign)) init_expr = try self.parseExpr();
                _ = try self.eat(.semicolon);
                try globals.append(self.arena, .{ .qualifier = qualifier, .location = layout.location, .ty = ty, .name = name, .init = init_expr, .array_len = alen });
            }
        }
        return .{
            .globals = try globals.toOwnedSlice(self.arena),
            .functions = try funcs.toOwnedSlice(self.arena),
            .structs = try structs.toOwnedSlice(self.arena),
            .uniform_blocks = try uniform_blocks.toOwnedSlice(self.arena),
            .local_size = local_size,
        };
    }

    /// Parse the body of a named uniform interface block after its name (`{ members } [inst];`,
    /// the block name already consumed). Each member becomes a `uniform` global appended to
    /// `globals` (so the lowering lays it out in the default block); the block itself is
    /// recorded in `blocks` with its name, binding, and member names. A member's type may be a
    /// scalar/vector/matrix, with an optional array size (`vec3 c[4];`).
    fn parseUniformBlock(
        self: *Parser,
        bname: []const u8,
        layout: Layout,
        globals: *std.ArrayList(GlobalVar),
        blocks: *std.ArrayList(UniformBlockDef),
    ) Error!void {
        _ = try self.eat(.lbrace);
        var member_names: std.ArrayList([]const u8) = .empty;
        while (self.cur.tag != .rbrace) {
            _ = try self.skipPrecisionQualifier();
            const fty = try self.parseType();
            const fname = (try self.eat(.ident)).text;
            const alen = try self.parseOptArraySize();
            _ = try self.eat(.semicolon);
            try globals.append(self.arena, .{ .qualifier = .uniform, .location = null, .ty = fty, .name = fname, .array_len = alen });
            try member_names.append(self.arena, fname);
        }
        _ = try self.eat(.rbrace);
        // An optional instance name (`... } Camera;`). The minimal path references members by
        // their bare name; an instance name is parsed and recorded but member access is by name.
        var instance_name: ?[]const u8 = null;
        if (self.cur.tag == .ident) instance_name = (try self.eat(.ident)).text;
        _ = try self.eat(.semicolon);
        try blocks.append(self.arena, .{
            .name = bname,
            .instance_name = instance_name,
            .binding = layout.binding,
            .member_names = try member_names.toOwnedSlice(self.arena),
        });
    }

    /// Parse `struct Name { <field>; <field>; ... }` (the leading `struct` already current).
    /// Does not consume the trailing `;` or a following variable name.
    fn parseStructDef(self: *Parser) Error!StructDef {
        _ = try self.eat(.kw_struct);
        const name = (try self.eat(.ident)).text;
        _ = try self.eat(.lbrace);
        var fields: std.ArrayList(StructField) = .empty;
        while (self.cur.tag != .rbrace) {
            _ = try self.skipPrecisionQualifier();
            // A field whose type is itself a named struct.
            if (self.cur.tag == .ident and self.isStructName(self.cur.text)) {
                const sname = (try self.eat(.ident)).text;
                const fname = (try self.eat(.ident)).text;
                const alen = try self.parseOptArraySize();
                _ = try self.eat(.semicolon);
                try fields.append(self.arena, .{ .ty = .void, .name = fname, .struct_name = sname, .array_len = alen });
                continue;
            }
            const fty = try self.parseType();
            const fname = (try self.eat(.ident)).text;
            const alen = try self.parseOptArraySize();
            _ = try self.eat(.semicolon);
            try fields.append(self.arena, .{ .ty = fty, .name = fname, .array_len = alen });
        }
        _ = try self.eat(.rbrace);
        return .{ .name = name, .fields = try fields.toOwnedSlice(self.arena) };
    }

    /// Parse an optional `[ <int-literal> ]` array size after a declarator name. Returns
    /// the length, or null if no bracket follows.
    fn parseOptArraySize(self: *Parser) Error!?u32 {
        if (!try self.accept(.lbracket)) return null;
        const v = parseIntLit(u32, (try self.eat(.int_lit)).text) catch return error.ParseError;
        _ = try self.eat(.rbracket);
        return v;
    }

    fn parseOptQualifier(self: *Parser) Error!Qualifier {
        const q: Qualifier = switch (self.cur.tag) {
            .kw_in => .in_,
            .kw_out => .out_,
            .kw_uniform => .uniform,
            .kw_attribute => .attribute,
            .kw_varying => .varying,
            .kw_const => .const_,
            else => return .none,
        };
        try self.advance();
        // An optional precision qualifier may follow a storage qualifier
        // (`uniform mediump float u;`, `varying lowp vec3 v;`). Skip it.
        _ = try self.skipPrecisionQualifier();
        return q;
    }

    /// Skip an optional precision qualifier (`lowp`/`mediump`/`highp`). Returns true if
    /// one was consumed. Precision is parsed and ignored for codegen.
    fn skipPrecisionQualifier(self: *Parser) Error!bool {
        switch (self.cur.tag) {
            .kw_lowp, .kw_mediump, .kw_highp => {
                try self.advance();
                return true;
            },
            else => return false,
        }
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
                const v = parseIntLit(u32, (try self.eat(.int_lit)).text) catch return error.ParseError;
                if (std.mem.eql(u8, key, "location")) {
                    layout.location = v;
                } else if (std.mem.eql(u8, key, "binding")) {
                    layout.binding = v;
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
            .kw_ivec2 => .ivec2,
            .kw_ivec3 => .ivec3,
            .kw_ivec4 => .ivec4,
            .kw_uvec2 => .uvec2,
            .kw_uvec3 => .uvec3,
            .kw_uvec4 => .uvec4,
            .kw_bvec2 => .bvec2,
            .kw_bvec3 => .bvec3,
            .kw_bvec4 => .bvec4,
            .kw_mat2 => .mat2,
            .kw_mat3 => .mat3,
            .kw_mat4 => .mat4,
            .kw_sampler2d => .sampler2d,
            .kw_samplercube => .sampler_cube,
            .kw_sampler3d => .sampler3d,
            .kw_sampler2darray => .sampler2darray,
            .kw_sampler2dshadow => .sampler2dshadow,
            .kw_samplercubeshadow => .samplercubeshadow,
            .kw_sampler2darrayshadow => .sampler2darrayshadow,
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
                _ = try self.accept(.kw_const);
                _ = try self.accept(.kw_in); // in/out/inout qualifier (ignored: by-value)
                _ = try self.accept(.kw_out);
                _ = try self.skipPrecisionQualifier();
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
        const line = self.cur.line;
        if (self.cur.tag == .kw_if) return self.parseIf();
        if (self.cur.tag == .kw_for) return self.parseFor();
        if (self.cur.tag == .kw_while) return self.parseWhile();
        if (self.cur.tag == .kw_switch) return self.parseSwitch();
        if (self.cur.tag == .kw_break) {
            try self.advance();
            _ = try self.eat(.semicolon);
            return stmt(line, .break_);
        }
        if (self.cur.tag == .kw_continue) {
            try self.advance();
            _ = try self.eat(.semicolon);
            return stmt(line, .continue_);
        }
        if (self.cur.tag == .kw_discard) {
            try self.advance();
            _ = try self.eat(.semicolon);
            return stmt(line, .discard_);
        }
        if (self.cur.tag == .kw_return) {
            try self.advance();
            if (try self.accept(.semicolon)) return stmt(line, .{ .ret = null });
            const e = try self.parseExpr();
            _ = try self.eat(.semicolon);
            return stmt(line, .{ .ret = e });
        }
        // A local declaration whose type is a user struct (`LightSource s;`, `S a[3];`).
        if (self.cur.tag == .ident and self.isStructName(self.cur.text)) {
            const sname = (try self.eat(.ident)).text;
            const name = (try self.eat(.ident)).text;
            const alen = try self.parseOptArraySize();
            var value: ?*Expr = null;
            if (try self.accept(.assign)) value = try self.parseExpr();
            _ = try self.eat(.semicolon);
            return stmt(line, .{ .decl = .{ .ty = .void, .name = name, .value = value, .array_len = alen, .struct_name = sname } });
        }
        if (isTypeTag(self.cur.tag)) {
            _ = try self.accept(.kw_const);
            _ = try self.skipPrecisionQualifier();
            const ty = try self.parseType();
            const name = (try self.eat(.ident)).text;
            const alen = try self.parseOptArraySize();
            var value: ?*Expr = null;
            if (try self.accept(.assign)) value = try self.parseExpr();
            _ = try self.eat(.semicolon);
            return stmt(line, .{ .decl = .{ .ty = ty, .name = name, .value = value, .array_len = alen } });
        }
        // Assignment (`name = expr`, `name += expr`, `name++`, `name.xy = expr`,
        // `a[i] = expr`, `a[i].field op= expr`, ...) or a bare expression. Parse a postfix
        // lvalue. If an assignment operator follows, build the matching store. Otherwise the
        // expression we parsed IS the statement expression (so the lvalue work is not wasted).
        if (self.cur.tag == .ident) {
            const lhs = try self.parsePostfix();
            if (isAssignOp(self.cur.tag)) {
                const st = try self.parseStoreRest(lhs, line);
                _ = try self.eat(.semicolon);
                return st;
            }
            // Not an assignment: continue parsing the rest of the expression from `lhs`
            // (it may be a call statement or a larger expression), then a `;`.
            const e = try self.continueExpr(lhs);
            _ = try self.eat(.semicolon);
            return stmt(line, .{ .expr = e });
        }
        const e = try self.parseExpr();
        _ = try self.eat(.semicolon);
        return stmt(line, .{ .expr = e });
    }

    /// Parse the rest of an assignment whose target `name` is already consumed: a plain
    /// `= expr`, a compound `+= expr` (desugared to `name = name + expr`), or `++`/`--`
    /// (desugared to `name = name + 1`).
    fn parseAssignRest(self: *Parser, name: []const u8) Error!Stmt {
        const line = self.cur.line;
        const tag = self.cur.tag;
        try self.advance();
        const value = switch (tag) {
            .assign => try self.parseExpr(),
            .plus_eq, .minus_eq, .star_eq, .slash_eq, .percent_eq, .amp_eq, .pipe_eq, .caret_eq, .lt_lt_eq, .gt_gt_eq => blk: {
                const op: BinOp = switch (tag) {
                    .plus_eq => .add,
                    .minus_eq => .sub,
                    .star_eq => .mul,
                    .slash_eq => .div,
                    .percent_eq => .mod,
                    .amp_eq => .bit_and,
                    .pipe_eq => .bit_or,
                    .caret_eq => .bit_xor,
                    .lt_lt_eq => .shl,
                    .gt_gt_eq => .shr,
                    else => unreachable,
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
        return stmt(line, .{ .assign = .{ .name = name, .value = value } });
    }

    fn parseIf(self: *Parser) Error!Stmt {
        const line = self.cur.line;
        _ = try self.eat(.kw_if);
        _ = try self.eat(.lparen);
        const cond = try self.parseExpr();
        _ = try self.eat(.rparen);
        const then = try self.parseBlockOrStmt();
        const els: []const Stmt = if (try self.accept(.kw_else)) try self.parseBlockOrStmt() else &.{};
        return stmt(line, .{ .if_ = .{ .cond = cond, .then = then, .@"else" = els } });
    }

    fn parseFor(self: *Parser) Error!Stmt {
        const line = self.cur.line;
        _ = try self.eat(.kw_for);
        _ = try self.eat(.lparen);
        const init_stmts: []const Stmt = if (try self.accept(.semicolon)) &.{} else try self.arena.dupe(Stmt, &.{try self.parseStmt()});
        const cond: ?*Expr = if (self.cur.tag == .semicolon) null else try self.parseExpr();
        _ = try self.eat(.semicolon);
        const incr: []const Stmt = if (self.cur.tag == .rparen) &.{} else try self.arena.dupe(Stmt, &.{try self.parseSimpleStmt()});
        _ = try self.eat(.rparen);
        const body = try self.parseBlockOrStmt();
        return stmt(line, .{ .for_ = .{ .init = init_stmts, .cond = cond, .incr = incr, .body = body } });
    }

    fn parseWhile(self: *Parser) Error!Stmt {
        const line = self.cur.line;
        _ = try self.eat(.kw_while);
        _ = try self.eat(.lparen);
        const cond = try self.parseExpr();
        _ = try self.eat(.rparen);
        const body = try self.parseBlockOrStmt();
        return stmt(line, .{ .for_ = .{ .init = &.{}, .cond = cond, .incr = &.{}, .body = body } });
    }

    /// Parse a `switch` and desugar it to a nested `if`/`else` chain. GLSL switch
    /// selectors and functions in this subset are side-effect free (functions are inlined
    /// by-value), so the selector is inlined into each `==` comparison rather than bound to
    /// a temporary. Each case body's trailing `break` is stripped (it is the case
    /// terminator). Constructs this cannot represent as straight if/else are rejected:
    ///   - fall-through (a non-last case whose body does not end in break/return/discard),
    ///   - a `break` anywhere but the tail of a case body (nothing to break out of once the
    ///     switch is if/else, and letting it target an enclosing loop would miscompile).
    /// `continue` is left intact: with no loop introduced here it correctly targets the
    /// enclosing loop, matching GLSL semantics.
    fn parseSwitch(self: *Parser) Error!Stmt {
        const line = self.cur.line;
        _ = try self.eat(.kw_switch);
        _ = try self.eat(.lparen);
        const sel = try self.parseExpr();
        _ = try self.eat(.rparen);
        _ = try self.eat(.lbrace);

        const Group = struct { labels: []const *Expr, is_default: bool, body: []const Stmt, terminates: bool };
        var groups: std.ArrayList(Group) = .empty;
        var default_seen = false;
        while (self.cur.tag != .rbrace) {
            // Consume one or more consecutive labels (grouped, e.g. `case 1: case 2:`).
            var labels: std.ArrayList(*Expr) = .empty;
            var is_default = false;
            var saw_label = false;
            while (self.cur.tag == .kw_case or self.cur.tag == .kw_default) {
                if (try self.accept(.kw_case)) {
                    try labels.append(self.arena, try self.parseExpr());
                    _ = try self.eat(.colon);
                } else {
                    _ = try self.eat(.kw_default);
                    _ = try self.eat(.colon);
                    if (default_seen) return error.ParseError; // duplicate default
                    is_default = true;
                    default_seen = true;
                }
                saw_label = true;
            }
            if (!saw_label) return error.ParseError; // a statement outside any case label
            // Collect this group's body up to the next label or the closing brace.
            var body: std.ArrayList(Stmt) = .empty;
            while (self.cur.tag != .kw_case and self.cur.tag != .kw_default and self.cur.tag != .rbrace) {
                try body.append(self.arena, try self.parseStmt());
            }
            // Determine whether the body terminates, then strip a trailing `break`.
            var terminates = false;
            var stmts = body.items;
            if (stmts.len > 0) switch (stmts[stmts.len - 1].kind) {
                .break_ => {
                    terminates = true;
                    stmts = stmts[0 .. stmts.len - 1];
                },
                .ret, .discard_, .continue_ => terminates = true,
                else => {},
            };
            if (bodyHasBreak(stmts)) return error.Unsupported; // break not at the case tail
            try groups.append(self.arena, .{ .labels = try labels.toOwnedSlice(self.arena), .is_default = is_default, .body = try self.arena.dupe(Stmt, stmts), .terminates = terminates });
        }
        _ = try self.eat(.rbrace);

        // A non-last group that does not terminate would fall through: unsupported.
        for (groups.items, 0..) |g, i| {
            if (i + 1 < groups.items.len and !g.terminates) return error.Unsupported;
        }

        // The default body is the innermost `else`; value groups wrap it in source order.
        var else_body: []const Stmt = &.{};
        for (groups.items) |g| if (g.is_default) {
            else_body = g.body;
        };
        var i = groups.items.len;
        while (i > 0) {
            i -= 1;
            const g = groups.items[i];
            if (g.labels.len == 0) continue; // a pure `default:` (already placed as else)
            const cond = try self.switchCond(sel, g.labels);
            const if_stmt = stmt(line, .{ .if_ = .{ .cond = cond, .then = g.body, .@"else" = else_body } });
            else_body = try self.arena.dupe(Stmt, &.{if_stmt});
        }
        // With only a `default` (or an empty switch) there is no wrapping `if`; guard the
        // body with `if (true) { ... }` so a single statement is returned.
        if (else_body.len != 1 or else_body[0].kind != .if_) {
            const t = try self.node(.{ .bool_lit = true });
            return stmt(line, .{ .if_ = .{ .cond = t, .then = else_body, .@"else" = &.{} } });
        }
        return else_body[0];
    }

    /// Build `sel == l0 || sel == l1 || ...` for a group's labels. The selector Expr is
    /// shared across comparisons (safe: lowering never mutates the AST).
    fn switchCond(self: *Parser, sel: *Expr, labels: []const *Expr) Error!*Expr {
        var acc: ?*Expr = null;
        for (labels) |lbl| {
            const cmp = try self.node(.{ .binary = .{ .op = .eq, .lhs = sel, .rhs = lbl } });
            acc = if (acc) |a| try self.node(.{ .binary = .{ .op = .logical_or, .lhs = a, .rhs = cmp } }) else cmp;
        }
        return acc.?;
    }

    /// A statement without a trailing semicolon (the increment of a `for`): an assignment
    /// (`name = expr`, `name += expr`, `name++`, ...) or a bare expression.
    fn parseSimpleStmt(self: *Parser) Error!Stmt {
        const line = self.cur.line;
        if (self.cur.tag == .ident) {
            const save = self.lx;
            const name = self.cur.text;
            try self.advance();
            if (isAssignOp(self.cur.tag)) return self.parseAssignRest(name);
            self.lx = save;
            self.cur = .{ .tag = .ident, .text = name };
        }
        return stmt(line, .{ .expr = try self.parseExpr() });
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
        return self.parseBinarySeed(try self.parseUnary(), min_prec);
    }

    /// Precedence-climbing continued from an already-parsed left operand (`seed`).
    fn parseBinarySeed(self: *Parser, seed: *Expr, min_prec: u8) Error!*Expr {
        var lhs = seed;
        while (binPrec(self.cur.tag)) |info| {
            if (info.prec < min_prec) break;
            const op = info.op;
            try self.advance();
            const rhs = try self.parseBinary(info.prec + 1);
            lhs = try self.node(.{ .binary = .{ .op = op, .lhs = lhs, .rhs = rhs } });
        }
        return lhs;
    }

    /// Continue parsing a full expression (binary + ternary) from an already-parsed postfix
    /// operand. Used when a statement-leading lvalue turns out to be a bare expression.
    fn continueExpr(self: *Parser, seed: *Expr) Error!*Expr {
        const cond = try self.parseBinarySeed(seed, 0);
        if (self.cur.tag != .question) return cond;
        try self.advance();
        const then = try self.parseExpr();
        _ = try self.eat(.colon);
        const els = try self.parseExpr();
        return self.node(.{ .ternary = .{ .cond = cond, .then = then, .@"else" = els } });
    }

    /// Parse the rest of a store whose target lvalue is already parsed: `= expr`, compound
    /// `+= -= *= /= expr` (desugared to `target = target OP expr`), or `++`/`--`.
    fn parseStoreRest(self: *Parser, target: *Expr, line: u32) Error!Stmt {
        const tag = self.cur.tag;
        try self.advance();
        const value = switch (tag) {
            .assign => try self.parseExpr(),
            .plus_eq, .minus_eq, .star_eq, .slash_eq, .percent_eq, .amp_eq, .pipe_eq, .caret_eq, .lt_lt_eq, .gt_gt_eq => blk: {
                const op: BinOp = switch (tag) {
                    .plus_eq => .add,
                    .minus_eq => .sub,
                    .star_eq => .mul,
                    .slash_eq => .div,
                    .percent_eq => .mod,
                    .amp_eq => .bit_and,
                    .pipe_eq => .bit_or,
                    .caret_eq => .bit_xor,
                    .lt_lt_eq => .shl,
                    .gt_gt_eq => .shr,
                    else => unreachable,
                };
                const rhs = try self.parseExpr();
                break :blk try self.node(.{ .binary = .{ .op = op, .lhs = target, .rhs = rhs } });
            },
            .plus_plus, .minus_minus => blk: {
                const op: BinOp = if (tag == .plus_plus) .add else .sub;
                break :blk try self.node(.{ .binary = .{ .op = op, .lhs = target, .rhs = try self.node(.{ .int_lit = 1 }) } });
            },
            else => return error.ParseError,
        };
        // Lower a plain `ident = ...` to the simple `assign` form (keeps the existing fast
        // path). A `ident.field = ...` to `swizzle_assign` when the field is a swizzle, and
        // any indexed / member / chained target to the general `store` form.
        switch (target.*) {
            .ident => |n| return stmt(line, .{ .assign = .{ .name = n, .value = value } }),
            .swizzle => |s| if (s.value.* == .ident) return stmt(line, .{ .swizzle_assign = .{ .name = s.value.ident, .field = s.field, .value = value } }),
            else => {},
        }
        return stmt(line, .{ .store = .{ .target = target, .value = value } });
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
        if (self.cur.tag == .tilde) {
            try self.advance();
            return self.node(.{ .unary = .{ .op = .bit_not, .operand = try self.parseUnary() } });
        }
        return self.parsePostfix();
    }

    /// A primary followed by zero or more `.field` accesses (vector swizzle or struct
    /// member) and `[index]` subscripts, in any order (`a[i].field.xy`, `m[2][3]`).
    fn parsePostfix(self: *Parser) Error!*Expr {
        var e = try self.parsePrimary();
        while (true) {
            if (self.cur.tag == .dot) {
                try self.advance();
                const field = (try self.eat(.ident)).text;
                e = try self.node(.{ .swizzle = .{ .value = e, .field = field } });
            } else if (self.cur.tag == .lbracket) {
                try self.advance();
                const idx = try self.parseExpr();
                _ = try self.eat(.rbracket);
                e = try self.node(.{ .index = .{ .value = e, .index = idx } });
            } else break;
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
                const v = parseIntLit(i64, self.cur.text) catch return error.ParseError;
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
            .ident, .kw_float, .kw_int, .kw_uint, .kw_bool, .kw_vec2, .kw_vec3, .kw_vec4, .kw_ivec2, .kw_ivec3, .kw_ivec4, .kw_uvec2, .kw_uvec3, .kw_uvec4, .kw_bvec2, .kw_bvec3, .kw_bvec4, .kw_mat2, .kw_mat3, .kw_mat4 => {
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
                    // A constructor for a user-declared struct vs a function/type-ctor call.
                    if (self.isStructName(name)) return self.node(.{ .struct_ctor = .{ .name = name, .args = try args.toOwnedSlice(self.arena) } });
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
    // C/GLSL precedence, loosest to tightest.
    return switch (tag) {
        .pipe_pipe => .{ .op = .logical_or, .prec = 1 },
        .amp_amp => .{ .op = .logical_and, .prec = 2 },
        .pipe => .{ .op = .bit_or, .prec = 3 },
        .caret => .{ .op = .bit_xor, .prec = 4 },
        .amp => .{ .op = .bit_and, .prec = 5 },
        .eq => .{ .op = .eq, .prec = 6 },
        .neq => .{ .op = .ne, .prec = 6 },
        .lt => .{ .op = .lt, .prec = 7 },
        .gt => .{ .op = .gt, .prec = 7 },
        .le => .{ .op = .le, .prec = 7 },
        .ge => .{ .op = .ge, .prec = 7 },
        .lt_lt => .{ .op = .shl, .prec = 8 },
        .gt_gt => .{ .op = .shr, .prec = 8 },
        .plus => .{ .op = .add, .prec = 9 },
        .minus => .{ .op = .sub, .prec = 9 },
        .star => .{ .op = .mul, .prec = 10 },
        .slash => .{ .op = .div, .prec = 10 },
        .percent => .{ .op = .mod, .prec = 10 },
        else => null,
    };
}

fn isAssignOp(tag: Tag) bool {
    return switch (tag) {
        .assign, .plus_eq, .minus_eq, .star_eq, .slash_eq, .percent_eq, .amp_eq, .pipe_eq, .caret_eq, .lt_lt_eq, .gt_gt_eq, .plus_plus, .minus_minus => true,
        else => false,
    };
}

fn isTypeTag(tag: Tag) bool {
    return switch (tag) {
        .kw_float, .kw_int, .kw_uint, .kw_bool, .kw_const, .kw_vec2, .kw_vec3, .kw_vec4, .kw_ivec2, .kw_ivec3, .kw_ivec4, .kw_uvec2, .kw_uvec3, .kw_uvec4, .kw_bvec2, .kw_bvec3, .kw_bvec4, .kw_mat2, .kw_mat3, .kw_mat4, .kw_sampler2d, .kw_samplercube, .kw_sampler3d, .kw_sampler2darray, .kw_sampler2dshadow, .kw_samplercubeshadow, .kw_sampler2darrayshadow => true,
        // A precision qualifier introduces a typed local declaration (`mediump float x;`).
        .kw_lowp, .kw_mediump, .kw_highp => true,
        else => false,
    };
}

fn trimFloatSuffix(s: []const u8) []const u8 {
    return if (s.len > 0 and (s[s.len - 1] == 'f' or s[s.len - 1] == 'F')) s[0 .. s.len - 1] else s;
}

/// Whether a statement list contains a `break` that targets the enclosing construct: a
/// bare `break` or one inside an `if`. It does not descend into `for_` (a nested loop owns
/// its own breaks). Used by `parseSwitch` to reject a `break` that is not a case tail.
fn bodyHasBreak(body: []const Stmt) bool {
    for (body) |s| if (stmtHasBreak(s)) return true;
    return false;
}
fn stmtHasBreak(s: Stmt) bool {
    return switch (s.kind) {
        .break_ => true,
        .if_ => |iff| bodyHasBreak(iff.then) or bodyHasBreak(iff.@"else"),
        else => false,
    };
}

/// Parse an integer literal with C/GLSL base semantics: `0x`/`0X` is hex, a leading
/// `0` followed by more digits is octal, everything else is decimal. Zig's base-0
/// auto-detect treats `0777` as decimal, so octal needs explicit handling.
fn parseIntLit(comptime T: type, raw: []const u8) !T {
    // Strip a trailing unsigned suffix (`u`/`U`) if present.
    const text = if (raw.len > 0 and (raw[raw.len - 1] == 'u' or raw[raw.len - 1] == 'U'))
        raw[0 .. raw.len - 1]
    else
        raw;
    if (text.len >= 2 and text[0] == '0' and (text[1] == 'x' or text[1] == 'X')) {
        return std.fmt.parseInt(T, text[2..], 16);
    }
    if (text.len >= 2 and text[0] == '0') {
        return std.fmt.parseInt(T, text[1..], 8);
    }
    return std.fmt.parseInt(T, text, 10);
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
    try std.testing.expect(f.body[0].kind == .ret);
    // x * 2.0 + 1.0 parses as (x * 2.0) + 1.0
    try std.testing.expect(f.body[0].kind.ret.?.binary.op == .add);
}
