//! Reconstructs a `Function` from Vulcan's functional text format, the
//! round-trip partner of the printer. Values and blocks are named positionally
//! (`v{n}`, `block{n}`), so the parser creates them in textual order and resolves
//! references by number.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");
const attribute = @import("attribute.zig");

const Function = function.Function;
const Block = function.Block;
const Value = function.Value;
const Type = types.Type;
const Attribute = attribute.Attribute;
const AttrValue = attribute.AttrValue;
const AttrTarget = function.AttrTarget;
const CmpOp = function.CmpOp;
const BinOp = function.BinOp;

/// Map a leading character to a binary arithmetic operator, if it is one.
fn arithOpOf(c: ?u8) ?BinOp {
    return switch (c orelse return null) {
        '+' => .add,
        '-' => .sub,
        '*' => .mul,
        '/' => .div,
        '%' => .rem,
        '&' => .bit_and,
        '|' => .bit_or,
        '^' => .bit_xor,
        else => null,
    };
}

pub const ParseError = error{InvalidSyntax} || types.ParseError;
pub const Error = ParseError || std.mem.Allocator.Error;

/// Parse a function from its text form. The returned function owns its memory.
/// The caller must `deinit` it.
pub fn parse(allocator: std.mem.Allocator, text: []const u8) Error!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();

    var p: FunctionParser = .{ .func = &func, .src = text };
    defer p.value_names.deinit(allocator);

    try p.parseFunction();
    return func;
}

fn isDigit(c: u8) bool {
    return std.ascii.isDigit(c);
}

fn isLetter(c: u8) bool {
    return std.ascii.isAlphabetic(c);
}

fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c);
}

fn allDigits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| if (!isDigit(c)) return false;
    return true;
}

const FunctionParser = struct {
    func: *Function,
    src: []const u8,
    pos: usize = 0,
    /// Values indexed by their positional name number.
    value_names: std.ArrayList(Value) = .empty,

    fn allocator(self: *FunctionParser) std.mem.Allocator {
        return self.func.allocator;
    }

    fn peek(self: *FunctionParser) ?u8 {
        return if (self.pos < self.src.len) self.src[self.pos] else null;
    }

    fn skipWs(self: *FunctionParser) void {
        while (self.pos < self.src.len) : (self.pos += 1) {
            switch (self.src[self.pos]) {
                ' ', '\t', '\n', '\r' => {},
                else => break,
            }
        }
    }

    fn eat(self: *FunctionParser, c: u8) Error!void {
        if (self.peek() == c) {
            self.pos += 1;
        } else {
            return error.InvalidSyntax;
        }
    }

    fn tryChar(self: *FunctionParser, c: u8) bool {
        if (self.peek() == c) {
            self.pos += 1;
            return true;
        }
        return false;
    }

    fn readWord(self: *FunctionParser) []const u8 {
        const start = self.pos;
        while (self.pos < self.src.len and isWordChar(self.src[self.pos])) : (self.pos += 1) {}
        return self.src[start..self.pos];
    }

    fn expectWord(self: *FunctionParser, word: []const u8) Error!void {
        if (!std.mem.eql(u8, self.readWord(), word)) return error.InvalidSyntax;
    }

    fn readUnsigned(self: *FunctionParser) Error!u64 {
        const start = self.pos;
        while (self.pos < self.src.len and isDigit(self.src[self.pos])) : (self.pos += 1) {}
        if (self.pos == start) return error.InvalidSyntax;
        return std.fmt.parseInt(u64, self.src[start..self.pos], 10) catch error.InvalidSyntax;
    }

    fn readSigned(self: *FunctionParser) Error!i64 {
        const neg = self.tryChar('-');
        const mag = try self.readUnsigned();
        if (neg) {
            // -(2^63) is a valid i64 even though +2^63 is not, so handle the
            // boundary explicitly; any larger magnitude is out of range.
            if (mag == @as(u64, std.math.maxInt(i64)) + 1) return std.math.minInt(i64);
            return -(std.math.cast(i64, mag) orelse return error.InvalidSyntax);
        }
        return std.math.cast(i64, mag) orelse error.InvalidSyntax;
    }

    /// Parse a type embedded at the cursor, advancing past it.
    fn parseType(self: *FunctionParser) Error!Type {
        const parsed = try self.func.types.parseTypePrefix(self.src[self.pos..]);
        self.pos += parsed.len;
        return parsed.ty;
    }

    /// Parse a value definition `vN`, asserting N is the next positional name.
    fn defineValueName(self: *FunctionParser) Error!void {
        try self.eat('v');
        const num = try self.readUnsigned();
        if (num != self.value_names.items.len) return error.InvalidSyntax;
    }

    fn recordValue(self: *FunctionParser, value: Value) Error!void {
        try self.value_names.append(self.allocator(), value);
    }

    /// Parse a value reference `vN`, resolving it.
    fn parseValueRef(self: *FunctionParser) Error!Value {
        try self.eat('v');
        const num = try self.readUnsigned();
        if (num >= self.value_names.items.len) return error.InvalidSyntax;
        return self.value_names.items[@intCast(num)];
    }

    /// Convert a parsed block number to a `Block`, rejecting an out-of-range value
    /// that would later index the block list out of bounds (blocks are precreated
    /// up front, so the count is final here).
    fn checkedBlock(self: *FunctionParser, bnum: u32) Error!Block {
        if (@as(usize, bnum) >= self.func.blockCount()) return error.InvalidSyntax;
        return @enumFromInt(bnum);
    }

    fn parseFunction(self: *FunctionParser) Error!void {
        try self.precreateBlocks();

        self.skipWs();
        try self.parseAttrs(.func);
        self.skipWs();
        try self.expectWord("fn");
        self.skipWs();
        try self.eat('{');

        while (true) {
            self.skipWs();
            if (self.peek() == '}') {
                self.pos += 1;
                break;
            }
            try self.parseBlock();
        }
    }

    /// Parse zero or more `#[...]` attributes, attaching each to `target`.
    fn parseAttrs(self: *FunctionParser, target: AttrTarget) Error!void {
        while (true) {
            self.skipWs();
            if (self.peek() != '#') break;
            try self.eat('#');
            try self.eat('[');
            self.skipWs();
            const attr = try self.parseAttrBody();
            self.skipWs();
            try self.eat(']');
            try self.func.addAttr(target, attr);
        }
    }

    fn parseAttrBody(self: *FunctionParser) Error!Attribute {
        const word = self.readWord();
        if (std.mem.eql(u8, word, "inline")) return .@"inline";
        if (std.mem.eql(u8, word, "noreturn")) return .noreturn;
        if (std.mem.eql(u8, word, "cold")) return .cold;
        if (std.mem.eql(u8, word, "align")) {
            try self.eat('(');
            const n = try self.readUnsigned();
            try self.eat(')');
            return .{ .@"align" = std.math.cast(u32, n) orelse return error.InvalidSyntax };
        }
        if (std.mem.eql(u8, word, "endian")) {
            try self.eat('(');
            const e = self.readWord();
            try self.eat(')');
            const order = std.meta.stringToEnum(attribute.Endianness, e) orelse return error.InvalidSyntax;
            return .{ .endian = order };
        }
        // Namespaced: `namespace.key` with an optional `= value`.
        try self.eat('.');
        const key = self.readWord();
        self.skipWs();
        var value: AttrValue = .flag;
        if (self.tryChar('=')) {
            self.skipWs();
            value = try self.parseAttrValue();
        }
        return .{ .custom = .{ .namespace = word, .key = key, .value = value } };
    }

    fn parseAttrValue(self: *FunctionParser) Error!AttrValue {
        if (self.peek() == '"') {
            self.pos += 1;
            const start = self.pos;
            while (self.pos < self.src.len and self.src[self.pos] != '"') : (self.pos += 1) {}
            const s = self.src[start..self.pos];
            try self.eat('"');
            return .{ .string = s };
        }
        return .{ .int = try self.readSigned() };
    }

    /// Pre-create one block per label line so forward references resolve.
    fn precreateBlocks(self: *FunctionParser) Error!void {
        var count: usize = 0;
        var it = std.mem.splitScalar(u8, self.src, '\n');
        while (it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, trimmed, "block") and std.mem.endsWith(u8, trimmed, ":")) {
                count += 1;
            }
        }
        var i: usize = 0;
        while (i < count) : (i += 1) _ = try self.func.appendBlock();
    }

    fn parseBlock(self: *FunctionParser) Error!void {
        self.skipWs();
        const label = self.readWord();
        if (!std.mem.startsWith(u8, label, "block")) return error.InvalidSyntax;
        const bnum = std.fmt.parseInt(u32, label["block".len..], 10) catch return error.InvalidSyntax;
        const block = try self.checkedBlock(bnum);

        try self.eat('(');
        self.skipWs();
        if (self.peek() != ')') {
            while (true) {
                self.skipWs();
                try self.parseBlockParam(block);
                self.skipWs();
                if (self.tryChar(',')) continue;
                break;
            }
        }
        self.skipWs();
        try self.eat(')');
        self.skipWs();
        try self.eat(':');

        try self.parseBody(block);
    }

    fn parseBlockParam(self: *FunctionParser, block: Block) Error!void {
        try self.defineValueName();
        self.skipWs();
        try self.eat(':');
        self.skipWs();
        const ty = try self.parseType();
        const value = try self.func.appendBlockParam(block, ty);
        try self.recordValue(value);
    }

    /// Parse instructions until a terminator ends the block.
    fn parseBody(self: *FunctionParser, block: Block) Error!void {
        while (true) {
            self.skipWs();

            var pending: std.ArrayList(Attribute) = .empty;
            defer pending.deinit(self.allocator());
            try self.collectAttrs(&pending);

            self.skipWs();
            const word = self.readWord();
            if (std.mem.eql(u8, word, "const")) {
                try self.attachAll(&pending, try self.parseConst(block));
            } else if (std.mem.eql(u8, word, "let")) {
                try self.attachAll(&pending, try self.parseLet(block));
            } else if (word.len >= 2 and word[0] == 'v' and allDigits(word[1..])) {
                try self.attachAll(&pending, try self.parseSelect(block, word));
            } else if (std.mem.eql(u8, word, "if")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseIf(block);
            } else if (std.mem.eql(u8, word, "store")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseStore(block);
            } else if (std.mem.eql(u8, word, "call")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseVoidCall(block);
            } else if (std.mem.eql(u8, word, "ret")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseRet(block);
                return;
            } else if (std.mem.startsWith(u8, word, "block")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseJump(block, word);
                return;
            } else {
                return error.InvalidSyntax;
            }
        }
    }

    /// Collect leading `#[...]` attributes into `list` without attaching them yet.
    fn collectAttrs(self: *FunctionParser, list: *std.ArrayList(Attribute)) Error!void {
        while (true) {
            self.skipWs();
            if (self.peek() != '#') break;
            try self.eat('#');
            try self.eat('[');
            self.skipWs();
            try list.append(self.allocator(), try self.parseAttrBody());
            self.skipWs();
            try self.eat(']');
        }
    }

    fn attachAll(self: *FunctionParser, list: *const std.ArrayList(Attribute), value: Value) Error!void {
        for (list.items) |attr| try self.func.addAttr(.{ .value = value }, attr);
    }

    fn parseConst(self: *FunctionParser, block: Block) Error!Value {
        self.skipWs();
        try self.defineValueName();
        self.skipWs();
        try self.eat(':');
        self.skipWs();
        const ty = try self.parseType();
        self.skipWs();
        try self.eat('=');
        self.skipWs();
        const op: function.Opcode = if (self.func.types.type_kind(ty) == .float)
            .{ .fconst = try self.readFloat() }
        else
            .{ .iconst = try self.readSigned() };
        const result = try self.func.appendInst(block, ty, op);
        try self.recordValue(result);
        return result;
    }

    /// Read a floating-point literal (a run of numeric characters).
    fn readFloat(self: *FunctionParser) Error!f64 {
        const start = self.pos;
        while (self.pos < self.src.len) : (self.pos += 1) {
            switch (self.src[self.pos]) {
                '0'...'9', '.', '-', '+', 'e', 'E' => {},
                else => break,
            }
        }
        return std.fmt.parseFloat(f64, self.src[start..self.pos]) catch error.InvalidSyntax;
    }

    /// Finish a binary op `lhs <bop> rhs` after the operator: a numeric rhs makes
    /// an `arith_imm`, a value reference an `arith`.
    fn finishArith(self: *FunctionParser, block: Block, lhs: Value, bop: BinOp) Error!Value {
        self.skipWs();
        const ty = self.func.valueType(lhs);
        const ch = self.peek() orelse 0;
        const result = if (ch == '-' or isDigit(ch))
            try self.func.appendArithImm(block, ty, bop, lhs, try self.readSigned())
        else
            try self.func.appendInst(block, ty, .{ .arith = .{ .op = bop, .lhs = lhs, .rhs = try self.parseValueRef() } });
        try self.recordValue(result);
        return result;
    }

    fn parseLet(self: *FunctionParser, block: Block) Error!Value {
        self.skipWs();
        try self.defineValueName();
        self.skipWs();
        try self.eat('=');
        self.skipWs();

        // A value reference first is either field extraction `vA.#i` or a
        // comparison `vA <op> vB`.
        if (self.peek() == 'v') {
            const lhs = try self.parseValueRef();

            if (self.peek() == '.') {
                self.pos += 1;
                try self.eat('#');
                const index = std.math.cast(u32, try self.readUnsigned()) orelse return error.InvalidSyntax;
                const field_ty = switch (self.func.types.type_kind(self.func.valueType(lhs))) {
                    .@"struct" => |flds| if (index < flds.len) flds[index] else return error.InvalidSyntax,
                    else => return error.InvalidSyntax,
                };
                const result = try self.func.appendInst(block, field_ty, .{ .extract = .{ .aggregate = lhs, .index = index } });
                try self.recordValue(result);
                return result;
            }

            self.skipWs();
            // Two-character shift operators come before the single-char checks so
            // `<<` is not mistaken for the `<` comparison.
            const shift: ?BinOp =
                if (std.mem.startsWith(u8, self.src[self.pos..], "<<")) .shl else if (std.mem.startsWith(u8, self.src[self.pos..], ">>")) .shr else null;
            if (shift) |bop| {
                self.pos += 2;
                return self.finishArith(block, lhs, bop);
            }
            // Arithmetic operators yield an arith, comparison operators an icmp.
            if (arithOpOf(self.peek())) |bop| {
                self.pos += 1;
                return self.finishArith(block, lhs, bop);
            }
            const op = try self.readCmpOp();
            self.skipWs();
            const rhs = try self.parseValueRef();
            const bool_t = try self.func.types.intern(.bool);
            const result = try self.func.appendInst(block, bool_t, .{ .icmp = .{ .op = op, .lhs = lhs, .rhs = rhs } });
            try self.recordValue(result);
            return result;
        }

        const op = self.readWord();
        if (std.mem.eql(u8, op, "load")) {
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const ptr = try self.parseValueRef();
            const result = try self.func.appendInst(block, ty, .{ .load = .{ .ptr = ptr } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "alloca")) {
            self.skipWs();
            const elem = try self.parseType();
            const ptr_t = try self.func.types.intern(.ptr);
            const result = try self.func.appendInst(block, ptr_t, .{ .alloca = .{ .elem = elem } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "call")) {
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            try self.eat('@');
            const name = self.readWord();
            self.skipWs();
            try self.eat('(');

            var args: std.ArrayList(Value) = .empty;
            defer args.deinit(self.allocator());
            self.skipWs();
            if (self.peek() != ')') {
                while (true) {
                    self.skipWs();
                    try args.append(self.allocator(), try self.parseValueRef());
                    self.skipWs();
                    if (self.tryChar(',')) continue;
                    break;
                }
            }
            self.skipWs();
            try self.eat(')');

            const result = try self.func.appendCall(block, ty, name, args.items);
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "convert")) {
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const value = try self.parseValueRef();
            const result = try self.func.appendInst(block, ty, .{ .convert = .{ .value = value } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "struct")) {
            self.skipWs();
            try self.eat('{');

            var fields: std.ArrayList(Value) = .empty;
            defer fields.deinit(self.allocator());
            self.skipWs();
            if (self.peek() != '}') {
                while (true) {
                    self.skipWs();
                    try fields.append(self.allocator(), try self.parseValueRef());
                    self.skipWs();
                    if (self.tryChar(',')) continue;
                    break;
                }
            }
            self.skipWs();
            try self.eat('}');

            // The struct type is inferred from the field value types.
            var field_types: std.ArrayList(Type) = .empty;
            defer field_types.deinit(self.allocator());
            for (fields.items) |f| try field_types.append(self.allocator(), self.func.valueType(f));
            const st = try self.func.types.intern(.{ .@"struct" = field_types.items });

            const result = try self.func.appendStructNew(block, st, fields.items);
            try self.recordValue(result);
            return result;
        }
        return error.InvalidSyntax;
    }

    fn parseStore(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        const value = try self.parseValueRef();
        self.skipWs();
        try self.eat(',');
        self.skipWs();
        const ptr = try self.parseValueRef();
        try self.func.appendStore(block, value, ptr);
    }

    /// Parse a void call statement: `call @name(args)`.
    fn parseVoidCall(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        try self.eat('@');
        const name = self.readWord();
        self.skipWs();
        try self.eat('(');

        var args: std.ArrayList(Value) = .empty;
        defer args.deinit(self.allocator());
        self.skipWs();
        if (self.peek() != ')') {
            while (true) {
                self.skipWs();
                try args.append(self.allocator(), try self.parseValueRef());
                self.skipWs();
                if (self.tryChar(',')) continue;
                break;
            }
        }
        self.skipWs();
        try self.eat(')');
        try self.func.appendVoidCall(block, name, args.items);
    }

    /// Parse the value form `vN := if vC { vT } else { vE }`. `name` is the
    /// already-read `vN` definition token.
    fn parseSelect(self: *FunctionParser, block: Block, name: []const u8) Error!Value {
        const num = std.fmt.parseInt(usize, name[1..], 10) catch return error.InvalidSyntax;
        if (num != self.value_names.items.len) return error.InvalidSyntax;

        self.skipWs();
        try self.eat(':');
        try self.eat('=');
        self.skipWs();
        try self.expectWord("if");
        self.skipWs();
        const cond = try self.parseValueRef();

        self.skipWs();
        try self.eat('{');
        self.skipWs();
        const then_v = try self.parseValueRef();
        self.skipWs();
        try self.eat('}');

        self.skipWs();
        try self.expectWord("else");
        self.skipWs();
        try self.eat('{');
        self.skipWs();
        const else_v = try self.parseValueRef();
        self.skipWs();
        try self.eat('}');

        const ty = self.func.valueType(then_v);
        const result = try self.func.appendInst(block, ty, .{ .select = .{ .cond = cond, .then = then_v, .@"else" = else_v } });
        try self.recordValue(result);
        return result;
    }

    fn readCmpOp(self: *FunctionParser) Error!CmpOp {
        const c = self.peek() orelse return error.InvalidSyntax;
        switch (c) {
            '=' => {
                try self.eat('=');
                try self.eat('=');
                return .eq;
            },
            '!' => {
                try self.eat('!');
                try self.eat('=');
                return .ne;
            },
            '<' => {
                self.pos += 1;
                return if (self.tryChar('=')) .le else .lt;
            },
            '>' => {
                self.pos += 1;
                return if (self.tryChar('=')) .ge else .gt;
            },
            else => return error.InvalidSyntax,
        }
    }

    fn parseIf(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        const cond = try self.parseValueRef();

        var then_args: std.ArrayList(Value) = .empty;
        defer then_args.deinit(self.allocator());
        var else_args: std.ArrayList(Value) = .empty;
        defer else_args.deinit(self.allocator());

        self.skipWs();
        try self.eat('{');
        const then_target = try self.parseEdgeInto(&then_args);
        self.skipWs();
        try self.eat('}');

        self.skipWs();
        try self.expectWord("else");

        self.skipWs();
        try self.eat('{');
        const else_target = try self.parseEdgeInto(&else_args);
        self.skipWs();
        try self.eat('}');

        try self.func.appendIf(
            block,
            cond,
            .{ .target = then_target, .args = then_args.items },
            .{ .target = else_target, .args = else_args.items },
        );
    }

    /// Parse a `blockN(args)` edge, filling `list` with its arguments and
    /// returning the target block.
    fn parseEdgeInto(self: *FunctionParser, list: *std.ArrayList(Value)) Error!Block {
        self.skipWs();
        const label = self.readWord();
        if (!std.mem.startsWith(u8, label, "block")) return error.InvalidSyntax;
        const tnum = std.fmt.parseInt(u32, label["block".len..], 10) catch return error.InvalidSyntax;
        const target = try self.checkedBlock(tnum);
        try self.parseEdgeArgs(list);
        return target;
    }

    fn parseJump(self: *FunctionParser, block: Block, label: []const u8) Error!void {
        const tnum = std.fmt.parseInt(u32, label["block".len..], 10) catch return error.InvalidSyntax;
        const target = try self.checkedBlock(tnum);

        var args: std.ArrayList(Value) = .empty;
        defer args.deinit(self.allocator());
        try self.parseEdgeArgs(&args);

        try self.func.setJump(block, target, args.items);
    }

    /// Parse a parenthesized, comma-separated list of value references.
    fn parseEdgeArgs(self: *FunctionParser, list: *std.ArrayList(Value)) Error!void {
        try self.eat('(');
        self.skipWs();
        if (self.peek() != ')') {
            while (true) {
                self.skipWs();
                try list.append(self.allocator(), try self.parseValueRef());
                self.skipWs();
                if (self.tryChar(',')) continue;
                break;
            }
        }
        self.skipWs();
        try self.eat(')');
    }

    fn parseRet(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        if (self.peek() == 'v') {
            // Could be `void` or a value reference `vN`.
            if (std.mem.startsWith(u8, self.src[self.pos..], "void") and
                (self.pos + 4 >= self.src.len or !isWordChar(self.src[self.pos + 4])))
            {
                self.pos += 4;
                self.func.setTerminator(block, .{ .ret = null });
                return;
            }
            const value = try self.parseValueRef();
            self.func.setTerminator(block, .{ .ret = value });
            return;
        }
        return error.InvalidSyntax;
    }
};

test "round-trips a float constant" {
    const text =
        \\fn {
        \\  block0():
        \\    const v0: f32 = 1.5
        \\    ret v0
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a minimal function" {
    const text =
        \\fn {
        \\  block0():
        \\    const v0: i32 = 42
        \\    ret v0
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a function-level attribute" {
    const text =
        \\#[inline]
        \\fn {
        \\  block0():
        \\    ret void
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a namespaced attribute" {
    const text =
        \\#[target.clone = "rv64gcv"]
        \\fn {
        \\  block0():
        \\    ret void
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips an endian attribute on a load" {
    const text =
        \\fn {
        \\  block0(v0: ptr):
        \\    #[endian(big)]
        \\    let v1 = load i32, v0
        \\    ret v1
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips the arithmetic and bitwise operators" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 + v1
        \\    let v3 = v0 - v1
        \\    let v4 = v0 * v1
        \\    let v5 = v0 / v1
        \\    let v6 = v0 % v1
        \\    let v7 = v0 & v1
        \\    let v8 = v0 | v1
        \\    let v9 = v0 ^ v1
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips shifts, disambiguated from comparisons" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 << v1
        \\    let v3 = v0 >> v1
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips struct construction" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = struct { v0, v1 }
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips field extraction" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = struct { v0, v1 }
        \\    let v3 = v2.#0
        \\    ret v3
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips loads and stores" {
    const text =
        \\fn {
        \\  block0(v0: ptr):
        \\    let v1 = load i32, v0
        \\    store v1, v0
        \\    ret void
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a call" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = call i32 @add(v0, v1)
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips immediate arithmetic" {
    const text =
        \\fn {
        \\  block0(v0: i32):
        \\    let v1 = v0 + 5
        \\    let v2 = v1 << 2
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a void call" {
    const text =
        \\fn {
        \\  block0(v0: i32):
        \\    call @sink(v0)
        \\    ret void
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips an alloca" {
    const text =
        \\fn {
        \\  block0(v0: i32):
        \\    let v1 = alloca i32
        \\    store v0, v1
        \\    let v2 = load i32, v1
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips an int-to-float conversion" {
    const text =
        \\fn {
        \\  block0(v0: i32):
        \\    let v1 = convert f32, v0
        \\    ret v1
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a select" {
    const text =
        \\fn {
        \\  block0(v0: bool, v1: i32, v2: i32):
        \\    v3 := if v0 { v1 } else { v2 }
        \\    ret v3
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a comparison" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 > v1
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips and verifies the canonical max function" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 > v1
        \\    if v2 { block1(v0) } else { block1(v1) }
        \\    ret void
        \\
        \\  block1(v3: i32):
        \\    ret v3
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    // Round-trips through the printer...
    try std.testing.expectFmt(text, "{f}", .{func});

    // ...and is well-formed in both profiles.
    const verify = @import("verify.zig");
    var high = try verify.verify(std.testing.allocator, &func, .high);
    defer high.deinit();
    try std.testing.expect(high.ok());
    var low = try verify.verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());
}

test "round-trips a value attribute" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    #[align(16)]
        \\    let v2 = v0 + v1
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a function with a conditional" {
    const text =
        \\fn {
        \\  block0():
        \\    const v0: bool = 1
        \\    const v1: i32 = 5
        \\    if v0 { block1(v1) } else { block2() }
        \\    ret void
        \\
        \\  block1():
        \\    ret void
        \\  block2():
        \\    ret void
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a function with iadd and a jump" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 + v1
        \\    block1(v2)
        \\
        \\  block1(v3: i32):
        \\    ret v3
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "regression: rejects an out-of-range block label instead of OOB indexing the block list" {
    // One block is precreated (the single label line), but it is named block7;
    // the pre-fix code did @enumFromInt(7) then indexed blocks.items[7] OOB.
    const text = "fn {\n  block7():\n    ret\n}";
    try std.testing.expectError(error.InvalidSyntax, parse(std.testing.allocator, text));
}
