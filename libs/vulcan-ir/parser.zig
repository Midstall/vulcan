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

/// An underscore is part of a word. The printer writes mnemonics such as `call_indirect`
/// and `global_addr`, and attribute keys such as `local_size_x`, so a word that stops at
/// the underscore cannot read back what the printer wrote.
fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
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
        try self.parseFunctionModifiers();
        self.skipWs();
        try self.eat('{');

        while (true) {
            self.skipWs();
            if (self.peek() == '}') {
                self.pos += 1;
                break;
            }
            // Attributes here sit above a block label, so they belong to that block. The
            // label is read first, so they attach to the block the label names.
            var pending: std.ArrayList(Attribute) = .empty;
            defer pending.deinit(self.allocator());
            try self.collectAttrs(&pending, null);
            try self.parseBlock(&pending);
        }
    }

    /// Read the whole-function metadata words the printer writes between `fn` and `{`:
    /// `local`, `variadic(n)` and `sret`, in any order and any number. An ordinary
    /// function has none of them and this stops at once.
    fn parseFunctionModifiers(self: *FunctionParser) Error!void {
        while (true) {
            const save = self.pos;
            self.skipWs();
            if (self.peek() == '{') return;
            const word = self.readWord();
            if (std.mem.eql(u8, word, "local")) {
                self.func.is_local = true;
            } else if (std.mem.eql(u8, word, "sret")) {
                self.func.sret = true;
            } else if (std.mem.eql(u8, word, "variadic")) {
                try self.eat('(');
                self.func.num_fixed_params = std.math.cast(u32, try self.readUnsigned()) orelse
                    return error.InvalidSyntax;
                try self.eat(')');
                self.func.is_variadic = true;
            } else {
                self.pos = save;
                return;
            }
        }
    }

    /// Try to read `word` at the cursor. On a different word the cursor does not move, so
    /// the caller can fall through to another spelling.
    fn tryWord(self: *FunctionParser, word: []const u8) bool {
        const save = self.pos;
        self.skipWs();
        if (std.mem.eql(u8, self.readWord(), word)) return true;
        self.pos = save;
        return false;
    }

    /// Read a `true` or `false` word.
    fn readBool(self: *FunctionParser) Error!bool {
        self.skipWs();
        const word = self.readWord();
        if (std.mem.eql(u8, word, "true")) return true;
        if (std.mem.eql(u8, word, "false")) return false;
        return error.InvalidSyntax;
    }

    /// Read an unsigned literal that may carry a `0x` prefix, as the printer writes the
    /// quant scale bits.
    fn readRadixUnsigned(self: *FunctionParser) Error!u64 {
        self.skipWs();
        const word = self.readWord();
        if (word.len == 0) return error.InvalidSyntax;
        return std.fmt.parseInt(u64, word, 0) catch error.InvalidSyntax;
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
        // Namespaced: `namespace.key` with an optional `= value`. A namespace may itself
        // hold dots (`vulcan.gpu` is the one the whole GPU path uses), so read the full
        // dotted path and split it at the LAST dot: everything before it is the
        // namespace, the final segment is the key.
        const path_start = self.pos - word.len;
        while (self.peek() == '.') {
            self.pos += 1;
            if (self.readWord().len == 0) return error.InvalidSyntax;
        }
        const path = self.src[path_start..self.pos];
        const dot = std.mem.lastIndexOfScalar(u8, path, '.') orelse return error.InvalidSyntax;
        const namespace = path[0..dot];
        const key = path[dot + 1 ..];
        self.skipWs();
        var value: AttrValue = .flag;
        if (self.tryChar('=')) {
            self.skipWs();
            value = try self.parseAttrValue();
        }
        return .{ .custom = .{ .namespace = namespace, .key = key, .value = value } };
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

    fn parseBlock(self: *FunctionParser, block_attrs: *const std.ArrayList(Attribute)) Error!void {
        self.skipWs();
        const label = self.readWord();
        if (!std.mem.startsWith(u8, label, "block")) return error.InvalidSyntax;
        const bnum = std.fmt.parseInt(u32, label["block".len..], 10) catch return error.InvalidSyntax;
        const block = try self.checkedBlock(bnum);
        for (block_attrs.items) |attr| try self.func.addAttr(.{ .block = block }, attr);

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
        // Attributes that follow the parameter attach to it, not to the block.
        try self.parseAttrs(.{ .value = value });
    }

    /// Parse instructions until a terminator ends the block.
    fn parseBody(self: *FunctionParser, block: Block) Error!void {
        while (true) {
            self.skipWs();

            var pending: std.ArrayList(Attribute) = .empty;
            defer pending.deinit(self.allocator());
            var pending_inst: std.ArrayList(Attribute) = .empty;
            defer pending_inst.deinit(self.allocator());
            try self.collectAttrs(&pending, &pending_inst);

            // An instruction attribute attaches to whatever instruction the statement
            // appends, so remember where the block ended before the statement is read.
            const insts_before = self.func.blockInsts(block).len;

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
            } else if (std.mem.eql(u8, word, "barrier")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseBarrier(block);
            } else if (std.mem.eql(u8, word, "call")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseVoidCall(block);
            } else if (std.mem.eql(u8, word, "call_indirect")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseVoidCallIndirect(block);
            } else if (std.mem.eql(u8, word, "prefetch")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                self.skipWs();
                try self.func.appendPrefetch(block, try self.parseValueRef());
            } else if (std.mem.eql(u8, word, "va_start")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                self.skipWs();
                try self.func.appendVaStart(block, try self.parseValueRef());
            } else if (std.mem.eql(u8, word, "va_end")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                self.skipWs();
                try self.func.appendVaEnd(block, try self.parseValueRef());
            } else if (std.mem.eql(u8, word, "matmul")) {
                if (pending.items.len != 0) return error.InvalidSyntax;
                try self.parseMatmul(block);
            } else if (std.mem.eql(u8, word, "ret")) {
                if (pending.items.len != 0 or pending_inst.items.len != 0) return error.InvalidSyntax;
                try self.parseRet(block);
                return;
            } else if (std.mem.startsWith(u8, word, "block")) {
                if (pending.items.len != 0 or pending_inst.items.len != 0) return error.InvalidSyntax;
                try self.parseJump(block, word);
                return;
            } else {
                return error.InvalidSyntax;
            }

            // A terminator appends no instruction, so both arms above return before this.
            if (pending_inst.items.len != 0) {
                const insts = self.func.blockInsts(block);
                if (insts.len == insts_before) return error.InvalidSyntax;
                const inst = insts[insts.len - 1];
                for (pending_inst.items) |attr| try self.func.addAttr(.{ .inst = inst }, attr);
            }
        }
    }

    /// Collect leading attributes without attaching them yet. A plain `#[...]` goes to
    /// `list`, a `#![...]` to `inst_list`, which names the instruction rather than the
    /// value it defines. When `inst_list` is null a `#!` form is a parse error, because
    /// there is no instruction at that position.
    fn collectAttrs(self: *FunctionParser, list: *std.ArrayList(Attribute), inst_list: ?*std.ArrayList(Attribute)) Error!void {
        while (true) {
            self.skipWs();
            if (self.peek() != '#') break;
            try self.eat('#');
            const on_inst = self.tryChar('!');
            try self.eat('[');
            self.skipWs();
            const attr = try self.parseAttrBody();
            if (on_inst) {
                const target = inst_list orelse return error.InvalidSyntax;
                try target.append(self.allocator(), attr);
            } else {
                try list.append(self.allocator(), attr);
            }
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
        const op: function.Opcode = switch (self.func.types.type_kind(ty)) {
            .float => |fk| if (fk == .f128)
                // The f64 carrier cannot hold 128 bits, so the text form parses the
                // decimal at f128 width and keeps the binary128 bit pattern.
                .{ .fconst128 = @bitCast(try self.readFloat128()) }
            else
                .{ .fconst = try self.readFloat() },
            else => .{ .iconst = try self.readSigned() },
        };
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

    /// The f128 form of readFloat: the same character set, parsed at binary128 width.
    fn readFloat128(self: *FunctionParser) Error!f128 {
        const start = self.pos;
        while (self.pos < self.src.len) : (self.pos += 1) {
            switch (self.src[self.pos]) {
                '0'...'9', '.', '-', '+', 'e', 'E' => {},
                else => break,
            }
        }
        return std.fmt.parseFloat(f128, self.src[start..self.pos]) catch error.InvalidSyntax;
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
            // `volatile` marks an access the optimizer must not remove, move or merge.
            const is_volatile = self.tryWord("volatile");
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const ptr = try self.parseValueRef();
            const result = try self.func.appendInst(block, ty, .{ .load = .{ .ptr = ptr, .@"volatile" = is_volatile } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "global_addr")) {
            const via_got = self.tryWord("got");
            self.skipWs();
            try self.eat('@');
            const name = self.readWord();
            const symbol = try self.func.internSymbol(name);
            // The printed form carries no result type, so the address space is the
            // default one, the same rule `alloca` already follows here.
            const ptr_t = try self.func.types.ptrGlobal();
            const result = try self.func.appendInst(block, ptr_t, .{ .global_addr = .{ .symbol = symbol, .via_got = via_got } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "dot")) {
            self.skipWs();
            const acc = try self.parseValueRef();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const a = try self.parseValueRef();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const b = try self.parseValueRef();
            const result = try self.func.appendDot(block, acc, a, b);
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "va_arg")) {
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const list = try self.parseValueRef();
            const result = try self.func.appendVaArg(block, list, ty);
            try self.recordValue(result);
            return result;
        }
        if (std.meta.stringToEnum(function.UnaryOp, op)) |uop| {
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            try self.eat(',');
            self.skipWs();
            const value = try self.parseValueRef();
            const result = try self.func.appendInst(block, ty, .{ .unary = .{ .op = uop, .value = value } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "call_indirect")) {
            self.skipWs();
            const ty = try self.parseType();
            self.skipWs();
            const target = try self.parseValueRef();

            var args: std.ArrayList(Value) = .empty;
            defer args.deinit(self.allocator());
            try self.parseCallArgs(&args);
            const extras = try self.parseCallExtras();

            const list = try self.func.internValues(args.items);
            const result = try self.func.appendInst(block, ty, .{ .call_indirect = .{
                .target = target,
                .args = list,
                .is_variadic = extras.is_variadic,
                .num_fixed = extras.num_fixed,
                .ret_dest = extras.ret_dest,
                .ret_regs = extras.ret_regs,
                .ret_pieces = extras.ret_pieces,
                .sret = extras.sret,
            } });
            try self.recordValue(result);
            return result;
        }
        if (std.mem.eql(u8, op, "alloca")) {
            self.skipWs();
            const elem = try self.parseType();
            const ptr_t = try self.func.types.ptrGlobal();
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

            var args: std.ArrayList(Value) = .empty;
            defer args.deinit(self.allocator());
            try self.parseCallArgs(&args);
            const extras = try self.parseCallExtras();

            const symbol = try self.func.internSymbol(name);
            const list = try self.func.internValues(args.items);
            const result = try self.func.appendInst(block, ty, .{ .call = .{
                .symbol = symbol,
                .args = list,
                .is_variadic = extras.is_variadic,
                .num_fixed = extras.num_fixed,
                .ret_dest = extras.ret_dest,
                .ret_regs = extras.ret_regs,
                .ret_pieces = extras.ret_pieces,
                .sret = extras.sret,
            } });
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
        // `volatile` marks an access the optimizer must not remove, move or merge.
        const is_volatile = self.tryWord("volatile");
        self.skipWs();
        const value = try self.parseValueRef();
        self.skipWs();
        try self.eat(',');
        self.skipWs();
        const ptr = try self.parseValueRef();
        try self.func.appendStoreVol(block, value, ptr, is_volatile);
    }

    /// Parse a barrier statement: `barrier workgroup`. A result-less statement with one
    /// word operand, the same shape as `store`.
    fn parseBarrier(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        const word = self.readWord();
        // The text is UNTRUSTED, so an unknown scope is a parse error and never an
        // invalid enum.
        const scope = std.meta.stringToEnum(function.BarrierScope, word) orelse
            return error.InvalidSyntax;
        try self.func.appendBarrier(block, scope);
    }

    /// Parse a void call statement: `call @name(args)` with its trailing extras.
    fn parseVoidCall(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        try self.eat('@');
        const name = self.readWord();

        var args: std.ArrayList(Value) = .empty;
        defer args.deinit(self.allocator());
        try self.parseCallArgs(&args);
        const extras = try self.parseCallExtras();

        const symbol = try self.func.internSymbol(name);
        const list = try self.func.internValues(args.items);
        _ = try self.func.appendStmtRaw(block, .{ .call = .{
            .symbol = symbol,
            .args = list,
            .is_variadic = extras.is_variadic,
            .num_fixed = extras.num_fixed,
            .ret_dest = extras.ret_dest,
            .ret_regs = extras.ret_regs,
            .ret_pieces = extras.ret_pieces,
            .sret = extras.sret,
        } });
    }

    /// Parse a void indirect call statement: `call_indirect vT(args)` with its extras.
    fn parseVoidCallIndirect(self: *FunctionParser, block: Block) Error!void {
        self.skipWs();
        const target = try self.parseValueRef();

        var args: std.ArrayList(Value) = .empty;
        defer args.deinit(self.allocator());
        try self.parseCallArgs(&args);
        const extras = try self.parseCallExtras();

        const list = try self.func.internValues(args.items);
        _ = try self.func.appendStmtRaw(block, .{ .call_indirect = .{
            .target = target,
            .args = list,
            .is_variadic = extras.is_variadic,
            .num_fixed = extras.num_fixed,
            .ret_dest = extras.ret_dest,
            .ret_regs = extras.ret_regs,
            .ret_pieces = extras.ret_pieces,
            .sret = extras.sret,
        } });
    }

    /// Parse a parenthesized, comma-separated argument list into `list`.
    fn parseCallArgs(self: *FunctionParser, list: *std.ArrayList(Value)) Error!void {
        self.skipWs();
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

    /// The extra fields a call carries after its argument list. The defaults are those of
    /// an ordinary, non-variadic call that returns a scalar or nothing.
    const CallExtras = struct {
        is_variadic: bool = false,
        num_fixed: u32 = 0,
        ret_dest: ?Value = null,
        ret_regs: u8 = 0,
        ret_pieces: [4]function.RetPiece = @splat(.{}),
        sret: bool = false,
    };

    /// Read the clauses the printer writes after a call's argument list: `variadic(n)`,
    /// `sret`, and `retdest(dest,[pieces])`. Each is optional and they come in any order.
    /// An unknown word ends the list and leaves the cursor on it, because it belongs to
    /// the next statement.
    fn parseCallExtras(self: *FunctionParser) Error!CallExtras {
        var out: CallExtras = .{};
        while (true) {
            const save = self.pos;
            self.skipWs();
            const word = self.readWord();
            if (std.mem.eql(u8, word, "variadic")) {
                try self.eat('(');
                out.num_fixed = std.math.cast(u32, try self.readUnsigned()) orelse
                    return error.InvalidSyntax;
                try self.eat(')');
                out.is_variadic = true;
            } else if (std.mem.eql(u8, word, "sret")) {
                out.sret = true;
            } else if (std.mem.eql(u8, word, "retdest")) {
                try self.eat('(');
                if (self.peek() == 'v') {
                    out.ret_dest = try self.parseValueRef();
                } else {
                    try self.expectWord("none");
                }
                try self.eat(',');
                try self.eat('[');
                var count: usize = 0;
                if (self.peek() != ']') {
                    while (true) {
                        if (count >= out.ret_pieces.len) return error.InvalidSyntax;
                        out.ret_pieces[count] = try self.parseRetPiece();
                        count += 1;
                        if (self.tryChar(',')) continue;
                        break;
                    }
                }
                try self.eat(']');
                try self.eat(')');
                out.ret_regs = @intCast(count);
            } else {
                self.pos = save;
                return out;
            }
        }
    }

    /// Read one return-register piece, `bank@offset:bytes`, where the bank is `i` for an
    /// integer register and `f` for a floating-point one.
    fn parseRetPiece(self: *FunctionParser) Error!function.RetPiece {
        const bank = self.readWord();
        const fp = if (std.mem.eql(u8, bank, "f"))
            true
        else if (std.mem.eql(u8, bank, "i"))
            false
        else
            return error.InvalidSyntax;
        try self.eat('@');
        const offset = std.math.cast(u8, try self.readUnsigned()) orelse return error.InvalidSyntax;
        try self.eat(':');
        const bytes = std.math.cast(u8, try self.readUnsigned()) orelse return error.InvalidSyntax;
        return .{ .fp = fp, .offset = offset, .bytes = bytes };
    }

    /// Parse a matmul statement, the shape `printInst` writes:
    /// `matmul c=vC, a=vA, b=vB [m x n x k] dtype [acc] [embedded] [signs] [quant(...)]`.
    fn parseMatmul(self: *FunctionParser, block: Block) Error!void {
        const c = try self.parseNamedOperand("c");
        self.skipWs();
        try self.eat(',');
        const a = try self.parseNamedOperand("a");
        self.skipWs();
        try self.eat(',');
        const b = try self.parseNamedOperand("b");

        self.skipWs();
        try self.eat('[');
        const m = std.math.cast(u16, try self.readRadixUnsigned()) orelse return error.InvalidSyntax;
        try self.expectWordAfterWs("x");
        const n = std.math.cast(u16, try self.readRadixUnsigned()) orelse return error.InvalidSyntax;
        try self.expectWordAfterWs("x");
        const k = std.math.cast(u16, try self.readRadixUnsigned()) orelse return error.InvalidSyntax;
        self.skipWs();
        try self.eat(']');

        self.skipWs();
        // The dtype word comes off UNTRUSTED text, so an unknown one is a parse error and
        // never an invalid enum.
        const dtype = std.meta.stringToEnum(function.MatMulType, self.readWord()) orelse
            return error.InvalidSyntax;
        const accumulate = self.tryWord("acc");
        const embedded = self.tryWord("embedded");

        var input_signs: ?function.InputSigns = null;
        if (self.tryWord("a_uns")) {
            try self.eat('=');
            const a_unsigned = try self.readBool();
            try self.eat(',');
            try self.expectWordAfterWs("b_uns");
            try self.eat('=');
            const b_unsigned = try self.readBool();
            input_signs = .{ .a_unsigned = a_unsigned, .b_unsigned = b_unsigned };
        }

        var quant: ?function.MatMulQuant = null;
        if (self.tryWord("quant")) quant = try self.parseMatmulQuant();

        _ = try self.func.appendStmtRaw(block, .{ .matmul = .{
            .a = a,
            .b = b,
            .c = c,
            .m = m,
            .n = n,
            .k = k,
            .dtype = dtype,
            .accumulate = accumulate,
            .embedded = embedded,
            .quant = quant,
            .input_signs = input_signs,
        } });
    }

    /// Read a `name=vN` operand of a matmul.
    fn parseNamedOperand(self: *FunctionParser, name: []const u8) Error!Value {
        try self.expectWordAfterWs(name);
        try self.eat('=');
        return self.parseValueRef();
    }

    /// `expectWord` with leading whitespace skipped first.
    fn expectWordAfterWs(self: *FunctionParser, word: []const u8) Error!void {
        self.skipWs();
        try self.expectWord(word);
    }

    /// Parse a matmul's requantize epilogue, the body inside `quant(...)`. The scale and
    /// the bias are constant data written as values, so they rebuild exactly.
    fn parseMatmulQuant(self: *FunctionParser) Error!function.MatMulQuant {
        try self.eat('(');
        self.skipWs();
        const scale_word = self.readWord();
        var scale: function.MatMulScale = undefined;
        if (std.mem.eql(u8, scale_word, "scalar")) {
            try self.eat('=');
            scale = .{ .scalar = std.math.cast(u32, try self.readRadixUnsigned()) orelse
                return error.InvalidSyntax };
        } else if (std.mem.eql(u8, scale_word, "per_col")) {
            try self.eat('[');
            var scales: std.ArrayList(u32) = .empty;
            defer scales.deinit(self.allocator());
            if (self.peek() != ']') {
                while (true) {
                    const s = std.math.cast(u32, try self.readRadixUnsigned()) orelse
                        return error.InvalidSyntax;
                    try scales.append(self.allocator(), s);
                    if (self.tryChar(',')) continue;
                    break;
                }
            }
            try self.eat(']');
            scale = .{ .per_column = try self.func.internScales(scales.items) };
        } else return error.InvalidSyntax;

        try self.eat(',');
        try self.expectWordAfterWs("relu");
        try self.eat('=');
        const relu = try self.readBool();

        try self.eat(',');
        self.skipWs();
        const out = std.meta.stringToEnum(function.MatMulQuantOut, self.readWord()) orelse
            return error.InvalidSyntax;

        try self.eat(',');
        try self.expectWordAfterWs("bias");
        var bias: ?function.BiasList = null;
        if (self.tryChar('=')) {
            try self.expectWordAfterWs("none");
        } else {
            try self.eat('[');
            var values: std.ArrayList(i32) = .empty;
            defer values.deinit(self.allocator());
            if (self.peek() != ']') {
                while (true) {
                    const v = std.math.cast(i32, try self.readSigned()) orelse
                        return error.InvalidSyntax;
                    try values.append(self.allocator(), v);
                    if (self.tryChar(',')) continue;
                    break;
                }
            }
            try self.eat(']');
            bias = try self.func.internBias(values.items);
        }

        // A zero zero-point is the symmetric default and does not print.
        var zero_point: i32 = 0;
        if (self.tryChar(',')) {
            try self.expectWordAfterWs("zp");
            try self.eat('=');
            zero_point = std.math.cast(i32, try self.readSigned()) orelse return error.InvalidSyntax;
        }
        try self.eat(')');
        return .{ .scale = scale, .relu = relu, .out = out, .bias = bias, .zero_point = zero_point };
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
            // Could be `void` or a comma-separated value-reference list `vN, vM, ...`.
            if (std.mem.startsWith(u8, self.src[self.pos..], "void") and
                (self.pos + 4 >= self.src.len or !isWordChar(self.src[self.pos + 4])))
            {
                self.pos += 4;
                self.func.setTerminator(block, .{ .ret = function.Ret.none() });
                return;
            }
            var values: [4]Value = undefined;
            var count: usize = 0;
            while (true) {
                if (count >= values.len) return error.InvalidSyntax;
                values[count] = try self.parseValueRef();
                count += 1;
                self.skipWs();
                if (self.tryChar(',')) {
                    self.skipWs();
                    continue;
                }
                break;
            }
            self.func.setTerminator(block, .{ .ret = function.Ret.many(values[0..count]) });
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

test "round-trips a workgroup barrier" {
    const text =
        \\fn {
        \\  block0(v0: ptr, v1: i32):
        \\    store v1, v0
        \\    barrier workgroup
        \\    let v2 = load i32, v0
        \\    ret v2
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "round-trips a subgroup barrier" {
    // The second scope round-trips too, even though no backend lowers it. A scope that
    // parses but does not print, or the reverse, would lose the operation silently.
    const text =
        \\fn {
        \\  block0(v0: i32):
        \\    barrier subgroup
        \\    ret v0
        \\}
    ;

    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    try std.testing.expectFmt(text, "{f}", .{func});
}

test "an unknown barrier scope is a parse error, not an invalid enum" {
    // Suspicious case: the text is untrusted, so stringToEnum must reject the word rather
    // than build an out-of-range tag every later exhaustive switch reads as undefined.
    //
    // The `barrier workgroup` control below is what makes this test mean anything. Every
    // other token in these three strings is identical, so the control proves the two
    // refusals come from the scope word and not from some unrelated syntax error.
    var good = try parse(std.testing.allocator, "fn {\n  block0():\n    barrier workgroup\n    ret void\n}");
    good.deinit();

    const unknown = "fn {\n  block0():\n    barrier nonsense\n    ret void\n}";
    try std.testing.expectError(error.InvalidSyntax, parse(std.testing.allocator, unknown));
    const missing = "fn {\n  block0():\n    barrier\n    ret void\n}";
    try std.testing.expectError(error.InvalidSyntax, parse(std.testing.allocator, missing));
}

test "a barrier verifies clean and keeps its scope through a clone" {
    const verify = @import("verify.zig");
    const text = "fn {\n  block0(v0: i32):\n    barrier workgroup\n    ret v0\n}";
    var func = try parse(std.testing.allocator, text);
    defer func.deinit();

    var diags = try verify.verify(std.testing.allocator, &func, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var copy = try func.clone(std.testing.allocator);
    defer copy.deinit();
    const insts = copy.blockInsts(@enumFromInt(0));
    try std.testing.expectEqual(@as(usize, 1), insts.len);
    try std.testing.expectEqual(
        function.BarrierScope.workgroup,
        copy.opcode(insts[0]).barrier.scope,
    );
}
