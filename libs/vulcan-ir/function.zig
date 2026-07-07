//! Owns an IR function's blocks, instructions, and values as entity references
//! into dense pools off a caller-supplied allocator. SSA values cross block
//! boundaries as block parameters, not phi nodes. The control-flow graph is
//! derived from terminators, never stored separately, so it cannot drift.

const std = @import("std");
const types = @import("types.zig");
const attribute = @import("attribute.zig");

const Type = types.Type;
const TypeTable = types.TypeTable;
pub const Attribute = attribute.Attribute;

/// A handle to a basic block within a function.
pub const Block = enum(u32) { _ };

/// A handle to an SSA value: a block parameter or an instruction result.
pub const Value = enum(u32) { _ };

/// A handle to an instruction within a function.
pub const Inst = enum(u32) { _ };

/// A binary arithmetic/bitwise relation. Signedness for division comes from the
/// operand types, so the relation stays signedness-agnostic.
pub const BinOp = enum {
    add,
    sub,
    mul,
    div,
    rem,
    bit_and,
    bit_or,
    bit_xor,
    shl,
    shr,

    /// The symbolic operator this relation prints as.
    pub fn symbol(self: BinOp) []const u8 {
        return switch (self) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .rem => "%",
            .bit_and => "&",
            .bit_or => "|",
            .bit_xor => "^",
            .shl => "<<",
            .shr => ">>",
        };
    }
};

/// A binary arithmetic/bitwise operation. The result type is the operand type.
pub const Arith = struct { op: BinOp, lhs: Value, rhs: Value };

/// Arithmetic against a constant operand: `lhs <op> imm`. Lowers to an immediate
/// instruction (addi/andi/.../slli) instead of materializing the constant.
pub const ArithImm = struct { op: BinOp, lhs: Value, imm: i64 };

/// A comparison relation. Signedness comes from the operand types, not the
/// relation, so the relation stays signedness-agnostic.
pub const CmpOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,

    /// The symbolic operator this relation prints as.
    pub fn symbol(self: CmpOp) []const u8 {
        return switch (self) {
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
        };
    }
};

/// An integer comparison: produces a `bool`.
pub const Compare = struct { op: CmpOp, lhs: Value, rhs: Value };

/// A value-producing conditional: `then` when `cond` is true, else `@"else"`.
/// The value form of `if` (`c := if {} else {}`).
pub const Select = struct { cond: Value, then: Value, @"else": Value };

/// Construct an aggregate value from field values. High profile only.
pub const StructNew = struct { fields: ValueList };

/// Extract field `index` from an aggregate value. High profile only.
pub const Extract = struct { aggregate: Value, index: u32 };

/// Convert a value to the instruction's result type (int<->float). The
/// direction is read from the source value's type versus the result type.
pub const Convert = struct { value: Value };

/// A single-operand operation. `reinterpret` reinterprets the bits as the result
/// type (int<->float, same width), the rest are floating-point math on the result
/// type.
pub const UnaryOp = enum { reinterpret, sqrt, ceil, floor, trunc, nearest };

/// A unary operation on `value`, producing the instruction's result type.
pub const Unary = struct { op: UnaryOp, value: Value };

/// Reserve a stack slot sized for `elem` and yield its address (a `ptr`).
pub const Alloca = struct { elem: Type };

/// Call a function named by symbol index, passing `args`. The result type is the
/// instruction's result type (use a zero-width result for a void call).
pub const Call = struct { symbol: u32, args: ValueList };

/// Call the function whose address is `target` (a `ptr`), passing `args`. The result
/// type is the instruction's result type.
pub const CallIndirect = struct { target: Value, args: ValueList };

/// The address of a named global symbol (a module-level constant or variable),
/// yielding a `ptr`. The symbol is resolved at link time.
pub const GlobalAddr = struct { symbol: u32 };

pub const Load = struct { ptr: Value };

/// A store to memory. Produces no result.
pub const Store = struct { value: Value, ptr: Value };

/// A run of values in the function's value-list pool, used for variadic operands
/// like the arguments passed across a control-flow edge.
pub const ValueList = struct { start: u32, len: u32 };

/// An edge to a block, passing arguments to its parameters. Used both as an
/// unconditional jump and as each side of a conditional branch.
pub const Jump = struct { target: Block, args: ValueList };

/// A conditional: take `then` when `cond` is true, `else` otherwise. In the high
/// profile this is a non-terminating instruction. Control continues to the
/// block's terminator afterward. Legalization lowers it to a flat
/// conditional-branch terminator in the low profile.
pub const If = struct { cond: Value, then: Jump, @"else": Jump };

/// How a block ends. The terminator transfers control out of the block. An unset
/// terminator is an implicit `ret void`.
pub const Terminator = union(enum) {
    /// Return from the function, optionally with a value.
    ret: ?Value,
    /// Branch unconditionally to a block, passing arguments to its parameters.
    jump: Jump,
};

/// A caller-facing description of a control-flow edge: a target block and the
/// arguments to pass it. The arguments are copied into the value-list pool.
pub const EdgeDesc = struct { target: Block, args: []const Value = &.{} };

/// What an attribute is attached to. Never a type (types are interned and shared).
pub const AttrTarget = union(enum) {
    func,
    block: Block,
    inst: Inst,
    value: Value,
};

/// One attribute attached to one target.
pub const AttrEntry = struct { target: AttrTarget, attr: Attribute };

/// Iterates the attributes attached to a single target.
pub const AttrIterator = struct {
    entries: []const AttrEntry,
    target: AttrTarget,
    index: usize = 0,

    pub fn next(self: *AttrIterator) ?Attribute {
        while (self.index < self.entries.len) {
            const entry = self.entries[self.index];
            self.index += 1;
            if (std.meta.eql(entry.target, self.target)) return entry.attr;
        }
        return null;
    }
};

/// The closed set of IR operations. Target-independent only, machine ops live in
/// codegen.
pub const Opcode = union(enum) {
    /// An integer constant of the instruction's result type.
    iconst: i64,
    /// A floating-point constant of the instruction's result type.
    fconst: f64,
    /// A binary arithmetic/bitwise operation.
    arith: Arith,
    /// Arithmetic against a constant operand (`lhs <op> imm`).
    arith_imm: ArithImm,
    /// Integer comparison of two operands, producing a `bool`.
    icmp: Compare,
    /// A value-producing conditional (`c := if {} else {}`).
    select: Select,
    /// Construct an aggregate from field values. High profile only.
    struct_new: StructNew,
    /// Extract a field from an aggregate. High profile only.
    extract: Extract,
    /// Convert a value to the result type (int<->float numeric conversion).
    convert: Convert,
    /// A single-operand op (bit reinterpret, or floating-point math) on the result type.
    unary: Unary,
    /// Reserve a stack slot and produce its address. Result type is `ptr`.
    alloca: Alloca,
    /// Call a function by symbol, passing arguments and producing its result.
    call: Call,
    /// Call a function through a computed address, passing arguments.
    call_indirect: CallIndirect,
    /// The address of a named global symbol, resolved at link time. Result `ptr`.
    global_addr: GlobalAddr,
    /// A load from memory, producing a value of the result type.
    load: Load,
    /// A store to memory. Produces no result.
    store: Store,
    /// A non-terminating conditional. Produces no result in its statement form.
    @"if": If,
};

/// Per-instruction storage: its opcode and the value it defines, if any.
const InstData = struct {
    op: Opcode,
    result: ?Value,
};

/// What defines a value.
const ValueDef = union(enum) {
    /// The `index`-th parameter of `block`.
    block_param: struct { block: Block, index: u32 },
    /// The result of an instruction.
    inst_result: Inst,
};

/// Per-value storage: its type and what defines it.
const ValueData = struct {
    ty: Type,
    def: ValueDef,
};

/// Per-block storage.
const BlockData = struct {
    params: std.ArrayList(Value),
    insts: std.ArrayList(Inst),
    term: ?Terminator = null,
};

/// Owns an IR function's blocks, instructions, and values.
pub const Function = struct {
    allocator: std.mem.Allocator,
    types: TypeTable,
    blocks: std.ArrayList(BlockData),
    insts: std.ArrayList(InstData),
    values: std.ArrayList(ValueData),
    value_lists: std.ArrayList(Value),
    attributes: std.ArrayList(AttrEntry),
    /// Callee names referenced by `call` instructions, owned by the function.
    symbols: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) Function {
        return .{
            .allocator = allocator,
            .types = TypeTable.init(allocator),
            .blocks = .empty,
            .insts = .empty,
            .values = .empty,
            .value_lists = .empty,
            .attributes = .empty,
            .symbols = .empty,
        };
    }

    pub fn deinit(self: *Function) void {
        for (self.blocks.items) |*block| {
            block.params.deinit(self.allocator);
            block.insts.deinit(self.allocator);
        }
        self.blocks.deinit(self.allocator);
        self.insts.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.value_lists.deinit(self.allocator);
        for (self.attributes.items) |entry| self.freeAttr(entry.attr);
        self.attributes.deinit(self.allocator);
        for (self.symbols.items) |s| self.allocator.free(s);
        self.symbols.deinit(self.allocator);
        self.types.deinit();
    }

    /// Intern a callee name, returning its symbol index. Equal names share an
    /// index, the function owns the copied string.
    pub fn internSymbol(self: *Function, name: []const u8) std.mem.Allocator.Error!u32 {
        for (self.symbols.items, 0..) |s, i| {
            if (std.mem.eql(u8, s, name)) return @intCast(i);
        }
        const owned = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned);
        try self.symbols.append(self.allocator, owned);
        return @intCast(self.symbols.items.len - 1);
    }

    /// The name of an interned callee symbol.
    pub fn symbolName(self: *const Function, symbol: u32) []const u8 {
        return self.symbols.items[symbol];
    }

    /// The number of interned symbols (indices are `0..symbolCount`).
    pub fn symbolCount(self: *const Function) usize {
        return self.symbols.items.len;
    }

    /// Attach an attribute to a target. Namespaced string payloads are copied
    /// into function-owned storage so they stay valid.
    pub fn addAttr(self: *Function, target: AttrTarget, attr: Attribute) std.mem.Allocator.Error!void {
        const owned = try self.ownAttr(attr);
        errdefer self.freeAttr(owned);
        try self.attributes.append(self.allocator, .{ .target = target, .attr = owned });
    }

    /// Iterate the attributes attached to a target.
    pub fn attributesOf(self: *const Function, target: AttrTarget) AttrIterator {
        return .{ .entries = self.attributes.items, .target = target };
    }

    /// All attribute entries, for whole-function checks like verification.
    pub fn attributeEntries(self: *const Function) []const AttrEntry {
        return self.attributes.items;
    }

    /// Copy any string payloads of an attribute into function-owned memory.
    fn ownAttr(self: *Function, attr: Attribute) std.mem.Allocator.Error!Attribute {
        switch (attr) {
            .custom => |c| {
                const namespace = try self.allocator.dupe(u8, c.namespace);
                errdefer self.allocator.free(namespace);
                const key = try self.allocator.dupe(u8, c.key);
                errdefer self.allocator.free(key);
                const value: attribute.AttrValue = switch (c.value) {
                    .string => |s| .{ .string = try self.allocator.dupe(u8, s) },
                    else => c.value,
                };
                return .{ .custom = .{ .namespace = namespace, .key = key, .value = value } };
            },
            else => return attr,
        }
    }

    /// Free any function-owned string payloads of an attribute.
    fn freeAttr(self: *Function, attr: Attribute) void {
        switch (attr) {
            .custom => |c| {
                self.allocator.free(c.namespace);
                self.allocator.free(c.key);
                switch (c.value) {
                    .string => |s| self.allocator.free(s),
                    else => {},
                }
            },
            else => {},
        }
    }

    /// Append a fresh, empty block and return its handle.
    pub fn appendBlock(self: *Function) std.mem.Allocator.Error!Block {
        const index: u32 = @intCast(self.blocks.items.len);
        try self.blocks.append(self.allocator, .{ .params = .empty, .insts = .empty });
        return @enumFromInt(index);
    }

    /// Append an instruction to a block, returning the result value it defines.
    pub fn appendInst(self: *Function, block: Block, ty: Type, op: Opcode) std.mem.Allocator.Error!Value {
        const inst: Inst = @enumFromInt(@as(u32, @intCast(self.insts.items.len)));
        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));

        try self.values.append(self.allocator, .{ .ty = ty, .def = .{ .inst_result = inst } });
        try self.insts.append(self.allocator, .{ .op = op, .result = value });
        try self.blocks.items[@intFromEnum(block)].insts.append(self.allocator, inst);
        return value;
    }

    /// Append a typed parameter to a block, returning the value it introduces.
    pub fn appendBlockParam(self: *Function, block: Block, ty: Type) std.mem.Allocator.Error!Value {
        const data = &self.blocks.items[@intFromEnum(block)];
        const param_index: u32 = @intCast(data.params.items.len);

        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));
        try self.values.append(self.allocator, .{
            .ty = ty,
            .def = .{ .block_param = .{ .block = block, .index = param_index } },
        });
        try data.params.append(self.allocator, value);
        return value;
    }

    /// The type of a value.
    pub fn valueType(self: *const Function, value: Value) Type {
        return self.values.items[@intFromEnum(value)].ty;
    }

    /// Retype a value in place (its defining instruction / param is unchanged). Used by
    /// the SIMD widener to re-type a scalar value to a packed vector without rebuilding
    /// the SSA graph. The caller is responsible for keeping the def consistent (e.g.
    /// splatting a scalar constant that becomes a vector).
    pub fn setValueType(self: *Function, value: Value, ty: Type) void {
        self.values.items[@intFromEnum(value)].ty = ty;
    }

    /// The textual name number of a value: its position in a deterministic walk
    /// (block by block, parameters then instruction results). A pure function of
    /// structure, so the text round-trips.
    fn valueName(self: *const Function, value: Value) u32 {
        var n: u32 = 0;
        for (self.blocks.items) |block| {
            for (block.params.items) |param| {
                if (param == value) return n;
                n += 1;
            }
            for (block.insts.items) |inst| {
                if (self.insts.items[@intFromEnum(inst)].result) |result| {
                    if (result == value) return n;
                    n += 1;
                }
            }
        }
        return n;
    }

    /// Set the block's terminator.
    pub fn setTerminator(self: *Function, block: Block, term: Terminator) void {
        self.blocks.items[@intFromEnum(block)].term = term;
    }

    /// Terminate a block with an unconditional jump, passing `args` to the
    /// target's block parameters.
    pub fn setJump(self: *Function, block: Block, target: Block, args: []const Value) std.mem.Allocator.Error!void {
        const list = try self.internValues(args);
        self.setTerminator(block, .{ .jump = .{ .target = target, .args = list } });
    }

    /// Append a result-less instruction (a statement) to a block, returning the
    /// instruction handle. Primitive behind the typed statement builders, also
    /// used to reconstruct instructions during deserialization.
    pub fn appendStmtRaw(self: *Function, block: Block, op: Opcode) std.mem.Allocator.Error!Inst {
        const inst: Inst = @enumFromInt(@as(u32, @intCast(self.insts.items.len)));
        try self.insts.append(self.allocator, .{ .op = op, .result = null });
        try self.blocks.items[@intFromEnum(block)].insts.append(self.allocator, inst);
        return inst;
    }

    /// Append a result-less instruction (a statement) to a block.
    fn appendStmt(self: *Function, block: Block, op: Opcode) std.mem.Allocator.Error!void {
        _ = try self.appendStmtRaw(block, op);
    }

    /// Append a non-terminating conditional to a block. Control continues to the
    /// block's terminator afterward.
    pub fn appendIf(self: *Function, block: Block, cond: Value, then_edge: EdgeDesc, else_edge: EdgeDesc) std.mem.Allocator.Error!void {
        const then_jump: Jump = .{ .target = then_edge.target, .args = try self.internValues(then_edge.args) };
        const else_jump: Jump = .{ .target = else_edge.target, .args = try self.internValues(else_edge.args) };
        try self.appendStmt(block, .{ .@"if" = .{ .cond = cond, .then = then_jump, .@"else" = else_jump } });
    }

    /// Append a store to a block.
    pub fn appendStore(self: *Function, block: Block, value: Value, ptr: Value) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .store = .{ .value = value, .ptr = ptr } });
    }

    /// Append `lhs <op> imm`, returning the result value of type `ty`.
    pub fn appendArithImm(self: *Function, block: Block, ty: Type, op: BinOp, lhs: Value, imm: i64) std.mem.Allocator.Error!Value {
        return self.appendInst(block, ty, .{ .arith_imm = .{ .op = op, .lhs = lhs, .imm = imm } });
    }

    /// Append a call to `name` with `args`, returning the result value of type `ty`.
    pub fn appendCall(self: *Function, block: Block, ty: Type, name: []const u8, args: []const Value) std.mem.Allocator.Error!Value {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        return self.appendInst(block, ty, .{ .call = .{ .symbol = symbol, .args = list } });
    }

    /// Append an indirect call through `target` with `args`, returning the result.
    pub fn appendCallIndirect(self: *Function, block: Block, ty: Type, target: Value, args: []const Value) std.mem.Allocator.Error!Value {
        const list = try self.internValues(args);
        return self.appendInst(block, ty, .{ .call_indirect = .{ .target = target, .args = list } });
    }

    /// Append a `global_addr` for the named symbol, returning its address as `ty`
    /// (which should be `ptr`). The symbol is resolved at link time.
    pub fn appendGlobalAddr(self: *Function, block: Block, ty: Type, name: []const u8) std.mem.Allocator.Error!Value {
        const symbol = try self.internSymbol(name);
        return self.appendInst(block, ty, .{ .global_addr = .{ .symbol = symbol } });
    }

    /// Append a call to `name` with `args` that discards its result (a statement).
    pub fn appendVoidCall(self: *Function, block: Block, name: []const u8, args: []const Value) std.mem.Allocator.Error!void {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call = .{ .symbol = symbol, .args = list } });
    }

    /// Append an aggregate construction, returning the struct value. `ty` must be
    /// the struct type of the field values.
    pub fn appendStructNew(self: *Function, block: Block, ty: Type, fields: []const Value) std.mem.Allocator.Error!Value {
        const list = try self.internValues(fields);
        return self.appendInst(block, ty, .{ .struct_new = .{ .fields = list } });
    }

    /// The instructions of a block, in order.
    pub fn blockInsts(self: *const Function, block: Block) []const Inst {
        return self.blocks.items[@intFromEnum(block)].insts.items;
    }

    /// The number of blocks in the function.
    pub fn blockCount(self: *const Function) usize {
        return self.blocks.items.len;
    }

    /// The parameters of a block, in order.
    pub fn blockParams(self: *const Function, block: Block) []const Value {
        return self.blocks.items[@intFromEnum(block)].params.items;
    }

    /// The value an instruction defines, if any.
    pub fn instResult(self: *const Function, inst: Inst) ?Value {
        return self.insts.items[@intFromEnum(inst)].result;
    }

    /// The number of instructions in the function.
    pub fn instCount(self: *const Function) usize {
        return self.insts.items.len;
    }

    // mutable accessors, for passes like legalization

    /// A mutable pointer to an instruction's opcode (to rewrite operands).
    pub fn opcodeMut(self: *Function, inst: Inst) *Opcode {
        return &self.insts.items[@intFromEnum(inst)].op;
    }

    /// A mutable pointer to a block's terminator slot.
    pub fn terminatorPtr(self: *Function, block: Block) *?Terminator {
        return &self.blocks.items[@intFromEnum(block)].term;
    }

    /// A mutable view of a value-list run (to rewrite variadic operands).
    pub fn valueListMut(self: *Function, list: ValueList) []Value {
        return self.value_lists.items[list.start..][0..list.len];
    }

    /// Replace every use of `from` with `to` across instruction operands, `if`
    /// edge arguments, and terminators (an SSA "replace all uses with").
    /// Definitions are untouched. The caller handles any now-dead `from`.
    pub fn replaceAllUses(self: *Function, from: Value, to: Value) void {
        const r = struct {
            fn repl(f: Value, t: Value, v: Value) Value {
                return if (v == f) t else v;
            }
        }.repl;
        for (0..self.instCount()) |i| {
            const op = self.opcodeMut(@enumFromInt(i));
            switch (op.*) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |*a| {
                    a.lhs = r(from, to, a.lhs);
                    a.rhs = r(from, to, a.rhs);
                },
                .arith_imm => |*a| a.lhs = r(from, to, a.lhs),
                .icmp => |*c| {
                    c.lhs = r(from, to, c.lhs);
                    c.rhs = r(from, to, c.rhs);
                },
                .select => |*s| {
                    s.cond = r(from, to, s.cond);
                    s.then = r(from, to, s.then);
                    s.@"else" = r(from, to, s.@"else");
                },
                .extract => |*e| e.aggregate = r(from, to, e.aggregate),
                .convert => |*cv| cv.value = r(from, to, cv.value),
                .unary => |*u| u.value = r(from, to, u.value),
                .load => |*l| l.ptr = r(from, to, l.ptr),
                .store => |*st| {
                    st.value = r(from, to, st.value);
                    st.ptr = r(from, to, st.ptr);
                },
                .struct_new => |sn| for (self.valueListMut(sn.fields)) |*f| {
                    f.* = r(from, to, f.*);
                },
                .call => |c| for (self.valueListMut(c.args)) |*arg| {
                    arg.* = r(from, to, arg.*);
                },
                .call_indirect => |*c| {
                    c.target = r(from, to, c.target);
                    for (self.valueListMut(c.args)) |*arg| arg.* = r(from, to, arg.*);
                },
                .@"if" => |*cf| {
                    cf.cond = r(from, to, cf.cond);
                    for (self.valueListMut(cf.then.args)) |*arg| arg.* = r(from, to, arg.*);
                    for (self.valueListMut(cf.@"else".args)) |*arg| arg.* = r(from, to, arg.*);
                },
            }
        }
        for (0..self.blockCount()) |bi| {
            const term = self.terminatorPtr(@enumFromInt(bi));
            if (term.*) |*t| switch (t.*) {
                .ret => |*v| if (v.*) |vv| {
                    v.* = r(from, to, vv);
                },
                .jump => |*j| for (self.valueListMut(j.args)) |*arg| {
                    arg.* = r(from, to, arg.*);
                },
            };
        }
    }

    /// A mutable pointer to a block's instruction list (to drop instructions).
    pub fn blockInstsMut(self: *Function, block: Block) *std.ArrayList(Inst) {
        return &self.blocks.items[@intFromEnum(block)].insts;
    }

    /// Create a fresh block-parameter value of type `ty`, without adding it to
    /// any block's parameter list. The caller installs it via `setBlockParams`.
    pub fn newParam(self: *Function, block: Block, ty: Type) std.mem.Allocator.Error!Value {
        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));
        try self.values.append(self.allocator, .{
            .ty = ty,
            .def = .{ .block_param = .{ .block = block, .index = 0 } },
        });
        return value;
    }

    /// Replace a block's parameter list with `params`.
    pub fn setBlockParams(self: *Function, block: Block, params: []const Value) std.mem.Allocator.Error!void {
        const data = &self.blocks.items[@intFromEnum(block)];
        data.params.clearRetainingCapacity();
        try data.params.appendSlice(self.allocator, params);
    }

    /// Create an instruction and its result value without adding it to any
    /// block. The caller places it via `setBlockInsts`.
    pub fn createInst(self: *Function, ty: Type, op: Opcode) std.mem.Allocator.Error!Value {
        const inst: Inst = @enumFromInt(@as(u32, @intCast(self.insts.items.len)));
        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));
        try self.values.append(self.allocator, .{ .ty = ty, .def = .{ .inst_result = inst } });
        try self.insts.append(self.allocator, .{ .op = op, .result = value });
        return value;
    }

    /// Replace a block's instruction list with `insts`.
    pub fn setBlockInsts(self: *Function, block: Block, insts: []const Inst) std.mem.Allocator.Error!void {
        const data = &self.blocks.items[@intFromEnum(block)];
        data.insts.clearRetainingCapacity();
        try data.insts.appendSlice(self.allocator, insts);
    }

    /// Intern a run of values into the value-list pool (for variadic operands).
    pub fn internValueList(self: *Function, vals: []const Value) std.mem.Allocator.Error!ValueList {
        return self.internValues(vals);
    }

    /// The total number of values in the function.
    pub fn valueCount(self: *const Function) usize {
        return self.values.items.len;
    }

    /// The argument values a jump (or branch edge) passes to its target's parameters.
    pub fn blockArgs(self: *const Function, jump: Jump) []const Value {
        return self.valueList(jump.args);
    }

    /// Copy values into the value-list pool, returning a handle to the run.
    pub fn internValues(self: *Function, vals: []const Value) std.mem.Allocator.Error!ValueList {
        const start: u32 = @intCast(self.value_lists.items.len);
        try self.value_lists.appendSlice(self.allocator, vals);
        return .{ .start = start, .len = @intCast(vals.len) };
    }

    /// Resolve a value-list handle to its slice.
    pub fn valueList(self: *const Function, list: ValueList) []const Value {
        return self.value_lists.items[list.start..][0..list.len];
    }

    /// The block's terminator, or null if it has not been set yet.
    pub fn terminator(self: *const Function, block: Block) ?Terminator {
        return self.blocks.items[@intFromEnum(block)].term;
    }

    /// The opcode of an instruction.
    pub fn opcode(self: *const Function, inst: Inst) Opcode {
        return self.insts.items[@intFromEnum(inst)].op;
    }

    /// The instruction that defines a value, or null if it is a block parameter.
    pub fn definingInst(self: *const Function, value: Value) ?Inst {
        return switch (self.values.items[@intFromEnum(value)].def) {
            .inst_result => |inst| inst,
            .block_param => null,
        };
    }

    /// Render the function in Vulcan's functional text format.
    pub fn format(self: *const Function, w: *std.Io.Writer) std.Io.Writer.Error!void {
        var func_attrs = self.attributesOf(.func);
        while (func_attrs.next()) |attr| {
            try w.writeAll("#[");
            try printAttr(w, attr);
            try w.writeAll("]\n");
        }
        try w.writeAll("fn {\n");
        for (self.blocks.items, 0..) |block, bi| {
            try w.print("  block{d}(", .{bi});
            for (block.params.items, 0..) |param, pi| {
                if (pi != 0) try w.writeAll(", ");
                try w.print("v{d}: {f}", .{ self.valueName(param), self.types.fmt(self.valueType(param)) });
            }
            try w.writeAll("):\n");

            for (block.insts.items) |inst| {
                if (self.instResult(inst)) |result| {
                    var attrs = self.attributesOf(.{ .value = result });
                    while (attrs.next()) |attr| {
                        try w.writeAll("    #[");
                        try printAttr(w, attr);
                        try w.writeAll("]\n");
                    }
                }
                try w.writeAll("    ");
                try printInst(self, w, inst);
                try w.writeByte('\n');
            }

            // An unset terminator prints as an implicit `ret void`.
            try w.writeAll("    ");
            try printTerminator(self, w, block.term);
            try w.writeByte('\n');

            // Blocks that transfer control onward (`if` or `jump`) get a trailing
            // blank line. Purely returning blocks do not.
            if (self.branchesOut(block)) try w.writeByte('\n');
        }
        try w.writeAll("}");
    }

    /// Whether a block transfers control to other blocks (has a conditional or a
    /// jump terminator).
    fn branchesOut(self: *const Function, block: BlockData) bool {
        for (block.insts.items) |inst| {
            switch (self.insts.items[@intFromEnum(inst)].op) {
                .@"if" => return true,
                else => {},
            }
        }
        if (block.term) |term| switch (term) {
            .jump => return true,
            .ret => {},
        };
        return false;
    }
};

/// Render an attribute's body (the text inside the `#[...]`).
fn printAttr(w: *std.Io.Writer, attr: Attribute) std.Io.Writer.Error!void {
    switch (attr) {
        .@"inline" => try w.writeAll("inline"),
        .noreturn => try w.writeAll("noreturn"),
        .cold => try w.writeAll("cold"),
        .@"align" => |a| try w.print("align({d})", .{a}),
        .endian => |e| try w.print("endian({s})", .{@tagName(e)}),
        .custom => |c| {
            try w.print("{s}.{s}", .{ c.namespace, c.key });
            switch (c.value) {
                .flag => {},
                .int => |i| try w.print(" = {d}", .{i}),
                .string => |s| try w.print(" = \"{s}\"", .{s}),
            }
        },
    }
}

/// Render an instruction statement. Constants bind with `const` and carry a type
/// annotation. Other results bind with `let`.
fn printInst(self: *const Function, w: *std.Io.Writer, inst: Inst) std.Io.Writer.Error!void {
    const data = self.insts.items[@intFromEnum(inst)];
    switch (data.op) {
        .iconst => |value| try w.print("const v{d}: {f} = {d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            value,
        }),
        .fconst => |value| try w.print("const v{d}: {f} = {d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            value,
        }),
        .arith => |a| try w.print("let v{d} = v{d} {s} v{d}", .{
            self.valueName(data.result.?),
            self.valueName(a.lhs),
            a.op.symbol(),
            self.valueName(a.rhs),
        }),
        .arith_imm => |a| try w.print("let v{d} = v{d} {s} {d}", .{
            self.valueName(data.result.?),
            self.valueName(a.lhs),
            a.op.symbol(),
            a.imm,
        }),
        .icmp => |cmp| try w.print("let v{d} = v{d} {s} v{d}", .{
            self.valueName(data.result.?),
            self.valueName(cmp.lhs),
            cmp.op.symbol(),
            self.valueName(cmp.rhs),
        }),
        .select => |sel| {
            try w.print("v{d} := if v{d} ", .{ self.valueName(data.result.?), self.valueName(sel.cond) });
            try w.writeAll("{ ");
            try w.print("v{d}", .{self.valueName(sel.then)});
            try w.writeAll(" } else { ");
            try w.print("v{d}", .{self.valueName(sel.@"else")});
            try w.writeAll(" }");
        },
        .struct_new => |sn| {
            try w.print("let v{d} = struct ", .{self.valueName(data.result.?)});
            try w.writeAll("{ ");
            for (self.valueList(sn.fields), 0..) |field, i| {
                if (i != 0) try w.writeAll(", ");
                try w.print("v{d}", .{self.valueName(field)});
            }
            try w.writeAll(" }");
        },
        .extract => |ex| try w.print("let v{d} = v{d}.#{d}", .{
            self.valueName(data.result.?),
            self.valueName(ex.aggregate),
            ex.index,
        }),
        .alloca => |al| try w.print("let v{d} = alloca {f}", .{
            self.valueName(data.result.?),
            self.types.fmt(al.elem),
        }),
        .global_addr => |ga| try w.print("let v{d} = global_addr @{s}", .{
            self.valueName(data.result.?),
            self.symbolName(ga.symbol),
        }),
        .call => |c| {
            if (data.result) |res| {
                try w.print("let v{d} = call {f} @{s}(", .{
                    self.valueName(res),
                    self.types.fmt(self.valueType(res)),
                    self.symbolName(c.symbol),
                });
            } else {
                try w.print("call @{s}(", .{self.symbolName(c.symbol)});
            }
            for (self.valueList(c.args), 0..) |arg, i| {
                if (i != 0) try w.writeAll(", ");
                try w.print("v{d}", .{self.valueName(arg)});
            }
            try w.writeAll(")");
        },
        .call_indirect => |c| {
            if (data.result) |res| {
                try w.print("let v{d} = call_indirect {f} v{d}(", .{ self.valueName(res), self.types.fmt(self.valueType(res)), self.valueName(c.target) });
            } else {
                try w.print("call_indirect v{d}(", .{self.valueName(c.target)});
            }
            for (self.valueList(c.args), 0..) |arg, i| {
                if (i != 0) try w.writeAll(", ");
                try w.print("v{d}", .{self.valueName(arg)});
            }
            try w.writeAll(")");
        },
        .convert => |cv| try w.print("let v{d} = convert {f}, v{d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            self.valueName(cv.value),
        }),
        .unary => |u| try w.print("let v{d} = {s} {f}, v{d}", .{
            self.valueName(data.result.?),
            @tagName(u.op),
            self.types.fmt(self.valueType(data.result.?)),
            self.valueName(u.value),
        }),
        .load => |ld| try w.print("let v{d} = load {f}, v{d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            self.valueName(ld.ptr),
        }),
        .store => |st| try w.print("store v{d}, v{d}", .{ self.valueName(st.value), self.valueName(st.ptr) }),
        .@"if" => |cond| {
            try w.print("if v{d} ", .{self.valueName(cond.cond)});
            try w.writeAll("{ ");
            try printEdge(self, w, cond.then);
            try w.writeAll(" } else { ");
            try printEdge(self, w, cond.@"else");
            try w.writeAll(" }");
        },
    }
}

/// Render a jump's target and the arguments it passes, e.g. `block1(v3, v4)`.
fn printEdge(self: *const Function, w: *std.Io.Writer, jump: Jump) std.Io.Writer.Error!void {
    try w.print("block{d}(", .{@intFromEnum(jump.target)});
    for (self.blockArgs(jump), 0..) |arg, i| {
        if (i != 0) try w.writeAll(", ");
        try w.print("v{d}", .{self.valueName(arg)});
    }
    try w.writeAll(")");
}

fn printTerminator(self: *const Function, w: *std.Io.Writer, term: ?Terminator) std.Io.Writer.Error!void {
    const t = term orelse {
        // An unset terminator is an implicit void return.
        try w.writeAll("ret void");
        return;
    };
    switch (t) {
        .ret => |value| {
            if (value) |v| {
                try w.print("ret v{d}", .{self.valueName(v)});
            } else {
                try w.writeAll("ret void");
            }
        },
        .jump => |j| try printEdge(self, w, j),
    }
}

test "attributes attach to an entity and read back" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 1 });

    try func.addAttr(.{ .value = v }, .{ .@"align" = 16 });

    var it = func.attributesOf(.{ .value = v });
    try std.testing.expectEqual(Attribute{ .@"align" = 16 }, it.next().?);
    try std.testing.expectEqual(@as(?Attribute, null), it.next());
}

test "namespaced attributes carry a typed value and are owned" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    try func.addAttr(.func, .{ .custom = .{
        .namespace = "target",
        .key = "clone",
        .value = .{ .string = "rv64gcv" },
    } });

    var it = func.attributesOf(.func);
    const got = it.next().?;
    try std.testing.expectEqualStrings("target", got.custom.namespace);
    try std.testing.expectEqualStrings("clone", got.custom.key);
    try std.testing.expectEqualStrings("rv64gcv", got.custom.value.string);
}

test "a function creates distinct blocks" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const a = try func.appendBlock();
    const b = try func.appendBlock();

    try std.testing.expect(a != b);
}

test "block parameters are typed values" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const p = try func.appendBlockParam(block, i32_t);

    try std.testing.expectEqual(i32_t, func.valueType(p));
}

test "an instruction result is a typed value" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v = try func.appendInst(block, i32_t, .{ .iconst = 42 });

    try std.testing.expectEqual(i32_t, func.valueType(v));
}

test "store writes a value to a pointer" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.appendStore(entry, x, p);

    const insts = func.blockInsts(entry);
    const op = func.opcode(insts[insts.len - 1]);
    try std.testing.expectEqual(x, op.store.value);
    try std.testing.expectEqual(p, op.store.ptr);
}

test "struct construction prints its fields" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const s = try func.appendStructNew(entry, st, &.{ a, b });
    func.setTerminator(entry, .{ .ret = s });

    try std.testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = struct { v0, v1 }
        \\    ret v2
        \\}
    , "{f}", .{func});
}

test "load reads a typed value from a pointer" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });

    try std.testing.expectEqual(i32_t, func.valueType(v));
    try std.testing.expectEqual(p, func.opcode(func.definingInst(v).?).load.ptr);
}

test "select picks between two values and is typed" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, i32_t, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } });

    try std.testing.expectEqual(i32_t, func.valueType(c));
    const op = func.opcode(func.definingInst(c).?);
    try std.testing.expectEqual(cond, op.select.cond);
    try std.testing.expectEqual(a, op.select.then);
    try std.testing.expectEqual(b, op.select.@"else");
}

test "icmp produces a bool and records its comparison" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });

    try std.testing.expectEqual(bool_t, func.valueType(c));
    const op = func.opcode(func.definingInst(c).?);
    try std.testing.expectEqual(CmpOp.gt, op.icmp.op);
    try std.testing.expectEqual(a, op.icmp.lhs);
    try std.testing.expectEqual(b, op.icmp.rhs);
}

test "arith records its operator and operands" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendInst(block, i32_t, .{ .iconst = 10 });
    const b = try func.appendInst(block, i32_t, .{ .iconst = 20 });
    const sum = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });

    const inst = func.definingInst(sum).?;
    const op = func.opcode(inst);
    try std.testing.expectEqual(BinOp.add, op.arith.op);
    try std.testing.expectEqual(a, op.arith.lhs);
    try std.testing.expectEqual(b, op.arith.rhs);
}

test "a block can be terminated with a return" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v = try func.appendInst(block, i32_t, .{ .iconst = 42 });
    func.setTerminator(block, .{ .ret = v });

    try std.testing.expectEqual(Terminator{ .ret = v }, func.terminator(block).?);
}

test "a jump passes arguments to its target block params" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    const entry = try func.appendBlock();
    const target = try func.appendBlock();
    _ = try func.appendBlockParam(target, i32_t);

    const v = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    try func.setJump(entry, target, &.{v});

    const term = func.terminator(entry).?;
    try std.testing.expectEqual(target, term.jump.target);
    try std.testing.expectEqualSlices(Value, &.{v}, func.blockArgs(term.jump));
}

test "a conditional selects between two targets" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{x} }, .{ .target = else_b });

    // The conditional is an instruction in the block body, not a terminator.
    const if_inst = func.blockInsts(entry)[func.blockInsts(entry).len - 1];
    const cf = func.opcode(if_inst).@"if";
    try std.testing.expectEqual(cond, cf.cond);
    try std.testing.expectEqual(then_b, cf.then.target);
    try std.testing.expectEqual(else_b, cf.@"else".target);
    try std.testing.expectEqualSlices(Value, &.{x}, func.blockArgs(cf.then));
    try std.testing.expectEqual(@as(?Terminator, null), func.terminator(entry));
}

test "printing a minimal function" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 42 });
    func.setTerminator(entry, .{ .ret = v });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    const v0: i32 = 42
        \\    ret v0
        \\}
    , "{f}", .{func});
}

test "printing a call names the callee, result type, and arguments" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const r = try func.appendCall(entry, i32_t, "add", &.{ a, b });
    func.setTerminator(entry, .{ .ret = r });

    try std.testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = call i32 @add(v0, v1)
        \\    ret v2
        \\}
    , "{f}", .{func});
}

test "printing an alloca names the slot type" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    func.setTerminator(entry, .{ .ret = p });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    let v0 = alloca i32
        \\    ret v0
        \\}
    , "{f}", .{func});
}

test "printing a convert names its target type and source" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const i = try func.appendInst(entry, i32_t, .{ .iconst = 3 });
    const f = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = i } });
    func.setTerminator(entry, .{ .ret = f });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    const v0: i32 = 3
        \\    let v1 = convert f32, v0
        \\    ret v1
        \\}
    , "{f}", .{func});
}

test "printing a function with params, iadd, and a jump" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const exit = try func.appendBlock();
    const r = try func.appendBlockParam(exit, i32_t);

    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    try func.setJump(entry, exit, &.{sum});
    func.setTerminator(exit, .{ .ret = r });

    try std.testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 + v1
        \\    block1(v2)
        \\
        \\  block1(v3: i32):
        \\    ret v3
        \\}
    , "{f}", .{func});
}

test "printing a conditional branch" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{x} }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = null });
    func.setTerminator(else_b, .{ .ret = null });

    try std.testing.expectFmt(
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
    , "{f}", .{func});
}
