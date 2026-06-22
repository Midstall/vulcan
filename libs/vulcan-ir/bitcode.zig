//! Binary IR serialization ("bitcode"): a compact, position-independent encoding
//! of a `Function` and a decoder that rebuilds an equivalent one. The binary
//! analog of the text printer/parser, used by cross-module work (LTO/PGO).
//!
//! Values are referenced by serial number (their order in a canonical walk: per
//! block, parameters then instruction results), not by raw handle, so the decoder
//! rebuilds the function with fresh handles and stays isomorphic. Round-trip
//! oracle: `print(decode(encode(f))) == print(f)`.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Value = function.Value;
const Inst = function.Inst;
const Block = function.Block;
const Type = types.Type;
const Opcode = function.Opcode;

pub const Error = std.mem.Allocator.Error || error{MalformedBitcode};

const magic = "VBC1";

// Opcode tags (stable on the wire).
const op_iconst: u8 = 0;
const op_fconst: u8 = 1;
const op_arith: u8 = 2;
const op_arith_imm: u8 = 3;
const op_icmp: u8 = 4;
const op_select: u8 = 5;
const op_struct_new: u8 = 6;
const op_extract: u8 = 7;
const op_convert: u8 = 8;
const op_alloca: u8 = 9;
const op_call: u8 = 10;
const op_load: u8 = 11;
const op_store: u8 = 12;
const op_if: u8 = 13;
const op_global_addr: u8 = 14;
const op_unary: u8 = 15;
const op_call_indirect: u8 = 16;

const Writer = struct {
    bytes: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    fn u8v(self: *Writer, v: u8) Error!void {
        try self.bytes.append(self.allocator, v);
    }
    fn u16v(self: *Writer, v: u16) Error!void {
        try self.bytes.appendSlice(self.allocator, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
    }
    fn u32v(self: *Writer, v: u32) Error!void {
        try self.bytes.appendSlice(self.allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    }
    fn u64v(self: *Writer, v: u64) Error!void {
        try self.bytes.appendSlice(self.allocator, &std.mem.toBytes(std.mem.nativeToLittle(u64, v)));
    }
};

/// Serialize `func` into bitcode. The caller owns the returned bytes.
pub fn encode(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    var w = Writer{ .allocator = allocator };
    errdefer w.bytes.deinit(allocator);

    // Canonical serial number for each value (params then results, block order).
    const serial = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(serial);
    {
        var next: u32 = 0;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockParams(block)) |p| {
                serial[@intFromEnum(p)] = next;
                next += 1;
            }
            for (func.blockInsts(block)) |inst| {
                if (func.instResult(inst)) |r| {
                    serial[@intFromEnum(r)] = next;
                    next += 1;
                }
            }
        }
    }
    const sv = struct {
        fn of(s: []const u32, v: Value) u32 {
            return s[@intFromEnum(v)];
        }
    }.of;

    try w.bytes.appendSlice(allocator, magic);

    // Types (interned in dependency order, so a kind's nested types precede it).
    try w.u32v(@intCast(func.types.count()));
    for (0..func.types.count()) |i| try writeType(&w, func.types.type_kind(@enumFromInt(i)));

    // Symbols.
    try w.u32v(@intCast(func.symbolCount()));
    for (0..func.symbolCount()) |i| {
        const name = func.symbolName(@intCast(i));
        try w.u32v(@intCast(name.len));
        try w.bytes.appendSlice(allocator, name);
    }

    // Blocks.
    try w.u32v(@intCast(func.blockCount()));
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        const params = func.blockParams(block);
        try w.u32v(@intCast(params.len));
        for (params) |p| try w.u32v(@intFromEnum(func.valueType(p)));

        const insts = func.blockInsts(block);
        try w.u32v(@intCast(insts.len));
        for (insts) |inst| try writeInst(&w, func, inst, serial, sv);

        try writeTerm(&w, func, block, serial, sv);
    }

    return w.bytes.toOwnedSlice(allocator);
}

fn writeType(w: *Writer, kind: types.TypeKind) Error!void {
    switch (kind) {
        .bool => try w.u8v(0),
        .int => |i| {
            try w.u8v(1);
            try w.u8v(if (i.signedness == .signed) 0 else 1);
            try w.u16v(i.bits);
        },
        .float => |f| {
            try w.u8v(2);
            try w.u8v(if (f == .f32) 0 else 1);
        },
        .ptr => try w.u8v(3),
        .vector => |v| {
            try w.u8v(4);
            try w.u32v(v.len);
            try w.u32v(@intFromEnum(v.elem));
        },
        .@"struct" => |fields| {
            try w.u8v(5);
            try w.u32v(@intCast(fields.len));
            for (fields) |f| try w.u32v(@intFromEnum(f));
        },
        .array => |a| {
            try w.u8v(6);
            try w.u64v(a.len);
            try w.u32v(@intFromEnum(a.elem));
        },
        .slice => |s| {
            try w.u8v(7);
            try w.u32v(@intFromEnum(s.elem));
        },
    }
}

fn writeInst(w: *Writer, func: *const Function, inst: Inst, serial: []const u32, sv: fn ([]const u32, Value) u32) Error!void {
    const result = func.instResult(inst);
    try w.u8v(if (result != null) 1 else 0);
    if (result) |r| try w.u32v(@intFromEnum(func.valueType(r)));

    switch (func.opcode(inst)) {
        .iconst => |v| {
            try w.u8v(op_iconst);
            try w.u64v(@bitCast(v));
        },
        .fconst => |v| {
            try w.u8v(op_fconst);
            try w.u64v(@bitCast(v));
        },
        .arith => |a| {
            try w.u8v(op_arith);
            try w.u8v(@intFromEnum(a.op));
            try w.u32v(sv(serial, a.lhs));
            try w.u32v(sv(serial, a.rhs));
        },
        .arith_imm => |a| {
            try w.u8v(op_arith_imm);
            try w.u8v(@intFromEnum(a.op));
            try w.u32v(sv(serial, a.lhs));
            try w.u64v(@bitCast(a.imm));
        },
        .icmp => |c| {
            try w.u8v(op_icmp);
            try w.u8v(@intFromEnum(c.op));
            try w.u32v(sv(serial, c.lhs));
            try w.u32v(sv(serial, c.rhs));
        },
        .select => |s| {
            try w.u8v(op_select);
            try w.u32v(sv(serial, s.cond));
            try w.u32v(sv(serial, s.then));
            try w.u32v(sv(serial, s.@"else"));
        },
        .struct_new => |sn| {
            try w.u8v(op_struct_new);
            const fields = func.valueList(sn.fields);
            try w.u32v(@intCast(fields.len));
            for (fields) |f| try w.u32v(sv(serial, f));
        },
        .extract => |e| {
            try w.u8v(op_extract);
            try w.u32v(sv(serial, e.aggregate));
            try w.u32v(e.index);
        },
        .convert => |cv| {
            try w.u8v(op_convert);
            try w.u32v(sv(serial, cv.value));
        },
        .unary => |u| {
            try w.u8v(op_unary);
            try w.u8v(@intFromEnum(u.op));
            try w.u32v(sv(serial, u.value));
        },
        .alloca => |al| {
            try w.u8v(op_alloca);
            try w.u32v(@intFromEnum(al.elem));
        },
        .call => |c| {
            try w.u8v(op_call);
            try w.u32v(c.symbol);
            const args = func.valueList(c.args);
            try w.u32v(@intCast(args.len));
            for (args) |a| try w.u32v(sv(serial, a));
        },
        .call_indirect => |c| {
            try w.u8v(op_call_indirect);
            try w.u32v(sv(serial, c.target));
            const args = func.valueList(c.args);
            try w.u32v(@intCast(args.len));
            for (args) |a| try w.u32v(sv(serial, a));
        },
        .load => |l| {
            try w.u8v(op_load);
            try w.u32v(sv(serial, l.ptr));
        },
        .store => |st| {
            try w.u8v(op_store);
            try w.u32v(sv(serial, st.value));
            try w.u32v(sv(serial, st.ptr));
        },
        .@"if" => |cf| {
            try w.u8v(op_if);
            try w.u32v(sv(serial, cf.cond));
            try writeJump(w, func, cf.then, serial, sv);
            try writeJump(w, func, cf.@"else", serial, sv);
        },
        .global_addr => |ga| {
            try w.u8v(op_global_addr);
            try w.u32v(ga.symbol);
        },
    }
}

fn writeJump(w: *Writer, func: *const Function, jump: function.Jump, serial: []const u32, sv: fn ([]const u32, Value) u32) Error!void {
    try w.u32v(@intFromEnum(jump.target));
    const args = func.blockArgs(jump);
    try w.u32v(@intCast(args.len));
    for (args) |a| try w.u32v(sv(serial, a));
}

fn writeTerm(w: *Writer, func: *const Function, block: Block, serial: []const u32, sv: fn ([]const u32, Value) u32) Error!void {
    const term = func.terminator(block) orelse {
        try w.u8v(0); // no terminator (implicit ret void)
        return;
    };
    switch (term) {
        .ret => |v| {
            try w.u8v(1);
            try w.u8v(if (v != null) 1 else 0);
            if (v) |vv| try w.u32v(sv(serial, vv));
        },
        .jump => |j| {
            try w.u8v(2);
            try writeJump(w, func, j, serial, sv);
        },
    }
}

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, comptime T: type) Error!T {
        const n = @sizeOf(T);
        if (self.pos + n > self.bytes.len) return error.MalformedBitcode;
        const v = std.mem.readInt(T, self.bytes[self.pos..][0..n], .little);
        self.pos += n;
        return v;
    }
    fn takeBytes(self: *Reader, n: usize) Error![]const u8 {
        if (self.pos + n > self.bytes.len) return error.MalformedBitcode;
        const s = self.bytes[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
};

/// Decode bitcode into an equivalent `Function`. The caller owns it (`deinit`).
pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) Error!Function {
    var r = Reader{ .bytes = bytes };
    if (!std.mem.eql(u8, try r.takeBytes(4), magic)) return error.MalformedBitcode;

    var func = Function.init(allocator);
    errdefer func.deinit();

    // Types (interned in order, nested references resolve to earlier handles).
    const type_count = try r.take(u32);
    var type_map = try allocator.alloc(Type, type_count);
    defer allocator.free(type_map);
    for (0..type_count) |i| type_map[i] = try readType(&r, &func, type_map);

    // Symbols.
    const sym_count = try r.take(u32);
    for (0..sym_count) |_| {
        const len = try r.take(u32);
        _ = try func.internSymbol(try r.takeBytes(len));
    }

    // Blocks. Create them all first so jump/if targets resolve.
    const block_count = try r.take(u32);
    for (0..block_count) |_| _ = try func.appendBlock();

    // The serial->Value table, filled as values are recreated in canonical order.
    var serial: std.ArrayList(Value) = .empty;
    defer serial.deinit(allocator);

    // Records of value operands to fix once every value exists.
    var fixups: std.ArrayList(Fixup) = .empty;
    defer {
        for (fixups.items) |*f| f.deinit(allocator);
        fixups.deinit(allocator);
    }

    const dummy: Value = @enumFromInt(0);
    for (0..block_count) |bi| {
        const block: Block = @enumFromInt(bi);
        const param_count = try r.take(u32);
        for (0..param_count) |_| {
            const ty = type_map[try r.take(u32)];
            try serial.append(allocator, try func.appendBlockParam(block, ty));
        }
        const inst_count = try r.take(u32);
        for (0..inst_count) |_| try readInst(&r, &func, block, type_map, dummy, &serial, &fixups, allocator);
        try readTerm(&r, &func, block, dummy, &fixups, allocator);
    }

    // Now resolve every operand from its serial number.
    for (fixups.items) |f| f.apply(&func, serial.items);

    return func;
}

fn readType(r: *Reader, func: *Function, type_map: []const Type) Error!Type {
    return switch (try r.take(u8)) {
        0 => try func.types.intern(.bool),
        1 => blk: {
            const s: std.builtin.Signedness = if (try r.take(u8) == 0) .signed else .unsigned;
            const bits = try r.take(u16);
            break :blk try func.types.intern(.{ .int = .{ .signedness = s, .bits = bits } });
        },
        2 => try func.types.intern(.{ .float = if (try r.take(u8) == 0) .f32 else .f64 }),
        3 => try func.types.intern(.ptr),
        4 => blk: {
            const len = try r.take(u32);
            const elem = type_map[try r.take(u32)];
            break :blk try func.types.intern(.{ .vector = .{ .len = len, .elem = elem } });
        },
        5 => blk: {
            const n = try r.take(u32);
            const fields = try func.allocator.alloc(Type, n);
            defer func.allocator.free(fields);
            for (fields) |*f| f.* = type_map[try r.take(u32)];
            break :blk try func.types.intern(.{ .@"struct" = fields });
        },
        6 => blk: {
            const len = try r.take(u64);
            const elem = type_map[try r.take(u32)];
            break :blk try func.types.intern(.{ .array = .{ .len = len, .elem = elem } });
        },
        7 => try func.types.intern(.{ .slice = .{ .elem = type_map[try r.take(u32)] } }),
        else => error.MalformedBitcode,
    };
}

/// A deferred operand fix: at decode time operands are recorded as serial
/// numbers (in `slots`), then patched to real values once all values exist.
const Fixup = struct {
    target: union(enum) {
        inst: Inst,
        terminator: Block,
    },
    slots: []u32,

    fn deinit(self: *Fixup, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
    }

    fn apply(self: Fixup, func: *Function, serial: []const Value) void {
        var i: usize = 0;
        const next = struct {
            fn n(idx: *usize, sl: []const u32, srl: []const Value) Value {
                const v = srl[sl[idx.*]];
                idx.* += 1;
                return v;
            }
        }.n;
        switch (self.target) {
            .inst => |inst| {
                const op = func.opcodeMut(inst);
                switch (op.*) {
                    .iconst, .fconst, .alloca, .global_addr => {},
                    .arith => |*a| {
                        a.lhs = next(&i, self.slots, serial);
                        a.rhs = next(&i, self.slots, serial);
                    },
                    .arith_imm => |*a| a.lhs = next(&i, self.slots, serial),
                    .icmp => |*c| {
                        c.lhs = next(&i, self.slots, serial);
                        c.rhs = next(&i, self.slots, serial);
                    },
                    .select => |*s| {
                        s.cond = next(&i, self.slots, serial);
                        s.then = next(&i, self.slots, serial);
                        s.@"else" = next(&i, self.slots, serial);
                    },
                    .extract => |*e| e.aggregate = next(&i, self.slots, serial),
                    .convert => |*cv| cv.value = next(&i, self.slots, serial),
                    .unary => |*u| u.value = next(&i, self.slots, serial),
                    .load => |*l| l.ptr = next(&i, self.slots, serial),
                    .store => |*st| {
                        st.value = next(&i, self.slots, serial);
                        st.ptr = next(&i, self.slots, serial);
                    },
                    .struct_new => |sn| for (func.valueListMut(sn.fields)) |*f| {
                        f.* = next(&i, self.slots, serial);
                    },
                    .call => |c| for (func.valueListMut(c.args)) |*a| {
                        a.* = next(&i, self.slots, serial);
                    },
                    .call_indirect => |*c| {
                        c.target = next(&i, self.slots, serial);
                        for (func.valueListMut(c.args)) |*a| a.* = next(&i, self.slots, serial);
                    },
                    .@"if" => |*cf| {
                        cf.cond = next(&i, self.slots, serial);
                        for (func.valueListMut(cf.then.args)) |*a| a.* = next(&i, self.slots, serial);
                        for (func.valueListMut(cf.@"else".args)) |*a| a.* = next(&i, self.slots, serial);
                    },
                }
            },
            .terminator => |block| {
                const tptr = func.terminatorPtr(block);
                if (tptr.* == null) return;
                switch (tptr.*.?) {
                    .ret => |v| {
                        if (v != null) tptr.* = .{ .ret = next(&i, self.slots, serial) };
                    },
                    .jump => |j| for (func.valueListMut(j.args)) |*a| {
                        a.* = next(&i, self.slots, serial);
                    },
                }
            },
        }
    }
};

fn readInst(r: *Reader, func: *Function, block: Block, type_map: []const Type, dummy: Value, serial: *std.ArrayList(Value), fixups: *std.ArrayList(Fixup), allocator: std.mem.Allocator) Error!void {
    const has_result = (try r.take(u8)) != 0;
    const rty: Type = if (has_result) type_map[try r.take(u32)] else undefined;
    const tag = try r.take(u8);

    var slots: std.ArrayList(u32) = .empty;
    errdefer slots.deinit(allocator);

    // Build the instruction with placeholder operands, recording the serials.
    const inst: Inst = switch (tag) {
        op_iconst => try appendRes(func, block, serial, rty, .{ .iconst = @bitCast(try r.take(u64)) }),
        op_fconst => try appendRes(func, block, serial, rty, .{ .fconst = @bitCast(try r.take(u64)) }),
        op_arith => blk: {
            const op: function.BinOp = @enumFromInt(try r.take(u8));
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .arith = .{ .op = op, .lhs = dummy, .rhs = dummy } });
        },
        op_arith_imm => blk: {
            const op: function.BinOp = @enumFromInt(try r.take(u8));
            try slots.append(allocator, try r.take(u32));
            const imm: i64 = @bitCast(try r.take(u64));
            break :blk try appendRes(func, block, serial, rty, .{ .arith_imm = .{ .op = op, .lhs = dummy, .imm = imm } });
        },
        op_icmp => blk: {
            const op: function.CmpOp = @enumFromInt(try r.take(u8));
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .icmp = .{ .op = op, .lhs = dummy, .rhs = dummy } });
        },
        op_select => blk: {
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .select = .{ .cond = dummy, .then = dummy, .@"else" = dummy } });
        },
        op_struct_new => blk: {
            const n = try r.take(u32);
            for (0..n) |_| try slots.append(allocator, try r.take(u32));
            const list = try internDummies(func, n, dummy);
            break :blk try appendRes(func, block, serial, rty, .{ .struct_new = .{ .fields = list } });
        },
        op_extract => blk: {
            try slots.append(allocator, try r.take(u32));
            const index = try r.take(u32);
            break :blk try appendRes(func, block, serial, rty, .{ .extract = .{ .aggregate = dummy, .index = index } });
        },
        op_convert => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .convert = .{ .value = dummy } });
        },
        op_unary => blk: {
            const uop = try r.take(u8);
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .unary = .{ .op = @enumFromInt(uop), .value = dummy } });
        },
        op_alloca => try appendRes(func, block, serial, rty, .{ .alloca = .{ .elem = type_map[try r.take(u32)] } }),
        op_call => blk: {
            const symbol = try r.take(u32);
            const n = try r.take(u32);
            for (0..n) |_| try slots.append(allocator, try r.take(u32));
            const list = try internDummies(func, n, dummy);
            const op: Opcode = .{ .call = .{ .symbol = symbol, .args = list } };
            if (has_result) {
                break :blk try appendRes(func, block, serial, rty, op);
            } else {
                break :blk try appendStmtOp(func, block, op);
            }
        },
        op_call_indirect => blk: {
            try slots.append(allocator, try r.take(u32));
            const n = try r.take(u32);
            for (0..n) |_| try slots.append(allocator, try r.take(u32));
            const list = try internDummies(func, n, dummy);
            const op: Opcode = .{ .call_indirect = .{ .target = dummy, .args = list } };
            if (has_result) {
                break :blk try appendRes(func, block, serial, rty, op);
            } else {
                break :blk try appendStmtOp(func, block, op);
            }
        },
        op_load => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .load = .{ .ptr = dummy } });
        },
        op_store => blk: {
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            break :blk try appendStmtOp(func, block, .{ .store = .{ .value = dummy, .ptr = dummy } });
        },
        op_if => blk: {
            try slots.append(allocator, try r.take(u32)); // cond
            const then_j = try readJumpDummy(r, func, &slots, dummy, allocator);
            const else_j = try readJumpDummy(r, func, &slots, dummy, allocator);
            break :blk try appendStmtOp(func, block, .{ .@"if" = .{ .cond = dummy, .then = then_j, .@"else" = else_j } });
        },
        op_global_addr => try appendRes(func, block, serial, rty, .{ .global_addr = .{ .symbol = try r.take(u32) } }),
        else => return error.MalformedBitcode,
    };

    if (slots.items.len > 0) {
        try fixups.append(allocator, .{ .target = .{ .inst = inst }, .slots = try slots.toOwnedSlice(allocator) });
    } else {
        slots.deinit(allocator);
    }
}

fn readJumpDummy(r: *Reader, func: *Function, slots: *std.ArrayList(u32), dummy: Value, allocator: std.mem.Allocator) Error!function.Jump {
    const target: Block = @enumFromInt(try r.take(u32));
    const n = try r.take(u32);
    for (0..n) |_| try slots.append(allocator, try r.take(u32));
    return .{ .target = target, .args = try internDummies(func, n, dummy) };
}

fn readTerm(r: *Reader, func: *Function, block: Block, dummy: Value, fixups: *std.ArrayList(Fixup), allocator: std.mem.Allocator) Error!void {
    var slots: std.ArrayList(u32) = .empty;
    errdefer slots.deinit(allocator);
    switch (try r.take(u8)) {
        0 => {}, // no terminator
        1 => {
            const has_value = (try r.take(u8)) != 0;
            if (has_value) {
                try slots.append(allocator, try r.take(u32));
                func.setTerminator(block, .{ .ret = dummy });
            } else {
                func.setTerminator(block, .{ .ret = null });
            }
        },
        2 => {
            const target: Block = @enumFromInt(try r.take(u32));
            const n = try r.take(u32);
            for (0..n) |_| try slots.append(allocator, try r.take(u32));
            const list = try internDummies(func, n, dummy);
            func.setTerminator(block, .{ .jump = .{ .target = target, .args = list } });
        },
        else => return error.MalformedBitcode,
    }
    if (slots.items.len > 0) {
        try fixups.append(allocator, .{ .target = .{ .terminator = block }, .slots = try slots.toOwnedSlice(allocator) });
    } else {
        slots.deinit(allocator);
    }
}

/// Append an instruction that produces a result, recording the new value.
fn appendRes(func: *Function, block: Block, serial: *std.ArrayList(Value), rty: Type, op: Opcode) Error!Inst {
    const v = try func.appendInst(block, rty, op);
    try serial.append(func.allocator, v);
    return func.definingInst(v).?;
}

/// Append a result-less instruction (store / void call / if).
fn appendStmtOp(func: *Function, block: Block, op: Opcode) Error!Inst {
    return func.appendStmtRaw(block, op);
}

/// Intern a value list of `n` placeholder values.
fn internDummies(func: *Function, n: u32, dummy: Value) Error!function.ValueList {
    if (n == 0) return func.internValues(&.{});
    const tmp = try func.allocator.alloc(Value, n);
    defer func.allocator.free(tmp);
    @memset(tmp, dummy);
    return func.internValues(tmp);
}

test "round-trips a function through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    // f(x, y): if x<y { jump m(x*y) } else { jump m(x+y) }, m(z): ret z + 1
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const merge = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const z = try func.appendBlockParam(merge, i32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    const prod = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    try func.appendIf(entry, c, .{ .target = merge, .args = &.{prod} }, .{ .target = merge, .args = &.{sum} });
    const r = try func.appendArithImm(merge, i32_t, .add, z, 1);
    func.setTerminator(merge, .{ .ret = r });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    // The text printer is a pure function of structure: round-tripped text matches.
    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "rejects truncated bitcode" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, "VBC1\x01"));
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, "nope"));
}
