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
const op_prefetch: u8 = 17;
const op_dot: u8 = 18;
const op_matmul: u8 = 19;

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
            // FloatKind now has 3 members, so this needs a full byte (was a
            // single 0/1 bit for f32/f64 only). Encode order must match
            // readType's decode order; f16 is the new third value. The decoder
            // is a hardcoded 0->f32/1->f64/2->f16 switch, so pin those tag
            // values here: a future reorder would otherwise desync the two
            // sides and silently corrupt streams instead of failing to build.
            comptime {
                std.debug.assert(@intFromEnum(types.FloatKind.f32) == 0);
                std.debug.assert(@intFromEnum(types.FloatKind.f64) == 1);
                std.debug.assert(@intFromEnum(types.FloatKind.f16) == 2);
            }
            try w.u8v(@intFromEnum(f));
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
        .prefetch => |pf| {
            try w.u8v(op_prefetch);
            try w.u32v(sv(serial, pf.ptr));
        },
        .dot => |d| {
            try w.u8v(op_dot);
            try w.u32v(sv(serial, d.acc));
            try w.u32v(sv(serial, d.a));
            try w.u32v(sv(serial, d.b));
        },
        .matmul => |mm| {
            try w.u8v(op_matmul);
            try w.u32v(sv(serial, mm.a));
            try w.u32v(sv(serial, mm.b));
            try w.u32v(sv(serial, mm.c));
            try w.u16v(mm.m);
            try w.u16v(mm.n);
            try w.u16v(mm.k);
            try w.u8v(@intFromEnum(mm.dtype)); // MatMulType (u3), widened to a byte
            try w.u8v(if (mm.accumulate) 1 else 0);
            try w.u8v(if (mm.embedded) 1 else 0); // self-contained (embedded) lowering flag
            if (mm.input_signs) |s| {
                try w.u8v(1);
                try w.u8v(if (s.a_unsigned) 1 else 0);
                try w.u8v(if (s.b_unsigned) 1 else 0);
            } else try w.u8v(0);
            if (mm.quant) |q| {
                try w.u8v(1);
                try w.u8v(if (q.relu) 1 else 0);
                try w.u8v(@intFromEnum(q.out)); // MatMulQuantOut (i8=0, u8=1)
                try w.u32v(@bitCast(q.zero_point)); // i32 zero-point as u32 bits
                if (q.bias) |bh| {
                    try w.u8v(1);
                    const bias = func.biasList(bh);
                    try w.u32v(@intCast(bias.len));
                    for (bias) |v| try w.u32v(@bitCast(v)); // i32 as u32 bits
                } else {
                    try w.u8v(0);
                }
                switch (q.scale) {
                    .scalar => |bits| {
                        try w.u8v(0);
                        try w.u32v(bits);
                    },
                    .per_column => |h| {
                        try w.u8v(1);
                        const scales = func.scaleList(h);
                        try w.u32v(@intCast(scales.len));
                        for (scales) |s| try w.u32v(s);
                    },
                }
            } else {
                try w.u8v(0);
            }
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
        // Subtraction form: pos <= len is an invariant, so it never underflows,
        // and unlike `pos + n > len` it cannot wrap when `n` is near usize max on
        // a 32-bit target (IronStyle: design for the most constrained target).
        if (n > self.bytes.len - self.pos) return error.MalformedBitcode;
        const v = std.mem.readInt(T, self.bytes[self.pos..][0..n], .little);
        self.pos += n;
        return v;
    }
    fn takeBytes(self: *Reader, n: usize) Error![]const u8 {
        if (n > self.bytes.len - self.pos) return error.MalformedBitcode;
        const s = self.bytes[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
};

/// Look up a type-table entry read from untrusted input. `valid` is the number
/// of entries decoded so far; an index at or beyond it is either out of range or
/// a forward reference into uninitialized memory. Both are malformed input.
fn mapType(type_map: []const Type, valid: usize, idx: u32) Error!Type {
    if (idx >= valid) return error.MalformedBitcode;
    return type_map[idx];
}

/// Convert an untrusted block number to a `Block`, rejecting out-of-range values
/// that would later index the block list out of bounds.
fn checkBlock(bnum: u32, block_count: u32) Error!Block {
    if (bnum >= block_count) return error.MalformedBitcode;
    return @enumFromInt(bnum);
}

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
    // Pass `i` as the valid-entry count so a nested type may only reference a
    // type decoded earlier, never a forward/uninitialized one.
    for (0..type_count) |i| type_map[i] = try readType(&r, &func, type_map, i);

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
            const ty = try mapType(type_map, type_map.len, try r.take(u32));
            try serial.append(allocator, try func.appendBlockParam(block, ty));
        }
        const inst_count = try r.take(u32);
        for (0..inst_count) |_| try readInst(&r, &func, block, type_map, block_count, dummy, &serial, &fixups, allocator);
        try readTerm(&r, &func, block, block_count, dummy, &fixups, allocator);
    }

    // Every fixup slot is a serial number read from input; validate the whole
    // set against the recovered value table before applying, so `apply` can index
    // it without bounds checks and a bad serial is a recoverable fault, not an OOB.
    for (fixups.items) |f| {
        for (f.slots) |slot| {
            if (slot >= serial.items.len) return error.MalformedBitcode;
        }
    }
    for (fixups.items) |f| f.apply(&func, serial.items);

    return func;
}

fn readType(r: *Reader, func: *Function, type_map: []const Type, valid: usize) Error!Type {
    return switch (try r.take(u8)) {
        0 => try func.types.intern(.bool),
        1 => blk: {
            const s: std.builtin.Signedness = if (try r.take(u8) == 0) .signed else .unsigned;
            const bits = try r.take(u16);
            break :blk try func.types.intern(.{ .int = .{ .signedness = s, .bits = bits } });
        },
        2 => blk: {
            const kind: types.FloatKind = switch (try r.take(u8)) {
                0 => .f32,
                1 => .f64,
                2 => .f16,
                else => return error.MalformedBitcode,
            };
            break :blk try func.types.intern(.{ .float = kind });
        },
        3 => try func.types.intern(.ptr),
        4 => blk: {
            const len = try r.take(u32);
            const elem = try mapType(type_map, valid, try r.take(u32));
            break :blk try func.types.intern(.{ .vector = .{ .len = len, .elem = elem } });
        },
        5 => blk: {
            const n = try r.take(u32);
            const fields = try func.allocator.alloc(Type, n);
            defer func.allocator.free(fields);
            for (fields) |*f| f.* = try mapType(type_map, valid, try r.take(u32));
            break :blk try func.types.intern(.{ .@"struct" = fields });
        },
        6 => blk: {
            const len = try r.take(u64);
            const elem = try mapType(type_map, valid, try r.take(u32));
            break :blk try func.types.intern(.{ .array = .{ .len = len, .elem = elem } });
        },
        7 => try func.types.intern(.{ .slice = .{ .elem = try mapType(type_map, valid, try r.take(u32)) } }),
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
                    .prefetch => |*pf| pf.ptr = next(&i, self.slots, serial),
                    .dot => |*d| {
                        d.acc = next(&i, self.slots, serial);
                        d.a = next(&i, self.slots, serial);
                        d.b = next(&i, self.slots, serial);
                    },
                    .matmul => |*mm| {
                        mm.a = next(&i, self.slots, serial);
                        mm.b = next(&i, self.slots, serial);
                        mm.c = next(&i, self.slots, serial);
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

fn readInst(r: *Reader, func: *Function, block: Block, type_map: []const Type, block_count: u32, dummy: Value, serial: *std.ArrayList(Value), fixups: *std.ArrayList(Fixup), allocator: std.mem.Allocator) Error!void {
    const has_result = (try r.take(u8)) != 0;
    const rty: Type = if (has_result) try mapType(type_map, type_map.len, try r.take(u32)) else undefined;
    const tag = try r.take(u8);

    var slots: std.ArrayList(u32) = .empty;
    errdefer slots.deinit(allocator);

    // Build the instruction with placeholder operands, recording the serials.
    const inst: Inst = switch (tag) {
        op_iconst => try appendRes(func, block, serial, rty, .{ .iconst = @bitCast(try r.take(u64)) }),
        op_fconst => try appendRes(func, block, serial, rty, .{ .fconst = @bitCast(try r.take(u64)) }),
        op_arith => blk: {
            const op = std.enums.fromInt(function.BinOp, try r.take(u8)) orelse return error.MalformedBitcode;
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .arith = .{ .op = op, .lhs = dummy, .rhs = dummy } });
        },
        op_arith_imm => blk: {
            const op = std.enums.fromInt(function.BinOp, try r.take(u8)) orelse return error.MalformedBitcode;
            try slots.append(allocator, try r.take(u32));
            const imm: i64 = @bitCast(try r.take(u64));
            break :blk try appendRes(func, block, serial, rty, .{ .arith_imm = .{ .op = op, .lhs = dummy, .imm = imm } });
        },
        op_icmp => blk: {
            const op = std.enums.fromInt(function.CmpOp, try r.take(u8)) orelse return error.MalformedBitcode;
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
            const uop = std.enums.fromInt(function.UnaryOp, try r.take(u8)) orelse return error.MalformedBitcode;
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .unary = .{ .op = uop, .value = dummy } });
        },
        op_alloca => try appendRes(func, block, serial, rty, .{ .alloca = .{ .elem = try mapType(type_map, type_map.len, try r.take(u32)) } }),
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
        op_prefetch => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendStmtOp(func, block, .{ .prefetch = .{ .ptr = dummy } });
        },
        op_dot => blk: {
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .dot = .{ .acc = dummy, .a = dummy, .b = dummy } });
        },
        op_matmul => blk: {
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            const m = try r.take(u16);
            const n = try r.take(u16);
            const k = try r.take(u16);
            const dtype = std.enums.fromInt(function.MatMulType, try r.take(u8)) orelse return error.MalformedBitcode;
            const accumulate = (try r.take(u8)) != 0;
            const embedded = (try r.take(u8)) != 0;
            const has_input_signs = (try r.take(u8)) != 0;
            const input_signs: ?function.InputSigns = if (has_input_signs) blk_signs: {
                const a_unsigned = (try r.take(u8)) != 0;
                const b_unsigned = (try r.take(u8)) != 0;
                break :blk_signs .{ .a_unsigned = a_unsigned, .b_unsigned = b_unsigned };
            } else null;
            const has_quant = (try r.take(u8)) != 0;
            const quant: ?function.MatMulQuant = if (has_quant) blk_quant: {
                const relu = (try r.take(u8)) != 0;
                const out = std.enums.fromInt(function.MatMulQuantOut, try r.take(u8)) orelse return error.MalformedBitcode;
                const zero_point: i32 = @bitCast(try r.take(u32));
                const bias_present = (try r.take(u8)) != 0;
                const bias: ?function.BiasList = if (bias_present) bias: {
                    const count = try r.take(u32);
                    const tmp = try allocator.alloc(i32, count);
                    defer allocator.free(tmp);
                    for (tmp) |*bv| bv.* = @bitCast(try r.take(u32));
                    break :bias try func.internBias(tmp);
                } else null;
                const scale_kind = try r.take(u8);
                const scale: function.MatMulScale = switch (scale_kind) {
                    0 => .{ .scalar = try r.take(u32) },
                    1 => scale: {
                        const count = try r.take(u32);
                        const tmp = try allocator.alloc(u32, count);
                        defer allocator.free(tmp);
                        for (tmp) |*s| s.* = try r.take(u32);
                        break :scale .{ .per_column = try func.internScales(tmp) };
                    },
                    else => return error.MalformedBitcode,
                };
                break :blk_quant .{ .scale = scale, .relu = relu, .out = out, .bias = bias, .zero_point = zero_point };
            } else null;
            break :blk try appendStmtOp(func, block, .{ .matmul = .{ .a = dummy, .b = dummy, .c = dummy, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate, .embedded = embedded, .quant = quant, .input_signs = input_signs } });
        },
        op_if => blk: {
            try slots.append(allocator, try r.take(u32)); // cond
            const then_j = try readJumpDummy(r, func, block_count, &slots, dummy, allocator);
            const else_j = try readJumpDummy(r, func, block_count, &slots, dummy, allocator);
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

fn readJumpDummy(r: *Reader, func: *Function, block_count: u32, slots: *std.ArrayList(u32), dummy: Value, allocator: std.mem.Allocator) Error!function.Jump {
    const target = try checkBlock(try r.take(u32), block_count);
    const n = try r.take(u32);
    for (0..n) |_| try slots.append(allocator, try r.take(u32));
    return .{ .target = target, .args = try internDummies(func, n, dummy) };
}

fn readTerm(r: *Reader, func: *Function, block: Block, block_count: u32, dummy: Value, fixups: *std.ArrayList(Fixup), allocator: std.mem.Allocator) Error!void {
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
            const target = try checkBlock(try r.take(u32), block_count);
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

test "round-trips a prefetch through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    try func.appendPrefetch(entry, p);
    func.setTerminator(entry, .{ .ret = null });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .prefetch);
    try std.testing.expectEqual(p, op.prefetch.ptr);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips a dot through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v16i8 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });
    const v4i32 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const entry = try func.appendBlock();
    const acc = try func.appendBlockParam(entry, v4i32);
    const a_val = try func.appendBlockParam(entry, v16i8);
    const b_val = try func.appendBlockParam(entry, v16i8);
    const result = try func.appendDot(entry, acc, a_val, b_val);
    func.setTerminator(entry, .{ .ret = result });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .dot);
    try std.testing.expectEqual(acc, op.dot.acc);
    try std.testing.expectEqual(a_val, op.dot.a);
    try std.testing.expectEqual(b_val, op.dot.b);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips a matmul through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmul(entry, a_val, b_val, c_val, 8, 12, 4, .uint8, true);
    func.setTerminator(entry, .{ .ret = null });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .matmul);
    try std.testing.expectEqual(a_val, op.matmul.a);
    try std.testing.expectEqual(b_val, op.matmul.b);
    try std.testing.expectEqual(c_val, op.matmul.c);
    try std.testing.expectEqual(@as(u16, 8), op.matmul.m);
    try std.testing.expectEqual(@as(u16, 12), op.matmul.n);
    try std.testing.expectEqual(@as(u16, 4), op.matmul.k);
    try std.testing.expectEqual(function.MatMulType.uint8, op.matmul.dtype);
    try std.testing.expectEqual(true, op.matmul.accumulate);
    try std.testing.expectEqual(@as(?function.MatMulQuant, null), op.matmul.quant);
    try std.testing.expectEqual(@as(?function.InputSigns, null), op.matmul.input_signs);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips a matmul with a mixed-signedness input_signs override through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulSigned(entry, a_val, b_val, c_val, 8, 12, 4, .int8, true, .{ .a_unsigned = true, .b_unsigned = false });
    func.setTerminator(entry, .{ .ret = null });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .matmul);
    try std.testing.expect(op.matmul.input_signs != null);
    try std.testing.expectEqual(true, op.matmul.input_signs.?.a_unsigned);
    try std.testing.expectEqual(false, op.matmul.input_signs.?.b_unsigned);
    try std.testing.expectEqual(@as(?function.MatMulQuant, null), op.matmul.quant);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips a matmul quant epilogue through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuant(entry, a_val, b_val, c_val, 8, 12, 4, .int8, true, .{ .scale = .{ .scalar = 0x3F000000 }, .relu = true });
    func.setTerminator(entry, .{ .ret = null });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .matmul);
    try std.testing.expect(op.matmul.quant != null);
    try std.testing.expect(op.matmul.quant.?.scale == .scalar);
    try std.testing.expectEqual(@as(u32, 0x3F000000), op.matmul.quant.?.scale.scalar);
    try std.testing.expectEqual(true, op.matmul.quant.?.relu);
    // appendMatmulQuant defaults bias/zero_point; the round-trip must preserve those defaults.
    try std.testing.expectEqual(@as(?function.BiasList, null), op.matmul.quant.?.bias);
    try std.testing.expectEqual(@as(i32, 0), op.matmul.quant.?.zero_point);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips a matmul per-column quant epilogue through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    const scales: []const u32 = &.{ 0x3F800000, 0x3F000000, 0x3E800000, 0x40000000 };
    // Exercises .u8 here (the scalar round-trip test above already covers the default .i8), so
    // the new `out` byte's encode/decode order is proven for both enum values across the suite.
    try func.appendMatmulQuantPerColumn(entry, a_val, b_val, c_val, 8, 4, 4, .int8, true, true, .u8, scales);
    func.setTerminator(entry, .{ .ret = null });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .matmul);
    try std.testing.expect(op.matmul.quant != null);
    try std.testing.expect(op.matmul.quant.?.scale == .per_column);
    try std.testing.expectEqual(function.MatMulQuantOut.u8, op.matmul.quant.?.out);
    try std.testing.expectEqualSlices(u32, scales, decoded.scaleList(op.matmul.quant.?.scale.per_column));
    try std.testing.expectEqual(true, op.matmul.quant.?.relu);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips an asymmetric-uint8 matmul quant epilogue (bias + zero_point) through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    const scales: []const u32 = &.{ 0x3F800000, 0x3F000000, 0x3E800000, 0x40000000 };
    const bias: []const i32 = &.{ 5, -7, 0, 128 }; // mix of positive, negative, and zero
    try func.appendMatmulQuantSpec(entry, a_val, b_val, c_val, 8, 4, 4, .int8, true, .{
        .scale_per_column = scales,
        .bias = bias,
        .zero_point = -12,
        .relu = false,
        .out = .u8,
    });
    func.setTerminator(entry, .{ .ret = null });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const op = decoded.opcode(insts[insts.len - 1]);
    try std.testing.expect(op == .matmul);
    try std.testing.expect(op.matmul.quant != null);
    try std.testing.expect(op.matmul.quant.?.scale == .per_column);
    try std.testing.expectEqualSlices(u32, scales, decoded.scaleList(op.matmul.quant.?.scale.per_column));
    try std.testing.expectEqual(function.MatMulQuantOut.u8, op.matmul.quant.?.out);
    try std.testing.expectEqual(false, op.matmul.quant.?.relu);
    try std.testing.expect(op.matmul.quant.?.bias != null);
    try std.testing.expectEqualSlices(i32, bias, decoded.biasList(op.matmul.quant.?.bias.?));
    try std.testing.expectEqual(@as(i32, -12), op.matmul.quant.?.zero_point);

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

test "regression: rejects an out-of-range type index instead of OOB reading the type table" {
    // A malformed module: 1 type, a vector whose element index points at type 0
    // (itself). That is a forward/self reference into a table entry not yet decoded.
    // The pre-fix code indexed `type_map[0]` (uninitialized memory) and interned a
    // garbage handle. Now it is a recoverable fault.
    const allocator = std.testing.allocator;
    const self_ref = "VBC1" ++ // magic
        "\x01\x00\x00\x00" ++ // type_count = 1
        "\x04" ++ // type 0: vector
        "\x01\x00\x00\x00" ++ // vector len = 1
        "\x00\x00\x00\x00"; // elem type index = 0 (not yet decoded)
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, self_ref));

    // Same, but an index past the whole table (5 >= 1).
    const past_end = "VBC1" ++
        "\x01\x00\x00\x00" ++
        "\x04" ++
        "\x01\x00\x00\x00" ++
        "\x05\x00\x00\x00"; // elem type index = 5, out of range
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, past_end));
}

test "regression: rejects an unknown arith operator byte instead of @enumFromInt UB" {
    // A hand-built module that reaches an `arith` instruction carrying operator
    // byte 0xFF. BinOp has 10 variants, so the pre-fix `@enumFromInt(0xFF)` was
    // undefined behavior; std.enums.fromInt must reject it as malformed.
    const allocator = std.testing.allocator;
    const bad_arith = "VBC1" ++ // magic
        "\x01\x00\x00\x00" ++ // type_count = 1
        "\x01\x00\x20\x00" ++ // type 0: int, signed, 32 bits
        "\x00\x00\x00\x00" ++ // sym_count = 0
        "\x01\x00\x00\x00" ++ // block_count = 1
        "\x01\x00\x00\x00" ++ // block 0 param_count = 1
        "\x00\x00\x00\x00" ++ // param 0 type index = 0
        "\x01\x00\x00\x00" ++ // inst_count = 1
        "\x01" ++ // has_result = 1
        "\x00\x00\x00\x00" ++ // result type index = 0
        "\x02" ++ // tag = op_arith
        "\xFF"; // operator byte = 0xFF (no such BinOp)
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad_arith));
}

test "round-trips f16 alongside f32 and f64 through bitcode" {
    // FloatKind grew from 2 to 3 members, so its wire encoding widened from a
    // single 0/1 bit to a full byte (see writeType/readType). This function
    // exercises all three float widths plus width-changing converts, so both
    // the new f16 case and the still-existing f32/f64 cases are proven not to
    // have regressed by the wire change.
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const f64_t = try func.types.intern(.{ .float = .f64 });

    const entry = try func.appendBlock();
    const p16 = try func.appendBlockParam(entry, f16_t);
    const p32 = try func.appendBlockParam(entry, f32_t);
    const p64 = try func.appendBlockParam(entry, f64_t);
    const widened = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = p16 } }); // f16 -> f32
    const narrowed = try func.appendInst(entry, f16_t, .{ .convert = .{ .value = p32 } }); // f32 -> f16
    const doubled = try func.appendInst(entry, f64_t, .{ .convert = .{ .value = p64 } }); // f64 -> f64 (identity, keeps p64 live)
    _ = narrowed;
    _ = doubled;
    func.setTerminator(entry, .{ .ret = widened });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);

    // The block param types decoded back to the exact same float widths, not
    // some other kind entirely (a wire-format mixup would likely land on a
    // structurally different type and fail this rather than print wrong text).
    try std.testing.expectEqual(types.TypeKind{ .float = .f16 }, decoded.types.type_kind(decoded.valueType(decoded.blockParams(entry)[0])));
    try std.testing.expectEqual(types.TypeKind{ .float = .f32 }, decoded.types.type_kind(decoded.valueType(decoded.blockParams(entry)[1])));
    try std.testing.expectEqual(types.TypeKind{ .float = .f64 }, decoded.types.type_kind(decoded.valueType(decoded.blockParams(entry)[2])));
}

test "regression: rejects an unknown float-kind byte instead of silently aliasing to f32/f64" {
    // A hand-built module with a `float` type carrying kind byte 3, which does
    // not exist (0=f32, 1=f64, 2=f16). Before f16 existed, this byte was a
    // simple `if (byte == 0) .f32 else .f64`, so any nonzero byte silently
    // meant f64; now it must be a recoverable fault instead of misreading the
    // type.
    const allocator = std.testing.allocator;
    const bad_float = "VBC1" ++ // magic
        "\x01\x00\x00\x00" ++ // type_count = 1
        "\x02" ++ // type 0: float
        "\x03"; // float-kind byte = 3 (no such FloatKind)
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad_float));
}
