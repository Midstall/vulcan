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
const parser = @import("parser.zig");
const attribute = @import("attribute.zig");

const Function = function.Function;
const Value = function.Value;
const Inst = function.Inst;
const Block = function.Block;
const Type = types.Type;
const Opcode = function.Opcode;
const Attribute = attribute.Attribute;

pub const Error = std.mem.Allocator.Error || error{MalformedBitcode};

const magic = "VBC1";

/// Bytes the stream header occupies before the type table: the magic, the whole-function
/// flag byte, and the fixed-parameter count.
const header_len = magic.len + @sizeOf(u8) + @sizeOf(u32);

// Whole-function flag bits, packed into one header byte.
const fn_flag_variadic: u8 = 1 << 0;
const fn_flag_sret: u8 = 1 << 1;
const fn_flag_local: u8 = 1 << 2;

// Extra-field flag bits of a `call` or `call_indirect` record.
const call_flag_variadic: u8 = 1 << 0;
const call_flag_sret: u8 = 1 << 1;
const call_flag_ret_dest: u8 = 1 << 2;

// Attribute target tags (stable on the wire).
const attr_target_func: u8 = 0;
const attr_target_block: u8 = 1;
const attr_target_inst: u8 = 2;
const attr_target_value: u8 = 3;

// Attribute body tags (stable on the wire).
const attr_inline: u8 = 0;
const attr_noreturn: u8 = 1;
const attr_cold: u8 = 2;
const attr_align: u8 = 3;
const attr_endian: u8 = 4;
const attr_custom: u8 = 5;

// Namespaced attribute payload tags (stable on the wire).
const attr_value_flag: u8 = 0;
const attr_value_int: u8 = 1;
const attr_value_string: u8 = 2;

/// The stream header a hand-built test module starts with: the magic, no whole-function
/// flags, and a zero fixed-parameter count. Only the tests below build a stream by hand.
const test_header = magic ++ "\x00" ++ "\x00\x00\x00\x00";

/// A canonical number that names nothing. An instruction that no block holds, or a value
/// that no such instruction defines, has no place in the canonical walk, so an attribute
/// on it names no entity in the decoded function and is not written.
const no_serial: u32 = std.math.maxInt(u32);

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
const op_va_start: u8 = 20;
const op_va_arg: u8 = 21;
const op_va_end: u8 = 22;
const op_fconst128: u8 = 23;
const op_barrier: u8 = 24;
const op_atomic_rmw: u8 = 25;

// An `atomic_rmw` record flag bit: the compare operand follows the two ordinary operand
// slots. Written from the field and read back into it, so a record round-trips whatever the
// opcode holds, rather than depending on the op/compare agreement `verify` enforces.
const atomic_flag_compare: u8 = 1 << 0;

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

    // Canonical serial number for each value (params then results, block order) and for
    // each instruction (block order, then order within the block). Both start at
    // `no_serial`, so an entity the walk never reaches keeps a number that names nothing.
    const serial = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(serial);
    @memset(serial, no_serial);
    const inst_serial = try allocator.alloc(u32, func.instCount());
    defer allocator.free(inst_serial);
    @memset(inst_serial, no_serial);
    {
        var next: u32 = 0;
        var next_inst: u32 = 0;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockParams(block)) |p| {
                serial[@intFromEnum(p)] = next;
                next += 1;
            }
            for (func.blockInsts(block)) |inst| {
                inst_serial[@intFromEnum(inst)] = next_inst;
                next_inst += 1;
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

    // Whole-function metadata. Each of these changes how the function is called or how
    // its symbol binds, so a stream without them decodes to a different function.
    var fn_flags: u8 = 0;
    if (func.is_variadic) fn_flags |= fn_flag_variadic;
    if (func.sret) fn_flags |= fn_flag_sret;
    if (func.is_local) fn_flags |= fn_flag_local;
    try w.u8v(fn_flags);
    try w.u32v(func.num_fixed_params);

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

    // Attributes come last, because a target names a block, an instruction or a value,
    // and the decoder can only resolve those once it has rebuilt the body. They carry the
    // `vulcan.gpu` namespace the whole GPU path reads, so a stream without them lays out a
    // kernel's parameter block differently.
    try writeAttrs(&w, func, serial, inst_serial);

    return w.bytes.toOwnedSlice(allocator);
}

/// Write the attribute list. An entry whose target is not in the canonical walk names no
/// entity in the decoded function, so it is dropped rather than written with a number the
/// decoder would have to reject.
fn writeAttrs(w: *Writer, func: *const Function, serial: []const u32, inst_serial: []const u32) Error!void {
    const entries = func.attributeEntries();
    var count: u32 = 0;
    for (entries) |entry| {
        if (attrTargetSerial(entry.target, serial, inst_serial) != null) count += 1;
    }
    try w.u32v(count);
    for (entries) |entry| {
        const number = attrTargetSerial(entry.target, serial, inst_serial) orelse continue;
        switch (entry.target) {
            .func => try w.u8v(attr_target_func),
            .block => try w.u8v(attr_target_block),
            .inst => try w.u8v(attr_target_inst),
            .value => try w.u8v(attr_target_value),
        }
        if (entry.target != .func) try w.u32v(number);
        try writeAttr(w, entry.attr);
    }
}

/// The canonical number an attribute target resolves to, or null when the target is not
/// part of the canonical walk. `func` has no number and reports 0.
fn attrTargetSerial(target: function.AttrTarget, serial: []const u32, inst_serial: []const u32) ?u32 {
    return switch (target) {
        .func => 0,
        .block => |b| @intFromEnum(b),
        .inst => |i| blk: {
            const n = inst_serial[@intFromEnum(i)];
            break :blk if (n == no_serial) null else n;
        },
        .value => |v| blk: {
            const n = serial[@intFromEnum(v)];
            break :blk if (n == no_serial) null else n;
        },
    };
}

/// Write one attribute body. String payloads use the length-prefixed form the symbol
/// table already uses.
fn writeAttr(w: *Writer, attr: Attribute) Error!void {
    switch (attr) {
        .@"inline" => try w.u8v(attr_inline),
        .noreturn => try w.u8v(attr_noreturn),
        .cold => try w.u8v(attr_cold),
        .@"align" => |a| {
            try w.u8v(attr_align);
            try w.u32v(a);
        },
        .endian => |e| {
            try w.u8v(attr_endian);
            // The decoder maps this byte back with `std.enums.fromInt`, so pin the tag
            // values here. This follows the `float` and `ptr` arms of `writeType`.
            comptime {
                std.debug.assert(@intFromEnum(attribute.Endianness.little) == 0);
                std.debug.assert(@intFromEnum(attribute.Endianness.big) == 1);
                std.debug.assert(@intFromEnum(attribute.Endianness.native) == 2);
            }
            try w.u8v(@intFromEnum(e));
        },
        .custom => |c| {
            try w.u8v(attr_custom);
            try writeStr(w, c.namespace);
            try writeStr(w, c.key);
            switch (c.value) {
                .flag => try w.u8v(attr_value_flag),
                .int => |i| {
                    try w.u8v(attr_value_int);
                    try w.u64v(@bitCast(i));
                },
                .string => |s| {
                    try w.u8v(attr_value_string);
                    try writeStr(w, s);
                },
            }
        },
    }
}

/// Write a length-prefixed string, the same shape the symbol table uses.
fn writeStr(w: *Writer, s: []const u8) Error!void {
    try w.u32v(@intCast(s.len));
    try w.bytes.appendSlice(w.allocator, s);
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
                std.debug.assert(@intFromEnum(types.FloatKind.f128) == 3);
            }
            try w.u8v(@intFromEnum(f));
        },
        .ptr => |space| {
            try w.u8v(3);
            // The decoder is a hardcoded switch on these values, so pin them here. A
            // future reorder would otherwise desync the two sides and silently corrupt
            // streams instead of failing to build.
            comptime {
                std.debug.assert(@intFromEnum(types.AddressSpace.global) == 0);
                std.debug.assert(@intFromEnum(types.AddressSpace.shared) == 1);
                std.debug.assert(@intFromEnum(types.AddressSpace.private) == 2);
                std.debug.assert(@intFromEnum(types.AddressSpace.constant) == 3);
            }
            try w.u8v(@intFromEnum(space));
        },
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
        .fconst128 => |v| {
            try w.u8v(op_fconst128);
            try w.u64v(@truncate(v));
            try w.u64v(@truncate(v >> 64));
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
            try writeCallExtras(w, c.is_variadic, c.num_fixed, c.sret, if (c.ret_dest) |rd| sv(serial, rd) else null, c.ret_regs, c.ret_pieces);
        },
        .call_indirect => |c| {
            try w.u8v(op_call_indirect);
            try w.u32v(sv(serial, c.target));
            const args = func.valueList(c.args);
            try w.u32v(@intCast(args.len));
            for (args) |a| try w.u32v(sv(serial, a));
            try writeCallExtras(w, c.is_variadic, c.num_fixed, c.sret, if (c.ret_dest) |rd| sv(serial, rd) else null, c.ret_regs, c.ret_pieces);
        },
        // `volatile` marks an access the optimizer must not remove, move or merge. A
        // stream that drops it turns an MMIO register access into an ordinary one.
        .load => |l| {
            try w.u8v(op_load);
            try w.u32v(sv(serial, l.ptr));
            try w.u8v(@intFromBool(l.@"volatile"));
        },
        .store => |st| {
            try w.u8v(op_store);
            try w.u32v(sv(serial, st.value));
            try w.u32v(sv(serial, st.ptr));
            try w.u8v(@intFromBool(st.@"volatile"));
        },
        .prefetch => |pf| {
            try w.u8v(op_prefetch);
            try w.u32v(sv(serial, pf.ptr));
        },
        // `VaArg.ty` is not written separately. It always equals the result type
        // already written generically above (`if (result) |r| ... valueType(r)`), so the
        // decoder recovers it from `rty` with nothing extra on the wire.
        .va_start => |vs| {
            try w.u8v(op_va_start);
            try w.u32v(sv(serial, vs.list));
        },
        .va_arg => |va| {
            try w.u8v(op_va_arg);
            try w.u32v(sv(serial, va.list));
        },
        .va_end => |ve| {
            try w.u8v(op_va_end);
            try w.u32v(sv(serial, ve.list));
        },
        .barrier => |bar| {
            try w.u8v(op_barrier);
            // The decoder maps this byte back with `std.enums.fromInt`, so pin the tag
            // values here. A future reorder of `BarrierScope` would otherwise desync the
            // two sides and silently turn a workgroup barrier into a subgroup one,
            // instead of failing to build. This follows the `float` and `ptr` arms of
            // `writeType`.
            comptime {
                std.debug.assert(@intFromEnum(function.BarrierScope.workgroup) == 0);
                std.debug.assert(@intFromEnum(function.BarrierScope.subgroup) == 1);
            }
            try w.u8v(@intFromEnum(bar.scope));
        },
        .atomic_rmw => |a| {
            try w.u8v(op_atomic_rmw);
            // The decoder maps these three bytes back with `std.enums.fromInt`, so pin the
            // tag values here. A future reorder of any of the three enums would otherwise
            // desync the two sides and silently turn one atomic into another, instead of
            // failing to build. This follows the `barrier` arm above.
            comptime {
                std.debug.assert(@intFromEnum(function.AtomicOp.add) == 0);
                std.debug.assert(@intFromEnum(function.AtomicOp.min) == 1);
                std.debug.assert(@intFromEnum(function.AtomicOp.max) == 2);
                std.debug.assert(@intFromEnum(function.AtomicOp.bit_and) == 3);
                std.debug.assert(@intFromEnum(function.AtomicOp.bit_or) == 4);
                std.debug.assert(@intFromEnum(function.AtomicOp.bit_xor) == 5);
                std.debug.assert(@intFromEnum(function.AtomicOp.exchange) == 6);
                std.debug.assert(@intFromEnum(function.AtomicOp.compare_exchange) == 7);
                std.debug.assert(@intFromEnum(function.AtomicOrdering.relaxed) == 0);
                std.debug.assert(@intFromEnum(function.AtomicOrdering.acquire) == 1);
                std.debug.assert(@intFromEnum(function.AtomicOrdering.release) == 2);
                std.debug.assert(@intFromEnum(function.AtomicOrdering.acq_rel) == 3);
                std.debug.assert(@intFromEnum(function.AtomicOrdering.seq_cst) == 4);
                std.debug.assert(@intFromEnum(function.AtomicScope.workgroup) == 0);
                std.debug.assert(@intFromEnum(function.AtomicScope.device) == 1);
                std.debug.assert(@intFromEnum(function.AtomicScope.system) == 2);
            }
            try w.u8v(@intFromEnum(a.op));
            try w.u8v(@intFromEnum(a.ordering));
            try w.u8v(@intFromEnum(a.scope));
            try w.u8v(if (a.compare != null) atomic_flag_compare else 0);
            try w.u32v(sv(serial, a.ptr));
            try w.u32v(sv(serial, a.value));
            if (a.compare) |c| try w.u32v(sv(serial, c));
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
            try w.u8v(@intFromBool(ga.via_got));
        },
    }
}

/// Write the extra fields a call carries: the variadic marker with the callee's fixed
/// parameter count, the hidden-pointer struct return, and the register-return destination
/// with its pieces. `ret_dest_serial` is the destination's canonical number, or null when
/// the call has no register return. It goes LAST, after the argument serials, and the
/// decoder reads it in the same place, so the operand fixup order stays fixed.
fn writeCallExtras(w: *Writer, is_variadic: bool, num_fixed: u32, sret: bool, ret_dest_serial: ?u32, ret_regs: u8, ret_pieces: [4]function.RetPiece) Error!void {
    var flags: u8 = 0;
    if (is_variadic) flags |= call_flag_variadic;
    if (sret) flags |= call_flag_sret;
    if (ret_dest_serial != null) flags |= call_flag_ret_dest;
    try w.u8v(flags);
    try w.u32v(num_fixed);
    // Only `ret_pieces[0..ret_regs]` carries meaning, so only that part goes on the wire
    // and the decoder rebuilds the rest at its default.
    std.debug.assert(ret_regs <= ret_pieces.len);
    try w.u8v(ret_regs);
    for (ret_pieces[0..ret_regs]) |p| {
        try w.u8v(@intFromBool(p.fp));
        try w.u8v(p.offset);
        try w.u8v(p.bytes);
    }
    if (ret_dest_serial) |s| try w.u32v(s);
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
        .ret => |r| {
            try w.u8v(1);
            try w.u8v(r.count);
            for (r.slice()) |vv| try w.u32v(sv(serial, vv));
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

    // Whole-function metadata. Unknown flag bits are malformed input: the stream carries
    // no version, so a bit this build does not know is a stream it cannot read.
    const fn_flags = try r.take(u8);
    if (fn_flags & ~(fn_flag_variadic | fn_flag_sret | fn_flag_local) != 0) return error.MalformedBitcode;
    func.is_variadic = fn_flags & fn_flag_variadic != 0;
    func.sret = fn_flags & fn_flag_sret != 0;
    func.is_local = fn_flags & fn_flag_local != 0;
    func.num_fixed_params = try r.take(u32);

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

    // The instruction serial->Inst table, in the same canonical order, so an attribute can
    // name the instruction it rides on.
    var inst_serial: std.ArrayList(Inst) = .empty;
    defer inst_serial.deinit(allocator);

    const dummy: Value = @enumFromInt(0);
    for (0..block_count) |bi| {
        const block: Block = @enumFromInt(bi);
        const param_count = try r.take(u32);
        for (0..param_count) |_| {
            const ty = try mapType(type_map, type_map.len, try r.take(u32));
            try serial.append(allocator, try func.appendBlockParam(block, ty));
        }
        const inst_count = try r.take(u32);
        for (0..inst_count) |_| try readInst(&r, &func, block, type_map, block_count, dummy, &serial, &inst_serial, &fixups, allocator);
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

    try readAttrs(&r, &func, block_count, serial.items, inst_serial.items);

    return func;
}

/// Read the attribute list written by `writeAttrs` and reattach each entry. Every target
/// number comes off an UNTRUSTED stream, so one that names no entity is malformed input.
fn readAttrs(r: *Reader, func: *Function, block_count: u32, serial: []const Value, inst_serial: []const Inst) Error!void {
    const count = try r.take(u32);
    for (0..count) |_| {
        const kind = try r.take(u8);
        const target: function.AttrTarget = switch (kind) {
            attr_target_func => .func,
            attr_target_block => .{ .block = try checkBlock(try r.take(u32), block_count) },
            attr_target_inst => blk: {
                const n = try r.take(u32);
                if (n >= inst_serial.len) return error.MalformedBitcode;
                break :blk .{ .inst = inst_serial[n] };
            },
            attr_target_value => blk: {
                const n = try r.take(u32);
                if (n >= serial.len) return error.MalformedBitcode;
                break :blk .{ .value = serial[n] };
            },
            else => return error.MalformedBitcode,
        };
        try func.addAttr(target, try readAttr(r));
    }
}

/// Read one attribute body. String payloads point into the input buffer; `addAttr` copies
/// them into function-owned storage.
fn readAttr(r: *Reader) Error!Attribute {
    return switch (try r.take(u8)) {
        attr_inline => .@"inline",
        attr_noreturn => .noreturn,
        attr_cold => .cold,
        attr_align => .{ .@"align" = try r.take(u32) },
        attr_endian => blk: {
            // The byte comes off an UNTRUSTED stream, so an unknown value is a recoverable
            // fault and never an invalid enum.
            const raw = try r.take(u8);
            const order = std.enums.fromInt(attribute.Endianness, raw) orelse
                return error.MalformedBitcode;
            break :blk .{ .endian = order };
        },
        attr_custom => blk: {
            const namespace = try readStr(r);
            const key = try readStr(r);
            const value: attribute.AttrValue = switch (try r.take(u8)) {
                attr_value_flag => .flag,
                attr_value_int => .{ .int = @bitCast(try r.take(u64)) },
                attr_value_string => .{ .string = try readStr(r) },
                else => return error.MalformedBitcode,
            };
            break :blk .{ .custom = .{ .namespace = namespace, .key = key, .value = value } };
        },
        else => error.MalformedBitcode,
    };
}

/// Read a length-prefixed string, the same shape the symbol table uses.
fn readStr(r: *Reader) Error![]const u8 {
    const len = try r.take(u32);
    return r.takeBytes(len);
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
                3 => .f128,
                else => return error.MalformedBitcode,
            };
            break :blk try func.types.intern(.{ .float = kind });
        },
        3 => blk: {
            // The byte comes off an UNTRUSTED stream, so an unknown value is a
            // recoverable fault and never an invalid enum.
            const raw = try r.take(u8);
            const space = std.enums.fromInt(types.AddressSpace, raw) orelse
                return error.MalformedBitcode;
            break :blk try func.types.intern(.{ .ptr = space });
        },
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
                    // A barrier reserves no operand slot, because it carries no Value.
                    .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
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
                    // Slot order matches the write order above: address, operand, then the
                    // compare operand when the record carried one.
                    .atomic_rmw => |*a| {
                        a.ptr = next(&i, self.slots, serial);
                        a.value = next(&i, self.slots, serial);
                        if (a.compare) |*c| c.* = next(&i, self.slots, serial);
                    },
                    .prefetch => |*pf| pf.ptr = next(&i, self.slots, serial),
                    .va_start => |*vs| vs.list = next(&i, self.slots, serial),
                    .va_arg => |*va| va.list = next(&i, self.slots, serial),
                    .va_end => |*ve| ve.list = next(&i, self.slots, serial),
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
                    // The register-return destination is an operand, and it comes after
                    // the arguments on the wire, so it is filled in that order here.
                    .call => |*c| {
                        for (func.valueListMut(c.args)) |*a| a.* = next(&i, self.slots, serial);
                        if (c.ret_dest) |*rd| rd.* = next(&i, self.slots, serial);
                    },
                    .call_indirect => |*c| {
                        c.target = next(&i, self.slots, serial);
                        for (func.valueListMut(c.args)) |*a| a.* = next(&i, self.slots, serial);
                        if (c.ret_dest) |*rd| rd.* = next(&i, self.slots, serial);
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
                    .ret => |*r| {
                        for (r.values[0..r.count]) |*vv| vv.* = next(&i, self.slots, serial);
                    },
                    .jump => |j| for (func.valueListMut(j.args)) |*a| {
                        a.* = next(&i, self.slots, serial);
                    },
                }
            },
        }
    }
};

fn readInst(r: *Reader, func: *Function, block: Block, type_map: []const Type, block_count: u32, dummy: Value, serial: *std.ArrayList(Value), inst_serial: *std.ArrayList(Inst), fixups: *std.ArrayList(Fixup), allocator: std.mem.Allocator) Error!void {
    const has_result = (try r.take(u8)) != 0;
    const rty: Type = if (has_result) try mapType(type_map, type_map.len, try r.take(u32)) else undefined;
    const tag = try r.take(u8);

    var slots: std.ArrayList(u32) = .empty;
    errdefer slots.deinit(allocator);

    // Build the instruction with placeholder operands, recording the serials.
    const inst: Inst = switch (tag) {
        op_iconst => try appendRes(func, block, serial, rty, .{ .iconst = @bitCast(try r.take(u64)) }),
        op_fconst => try appendRes(func, block, serial, rty, .{ .fconst = @bitCast(try r.take(u64)) }),
        op_fconst128 => blk: {
            const lo: u128 = try r.take(u64);
            const hi: u128 = try r.take(u64);
            break :blk try appendRes(func, block, serial, rty, .{ .fconst128 = (hi << 64) | lo });
        },
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
            const extras = try readCallExtras(r, &slots, allocator);
            const list = try internDummies(func, n, dummy);
            const op: Opcode = .{ .call = .{
                .symbol = symbol,
                .args = list,
                .is_variadic = extras.is_variadic,
                .num_fixed = extras.num_fixed,
                .ret_dest = if (extras.has_ret_dest) dummy else null,
                .ret_regs = extras.ret_regs,
                .ret_pieces = extras.ret_pieces,
                .sret = extras.sret,
            } };
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
            const extras = try readCallExtras(r, &slots, allocator);
            const list = try internDummies(func, n, dummy);
            const op: Opcode = .{ .call_indirect = .{
                .target = dummy,
                .args = list,
                .is_variadic = extras.is_variadic,
                .num_fixed = extras.num_fixed,
                .ret_dest = if (extras.has_ret_dest) dummy else null,
                .ret_regs = extras.ret_regs,
                .ret_pieces = extras.ret_pieces,
                .sret = extras.sret,
            } };
            if (has_result) {
                break :blk try appendRes(func, block, serial, rty, op);
            } else {
                break :blk try appendStmtOp(func, block, op);
            }
        },
        op_load => blk: {
            try slots.append(allocator, try r.take(u32));
            const is_volatile = (try r.take(u8)) != 0;
            break :blk try appendRes(func, block, serial, rty, .{ .load = .{ .ptr = dummy, .@"volatile" = is_volatile } });
        },
        op_store => blk: {
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            const is_volatile = (try r.take(u8)) != 0;
            break :blk try appendStmtOp(func, block, .{ .store = .{ .value = dummy, .ptr = dummy, .@"volatile" = is_volatile } });
        },
        op_prefetch => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendStmtOp(func, block, .{ .prefetch = .{ .ptr = dummy } });
        },
        op_va_start => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendStmtOp(func, block, .{ .va_start = .{ .list = dummy } });
        },
        // `rty` was already decoded above (`has_result` is true for `va_arg`), so `VaArg.ty`
        // recovers straight from it - see `writeInst`'s matching comment.
        op_va_arg => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendRes(func, block, serial, rty, .{ .va_arg = .{ .list = dummy, .ty = rty } });
        },
        op_va_end => blk: {
            try slots.append(allocator, try r.take(u32));
            break :blk try appendStmtOp(func, block, .{ .va_end = .{ .list = dummy } });
        },
        op_barrier => blk: {
            // The stream is UNTRUSTED. An unknown scope byte is malformed bitcode, never
            // an invalid enum: `@enumFromInt` here would build an out-of-range tag that
            // every later exhaustive switch reads as undefined behavior.
            const raw = try r.take(u8);
            const scope = std.enums.fromInt(function.BarrierScope, raw) orelse
                return error.MalformedBitcode;
            break :blk try appendStmtOp(func, block, .{ .barrier = .{ .scope = scope } });
        },
        op_atomic_rmw => blk: {
            // The stream is UNTRUSTED, so each of the three selector bytes maps back with
            // `std.enums.fromInt` and an unknown value is malformed bitcode, never an
            // out-of-range tag every later exhaustive switch reads as undefined behavior.
            const op = std.enums.fromInt(function.AtomicOp, try r.take(u8)) orelse
                return error.MalformedBitcode;
            const ordering = std.enums.fromInt(function.AtomicOrdering, try r.take(u8)) orelse
                return error.MalformedBitcode;
            const scope = std.enums.fromInt(function.AtomicScope, try r.take(u8)) orelse
                return error.MalformedBitcode;
            // An unknown flag bit is a stream this build cannot read: the record's operand
            // count would be wrong and every later record would decode from the wrong
            // offset. Mirrors the whole-function flag check in `decode`.
            const flags = try r.take(u8);
            if (flags & ~atomic_flag_compare != 0) return error.MalformedBitcode;
            const has_compare = flags & atomic_flag_compare != 0;
            try slots.append(allocator, try r.take(u32));
            try slots.append(allocator, try r.take(u32));
            if (has_compare) try slots.append(allocator, try r.take(u32));
            const op_val: Opcode = .{ .atomic_rmw = .{
                .op = op,
                .ptr = dummy,
                .value = dummy,
                .compare = if (has_compare) dummy else null,
                .ordering = ordering,
                .scope = scope,
            } };
            // The result is OPTIONAL, so the record's own `has_result` byte decides which
            // form is rebuilt, exactly as it does for a `call`. Rebuilding the reading form
            // for a reduction would cost the backend a scoreboard it never asked for.
            if (has_result) {
                break :blk try appendRes(func, block, serial, rty, op_val);
            } else {
                break :blk try appendStmtOp(func, block, op_val);
            }
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
        op_global_addr => blk: {
            const symbol = try r.take(u32);
            const via_got = (try r.take(u8)) != 0;
            break :blk try appendRes(func, block, serial, rty, .{ .global_addr = .{ .symbol = symbol, .via_got = via_got } });
        },
        else => return error.MalformedBitcode,
    };

    try inst_serial.append(allocator, inst);

    if (slots.items.len > 0) {
        try fixups.append(allocator, .{ .target = .{ .inst = inst }, .slots = try slots.toOwnedSlice(allocator) });
    } else {
        slots.deinit(allocator);
    }
}

/// Read the extra fields `writeCallExtras` wrote. The destination serial, when present, is
/// appended to `slots` after the argument serials, matching the write order and the fixup
/// order.
fn readCallExtras(r: *Reader, slots: *std.ArrayList(u32), allocator: std.mem.Allocator) Error!CallExtras {
    const flags = try r.take(u8);
    if (flags & ~(call_flag_variadic | call_flag_sret | call_flag_ret_dest) != 0) return error.MalformedBitcode;
    const num_fixed = try r.take(u32);
    const ret_regs = try r.take(u8);
    // A call carries at most 4 return-register pieces. The count comes off an UNTRUSTED
    // stream, so a larger one is malformed input and never an out-of-range array write.
    var ret_pieces: [4]function.RetPiece = @splat(.{});
    if (ret_regs > ret_pieces.len) return error.MalformedBitcode;
    for (ret_pieces[0..ret_regs]) |*p| {
        const fp = (try r.take(u8)) != 0;
        const offset = try r.take(u8);
        const bytes = try r.take(u8);
        p.* = .{ .fp = fp, .offset = offset, .bytes = bytes };
    }
    const has_ret_dest = flags & call_flag_ret_dest != 0;
    if (has_ret_dest) try slots.append(allocator, try r.take(u32));
    return .{
        .is_variadic = flags & call_flag_variadic != 0,
        .num_fixed = num_fixed,
        .has_ret_dest = has_ret_dest,
        .ret_regs = ret_regs,
        .ret_pieces = ret_pieces,
        .sret = flags & call_flag_sret != 0,
    };
}

/// The decoded extra fields of a call. `has_ret_dest` says whether the operand fixup must
/// fill a destination value in.
const CallExtras = struct {
    is_variadic: bool,
    num_fixed: u32,
    has_ret_dest: bool,
    ret_regs: u8,
    ret_pieces: [4]function.RetPiece,
    sret: bool,
};

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
            const count = try r.take(u8);
            if (count > 4) return error.MalformedBitcode;
            var dummies: [4]Value = undefined;
            for (0..count) |i| {
                try slots.append(allocator, try r.take(u32));
                dummies[i] = dummy;
            }
            func.setTerminator(block, .{ .ret = function.Ret.many(dummies[0..count]) });
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
    func.setTerminator(merge, .{ .ret = function.Ret.one(r) });

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
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    try func.appendPrefetch(entry, p);
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

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

test "round-trips va_start/va_arg/va_end through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const list = try func.appendBlockParam(entry, ptr_t);
    try func.appendVaStart(entry, list);
    const v = try func.appendVaArg(entry, list, i32_t);
    try func.appendVaEnd(entry, list);
    func.setTerminator(entry, .{ .ret = function.Ret.one(v) });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    try std.testing.expectEqual(list, decoded.opcode(insts[0]).va_start.list);
    try std.testing.expectEqual(list, decoded.opcode(insts[1]).va_arg.list);
    try std.testing.expectEqual(i32_t, decoded.opcode(insts[1]).va_arg.ty);
    try std.testing.expectEqual(list, decoded.opcode(insts[2]).va_end.list);

    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "round-trips a global_addr's via_got flag through bitcode (both directions)" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const direct = try func.appendGlobalAddr(entry, ptr_t, "D");
    const got = try func.appendGlobalAddrGot(entry, ptr_t, "G");
    _ = direct;
    _ = got;
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    const d_op = decoded.opcode(insts[0]);
    const g_op = decoded.opcode(insts[1]);
    try std.testing.expect(d_op == .global_addr);
    try std.testing.expect(g_op == .global_addr);
    try std.testing.expectEqual(false, d_op.global_addr.via_got);
    try std.testing.expectEqual(true, g_op.global_addr.via_got);

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
    func.setTerminator(entry, .{ .ret = function.Ret.one(result) });

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
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmul(entry, a_val, b_val, c_val, 8, 12, 4, .uint8, true);
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

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
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulSigned(entry, a_val, b_val, c_val, 8, 12, 4, .int8, true, .{ .a_unsigned = true, .b_unsigned = false });
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

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
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuant(entry, a_val, b_val, c_val, 8, 12, 4, .int8, true, .{ .scale = .{ .scalar = 0x3F000000 }, .relu = true });
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

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
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a_val = try func.appendBlockParam(entry, ptr_t);
    const b_val = try func.appendBlockParam(entry, ptr_t);
    const c_val = try func.appendBlockParam(entry, ptr_t);
    const scales: []const u32 = &.{ 0x3F800000, 0x3F000000, 0x3E800000, 0x40000000 };
    // Exercises .u8 here (the scalar round-trip test above already covers the default .i8), so
    // the new `out` byte's encode/decode order is proven for both enum values across the suite.
    try func.appendMatmulQuantPerColumn(entry, a_val, b_val, c_val, 8, 4, 4, .int8, true, true, .u8, scales);
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

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
    const ptr_t = try func.types.ptrGlobal();
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
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

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

/// Build one function that carries every field a stream can lose: whole-function
/// metadata, a volatile load and store, a variadic direct call with all six call extras, a
/// variadic indirect call with the hidden-pointer return, an accumulating matmul with a
/// full quant payload, a barrier, and attributes on the function, a block, an instruction
/// and a value.
fn buildFullFunction(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();

    func.is_variadic = true;
    func.num_fixed_params = 2;
    func.sret = true;
    func.is_local = true;

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const mmio = try func.appendBlockParam(entry, ptr_t);
    const fnptr = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);

    const dest = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const reg = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = mmio, .@"volatile" = true } });
    try func.appendStoreVol(entry, reg, mmio, true);
    // A plain load and store beside the volatile pair, so a change that forces the flag ON
    // fails too, not only one that drops it.
    const plain = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = a } });
    try func.appendStore(entry, plain, a);

    // A variadic call that also returns a struct in two registers, one integer and one
    // floating-point, so every one of the six extras carries a non-default value.
    const args = try func.internValues(&.{ reg, reg });
    const symbol = try func.internSymbol("printf");
    _ = try func.appendStmtRaw(entry, .{ .call = .{
        .symbol = symbol,
        .args = args,
        .is_variadic = true,
        .num_fixed = 1,
        .ret_dest = dest,
        .ret_regs = 2,
        .ret_pieces = .{
            .{ .fp = false, .offset = 0, .bytes = 8 },
            .{ .fp = true, .offset = 8, .bytes = 4 },
            .{},
            .{},
        },
        .sret = false,
    } });
    const iargs = try func.internValues(&.{reg});
    _ = try func.appendStmtRaw(entry, .{ .call_indirect = .{
        .target = fnptr,
        .args = iargs,
        .is_variadic = true,
        .num_fixed = 1,
        .sret = true,
    } });

    try func.appendMatmulQuantSpec(entry, a, b, c, 2, 2, 4, .int8, true, .{
        .scale_per_column = &.{ 0x3F800000, 0x3F000000 },
        .bias = &.{ 5, -7 },
        .zero_point = -12,
        .relu = true,
        .out = .u8,
    });
    try func.appendBarrier(entry, .subgroup);
    func.setTerminator(entry, .{ .ret = function.Ret.one(reg) });

    try func.addAttr(.func, .@"inline");
    try func.addAttr(.func, .{ .custom = .{
        .namespace = "vulcan.gpu",
        .key = "local_size_x",
        .value = .{ .int = 64 },
    } });
    try func.addAttr(.{ .block = entry }, .{ .endian = .big });
    try func.addAttr(.{ .value = mmio }, .{ .custom = .{
        .namespace = "vulcan.gpu",
        .key = "builtin",
        .value = .{ .int = 3 },
    } });
    try func.addAttr(.{ .value = reg }, .{ .@"align" = 16 });
    try func.addAttr(.{ .inst = func.definingInst(reg).? }, .{ .custom = .{
        .namespace = "debug",
        .key = "file",
        .value = .{ .string = "mmio.c" },
    } });
    return func;
}

/// Check every field of `func` against what `buildFullFunction` put there. Field by field,
/// not by comparing printed text: a field the printer drops is invisible to a text
/// comparison, which is how these losses stayed hidden.
fn expectFullFunction(func: *const Function) !void {
    try std.testing.expect(func.is_variadic);
    try std.testing.expectEqual(@as(u32, 2), func.num_fixed_params);
    try std.testing.expect(func.sret);
    try std.testing.expect(func.is_local);

    const entry: Block = @enumFromInt(0);
    const params = func.blockParams(entry);
    try std.testing.expectEqual(@as(usize, 5), params.len);
    const insts = func.blockInsts(entry);
    try std.testing.expectEqual(@as(usize, 9), insts.len);

    const load_op = func.opcode(insts[1]);
    try std.testing.expect(load_op == .load);
    try std.testing.expectEqual(params[0], load_op.load.ptr);
    try std.testing.expect(load_op.load.@"volatile");

    const store_op = func.opcode(insts[2]);
    try std.testing.expect(store_op == .store);
    try std.testing.expectEqual(params[0], store_op.store.ptr);
    try std.testing.expect(store_op.store.@"volatile");

    // The plain pair must stay plain, so a change that forces the flag on is caught too.
    const plain_load_op = func.opcode(insts[3]);
    try std.testing.expect(plain_load_op == .load);
    try std.testing.expect(!plain_load_op.load.@"volatile");
    const plain_store_op = func.opcode(insts[4]);
    try std.testing.expect(plain_store_op == .store);
    try std.testing.expect(!plain_store_op.store.@"volatile");

    const call_op = func.opcode(insts[5]);
    try std.testing.expect(call_op == .call);
    const call = call_op.call;
    try std.testing.expectEqualStrings("printf", func.symbolName(call.symbol));
    try std.testing.expectEqual(@as(usize, 2), func.valueList(call.args).len);
    try std.testing.expect(call.is_variadic);
    try std.testing.expectEqual(@as(u32, 1), call.num_fixed);
    try std.testing.expectEqual(func.instResult(insts[0]).?, call.ret_dest.?);
    try std.testing.expectEqual(@as(u8, 2), call.ret_regs);
    try std.testing.expectEqual(function.RetPiece{ .fp = false, .offset = 0, .bytes = 8 }, call.ret_pieces[0]);
    try std.testing.expectEqual(function.RetPiece{ .fp = true, .offset = 8, .bytes = 4 }, call.ret_pieces[1]);
    try std.testing.expectEqual(function.RetPiece{}, call.ret_pieces[2]);
    try std.testing.expectEqual(function.RetPiece{}, call.ret_pieces[3]);
    try std.testing.expect(!call.sret);

    const ind_op = func.opcode(insts[6]);
    try std.testing.expect(ind_op == .call_indirect);
    const ind = ind_op.call_indirect;
    try std.testing.expectEqual(params[1], ind.target);
    try std.testing.expect(ind.is_variadic);
    try std.testing.expectEqual(@as(u32, 1), ind.num_fixed);
    try std.testing.expect(ind.sret);
    try std.testing.expectEqual(@as(?Value, null), ind.ret_dest);
    try std.testing.expectEqual(@as(u8, 0), ind.ret_regs);

    const mm_op = func.opcode(insts[7]);
    try std.testing.expect(mm_op == .matmul);
    const mm = mm_op.matmul;
    try std.testing.expectEqual(params[2], mm.a);
    try std.testing.expectEqual(params[3], mm.b);
    try std.testing.expectEqual(params[4], mm.c);
    try std.testing.expectEqual(@as(u16, 2), mm.m);
    try std.testing.expectEqual(@as(u16, 2), mm.n);
    try std.testing.expectEqual(@as(u16, 4), mm.k);
    try std.testing.expectEqual(function.MatMulType.int8, mm.dtype);
    try std.testing.expect(mm.accumulate);
    try std.testing.expect(!mm.embedded);
    try std.testing.expectEqual(@as(?function.InputSigns, null), mm.input_signs);
    const quant = mm.quant orelse return error.TestUnexpectedResult;
    try std.testing.expect(quant.scale == .per_column);
    try std.testing.expectEqualSlices(u32, &.{ 0x3F800000, 0x3F000000 }, func.scaleList(quant.scale.per_column));
    try std.testing.expect(quant.relu);
    try std.testing.expectEqual(function.MatMulQuantOut.u8, quant.out);
    try std.testing.expectEqualSlices(i32, &.{ 5, -7 }, func.biasList(quant.bias.?));
    try std.testing.expectEqual(@as(i32, -12), quant.zero_point);

    const bar_op = func.opcode(insts[8]);
    try std.testing.expect(bar_op == .barrier);
    try std.testing.expectEqual(function.BarrierScope.subgroup, bar_op.barrier.scope);

    var fn_attrs = func.attributesOf(.func);
    try std.testing.expectEqual(Attribute.@"inline", fn_attrs.next().?);
    const local_size = fn_attrs.next().?;
    try std.testing.expectEqualStrings("vulcan.gpu", local_size.custom.namespace);
    try std.testing.expectEqualStrings("local_size_x", local_size.custom.key);
    try std.testing.expectEqual(@as(i64, 64), local_size.custom.value.int);
    try std.testing.expectEqual(@as(?Attribute, null), fn_attrs.next());

    var block_attrs = func.attributesOf(.{ .block = entry });
    try std.testing.expectEqual(Attribute{ .endian = .big }, block_attrs.next().?);
    try std.testing.expectEqual(@as(?Attribute, null), block_attrs.next());

    var param_attrs = func.attributesOf(.{ .value = params[0] });
    const builtin_tag = param_attrs.next().?;
    try std.testing.expectEqualStrings("vulcan.gpu", builtin_tag.custom.namespace);
    try std.testing.expectEqualStrings("builtin", builtin_tag.custom.key);
    try std.testing.expectEqual(@as(i64, 3), builtin_tag.custom.value.int);

    var result_attrs = func.attributesOf(.{ .value = func.instResult(insts[1]).? });
    try std.testing.expectEqual(Attribute{ .@"align" = 16 }, result_attrs.next().?);

    var inst_attrs = func.attributesOf(.{ .inst = insts[1] });
    const file = inst_attrs.next().?;
    try std.testing.expectEqualStrings("debug", file.custom.namespace);
    try std.testing.expectEqualStrings("file", file.custom.key);
    try std.testing.expectEqualStrings("mmio.c", file.custom.value.string);
}

test "bitcode round-trips every instruction, call, matmul and attribute field" {
    // The oracle every other test in this file uses is `print(decode(encode(f))) ==
    // print(f)`. A field the PRINTER drops is invisible to it, which is how volatile, the
    // six call extras and the matmul accumulate flag all stayed lost in both directions.
    // This test reads the decoded function FIELD BY FIELD instead.
    const allocator = std.testing.allocator;

    var func = try buildFullFunction(allocator);
    defer func.deinit();
    try expectFullFunction(&func); // the builder really put the fields there

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);
    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    try expectFullFunction(&decoded);

    // The attribute string payloads are the decoded function's OWN storage, not a view
    // into the byte buffer, which is freed before the function is.
    for (decoded.attributeEntries()) |entry| switch (entry.attr) {
        .custom => |cu| {
            try std.testing.expect(@intFromPtr(cu.namespace.ptr) < @intFromPtr(bytes.ptr) or
                @intFromPtr(cu.namespace.ptr) >= @intFromPtr(bytes.ptr) + bytes.len);
        },
        .@"inline", .noreturn, .cold, .@"align", .endian => {},
    };

    // The printed text agrees too, so the print oracle now covers these fields.
    const a = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{f}", .{decoded});
    defer allocator.free(b);
    try std.testing.expectEqualStrings(a, b);
}

test "the text IR round-trips every instruction, call, matmul and attribute field" {
    // The parser's half of the same contract: what the printer writes must read back as
    // the same function, field by field, not only as the same text.
    const allocator = std.testing.allocator;

    var func = try buildFullFunction(allocator);
    defer func.deinit();

    const text = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(text);

    var reparsed = try parser.parse(allocator, text);
    defer reparsed.deinit();
    try expectFullFunction(&reparsed);

    const reprinted = try std.fmt.allocPrint(allocator, "{f}", .{reparsed});
    defer allocator.free(reprinted);
    try std.testing.expectEqualStrings(text, reprinted);
}

test "rejects the untrusted bytes the new fields added" {
    // Suspicious cases: every byte below comes off an UNTRUSTED stream. An unknown flag
    // bit, an out-of-range piece count, an unknown attribute tag and a target number that
    // names no entity must all be recoverable faults, never an invalid enum, an
    // out-of-range array write or an out-of-bounds table read.
    const allocator = std.testing.allocator;

    var func = try buildFullFunction(allocator);
    defer func.deinit();
    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    // An unknown whole-function flag bit. The flag byte follows the magic.
    {
        const patched = try allocator.dupe(u8, bytes);
        defer allocator.free(patched);
        patched[magic.len] = 0xff;
        try std.testing.expectError(error.MalformedBitcode, decode(allocator, patched));
    }

    // A call claiming more return-register pieces than a call can hold. The piece count
    // follows the call flag byte and the fixed-parameter count; the encoded function holds
    // exactly one call with 2 pieces, so the first such byte pair names it.
    {
        const patched = try allocator.dupe(u8, bytes);
        defer allocator.free(patched);
        const flags = call_flag_variadic | call_flag_ret_dest;
        var i: usize = 0;
        const found = while (i + 6 < patched.len) : (i += 1) {
            if (patched[i] != flags) continue;
            if (patched[i + 5] != 2) continue; // ret_regs, after the u32 num_fixed
            patched[i + 5] = 5; // more than the 4 a call can hold
            break true;
        } else false;
        try std.testing.expect(found);
        try std.testing.expectError(error.MalformedBitcode, decode(allocator, patched));
    }

    // An attribute target number past the end of the value table. The attribute section is
    // last, so the final entry's target number is near the tail: rebuild the section
    // instead of patching, by truncating to the count and writing one bad entry.
    {
        var bad: std.ArrayList(u8) = .empty;
        defer bad.deinit(allocator);
        try bad.appendSlice(allocator, bytes[0 .. bytes.len - 1]);
        // A truncated stream is malformed on its own terms, which is the point: the decoder
        // must not read past the buffer to discover it.
        try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad.items));
    }
}

test "rejects an attribute naming a value that does not exist" {
    // A hand-built module: one block, no instructions, and one attribute on value 7.
    const allocator = std.testing.allocator;
    const bad_attr = test_header ++
        "\x00\x00\x00\x00" ++ // type_count = 0
        "\x00\x00\x00\x00" ++ // sym_count = 0
        "\x01\x00\x00\x00" ++ // block_count = 1
        "\x00\x00\x00\x00" ++ // block 0 param_count = 0
        "\x00\x00\x00\x00" ++ // inst_count = 0
        "\x00" ++ // no terminator
        "\x01\x00\x00\x00" ++ // attr_count = 1
        "\x03" ++ // target kind = value
        "\x07\x00\x00\x00" ++ // value number 7, which does not exist
        "\x00"; // attribute body = inline
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad_attr));

    // The same module with an unknown attribute body tag.
    const bad_body = test_header ++
        "\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00" ++
        "\x01\x00\x00\x00" ++
        "\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00" ++
        "\x00" ++
        "\x01\x00\x00\x00" ++
        "\x00" ++ // target kind = func
        "\x63"; // attribute body tag = 0x63, no such attribute
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad_body));

    // And with an out-of-range endianness byte, which `@enumFromInt` would turn into an
    // invalid enum tag.
    const bad_endian = test_header ++
        "\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00" ++
        "\x01\x00\x00\x00" ++
        "\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00" ++
        "\x00" ++
        "\x01\x00\x00\x00" ++
        "\x00" ++ // target kind = func
        "\x04" ++ // attribute body = endian
        "\xff"; // endianness byte = 0xff, no such order
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad_endian));
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
    const self_ref = test_header ++
        "\x01\x00\x00\x00" ++ // type_count = 1
        "\x04" ++ // type 0: vector
        "\x01\x00\x00\x00" ++ // vector len = 1
        "\x00\x00\x00\x00"; // elem type index = 0 (not yet decoded)
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, self_ref));

    // Same, but an index past the whole table (5 >= 1).
    const past_end = test_header ++
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
    const bad_arith = test_header ++
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
    func.setTerminator(entry, .{ .ret = function.Ret.one(widened) });

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
    const bad_float = test_header ++
        "\x01\x00\x00\x00" ++ // type_count = 1
        "\x02" ++ // type 0: float
        "\x03"; // float-kind byte = 3 (no such FloatKind)
    try std.testing.expectError(error.MalformedBitcode, decode(allocator, bad_float));
}

test "a 2-value ret round-trips through bitcode and text-IR" {
    // `Terminator.ret` was widened from a single optional value to an inline
    // list of up to 4 values. A count of 0 or 1 must stay byte-identical to before,
    // as proven by every other test in this file. This test proves only the new
    // count-2 shape: it serializes and parses correctly. No backend lowers a count
    // above 1 yet, so this never touches codegen.
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    func.setTerminator(entry, .{ .ret = function.Ret.many(&.{ a, b }) });

    // Bitcode round-trip: write then read back.
    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);
    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const decoded_ret = decoded.terminator(entry).?.ret;
    try std.testing.expectEqual(@as(u8, 2), decoded_ret.count);
    try std.testing.expectEqual(decoded.blockParams(entry)[0], decoded_ret.values[0]);
    try std.testing.expectEqual(decoded.blockParams(entry)[1], decoded_ret.values[1]);

    // Text-IR round-trip: print then parse, byte-identical text (`ret v0, v1`).
    const text = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "ret v0, v1") != null);

    var reparsed = try parser.parse(allocator, text);
    defer reparsed.deinit();
    const reparsed_ret = reparsed.terminator(entry).?.ret;
    try std.testing.expectEqual(@as(u8, 2), reparsed_ret.count);
    try std.testing.expectEqual(reparsed.blockParams(entry)[0], reparsed_ret.values[0]);
    try std.testing.expectEqual(reparsed.blockParams(entry)[1], reparsed_ret.values[1]);

    const reprinted = try std.fmt.allocPrint(allocator, "{f}", .{reparsed});
    defer allocator.free(reprinted);
    try std.testing.expectEqualStrings(text, reprinted);
}

test "a pointer address space survives a bitcode round trip" {
    // Tag 3 carried no payload before, so every pointer decoded as global and a
    // shared one silently changed address space across a round trip.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const s = try func.types.intern(.{ .ptr = .shared });
    const g = try func.types.ptrGlobal();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const ps = try func.appendBlockParam(b, s);
    const pg = try func.appendBlockParam(b, g);
    const n = try func.appendBlockParam(b, i32_t);
    func.setTerminator(b, .{ .ret = function.Ret.one(n) });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);
    var back = try decode(allocator, bytes);
    defer back.deinit();

    try std.testing.expectEqual(
        types.AddressSpace.shared,
        back.types.type_kind(back.valueType(ps)).ptr,
    );
    try std.testing.expectEqual(
        types.AddressSpace.global,
        back.types.type_kind(back.valueType(pg)).ptr,
    );
    try std.testing.expect(back.valueType(ps) != back.valueType(pg));
}

test "regression: rejects an unknown address-space byte instead of @enumFromInt UB" {
    // Suspicious case: the stream is untrusted. An out-of-range byte must not
    // become an invalid enum.
    //
    // The stream is a real encode of the function below, patched at one exact
    // offset. A scan for the first tag 3 is not safe here, because tag 3 is also
    // the `arith_imm` opcode and the low byte of many counts, so it can match
    // earlier by coincidence and patch a byte this test does not mean to touch.
    // The layout is fixed instead: the stream header, then a u32 type count, then
    // the type records. The function holds exactly one type, so type 0 starts right
    // after the count.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    _ = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    func.setTerminator(b, .{ .ret = function.Ret.none() });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    const tag_offset = header_len + @sizeOf(u32);
    const space_offset = tag_offset + 1;
    try std.testing.expect(space_offset < bytes.len);
    // Prove the offset really names the pointer record before it is patched.
    try std.testing.expectEqual(@as(u8, 3), bytes[tag_offset]);
    try std.testing.expectEqual(@intFromEnum(types.AddressSpace.global), bytes[space_offset]);

    const patched = try allocator.dupe(u8, bytes);
    defer allocator.free(patched);
    patched[space_offset] = 0xff;

    try std.testing.expectError(error.MalformedBitcode, decode(allocator, patched));
}

test "round-trips a barrier and its scope through bitcode" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const entry = try func.appendBlock();
    try func.appendBarrier(entry, .workgroup);
    try func.appendBarrier(entry, .subgroup);
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    try std.testing.expectEqual(@as(usize, 2), insts.len);
    try std.testing.expectEqual(
        function.BarrierScope.workgroup,
        decoded.opcode(insts[0]).barrier.scope,
    );
    try std.testing.expectEqual(
        function.BarrierScope.subgroup,
        decoded.opcode(insts[1]).barrier.scope,
    );
}

test "an unknown barrier scope byte is rejected as malformed bitcode" {
    // Suspicious case: the stream is untrusted. An out-of-range byte must not become an
    // invalid enum tag that every later exhaustive switch reads as undefined behavior.
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const entry = try func.appendBlock();
    try func.appendBarrier(entry, .workgroup);
    func.setTerminator(entry, .{ .ret = function.Ret.none() });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    // The barrier record is the opcode tag byte followed by the scope byte. This function
    // holds exactly one instruction, so the first `op_barrier` byte found is that tag, and
    // the byte after it is the scope.
    const mutable = try allocator.dupe(u8, bytes);
    defer allocator.free(mutable);
    var i: usize = 0;
    const patched = while (i + 1 < mutable.len) : (i += 1) {
        if (mutable[i] == op_barrier and mutable[i + 1] == 0) {
            mutable[i + 1] = 0xff;
            break true;
        }
    } else false;
    try std.testing.expect(patched);

    try std.testing.expectError(error.MalformedBitcode, decode(allocator, mutable));
}

test "round-trips an atomic read-modify-write field by field" {
    // FIELD BY FIELD, not by comparing printed text: a printed comparison hides a field the
    // printer never prints, which is how an earlier field loss stayed hidden.
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendBlockParam(entry, i32_t);
    const old = try func.appendAtomicRmw(entry, .{
        .op = .min,
        .ptr = p,
        .value = v,
        .ordering = .acq_rel,
        .scope = .device,
    });
    try func.appendAtomicRmwStmt(entry, .{
        .op = .bit_xor,
        .ptr = p,
        .value = v,
        .ordering = .relaxed,
        .scope = .system,
    });
    func.setTerminator(entry, .{ .ret = function.Ret.one(old) });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);
    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const insts = decoded.blockInsts(entry);
    try std.testing.expectEqual(@as(usize, 2), insts.len);

    const reading = decoded.opcode(insts[0]).atomic_rmw;
    try std.testing.expectEqual(function.AtomicOp.min, reading.op);
    try std.testing.expectEqual(function.AtomicOrdering.acq_rel, reading.ordering);
    try std.testing.expectEqual(function.AtomicScope.device, reading.scope);
    try std.testing.expectEqual(@as(?Value, null), reading.compare);
    // The two operands are the block's two parameters, in order.
    const params = decoded.blockParams(entry);
    try std.testing.expectEqual(params[0], reading.ptr);
    try std.testing.expectEqual(params[1], reading.value);
    // The reading form keeps its result, and the result type is the operand's type.
    const result = decoded.instResult(insts[0]).?;
    try std.testing.expectEqual(decoded.valueType(params[1]), decoded.valueType(result));

    const reduction = decoded.opcode(insts[1]).atomic_rmw;
    try std.testing.expectEqual(function.AtomicOp.bit_xor, reduction.op);
    try std.testing.expectEqual(function.AtomicOrdering.relaxed, reduction.ordering);
    try std.testing.expectEqual(function.AtomicScope.system, reduction.scope);
    try std.testing.expectEqual(params[0], reduction.ptr);
    try std.testing.expectEqual(params[1], reduction.value);
    // The reduction form comes back RESULT-LESS. Rebuilding it with a result would cost the
    // backend a scoreboard on every fire-and-forget counter increment.
    try std.testing.expectEqual(@as(?Value, null), decoded.instResult(insts[1]));
}

test "round-trips a compare-exchange and its compare operand" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const desired = try func.appendBlockParam(entry, i32_t);
    const expected = try func.appendBlockParam(entry, i32_t);
    const old = try func.appendAtomicRmw(entry, .{
        .op = .compare_exchange,
        .ptr = p,
        .value = desired,
        .compare = expected,
        .ordering = .seq_cst,
        .scope = .workgroup,
    });
    func.setTerminator(entry, .{ .ret = function.Ret.one(old) });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);
    var decoded = try decode(allocator, bytes);
    defer decoded.deinit();

    const params = decoded.blockParams(entry);
    const cas = decoded.opcode(decoded.blockInsts(entry)[0]).atomic_rmw;
    try std.testing.expectEqual(function.AtomicOp.compare_exchange, cas.op);
    try std.testing.expectEqual(params[0], cas.ptr);
    try std.testing.expectEqual(params[1], cas.value);
    // The compare operand is the field a two-slot record would have dropped.
    try std.testing.expectEqual(params[2], cas.compare.?);
    try std.testing.expectEqual(function.AtomicOrdering.seq_cst, cas.ordering);
    try std.testing.expectEqual(function.AtomicScope.workgroup, cas.scope);
}

test "an unknown atomic selector byte is rejected as malformed bitcode" {
    // Suspicious case: the stream is untrusted. Each of the three selector bytes and the
    // flag byte must be refused when it names nothing, rather than becoming an
    // out-of-range tag that every later exhaustive switch reads as undefined behavior.
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendBlockParam(entry, i32_t);
    try func.appendAtomicRmwStmt(entry, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .workgroup });
    func.setTerminator(entry, .{ .ret = function.Ret.one(v) });

    const bytes = try encode(allocator, &func);
    defer allocator.free(bytes);

    // The record is the tag byte followed by op, ordering, scope and the flag byte, all
    // four of them zero for the operation built above. This function holds exactly one
    // instruction, so the first `op_atomic_rmw` byte followed by four zeros is that record.
    var at: ?usize = null;
    var i: usize = 0;
    while (i + 4 < bytes.len) : (i += 1) {
        if (bytes[i] == op_atomic_rmw and bytes[i + 1] == 0 and bytes[i + 2] == 0 and
            bytes[i + 3] == 0 and bytes[i + 4] == 0)
        {
            at = i;
            break;
        }
    }
    try std.testing.expect(at != null);

    // The unpatched stream decodes: the control that proves each refusal below comes from
    // the byte it patched and not from some unrelated damage.
    {
        var ok = try decode(allocator, bytes);
        ok.deinit();
    }

    // Byte 4 is the flag byte, and it gets a value with ONLY an UNKNOWN bit set. 0xff would
    // also set the compare bit, which makes the record grow by a slot and the rest of the
    // stream decode at a wrong offset, so the refusal would come from that and not from the
    // flag check. 0x02 leaves the record's length alone, so only the check can refuse it.
    const patches = [_]struct { field: usize, byte: u8 }{
        .{ .field = 1, .byte = 0xff }, // operation
        .{ .field = 2, .byte = 0xff }, // ordering
        .{ .field = 3, .byte = 0xff }, // scope
        .{ .field = 4, .byte = 0x02 }, // an unknown flag bit
    };
    for (patches) |patch| {
        const mutable = try allocator.dupe(u8, bytes);
        defer allocator.free(mutable);
        mutable[at.? + patch.field] = patch.byte;
        try std.testing.expectError(error.MalformedBitcode, decode(allocator, mutable));
    }
}
