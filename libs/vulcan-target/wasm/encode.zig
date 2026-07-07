//! Wasm (MVP) binary encoding helpers. Section IDs, value types, opcode bytes,
//! and LEB128 variable-length integer encoding.

const std = @import("std");
const ir = @import("vulcan-ir");

// Wasm binary format constants

/// Section IDs in the Wasm binary format.
pub const SectionId = enum(u8) {
    custom,
    type,
    import,
    function,
    table,
    memory,
    global,
    @"export",
    start,
    element,
    code,
    data,

    pub fn toByte(id: SectionId) u8 {
        return @intFromEnum(id);
    }
};

/// Wasm value types (opcode bytes).
pub const ValType = enum(u8) {
    i32,
    i64,
    f32,
    f64,

    pub fn toByte(vt: ValType) u8 {
        return switch (vt) {
            .i32 => 0x7F,
            .i64 => 0x7E,
            .f32 => 0x7D,
            .f64 => 0x7C,
        };
    }
};

/// Wasm block return types (empty or a single value type).
pub const BlockType = enum(u8) {
    empty,
    i32,
    i64,
    f32,
    f64,

    pub fn toByte(bt: BlockType) u8 {
        return switch (bt) {
            .empty => 0x40,
            .i32 => 0x7F,
            .i64 => 0x7E,
            .f32 => 0x7D,
            .f64 => 0x7C,
        };
    }
};

/// Wasm memory descriptor flags.
pub const MemFlags = enum(u8) {
    no_max,
    has_max,

    pub fn toByte(f: MemFlags) u8 {
        return switch (f) {
            .no_max => 0x00,
            .has_max => 0x01,
        };
    }
};

/// Wasm global mutability.
pub const GlobalMut = enum(u8) {
    const_val,
    var_val,

    pub fn toByte(m: GlobalMut) u8 {
        return switch (m) {
            .const_val => 0x00,
            .var_val => 0x01,
        };
    }
};

// Opcodes

/// Block control flow opcodes.
pub const ControlOp = struct {
    pub const unreachable_: u8 = 0x00;
    pub const nop: u8 = 0x01;
    pub const block: u8 = 0x02;
    pub const loop: u8 = 0x03;
    pub const if_: u8 = 0x04;
    pub const else_: u8 = 0x05;
    pub const end: u8 = 0x0B;
    pub const br: u8 = 0x0C;
    pub const br_if: u8 = 0x0D;
    pub const br_table: u8 = 0x0E;
    pub const return_: u8 = 0x0F;
    pub const call: u8 = 0x10;
    pub const call_indirect: u8 = 0x11;
    pub const drop: u8 = 0x1A;
    pub const select: u8 = 0x1B;
};

/// Memory opcodes.
pub const MemOp = struct {
    pub const load32: u8 = 0x28;
    pub const load64: u8 = 0x29;
    pub const load32f32: u8 = 0x2A;
    pub const load64f64: u8 = 0x2B;
    pub const load8_s: u8 = 0x2C;
    pub const load8_u: u8 = 0x2D;
    pub const load16_s: u8 = 0x2E;
    pub const load16_u: u8 = 0x2F;
    pub const load64_8s: u8 = 0x30;
    pub const load64_8u: u8 = 0x31;
    pub const load64_16s: u8 = 0x32;
    pub const load64_16u: u8 = 0x33;
    pub const load64_32s: u8 = 0x34;
    pub const load64_32u: u8 = 0x35;
    pub const store32: u8 = 0x36;
    pub const store64: u8 = 0x37;
    pub const store32f32: u8 = 0x38;
    pub const store64f64: u8 = 0x39;
    pub const store8: u8 = 0x3A;
    pub const store16: u8 = 0x3B;
    pub const store64_8: u8 = 0x3C;
    pub const store64_16: u8 = 0x3D;
    pub const store64_32: u8 = 0x3E;
    pub const memory_size: u8 = 0x3F;
    pub const memory_grow: u8 = 0x40;
};

/// Local variable opcodes.
pub const LocalOp = struct {
    pub const local_get: u8 = 0x20;
    pub const local_set: u8 = 0x21;
    pub const local_tee: u8 = 0x22;
};

/// Global opcodes.
pub const GlobalOp = struct {
    pub const global_get: u8 = 0x23;
    pub const global_set: u8 = 0x24;
};

/// Constant opcodes.
pub const ConstOp = struct {
    pub const i32_const: u8 = 0x41;
    pub const i64_const: u8 = 0x42;
    pub const f32_const: u8 = 0x43;
    pub const f64_const: u8 = 0x44;
    pub const ref_null: u8 = 0x50;
};

/// i32 integer opcodes.
pub const I32Op = struct {
    pub const eqz: u8 = 0x45;
    pub const eq: u8 = 0x46;
    pub const ne: u8 = 0x47;
    pub const lt_s: u8 = 0x48;
    pub const lt_u: u8 = 0x49;
    pub const gt_s: u8 = 0x4A;
    pub const gt_u: u8 = 0x4B;
    pub const le_s: u8 = 0x4C;
    pub const le_u: u8 = 0x4D;
    pub const ge_s: u8 = 0x4E;
    pub const ge_u: u8 = 0x4F;
    pub const add: u8 = 0x6A;
    pub const sub: u8 = 0x6B;
    pub const mul: u8 = 0x6C;
    pub const div_s: u8 = 0x6D;
    pub const div_u: u8 = 0x6E;
    pub const rem_s: u8 = 0x6F;
    pub const rem_u: u8 = 0x70;
    pub const bit_and: u8 = 0x71;
    pub const bit_or: u8 = 0x72;
    pub const bit_xor: u8 = 0x73;
    pub const shl: u8 = 0x74;
    pub const shr_s: u8 = 0x75;
    pub const shr_u: u8 = 0x76;
    pub const rotl: u8 = 0x77;
    pub const rotr: u8 = 0x78;
    pub const clz: u8 = 0x67;
    pub const ctz: u8 = 0x68;
    pub const popcnt: u8 = 0x69;
    pub const wrap_i64: u8 = 0xA7;
    // These are the non-saturating i32.trunc_f* ops (they trap on NaN/overflow). The
    // saturating forms are 0xFC-prefixed. The isel uses these for in-range floats.
    pub const trunc_sat_f32_s: u8 = 0xA8;
    pub const trunc_sat_f32_u: u8 = 0xA9;
    pub const trunc_sat_f64_s: u8 = 0xAA;
    pub const trunc_sat_f64_u: u8 = 0xAB;
    pub const reinterpret_i32: u8 = 0xBC;
};

/// i64 integer opcodes.
pub const I64Op = struct {
    pub const eqz: u8 = 0x45;
    pub const eq: u8 = 0x51;
    pub const ne: u8 = 0x52;
    pub const lt_s: u8 = 0x53;
    pub const lt_u: u8 = 0x54;
    pub const gt_s: u8 = 0x55;
    pub const gt_u: u8 = 0x56;
    pub const le_s: u8 = 0x57;
    pub const le_u: u8 = 0x58;
    pub const ge_s: u8 = 0x59;
    pub const ge_u: u8 = 0x5A;
    pub const add: u8 = 0x7C;
    pub const sub: u8 = 0x7D;
    pub const mul: u8 = 0x7E;
    pub const div_s: u8 = 0x7F;
    pub const div_u: u8 = 0x80;
    pub const rem_s: u8 = 0x81;
    pub const rem_u: u8 = 0x82;
    pub const bit_and: u8 = 0x83;
    pub const bit_or: u8 = 0x84;
    pub const bit_xor: u8 = 0x85;
    pub const shl: u8 = 0x86;
    pub const shr_s: u8 = 0x87;
    pub const shr_u: u8 = 0x88;
    pub const rotl: u8 = 0x89;
    pub const rotr: u8 = 0x8A;
    pub const clz: u8 = 0x67;
    pub const ctz: u8 = 0x68;
    pub const popcnt: u8 = 0x69;
    pub const extend_i32_s: u8 = 0xAC;
    pub const extend_i32_u: u8 = 0xAD;
    // Non-saturating i64.trunc_f* (the saturating forms are 0xFC-prefixed).
    pub const trunc_sat_f32_s: u8 = 0xAE;
    pub const trunc_sat_f32_u: u8 = 0xAF;
    pub const trunc_sat_f64_s: u8 = 0xB0;
    pub const trunc_sat_f64_u: u8 = 0xB1;
    pub const reinterpret_i64: u8 = 0xBD;
};

/// f32 float opcodes.
pub const F32Op = struct {
    pub const eq: u8 = 0x5B;
    pub const ne: u8 = 0x5C;
    pub const lt: u8 = 0x5D;
    pub const gt: u8 = 0x5E;
    pub const le: u8 = 0x5F;
    pub const ge: u8 = 0x60;
    pub const abs: u8 = 0x8B;
    pub const neg: u8 = 0x8C;
    pub const add: u8 = 0x92;
    pub const sub: u8 = 0x93;
    pub const mul: u8 = 0x94;
    pub const div: u8 = 0x95;
    pub const ceil: u8 = 0x8D;
    pub const floor: u8 = 0x8E;
    pub const trunc: u8 = 0x8F;
    pub const nearest: u8 = 0x90;
    pub const sqrt: u8 = 0x91;
    pub const min: u8 = 0x96;
    pub const max: u8 = 0x97;
    pub const copysign: u8 = 0x98;
    pub const convert_i32_s: u8 = 0xB2;
    pub const convert_i32_u: u8 = 0xB3;
    pub const convert_i64_s: u8 = 0xB4;
    pub const convert_i64_u: u8 = 0xB5;
    pub const demote_f64: u8 = 0xB6;
    pub const reinterpret_i32: u8 = 0xBE;
};

/// f64 float opcodes.
pub const F64Op = struct {
    pub const eq: u8 = 0x61;
    pub const ne: u8 = 0x62;
    pub const lt: u8 = 0x63;
    pub const gt: u8 = 0x64;
    pub const le: u8 = 0x65;
    pub const ge: u8 = 0x66;
    pub const abs: u8 = 0x99;
    pub const neg: u8 = 0x9A;
    pub const add: u8 = 0xA0;
    pub const sub: u8 = 0xA1;
    pub const mul: u8 = 0xA2;
    pub const div: u8 = 0xA3;
    pub const ceil: u8 = 0x9B;
    pub const floor: u8 = 0x9C;
    pub const trunc: u8 = 0x9D;
    pub const nearest: u8 = 0x9E;
    pub const sqrt: u8 = 0x9F;
    pub const min: u8 = 0xA4;
    pub const max: u8 = 0xA5;
    pub const copysign: u8 = 0xA6;
    pub const convert_i32_s: u8 = 0xB7;
    pub const convert_i32_u: u8 = 0xB8;
    pub const convert_i64_s: u8 = 0xB9;
    pub const convert_i64_u: u8 = 0xBA;
    pub const promote_f32: u8 = 0xBB;
    pub const reinterpret_i64: u8 = 0xBF;
};

// LEB128 encoding

pub fn encodeU32leb(buf: []u8, value: u32) usize {
    var v = value;
    var i: usize = 0;
    // Mask the low 7 bits before narrowing to u8: without the mask `@intCast(v)` panics as
    // soon as `v` exceeds 255 (e.g. a function body 256+ bytes long).
    while (v > 0x7F) {
        buf[i] = @as(u8, @intCast(v & 0x7F)) | 0x80;
        i += 1;
        v >>= 7;
    }
    buf[i] = @as(u8, @intCast(v));
    return i + 1;
}

pub fn encodeS32leb(buf: []u8, value: i32) usize {
    var v: i32 = value;
    var i: usize = 0;
    while (true) {
        const byte: u8 = @as(u8, @intCast(v & 0x7F));
        v >>= 7;
        const done: bool = (v == 0 and (byte & 0x40) == 0) or (v == -1 and (byte & 0x40) != 0);
        if (done) {
            buf[i] = byte;
        } else {
            buf[i] = byte | 0x80;
        }
        i += 1;
        if (done) break;
    }
    return i;
}

pub fn encodeU64leb(buf: []u8, value: u64) usize {
    var v = value;
    var i: usize = 0;
    // Mask the low 7 bits before narrowing (see encodeU32leb): an unmasked `@intCast` panics
    // once the value exceeds 255.
    while (v > 0x7F) {
        buf[i] = @as(u8, @intCast(v & 0x7F)) | 0x80;
        i += 1;
        v >>= 7;
    }
    buf[i] = @as(u8, @intCast(v));
    return i + 1;
}

pub fn encodeS64leb(buf: []u8, value: i64) usize {
    var v: i64 = value;
    var i: usize = 0;
    while (true) {
        const byte: u8 = @as(u8, @intCast(v & 0x7F));
        v >>= 7;
        const done: bool = (v == 0 and (byte & 0x40) == 0) or (v == -1 and (byte & 0x40) != 0);
        if (done) {
            buf[i] = byte;
        } else {
            buf[i] = byte | 0x80;
        }
        i += 1;
        if (done) break;
    }
    return i;
}

// Type mapping

/// Map an IR type handle to a Wasm value type using the function's type table.
/// Returns null for unsupported types (pointers, vectors, aggregates, etc.).
pub fn irTypeToWasm(types: *const ir.types.TypeTable, ty: ir.types.Type) ?ValType {
    return switch (types.type_kind(ty)) {
        .bool => .i32,
        .int => |info| switch (info.bits) {
            8, 16, 32 => .i32,
            64 => .i64,
            else => null,
        },
        .float => |kind| switch (kind) {
            .f32 => .f32,
            .f64 => .f64,
        },
        else => null,
    };
}

/// Map an IR type to the Wasm memory load opcode for the given width and signedness.
pub fn irLoadOp(types: *const ir.types.TypeTable, ty: ir.types.Type) u8 {
    const kind = types.type_kind(ty);
    return switch (kind) {
        .int => |info| switch (info.bits) {
            8 => if (info.signedness == .signed) MemOp.load8_s else MemOp.load8_u,
            16 => if (info.signedness == .signed) MemOp.load16_s else MemOp.load16_u,
            32 => MemOp.load32,
            64 => MemOp.load64,
            else => unreachable,
        },
        .float => |f| switch (f) {
            .f32 => MemOp.load32f32,
            .f64 => MemOp.load64f64,
        },
        else => unreachable,
    };
}

/// Map an IR type to the Wasm memory store opcode for the given width.
pub fn irStoreOp(types: *const ir.types.TypeTable, ty: ir.types.Type) u8 {
    const kind = types.type_kind(ty);
    return switch (kind) {
        .int => |info| switch (info.bits) {
            8 => MemOp.store8,
            16 => MemOp.store16,
            32 => MemOp.store32,
            64 => MemOp.store64,
            else => unreachable,
        },
        .float => |f| switch (f) {
            .f32 => MemOp.store32f32,
            .f64 => MemOp.store64f64,
        },
        else => unreachable,
    };
}

/// Get the i32 comparison opcode for the given CmpOp and signedness.
pub fn irCmpI32Op(op: ir.function.CmpOp, signed: std.builtin.Signedness) u8 {
    return switch (op) {
        .eq => I32Op.eq,
        .ne => I32Op.ne,
        .lt => if (signed == .signed) I32Op.lt_s else I32Op.lt_u,
        .le => if (signed == .signed) I32Op.le_s else I32Op.le_u,
        .gt => if (signed == .signed) I32Op.gt_s else I32Op.gt_u,
        .ge => if (signed == .signed) I32Op.ge_s else I32Op.ge_u,
    };
}

/// Get the i64 comparison opcode for the given CmpOp and signedness.
pub fn irCmpI64Op(op: ir.function.CmpOp, signed: std.builtin.Signedness) u8 {
    return switch (op) {
        .eq => I64Op.eq,
        .ne => I64Op.ne,
        .lt => if (signed == .signed) I64Op.lt_s else I64Op.lt_u,
        .le => if (signed == .signed) I64Op.le_s else I64Op.le_u,
        .gt => if (signed == .signed) I64Op.gt_s else I64Op.gt_u,
        .ge => if (signed == .signed) I64Op.ge_s else I64Op.ge_u,
    };
}

/// Get the i32 arithmetic opcode for the given BinOp.
pub fn irArithI32Op(op: ir.function.BinOp) u8 {
    return switch (op) {
        .add => I32Op.add,
        .sub => I32Op.sub,
        .mul => I32Op.mul,
        .div => I32Op.div_s,
        .rem => I32Op.rem_s,
        .bit_and => I32Op.bit_and,
        .bit_or => I32Op.bit_or,
        .bit_xor => I32Op.bit_xor,
        .shl => I32Op.shl,
        .shr => I32Op.shr_s,
    };
}

/// Get the i64 arithmetic opcode for the given BinOp.
pub fn irArithI64Op(op: ir.function.BinOp) u8 {
    return switch (op) {
        .add => I64Op.add,
        .sub => I64Op.sub,
        .mul => I64Op.mul,
        .div => I64Op.div_s,
        .rem => I64Op.rem_s,
        .bit_and => I64Op.bit_and,
        .bit_or => I64Op.bit_or,
        .bit_xor => I64Op.bit_xor,
        .shl => I64Op.shl,
        .shr => I64Op.shr_s,
    };
}

// Test

test "LEB128 encodes zero" {
    var buf: [10]u8 = undefined;
    try std.testing.expectEqual(1, encodeU32leb(&buf, 0));
    try std.testing.expectEqual(@as(u8, 0), buf[0]);
}

test "LEB128 encodes 127" {
    var buf: [10]u8 = undefined;
    try std.testing.expectEqual(1, encodeU32leb(&buf, 127));
    try std.testing.expectEqual(@as(u8, 127), buf[0]);
}

test "LEB128 encodes 128" {
    var buf: [10]u8 = undefined;
    try std.testing.expectEqual(2, encodeU32leb(&buf, 128));
    try std.testing.expectEqual(@as(u8, 128), buf[0]);
    try std.testing.expectEqual(@as(u8, 1), buf[1]);
}

test "LEB128 encodes signed negative" {
    var buf: [10]u8 = undefined;
    try std.testing.expectEqual(1, encodeS32leb(&buf, -1));
    try std.testing.expectEqual(@as(u8, 0x7F), buf[0]);
}

test "LEB128 encodes values that need more than one byte" {
    var buf: [10]u8 = undefined;
    // 256 crossed the u8 boundary and used to panic in the unmasked cast.
    try std.testing.expectEqual(2, encodeU32leb(&buf, 256));
    try std.testing.expectEqualSlices(u8, &.{ 0x80, 0x02 }, buf[0..2]);
    // 300 = [0xAC, 0x02].
    try std.testing.expectEqual(2, encodeU32leb(&buf, 300));
    try std.testing.expectEqualSlices(u8, &.{ 0xAC, 0x02 }, buf[0..2]);
    // 624485 = [0xE5, 0x8E, 0x26] (the canonical LEB128 example).
    try std.testing.expectEqual(3, encodeU32leb(&buf, 624485));
    try std.testing.expectEqualSlices(u8, &.{ 0xE5, 0x8E, 0x26 }, buf[0..3]);
    // Same for the 64-bit encoder.
    try std.testing.expectEqual(3, encodeU64leb(&buf, 624485));
    try std.testing.expectEqualSlices(u8, &.{ 0xE5, 0x8E, 0x26 }, buf[0..3]);
}
