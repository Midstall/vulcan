//! Wasm instruction selection: lowers a Vulcan IR function to Wasm (MVP) bytecode.
//! Wasm is a stack machine with locals. Each IR value is assigned a Wasm local
//! variable, and instructions emit local.get/local.set sequences.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const disasm = @import("disasm.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Opcode = ir.function.Opcode;
const BinOp = ir.function.BinOp;
const CmpOp = ir.function.CmpOp;
const Terminator = ir.function.Terminator;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// The emitted Wasm bytecode for a single function (as raw bytes).
pub const Compiled = struct {
    code: []u8,

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
    }
};

/// A Wasm function type signature: parameter and result value types.
pub const Signature = struct {
    params: []const encode.ValType,
    results: []const encode.ValType,
};

/// Module-level information the linker owns and isel needs: the deduplicated type
/// section (so `call_indirect` can name a type index) and the function names in
/// module order (so a direct `call` resolves a symbol name to a function index,
/// rather than trusting the caller's interning order to match the module layout).
pub const ModuleResolver = struct {
    sigs: []const Signature,
    func_names: []const []const u8,
    /// The index of the mutable i32 stack-pointer global, if the module declares one
    /// (it does when any function allocates). alloca-heavy functions carve their frame
    /// from this descending stack so allocas never alias across calls.
    sp_global: ?u32 = null,

    pub fn indexOf(self: *const ModuleResolver, params: []const encode.ValType, results: []const encode.ValType) ?u32 {
        for (self.sigs, 0..) |s, i| {
            if (std.mem.eql(encode.ValType, s.params, params) and
                std.mem.eql(encode.ValType, s.results, results)) return @intCast(i);
        }
        return null;
    }

    pub fn funcIndex(self: *const ModuleResolver, name: []const u8) ?u32 {
        for (self.func_names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return @intCast(i);
        }
        return null;
    }
};

/// A function's stack frame carved from the shared descending stack: which global
/// holds the pointer, the local that saves the caller's value, and the frame size.
const FrameCtx = struct { sp_global: u32, saved_sp_local: u32, size: u32 };

/// Restore the caller's stack pointer, emitted before every return of a framed
/// function. A no-op when the function has no frame.
fn emitEpilogue(code: *std.ArrayList(u8), allocator: std.mem.Allocator, frame: ?FrameCtx) Error!void {
    const fr = frame orelse return;
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(fr.saved_sp_local)));
    try code.append(allocator, encode.GlobalOp.global_set);
    try code.append(allocator, @as(u8, @intCast(fr.sp_global)));
}

/// Compile a single IR function to Wasm bytecode. `resolver` supplies type indices
/// for `call_indirect`. Pass null when compiling standalone (indirect calls then
/// return error.Unsupported).
pub fn selectFunction(allocator: std.mem.Allocator, func: *const Function, resolver: ?*const ModuleResolver) Error!Compiled {
    // f16 is emulated in software: wasm has no f16 value type nor f16 arithmetic, so an f16
    // SSA value is held as its f32 widening in an f32 local and every memory/round boundary
    // converts in software (see `emitHalfExtend` / `emitHalfTruncate` / `emitRoundToHalf`).
    // This mirrors the riscv64/aarch64 held-as-f32 model, so no f16 rejection gate remains.
    // Only SCALAR f16 is handled; f16 nested in a vector/aggregate would fall through to the
    // raw path and miscompile the half lanes, so reject that composite case cleanly.
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    var code = std.ArrayList(u8).empty;
    errdefer code.deinit(allocator);

    var leb_buf: [10]u8 = undefined;
    try emitFunction(func, &code, &leb_buf, allocator, resolver);
    return .{ .code = try code.toOwnedSlice(allocator) };
}

/// The Wasm value type occupied by each scalar leaf of `ty`. Wasm has no pointer
/// type, so pointers (and any unmapped scalar) become an i32 address.
fn scalarValtype(types: *const ir.types.TypeTable, ty: ir.types.Type) encode.ValType {
    return encode.irTypeToWasm(types, ty) orelse .i32;
}

/// How many contiguous wasm locals `ty` occupies: one per scalar, or one per field
/// of a struct or vector (aggregates are scalarized into consecutive locals).
fn leafCount(types: *const ir.types.TypeTable, ty: ir.types.Type) u32 {
    return switch (types.type_kind(ty)) {
        .@"struct" => |fields| @intCast(fields.len),
        .vector => |v| @intCast(v.len),
        else => 1,
    };
}

/// The storage size in bytes of a type, for laying out alloca slots in memory.
fn typeSize(types: *const ir.types.TypeTable, ty: ir.types.Type) u32 {
    return switch (types.type_kind(ty)) {
        .bool => 1,
        .int => |i| (@as(u32, i.bits) + 7) / 8,
        .ptr => 4,
        // An f16 in memory is a 2-byte IEEE half (loaded/stored via load16_u/store16); the
        // f32 widening lives only in the local, never in memory. Matches riscv64's typeSize.
        .float => |f| switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
        },
        .array => |a| @as(u32, @intCast(a.len)) * typeSize(types, a.elem),
        .vector => |v| @as(u32, v.len) * typeSize(types, v.elem),
        else => 8,
    };
}

/// The natural alignment of a type's storage.
fn typeAlign(types: *const ir.types.TypeTable, ty: ir.types.Type) u32 {
    const sz = switch (types.type_kind(ty)) {
        .array => |a| typeSize(types, a.elem),
        .vector => |v| typeSize(types, v.elem),
        else => typeSize(types, ty),
    };
    return if (sz <= 1) 1 else if (sz <= 2) 2 else if (sz <= 4) 4 else 8;
}

/// How a scalar type participates in a `convert`: whether it is a float, a double,
/// a 64-bit width, and (for ints) unsigned. Bools count as unsigned i32.
const ConvClass = struct { float: bool, dbl: bool, bits64: bool, unsigned: bool };

fn convClass(types: *const ir.types.TypeTable, ty: ir.types.Type) ?ConvClass {
    return switch (types.type_kind(ty)) {
        .int => |i| .{ .float = false, .dbl = false, .bits64 = i.bits == 64, .unsigned = i.signedness == .unsigned },
        .bool => .{ .float = false, .dbl = false, .bits64 = false, .unsigned = true },
        .float => |f| .{ .float = true, .dbl = f == .f64, .bits64 = f == .f64, .unsigned = false },
        else => null,
    };
}

/// The single valtype a value's locals share, or null if it has no wasm
/// representation or is a non-uniform aggregate (which isel cannot lower).
fn valueValtype(types: *const ir.types.TypeTable, ty: ir.types.Type) ?encode.ValType {
    return switch (types.type_kind(ty)) {
        .@"struct" => |fields| {
            if (fields.len == 0) return null;
            const first = encode.irTypeToWasm(types, fields[0]) orelse return null;
            for (fields[1..]) |f| {
                if ((encode.irTypeToWasm(types, f) orelse return null) != first) return null;
            }
            return first;
        },
        .vector => |v| encode.irTypeToWasm(types, v.elem),
        else => scalarValtype(types, ty),
    };
}

/// Whether `ty` is the half-precision float `f16`. Wasm has no f16, so an f16 value is
/// emulated as its f32 widening held in an f32 local, with software convert at every
/// boundary (mirrors the riscv64/aarch64 model).
fn isF16(types: *const ir.types.TypeTable, ty: ir.types.Type) bool {
    return switch (types.type_kind(ty)) {
        .float => |f| f == .f16,
        else => false,
    };
}

/// The wasm locals reserved for the software f16 convert routines when a function uses f16.
/// Three scratch i32 locals and one scratch f32 local are enough for both the extend and
/// truncate sequences (they never run concurrently, so the two sequences reuse `i0`/`i1`).
/// Locals are cheap in wasm, so reserving a fixed handful is simpler than juggling the
/// operand stack alone. Absent (null) in every non-f16 function, so those stay byte-identical.
const HalfScratch = struct { i0: u32, i1: u32, i2: u32, f0: u32 };

/// Append an `i32.const` carrying the raw 32-bit pattern `bits` (as a signed LEB, so a
/// pattern with the top bit set still encodes as the correct i32).
fn emitI32ConstBits(code: *std.ArrayList(u8), allocator: std.mem.Allocator, bits: u32) Error!void {
    var leb: [10]u8 = undefined;
    try code.append(allocator, encode.ConstOp.i32_const);
    const n = encode.encodeS32leb(&leb, @bitCast(bits));
    try code.appendSlice(allocator, leb[0..n]);
}

/// Software EXTEND f16 -> f32 (exact, no rounding). Consumes the raw 16-bit half pattern
/// (an i32, zero-extended, e.g. straight from `i32.load16_u`) from the top of the operand
/// stack and leaves the f32 widening of the same value on the stack. Fabian Giesen's
/// magic-multiply half->float: shift the 15 exponent+mantissa bits into an f32 whose
/// exponent is biased low, multiply by the exact power of two 2^112 (0x77800000) to rebias
/// (renormalizing subnormals for free), patch the inf/NaN exponent, then OR in the sign.
/// This is the byte-for-byte port of riscv64's `emitHalfToFloat`, proven bit-exact.
fn emitHalfExtend(code: *std.ArrayList(u8), allocator: std.mem.Allocator, sc: HalfScratch) Error!void {
    const h = sc.i0; // raw 16-bit half pattern
    const of = sc.f0; // o.f (the low-biased f32 before the exponent patch)
    const ou = sc.i1; // accumulating o.u bits

    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(h)));

    // o.f = reinterpret((h & 0x7fff) << 13) * 2^112. The mask drops the sign, the shift
    // places the 15 bits at f32 bit 13, and the pure-power-of-two multiply is exact in any
    // rounding mode.
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(h)));
    try emitI32ConstBits(code, allocator, 0x7FFF);
    try code.append(allocator, encode.I32Op.bit_and);
    try emitI32ConstBits(code, allocator, 13);
    try code.append(allocator, encode.I32Op.shl);
    try code.append(allocator, encode.F32Op.reinterpret_i32);
    try emitI32ConstBits(code, allocator, 0x77800000);
    try code.append(allocator, encode.F32Op.reinterpret_i32);
    try code.append(allocator, encode.F32Op.mul);
    try code.append(allocator, encode.LocalOp.local_tee); // of = o.f, keep it on the stack
    try code.append(allocator, @as(u8, @intCast(of)));
    try code.append(allocator, encode.I32Op.reinterpret_i32); // o.u = bits(o.f)
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(ou)));

    // inf/NaN: a half with exponent 31 lands at >= 2^16 = 65536.0 (0x47800000) after the
    // multiply, so OR in the f32 all-ones exponent 0x7f800000 whenever o.f >= 65536.0.
    // `f32.ge` yields 0/1, and `* 0x7f800000` selects the exponent mask branchlessly.
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(of)));
    try emitI32ConstBits(code, allocator, 0x47800000);
    try code.append(allocator, encode.F32Op.reinterpret_i32);
    try code.append(allocator, encode.F32Op.ge);
    try emitI32ConstBits(code, allocator, 0x7F800000);
    try code.append(allocator, encode.I32Op.mul);
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(ou)));
    try code.append(allocator, encode.I32Op.bit_or);
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(ou)));

    // sign: bit 15 of the half -> bit 31 of the f32, then reinterpret the assembled bits.
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(h)));
    try emitI32ConstBits(code, allocator, 15);
    try code.append(allocator, encode.I32Op.shr_u);
    try emitI32ConstBits(code, allocator, 31);
    try code.append(allocator, encode.I32Op.shl);
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(ou)));
    try code.append(allocator, encode.I32Op.bit_or);
    try code.append(allocator, encode.F32Op.reinterpret_i32);
}

/// Software TRUNCATE f32 -> f16 with round-to-nearest-EVEN. Consumes the held f32 from the
/// top of the operand stack and leaves the 16-bit half pattern (in the low 16 bits of an
/// i32, sign already merged) on the stack. Branchless port of Fabian Giesen's
/// `float_to_half_fast3_rtne`: it computes the normal, subnormal, and inf/NaN candidates and
/// blends them with `select` on masks derived from the input's exponent range. Handles RNE
/// ties (the mant-odd bias), overflow to inf, gradual underflow into f16 subnormals or
/// signed zero, and NaN (mapped to a quiet NaN). This is the port of riscv64's
/// `emitFloatToHalf`, proven bit-exact.
fn emitHalfTruncate(code: *std.ArrayList(u8), allocator: std.mem.Allocator, sc: HalfScratch) Error!void {
    const inbits = sc.i0; // the f32 bit pattern
    const abs = sc.i1; // |f| bits, kept live for the whole routine
    const out = sc.i2; // the running candidate half pattern

    try code.append(allocator, encode.I32Op.reinterpret_i32); // f32 -> its bits
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(inbits)));

    // abs = inbits & 0x7fffffff (strip the sign).
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(inbits)));
    try emitI32ConstBits(code, allocator, 0x7FFFFFFF);
    try code.append(allocator, encode.I32Op.bit_and);
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(abs)));

    // NORMAL candidate: out = (abs + ((15-127)<<23) + 0xfff + mant_odd) >>_u 13, where
    // mant_odd = (abs >> 13) & 1 is the RNE bias. ((15-127)<<23)+0xfff = 0xC8000FFF. The
    // add wraps mod 2^32 (like the riscv 32-bit low word), and the >>_u 13 realigns.
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(abs)));
    try emitI32ConstBits(code, allocator, 0xC8000FFF);
    try code.append(allocator, encode.I32Op.add);
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(abs)));
    try emitI32ConstBits(code, allocator, 13);
    try code.append(allocator, encode.I32Op.shr_u);
    try emitI32ConstBits(code, allocator, 1);
    try code.append(allocator, encode.I32Op.bit_and);
    try code.append(allocator, encode.I32Op.add);
    try emitI32ConstBits(code, allocator, 13);
    try code.append(allocator, encode.I32Op.shr_u);
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(out)));

    // SUBNORMAL candidate, chosen when abs < (113<<23) = 0x38800000. Adding the magic 0.5
    // (0x3f000000) to |f| as an f32 aligns the 10 mantissa bits at the bottom under RNE;
    // the integer subtract of the bias yields the half. `select(o_sub, out, flag_sub)`.
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(abs)));
    try code.append(allocator, encode.F32Op.reinterpret_i32);
    try emitI32ConstBits(code, allocator, 0x3F000000);
    try code.append(allocator, encode.F32Op.reinterpret_i32);
    try code.append(allocator, encode.F32Op.add);
    try code.append(allocator, encode.I32Op.reinterpret_i32);
    try emitI32ConstBits(code, allocator, 0x3F000000);
    try code.append(allocator, encode.I32Op.sub); // o_sub
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(out)));
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(abs)));
    try emitI32ConstBits(code, allocator, 0x38800000);
    try code.append(allocator, encode.I32Op.lt_u); // flag_sub
    try code.append(allocator, encode.ControlOp.select); // flag_sub ? o_sub : out
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(out)));

    // INF/NaN candidate, chosen when abs >= (143<<23) = f16max = 0x47800000. o_inf =
    // 0x7c00 | (abs > 0x7f800000 ? 0x200 : 0): Inf stays Inf, any NaN becomes a quiet NaN.
    try emitI32ConstBits(code, allocator, 0x7C00);
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(abs)));
    try emitI32ConstBits(code, allocator, 0x7F800000);
    try code.append(allocator, encode.I32Op.gt_u); // is_nan (0/1)
    try emitI32ConstBits(code, allocator, 9);
    try code.append(allocator, encode.I32Op.shl); // 0x200 or 0
    try code.append(allocator, encode.I32Op.bit_or); // o_inf
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(out)));
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(abs)));
    try emitI32ConstBits(code, allocator, 0x47800000);
    try code.append(allocator, encode.I32Op.ge_u); // flag_inf
    try code.append(allocator, encode.ControlOp.select); // flag_inf ? o_inf : out
    try code.append(allocator, encode.LocalOp.local_set);
    try code.append(allocator, @as(u8, @intCast(out)));

    // Mask to 16 bits, then OR in the sign (bit 31 of the input -> bit 15 of the half).
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(out)));
    try emitI32ConstBits(code, allocator, 0xFFFF);
    try code.append(allocator, encode.I32Op.bit_and);
    try code.append(allocator, encode.LocalOp.local_get);
    try code.append(allocator, @as(u8, @intCast(inbits)));
    try emitI32ConstBits(code, allocator, 31);
    try code.append(allocator, encode.I32Op.shr_u);
    try emitI32ConstBits(code, allocator, 15);
    try code.append(allocator, encode.I32Op.shl);
    try code.append(allocator, encode.I32Op.bit_or);
}

/// Round the held f32-widening f16 value on the top of the operand stack to nearest-even
/// half and re-widen it, leaving the rounded f32 on the stack. This is the per-op rounding
/// an f16 arithmetic result (or an f32/f64/int -> f16 convert) needs: truncate to half then
/// extend back, both in software. The truncate leaves the 16-bit half on the stack, which
/// the extend consumes; the two sequences reuse the scratch locals since they run in turn.
fn emitRoundToHalf(code: *std.ArrayList(u8), allocator: std.mem.Allocator, sc: HalfScratch) Error!void {
    try emitHalfTruncate(code, allocator, sc);
    try emitHalfExtend(code, allocator, sc);
}

/// Emit an entire function: locals declaration + all blocks.
fn emitFunction(
    func: *const Function,
    code: *std.ArrayList(u8),
    leb_buf: *[10]u8,
    allocator: std.mem.Allocator,
    resolver: ?*const ModuleResolver,
) Error!void {
    const types = &func.types;
    const val_count = func.valueCount();

    // Assign each IR value a contiguous run of wasm locals. Entry-block params are
    // the wasm function parameters, positional at indices 0..n_params-1. Every other
    // value (non-entry block params and instruction results) becomes a declared
    // local. Wasm requires the locals vector grouped by type, so declared locals are
    // laid out grouped by valtype and the index assignment mirrors that order.
    // Aggregates scalarize into `leafCount` consecutive locals of one valtype.
    const leb_order = [_]encode.ValType{ .i32, .i64, .f32, .f64 };

    var value_local = try func.allocator.alloc(u32, val_count);
    defer func.allocator.free(value_local);
    @memset(value_local, 0xFFFFFFFF);

    var is_entry_param = try func.allocator.alloc(bool, val_count);
    defer func.allocator.free(is_entry_param);
    @memset(is_entry_param, false);

    const entry: Block = @enumFromInt(0);
    var next_local: u32 = 0;
    for (func.blockParams(entry)) |param| {
        is_entry_param[@intFromEnum(param)] = true;
        value_local[@intFromEnum(param)] = next_local;
        next_local += 1;
    }

    // Declared locals, grouped by valtype, in value order within each group.
    for (leb_order) |vt| {
        for (0..val_count) |vi| {
            if (is_entry_param[vi]) continue;
            const ty = func.valueType(@enumFromInt(vi));
            const vvt = valueValtype(types, ty) orelse continue;
            if (vvt != vt) continue;
            value_local[vi] = next_local;
            next_local += leafCount(types, ty);
        }
    }

    // Lay each alloca out at a distinct byte offset within the function's frame.
    const alloca_off = try func.allocator.alloc(u32, val_count);
    defer func.allocator.free(alloca_off);
    @memset(alloca_off, 0);
    var frame_size: u32 = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| {
                    frame_size = std.mem.alignForward(u32, frame_size, typeAlign(types, al.elem));
                    if (func.instResult(inst)) |rv| alloca_off[@intFromEnum(rv)] = frame_size;
                    frame_size += typeSize(types, al.elem);
                },
                else => {},
            }
        }
    }

    // When the function allocates and the linker provided a stack-pointer global,
    // carve the frame from the shared descending stack (allocas then never alias
    // across calls). The frame adds one i32 local to hold the caller's sp. Without a
    // resolver (standalone compile) fall back to static offsets from memory base 0.
    const sp_global: ?u32 = if (frame_size > 0) (if (resolver) |r| r.sp_global else null) else null;
    const frame: ?FrameCtx = if (sp_global) |g| blk: {
        const saved = next_local;
        next_local += 1;
        break :blk .{ .sp_global = g, .saved_sp_local = saved, .size = frame_size };
    } else null;

    // When the function uses f16, reserve the software-convert scratch locals (three i32
    // and one f32) as trailing groups after the saved-sp local. Their indices come last, so
    // they never shift the grouped value-local indices, and a non-f16 function reserves
    // nothing (byte-identical codegen). See `HalfScratch` / `emitHalfExtend` / `emitHalfTruncate`.
    const uses_f16 = ir.function.functionUsesF16(func);
    const half: ?HalfScratch = if (uses_f16) blk: {
        const base = next_local;
        next_local += 4;
        break :blk .{ .i0 = base, .i1 = base + 1, .i2 = base + 2, .f0 = base + 3 };
    } else null;

    // Emit the locals vector: group count, then (count, valtype) per present group.
    // The saved-sp local is appended as its own trailing i32 group so it does not
    // shift the grouped value-local indices.
    var group_counts = [_]u32{0} ** leb_order.len;
    var n_groups: u32 = 0;
    for (leb_order, 0..) |vt, gi| {
        var count: u32 = 0;
        for (0..val_count) |vi| {
            if (is_entry_param[vi]) continue;
            const ty = func.valueType(@enumFromInt(vi));
            const vvt = valueValtype(types, ty) orelse continue;
            if (vvt != vt) continue;
            count += leafCount(types, ty);
        }
        group_counts[gi] = count;
        if (count > 0) n_groups += 1;
    }
    if (frame != null) n_groups += 1;
    // The f16 scratch adds two trailing groups: three i32 locals and one f32 local.
    if (half != null) n_groups += 2;
    {
        const n = encode.encodeU32leb(leb_buf, n_groups);
        try code.appendSlice(allocator, leb_buf[0..n]);
    }
    for (leb_order, 0..) |vt, gi| {
        if (group_counts[gi] == 0) continue;
        const n = encode.encodeU32leb(leb_buf, group_counts[gi]);
        try code.appendSlice(allocator, leb_buf[0..n]);
        try code.append(allocator, vt.toByte());
    }
    if (frame != null) {
        try code.append(allocator, 0x01); // one saved-sp local
        try code.append(allocator, encode.ValType.i32.toByte());
    }
    // The f16 scratch groups, emitted in index order (i0,i1,i2 then f0) so the declaration
    // order matches the indices assigned in `half` above.
    if (half != null) {
        try code.append(allocator, 0x03); // three i32 scratch locals (i0, i1, i2)
        try code.append(allocator, encode.ValType.i32.toByte());
        try code.append(allocator, 0x01); // one f32 scratch local (f0)
        try code.append(allocator, encode.ValType.f32.toByte());
    }

    // Stack-frame prologue: save the caller's sp, then reserve this frame.
    if (frame) |fr| {
        try code.append(allocator, encode.GlobalOp.global_get);
        try code.append(allocator, @as(u8, @intCast(fr.sp_global)));
        try code.append(allocator, encode.LocalOp.local_set);
        try code.append(allocator, @as(u8, @intCast(fr.saved_sp_local)));
        try code.append(allocator, encode.GlobalOp.global_get);
        try code.append(allocator, @as(u8, @intCast(fr.sp_global)));
        try code.append(allocator, encode.ConstOp.i32_const);
        const n = encode.encodeS32leb(leb_buf, @intCast(fr.size));
        try code.appendSlice(allocator, leb_buf[0..n]);
        try code.append(allocator, encode.I32Op.sub);
        try code.append(allocator, encode.GlobalOp.global_set);
        try code.append(allocator, @as(u8, @intCast(fr.sp_global)));
    }

    // A single-block function needs no explicit block wrapper: the function body is
    // already an implicit block, so its instructions and the returned value sit at
    // function scope where the final `end` returns them.
    if (func.blockCount() == 1) {
        const block: Block = @enumFromInt(0);
        for (func.blockInsts(block)) |inst| {
            try emitInst(func, types, value_local, alloca_off, frame, half, inst, code, leb_buf, allocator, resolver);
        }
        try emitEpilogue(code, allocator, frame); // restore sp before the value is returned
        if (func.terminator(block)) |term| {
            switch (term) {
                .ret => |rv| if (rv) |v| {
                    try code.append(allocator, encode.LocalOp.local_get);
                    try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(v)])));
                },
                .jump => return error.Unsupported,
            }
        }
        try code.append(allocator, encode.ControlOp.end);
        return;
    }

    // Multi-block: emit structured Wasm control flow. IR `if` instructions become
    // wasm if/else (selection) or block+loop (when the head carries a cf.continue
    // attribute), and block-parameter phis become edge moves into param locals.
    const visited = try func.allocator.alloc(bool, func.blockCount());
    defer func.allocator.free(visited);
    @memset(visited, false);
    var control = std.ArrayList(usize).empty;
    defer control.deinit(func.allocator);
    var stops = std.ArrayList(usize).empty;
    defer stops.deinit(func.allocator);

    const ctx = EmitCtx{
        .func = func,
        .types = types,
        .value_local = value_local,
        .alloca_off = alloca_off,
        .frame = frame,
        .half = half,
        .code = code,
        .leb_buf = leb_buf,
        .allocator = allocator,
        .resolver = resolver,
        .control = &control,
        .stops = &stops,
        .visited = visited,
    };
    try emitRegion(&ctx, 0);

    // Terminate the implicit function body block.
    try code.append(allocator, encode.ControlOp.end);
}

/// The shared state threaded through the structured control-flow emission.
const EmitCtx = struct {
    func: *const Function,
    types: *const ir.types.TypeTable,
    value_local: []const u32,
    alloca_off: []const u32,
    frame: ?FrameCtx,
    half: ?HalfScratch,
    code: *std.ArrayList(u8),
    leb_buf: *[10]u8,
    allocator: std.mem.Allocator,
    resolver: ?*const ModuleResolver,
    /// IR blocks that an enclosing wasm block/loop targets, innermost last. A `br` to
    /// one of these uses its relative depth.
    control: *std.ArrayList(usize),
    /// Selection merge blocks of enclosing `if`s. A jump to one of these falls through
    /// the wasm if/else rather than branching.
    stops: *std.ArrayList(usize),
    visited: []bool,
};

/// The `if` instruction of a block, if it has one.
fn blockIf(func: *const Function, block: Block) ?ir.function.If {
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .@"if" => |iff| return iff,
            else => {},
        }
    }
    return null;
}

/// Read an integer `cf.<key>` block attribute (a structured-control-flow target the
/// frontend records on a selection or loop head).
fn cfAttrInt(func: *const Function, block: Block, key: []const u8) ?usize {
    var it = func.attributesOf(.{ .block = block });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (std.mem.eql(u8, c.namespace, "cf") and std.mem.eql(u8, c.key, key)) {
                switch (c.value) {
                    .int => |v| return @intCast(v),
                    else => {},
                }
            }
        },
        else => {},
    };
    return null;
}

/// The merge (continuation) block of a block's `if`: the shared target when both arms
/// jump to one block, else the recorded `cf.merge`, else a simple diamond's join.
fn mergeOf(func: *const Function, block: Block) ?usize {
    const iff = blockIf(func, block) orelse return null;
    if (iff.then.target == iff.@"else".target) return @intFromEnum(iff.then.target);
    if (cfAttrInt(func, block, "merge")) |m| return m;
    const tt = func.terminator(iff.then.target) orelse return null;
    const et = func.terminator(iff.@"else".target) orelse return null;
    if (tt == .jump and et == .jump and tt.jump.target == et.jump.target) return @intFromEnum(tt.jump.target);
    return null;
}

/// A block heads a loop when it branches and carries a `cf.continue` attribute.
fn isLoopHeader(func: *const Function, block: Block) bool {
    return blockIf(func, block) != null and cfAttrInt(func, block, "continue") != null;
}

/// A `control` entry standing for an enclosing wasm `if` scope. It occupies a nesting
/// level (so `scopeDepth` counts it) but is never a real branch target: no IR block has
/// this index, so `scopeDepth(real_block)` never matches it.
const IF_SCOPE_SENTINEL: usize = std.math.maxInt(usize);

/// The relative wasm depth of the innermost control scope targeting `block_idx`.
fn scopeDepth(ctx: *const EmitCtx, block_idx: usize) ?u32 {
    var i = ctx.control.items.len;
    while (i > 0) {
        i -= 1;
        if (ctx.control.items[i] == block_idx) return @intCast(ctx.control.items.len - 1 - i);
    }
    return null;
}

fn isStop(ctx: *const EmitCtx, block_idx: usize) bool {
    for (ctx.stops.items) |s| if (s == block_idx) return true;
    return false;
}

/// Move edge `args` into the parameter locals of `target`. All sources are pushed
/// first, then popped into the destinations, so the move is a correct parallel move
/// even when a source aliases a destination (e.g. a loop back-edge that swaps).
fn emitEdgeMoves(ctx: *const EmitCtx, args: ir.function.ValueList, target: Block) Error!void {
    const arg_vals = ctx.func.valueList(args);
    const params = ctx.func.blockParams(target);
    if (arg_vals.len != params.len) return error.Unsupported;
    for (arg_vals) |a| {
        try ctx.code.append(ctx.allocator, encode.LocalOp.local_get);
        try ctx.code.append(ctx.allocator, @as(u8, @intCast(ctx.value_local[@intFromEnum(a)])));
    }
    var k = params.len;
    while (k > 0) {
        k -= 1;
        try ctx.code.append(ctx.allocator, encode.LocalOp.local_set);
        try ctx.code.append(ctx.allocator, @as(u8, @intCast(ctx.value_local[@intFromEnum(params[k])])));
    }
}

/// Take an edge to `target` carrying `args`: move the args, then branch to an
/// enclosing scope, fall through to an enclosing merge, or inline the target region.
fn emitEdge(ctx: *const EmitCtx, target: Block, args: ir.function.ValueList) Error!void {
    try emitEdgeMoves(ctx, args, target);
    const tidx = @intFromEnum(target);
    if (scopeDepth(ctx, tidx)) |depth| {
        try ctx.code.append(ctx.allocator, encode.ControlOp.br);
        try ctx.code.append(ctx.allocator, @as(u8, @intCast(depth)));
    } else if (!isStop(ctx, tidx)) {
        try emitRegion(ctx, tidx);
    }
}

fn emitRet(ctx: *const EmitCtx, v: ?Value) Error!void {
    try emitEpilogue(ctx.code, ctx.allocator, ctx.frame);
    if (v) |val| {
        try ctx.code.append(ctx.allocator, encode.LocalOp.local_get);
        try ctx.code.append(ctx.allocator, @as(u8, @intCast(ctx.value_local[@intFromEnum(val)])));
    }
    try ctx.code.append(ctx.allocator, encode.ControlOp.return_);
}

/// Emit the non-`if` instructions of a block.
fn emitBlockBody(ctx: *const EmitCtx, block: Block) Error!void {
    for (ctx.func.blockInsts(block)) |inst| {
        if (ctx.func.opcode(inst) != .@"if") {
            try emitInst(ctx.func, ctx.types, ctx.value_local, ctx.alloca_off, ctx.frame, ctx.half, inst, ctx.code, ctx.leb_buf, ctx.allocator, ctx.resolver);
        }
    }
}

/// Emit a block and the structured region it heads.
fn emitRegion(ctx: *const EmitCtx, block_idx: usize) Error!void {
    if (ctx.visited[block_idx]) return;
    ctx.visited[block_idx] = true;
    const block: Block = @enumFromInt(block_idx);

    if (isLoopHeader(ctx.func, block)) {
        try emitLoop(ctx, block_idx);
        return;
    }

    try emitBlockBody(ctx, block);

    if (blockIf(ctx.func, block)) |iff| {
        try emitSelection(ctx, block_idx, iff);
    } else if (ctx.func.terminator(block)) |term| {
        switch (term) {
            .jump => |j| try emitEdge(ctx, j.target, j.args),
            .ret => |v| try emitRet(ctx, v),
        }
    } else {
        try emitRet(ctx, null);
    }
}

/// Emit an `if` as wasm if/else. Both arms take their edge to the merge, which is
/// emitted after the if/else closes.
fn emitSelection(ctx: *const EmitCtx, block_idx: usize, iff: ir.function.If) Error!void {
    const merge = mergeOf(ctx.func, @enumFromInt(block_idx)) orelse return error.Unsupported;

    try ctx.code.append(ctx.allocator, encode.LocalOp.local_get);
    try ctx.code.append(ctx.allocator, @as(u8, @intCast(ctx.value_local[@intFromEnum(iff.cond)])));
    try ctx.code.append(ctx.allocator, encode.ControlOp.if_);
    try ctx.code.append(ctx.allocator, encode.BlockType.empty.toByte());

    // The wasm `if` is a real nesting level: push a sentinel onto `control` so that a
    // `br` to an OUTER scope (a break/continue reaching an enclosing loop) counts this
    // `if` in its relative depth. The merge stays on `stops` (a jump to it falls through
    // the if/else); the sentinel is never itself a branch target.
    try ctx.stops.append(ctx.allocator, merge);
    try ctx.control.append(ctx.allocator, IF_SCOPE_SENTINEL);
    try emitEdge(ctx, iff.then.target, iff.then.args);
    try ctx.code.append(ctx.allocator, encode.ControlOp.else_);
    try emitEdge(ctx, iff.@"else".target, iff.@"else".args);
    _ = ctx.control.pop();
    _ = ctx.stops.pop();
    try ctx.code.append(ctx.allocator, encode.ControlOp.end);

    // Emit the merge here only if no enclosing construct owns it (otherwise that
    // construct emits it after its own body, at the correct nesting).
    if (!isStop(ctx, merge) and scopeDepth(ctx, merge) == null) try emitRegion(ctx, merge);
}

/// Emit a loop head as `block { loop { <cond> break-if-done <body> br } }`. The head's
/// condition is re-evaluated each iteration, one arm continues the loop and the other
/// (the merge) exits it.
fn emitLoop(ctx: *const EmitCtx, header_idx: usize) Error!void {
    const header: Block = @enumFromInt(header_idx);
    const iff = blockIf(ctx.func, header).?;
    const exit = mergeOf(ctx.func, header) orelse return error.Unsupported;

    // One arm re-enters the loop body, the other exits to the merge.
    var body_target: Block = undefined;
    var body_args: ir.function.ValueList = undefined;
    var exit_args: ir.function.ValueList = undefined;
    var break_when_true: bool = undefined;
    if (@intFromEnum(iff.@"else".target) == exit) {
        body_target = iff.then.target;
        body_args = iff.then.args;
        exit_args = iff.@"else".args;
        break_when_true = false;
    } else if (@intFromEnum(iff.then.target) == exit) {
        body_target = iff.@"else".target;
        body_args = iff.@"else".args;
        exit_args = iff.then.args;
        break_when_true = true;
    } else return error.Unsupported;

    try ctx.code.append(ctx.allocator, encode.ControlOp.block);
    try ctx.code.append(ctx.allocator, encode.BlockType.empty.toByte());
    try ctx.control.append(ctx.allocator, exit);
    try ctx.code.append(ctx.allocator, encode.ControlOp.loop);
    try ctx.code.append(ctx.allocator, encode.BlockType.empty.toByte());
    try ctx.control.append(ctx.allocator, header_idx);

    try emitBlockBody(ctx, header);

    // Set the exit block's params, then break out when the loop condition is done.
    try emitEdgeMoves(ctx, exit_args, @enumFromInt(exit));
    try ctx.code.append(ctx.allocator, encode.LocalOp.local_get);
    try ctx.code.append(ctx.allocator, @as(u8, @intCast(ctx.value_local[@intFromEnum(iff.cond)])));
    if (!break_when_true) try ctx.code.append(ctx.allocator, encode.I32Op.eqz);
    try ctx.code.append(ctx.allocator, encode.ControlOp.br_if);
    try ctx.code.append(ctx.allocator, @as(u8, @intCast(scopeDepth(ctx, exit).?)));

    // The loop's continue block (increment + back-edge) must be a real branch target, not
    // inlined at its first use: with a `continue` in one arm of an `if` and the fall-through
    // in the other, two paths reach it, and inlining emits it once (in the first path),
    // leaving the second with no increment/back-edge. Wrap the body in an inner `block`
    // labeled with the continue block so every path `br`s to it, then emit its code once
    // after the inner block closes.
    const cont_b = cfAttrInt(ctx.func, header, "continue");
    if (cont_b != null and cont_b.? != @intFromEnum(body_target)) {
        const cb = cont_b.?;
        try ctx.code.append(ctx.allocator, encode.ControlOp.block);
        try ctx.code.append(ctx.allocator, encode.BlockType.empty.toByte());
        try ctx.control.append(ctx.allocator, cb);
        try emitEdge(ctx, body_target, body_args);
        _ = ctx.control.pop();
        try ctx.code.append(ctx.allocator, encode.ControlOp.end); // inner continue block

        // The continue block itself: its increment instructions, then its back-edge.
        ctx.visited[cb] = true;
        try emitBlockBody(ctx, @enumFromInt(cb));
        if (ctx.func.terminator(@enumFromInt(cb))) |term| switch (term) {
            .jump => |j| try emitEdge(ctx, j.target, j.args),
            .ret => |v| try emitRet(ctx, v),
        } else try emitRet(ctx, null);
    } else {
        // No distinct continue block: fall through into the body (back-edge branches home).
        try emitEdge(ctx, body_target, body_args);
    }

    try ctx.code.append(ctx.allocator, encode.ControlOp.end); // loop
    _ = ctx.control.pop();
    try ctx.code.append(ctx.allocator, encode.ControlOp.end); // block
    _ = ctx.control.pop();

    // Emit the exit here only if no enclosing construct owns it.
    if (!isStop(ctx, exit) and scopeDepth(ctx, exit) == null) try emitRegion(ctx, exit);
}

/// Emit a single IR instruction as Wasm bytecode. Control-flow (`if`) is handled at
/// the block level by the structured emitter, never here.
fn emitInst(
    func: *const Function,
    types: *const ir.types.TypeTable,
    value_local: []const u32,
    alloca_off: []const u32,
    frame: ?FrameCtx,
    half: ?HalfScratch,
    inst: Inst,
    code: *std.ArrayList(u8),
    leb_buf: *[10]u8,
    allocator: std.mem.Allocator,
    resolver: ?*const ModuleResolver,
) Error!void {
    const op = func.opcode(inst);
    const result = func.instResult(inst);

    switch (op) {
        .iconst => |val| {
            const vt = encode.irTypeToWasm(types, func.valueType(result.?)).?;
            switch (vt) {
                .i32 => {
                    try code.append(allocator, encode.ConstOp.i32_const);
                    // Take the low 32 bits as the i32 bit pattern: a uint constant like
                    // 0xFFFFFFFF is a valid i32 (-1) that `@intCast` would reject.
                    const n = encode.encodeS32leb(leb_buf, @as(i32, @truncate(val)));
                    try code.appendSlice(allocator, leb_buf[0..n]);
                },
                .i64 => {
                    try code.append(allocator, encode.ConstOp.i64_const);
                    const n = encode.encodeS64leb(leb_buf, val);
                    try code.appendSlice(allocator, leb_buf[0..n]);
                },
                else => return error.Unsupported,
            }
            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .fconst => |fval| {
            const res_ty = func.valueType(result.?);
            const vt = encode.irTypeToWasm(types, res_ty).?;
            switch (vt) {
                .f32 => {
                    try code.append(allocator, encode.ConstOp.f32_const);
                    // wasm encodes float immediates little-endian; write in that order
                    // explicitly so the output is correct on a big-endian host too. An f16
                    // constant is pre-rounded to half (`@as(f16, val)`) then widened back to
                    // f32, so the materialized value already satisfies the held-as-f32 invariant.
                    const f32_val: f32 = if (isF16(types, res_ty))
                        @as(f32, @as(f16, @floatCast(fval)))
                    else
                        @as(f32, @floatCast(fval));
                    var bits: [4]u8 = undefined;
                    std.mem.writeInt(u32, &bits, @bitCast(f32_val), .little);
                    try code.appendSlice(allocator, &bits);
                },
                .f64 => {
                    try code.append(allocator, encode.ConstOp.f64_const);
                    var bits: [8]u8 = undefined;
                    std.mem.writeInt(u64, &bits, @bitCast(fval), .little);
                    try code.appendSlice(allocator, &bits);
                },
                else => return error.Unsupported,
            }
            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .arith => |a| {
            try emitArith(func, types, value_local, half, a.lhs, a.rhs, a.op, result, code);
        },
        .arith_imm => |a| {
            if (a.op == .mulh) return error.Unsupported; // no immediate high-multiply form
            // Materialize imm as a const, then arith.
            const ty = if (result) |r| func.valueType(r) else func.valueType(a.lhs);
            const lhs_local = value_local[@intFromEnum(a.lhs)];

            // Push lhs
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(lhs_local)));

            // Push immediate
            try emitIconst(func, code, leb_buf, a.imm);

            // Operate
            switch (types.type_kind(ty)) {
                .int => |info| {
                    const op_byte = switch (info.bits) {
                        64 => arithI64(a.op, info.signedness),
                        else => arithI32(a.op, info.signedness),
                    };
                    try code.append(allocator, op_byte);
                },
                // Bools are i32 (0/1), same as the `.arith` path.
                .bool => try code.append(allocator, arithI32(a.op, .unsigned)),
                .float => |f| {
                    const op_byte = switch (f) {
                        .f32 => switch (a.op) {
                            .add => encode.F32Op.add,
                            .sub => encode.F32Op.sub,
                            .mul => encode.F32Op.mul,
                            .div => encode.F32Op.div,
                            else => return error.Unsupported, // no wasm float rem/bitwise/shift
                        },
                        .f64 => switch (a.op) {
                            .add => encode.F64Op.add,
                            .sub => encode.F64Op.sub,
                            .mul => encode.F64Op.mul,
                            .div => encode.F64Op.div,
                            else => return error.Unsupported,
                        },
                        // f16 has no native wasm arith op; wasm f16 lowering is a
                        // later task, not this IR-only change.
                        .f16 => return error.Unsupported,
                    };
                    try code.append(allocator, op_byte);
                },
                else => return error.Unsupported,
            }

            // Store result
            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .icmp => |c| {
            // The comparison op is chosen from the operand type (the result is always
            // bool, so keying off it would always pick eq).
            const ty = func.valueType(c.lhs);
            const lhs_local = value_local[@intFromEnum(c.lhs)];
            const rhs_local = value_local[@intFromEnum(c.rhs)];

            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(lhs_local)));
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(rhs_local)));

            switch (types.type_kind(ty)) {
                .int => |info| {
                    const op_byte = switch (info.bits) {
                        64 => cmpI64(c.op, info.signedness),
                        else => cmpI32(c.op, info.signedness),
                    };
                    try code.append(allocator, op_byte);
                },
                .bool => {
                    // Bools are i32 (0/1), compare with the actual op, not always eq.
                    try code.append(allocator, cmpI32(c.op, .unsigned));
                },
                .float => |f| {
                    // The IR uses `icmp` for float comparisons too (keyed off operands).
                    try code.append(allocator, cmpFloat(c.op, f == .f64));
                },
                else => return error.Unsupported,
            }

            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .select => |s| {
            // Wasm `select` is [v1, v2, cond] -> cond ? v1 : v2, so push `then` first
            // (v1) and `else` second (v2), matching the IR `cond ? then : else`.
            const else_local = value_local[@intFromEnum(s.@"else")];
            const then_local = value_local[@intFromEnum(s.then)];
            const cond_local = value_local[@intFromEnum(s.cond)];
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(then_local)));
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(else_local)));
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(cond_local)));
            try code.append(allocator, encode.ControlOp.select);
            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .convert => |cv| {
            const src_ty = func.valueType(cv.value);
            const dst_ty = if (result) |r| func.valueType(r) else src_ty;
            const s = convClass(types, src_ty) orelse return error.Unsupported;
            const d = convClass(types, dst_ty) orelse return error.Unsupported;
            const src_local = value_local[@intFromEnum(cv.value)];

            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(src_local)));

            if (!s.float and !d.float) {
                // Integer resize: widen (extend), narrow (wrap), or same-width no-op.
                if (!s.bits64 and d.bits64) {
                    try code.append(allocator, if (s.unsigned) encode.I64Op.extend_i32_u else encode.I64Op.extend_i32_s);
                } else if (s.bits64 and !d.bits64) {
                    try code.append(allocator, encode.I32Op.wrap_i64);
                }
            } else if (!s.float and d.float) {
                // int -> float, using the source signedness.
                try code.append(allocator, if (d.dbl)
                    (if (s.bits64) (if (s.unsigned) encode.F64Op.convert_i64_u else encode.F64Op.convert_i64_s) else (if (s.unsigned) encode.F64Op.convert_i32_u else encode.F64Op.convert_i32_s))
                else
                    (if (s.bits64) (if (s.unsigned) encode.F32Op.convert_i64_u else encode.F32Op.convert_i64_s) else (if (s.unsigned) encode.F32Op.convert_i32_u else encode.F32Op.convert_i32_s)));
            } else if (s.float and !d.float) {
                // float -> int, saturating truncation chosen by the destination signedness.
                try code.append(allocator, if (d.bits64)
                    (if (s.dbl) (if (d.unsigned) encode.I64Op.trunc_f64_u else encode.I64Op.trunc_f64_s) else (if (d.unsigned) encode.I64Op.trunc_f32_u else encode.I64Op.trunc_f32_s))
                else
                    (if (s.dbl) (if (d.unsigned) encode.I32Op.trunc_f64_u else encode.I32Op.trunc_f64_s) else (if (d.unsigned) encode.I32Op.trunc_f32_u else encode.I32Op.trunc_f32_s)));
            } else {
                // float -> float: promote, demote, or same no-op. An f16 is held as its f32
                // widening, so f16<->f32 is a no-op here (the round below handles f32->f16),
                // f16->f64 is the plain promote (the exact half widens), and f64->f16 first
                // demotes to f32 then rounds below.
                if (!s.dbl and d.dbl) {
                    try code.append(allocator, encode.F64Op.promote_f32);
                } else if (s.dbl and !d.dbl) {
                    try code.append(allocator, encode.F32Op.demote_f64);
                }
            }

            // Any conversion whose destination is f16 rounds the produced f32 to nearest-even
            // half (int->f16, f32->f16, f64->f16), keeping the held-as-f32 invariant. A
            // destination of f32/f64/int needs no rounding. f16->int already truncated above.
            if (isF16(types, dst_ty)) {
                const sc = half orelse return error.Unsupported;
                try emitRoundToHalf(code, allocator, sc);
            }

            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .unary => |u| {
            const ty = if (result) |r| func.valueType(r) else func.valueType(u.value);
            const kind = types.type_kind(ty);
            const src_local = value_local[@intFromEnum(u.value)];

            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(src_local)));

            switch (u.op) {
                .reinterpret => switch (kind) {
                    .int => |info| switch (info.bits) {
                        32 => try code.append(allocator, encode.I32Op.reinterpret_i32),
                        64 => try code.append(allocator, encode.I64Op.reinterpret_i64),
                        else => return error.Unsupported,
                    },
                    // f16 has no native wasm reinterpret op; wasm f16 lowering
                    // is a later task, not this IR-only change.
                    .float => |f| switch (f) {
                        .f32 => try code.append(allocator, encode.F32Op.reinterpret_i32),
                        .f64 => try code.append(allocator, encode.F64Op.reinterpret_i64),
                        .f16 => return error.Unsupported,
                    },
                    else => return error.Unsupported,
                },
                .sqrt, .ceil, .floor, .trunc, .nearest => {
                    try code.append(allocator, switch (kind) {
                        .float => |f| switch (u.op) {
                            .sqrt => switch (f) {
                                .f32 => encode.F32Op.sqrt,
                                .f64 => encode.F64Op.sqrt,
                                .f16 => return error.Unsupported,
                            },
                            .ceil => switch (f) {
                                .f32 => encode.F32Op.ceil,
                                .f64 => encode.F64Op.ceil,
                                .f16 => return error.Unsupported,
                            },
                            .floor => switch (f) {
                                .f32 => encode.F32Op.floor,
                                .f64 => encode.F64Op.floor,
                                .f16 => return error.Unsupported,
                            },
                            .trunc => switch (f) {
                                .f32 => encode.F32Op.trunc,
                                .f64 => encode.F64Op.trunc,
                                .f16 => return error.Unsupported,
                            },
                            .nearest => switch (f) {
                                .f32 => encode.F32Op.nearest,
                                .f64 => encode.F64Op.nearest,
                                .f16 => return error.Unsupported,
                            },
                            .reinterpret => unreachable,
                        },
                        else => return error.Unsupported,
                    });
                },
            }

            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .load => |ld| {
            const ptr_local = value_local[@intFromEnum(ld.ptr)];
            const ty = if (result) |r| func.valueType(r) else func.valueType(ld.ptr);
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(ptr_local)));
            try code.append(allocator, try encode.irLoadOp(types, ty));
            try code.append(allocator, 0x00); // align
            try code.append(allocator, 0x00); // offset
            // f16 loads read the raw 2-byte half (via load16_u above); widen those bits to
            // the held f32 in software before the value lands in its f32 local.
            if (isF16(types, ty)) {
                const sc = half orelse return error.Unsupported;
                try emitHalfExtend(code, allocator, sc);
            }
            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .store => |st| {
            const val_local = value_local[@intFromEnum(st.value)];
            const ptr_local = value_local[@intFromEnum(st.ptr)];
            const val_ty = func.valueType(st.value);
            // Wasm store pops the value (top of stack) then the address, so push the
            // address first.
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(ptr_local)));
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(val_local)));
            // f16 stores truncate the held f32 (now on the stack, above the address) to the
            // 2-byte half in software, then store16 those low 16 bits. The address pushed
            // first stays untouched underneath the truncate's stack work.
            if (isF16(types, val_ty)) {
                const sc = half orelse return error.Unsupported;
                try emitHalfTruncate(code, allocator, sc);
            }
            try code.append(allocator, try encode.irStoreOp(types, val_ty));
            try code.append(allocator, 0x00); // align
            try code.append(allocator, 0x00); // offset
        },
        .prefetch => {}, // a hint, Wasm has no prefetch, dropped
        // dot is aarch64+dotprod-only in practice; Wasm has no lowering for it.
        .dot => return error.Unsupported,
        // matmul is et-soc-only (a later task); Wasm has no lowering for it.
        .matmul => return error.Unsupported,
        .call => |c| {
            // Resolve the callee by NAME to its module function index. `c.symbol` is a
            // per-function interned id whose ordering need not match the module layout,
            // so using it as the index directly would call the wrong function.
            const res = resolver orelse return error.Unsupported;
            const callee = res.funcIndex(func.symbolName(c.symbol)) orelse return error.Unsupported;
            for (func.valueList(c.args)) |arg| {
                try code.append(allocator, encode.LocalOp.local_get);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(arg)])));
            }
            try code.append(allocator, encode.ControlOp.call);
            const n = encode.encodeU32leb(leb_buf, callee);
            try code.appendSlice(allocator, leb_buf[0..n]);
            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .call_indirect => |ci| {
            // Push the args, then the table index (`ci.target`), then call_indirect
            // naming the callee signature's type index and table 0. The linker owns
            // the type section, so it supplies the resolver.
            const res = resolver orelse return error.Unsupported;
            const args = func.valueList(ci.args);

            for (args) |arg| {
                try code.append(allocator, encode.LocalOp.local_get);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(arg)])));
            }
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(ci.target)])));

            var pbuf: [16]encode.ValType = undefined;
            if (args.len > pbuf.len) return error.Unsupported;
            for (args, 0..) |arg, i| {
                pbuf[i] = encode.irTypeToWasm(types, func.valueType(arg)) orelse return error.Unsupported;
            }
            var rbuf: [1]encode.ValType = undefined;
            var n_res: usize = 0;
            if (result) |rv| {
                rbuf[0] = encode.irTypeToWasm(types, func.valueType(rv)) orelse return error.Unsupported;
                n_res = 1;
            }
            const type_idx = res.indexOf(pbuf[0..args.len], rbuf[0..n_res]) orelse return error.Unsupported;

            try code.append(allocator, encode.ControlOp.call_indirect);
            const nt = encode.encodeU32leb(leb_buf, type_idx);
            try code.appendSlice(allocator, leb_buf[0..nt]);
            const nb = encode.encodeU32leb(leb_buf, 0); // table 0
            try code.appendSlice(allocator, leb_buf[0..nb]);

            if (result) |rv| {
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
            }
        },
        .alloca => {
            // The address of this alloca's slot: sp + offset when the function has a
            // stack frame, else a static offset from memory base 0.
            const rv = result orelse return error.Unsupported;
            const off = alloca_off[@intFromEnum(rv)];
            if (frame) |fr| {
                try code.append(allocator, encode.GlobalOp.global_get);
                try code.append(allocator, @as(u8, @intCast(fr.sp_global)));
                if (off != 0) {
                    try code.append(allocator, encode.ConstOp.i32_const);
                    const n = encode.encodeS32leb(leb_buf, @intCast(off));
                    try code.appendSlice(allocator, leb_buf[0..n]);
                    try code.append(allocator, encode.I32Op.add);
                }
            } else {
                try code.append(allocator, encode.ConstOp.i32_const);
                const n = encode.encodeS32leb(leb_buf, @intCast(off));
                try code.appendSlice(allocator, leb_buf[0..n]);
            }
            try code.append(allocator, encode.LocalOp.local_set);
            try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
        },
        .global_addr => {
            // Named globals need a data/global layout the wasm target does not lay out
            // yet. Fail loudly rather than returning a bogus address.
            return error.Unsupported;
        },
        .struct_new => |sn| {
            // The aggregate occupies a contiguous run of locals. Copy each field
            // value into its slot.
            const rv = result orelse return error.Unsupported;
            const base = value_local[@intFromEnum(rv)];
            if (base == 0xFFFFFFFF) return error.Unsupported;
            for (func.valueList(sn.fields), 0..) |field, i| {
                try code.append(allocator, encode.LocalOp.local_get);
                try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(field)])));
                try code.append(allocator, encode.LocalOp.local_set);
                try code.append(allocator, @as(u8, @intCast(base + i)));
            }
        },
        .extract => |ex| {
            // Read the field's slot out of the aggregate's contiguous local run.
            const rv = result orelse return error.Unsupported;
            const base = value_local[@intFromEnum(ex.aggregate)];
            if (base == 0xFFFFFFFF) return error.Unsupported;
            try code.append(allocator, encode.LocalOp.local_get);
            try code.append(allocator, @as(u8, @intCast(base + ex.index)));
            try code.append(allocator, encode.LocalOp.local_set);
            try code.append(allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
        },
        .@"if" => {
            // The structured emitter handles `if` at the block level and never emits
            // it as an instruction, so reaching here is a bug.
            return error.Unsupported;
        },
    }
}

/// Emit an integer constant as Wasm i32.const.
fn emitIconst(
    func: *const Function,
    code: *std.ArrayList(u8),
    leb_buf: *[10]u8,
    val: i64,
) Error!void {
    try code.append(func.allocator, encode.ConstOp.i32_const);
    // Low 32 bits as the i32 bit pattern (see the .iconst case): a full-range uint immediate
    // is a valid i32 that a range-checked `@intCast` would reject.
    const n = encode.encodeS32leb(leb_buf, @as(i32, @truncate(val)));
    try code.appendSlice(func.allocator, leb_buf[0..n]);
}

// Helper functions for opcode selection

fn arithI32(op: BinOp, signed: std.builtin.Signedness) u8 {
    return switch (op) {
        .add => encode.I32Op.add,
        .sub => encode.I32Op.sub,
        .mul => encode.I32Op.mul,
        .div => if (signed == .signed) encode.I32Op.div_s else encode.I32Op.div_u,
        .rem => if (signed == .signed) encode.I32Op.rem_s else encode.I32Op.rem_u,
        .bit_and => encode.I32Op.bit_and,
        .bit_or => encode.I32Op.bit_or,
        .bit_xor => encode.I32Op.bit_xor,
        .shl => encode.I32Op.shl,
        .shr => if (signed == .signed) encode.I32Op.shr_s else encode.I32Op.shr_u,
        // wasm has no high-multiply; `expandMulh` rewrites `mulh` into plain multiplies/shifts
        // before this backend's isel, so it never reaches here.
        .mulh => unreachable,
    };
}

fn arithI64(op: BinOp, signed: std.builtin.Signedness) u8 {
    return switch (op) {
        .add => encode.I64Op.add,
        .sub => encode.I64Op.sub,
        .mul => encode.I64Op.mul,
        .div => if (signed == .signed) encode.I64Op.div_s else encode.I64Op.div_u,
        .rem => if (signed == .signed) encode.I64Op.rem_s else encode.I64Op.rem_u,
        .bit_and => encode.I64Op.bit_and,
        .bit_or => encode.I64Op.bit_or,
        .bit_xor => encode.I64Op.bit_xor,
        .shl => encode.I64Op.shl,
        .shr => if (signed == .signed) encode.I64Op.shr_s else encode.I64Op.shr_u,
        // wasm has no high-multiply; `expandMulh` rewrites `mulh` into plain multiplies/shifts
        // before this backend's isel, so it never reaches here.
        .mulh => unreachable,
    };
}

fn cmpI32(op: CmpOp, signed: std.builtin.Signedness) u8 {
    return switch (op) {
        .eq => encode.I32Op.eq,
        .ne => encode.I32Op.ne,
        .lt => if (signed == .signed) encode.I32Op.lt_s else encode.I32Op.lt_u,
        .le => if (signed == .signed) encode.I32Op.le_s else encode.I32Op.le_u,
        .gt => if (signed == .signed) encode.I32Op.gt_s else encode.I32Op.gt_u,
        .ge => if (signed == .signed) encode.I32Op.ge_s else encode.I32Op.ge_u,
    };
}

fn cmpI64(op: CmpOp, signed: std.builtin.Signedness) u8 {
    return switch (op) {
        .eq => encode.I64Op.eq,
        .ne => encode.I64Op.ne,
        .lt => if (signed == .signed) encode.I64Op.lt_s else encode.I64Op.lt_u,
        .le => if (signed == .signed) encode.I64Op.le_s else encode.I64Op.le_u,
        .gt => if (signed == .signed) encode.I64Op.gt_s else encode.I64Op.gt_u,
        .ge => if (signed == .signed) encode.I64Op.ge_s else encode.I64Op.ge_u,
    };
}

fn cmpFloat(op: CmpOp, dbl: bool) u8 {
    return if (dbl) switch (op) {
        .eq => encode.F64Op.eq,
        .ne => encode.F64Op.ne,
        .lt => encode.F64Op.lt,
        .le => encode.F64Op.le,
        .gt => encode.F64Op.gt,
        .ge => encode.F64Op.ge,
    } else switch (op) {
        .eq => encode.F32Op.eq,
        .ne => encode.F32Op.ne,
        .lt => encode.F32Op.lt,
        .le => encode.F32Op.le,
        .gt => encode.F32Op.gt,
        .ge => encode.F32Op.ge,
    };
}

fn emitArith(
    func: *const Function,
    types: *const ir.types.TypeTable,
    value_local: []const u32,
    half: ?HalfScratch,
    lhs: Value,
    rhs: Value,
    op: BinOp,
    result: ?Value,
    code: *std.ArrayList(u8),
) Error!void {
    // wasm has no high-multiply and this backend runs no `expandMulh`; reject cleanly rather than
    // reach the `unreachable` in arithI32/arithI64. Only the 64-bit magic-divide lowering emits it.
    if (op == .mulh) return error.Unsupported;
    const ty = if (result) |r| func.valueType(r) else func.valueType(lhs);
    const lhs_local = value_local[@intFromEnum(lhs)];
    const rhs_local = value_local[@intFromEnum(rhs)];

    try code.append(func.allocator, encode.LocalOp.local_get);
    try code.append(func.allocator, @as(u8, @intCast(lhs_local)));
    try code.append(func.allocator, encode.LocalOp.local_get);
    try code.append(func.allocator, @as(u8, @intCast(rhs_local)));

    switch (types.type_kind(ty)) {
        .int => |info| {
            const op_byte = switch (info.bits) {
                64 => arithI64(op, info.signedness),
                else => arithI32(op, info.signedness),
            };
            try code.append(func.allocator, op_byte);
        },
        // Bools are i32 (0/1). `&&`/`||` lower to bit_and/bit_or on bool operands.
        .bool => try code.append(func.allocator, arithI32(op, .unsigned)),
        .float => |f| {
            const op_byte = switch (f) {
                .f32 => switch (op) {
                    .add => encode.F32Op.add,
                    .sub => encode.F32Op.sub,
                    .mul => encode.F32Op.mul,
                    .div => encode.F32Op.div,
                    else => return error.Unsupported, // no wasm float rem/bitwise/shift
                },
                .f64 => switch (op) {
                    .add => encode.F64Op.add,
                    .sub => encode.F64Op.sub,
                    .mul => encode.F64Op.mul,
                    .div => encode.F64Op.div,
                    else => return error.Unsupported,
                },
                // f16 is emulated as its f32 widening: the operands are already the f32
                // widenings held in f32 locals, so the op runs in f32 and the result is
                // rounded back to half below (per-op rounding = correct IEEE f16 semantics).
                .f16 => switch (op) {
                    .add => encode.F32Op.add,
                    .sub => encode.F32Op.sub,
                    .mul => encode.F32Op.mul,
                    .div => encode.F32Op.div,
                    else => return error.Unsupported,
                },
            };
            try code.append(func.allocator, op_byte);
            // Round the f32 result back to nearest-even half, preserving the held-as-f32
            // widening invariant. The truncate+extend consumes and re-produces the stack top.
            if (f == .f16) {
                const sc = half orelse return error.Unsupported;
                try emitRoundToHalf(code, func.allocator, sc);
            }
        },
        else => return error.Unsupported,
    }

    if (result) |rv| {
        try code.append(func.allocator, encode.LocalOp.local_set);
        try code.append(func.allocator, @as(u8, @intCast(value_local[@intFromEnum(rv)])));
    }
}

test "codegen+disasm round-trip: integer add" {
    // Lower an IR add to Wasm bytecode and assert the disassembled function body: the locals
    // header plus the stack-machine expression. No execution, so it runs on any host.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, i32_t);
    const y = try func.appendBlockParam(e, i32_t);
    const s = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(e, .{ .ret = s });

    var compiled = try selectFunction(a, &func, null);
    defer compiled.deinit(a);
    const text = try disasm.formatBody(a, compiled.code);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\locals: 1 x i32
        \\0000: local.get 0
        \\0002: local.get 1
        \\0004: i32.add
        \\0005: local.set 2
        \\0007: local.get 2
        \\0009: end
        \\
    , text);
}

test "an f16 function lowers: held as f32, with reserved software-convert scratch locals" {
    // f16 is emulated as its f32 widening (no wasm f16 type), so an f16 add now lowers rather
    // than being rejected. The function must declare the reserved f16 scratch locals (three
    // i32 and one f32) and round the f32 add result back to half.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, f16_t);
    const y = try func.appendBlockParam(e, f16_t);
    const s = try func.appendInst(e, f16_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(e, .{ .ret = s });

    var compiled = try selectFunction(a, &func, null);
    defer compiled.deinit(a);
    const text = try disasm.formatBody(a, compiled.code);
    defer a.free(text);
    // The result f32 local plus the four reserved f16 scratch locals (3 i32, 1 f32); the
    // body starts by adding the two f32-widening params, then rounds to half.
    try std.testing.expect(std.mem.indexOf(u8, text, "1 x f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "3 x i32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "f32.add") != null);
}
