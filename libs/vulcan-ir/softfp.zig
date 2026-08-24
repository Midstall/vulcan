//! Soft-float lowering for binary128 (`f128`). No CPU backend has a hardware binary128
//! unit, so f128 arithmetic, compares, conversions, and sqrt lower to the standard
//! libgcc/compiler-rt soft-fp symbols (`__addtf3`, `__eqtf2`, `__extenddftf2`, ...). Data
//! movement (constant, register move, memory load and store) stays native 128-bit xmm
//! traffic and is NOT touched here.
//!
//! The lowering is target-independent: it only rewrites IR, and each backend's own call ABI
//! then places the f128 arguments (in an xmm register by value on System V, per the psABI)
//! and reads the result. So one pass serves every machine backend. The in-memory JIT linker
//! rejects the undefined soft-fp symbols with `error.UndefinedSymbol`, which is the honest
//! JIT answer until vulcan grows its own binary128 soft-fp; the object path emits a real
//! undefined symbol that the final link resolves from the system libgcc/compiler-rt.
//!
//! The C backend keeps native `_Float128` source text (the host compiler supplies the
//! soft-fp), so it does NOT run this pass. Only the machine backends do.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Value = function.Value;
const Inst = function.Inst;
const Block = function.Block;
const Opcode = function.Opcode;

/// Every soft-fp symbol this pass can emit, as static string literals. A backend that clones
/// the function before compiling frees the clone's symbol storage on return, so it must
/// re-point each borrowed relocation name to a longer-lived string; a soft-fp name the pass
/// added to the clone has no match in the original function, so the backend maps it back to
/// one of these static literals instead (see `staticName`).
pub const symbols = [_][]const u8{
    "__addtf3",      "__subtf3",     "__multf3",      "__divtf3",
    "__eqtf2",       "__netf2",      "__lttf2",       "__letf2",
    "__gttf2",       "__getf2",      "__extendsftf2", "__extenddftf2",
    "__extendhftf2", "__trunctfsf2", "__trunctfdf2",  "__trunctfhf2",
    "__floatsitf",   "__floatditf",  "__floatunsitf", "__floatunditf",
    "__fixtfsi",     "__fixtfdi",    "__fixunstfsi",  "__fixunstfdi",
    "__sqrttf2",
};

/// The static soft-fp literal whose text equals `name`, or null if `name` is not one of this
/// pass's soft-fp symbols. The returned slice has static lifetime.
pub fn staticName(name: []const u8) ?[]const u8 {
    for (symbols) |s| {
        if (std.mem.eql(u8, s, name)) return s;
    }
    return null;
}

/// Whether `v` is a binary128 scalar float.
fn isF128(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f128,
        else => false,
    };
}

/// The float width of `v`, or null if it is not a float.
fn floatKind(func: *const Function, v: Value) ?types.FloatKind {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f,
        else => null,
    };
}

/// The (bit width, signedness) of an integer value, or null if it is not an integer.
const IntInfo = struct { bits: u16, signed: bool };
fn intInfo(func: *const Function, v: Value) ?IntInfo {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| .{ .bits = i.bits, .signed = i.signedness == .signed },
        else => null,
    };
}

/// The soft-fp name for a binary arithmetic op on f128, or null if the op has no soft-fp
/// form (only add/sub/mul/div do; a real f128 never carries the others).
fn arithSym(op: function.BinOp) ?[]const u8 {
    return switch (op) {
        .add => "__addtf3",
        .sub => "__subtf3",
        .mul => "__multf3",
        .div => "__divtf3",
        else => null,
    };
}

/// The soft-fp compare name for `op`. Each returns an integer status whose sign against zero,
/// under the SAME relation, reproduces the ordered compare (including the NaN-unordered
/// result): `__lttf2(a,b) < 0` iff `a < b`, and so on.
fn cmpSym(op: function.CmpOp) []const u8 {
    return switch (op) {
        .eq => "__eqtf2",
        .ne => "__netf2",
        .lt => "__lttf2",
        .le => "__letf2",
        .gt => "__gttf2",
        .ge => "__getf2",
    };
}

/// Run the lowering in place over `func`. Returns whether anything changed. Idempotent: a
/// second run finds no f128 op left and changes nothing.
pub fn lower(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    const i32_ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f128_ty = try func.types.intern(.{ .float = .f128 });

    var changed = false;
    var order: std.ArrayList(Inst) = .empty;
    defer order.deinit(allocator);

    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        const old = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(old);
        order.clearRetainingCapacity();

        for (old) |inst| {
            const did = try lowerInst(allocator, func, inst, &order, i32_ty, f128_ty);
            if (did) changed = true else try order.append(allocator, inst);
        }
        if (changed) try func.setBlockInsts(block, order.items);
    }
    return changed;
}

/// Try to lower one instruction. On success it appends the replacement instructions to
/// `order` (in program order), redirects the original result to the replacement via
/// `replaceAllUses`, and returns true. On a non-f128 instruction it returns false and appends
/// nothing (the caller keeps the original). The original f128 instruction is simply dropped
/// from the rebuilt block; its now-unused result value is left orphaned, like any dead IR.
fn lowerInst(
    allocator: std.mem.Allocator,
    func: *Function,
    inst: Inst,
    order: *std.ArrayList(Inst),
    i32_ty: types.Type,
    f128_ty: types.Type,
) std.mem.Allocator.Error!bool {
    const result = func.instResult(inst) orelse return false;
    switch (func.opcode(inst)) {
        .arith => |a| {
            if (!isF128(func, result)) return false;
            const sym = arithSym(a.op) orelse return false;
            const call = try emitCall(allocator, func, order, f128_ty, sym, &.{ a.lhs, a.rhs });
            func.replaceAllUses(result, call);
            return true;
        },
        .icmp => |c| {
            if (!isF128(func, c.lhs)) return false;
            // status = __cmptf2(a, b); result = (status <op> 0), a SIGNED i32 compare.
            const status = try emitCall(allocator, func, order, i32_ty, cmpSym(c.op), &.{ c.lhs, c.rhs });
            const zero = try emitInst(allocator, func, order, i32_ty, .{ .iconst = 0 });
            const cmp = try emitInst(allocator, func, order, func.valueType(result), .{ .icmp = .{ .op = c.op, .lhs = status, .rhs = zero } });
            func.replaceAllUses(result, cmp);
            return true;
        },
        .convert => |cv| {
            const src_q = isF128(func, cv.value);
            const dst_q = isF128(func, result);
            if (!src_q and !dst_q) return false;
            if (src_q and dst_q) return false; // f128 -> f128 is a no-op, not a conversion
            const sym = if (dst_q) toF128Sym(func, cv.value) else fromF128Sym(func, result);
            const call = try emitCall(allocator, func, order, func.valueType(result), sym, &.{cv.value});
            func.replaceAllUses(result, call);
            return true;
        },
        .unary => |u| {
            // Only sqrt has a soft-fp form here (`__sqrttf2`); the other unary math ops on
            // f128 are left for isel to reject rather than silently mislowered.
            if (u.op != .sqrt or !isF128(func, result)) return false;
            const call = try emitCall(allocator, func, order, f128_ty, "__sqrttf2", &.{u.value});
            func.replaceAllUses(result, call);
            return true;
        },
        else => return false,
    }
}

/// The soft-fp name that widens `src` (an f32/f64/f16, or an integer) to f128.
fn toF128Sym(func: *const Function, src: Value) []const u8 {
    if (floatKind(func, src)) |fk| return switch (fk) {
        .f32 => "__extendsftf2",
        .f64 => "__extenddftf2",
        .f16 => "__extendhftf2",
        .f128 => unreachable, // f128 -> f128 handled by the caller
    };
    const ii = intInfo(func, src) orelse unreachable; // a convert source is a number
    return if (ii.signed)
        (if (ii.bits <= 32) "__floatsitf" else "__floatditf")
    else
        (if (ii.bits <= 32) "__floatunsitf" else "__floatunditf");
}

/// The soft-fp name that narrows f128 to `dst` (an f32/f64/f16, or an integer).
fn fromF128Sym(func: *const Function, dst: Value) []const u8 {
    if (floatKind(func, dst)) |fk| return switch (fk) {
        .f32 => "__trunctfsf2",
        .f64 => "__trunctfdf2",
        .f16 => "__trunctfhf2",
        .f128 => unreachable,
    };
    const ii = intInfo(func, dst) orelse unreachable;
    return if (ii.signed)
        (if (ii.bits <= 32) "__fixtfsi" else "__fixtfdi")
    else
        (if (ii.bits <= 32) "__fixunstfsi" else "__fixunstfdi");
}

/// Create a detached `call name(args) : ty` instruction, append its handle to `order`, and
/// return its result value.
fn emitCall(
    allocator: std.mem.Allocator,
    func: *Function,
    order: *std.ArrayList(Inst),
    ty: types.Type,
    name: []const u8,
    args: []const Value,
) std.mem.Allocator.Error!Value {
    const symbol = try func.internSymbol(name);
    const list = try func.internValues(args);
    return emitInst(allocator, func, order, ty, .{ .call = .{ .symbol = symbol, .args = list } });
}

/// Create a detached instruction, append its handle to `order`, and return its result value.
fn emitInst(
    allocator: std.mem.Allocator,
    func: *Function,
    order: *std.ArrayList(Inst),
    ty: types.Type,
    op: Opcode,
) std.mem.Allocator.Error!Value {
    const value = try func.createInst(ty, op);
    try order.append(allocator, func.definingInst(value).?);
    return value;
}

test {
    std.testing.refAllDecls(@This());
}

const testing = std.testing;

/// The single call symbol in `func`'s entry block, asserting there is exactly one call and no
/// leftover instruction of `banned` opcode tag.
fn onlyCallSym(func: *Function, banned: std.meta.Tag(Opcode)) ![]const u8 {
    var found: ?[]const u8 = null;
    var calls: usize = 0;
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        const op = func.opcode(inst);
        try testing.expect(op != banned);
        if (op == .call) {
            calls += 1;
            found = func.symbolName(op.call.symbol);
        }
    }
    try testing.expectEqual(@as(usize, 1), calls);
    return found.?;
}

test "f128 add lowers to a single __addtf3 call, no arith left" {
    var f = Function.init(testing.allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    f.setTerminator(b, .{ .ret = function.Ret.one(r) });

    try testing.expect(try lower(testing.allocator, &f));
    try testing.expectEqualStrings("__addtf3", try onlyCallSym(&f, .arith));
    // The call result, an f128, is what the return now yields.
    const call = f.blockInsts(@enumFromInt(0))[0];
    try testing.expect(isF128(&f, f.instResult(call).?));
}

test "f128 sub/mul/div select the matching soft-fp symbol" {
    const cases = [_]struct { op: function.BinOp, sym: []const u8 }{
        .{ .op = .sub, .sym = "__subtf3" },
        .{ .op = .mul, .sym = "__multf3" },
        .{ .op = .div, .sym = "__divtf3" },
    };
    for (cases) |c| {
        var f = Function.init(testing.allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f128 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = c.op, .lhs = x, .rhs = y } });
        f.setTerminator(b, .{ .ret = function.Ret.one(r) });
        try testing.expect(try lower(testing.allocator, &f));
        try testing.expectEqualStrings(c.sym, try onlyCallSym(&f, .arith));
    }
}

test "f128 compare lowers to __lttf2 plus a signed status compare against zero" {
    var f = Function.init(testing.allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const boolt = try f.types.intern(.bool);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, boolt, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    f.setTerminator(b, .{ .ret = function.Ret.one(r) });

    try testing.expect(try lower(testing.allocator, &f));
    // Exactly one call (__lttf2). A leftover integer icmp against the status is expected, so
    // the banned tag is .arith here, not .icmp.
    try testing.expectEqualStrings("__lttf2", try onlyCallSym(&f, .arith));
    // The final result is now an integer icmp of the i32 status against a zero constant.
    var saw_zero = false;
    var saw_status_cmp = false;
    for (f.blockInsts(@enumFromInt(0))) |inst| switch (f.opcode(inst)) {
        .iconst => |v| if (v == 0) {
            saw_zero = true;
        },
        .icmp => |c| {
            saw_status_cmp = true;
            try testing.expectEqual(function.CmpOp.lt, c.op);
        },
        else => {},
    };
    try testing.expect(saw_zero and saw_status_cmp);
}

test "f128 conversions pick the right extend/trunc/float/fix symbol" {
    const f64_t: types.FloatKind = .f64;
    _ = f64_t;
    // f64 -> f128
    {
        var f = Function.init(testing.allocator);
        defer f.deinit();
        const src_t = try f.types.intern(.{ .float = .f64 });
        const dst_t = try f.types.intern(.{ .float = .f128 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, src_t);
        const r = try f.appendInst(b, dst_t, .{ .convert = .{ .value = x } });
        f.setTerminator(b, .{ .ret = function.Ret.one(r) });
        try testing.expect(try lower(testing.allocator, &f));
        try testing.expectEqualStrings("__extenddftf2", try onlyCallSym(&f, .convert));
    }
    // f128 -> i64 (signed)
    {
        var f = Function.init(testing.allocator);
        defer f.deinit();
        const src_t = try f.types.intern(.{ .float = .f128 });
        const dst_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, src_t);
        const r = try f.appendInst(b, dst_t, .{ .convert = .{ .value = x } });
        f.setTerminator(b, .{ .ret = function.Ret.one(r) });
        try testing.expect(try lower(testing.allocator, &f));
        try testing.expectEqualStrings("__fixtfdi", try onlyCallSym(&f, .convert));
    }
    // u32 -> f128
    {
        var f = Function.init(testing.allocator);
        defer f.deinit();
        const src_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const dst_t = try f.types.intern(.{ .float = .f128 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, src_t);
        const r = try f.appendInst(b, dst_t, .{ .convert = .{ .value = x } });
        f.setTerminator(b, .{ .ret = function.Ret.one(r) });
        try testing.expect(try lower(testing.allocator, &f));
        try testing.expectEqualStrings("__floatunsitf", try onlyCallSym(&f, .convert));
    }
}

test "every emittable soft-fp symbol is listed in `symbols`" {
    // `rebindSymbolName` in a backend maps a borrowed relocation name back to a `symbols`
    // literal; a name the mappers can emit but that is missing from `symbols` would crash
    // there. Assert every mapper output is present.
    inline for (.{ function.BinOp.add, .sub, .mul, .div }) |op| {
        try testing.expect(staticName(arithSym(op).?) != null);
    }
    inline for (.{ function.CmpOp.eq, .ne, .lt, .le, .gt, .ge }) |op| {
        try testing.expect(staticName(cmpSym(op)) != null);
    }
    // Conversions: build tiny functions so the type-driven mappers have real values to read.
    const FK = types.FloatKind;
    inline for (.{ FK.f32, FK.f64, FK.f16 }) |fk| {
        var f = Function.init(testing.allocator);
        defer f.deinit();
        const b = try f.appendBlock();
        const fv = try f.appendBlockParam(b, try f.types.intern(.{ .float = fk }));
        try testing.expect(staticName(toF128Sym(&f, fv)) != null); // widen fk -> f128
        try testing.expect(staticName(fromF128Sym(&f, fv)) != null); // narrow f128 -> fk
    }
    inline for (.{ true, false }) |signed| inline for (.{ 32, 64 }) |bits| {
        var f = Function.init(testing.allocator);
        defer f.deinit();
        const b = try f.appendBlock();
        const iv = try f.appendBlockParam(b, try f.types.intern(.{ .int = .{ .signedness = if (signed) .signed else .unsigned, .bits = bits } }));
        try testing.expect(staticName(toF128Sym(&f, iv)) != null); // int -> f128
        try testing.expect(staticName(fromF128Sym(&f, iv)) != null); // f128 -> int
    };
    try testing.expect(staticName("__sqrttf2") != null);
}

test "a function with no f128 is unchanged" {
    var f = Function.init(testing.allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f64 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    f.setTerminator(b, .{ .ret = function.Ret.one(r) });
    try testing.expect(!try lower(testing.allocator, &f));
}
