//! Native execution validation for the AArch64 backend (test-only). The host is
//! aarch64, so generated A64 code maps into W^X memory and is called directly, no
//! emulator needed. The codegen oracle: a wrong encoding or selection returns the
//! wrong value or faults.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const isel = @import("../isel.zig");
const link = @import("../link.zig");
const jit = @import("../jit.zig");

const Function = ir.function.Function;

/// A named function for `runModule`. The entry must be first.
pub const NamedFunc = struct { name: []const u8, func: *Function };

/// Compile `func` to A64, JIT-map it, and call it with `args` (each loaded into an
/// argument register). Returns the i32 result. Skips when not on aarch64.
pub fn run(allocator: std.mem.Allocator, func: *const Function, args: []const i32) !i32 {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    return callI32(&buf, args);
}

/// Compile a single function and call it, returning its f64 result (in d0).
/// Integer arguments are passed in x-registers as usual. Skips off aarch64.
pub fn runF64(allocator: std.mem.Allocator, func: *const Function, args: []const i32) !f64 {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const ptr = buf.memory.ptr;
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) f64, @ptrCast(ptr))(),
        1 => @as(*const fn (i32) callconv(.c) f64, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i32, i32) callconv(.c) f64, @ptrCast(ptr))(args[0], args[1]),
        else => error.Unsupported,
    };
}

/// Link a set of functions into one image (resolving each `bl`), JIT-map it, and
/// call the entry (the first function). Skips when not on aarch64.
pub fn runModule(allocator: std.mem.Allocator, funcs: []const NamedFunc, args: []const i32) !i32 {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    var module: link.Module = .{};
    defer module.deinit(allocator);
    for (funcs) |nf| try module.addFunction(allocator, nf.name, nf.func);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    defer buf.deinit();
    return callI32(&buf, args);
}

fn callI32(buf: *const jit.CodeBuffer, args: []const i32) !i32 {
    const ptr = buf.memory.ptr; // page-aligned, satisfies the function-pointer alignment
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) i32, @ptrCast(ptr))(),
        1 => @as(*const fn (i32) callconv(.c) i32, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i32, i32) callconv(.c) i32, @ptrCast(ptr))(args[0], args[1]),
        3 => @as(*const fn (i32, i32, i32) callconv(.c) i32, @ptrCast(ptr))(args[0], args[1], args[2]),
        else => error.Unsupported,
    };
}

fn expectRun(allocator: std.mem.Allocator, func: *Function, args: []const i32, expected: i32) !void {
    try std.testing.expectEqual(expected, try run(allocator, func, args));
}

fn i32func(allocator: std.mem.Allocator) !Function {
    return Function.init(allocator);
}

test "native: a*b + a" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });
    try expectRun(allocator, &func, &.{ 3, 4 }, 15); // 3*4 + 3
}

test "neon: a vector crosses a block edge whole (block-param move, no truncation)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // out = cond ? in[0] : in[1], where the chosen vector reaches the merge block as a
    // <4 x f32> block parameter, so a parallel move carries it across the edge.
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const entry = try func.appendBlock();
    const out = try func.appendBlockParam(entry, ptr_t);
    const in = try func.appendBlockParam(entry, ptr_t);
    const cond = try func.appendBlockParam(entry, i32_t);
    const va = try func.appendInst(entry, v4, .{ .load = .{ .ptr = in } });
    const p1 = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = 16 } });
    const vb = try func.appendInst(entry, v4, .{ .load = .{ .ptr = p1 } });
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const m = try func.appendBlockParam(merge, v4);
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{va}) } });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{vb}) } });
    try func.appendStore(merge, m, out);
    func.setTerminator(merge, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [2][4]f32, i32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const input: [2][4]f32 align(16) = .{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 } };
    var r1: [4]f32 align(16) = .{ 0, 0, 0, 0 };
    f(&r1, &input, 1);
    try std.testing.expectEqual([4]f32{ 1, 2, 3, 4 }, r1); // chose in[0], all lanes intact
    var r0: [4]f32 align(16) = .{ 0, 0, 0, 0 };
    f(&r0, &input, 0);
    try std.testing.expectEqual([4]f32{ 5, 6, 7, 8 }, r0); // chose in[1], all lanes intact
}

test "neon: high vector pressure spills and reloads all 128 bits (no truncation)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // Load N vectors and sum them. N exceeds the FP register pool, so some vectors spill.
    // The sum is only correct if a spilled vector reloads whole (a 64-bit reload would lose
    // lanes 2 and 3).
    const N = 18;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const in = try func.appendBlockParam(blk, ptr_t);
    var v: [N]ir.function.Value = undefined;
    for (0..N) |i| {
        const p = if (i == 0) in else try func.appendInst(blk, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = in, .imm = @intCast(i * 16) } });
        v[i] = try func.appendInst(blk, v4, .{ .load = .{ .ptr = p } });
    }
    var s = v[0];
    for (1..N) |i| s = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = s, .rhs = v[i] } });
    try func.appendStore(blk, s, out);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (*[4]f32, *const [N][4]f32) callconv(.c) void;
    var input: [N][4]f32 align(16) = undefined;
    for (0..N) |i| input[i] = .{ 1, 2, 3, 4 };
    var result: [4]f32 align(16) = .{ 0, 0, 0, 0 };
    @as(Fn, @ptrCast(buf.memory.ptr))(&result, &input);
    try std.testing.expectEqual([4]f32{ N, 2 * N, 3 * N, 4 * N }, result); // every lane summed
}

fn runF32x8(allocator: std.mem.Allocator, func: *const Function, args: [8]f32) !f32 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (f32, f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

test "vectorize: 4 parallel f32 adds fuse into a NEON vector add (same result)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // s = (a0+b0) + (a1+b1) + (a2+b2) + (a3+b3): four parallel adds, then a reduction.
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    var a: [4]ir.function.Value = undefined;
    var b: [4]ir.function.Value = undefined;
    for (0..4) |i| a[i] = try func.appendBlockParam(blk, f32_t);
    for (0..4) |i| b[i] = try func.appendBlockParam(blk, f32_t);
    var c: [4]ir.function.Value = undefined;
    for (0..4) |i| c[i] = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = a[i], .rhs = b[i] } });
    var s = c[0];
    for (1..4) |i| s = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = c[i] } });
    func.setTerminator(blk, .{ .ret = s });

    const args = [8]f32{ 1, 2, 3, 4, 10, 20, 30, 40 };
    const scalar_res = try runF32x8(allocator, &func, args); // 11+22+33+44 = 110

    const changed = try opt.vectorize.run(allocator, &func);
    try std.testing.expect(changed);

    // A vector-typed arith now exists (the four scalar adds became one).
    var has_vec = false;
    for (0..func.instCount()) |i| {
        if (func.opcodeMut(@enumFromInt(i)).* == .arith) {
            const res = func.instResult(@enumFromInt(i)).?;
            if (func.types.type_kind(func.valueType(res)) == .vector) has_vec = true;
        }
    }
    try std.testing.expect(has_vec);

    const vec_res = try runF32x8(allocator, &func, args);
    try std.testing.expectEqual(scalar_res, vec_res);
    try std.testing.expectEqual(@as(f32, 110), vec_res);
}

fn loadLane(func: *Function, blk: ir.function.Block, ptr_t: ir.types.Type, f32_t: ir.types.Type, base: ir.function.Value, i: usize) !ir.function.Value {
    if (i == 0) return func.appendInst(blk, f32_t, .{ .load = .{ .ptr = base } });
    const p = try func.appendInst(blk, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = @intCast(i * 4) } });
    return func.appendInst(blk, f32_t, .{ .load = .{ .ptr = p } });
}

test "vectorize: chained (a+b)*c keeps the intermediate in a vector (pack reuse)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // s = sum_i (a_i + b_i) * c_i. The add group feeds the mul group. Pack-reuse should wire
    // the vector add's result straight into the vector mul (no re-pack between them).
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const pc = try func.appendBlockParam(blk, ptr_t);
    var a: [4]ir.function.Value = undefined;
    var b: [4]ir.function.Value = undefined;
    var cc: [4]ir.function.Value = undefined;
    for (0..4) |i| a[i] = try loadLane(&func, blk, ptr_t, f32_t, pa, i);
    for (0..4) |i| b[i] = try loadLane(&func, blk, ptr_t, f32_t, pb, i);
    for (0..4) |i| cc[i] = try loadLane(&func, blk, ptr_t, f32_t, pc, i);
    var t: [4]ir.function.Value = undefined;
    for (0..4) |i| t[i] = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = a[i], .rhs = b[i] } });
    var r: [4]ir.function.Value = undefined;
    for (0..4) |i| r[i] = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .mul, .lhs = t[i], .rhs = cc[i] } });
    var s = r[0];
    for (1..4) |i| s = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = r[i] } });
    func.setTerminator(blk, .{ .ret = s });

    const av = [4]f32{ 1, 2, 3, 4 };
    const bv = [4]f32{ 10, 20, 30, 40 };
    const cv = [4]f32{ 2, 2, 2, 2 };
    const Run = struct {
        fn call(al: std.mem.Allocator, fnc: *const Function, x: *const [4]f32, y: *const [4]f32, z: *const [4]f32) !f32 {
            const code = try isel.selectFunction(al, fnc);
            defer al.free(code);
            var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
            defer buf.deinit();
            const Fn = *const fn (*const [4]f32, *const [4]f32, *const [4]f32) callconv(.c) f32;
            return @as(Fn, @ptrCast(buf.memory.ptr))(x, y, z);
        }
    };
    const scalar_res = try Run.call(allocator, &func, &av, &bv, &cv); // (11*2)+(22*2)+(33*2)+(44*2) = 220

    try std.testing.expect(try opt.vectorize.run(allocator, &func));

    // The vector mul should take the vector add's result directly (the chain stayed vector).
    var vadd_res: ?ir.function.Value = null;
    var vmul_lhs: ?ir.function.Value = null;
    for (0..func.instCount()) |i| {
        const o = func.opcodeMut(@enumFromInt(i)).*;
        if (o != .arith) continue;
        const res = func.instResult(@enumFromInt(i)).?;
        if (func.types.type_kind(func.valueType(res)) != .vector) continue;
        if (o.arith.op == .add) vadd_res = res;
        if (o.arith.op == .mul) vmul_lhs = o.arith.lhs;
    }
    try std.testing.expect(vadd_res != null and vmul_lhs != null);
    try std.testing.expectEqual(vadd_res.?, vmul_lhs.?); // pack-reuse: add's vector feeds the mul

    // The intermediate add's lanes are extracted as register ops, pack-reuse makes them dead,
    // so DCE removes them. Only the final result's 4 extracts should survive.
    var analyses = opt.pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    _ = try opt.dce.run(allocator, &func, &analyses);
    var extracts: usize = 0;
    var allocas: usize = 0;
    for (func.blockInsts(blk)) |inst| {
        switch (func.opcode(inst)) {
            .extract => extracts += 1,
            .alloca => allocas += 1,
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 4), extracts); // the dead intermediate extracts are gone
    try std.testing.expectEqual(@as(usize, 0), allocas); // pack/unpack are register ops, no stack

    const vec_res = try Run.call(allocator, &func, &av, &bv, &cv);
    try std.testing.expectEqual(scalar_res, vec_res);
    try std.testing.expectEqual(@as(f32, 220), vec_res);
}

test "neon: <4 x f32> lane-wise add/mul through pointers" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // out = a * b + a, computed a full 4-wide vector at a time (one fmul + one fadd).
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const pa = try func.appendBlockParam(blk, ptr_t);
    const pb = try func.appendBlockParam(blk, ptr_t);
    const va = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pa } });
    const vb = try func.appendInst(blk, v4, .{ .load = .{ .ptr = pb } });
    const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const sum = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = va } });
    try func.appendStore(blk, sum, out);
    func.setTerminator(blk, .{ .ret = null });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();

    const Fn = *const fn (*[4]f32, *const [4]f32, *const [4]f32) callconv(.c) void;
    const f: Fn = @ptrCast(buf.memory.ptr);
    const a align(16) = [4]f32{ 1, 2, 3, 4 };
    const b align(16) = [4]f32{ 10, 20, 30, 40 };
    var result align(16) = [4]f32{ 0, 0, 0, 0 };
    f(&result, &a, &b);
    try std.testing.expectEqual([4]f32{ 11, 42, 93, 164 }, result); // a*b + a per lane
}

test "native: signed division and remainder" {
    const allocator = std.testing.allocator;
    inline for (.{ .{ ir.function.BinOp.div, @as(i32, 6) }, .{ ir.function.BinOp.rem, @as(i32, 2) } }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, t);
        const y = try func.appendBlockParam(b, t);
        const r = try func.appendInst(b, t, .{ .arith = .{ .op = case[0], .lhs = x, .rhs = y } });
        func.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &func, &.{ 20, 3 }, case[1]); // 20/3 = 6, 20%3 = 2
    }
}

test "native: shifts (left and arithmetic right)" {
    const allocator = std.testing.allocator;
    inline for (.{ .{ ir.function.BinOp.shl, @as(i32, -16) }, .{ ir.function.BinOp.shr, @as(i32, -1) } }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, t);
        const y = try func.appendBlockParam(b, t);
        const r = try func.appendInst(b, t, .{ .arith = .{ .op = case[0], .lhs = x, .rhs = y } });
        func.setTerminator(b, .{ .ret = r });
        try expectRun(allocator, &func, &.{ -4, 2 }, case[1]); // -4<<2 = -16, -4>>2 = -1 (asr)
    }
}

test "native: select picks the smaller operand" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const c = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
    const m = try func.appendInst(b, t, .{ .select = .{ .cond = c, .then = x, .@"else" = y } });
    func.setTerminator(b, .{ .ret = m });
    try expectRun(allocator, &func, &.{ 7, 3 }, 3); // min(7,3)
}

test "native: subtraction yields a negative result" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const d = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = d });
    try expectRun(allocator, &func, &.{ 3, 10 }, -7);
}

test "native: constants and immediate arithmetic" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    // (x + 100) ^ 0xFF00
    const c = try func.appendArithImm(b, t, .add, x, 100);
    const mask = try func.appendInst(b, t, .{ .iconst = 0xFF00 });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .bit_xor, .lhs = c, .rhs = mask } });
    func.setTerminator(b, .{ .ret = r });
    try expectRun(allocator, &func, &.{5}, (5 + 100) ^ 0xFF00);
}

test "native: a wide constant via movz/movk" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, t, .{ .iconst = 0x1234_5678 });
    func.setTerminator(b, .{ .ret = c });
    try expectRun(allocator, &func, &.{}, 0x1234_5678);
}

test "native: max via a conditional branch to two return blocks" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = a });
    func.setTerminator(else_b, .{ .ret = b });

    try expectRun(allocator, &func, &.{ 7, 3 }, 7);
    try expectRun(allocator, &func, &.{ 3, 4 }, 4);
}

test "native: max via a merge block with parameters" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const merge = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const z = try func.appendBlockParam(merge, t);
    // if a < b -> merge(b) else merge(a): the larger flows through the param.
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = merge, .args = &.{b} }, .{ .target = merge, .args = &.{a} });
    func.setTerminator(merge, .{ .ret = z });

    try expectRun(allocator, &func, &.{ 7, 3 }, 7);
    try expectRun(allocator, &func, &.{ 3, 4 }, 4);
}

test "native: a non-leaf function calls another and uses the result" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee(a) = a * 2   (leaf)
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(t_kind);
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        const two = try callee.appendInst(b, t, .{ .iconst = 2 });
        const r = try callee.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = two } });
        callee.setTerminator(b, .{ .ret = r });
    }
    // caller(x) = callee(x) + 1   (non-leaf: saves fp/lr + a callee-saved reg)
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(t_kind);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const call = try caller.appendCall(b, t, "callee", &.{x});
        const r = try caller.appendArithImm(b, t, .add, call, 1);
        caller.setTerminator(b, .{ .ret = r });
    }

    // caller(5) = (5*2) + 1 = 11.
    try std.testing.expectEqual(@as(i32, 11), try runModule(allocator, &.{
        .{ .name = "caller", .func = &caller },
        .{ .name = "callee", .func = &callee },
    }, &.{5}));
}

test "native: a non-leaf function with a value live across a call" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a   (leaf)
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(t_kind);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    // f(x) = dbl(x) + x   (x is live across the call, so it must survive in a
    // callee-saved register)
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const t = try f.types.intern(t_kind);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const d = try f.appendCall(b, t, "dbl", &.{x});
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = d, .rhs = x } });
        f.setTerminator(b, .{ .ret = r });
    }

    // f(10) = (10+10) + 10 = 30.
    try std.testing.expectEqual(@as(i32, 30), try runModule(allocator, &.{
        .{ .name = "f", .func = &f },
        .{ .name = "dbl", .func = &dbl },
    }, &.{10}));
}

test "native: a counted loop sums 0..n (back-edge)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    try expectRun(allocator, &func, &.{5}, 10); // 0+1+2+3+4
}

test "native: register spilling under high pressure (leaf)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e = try func.appendBlock();
    const p0 = try func.appendBlockParam(e, t);
    const p1 = try func.appendBlockParam(e, t);

    // 20 values all live at once (defined before any is consumed) exhausts the
    // ~10-register leaf pool, forcing spills. Folding them back exercises reloads.
    var vals: [20]ir.function.Value = undefined;
    for (&vals) |*v| v.* = try func.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(e, .{ .ret = acc });

    // Each value is p0 + p1 = 2, summing 20 of them = 40.
    try expectRun(allocator, &func, &.{ 1, 1 }, 40);
}

test "native: register spilling under high pressure (non-leaf)" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // id(a) = a
    var id = Function.init(allocator);
    defer id.deinit();
    {
        const t = try id.types.intern(t_kind);
        const b = try id.appendBlock();
        const a = try id.appendBlockParam(b, t);
        id.setTerminator(b, .{ .ret = a });
    }
    // f(p0, p1): 15 values live across a call to id, then folded. The call makes
    // this non-leaf (callee-saved pool of ~10), so the long-lived values spill.
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const t = try f.types.intern(t_kind);
        const e = try f.appendBlock();
        const p0 = try f.appendBlockParam(e, t);
        const p1 = try f.appendBlockParam(e, t);
        var vals: [15]ir.function.Value = undefined;
        for (&vals) |*v| v.* = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
        const r = try f.appendCall(e, t, "id", &.{p0}); // clobbers caller-saved regs
        var acc = r;
        for (vals) |v| acc = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
        f.setTerminator(e, .{ .ret = acc });
    }
    // f(1,1) = id(1) + 15*(1+1) = 1 + 30 = 31.
    try std.testing.expectEqual(@as(i32, 31), try runModule(allocator, &.{
        .{ .name = "f", .func = &f },
        .{ .name = "id", .func = &id },
    }, &.{ 1, 1 }));
}

test "native: a call with ten arguments (stack args)" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee(a0..a9) = a0 + a1 + ... + a9 (a8, a9 arrive on the stack)
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(t_kind);
        const b = try callee.appendBlock();
        var ps: [10]ir.function.Value = undefined;
        for (&ps) |*p| p.* = try callee.appendBlockParam(b, t);
        var sum = ps[0];
        for (ps[1..]) |p| sum = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = p } });
        callee.setTerminator(b, .{ .ret = sum });
    }
    // caller() = callee(1, 2, ..., 10)
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(t_kind);
        const b = try caller.appendBlock();
        var args: [10]ir.function.Value = undefined;
        for (&args, 0..) |*a, i| a.* = try caller.appendInst(b, t, .{ .iconst = @intCast(i + 1) });
        const r = try caller.appendCall(b, t, "callee", &args);
        caller.setTerminator(b, .{ .ret = r });
    }
    // 1 + 2 + ... + 10 = 55.
    try std.testing.expectEqual(@as(i32, 55), try runModule(allocator, &.{
        .{ .name = "caller", .func = &caller },
        .{ .name = "callee", .func = &callee },
    }, &.{}));
}

test "native: alloca stores and reloads through a stack frame" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(e, t);
    const slot = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(e, x, slot);
    const v = try func.appendInst(e, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(e, .{ .ret = v });
    try expectRun(allocator, &func, &.{42}, 42);
}

test "native: sub-word store and sign-extending load (i8)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const a = try func.appendBlockParam(e, i8_t);
    const slot = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i8_t } });
    try func.appendStore(e, a, slot); // strb (low byte)
    const v = try func.appendInst(e, i8_t, .{ .load = .{ .ptr = slot } }); // ldrsb (sign-extend)
    func.setTerminator(e, .{ .ret = v });
    // 200 stored as a byte is 0xC8, loaded with ldrsb it sign-extends to -56.
    try expectRun(allocator, &func, &.{200}, -56);
}

test "native: a stack slot survives a call (alloca in a non-leaf frame)" {
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // id(a) = a   (leaf)
    var id = Function.init(allocator);
    defer id.deinit();
    {
        const t = try id.types.intern(t_kind);
        const b = try id.appendBlock();
        const a = try id.appendBlockParam(b, t);
        id.setTerminator(b, .{ .ret = a });
    }
    // f(x): slot = alloca, *slot = x, r = id(x), return *slot + r
    // (the alloca lives above the saved registers in the non-leaf frame)
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const t = try f.types.intern(t_kind);
        const ptr_t = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, x, slot);
        const r = try f.appendCall(b, t, "id", &.{x});
        const v = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        const sum = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = r } });
        f.setTerminator(b, .{ .ret = sum });
    }
    // f(5) = 5 + id(5) = 10.
    try std.testing.expectEqual(@as(i32, 10), try runModule(allocator, &.{
        .{ .name = "f", .func = &f },
        .{ .name = "id", .func = &id },
    }, &.{5}));
}

test "native: f64 constant returned in d0" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, f64_t, .{ .fconst = 3.5 });
    func.setTerminator(b, .{ .ret = c });
    try std.testing.expectEqual(@as(f64, 3.5), try runF64(allocator, &func, &.{}));
}

test "native: f64 arithmetic (add/sub/mul/div)" {
    const allocator = std.testing.allocator;
    const cases = .{
        .{ ir.function.BinOp.add, @as(f64, 1.5), @as(f64, 2.5), @as(f64, 4.0) },
        .{ ir.function.BinOp.sub, @as(f64, 5.0), @as(f64, 1.5), @as(f64, 3.5) },
        .{ ir.function.BinOp.mul, @as(f64, 1.5), @as(f64, 4.0), @as(f64, 6.0) },
        .{ ir.function.BinOp.div, @as(f64, 7.0), @as(f64, 2.0), @as(f64, 3.5) },
    };
    inline for (cases) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const f64_t = try func.types.intern(.{ .float = .f64 });
        const b = try func.appendBlock();
        const x = try func.appendInst(b, f64_t, .{ .fconst = case[1] });
        const y = try func.appendInst(b, f64_t, .{ .fconst = case[2] });
        const r = try func.appendInst(b, f64_t, .{ .arith = .{ .op = case[0], .lhs = x, .rhs = y } });
        func.setTerminator(b, .{ .ret = r });
        try std.testing.expectEqual(case[3], try runF64(allocator, &func, &.{}));
    }
}

test "native: int<->f64 conversions" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const fx = try func.appendInst(b, f64_t, .{ .convert = .{ .value = x } }); // scvtf
    const half = try func.appendInst(b, f64_t, .{ .fconst = 0.5 });
    const prod = try func.appendInst(b, f64_t, .{ .arith = .{ .op = .mul, .lhs = fx, .rhs = half } });
    const sum = try func.appendInst(b, f64_t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = half } });
    const r = try func.appendInst(b, i32_t, .{ .convert = .{ .value = sum } }); // fcvtzs
    func.setTerminator(b, .{ .ret = r });
    // int(7.0 * 0.5 + 0.5) = int(4.0) = 4.
    try expectRun(allocator, &func, &.{7}, 4);
}

test "native: f32 arithmetic then narrow to int" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendInst(b, f32_t, .{ .fconst = 2.5 });
    const y = try func.appendInst(b, f32_t, .{ .fconst = 1.5 });
    const s = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const r = try func.appendInst(b, i32_t, .{ .convert = .{ .value = s } });
    func.setTerminator(b, .{ .ret = r });
    try expectRun(allocator, &func, &.{}, 4); // int(2.5 + 1.5) = 4
}

test "native: f64 compare and select" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const a = try func.appendInst(b, f64_t, .{ .fconst = 1.0 });
    const c = try func.appendInst(b, f64_t, .{ .fconst = 2.0 });
    const lt = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = c } }); // fcmp
    const then = try func.appendInst(b, f64_t, .{ .fconst = 3.5 });
    const els = try func.appendInst(b, f64_t, .{ .fconst = 9.5 });
    const m = try func.appendInst(b, f64_t, .{ .select = .{ .cond = lt, .then = then, .@"else" = els } });
    func.setTerminator(b, .{ .ret = m });
    try std.testing.expectEqual(@as(f64, 3.5), try runF64(allocator, &func, &.{})); // 1.0 < 2.0 -> 3.5
}

test "native: a call passing f64 arguments in v-registers" {
    const allocator = std.testing.allocator;
    const f64_kind = ir.types.TypeKind{ .float = .f64 };

    // addf(a, b) = a + b
    var addf = Function.init(allocator);
    defer addf.deinit();
    {
        const t = try addf.types.intern(f64_kind);
        const b = try addf.appendBlock();
        const a = try addf.appendBlockParam(b, t);
        const bb = try addf.appendBlockParam(b, t);
        const r = try addf.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bb } });
        addf.setTerminator(b, .{ .ret = r });
    }
    // caller() = int(addf(1.5, 2.5))
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(f64_kind);
        const i32_t = try caller.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try caller.appendBlock();
        const c1 = try caller.appendInst(b, t, .{ .fconst = 1.5 });
        const c2 = try caller.appendInst(b, t, .{ .fconst = 2.5 });
        const s = try caller.appendCall(b, t, "addf", &.{ c1, c2 });
        const r = try caller.appendInst(b, i32_t, .{ .convert = .{ .value = s } });
        caller.setTerminator(b, .{ .ret = r });
    }
    // int(1.5 + 2.5) = int(4.0) = 4.
    try std.testing.expectEqual(@as(i32, 4), try runModule(allocator, &.{
        .{ .name = "caller", .func = &caller },
        .{ .name = "addf", .func = &addf },
    }, &.{}));
}

test "jit: compile a module and call functions by name" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a   (leaf)
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(t_kind);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    // caller(x) = dbl(x) + 1
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(t_kind);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = r });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "dbl", &dbl);

    var compiled = try jit.compileModule(allocator, &module);
    defer compiled.deinit(allocator);

    // Both functions are individually callable by name through the JIT.
    const caller_fn = compiled.funcPointer(*const fn (i32) callconv(.c) i32, "caller").?;
    try std.testing.expectEqual(@as(i32, 11), caller_fn(5)); // dbl(5) + 1
    const dbl_fn = compiled.funcPointer(*const fn (i32) callconv(.c) i32, "dbl").?;
    try std.testing.expectEqual(@as(i32, 14), dbl_fn(7));
    try std.testing.expect(compiled.funcPointer(*const fn () callconv(.c) i32, "missing") == null);
}

test "pipeline: an optimized function runs correctly on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // f(x) = (2 + 3) * 4 + x, with a dead x*x. Constant folding collapses the
    // arithmetic to 20 and DCE drops the dead product, then aarch64 codegen runs it.
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const c3 = try func.appendInst(b, t, .{ .iconst = 3 });
    const c4 = try func.appendInst(b, t, .{ .iconst = 4 });
    const a = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c2, .rhs = c3 } });
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c4 } });
    _ = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } }); // dead
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = r });

    try std.testing.expect(try opt.optimize(allocator, &func));
    try expectRun(allocator, &func, &.{22}, 42); // 20 + 22
}

test "pipeline: inlining composes with aarch64 codegen" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // madd(a, b) = a*b + a
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const cb = try callee.appendBlock();
        const a = try callee.appendBlockParam(cb, t);
        const bb = try callee.appendBlockParam(cb, t);
        const prod = try callee.appendInst(cb, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
        const sum = try callee.appendInst(cb, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
        callee.setTerminator(cb, .{ .ret = sum });
    }
    // f() = madd(2, 3)
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const b = try caller.appendBlock();
    const c2 = try caller.appendInst(b, t, .{ .iconst = 2 });
    const c3 = try caller.appendInst(b, t, .{ .iconst = 3 });
    const r = try caller.appendCall(b, t, "madd", &.{ c2, c3 });
    caller.setTerminator(b, .{ .ret = r });

    const Lk = struct {
        callee: *const Function,
        fn get(ctx: *anyopaque, name: []const u8) ?*const Function {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return if (std.mem.eql(u8, name, "madd")) self.callee else null;
        }
    };
    var lk = Lk{ .callee = &callee };
    const lookup = opt.inlining.Lookup{ .context = &lk, .func = Lk.get };
    try std.testing.expect(try opt.inlining.run(allocator, &caller, lookup));
    _ = try opt.optimize(allocator, &caller); // fold across the inlined call
    for (caller.blockInsts(b)) |inst| try std.testing.expect(caller.opcode(inst) != .call);

    try expectRun(allocator, &caller, &.{}, 8); // madd(2,3) = 8, after inlining
}

test "pipeline: LTO across modules then aarch64 codegen" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var src = opt.lto.Module.init(allocator);
    {
        // helper(a, b) = a*b + a
        var f = Function.init(allocator);
        const t = try f.types.intern(i32k);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const bb = try f.appendBlockParam(b, t);
        const prod = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
        const sum = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
        f.setTerminator(b, .{ .ret = sum });
        try src.add("helper", f);
        // entry(x) = helper(x, x) + 1
        var g = Function.init(allocator);
        const gt = try g.types.intern(i32k);
        const gb = try g.appendBlock();
        const x = try g.appendBlockParam(gb, gt);
        const call = try g.appendCall(gb, gt, "helper", &.{ x, x });
        const r = try g.appendArithImm(gb, gt, .add, call, 1);
        g.setTerminator(gb, .{ .ret = r });
        try src.add("entry", g);
    }
    const blob = try opt.lto.encode(allocator, &src);
    src.deinit();
    defer allocator.free(blob);
    var module = try opt.lto.decode(allocator, blob);
    defer module.deinit();
    _ = try opt.lto.link(allocator, &module, &.{"entry"});
    try std.testing.expectEqual(@as(usize, 1), module.count()); // helper inlined + pruned

    // entry(x) = x*x + x + 1.  x=4: 16 + 4 + 1 = 21.
    try expectRun(allocator, module.get("entry").?, &.{4}, 21);
}

test "object+ld: emit ELF .o, link it, and JIT-run the result" {
    const allocator = std.testing.allocator;
    const object = @import("../object.zig");
    const ld = @import("../ld.zig");
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a   (leaf)
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    // caller(x) = dbl(x) + 1   (a cross-function CALL26 relocation)
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = r });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "dbl", &dbl);

    // Module -> ELF .o -> the linker -> image.
    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var image = try ld.linkObjects(allocator, &.{obj}, 0);
    defer image.deinit(allocator);

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // JIT-map the linked image and call through it.
    var buf = try jit.CodeBuffer.map(image.code);
    defer buf.deinit();
    const caller_fn = buf.entry(*const fn (i32) callconv(.c) i32, image.addressOf("caller").?);
    try std.testing.expectEqual(@as(i32, 11), caller_fn(5)); // dbl(5) + 1
    const dbl_fn = buf.entry(*const fn (i32) callconv(.c) i32, image.addressOf("dbl").?);
    try std.testing.expectEqual(@as(i32, 14), dbl_fn(7));
}

test "spirv: lower a SPIR-V function to IR and run it natively (x*y + x)" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");

    // Hand-assembled SPIR-V: int f(int x, int y) { return x*y + x }
    // ids: int=1, fnty=2, f=3, x=4, y=5, entry=6, prod=7, sum=8.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    const op = spirv.opcodes;
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(allocator, op.IAdd, &.{ 1, 8, 7, 4 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // SPIR-V -> Vulcan IR -> AArch64 -> run on the host: f(3, 4) = 3*4 + 3 = 15.
    try expectRun(allocator, &func, &.{ 3, 4 }, 15);
}

test "spirv: an optimized SPIR-V function runs natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const op = spirv.opcodes;

    // int f(int x) { return (x + 7) * x }  (a constant plus two ops)
    // ids: int=1, fnty=2, c7=3, f=4, x=5, entry=6, sum=7, prod=8.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 1, 3, 7 });
    try b.emit(allocator, op.Function, &.{ 1, 4, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IAdd, &.{ 1, 7, 5, 3 });
    try b.emit(allocator, op.IMul, &.{ 1, 8, 7, 5 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    _ = try opt.optimize(allocator, &func); // the IR composes with the optimizer
    // f(5) = (5 + 7) * 5 = 60.
    try expectRun(allocator, &func, &.{5}, 60);
}

test "spirv: int<->float conversions lowered and run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const op = spirv.opcodes;

    // int f(int x) { return int(float(x) * 2.5) }
    // ids: int=1, float=2, fnty=3, c2.5=4, f=5, x=6, entry=7, fx=8, scaled=9, r=10.
    var b = try spirv.binary.Builder.init(allocator, 11);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeFunction, &.{ 3, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 4, 0x40200000 }); // 2.5f
    try b.emit(allocator, op.Function, &.{ 1, 5, 0, 3 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 6 });
    try b.emit(allocator, op.Label, &.{7});
    try b.emit(allocator, op.ConvertSToF, &.{ 2, 8, 6 }); // fx = float(x)
    try b.emit(allocator, op.FMul, &.{ 2, 9, 8, 4 }); // scaled = fx * 2.5
    try b.emit(allocator, op.ConvertFToS, &.{ 1, 10, 9 }); // r = int(scaled)
    try b.emit(allocator, op.ReturnValue, &.{10});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    // f(4) = int(4.0 * 2.5) = int(10.0) = 10.
    try expectRun(allocator, &func, &.{4}, 10);
}

test "native: 64-bit pointer arithmetic into a stack array (base + i*4)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const arr_t = try func.types.intern(.{ .array = .{ .len = 8, .elem = i32_t } });

    // f(i) { int buf[8], int* p = buf + i, *p = i*10 + 1, return *p }
    // Exercises a 32-byte array alloca, a 64-bit pointer add, and load/store at a
    // computed address. The byte offset is i*4 (element stride).
    const e = try func.appendBlock();
    const i = try func.appendBlockParam(e, i32_t);
    const buf = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = arr_t } });
    const off = try func.appendArithImm(e, i32_t, .shl, i, 2); // i*4 bytes
    const p = try func.appendInst(e, ptr_t, .{ .arith = .{ .op = .add, .lhs = buf, .rhs = off } }); // 64-bit add
    const ten_i = try func.appendInst(e, i32_t, .{ .iconst = 10 });
    const scaled = try func.appendInst(e, i32_t, .{ .arith = .{ .op = .mul, .lhs = i, .rhs = ten_i } });
    const val = try func.appendArithImm(e, i32_t, .add, scaled, 1);
    try func.appendStore(e, val, p);
    const got = try func.appendInst(e, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(e, .{ .ret = got });

    // f(3) = 3*10 + 1 = 31, written at buf+12 and read back.
    try expectRun(allocator, &func, &.{3}, 31);
}

/// Run a lowered compute kernel `main(invocation_id, buf_ptr)` natively against a
/// real buffer, mutating it in place. The signature matches the SPIR-V frontend's
/// synthesized entry params (the invocation id, then the storage-buffer pointer).
fn runCompute(allocator: std.mem.Allocator, func: *const Function, invocation_id: i32, buf: []i32) !void {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    var cb = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer cb.deinit();
    const main_fn = cb.entry(*const fn (i32, [*]i32) callconv(.c) void, 0);
    main_fn(invocation_id, buf.ptr);
}

test "spirv compute: storage-buffer shader runs natively (data[gid] *= 2)" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // void main() { data[gl_GlobalInvocationID.x] = data[gid] * 2 }
    var b = try spirv.binary.Builder.init(allocator, 23);
    defer b.deinit(allocator);
    try b.emit(allocator, o.Decorate, &.{ 14, o.Decoration.builtin, o.BuiltIn.global_invocation_id });
    try b.emit(allocator, o.TypeVoid, &.{1});
    try b.emit(allocator, o.TypeInt, &.{ 2, 32, 1 });
    try b.emit(allocator, o.TypeInt, &.{ 3, 32, 0 });
    try b.emit(allocator, o.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, o.TypePointer, &.{ 5, o.StorageClass.input, 4 });
    try b.emit(allocator, o.TypePointer, &.{ 6, o.StorageClass.input, 3 });
    try b.emit(allocator, o.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, o.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, o.TypePointer, &.{ 9, o.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, o.TypePointer, &.{ 10, o.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, o.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, o.Constant, &.{ 3, 12, 0 });
    try b.emit(allocator, o.Constant, &.{ 2, 13, 2 });
    try b.emit(allocator, o.Variable, &.{ 5, 14, o.StorageClass.input });
    try b.emit(allocator, o.Variable, &.{ 9, 15, o.StorageClass.storage_buffer });
    try b.emit(allocator, o.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, o.Label, &.{17});
    try b.emit(allocator, o.AccessChain, &.{ 6, 18, 14, 12 });
    try b.emit(allocator, o.Load, &.{ 3, 19, 18 });
    try b.emit(allocator, o.AccessChain, &.{ 10, 20, 15, 12, 19 });
    try b.emit(allocator, o.Load, &.{ 2, 21, 20 });
    try b.emit(allocator, o.IMul, &.{ 2, 22, 21, 13 });
    try b.emit(allocator, o.Store, &.{ 20, 22 });
    try b.emit(allocator, o.Return, &.{});
    try b.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // Run one invocation (thread 2) over a real buffer: only data[2] doubles.
    var data = [_]i32{ 10, 20, 30, 40 };
    try runCompute(allocator, &func, 2, &data);
    try std.testing.expectEqualSlices(i32, &.{ 10, 20, 60, 40 }, &data);

    // A different thread index hits a different element.
    var data2 = [_]i32{ 5, 6, 7, 8 };
    try runCompute(allocator, &func, 0, &data2);
    try std.testing.expectEqualSlices(i32, &.{ 10, 6, 7, 8 }, &data2);
}

/// Build f(a, b) = a <op> b with the given signedness, lower its division to
/// division-free IR, and run it natively. Validates the lowering against the
/// host's real divide.
fn runLoweredDiv(allocator: std.mem.Allocator, signedness: std.builtin.Signedness, bop: ir.function.BinOp, a: i32, b: i32) !i32 {
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = signedness, .bits = 32 } });
    const blk = try func.appendBlock();
    const x = try func.appendBlockParam(blk, t);
    const y = try func.appendBlockParam(blk, t);
    const r = try func.appendInst(blk, t, .{ .arith = .{ .op = bop, .lhs = x, .rhs = y } });
    func.setTerminator(blk, .{ .ret = r });
    try std.testing.expect(try opt.lowerdiv.run(allocator, &func));
    return run(allocator, &func, &.{ a, b });
}

test "lowerdiv: unsigned division and remainder run correctly" {
    const allocator = std.testing.allocator;
    const U = std.builtin.Signedness.unsigned;
    try std.testing.expectEqual(@as(i32, 6), try runLoweredDiv(allocator, U, .div, 20, 3));
    try std.testing.expectEqual(@as(i32, 2), try runLoweredDiv(allocator, U, .rem, 20, 3));
    try std.testing.expectEqual(@as(i32, 14), try runLoweredDiv(allocator, U, .div, 100, 7));
    try std.testing.expectEqual(@as(i32, 15), try runLoweredDiv(allocator, U, .div, 255, 16));
    // The high bit must be treated as magnitude, not sign: 0x80000000 / 2.
    try std.testing.expectEqual(@as(i32, @bitCast(@as(u32, 0x40000000))), try runLoweredDiv(allocator, U, .div, @bitCast(@as(u32, 0x80000000)), 2));
}

test "lowerdiv: signed division and remainder run correctly (round toward zero)" {
    const allocator = std.testing.allocator;
    const S = std.builtin.Signedness.signed;
    try std.testing.expectEqual(@as(i32, 6), try runLoweredDiv(allocator, S, .div, 20, 3));
    try std.testing.expectEqual(@as(i32, -6), try runLoweredDiv(allocator, S, .div, -20, 3));
    try std.testing.expectEqual(@as(i32, -6), try runLoweredDiv(allocator, S, .div, 20, -3));
    try std.testing.expectEqual(@as(i32, 6), try runLoweredDiv(allocator, S, .div, -20, -3));
    // Remainder takes the sign of the dividend.
    try std.testing.expectEqual(@as(i32, -2), try runLoweredDiv(allocator, S, .rem, -20, 3));
    try std.testing.expectEqual(@as(i32, 2), try runLoweredDiv(allocator, S, .rem, 20, -3));
}

test "spirv: unary negate and bitwise not run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x) { return -x }  (OpSNegate)
    {
        var b = try spirv.binary.Builder.init(allocator, 8);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
        try b.emit(allocator, o.Label, &.{5});
        try b.emit(allocator, o.SNegate, &.{ 1, 6, 4 });
        try b.emit(allocator, o.ReturnValue, &.{6});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{7}, -7);
    }
    // int f(int x) { return ~x }  (OpNot, ~5 = -6)
    {
        var b = try spirv.binary.Builder.init(allocator, 8);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
        try b.emit(allocator, o.Label, &.{5});
        try b.emit(allocator, o.Not, &.{ 1, 6, 4 });
        try b.emit(allocator, o.ReturnValue, &.{6});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{5}, -6);
    }
}

test "spirv: GLSL clamp and abs (OpExtInst) run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x) { return clamp(x, 2, 8) }  (SClamp, set operand is ignored)
    // ids: int=1, fnty=2, c2=3, c8=4, f=5, x=6, entry=7, r=8.
    {
        var b = try spirv.binary.Builder.init(allocator, 9);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Constant, &.{ 1, 3, 2 });
        try b.emit(allocator, o.Constant, &.{ 1, 4, 8 });
        try b.emit(allocator, o.Function, &.{ 1, 5, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 6 });
        try b.emit(allocator, o.Label, &.{7});
        try b.emit(allocator, o.ExtInst, &.{ 1, 8, 99, o.Glsl.s_clamp, 6, 3, 4 });
        try b.emit(allocator, o.ReturnValue, &.{8});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{5}, 5); // in range
        var f2 = try spirv.lowerModule(allocator, b.words.items);
        defer f2.deinit();
        try expectRun(allocator, &f2, &.{1}, 2); // clamped up to lo
        var f3 = try spirv.lowerModule(allocator, b.words.items);
        defer f3.deinit();
        try expectRun(allocator, &f3, &.{10}, 8); // clamped down to hi
    }
    // int f(int x) { return abs(x) }  (SAbs)
    {
        var b = try spirv.binary.Builder.init(allocator, 8);
        defer b.deinit(allocator);
        try b.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
        try b.emit(allocator, o.TypeFunction, &.{ 2, 1, 1 });
        try b.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
        try b.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
        try b.emit(allocator, o.Label, &.{5});
        try b.emit(allocator, o.ExtInst, &.{ 1, 6, 99, o.Glsl.s_abs, 4 });
        try b.emit(allocator, o.ReturnValue, &.{6});
        try b.emit(allocator, o.FunctionEnd, &.{});
        var func = try spirv.lowerModule(allocator, b.words.items);
        defer func.deinit();
        try expectRun(allocator, &func, &.{-7}, 7);
        var f2 = try spirv.lowerModule(allocator, b.words.items);
        defer f2.deinit();
        try expectRun(allocator, &f2, &.{7}, 7);
    }
}

test "spirv compute: multi-member buffer struct runs natively (data[i] *= scale)" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // buffer Buf { uint scale, uint data[] }  // member 0 at offset 0, member 1 at 4
    // void main() { uint i = gid.x, data[i] = data[i] * scale }
    // ids: void=1 uint=2 v3=3 pInV3=4 pInU=5 arr=6 Buf=7 pSbBuf=8 pSbU=9 voidfn=10
    //      u0=11 u1=12 gid=13 buf=14 main=15 entry=16 xptr=17 i=18 sptr=19 scale=20
    //      eptr=21 v=22 v2=23.
    var b = try spirv.binary.Builder.init(allocator, 24);
    defer b.deinit(allocator);
    // Annotations (before the types they decorate).
    try b.emit(allocator, o.MemberDecorate, &.{ 7, 0, o.Decoration.offset, 0 }); // scale @0
    try b.emit(allocator, o.MemberDecorate, &.{ 7, 1, o.Decoration.offset, 4 }); // data @4
    try b.emit(allocator, o.Decorate, &.{ 6, o.Decoration.array_stride, 4 });
    try b.emit(allocator, o.Decorate, &.{ 13, o.Decoration.builtin, o.BuiltIn.global_invocation_id });
    // Types.
    try b.emit(allocator, o.TypeVoid, &.{1});
    try b.emit(allocator, o.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, o.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, o.TypePointer, &.{ 4, o.StorageClass.input, 3 });
    try b.emit(allocator, o.TypePointer, &.{ 5, o.StorageClass.input, 2 });
    try b.emit(allocator, o.TypeRuntimeArray, &.{ 6, 2 });
    try b.emit(allocator, o.TypeStruct, &.{ 7, 2, 6 }); // { uint scale, uint data[] }
    try b.emit(allocator, o.TypePointer, &.{ 8, o.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, o.TypePointer, &.{ 9, o.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, o.TypeFunction, &.{ 10, 1 });
    try b.emit(allocator, o.Constant, &.{ 2, 11, 0 });
    try b.emit(allocator, o.Constant, &.{ 2, 12, 1 });
    try b.emit(allocator, o.Variable, &.{ 4, 13, o.StorageClass.input });
    try b.emit(allocator, o.Variable, &.{ 8, 14, o.StorageClass.storage_buffer });
    try b.emit(allocator, o.Function, &.{ 1, 15, 0, 10 });
    try b.emit(allocator, o.Label, &.{16});
    try b.emit(allocator, o.AccessChain, &.{ 5, 17, 13, 11 }); // &gid.x
    try b.emit(allocator, o.Load, &.{ 2, 18, 17 }); // i
    try b.emit(allocator, o.AccessChain, &.{ 9, 19, 14, 11 }); // &buf.scale (member 0)
    try b.emit(allocator, o.Load, &.{ 2, 20, 19 }); // scale
    try b.emit(allocator, o.AccessChain, &.{ 9, 21, 14, 12, 18 }); // &buf.data[i] (member 1 + i)
    try b.emit(allocator, o.Load, &.{ 2, 22, 21 }); // data[i]
    try b.emit(allocator, o.IMul, &.{ 2, 23, 22, 20 }); // data[i] * scale
    try b.emit(allocator, o.Store, &.{ 21, 23 });
    try b.emit(allocator, o.Return, &.{});
    try b.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // Buffer layout: [scale=3, data[0]=10, data[1]=20, data[2]=30].
    var data = [_]i32{ 3, 10, 20, 30 };
    try runCompute(allocator, &func, 1, &data); // thread 1: data[1] = 20 * 3 = 60
    try std.testing.expectEqualSlices(i32, &.{ 3, 10, 60, 30 }, &data);

    var data2 = [_]i32{ 3, 10, 20, 30 };
    try runCompute(allocator, &func, 0, &data2); // thread 0: data[0] = 10 * 3 = 30
    try std.testing.expectEqualSlices(i32, &.{ 3, 30, 20, 30 }, &data2);
}

test "spirv vectors: construct, component-wise add, extract run natively" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x, int y) { ivec2 a = ivec2(x,y), ivec2 b = ivec2(y,x),
    //                       ivec2 c = a + b, return c.x }   // = x + y
    // ids: int=1 v2=2 fnty=3 f=4 x=5 y=6 entry=7 a=8 b=9 c=10 r=11.
    var bld = try spirv.binary.Builder.init(allocator, 12);
    defer bld.deinit(allocator);
    try bld.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
    try bld.emit(allocator, o.TypeVector, &.{ 2, 1, 2 });
    try bld.emit(allocator, o.TypeFunction, &.{ 3, 1, 1, 1 });
    try bld.emit(allocator, o.Function, &.{ 1, 4, 0, 3 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 5 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 6 });
    try bld.emit(allocator, o.Label, &.{7});
    try bld.emit(allocator, o.CompositeConstruct, &.{ 2, 8, 5, 6 }); // a = (x, y)
    try bld.emit(allocator, o.CompositeConstruct, &.{ 2, 9, 6, 5 }); // b = (y, x)
    try bld.emit(allocator, o.IAdd, &.{ 2, 10, 8, 9 }); // c = a + b (component-wise)
    try bld.emit(allocator, o.CompositeExtract, &.{ 1, 11, 10, 0 }); // c.x
    try bld.emit(allocator, o.ReturnValue, &.{11});
    try bld.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, bld.words.items);
    defer func.deinit();
    try expectRun(allocator, &func, &.{ 3, 4 }, 7); // c.x = x + y = 7
}

test "spirv vectors: VectorShuffle swaps components" {
    const allocator = std.testing.allocator;
    const spirv = @import("vulcan-spirv");
    const o = spirv.opcodes;

    // int f(int x, int y) { ivec2 a = ivec2(x,y), ivec2 b = a.yx, return b.x }  // = y
    var bld = try spirv.binary.Builder.init(allocator, 12);
    defer bld.deinit(allocator);
    try bld.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
    try bld.emit(allocator, o.TypeVector, &.{ 2, 1, 2 });
    try bld.emit(allocator, o.TypeFunction, &.{ 3, 1, 1, 1 });
    try bld.emit(allocator, o.Function, &.{ 1, 4, 0, 3 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 5 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 6 });
    try bld.emit(allocator, o.Label, &.{7});
    try bld.emit(allocator, o.CompositeConstruct, &.{ 2, 8, 5, 6 }); // a = (x, y)
    try bld.emit(allocator, o.VectorShuffle, &.{ 2, 9, 8, 8, 1, 0 }); // b = a.yx = (y, x)
    try bld.emit(allocator, o.CompositeExtract, &.{ 1, 10, 9, 0 }); // b.x
    try bld.emit(allocator, o.ReturnValue, &.{10});
    try bld.emit(allocator, o.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, bld.words.items);
    defer func.deinit();
    try expectRun(allocator, &func, &.{ 3, 4 }, 4); // b.x = y = 4
}

test "uefi: IR -> aarch64 -> PE32+ image, and the embedded code runs" {
    const allocator = std.testing.allocator;
    const pe = @import("../../pe.zig");

    // A trivial efi_main-like function: returns 42 (a constant EFI_STATUS).
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, t, .{ .iconst = 42 });
    func.setTerminator(b, .{ .ret = c });

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "efi_main", &func);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);

    const entry = linked.addressOf("efi_main").?;
    const img = try pe.writeUefiImage(allocator, std.mem.sliceAsBytes(linked.code), linked.code.len * 4, entry, .aarch64);
    defer allocator.free(img);

    // A well-formed UEFI image: MZ / PE / AArch64 machine / EFI_APPLICATION subsystem.
    try std.testing.expectEqual(@as(u8, 'M'), img[0]);
    const lfanew = std.mem.readInt(u32, img[0x3c..0x40], .little);
    try std.testing.expectEqual(@as(u16, 0xAA64), std.mem.readInt(u16, img[lfanew + 4 ..][0..2], .little));
    const opt_hdr = img[lfanew + 4 + 20 ..];
    try std.testing.expectEqual(@as(u16, 10), std.mem.readInt(u16, opt_hdr[68..70], .little)); // EFI_APPLICATION

    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // The .text bytes are the real function: extract from PointerToRawData and run.
    const sect = img[lfanew + 4 + 20 + 240 ..];
    const praw = std.mem.readInt(u32, sect[20..24], .little);
    const text = img[praw..][0 .. linked.code.len * 4];
    var buf = try jit.CodeBuffer.map(text);
    defer buf.deinit();
    const fp = buf.entry(*const fn () callconv(.c) i32, entry);
    try std.testing.expectEqual(@as(i32, 42), fp());
}

test "object+ld+exec: link two functions into a runnable ELF and execute it natively" {
    const allocator = std.testing.allocator;
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // executes the AArch64 ELF directly
    const object = @import("../object.zig");
    const ld = @import("../ld.zig");
    const encode = @import("../encode.zig");
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a, main(x) = dbl(x) + 2.  main(20) = 42.
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(i32k);
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const d = try main.appendCall(b, t, "dbl", &.{x});
        const r = try main.appendArithImm(b, t, .add, d, 2);
        main.setTerminator(b, .{ .ret = r });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "dbl", &dbl);
    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    const base: u64 = 0x400000;
    var image = try ld.linkObjects(allocator, &.{obj}, base);
    defer image.deinit(allocator);

    // A tiny entry stub: set the argument, call main, then exit with its result. main sits
    // right past the 16-byte stub. bl is at offset 4.
    const main_off: i64 = @intCast(image.addressOf("main").? - base);
    const stub = [_]u32{
        encode.movz(.x0, 20, 0), // x0 = 20
        encode.bl(@intCast((16 + main_off) - 4)), // bl main (it sits past the 16-byte stub)
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(allocator, program.items, program.items.len, base, base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });
    const proc = std.process.run(allocator, std.testing.io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound, error.AccessDenied => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // dbl(20) + 2
        else => return error.BackendFailed,
    }
}
