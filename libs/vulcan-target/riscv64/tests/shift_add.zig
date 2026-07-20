//! Zba sh-add fusion (Task 6), executed on qemu-riscv64 with Zba enabled (the oracle). Builds
//! `x + (b << k)` for k in {1, 2, 3} - the shl immediately preceding its single-use add, exactly the
//! shape `fusesIntoNextShiftAdd` (isel.zig) folds into one `sh{k}add rd, b, x` - and checks that the
//! qemu-executed result is correct AND that the fused instruction actually fired (the `sh{k}add` is
//! present and the standalone `slli` is gone). The fold is gated on `caps.fuse_shift_add`, TRUE only
//! for a Zba model: with the default caps (fuse_shift_add = false) the SAME function compiles to the
//! plain `slli`+`add` path, proving strict gating.
//!
//! sh1add/sh2add/sh3add are Zba instructions, so the fold tests run under `-cpu rv64,zba=true`
//! (harness.qemu_user_zba). A CPU without Zba would reject them as illegal.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// Caps with ONLY the Zba sh-add fold turned on (every other flag at its no-extension default). A
/// Zba model would set this via `capsForModel`; a test flips it in isolation to compile the SAME
/// function both folded and not.
const zba_caps = isel.ModelCaps{ .fuse_shift_add = true };

/// `f(x, b) = x + (b << k)` (or `(b << k) + x` when `shl_on_lhs`), all `bits`-wide signed integers,
/// with the shl immediately preceding its single consuming add - exactly the shape
/// `fusesIntoNextShiftAdd` folds into one `sh{k}add`.
fn buildShiftAdd(allocator: std.mem.Allocator, bits: u16, k: i64, shl_on_lhs: bool) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = bits } });
    const blk = try func.appendBlock();
    const x = try func.appendBlockParam(blk, ty);
    const b = try func.appendBlockParam(blk, ty);
    const shl = try func.appendInst(blk, ty, .{ .arith_imm = .{ .op = .shl, .lhs = b, .imm = k } });
    const add = if (shl_on_lhs)
        try func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = shl, .rhs = x } })
    else
        try func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = x, .rhs = shl } });
    func.setTerminator(blk, .{ .ret = add });
    return func;
}

/// The `sh{k}add` mnemonic for shift amount `k`.
fn mnemonicFor(k: i64) []const u8 {
    return switch (k) {
        1 => "sh1add",
        2 => "sh2add",
        3 => "sh3add",
        else => unreachable,
    };
}

/// Compile `func` with `caps` and return the disassembly text (caller frees).
fn disasmWithCaps(allocator: std.mem.Allocator, func: *Function, caps: isel.ModelCaps) ![]u8 {
    var words = try harness.compileFuncWithCaps(allocator, func, caps);
    defer words.deinit(allocator);
    return disasm.format(allocator, words.items);
}

/// Compile `func` with `caps`, run it under qemu (Zba enabled) with `args`, and return the raw a0.
fn runWithCaps(allocator: std.mem.Allocator, func: *Function, caps: isel.ModelCaps, args: []const i64) !i64 {
    var words = try harness.compileFuncWithCaps(allocator, func, caps);
    defer words.deinit(allocator);
    return harness.runCode(std.testing.io, allocator, words.items, args, harness.qemu_user_zba);
}

/// The full Zba fold check for a given `k` and operand order: the compiled code contains the fused
/// `sh{k}add` and NO standalone `slli` (proving the fold fired), and the qemu-executed result equals
/// `x + (b << k)`.
fn checkFold(k: i64, shl_on_lhs: bool) !void {
    const allocator = std.testing.allocator;
    const x: i64 = 100;
    const b: i64 = 7;

    var func = try buildShiftAdd(allocator, 64, k, shl_on_lhs);
    defer func.deinit();

    const text = try disasmWithCaps(allocator, &func, zba_caps);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, mnemonicFor(k)) != null); // the fused sh-add fired
    try std.testing.expect(std.mem.indexOf(u8, text, "slli") == null); // the standalone shift is gone

    const got = runWithCaps(allocator, &func, zba_caps, &.{ x, b }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest, // qemu not installed
        else => return e,
    };
    try std.testing.expectEqual(x + (b << @as(u6, @intCast(k))), got);
}

test "riscv64 shift_add: x + (b<<1) folds to sh1add under a Zba model and computes correctly (qemu-riscv64)" {
    try checkFold(1, false);
}

test "riscv64 shift_add: x + (b<<2) folds to sh2add under a Zba model and computes correctly (qemu-riscv64)" {
    try checkFold(2, false);
}

test "riscv64 shift_add: x + (b<<3) folds to sh3add under a Zba model and computes correctly (qemu-riscv64)" {
    try checkFold(3, false);
}

test "riscv64 shift_add: (b<<2) + x also folds (commutative) (qemu-riscv64)" {
    try checkFold(2, true);
}

test "riscv64 shift_add does NOT fold without Zba (default caps): plain slli+add, correct (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const x: i64 = 100;
    const b: i64 = 7;

    var func = try buildShiftAdd(allocator, 64, 2, false);
    defer func.deinit();

    // Default caps: fuse_shift_add is false, so the fold is declined - the shift materializes.
    const text = try disasmWithCaps(allocator, &func, .{});
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "slli") != null); // the shift is still materialized
    try std.testing.expect(std.mem.indexOf(u8, text, "add") != null); // and added separately
    try std.testing.expect(std.mem.indexOf(u8, text, "sh2add") == null); // no fused sh-add

    const got = runWithCaps(allocator, &func, .{}, &.{ x, b }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(x + (b << 2), got);
}

test "riscv64 shift_add: a multi-use shift does NOT fold (plain slli+add, correct, qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const x: i64 = 100;
    const b: i64 = 7;

    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const blk = try func.appendBlock();
    const px = try func.appendBlockParam(blk, ty);
    const pb = try func.appendBlockParam(blk, ty);
    const shl = try func.appendInst(blk, ty, .{ .arith_imm = .{ .op = .shl, .lhs = pb, .imm = 2 } });
    const r1 = try func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = px, .rhs = shl } }); // fusible shape...
    const r2 = try func.appendInst(blk, ty, .{ .arith = .{ .op = .bit_or, .lhs = shl, .rhs = px } }); // ...but shl is reused here
    const ret = try func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = r1, .rhs = r2 } });
    func.setTerminator(blk, .{ .ret = ret });

    // Even with Zba enabled, the shl has two uses, so the fold is declined and the shift stays.
    const text = try disasmWithCaps(allocator, &func, zba_caps);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "slli") != null); // still materialized
    try std.testing.expect(std.mem.indexOf(u8, text, "sh2add") == null); // not fused

    const got = runWithCaps(allocator, &func, zba_caps, &.{ x, b }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    const shifted = b << 2;
    try std.testing.expectEqual((x + shifted) + (shifted | x), got);
}

test "riscv64 shift_add: shift by 4 does NOT fold (sh-add only supports 1/2/3, qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const x: i64 = 100;
    const b: i64 = 7;

    var func = try buildShiftAdd(allocator, 64, 4, false);
    defer func.deinit();

    // k = 4 is outside the sh{1,2,3}add range, so the fold is declined even with Zba on.
    const text = try disasmWithCaps(allocator, &func, zba_caps);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "slli") != null); // materialized as a plain shift
    try std.testing.expect(std.mem.indexOf(u8, text, "sh1add") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sh2add") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sh3add") == null);

    const got = runWithCaps(allocator, &func, zba_caps, &.{ x, b }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(x + (b << 4), got);
}

test "riscv64 shift_add: a 32-bit add does NOT fold (needs sh-add.uw, deferred; qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const x: i64 = 100;
    const b: i64 = 7;

    var func = try buildShiftAdd(allocator, 32, 2, false);
    defer func.deinit();

    // sh{k}add produces a 64-bit sum; a 32-bit result would need sh{k}add.uw, so the fold is declined.
    const text = try disasmWithCaps(allocator, &func, zba_caps);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "sh2add") == null); // 32-bit add is not folded

    const got = runWithCaps(allocator, &func, zba_caps, &.{ x, b }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    // Compare only the low 32 bits: a 32-bit result's upper bits are don't-care in the 64-bit a0.
    try std.testing.expectEqual(@as(i64, x + (b << 2)), got & 0xffffffff);
}
