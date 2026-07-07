//! RISC-V 64-bit target: encoding, registers, calling convention, and codegen.
//! The first-class Vulcan backend.

const std = @import("std");

pub const encode = @import("riscv64/encode.zig");
pub const disasm = @import("riscv64/disasm.zig");
pub const isel = @import("riscv64/isel.zig");
pub const emit = @import("riscv64/emit.zig");
pub const schedule = @import("riscv64/schedule.zig");
pub const link = @import("riscv64/link.zig");
pub const object = @import("riscv64/object.zig");
pub const ld = @import("riscv64/ld.zig");
pub const jit = @import("riscv64/jit.zig");
pub const compress = @import("riscv64/compress.zig");

/// Enabled RISC-V ISA extensions for a codegen target. Codegen consults this to
/// decide optional output transforms; today the only one wired is `c` (RVC), which
/// makes `emitImage` compress the resolved word stream to mixed 16/32-bit RV64GC.
pub const Features = struct {
    /// M: integer multiply/divide.
    m: bool = true,
    /// F: single-precision float.
    f: bool = true,
    /// D: double-precision float.
    d: bool = true,
    /// C: compressed 16-bit instructions.
    c: bool = false,
    /// V: vectors (RVV).
    v: bool = false,

    /// RV64GC: the common general-purpose profile (IMAFD + C).
    pub const rv64gc: Features = .{ .c = true };
    /// RV64G: general-purpose without compression (fixed 32-bit).
    pub const rv64g: Features = .{};
};

/// Turn a resolved 32-bit word stream into machine-code bytes for the target. When
/// the C extension is enabled this compresses eligible instructions to their 2-byte
/// RVC forms; otherwise it emits the words unchanged. This is the single seam the
/// codegen pipeline uses, so RVC is applied automatically from the target's features
/// rather than bolted on at each call site. Caller owns the result.
pub fn emitImage(allocator: std.mem.Allocator, words: []const u32, features: Features) std.mem.Allocator.Error![]u8 {
    return if (features.c) compress.compress(allocator, words) else emit.emitBytes(allocator, words);
}

test "emitImage leaves the stream 32-bit when C is disabled" {
    const a = std.testing.allocator;
    const words = [_]u32{ encode.addi(.x10, .x0, 5), encode.jalr(.x0, .x1, 0) };
    const bytes = try emitImage(a, &words, Features.rv64g);
    defer a.free(bytes);
    // Two full 32-bit words: no compression happened.
    try std.testing.expectEqual(@as(usize, 8), bytes.len);
    const plain = try emit.emitBytes(a, &words);
    defer a.free(plain);
    try std.testing.expectEqualSlices(u8, plain, bytes);
}

test "emitImage compresses to RVC when C is enabled" {
    const a = std.testing.allocator;
    const words = [_]u32{ encode.addi(.x10, .x0, 5), encode.jalr(.x0, .x1, 0) };
    const bytes = try emitImage(a, &words, Features.rv64gc);
    defer a.free(bytes);
    // Both compress: c.li (2) + ret (2) = 4 bytes, half the fixed-width size.
    try std.testing.expectEqual(@as(usize, 4), bytes.len);
    // And it still disassembles to the same instructions.
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "li x10, 5") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "ret") != null);
}

/// Execution-test runners, one per backend (see each file). Each skips when its
/// backend's tool is unavailable or its machine is incompatible.
const tests = struct {
    pub const harness = @import("riscv64/tests/harness.zig");
    pub const river = @import("riscv64/tests/river.zig");
    pub const spike = @import("riscv64/tests/spike.zig");
    pub const qemu = @import("riscv64/tests/qemu.zig");
    pub const qemu_user = @import("riscv64/tests/qemu_user.zig");
    pub const qemu_user_rvc = @import("riscv64/tests/qemu_user_rvc.zig");
    pub const compressed = @import("riscv64/tests/compressed.zig");
    pub const native = @import("riscv64/tests/native.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
