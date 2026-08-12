//! Per-target C integer layout.
//! The `int`, `char`, `short`, and `long long` widths are fixed for every Vulcan target.
//! Only the `long` and pointer width, and the `char` signedness, change (LP64 or ILP32).

const std = @import("std");
const builtin = @import("builtin");

/// The architecture a `TargetLayout` targets.
/// The fields `long_bits`, `ptr_bits`, and `char_signed` alone cannot tell aarch64 and
/// riscv64 apart, because both are LP64 with an unsigned plain `char`.
/// `ctype.builtinVaList` needs the architecture itself, because its per-target
/// `__builtin_va_list` shape is an ABI fact. It is not derivable from the width and
/// signedness alone. This is the one place that fact is carried. Every other
/// `TargetLayout` consumer ignores it.
pub const Arch = enum { aarch64, riscv64, x86_64, x86 };

/// The bit widths and char signedness that vary by target.
/// `int` is 32 bits, `char` is 8 bits, `short` is 16 bits, and `long long` is 64 bits
/// on every target. This struct does not store those fixed widths.
pub const TargetLayout = struct {
    long_bits: u16,
    ptr_bits: u16,
    char_signed: bool,
    /// Defaults to `.x86_64` so a hand-written `TargetLayout{...}` literal still compiles
    /// without naming this field. The default matters only for tests that do not care
    /// about `builtinVaList`, such as this file's own tests and some `ctype.zig` and
    /// `parser.zig` tests. Every real caller gets a `TargetLayout` through `host()` or
    /// `forTarget()` below, which always set this field.
    arch: Arch = .x86_64,
};

/// The layout that matches the build host. The host-JIT differential path uses this
/// layout so it agrees with the host gcc. aarch64, riscv64, and x86_64 are LP64.
/// `char` is unsigned on arm and signed on x86.
pub fn host() TargetLayout {
    return switch (builtin.cpu.arch) {
        .aarch64 => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .aarch64 },
        .riscv64 => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .riscv64 },
        .x86_64 => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = true, .arch = .x86_64 },
        .x86 => .{ .long_bits = 32, .ptr_bits = 32, .char_signed = true, .arch = .x86 },
        else => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .aarch64 },
    };
}

/// Layout for a known `Arch` directly. Some cross-arch tests already hold each arch as
/// an `Arch` value, so this function saves them from building a triple string only to
/// route it back through `forTarget`'s prefix match. It reports the same width and
/// signedness facts as `forTarget`.
pub fn forArch(arch: Arch) TargetLayout {
    return switch (arch) {
        .x86_64 => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = true, .arch = .x86_64 },
        .aarch64 => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .aarch64 },
        .riscv64 => .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .riscv64 },
        .x86 => .{ .long_bits = 32, .ptr_bits = 32, .char_signed = true, .arch = .x86 },
    };
}

/// Layout for an explicit `-target` triple prefix. Falls back to `host()` when the
/// triple is null or not recognized.
pub fn forTarget(triple: ?[]const u8) TargetLayout {
    const t = triple orelse return host();
    if (std.mem.startsWith(u8, t, "x86_64")) return .{ .long_bits = 64, .ptr_bits = 64, .char_signed = true, .arch = .x86_64 };
    if (std.mem.startsWith(u8, t, "aarch64")) return .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .aarch64 };
    if (std.mem.startsWith(u8, t, "riscv64")) return .{ .long_bits = 64, .ptr_bits = 64, .char_signed = false, .arch = .riscv64 };
    if (std.mem.startsWith(u8, t, "i386") or std.mem.startsWith(u8, t, "i686") or std.mem.startsWith(u8, t, "x86")) return .{ .long_bits = 32, .ptr_bits = 32, .char_signed = true, .arch = .x86 };
    return host();
}
