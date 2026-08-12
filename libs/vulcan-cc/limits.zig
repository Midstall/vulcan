//! `<limits.h>`: this compiler's own built-in copy, the twin of `stddef.zig`. A real GNU
//! libc `<limits.h>` does not define the ISO integer limits itself. It defers them to THE
//! COMPILER's `<limits.h>` through `#include_next <limits.h>`, guarded by
//! `#if defined __GNUC__ && !defined _GCC_LIMITS_H_`. VCC defines `__GNUC__`, so that guard
//! fires, and VCC must supply the compiler-level header the guard reaches for. This file is it.
//!
//! Served FIRST for a system `<limits.h>` (ahead of the disk glibc header), exactly as GCC's
//! own include directory sits ahead of `/usr/include`. It defines `_GCC_LIMITS_H_` plus every
//! ISO integer limit, then `#include_next <limits.h>` to layer glibc's POSIX extensions
//! (`PATH_MAX`, `MB_LEN_MAX`, ...) on top. When glibc's header runs, `_GCC_LIMITS_H_` is
//! already defined, so its bottom guard does not bounce back here and its ISO fallback block
//! stays inert. The limits are spelled off the `__*_MAX__` macros VCC predefines per target
//! (`preproc.seedSystemPredefs`), so a 32-bit-`long` target gets the right `LONG_MAX`.
//!
//! CHAR_MIN/CHAR_MAX follow `__CHAR_UNSIGNED__` (predefined on an unsigned-`char` target),
//! matching GCC. `MB_LEN_MAX` is left to glibc.

const std = @import("std");

/// `<limits.h>`'s full text.
pub const bytes =
    "#ifndef _GCC_LIMITS_H_\n" ++
    "#define _GCC_LIMITS_H_ 1\n" ++
    "#define CHAR_BIT __CHAR_BIT__\n" ++
    "#define SCHAR_MAX __SCHAR_MAX__\n" ++
    "#define SCHAR_MIN (-SCHAR_MAX - 1)\n" ++
    "#define UCHAR_MAX (SCHAR_MAX * 2 + 1)\n" ++
    "#ifdef __CHAR_UNSIGNED__\n" ++
    "#define CHAR_MIN 0\n" ++
    "#define CHAR_MAX UCHAR_MAX\n" ++
    "#else\n" ++
    "#define CHAR_MIN SCHAR_MIN\n" ++
    "#define CHAR_MAX SCHAR_MAX\n" ++
    "#endif\n" ++
    "#define SHRT_MAX __SHRT_MAX__\n" ++
    "#define SHRT_MIN (-SHRT_MAX - 1)\n" ++
    "#define USHRT_MAX (SHRT_MAX * 2 + 1)\n" ++
    "#define INT_MAX __INT_MAX__\n" ++
    "#define INT_MIN (-INT_MAX - 1)\n" ++
    "#define UINT_MAX (INT_MAX * 2U + 1U)\n" ++
    "#define LONG_MAX __LONG_MAX__\n" ++
    "#define LONG_MIN (-LONG_MAX - 1L)\n" ++
    "#define ULONG_MAX (LONG_MAX * 2UL + 1UL)\n" ++
    "#define LLONG_MAX __LONG_LONG_MAX__\n" ++
    "#define LLONG_MIN (-LLONG_MAX - 1LL)\n" ++
    "#define ULLONG_MAX (LLONG_MAX * 2ULL + 1ULL)\n" ++
    "#endif\n" ++
    "#include_next <limits.h>\n";

/// Resolve an `#include` target against this built-in header: `bytes` for a SYSTEM (`<...>`)
/// include of exactly `"limits.h"`, `null` for anything else, so a caller chaining this ahead
/// of a disk resolver falls through unchanged. Mirrors `stddef.resolve`.
pub fn resolve(name: []const u8, is_system: bool) ?[]const u8 {
    if (is_system and std.mem.eql(u8, name, "limits.h")) return bytes;
    return null;
}

test "resolve serves limits.h bytes only for a system include of exactly that name" {
    try std.testing.expectEqualStrings(bytes, resolve("limits.h", true).?);
    try std.testing.expectEqual(@as(?[]const u8, null), resolve("limits.h", false)); // quoted, not system
    try std.testing.expectEqual(@as(?[]const u8, null), resolve("stddef.h", true)); // different name
}
