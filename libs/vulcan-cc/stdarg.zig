//! `<stdarg.h>`: the ONLY standard header this frontend serves ITSELF.
//! `__builtin_va_list`/`__builtin_va_start`/etc. are names this compiler alone gives
//! meaning to (see `ctype.builtinVaList`). A host's own installed `<stdarg.h>` would
//! define them differently, or not at all, so this header's bytes must win regardless of
//! what's on the host's include path. A driver wires this in by chaining `resolve` below
//! AHEAD of its disk resolver. Every OTHER standard header still comes from
//! the host / `-I` search path unchanged.

const std = @import("std");

/// `<stdarg.h>`'s full text, following gcc's own header protocol so real glibc headers work.
///
/// glibc's `<stdio.h>` (and other headers) do `#define __need___va_list` then `#include
/// <stdarg.h>`, expecting ONLY the internal `__gnuc_va_list` typedef back (NOT `va_list` or the
/// `va_*` macros). glibc then writes `typedef __gnuc_va_list va_list;` itself. This built-in
/// header wins over gcc's on-disk copy, so it must play gcc's role fully. It mirrors gcc:
///   * `__gnuc_va_list` aliases `__builtin_va_list` (`ctype.builtinVaList`) and is ALWAYS defined,
///     guarded by `__GNUC_VA_LIST` so a re-include never redefines it.
///   * In `__need___va_list` mode it stops there (and `#undef`s that request flag).
///   * Otherwise (a plain `#include <stdarg.h>`) it ALSO gives the user `va_list` (from
///     `__gnuc_va_list`) plus the `va_start`/`va_arg`/`va_end`/`va_copy` macros the lowering
///     code recognizes, guarded by `_VCC_STDARG_H` so re-inclusion is a no-op.
pub const bytes =
    "#ifndef __GNUC_VA_LIST\n" ++
    "#define __GNUC_VA_LIST\n" ++
    "typedef __builtin_va_list __gnuc_va_list;\n" ++
    "#endif\n" ++
    "#ifdef __need___va_list\n" ++
    "#undef __need___va_list\n" ++
    "#else\n" ++
    "#ifndef _VCC_STDARG_H\n" ++
    "#define _VCC_STDARG_H\n" ++
    "typedef __gnuc_va_list va_list;\n" ++
    "#define va_start(ap,last) __builtin_va_start(ap,last)\n" ++
    "#define va_arg(ap,ty) __builtin_va_arg(ap,ty)\n" ++
    "#define va_end(ap) __builtin_va_end(ap)\n" ++
    "#define va_copy(d,s) __builtin_va_copy(d,s)\n" ++
    "#endif\n" ++
    "#endif\n";

/// Resolve an `#include` target against this built-in header: `bytes` for a SYSTEM
/// (`<...>`) include of exactly `"stdarg.h"`, `null` for anything else (a `"..."`-quoted
/// include, or any other name) so a caller chaining this ahead of a disk resolver falls
/// through unchanged.
pub fn resolve(name: []const u8, is_system: bool) ?[]const u8 {
    if (is_system and std.mem.eql(u8, name, "stdarg.h")) return bytes;
    return null;
}

test "resolve serves stdarg.h bytes only for a system include of exactly that name" {
    try std.testing.expectEqualStrings(bytes, resolve("stdarg.h", true).?);
    try std.testing.expectEqual(@as(?[]const u8, null), resolve("stdarg.h", false)); // quoted, not system
    try std.testing.expectEqual(@as(?[]const u8, null), resolve("stdio.h", true)); // different name
}
