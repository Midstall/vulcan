//! `<stddef.h>`: this compiler's own built-in copy, mirroring
//! `stdarg.zig`'s pattern. It provides a `pub const bytes` header text plus a `resolve`
//! helper that a host-side resolver chains ahead of its disk search. Unlike `<stdarg.h>`, the
//! names this header defines (`size_t`, `ptrdiff_t`, `wchar_t`, `NULL`, `offsetof`) are not
//! this compiler's own inventions. A host's installed `<stddef.h>` would work just as well,
//! but serving it ourselves means it always exists, even with no `-isystem` path
//! configured. Its `typedef`s are spelled straight off this compiler's own predefined
//! `__SIZE_TYPE__`/`__PTRDIFF_TYPE__`/`__WCHAR_TYPE__` macros (`preproc.SystemPredef`,
//! `Options.system`), so they always match this compiler's own type sizes.
//!
//! `offsetof` expands to `__builtin_offsetof(t, m)`, the GCC/Clang builtin spelling, not
//! the `((size_t)&((t*)0)->m)` alternative that C also allows. This header only handles the
//! preprocessor stage, so `offsetof(t, m)` only needs to EXPAND correctly here, not parse.
//! `__builtin_offsetof` is the better choice. It is the name a later stage can give real
//! meaning to directly, without this header needing a revisit.

const std = @import("std");

/// `<stddef.h>`'s full text.
pub const bytes =
    "typedef __SIZE_TYPE__ size_t;\n" ++
    "typedef __PTRDIFF_TYPE__ ptrdiff_t;\n" ++
    "typedef __WCHAR_TYPE__ wchar_t;\n" ++
    "#define NULL ((void*)0)\n" ++
    "#define offsetof(t, m) __builtin_offsetof(t, m)\n";

/// Resolve an `#include` target against this built-in header: `bytes` for a SYSTEM
/// (`<...>`) include of exactly `"stddef.h"`, `null` for anything else (a `"..."`-quoted
/// include, or any other name) so a caller chaining this ahead of a disk resolver falls
/// through unchanged.
pub fn resolve(name: []const u8, is_system: bool) ?[]const u8 {
    if (is_system and std.mem.eql(u8, name, "stddef.h")) return bytes;
    return null;
}

test "resolve serves stddef.h bytes only for a system include of exactly that name" {
    try std.testing.expectEqualStrings(bytes, resolve("stddef.h", true).?);
    try std.testing.expectEqual(@as(?[]const u8, null), resolve("stddef.h", false)); // quoted, not system
    try std.testing.expectEqual(@as(?[]const u8, null), resolve("stdarg.h", true)); // different name
}
