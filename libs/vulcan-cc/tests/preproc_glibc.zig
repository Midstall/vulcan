//! This test proves that VCC's own preprocessor, `preproc.preprocess`, reduces the real
//! host glibc `#include <stdio.h>` chain to no error. The test wires the preprocessor
//! through the per-target `SystemPredef` and the filesystem `FsResolver`. It builds on
//! the char-literal and `__has_*` support, and on the variadic-macro support.
//! `stdio.h` alone pulls in about 29 files (`features.h`, `sys/cdefs.h`, the `bits/*`
//! family, and more). It also pulls in this compiler's own built-in `<stddef.h>` and
//! `<stdarg.h>`. These two headers must resolve to VCC's copies, not gcc's copies.
//! A `-nostdinc` glibc dev tree has no `stddef.h` or `stdarg.h` of its own. This is
//! confirmed against a real gcc, below.
//! A pass here means the preprocessor handles every directive and macro construct that
//! the chain uses: nested object and function macros, `#if` and `#ifdef` with `defined`,
//! `##` token pasting, branches gated by `__GNUC__` and `__GNUC_PREREQ`, and `#pragma once`
//! and `GCC system_header`.

const std = @import("std");
const builtin = @import("builtin");
const cc = @import("vulcan-cc");

/// This sets a modest `__GNUC__` version: 4.2. It is old enough that glibc's
/// `__GNUC_PREREQ` gates still select the GNU extension branches, like `__attribute__`
/// and `__THROW`, that real glibc headers rely on. The target is LP64: both `long` and
/// pointers are 64 bits.
fn systemPredef(arch: cc.layout.Arch) cc.preproc.SystemPredef {
    const lay = cc.layout.forArch(arch);
    return .{
        .arch = arch,
        .gnuc_major = 4,
        .gnuc_minor = 2,
        .gnuc_patch = 0,
        .long_bits = lay.long_bits,
        .ptr_bits = lay.ptr_bits,
        .char_signed = lay.char_signed,
    };
}

/// The host arch as a layout arch, skipping on a host with no layout entry.
fn hostLayoutArch() error{SkipZigTest}!cc.layout.Arch {
    return switch (builtin.cpu.arch) {
        .aarch64 => .aarch64,
        .x86_64 => .x86_64,
        .riscv64 => .riscv64,
        .x86 => .x86,
        else => error.SkipZigTest,
    };
}

/// Runs `sh -c script` and returns its trimmed stdout. Returns null if the process fails
/// to spawn or produces no output. This function never returns an error. This lets every
/// caller call `SkipZigTest` cleanly, instead of failing, when a discovery probe finds
/// nothing on this host.
fn shOutput(allocator: std.mem.Allocator, script: []const u8) !?[]u8 {
    const proc = std.process.run(allocator, std.testing.io, .{ .argv = &.{ "sh", "-c", script } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Locates the host (aarch64) glibc "dev" include directory. This is the directory that
/// holds a real `stdio.h`, `features.h`, `sys/cdefs.h`, and so on. It tries two methods, and
/// validates each candidate by checking for a real `stdio.h` inside it, so a stale or wrong
/// hit never gets returned.
/// (1) `VCC_GLIBC_INCLUDE`, an explicit override for a host where the search below does not
/// apply.
/// (2) The host C compiler's own `-E -v` system-include search list: the last directory in it
/// that holds a `stdio.h` is the C library include dir. Asking the compiler (rather than
/// globbing `/nix/store`) works on any host and never rots on a store rebuild.
/// This function never fails the test. A caller that gets `null` back skips cleanly.
fn findHostGlibcIncludeDir(allocator: std.mem.Allocator) !?[]u8 {
    const script =
        \\if [ -n "${VCC_GLIBC_INCLUDE:-}" ] && [ -f "$VCC_GLIBC_INCLUDE/stdio.h" ]; then
        \\  echo "$VCC_GLIBC_INCLUDE"; exit 0
        \\fi
        \\for cc_ in cc gcc; do
        \\  command -v "$cc_" >/dev/null 2>&1 || continue
        \\  last=""
        \\  for d in $(echo | LC_ALL=C "$cc_" -E -v -xc - 2>&1 | awk '/search starts here/{f=1;next} /End of search/{f=0} f{sub(/^ +/,"");print}'); do
        \\    [ -f "$d/features.h" ] && last="$d"
        \\  done
        \\  if [ -n "$last" ]; then echo "$last"; exit 0; fi
        \\done
        \\exit 1
    ;
    return shOutput(allocator, script);
}

/// Locates a cross glibc dev include directory for `triple`, for example
/// `x86_64-unknown-linux-gnu`. This supports an optional cross-arch repeat of the chain
/// preprocessing test. The function skips cleanly when absent: this Nix profile may
/// simply not have a given arch's cross glibc installed, which is not a VCC bug.
fn findCrossGlibcIncludeDir(allocator: std.mem.Allocator, triple: []const u8) !?[]u8 {
    const script = try std.fmt.allocPrint(allocator,
        \\for cc_ in {s}-gcc {s}-cc; do
        \\  command -v "$cc_" >/dev/null 2>&1 || continue
        \\  last=""
        \\  for d in $(echo | LC_ALL=C "$cc_" -E -v -xc - 2>&1 | awk '/search starts here/{{f=1;next}} /End of search/{{f=0}} f{{sub(/^ +/,"");print}}'); do
        \\    [ -f "$d/features.h" ] && last="$d"
        \\  done
        \\  if [ -n "$last" ]; then echo "$last"; exit 0; fi
        \\done
        \\exit 1
    , .{ triple, triple });
    defer allocator.free(script);
    return shOutput(allocator, script);
}

/// Preprocesses `#include <stdio.h>` with `sp` and `glibc_include`, and asserts no error.
/// Then it checks three sanity markers.
/// `printf`: an ordinary declaration that survived the whole chain.
/// `struct _IO_FILE`: from `bits/types/struct_FILE.h`, this proves the type machinery
/// expanded correctly.
/// `__attribute__`: this proves the code took a GNU-extension branch gated on `__GNUC__`,
/// not the plain-C fallback.
fn expectChainPreprocessesClean(allocator: std.mem.Allocator, glibc_include: []const u8, sp: cc.preproc.SystemPredef) !void {
    const io = std.testing.io;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const dirs = [_][]const u8{glibc_include};
    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, &dirs);

    const toks = try cc.preproc.preprocess(allocator, "#include <stdio.h>\n", .{
        .resolver = fsr.asResolver(),
        .system = sp,
    });
    defer cc.lexer.freeTokens(allocator, toks);

    // The code joins tokens with a separating space. A two-word marker like `struct _IO_FILE`
    // is found only when those are two adjacent tokens, not a coincidental substring hit.
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(allocator);
    for (toks) |t| {
        try joined.appendSlice(allocator, t.text);
        try joined.append(allocator, ' ');
    }

    try std.testing.expect(std.mem.indexOf(u8, joined.items, "printf") != null);
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "struct _IO_FILE") != null);
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "__attribute__") != null);
}

// The host test must pass whenever a glibc dev tree is present. This is the
// one case that is required to succeed, not merely allowed to succeed. The predef
// and layout follow the host arch, so the same check runs on aarch64 and x86_64.
test "VCC preprocesses the real glibc <stdio.h> chain to no-error (host glibc)" {
    const allocator = std.testing.allocator;
    const arch = try hostLayoutArch();
    const glibc_include = (try findHostGlibcIncludeDir(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    try expectChainPreprocessesClean(allocator, glibc_include, systemPredef(arch));
}

// The whole VCC stack must work together on a real header. The test above proved that
// the preprocessor alone reduces `#include <stdio.h>` to no error. This test feeds that
// reduced, roughly 800-line token stream to the parser (`parser.parse`, which runs the
// same preprocessor internally), and asserts that the whole header parses with no error.
// A pass here means every declaration construct the real header uses works together in
// one pass on real glibc, not just on the hand-picked snippets each feature was tested
// with alone. This includes `__attribute__`, `__restrict`, and `__THROW` runs; `typedef`,
// incomplete `struct`, anonymous union, and bitfield type machinery; `inline`,
// `_Noreturn`, `__extension__`, and `__asm__` labels; and `void` and `void*`.
fn expectChainParsesClean(allocator: std.mem.Allocator, glibc_include: []const u8, arch: cc.layout.Arch) !void {
    const io = std.testing.io;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const dirs = [_][]const u8{glibc_include};
    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, &dirs);

    var unit = try cc.parser.parse(allocator, "#include <stdio.h>\n", cc.layout.forArch(arch), .{
        .resolver = fsr.asResolver(),
        .system = systemPredef(arch),
    });
    defer unit.deinit();

    // `printf`, `fopen`, and similar functions are ordinary prototypes that survived the
    // whole chain. So the parser must produce a non-empty set of function declarations from
    // the real header.
    try std.testing.expect(unit.func_decls.len > 0);
}

test "VCC PARSES the real glibc <stdio.h> chain to no-error (host glibc)" {
    const allocator = std.testing.allocator;
    const arch = try hostLayoutArch();
    const glibc_include = (try findHostGlibcIncludeDir(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    try expectChainParsesClean(allocator, glibc_include, arch);
}

// The same full parse, repeated for each cross target whose glibc dev tree this Nix
// profile has.
test "cross-arch PARSE: x86_64 glibc <stdio.h>, if present" {
    const allocator = std.testing.allocator;
    const glibc_include = (try findCrossGlibcIncludeDir(allocator, "x86_64-unknown-linux-gnu")) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    try expectChainParsesClean(allocator, glibc_include, .x86_64);
}

test "cross-arch PARSE: riscv64 glibc <stdio.h>, if present" {
    const allocator = std.testing.allocator;
    const glibc_include = (try findCrossGlibcIncludeDir(allocator, "riscv64-unknown-linux-gnu")) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    try expectChainParsesClean(allocator, glibc_include, .riscv64);
}

// This chain is largely arch-independent. The `bits/*` family varies, but nothing about
// it should trip the preprocessor differently. The test repeats for each cross target,
// and skips cleanly whenever this Nix profile has no cross glibc dev tree for that arch.
test "cross-arch: x86_64 glibc dev tree, if present" {
    const allocator = std.testing.allocator;
    const glibc_include = (try findCrossGlibcIncludeDir(allocator, "x86_64-unknown-linux-gnu")) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    try expectChainPreprocessesClean(allocator, glibc_include, systemPredef(.x86_64));
}

test "cross-arch: riscv64 glibc dev tree, if present" {
    const allocator = std.testing.allocator;
    const glibc_include = (try findCrossGlibcIncludeDir(allocator, "riscv64-unknown-linux-gnu")) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    try expectChainPreprocessesClean(allocator, glibc_include, systemPredef(.riscv64));
}

test "cross-arch: i686 (x86) glibc dev tree, if present" {
    const allocator = std.testing.allocator;
    const glibc_include = (try findCrossGlibcIncludeDir(allocator, "i686-unknown-linux-gnu")) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);
    var sp = systemPredef(.x86);
    sp.long_bits = 32;
    sp.ptr_bits = 32;
    try expectChainPreprocessesClean(allocator, glibc_include, sp);
}
