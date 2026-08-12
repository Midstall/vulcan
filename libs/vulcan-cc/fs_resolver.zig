//! A filesystem `#include` resolver. It is the host-side counterpart to the pure,
//! in-memory `IncludeResolver`s that tests build by hand (`tests/native.zig`'s
//! `ppTestResolve`, `tests/variadic.zig`'s `stdargOnlyResolve`). `preproc.zig` itself never
//! touches a clock or a filesystem (see `IncludeResolver`'s doc comment). A driver that wants
//! real `#include` resolution builds an `FsResolver` and wires `asResolver()` into
//! `Options.resolver`.
//!
//! The search order mirrors gcc. For a SYSTEM (`<...>`) include, this compiler checks its own
//! BUILT-IN header set first (`<stddef.h>`/`<stdarg.h>`, see `stddef.zig`/`stdarg.zig` for why
//! they must win over anything on disk). It then checks `system_dirs` (the ordered `-isystem`
//! list) in order. For a `"..."` (quoted) include, the compiler searches `includer_dir` first
//! (the including file's own directory, when known), then the same `system_dirs` list. The
//! built-in set is SYSTEM-only, matching a real libc: `#include "stddef.h"` never sees it. If
//! the header is not found anywhere, the resolver returns `null`. `preproc.zig`'s
//! `handleInclude` turns that into a clear `error.PreprocError` that names the include.

const std = @import("std");
const preproc = @import("preproc.zig");
const lexer = @import("lexer.zig");
const layout = @import("layout.zig");
const stddef = @import("stddef.zig");
const stdarg = @import("stdarg.zig");
const limits = @import("limits.zig");

/// Serves this compiler's own built-in headers ahead of any disk search. See the module
/// doc comment for the search order. Returns `null` for anything else, so a caller falls
/// through to the disk search.
fn resolveBuiltin(name: []const u8, is_system: bool) ?[]const u8 {
    if (stddef.resolve(name, is_system)) |b| return b;
    if (stdarg.resolve(name, is_system)) |b| return b;
    if (limits.resolve(name, is_system)) |b| return b;
    return null;
}

/// A filesystem-backed `IncludeResolver`. `system_dirs` is the ordered `-isystem`
/// search list. It is a caller-owned slice, `FsResolver` only reads it and never frees it.
/// `allocator` and `io` back every disk read this resolver performs. `IncludeResolver.resolveFn`
/// takes no allocator or `std.Io` of its own (see `preproc.zig`), so both are captured here
/// at `init` instead. LIFETIME: every `ResolvedFile.bytes` and `.identity` this resolver
/// returns is allocated from `allocator`. Each one must outlive the whole `preprocess` call
/// it was returned into, because a header's bytes must survive until the TU is fully
/// expanded. The simplest way to satisfy that is to pass an arena allocator here and call
/// `deinit` once, after `preprocess` returns, rather than freeing each read individually.
pub const FsResolver = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    system_dirs: []const []const u8,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, system_dirs: []const []const u8) FsResolver {
        return .{ .allocator = allocator, .io = io, .system_dirs = system_dirs };
    }

    /// The `IncludeResolver` vtable view of this resolver. Wire this into
    /// `preproc.Options.resolver`. `ctx` is `self`, so `self` must outlive every
    /// `preprocess` call it's passed to.
    pub fn asResolver(self: *FsResolver) preproc.IncludeResolver {
        return .{ .ctx = @ptrCast(self), .resolveFn = resolveFn };
    }

    fn resolveFn(ctx: ?*anyopaque, name: []const u8, is_system: bool, includer_dir: ?[]const u8, next_after: ?[]const u8) preproc.Error!?preproc.ResolvedFile {
        const self: *FsResolver = @ptrCast(@alignCast(ctx.?));
        // `#include_next` (`next_after` non-null) re-resolves `name` starting AFTER the
        // search directory the current file came from. It never re-serves a built-in header.
        // It never consults `includer_dir`, only the `system_dirs` list, from `start`
        // onward. `start` is the entry just past the directory whose path is a prefix of
        // `next_after` (the current file's absolute identity). A current file with no matching
        // directory (a built-in, whose identity is a bare name) leaves `start` at 0, so the
        // search covers the whole `system_dirs` list. This is how VCC's own `<limits.h>`
        // reaches glibc's `<limits.h>` underneath it.
        if (next_after) |after| {
            // `after` is the current file's `identity` (an absolute canonical path for a disk
            // header, or a bare name for a built-in). A search directory may be spelled relative
            // (`-Ilib`) while the identity is absolute, and two spellings (`lib` and `./lib`)
            // may name the same directory, so a raw string prefix does not work. Canonicalize
            // each search directory and find the DEEPEST one that contains `after`. That is
            // the directory the current file came from. `start` is one past the last spelling
            // of that deepest directory, so every spelling of the origin is skipped. A
            // built-in origin (a bare name, under no directory) leaves `start` at 0, so the
            // whole list is searched. This is how VCC's `<limits.h>` reaches glibc's.
            var start: usize = 0;
            var best_len: usize = 0;
            var buf: [std.fs.max_path_bytes]u8 = undefined;
            for (self.system_dirs, 0..) |dir, k| {
                const real = self.dirRealPath(dir, &buf) orelse continue;
                if (!std.mem.startsWith(u8, after, real)) continue;
                if (after.len != real.len and after[real.len] != '/') continue; // path boundary
                if (real.len > best_len) {
                    best_len = real.len;
                    start = k + 1;
                } else if (real.len == best_len) {
                    start = k + 1;
                }
            }
            for (self.system_dirs[start..]) |dir| {
                if (try self.tryDir(dir, name)) |rf| return rf;
            }
            return null;
        }
        if (is_system) {
            if (resolveBuiltin(name, is_system)) |b| return .{ .identity = name, .bytes = b };
        } else if (includer_dir) |dir| {
            if (try self.tryDir(dir, name)) |rf| return rf;
        }
        for (self.system_dirs) |dir| {
            if (try self.tryDir(dir, name)) |rf| return rf;
        }
        return null;
    }

    /// The canonical absolute path of `dir`, written into `buf` and returned as a sub-slice.
    /// Returns `null` for a directory that cannot be opened or canonicalized (a `-I` entry
    /// naming a missing directory). A directory that does not exist can never be an
    /// `#include_next` origin, so skipping it is right. Used only by the `#include_next`
    /// origin search.
    fn dirRealPath(self: *FsResolver, dir: []const u8, buf: []u8) ?[]const u8 {
        var d = std.Io.Dir.cwd().openDir(self.io, dir, .{}) catch return null;
        defer d.close(self.io);
        const n = d.realPath(self.io, buf) catch return null;
        return buf[0..n];
    }

    /// Try resolving `name` under `dir`: open `dir`, read `name` inside it whole, and
    /// resolve `name`'s ABSOLUTE path for `ResolvedFile.identity` (the include-guard and
    /// `#pragma once` bookkeeping key, see `ResolvedFile`'s doc comment, needs a stable,
    /// canonical identity, not a directory-relative spelling that could name a different
    /// file depending on the current directory). Returns `null` for any ordinary failure (a
    /// missing directory, a missing file, or an unreadable one). A search-list entry that
    /// doesn't pan out just falls through to the next one. An actual `error.OutOfMemory`
    /// still propagates, since that isn't "not found".
    fn tryDir(self: *FsResolver, dir: []const u8, name: []const u8) preproc.Error!?preproc.ResolvedFile {
        var d = std.Io.Dir.cwd().openDir(self.io, dir, .{}) catch return null;
        defer d.close(self.io);
        const file_bytes = d.readFileAlloc(self.io, name, self.allocator, .limited(16 * 1024 * 1024)) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => return null,
        };
        // `realPathFileAlloc` returns a SENTINEL-terminated `[:0]u8`. It is copied here into a
        // plain, non-sentineled allocation before it's handed back as `ResolvedFile.identity`
        // ([]const u8), so a caller freeing `.identity` with an ordinary `allocator.free`
        // sees the same byte count it allocated (a sentineled slice's `.len` excludes its
        // trailing 0, so freeing it back through a non-sentineled slice type would report a
        // mismatched size to the allocator).
        const identity_z = d.realPathFileAlloc(self.io, name, self.allocator) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => {
                self.allocator.free(file_bytes);
                return null;
            },
        };
        defer self.allocator.free(identity_z);
        const identity = self.allocator.dupe(u8, identity_z) catch |err| {
            self.allocator.free(file_bytes);
            return err;
        };
        return .{ .identity = identity, .bytes = file_bytes };
    }
};

/// An aarch64 (LP64) `SystemPredef`, matching `preproc.zig`'s own test constant, so
/// `__SIZE_TYPE__` etc. expand the way a real target's headers would see them.
const aarch64_system_predef: preproc.SystemPredef = .{
    .arch = .aarch64,
    .gnuc_major = 4,
    .gnuc_minor = 2,
    .gnuc_patch = 0,
    .long_bits = 64,
    .ptr_bits = 64,
    .char_signed = false,
};

test "asResolver serves the built-in stddef.h even with no system_dirs at all" {
    const allocator = std.testing.allocator;
    var fsr = FsResolver.init(allocator, std.testing.io, &.{});
    const toks = try preproc.preprocess(allocator, "#include <stddef.h>\nsize_t x;\n", .{
        .resolver = fsr.asResolver(),
        .system = aarch64_system_predef,
    });
    defer lexer.freeTokens(allocator, toks);
    // `__SIZE_TYPE__`/`__PTRDIFF_TYPE__`/`__WCHAR_TYPE__` expand to the aarch64 LP64
    // spellings (3/2/1 tokens respectively, see `preproc.zig`'s own `seedSystemPredefs`
    // test). Each `typedef` line below is longer than its source text. The `#define`
    // lines are directives, consumed with no emitted tokens. The including source's own
    // `size_t x;` line comes last, unexpanded (`size_t` is a plain identifier here, the
    // preprocessor has no notion of typedefs).
    const expected_text = [_][]const u8{
        "typedef", "long", "unsigned", "int", "size_t", ";",
        "typedef", "long", "int",      "ptrdiff_t", ";",
        "typedef", "int",  "wchar_t",  ";",
        "size_t",  "x",    ";",
    };
    try std.testing.expectEqual(expected_text.len + 1, toks.len); // +1 for the trailing eof
    for (expected_text, toks[0..expected_text.len]) |want, got| try std.testing.expectEqualStrings(want, got.text);
    try std.testing.expectEqual(lexer.Kind.kw_typedef, toks[0].kind);
    try std.testing.expectEqual(lexer.Kind.semicolon, toks[5].kind);
    try std.testing.expectEqual(lexer.Kind.eof, toks[toks.len - 1].kind);
}

test "the built-in stddef.h also defines NULL and offsetof" {
    const allocator = std.testing.allocator;
    var fsr = FsResolver.init(allocator, std.testing.io, &.{});
    const toks = try preproc.preprocess(allocator,
        "#include <stddef.h>\n" ++
            "#ifdef NULL\nint has_null;\n#else\nint no_null;\n#endif\n" ++
            "#ifdef offsetof\nint has_offsetof;\n#else\nint no_offsetof;\n#endif\n",
        .{ .resolver = fsr.asResolver(), .system = aarch64_system_predef },
    );
    defer lexer.freeTokens(allocator, toks);
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(allocator);
    for (toks) |t| {
        try joined.appendSlice(allocator, t.text);
        try joined.append(allocator, ' ');
    }
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "has_null") != null);
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "no_null") == null);
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "has_offsetof") != null);
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "no_offsetof") == null);
}

test "asResolver resolves a system include from a real directory on disk" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "foo.h", .data = "int from_foo_h;\n" });

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);
    const dirs = [_][]const u8{dir_path};

    // The FsResolver's OWN disk reads come from an arena (see `FsResolver`'s doc comment on
    // lifetime and ownership). They are freed as one block once `preprocess` (which copies out
    // everything it needs) has returned, rather than tracked read-by-read.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var fsr = FsResolver.init(arena.allocator(), io, &dirs);
    const toks = try preproc.preprocess(allocator, "#include <foo.h>\nint y;\n", .{ .resolver = fsr.asResolver() });
    defer lexer.freeTokens(allocator, toks);
    try std.testing.expectEqualStrings("from_foo_h", toks[1].text);
    try std.testing.expectEqualStrings("y", toks[4].text);
}

test "a missing header resolves to null, surfacing as a clear preprocessor error" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);
    const dirs = [_][]const u8{dir_path};

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var fsr = FsResolver.init(arena.allocator(), io, &dirs);
    try std.testing.expectError(error.PreprocError, preproc.preprocess(allocator, "#include <missing.h>\nint y;\n", .{ .resolver = fsr.asResolver() }));
}

test "a quoted include searches includer_dir before system_dirs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "local.h", .data = "int from_local_h;\n" });

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);

    var fsr = FsResolver.init(allocator, io, &.{});
    const resolver = fsr.asResolver();
    const rf = (try resolver.resolveFn(resolver.ctx, "local.h", false, dir_path, null)).?;
    defer allocator.free(rf.bytes);
    defer allocator.free(rf.identity);
    try std.testing.expectEqualStrings("int from_local_h;\n", rf.bytes);
}
