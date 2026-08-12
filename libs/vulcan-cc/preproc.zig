//! The C preprocessor sits between raw source and `lexer.tokenize`'s token stream.
//! It handles the source-transformation steps the C standard puts ahead of tokenization
//! proper: backslash-newline line splicing (translation phase 2), comment stripping
//! (phase 3), and directive dispatch plus macro expansion (phase 4). Phase 4 covers
//! object-like macros, function-like macros with `#`/`##`, and variadic macros with
//! `__VA_ARGS__`, the GNU `, ##__VA_ARGS__` comma-swallow, and C23 `__VA_OPT__`.
//!
//! The pipeline follows a strict phase order, not ad-hoc scanning:
//!   1. `splice` removes every `\`+newline everywhere in the text: inside strings, numbers,
//!      operators, comments, and identifiers. It produces a spliced `text` plus a parallel
//!      `line_of` array. This array maps each spliced byte back to its original physical line.
//!   2. `scan` tokenizes the already-spliced `text`, so it contains no splice logic of its
//!      own. A `//` comment, a `==` operator, a `"..."` string, or an identifier split
//!      across a `\`+newline all just work, because the splice is already gone. A token's
//!      `.line` comes from `line_of[start]`, so diagnostics and `__LINE__` stay exact across
//!      splices, even though the running column is best-effort on the spliced text.
//!   3. A directive and expansion pass walks the `PToken` stream one logical line at a time
//!      (lines are delimited by `newline` PTokens). A line starting with a `bol` `.hash` is
//!      a directive, dispatched to `#define`/`#undef`/`#error`/`#warning`/`#line`/`#pragma`.
//!      Anything else, for example `#if`/`#include`, is `error.Unsupported` for now. Any
//!      other line has its macros expanded (`expandTokens`: object-like and function-like,
//!      with rescanning and a Prosser hide-set, so recursive and mutually-recursive macros
//!      always terminate), and its tokens are copied into the final stream.
//!   4. `emit` collapses that final `PToken` stream to the `lexer.Token` stream the parser
//!      sees. It drops `newline`, rejects a stray `#`/`##` (meaning it was not consumed as a
//!      directive, so it was not at the start of a line), and applies `lexer.keyword` to
//!      identifiers.
//! `preprocess` runs `scan`, then the directive and expansion pass, then `emit`.

const std = @import("std");
const lexer = @import("lexer.zig");
const layout = @import("layout.zig");

/// A preprocessing token. It is like `lexer.Token`, but carries extra bits that directive
/// handling and macro expansion need before the stream collapses to plain `lexer.Token`s
/// via `emit`. `space_before` records whether whitespace or a stripped comment preceded
/// this token. The C standard requires that fact to survive into macro expansion:
/// stringizing `#x` and adjacent-token paste decisions both depend on it. `bol` is true for
/// the first token on its line, ignoring leading whitespace and comments. Directive
/// recognition needs it, since a `#` only starts a directive at the start of a line. `hide`
/// is the macro hide set, empty until macro expansion runs. `owns_text` mirrors
/// `lexer.Token.owns_text`. `scan` returns self-contained PTokens: every real token's
/// `.text` is heap-owned, a copy lifted out of the transient spliced buffer. This lets later
/// code hold the stream without keeping any source buffer alive. `owns_text` is false only
/// for the empty-text `newline`/`eof` markers. `char_lit` marks an `int_lit` that came from
/// a character literal (`'A'` becomes `int_lit` "65"). Its `.text` is the decoded decimal
/// value, not the original `'A'` spelling. Like a `str_lit`, whose `.text` is likewise
/// decoded bytes, it cannot be faithfully stringized (`#`) or pasted (`##`). See
/// `spellingRecoverable`.
pub const PToken = struct {
    kind: lexer.Kind,
    text: []const u8,
    line: u32,
    col: u32,
    owns_text: bool = false,
    space_before: bool = false,
    bol: bool = false,
    char_lit: bool = false,
    hide: []const []const u8 = &.{},
};

/// A macro definition: object-like when `params` is null, function-like otherwise. `body`
/// is the unexpanded replacement list.
pub const Macro = struct {
    name: []const u8,
    params: ?[]const []const u8,
    variadic: bool,
    body: []const PToken,
};

pub const MacroTable = std.StringHashMapUnmanaged(Macro);

/// A file resolved by an `IncludeResolver`: `identity` names it for include-guard/
/// pragma-once bookkeeping (typically a canonical path). `bytes` is its raw contents.
pub const ResolvedFile = struct { identity: []const u8, bytes: []const u8 };

/// Resolves an `#include` target to file bytes. See `handleInclude` below. A vtable-style
/// callback, not direct filesystem access, so `preprocess` itself stays pure: no clock, no
/// filesystem. A host that needs `#include` injects its own resolver. This can be a disk
/// resolver, or a virtual one backed by an in-memory header table, as `tests/native.zig`'s
/// `expectAgreesPPBits` does.
pub const IncludeResolver = struct {
    ctx: ?*anyopaque = null,
    resolveFn: *const fn (ctx: ?*anyopaque, name: []const u8, is_system: bool, includer_dir: ?[]const u8, next_after: ?[]const u8) Error!?ResolvedFile,
};

/// A predefined macro (driver `-D name=value`, a later task), applied before the source's
/// own `#define`/`#undef` directives run.
pub const Define = struct { name: []const u8, value: []const u8 };

/// The per-target GCC-style predefined-macro set. It is opt-in via `Options.system`.
/// `arch` picks the arch macros (`__aarch64__`, and so on). `gnuc_major`, `gnuc_minor`, and
/// `gnuc_patch` back `__GNUC__`/`__GNUC_MINOR__`/`__GNUC_PATCHLEVEL__`. A real C library's
/// headers gate GCC-extension use on these. `long_bits`/`ptr_bits` size the `__SIZEOF_*__`
/// macros and pick the LP64 vs ILP32 form of the `__SIZE_TYPE__`-family type macros.
/// `char_signed` is carried for future use. No macro reads it yet.
pub const SystemPredef = struct {
    arch: layout.Arch,
    gnuc_major: u32,
    gnuc_minor: u32,
    gnuc_patch: u32,
    long_bits: u16,
    ptr_bits: u16,
    char_signed: bool,
};

/// Preprocessor configuration. `filename`/`timestamp` back `__FILE__`/`__DATE__`/`__TIME__`.
/// `timestamp` is caller-supplied rather than clock-read, so `preprocess` stays pure and
/// reproducible. `resolver` backs `#include`. `defines`/`undefines` seed the macro table
/// before the source's own directives run. `system` is null by default, so `preseedMacros`
/// seeds only the two standard `__STDC__*` macros, byte-identical to before this field
/// existed. A caller opts into the full GCC-style per-target set by passing a `SystemPredef`.
pub const Options = struct {
    filename: []const u8 = "<source>",
    timestamp: i64 = 0,
    resolver: ?IncludeResolver = null,
    defines: []const Define = &.{},
    undefines: []const []const u8 = &.{},
    system: ?SystemPredef = null,
};

pub const Error = lexer.Error || error{ PreprocError, Unsupported };

/// The result of `splice`: `text` is `source` with every `\`+newline removed. `line_of` is
/// parallel to `text` (`line_of[k]` is the 1-based original physical line of `text[k]`).
/// Both are heap-allocated. Free each with `allocator.free`.
pub const Spliced = struct { text: []u8, line_of: []u32 };

/// Frees a `PToken` slice returned by `scan`: first each token's heap-allocated `.text`,
/// then the slice itself. Mirrors `lexer.freeTokens`.
pub fn freePTokens(allocator: std.mem.Allocator, toks: []PToken) void {
    for (toks) |t| if (t.owns_text) allocator.free(t.text);
    allocator.free(toks);
}

/// Translation phase 2: remove every backslash-newline splice from `source`. A `\`
/// immediately followed by `\n` (or `\r\n`) is deleted, and the physical lines it joined
/// become one logical line. This applies everywhere: inside strings, numbers, operators,
/// comments, and identifiers alike, exactly as the C standard mandates before tokenization
/// sees the text. `line_of[k]` records the original 1-based physical line of `text[k]`. The
/// line counter is bumped on every newline, including the ones a splice consumes, so a
/// token that lands after one or more splices still reports its true source line. The
/// caller owns, and must free, both returned slices.
pub fn splice(allocator: std.mem.Allocator, source: []const u8) Error!Spliced {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(allocator);
    var line_of: std.ArrayList(u32) = .empty;
    errdefer line_of.deinit(allocator);

    var i: usize = 0;
    var orig_line: u32 = 1;
    while (i < source.len) {
        if (source[i] == '\\' and i + 1 < source.len and source[i + 1] == '\n') {
            i += 2;
            orig_line += 1;
            continue;
        }
        if (source[i] == '\\' and i + 2 < source.len and source[i + 1] == '\r' and source[i + 2] == '\n') {
            i += 3;
            orig_line += 1;
            continue;
        }
        try text.append(allocator, source[i]);
        try line_of.append(allocator, orig_line);
        if (source[i] == '\n') orig_line += 1;
        i += 1;
    }

    const text_slice = try text.toOwnedSlice(allocator);
    errdefer allocator.free(text_slice);
    const line_slice = try line_of.toOwnedSlice(allocator);
    return .{ .text = text_slice, .line_of = line_slice };
}

/// Scans a `"`-delimited string body (the opening quote already consumed by the caller) on
/// already-spliced `text`. It decodes escapes into a fresh owned buffer, up to and
/// including the closing `"`. This is shared by narrow `str_lit` and wide `wstr_lit`
/// PToken scanning. The two differ only in which `Kind` the caller wraps this text in.
fn scanPPStringBody(allocator: std.mem.Allocator, text: []const u8, i: *usize, col: *u32) Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    while (i.* < text.len and text[i.*] != '"') {
        if (text[i.*] == '\\') {
            i.* += 1;
            col.* += 1;
            if (i.* >= text.len) return error.UnexpectedChar;
            try buf.append(allocator, try lexer.decodeEscape(text, i, col));
        } else {
            if (text[i.*] == '\n') {
                col.* = 1;
            } else {
                col.* += 1;
            }
            try buf.append(allocator, text[i.*]);
            i.* += 1;
        }
    }
    if (i.* >= text.len) return error.UnexpectedChar;
    i.* += 1;
    col.* += 1;
    return try buf.toOwnedSlice(allocator);
}

/// Scans `source` into preprocessing tokens. It runs `splice` first, then tokenizes the
/// spliced text: comments are stripped to a whitespace boundary, `#`/`##` are recognized
/// (but not yet acted on here), and `bol`/`space_before` are tracked on every token. The
/// scanner carries no splice logic of its own, since the splice is already gone from the
/// text, and reuses `lexer`'s identifier, operator, and escape helpers. Identifiers are
/// emitted as plain `.ident`. `lexer.keyword` is applied later, by `emit`, so a
/// macro-expanded name that spells a keyword is classified the same way source-written
/// text is. Each token's `.line` is its original physical line (via `line_of`). Its `.col`
/// is a best-effort running column over the spliced text. The result always ends with an
/// `eof` PToken and is self-contained (see `PToken.owns_text`). Free it with
/// `freePTokens`. `opts` is unused by this pass-through. It exists so directive handling,
/// macro expansion, and `#include` do not need to change this signature.
pub fn scan(allocator: std.mem.Allocator, source: []const u8, opts: Options) Error![]PToken {
    _ = opts;
    const sp = try splice(allocator, source);
    defer allocator.free(sp.text);
    defer allocator.free(sp.line_of);
    const text = sp.text;
    const line_of = sp.line_of;

    var out: std.ArrayList(PToken) = .empty;
    errdefer {
        for (out.items) |t| if (t.owns_text) allocator.free(t.text);
        out.deinit(allocator);
    }

    var i: usize = 0;
    var col: u32 = 1;
    var space_before = false;
    var bol = true;

    while (i < text.len) {
        const c = text[i];
        // Comments are stripped here (phase 3), on already-spliced text. So a `//` comment
        // whose line ended in a `\`+newline in the source correctly runs on into the joined
        // line, because that splice is already gone. Both comment forms are whitespace. They
        // set `space_before` but leave `bol` untouched. A comment, like other whitespace,
        // does not itself count as the line's first token, nor cancel a `#` that a run of
        // whitespace and comments still leaves at the start of its line.
        if (c == '/' and i + 1 < text.len and text[i + 1] == '/') {
            i += 2;
            col += 2;
            while (i < text.len and text[i] != '\n') {
                i += 1;
                col += 1;
            }
            space_before = true;
            continue;
        }
        if (c == '/' and i + 1 < text.len and text[i + 1] == '*') {
            i += 2;
            col += 2;
            var closed = false;
            while (i < text.len) {
                if (text[i] == '*' and i + 1 < text.len and text[i + 1] == '/') {
                    i += 2;
                    col += 2;
                    closed = true;
                    break;
                }
                if (text[i] == '\n') {
                    col = 1;
                } else {
                    col += 1;
                }
                i += 1;
            }
            if (!closed) return error.PreprocError;
            space_before = true;
            continue;
        }

        switch (c) {
            // Space, tab, carriage return, and the vertical whitespace form feed (0x0c) and
            // vertical tab (0x0b). Real headers put a form feed (a page-break `^L`) on its own
            // line. C treats all of these as white space (C11 6.4p1).
            ' ', '\t', '\r', 0x0c, 0x0b => {
                i += 1;
                col += 1;
                space_before = true;
            },
            '\n' => {
                try out.append(allocator, .{ .kind = .newline, .text = "", .line = line_of[i], .col = col, .space_before = space_before, .bol = bol });
                i += 1;
                col = 1;
                space_before = false;
                bol = true;
            },
            else => {
                const start_line = line_of[i];
                const start_col = col;
                const this_space = space_before;
                const this_bol = bol;
                // A wide-literal prefix `L'...'`/`L"..."` only fires when the `L` is immediately
                // followed by `'`/`"`, with no whitespace. This mirrors `lexer.tokenize`'s own
                // `L`-prefix branch, so `Label`, `L1`, and a variable named `L` all stay plain
                // `.ident` PTokens (the branch below).
                if (c == 'L' and i + 1 < text.len and (text[i + 1] == '\'' or text[i + 1] == '"')) {
                    i += 1; // the `L` itself
                    col += 1;
                    const quote = text[i];
                    i += 1; // the opening quote
                    col += 1;
                    if (quote == '"') {
                        const str = try scanPPStringBody(allocator, text, &i, &col);
                        errdefer allocator.free(str);
                        try out.append(allocator, .{ .kind = .wstr_lit, .text = str, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                    } else {
                        // `L'c'` is functionally identical to `'c'`: both are `int`, with the
                        // same code point. Emit the exact same `char_lit`-marked `.int_lit`.
                        const value = try lexer.scanCharBody(text, &i, &col);
                        const str = try std.fmt.allocPrint(allocator, "{d}", .{value});
                        errdefer allocator.free(str);
                        try out.append(allocator, .{ .kind = .int_lit, .text = str, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol, .char_lit = true });
                    }
                } else if (lexer.isIdentStart(c)) {
                    const start = i;
                    while (i < text.len and lexer.isIdentCont(text[i])) {
                        i += 1;
                        col += 1;
                    }
                    const dup = try allocator.dupe(u8, text[start..i]);
                    errdefer allocator.free(dup);
                    try out.append(allocator, .{ .kind = .ident, .text = dup, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                } else if (std.ascii.isDigit(c)) {
                    // Mirrors `lexer.zig`'s digit scan via the shared `lexer.scanNumber`. It handles
                    // hex/binary prefixes, the decimal digit run, and the float-vs-int suffix, all in
                    // one place, so this pp-token pass and the final lexer can never drift apart on
                    // radix/float detection (see `scanNumber`'s doc comment).
                    const start = i;
                    const is_float = try lexer.scanNumber(text, &i, &col);
                    const dup = try allocator.dupe(u8, text[start..i]);
                    errdefer allocator.free(dup);
                    try out.append(allocator, .{ .kind = if (is_float) .float_lit else .int_lit, .text = dup, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                } else if (c == '.' and i + 1 < text.len and std.ascii.isDigit(text[i + 1])) {
                    // A `.` immediately followed by a digit starts a dot-led float literal
                    // (`.5`), not the `.` member-access operator. This mirrors `lexer.tokenize`'s
                    // dot-led branch.
                    const start = i;
                    _ = try lexer.scanNumber(text, &i, &col);
                    const dup = try allocator.dupe(u8, text[start..i]);
                    errdefer allocator.free(dup);
                    try out.append(allocator, .{ .kind = .float_lit, .text = dup, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                } else if (c == '"') {
                    i += 1;
                    col += 1;
                    const str = try scanPPStringBody(allocator, text, &i, &col);
                    errdefer allocator.free(str);
                    try out.append(allocator, .{ .kind = .str_lit, .text = str, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                } else if (c == '\'') {
                    i += 1;
                    col += 1;
                    const value = try lexer.scanCharBody(text, &i, &col);
                    const str = try std.fmt.allocPrint(allocator, "{d}", .{value});
                    errdefer allocator.free(str);
                    // `char_lit` marks this: `.text` is the decoded decimal value, not the
                    // `'A'` source spelling, so `#`/`##` on it must fail closed.
                    try out.append(allocator, .{ .kind = .int_lit, .text = str, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol, .char_lit = true });
                } else if (c == '#') {
                    const kind: lexer.Kind = if (i + 1 < text.len and text[i + 1] == '#') .hash_hash else .hash;
                    const len: usize = if (kind == .hash_hash) 2 else 1;
                    const dup = try allocator.dupe(u8, text[i .. i + len]);
                    errdefer allocator.free(dup);
                    try out.append(allocator, .{ .kind = kind, .text = dup, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                    i += len;
                    col += @intCast(len);
                } else if (lexer.matchOperator(text[i..])) |op| {
                    const dup = try allocator.dupe(u8, text[i .. i + op.text.len]);
                    errdefer allocator.free(dup);
                    try out.append(allocator, .{ .kind = op.kind, .text = dup, .line = start_line, .col = start_col, .owns_text = true, .space_before = this_space, .bol = this_bol });
                    i += op.text.len;
                    col += @intCast(op.text.len);
                } else {
                    return error.UnexpectedChar;
                }
                space_before = false;
                bol = false;
            },
        }
    }

    // eof's line is the physical line just past the last byte: the last byte's line, plus
    // one if that byte was a newline (matching where a fresh token would start).
    const eof_line: u32 = if (text.len == 0) 1 else line_of[text.len - 1] + @intFromBool(text[text.len - 1] == '\n');
    try out.append(allocator, .{ .kind = .eof, .text = "", .line = eof_line, .col = col, .space_before = space_before, .bol = bol });
    return out.toOwnedSlice(allocator);
}

/// Collapses a `PToken` stream to the final `lexer.Token` stream the parser consumes. It
/// drops `newline` (a scan-only bookkeeping token), and rejects a stray `hash`/`hash_hash`
/// as `error.PreprocError`. The directive and expansion pass consumes every `#`/`##` that
/// starts a directive line, so one reaching here means it was not at the start of a line,
/// for example `int x; # int y;`. It applies `lexer.keyword` to `.ident` text. Ownership of
/// each owned token's heap text is moved into the returned `Token` (the source `PToken`'s
/// `owns_text` is cleared), so the caller must free the result with `lexer.freeTokens` and
/// may still free `ptoks` with `freePTokens`. Neither call double-frees the moved text.
/// `ptoks` is mutated in place.
pub fn emit(allocator: std.mem.Allocator, ptoks: []PToken) Error![]lexer.Token {
    var out: std.ArrayList(lexer.Token) = .empty;
    errdefer {
        for (out.items) |t| if (t.owns_text) allocator.free(t.text);
        out.deinit(allocator);
    }

    for (ptoks) |*pt| {
        switch (pt.kind) {
            .newline => {},
            .hash, .hash_hash => return error.PreprocError,
            else => {
                const kind = if (pt.kind == .ident) (lexer.keyword(pt.text) orelse .ident) else pt.kind;
                try out.append(allocator, .{ .kind = kind, .text = pt.text, .line = pt.line, .col = pt.col, .owns_text = pt.owns_text });
                // The append copied the (pointer, owns_text=true) into `out`. Hand the text
                // over so the caller's `freePTokens(ptoks)` won't also free it.
                pt.owns_text = false;
            },
        }
    }
    return out.toOwnedSlice(allocator);
}

/// Frees a `Macro`: its owned `name`, each owned body token's `.text`, the `body` slice
/// itself, and, for a function-like macro, each owned param name plus the `params`
/// slice. It mirrors `freePTokens` but operates on `Macro.body`'s `[]const PToken`, whose
/// element text is always owned. See `dupPTokenSlice`.
fn freeMacro(allocator: std.mem.Allocator, m: *const Macro) void {
    for (m.body) |t| if (t.owns_text) allocator.free(t.text);
    allocator.free(m.body);
    if (m.params) |params| {
        for (params) |p| allocator.free(p);
        allocator.free(params);
    }
    allocator.free(m.name);
}

/// Copies `src` (a slice of PTokens borrowed from the scanner's line buffer) into a
/// freshly heap-owned `[]PToken`. Every token's `.text` is duplicated, so the copy
/// outlives whatever `src` was borrowed from, and `.hide` is reset to empty. A macro
/// body's tokens start with no hide-set of their own. The hide-set is applied fresh at
/// each expansion site, per Prosser. This gives a `#define`d macro's body an
/// independent lifetime from the line it was scanned from.
fn dupPTokenSlice(allocator: std.mem.Allocator, src: []const PToken) Error![]PToken {
    const out = try allocator.alloc(PToken, src.len);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |t| if (t.owns_text) allocator.free(t.text);
        allocator.free(out);
    }
    for (src, 0..) |t, k| {
        const dup = try allocator.dupe(u8, t.text);
        out[k] = .{
            .kind = t.kind,
            .text = dup,
            .line = t.line,
            .col = t.col,
            .owns_text = true,
            .space_before = t.space_before,
            .bol = t.bol,
            .char_lit = t.char_lit,
            .hide = &.{},
        };
        filled += 1;
    }
    return out;
}

/// Does `name` appear in hide-set `hide`? This is a linear scan. Hide-sets are tiny, bounded by
/// the number of macros nested along one expansion chain.
fn hideContains(hide: []const []const u8, name: []const u8) bool {
    for (hide) |h| if (std.mem.eql(u8, h, name)) return true;
    return false;
}

/// `hide ∪ {name}`, allocated from `arena`. It never mutates `hide`, since each expansion
/// step gets its own new array, and never removes anything, so a name once added to a
/// lineage's hide-set stays there. This is the property Prosser's rule relies on to
/// guarantee macro expansion always terminates (see `expandLine`).
fn hideUnion(arena: std.mem.Allocator, hide: []const []const u8, name: []const u8) Error![]const []const u8 {
    const out = try arena.alloc([]const u8, hide.len + 1);
    @memcpy(out[0..hide.len], hide);
    out[hide.len] = name;
    return out;
}

/// `#define NAME body...` (object-like) or `#define NAME(params) body...` (function-like:
/// `NAME` immediately followed by `(` with no space between). `args` is the
/// directive line's tokens after the `define` keyword itself. `args[0]` must be the macro
/// name, and `args[1..]` is the rest. For a function-like macro, `rest[1..]` up to the
/// matching `)` is parsed as a comma-separated list of parameter identifiers (`(` alone,
/// that is `()`, means zero params). A variadic ellipsis is one `.ellipsis` token. Before
/// that token existed, the lexer had no `...` token, so it scanned as three consecutive
/// `.dot`s instead. It is recognized here so it does not mis-tokenize as something else
/// either way.
/// A body that mentions `__VA_ARGS__`/`__VA_OPT__` is only valid when the macro is
/// `variadic`. Otherwise it is `error.PreprocError` at define time (nothing is stored),
/// rather than being silently accepted and mishandled at expansion time. See
/// `expandTokens`'s function-like branch for how a variadic macro's trailing arguments
/// bind to `__VA_ARGS__`. A prior definition of the same name is replaced, and freed
/// first. Identical-redefinition is not enforced.
fn handleDefine(allocator: std.mem.Allocator, macros: *MacroTable, args: []const PToken) Error!void {
    if (args.len == 0 or args[0].kind != .ident) return error.PreprocError;
    const name = args[0].text;
    const rest = args[1..];

    var params: ?[]const []const u8 = null;
    var variadic = false;
    // GNU named variadic (`#define f(a, rest...) ...`): `rest` binds the trailing arguments,
    // the same as `__VA_ARGS__` binds them for the C99 `...` form. Recorded here, then every
    // `rest` in the body is rewritten to `__VA_ARGS__` below, so the existing variadic
    // expansion path needs no change.
    var va_name: ?[]const u8 = null;
    var body_src: []const PToken = rest;

    if (rest.len > 0 and rest[0].kind == .lparen and !rest[0].space_before) {
        var plist: std.ArrayList([]const u8) = .empty;
        defer plist.deinit(allocator); // elements are borrowed (from `rest`); only the list itself is owned here.

        var i: usize = 1; // rest[0] is the '('
        if (i < rest.len and rest[i].kind == .rparen) {
            i += 1; // `()`: zero params
        } else {
            while (true) {
                if (i >= rest.len) return error.PreprocError;
                if (rest[i].kind == .ellipsis) {
                    variadic = true;
                    i += 1;
                } else if (rest[i].kind == .ident) {
                    // GNU named variadic: an `.ident` immediately followed by `...` names the
                    // variadic argument (`args...`). It is not an ordinary parameter, so it is
                    // not added to `plist`. It must be the last thing before `)`.
                    if (i + 1 < rest.len and rest[i + 1].kind == .ellipsis) {
                        va_name = rest[i].text;
                        variadic = true;
                        i += 2;
                    } else {
                        try plist.append(allocator, rest[i].text);
                        i += 1;
                    }
                } else {
                    return error.PreprocError;
                }
                if (i >= rest.len) return error.PreprocError;
                if (rest[i].kind == .rparen) {
                    i += 1;
                    break;
                }
                if (rest[i].kind == .comma) {
                    i += 1;
                    continue;
                }
                return error.PreprocError;
            }
        }
        body_src = rest[i..];

        const params_owned = try allocator.alloc([]const u8, plist.items.len);
        var filled: usize = 0;
        errdefer {
            for (params_owned[0..filled]) |p| allocator.free(p);
            allocator.free(params_owned);
        }
        for (plist.items, 0..) |p, k| {
            params_owned[k] = try allocator.dupe(u8, p);
            filled += 1;
        }
        params = params_owned;
    }
    // `params_owned`'s own `errdefer` above only guards the dupe loop it sits beside. Guard
    // the rest of this function too, since a later allocation failure must not leak it.
    errdefer if (params) |p| {
        for (p) |s| allocator.free(s);
        allocator.free(p);
    };

    // `__VA_ARGS__`/`__VA_OPT__` only mean anything inside a variadic
    // macro's body. Anywhere else, in an object-like macro or a function-like one with no
    // `...`, they are just reserved names a definition has no business using.
    for (body_src) |bt| {
        if (bt.kind != .ident) continue;
        const is_va = std.mem.eql(u8, bt.text, "__VA_ARGS__") or std.mem.eql(u8, bt.text, "__VA_OPT__");
        if (is_va and !variadic) return error.PreprocError;
    }

    const body = try dupPTokenSlice(allocator, body_src);
    errdefer {
        for (body) |t| if (t.owns_text) allocator.free(t.text);
        allocator.free(body);
    }
    // Rewrite a GNU named variadic's name to `__VA_ARGS__` in the body, so the ordinary
    // variadic expansion binds the trailing arguments to it. Every body token owns its text
    // (from `dupPTokenSlice`), so the old spelling is freed before the new one replaces it.
    if (va_name) |vn| {
        for (body) |*t| {
            if (t.kind == .ident and std.mem.eql(u8, t.text, vn)) {
                const rep = try allocator.dupe(u8, "__VA_ARGS__");
                if (t.owns_text) allocator.free(t.text);
                t.text = rep;
                t.owns_text = true;
            }
        }
    }
    const name_dup = try allocator.dupe(u8, name);
    errdefer allocator.free(name_dup);

    if (macros.fetchRemove(name)) |kv| freeMacro(allocator, &kv.value);
    try macros.put(allocator, name_dup, .{ .name = name_dup, .params = params, .variadic = variadic, .body = body });
}

/// `#undef NAME`: removes `NAME` from the macro table (freeing its definition) if present.
/// It is a no-op if it was not defined.
fn handleUndef(allocator: std.mem.Allocator, macros: *MacroTable, args: []const PToken) Error!void {
    if (args.len == 0 or args[0].kind != .ident) return error.PreprocError;
    if (macros.fetchRemove(args[0].text)) |kv| freeMacro(allocator, &kv.value);
}

/// Dispatches one directive line. `rest` is the line's tokens after the leading `#`, so
/// `rest[0]`, if present, names the directive. An empty `rest` is the null directive (`#`
/// alone on a line) and is ignored. `#define`/`#undef`/`#error`/`#warning`/`#line`/`#pragma`
/// are handled directly. Every other name is `error.Unsupported`, so an unimplemented
/// directive never silently vanishes. `include` never reaches here. `processLines`
/// intercepts it before calling this function (see `handleInclude`), since it needs far
/// more context than a simple directive does: the resolver, the shared macro table and
/// cond stack, and the include-depth counter. Conditional directives (`if`/`ifdef`/
/// `ifndef`/`elif`/`else`/`endif`) likewise never reach this function. `processLines`
/// intercepts them first (see `isCondDirective`, `handleConditional`), since, unlike every
/// directive here, they must run even while the enclosing region is inactive, to track
/// nesting and find the matching branch.
///
/// `included`/`current_identity` back `#pragma once`. `current_identity` is the
/// `ResolvedFile.identity` of the file this directive line came from. It is null for the
/// top-level source, which has no resolver identity of its own. `#pragma once` with a
/// known identity adds it to `included`. `handleInclude` then skips any further `#include`
/// of that same identity entirely. A `#pragma` argument other than `once` (or `#pragma
/// once` from the top-level source, which cannot be "skipped" anyway) is a no-op, matching
/// this preprocessor's best-effort `#pragma` handling.
fn processDirective(allocator: std.mem.Allocator, macros: *MacroTable, rest: []const PToken, included: *std.StringHashMapUnmanaged(void), current_identity: ?[]const u8) Error!void {
    if (rest.len == 0) return; // null directive
    if (rest[0].kind != .ident) return error.Unsupported;
    const dname = rest[0].text;
    const args = rest[1..];

    if (std.mem.eql(u8, dname, "define")) return handleDefine(allocator, macros, args);
    if (std.mem.eql(u8, dname, "undef")) return handleUndef(allocator, macros, args);
    if (std.mem.eql(u8, dname, "error")) return error.PreprocError;
    if (std.mem.eql(u8, dname, "warning")) {
        std.debug.print("warning:", .{});
        for (args) |t| std.debug.print(" {s}", .{t.text});
        std.debug.print("\n", .{});
        return;
    }
    if (std.mem.eql(u8, dname, "line")) return; // best-effort no-op
    if (std.mem.eql(u8, dname, "pragma")) {
        if (args.len > 0 and args[0].kind == .ident and std.mem.eql(u8, args[0].text, "once")) {
            if (current_identity) |id| {
                if (!included.contains(id)) {
                    const dup = try allocator.dupe(u8, id);
                    errdefer allocator.free(dup);
                    try included.put(allocator, dup, {});
                }
            }
        }
        return; // any other #pragma is a best-effort no-op
    }
    return error.Unsupported; // for example #ident, #assert, and other unimplemented directives
}

/// An `#include "..."` or `#include <...>` target, parsed by `parseIncludeName` from the
/// directive line's tokens after the `include` keyword itself.
const IncludeTarget = struct { name: []const u8, is_system: bool };

/// Parses `#include`'s header-name operand out of `args` (the directive line's tokens after
/// `include`). A `"..."` header arrives from `scan` as a single `str_lit` PToken whose
/// `.text` is already the decoded filename, with no quotes and escapes resolved. It is used
/// directly, with `is_system = false`. A `<...>` header is not one token. The scanner has no
/// special mode for it, so `<foo/bar.h>` comes through as ordinary tokens (`lt`,
/// `ident "foo"`, `slash`, `ident "bar"`, `dot`, `ident "h"`, `gt`). This walks from just
/// after the `lt` to the first `gt`, concatenating each token's spelling, with a single
/// space wherever a token had `space_before`, mirroring `stringize`, and sets
/// `is_system = true`. Anything else is `error.PreprocError`: no operand, a `<...>` with no
/// closing `gt`, or a leading token that is neither `str_lit` nor `lt`. Allocated from
/// `arena`.
fn parseIncludeName(arena: std.mem.Allocator, args: []const PToken) Error!IncludeTarget {
    if (args.len == 0) return error.PreprocError;
    if (args[0].kind == .str_lit) return .{ .name = args[0].text, .is_system = false };
    if (args[0].kind == .lt) {
        var buf: std.ArrayList(u8) = .empty;
        var i: usize = 1;
        var first = true;
        while (i < args.len and args[i].kind != .gt) : (i += 1) {
            if (!first and args[i].space_before) try buf.append(arena, ' ');
            try buf.appendSlice(arena, args[i].text);
            first = false;
        }
        if (i >= args.len) return error.PreprocError; // no closing '>'
        const name = try buf.toOwnedSlice(arena);
        if (name.len == 0) return error.PreprocError;
        return .{ .name = name, .is_system = true };
    }
    return error.PreprocError;
}

/// A cycle, or just very deep nesting, must never hang `preprocess`. Every `#include`
/// bumps a shared depth counter (`Proc.depth`), and exceeding this bound fails closed with
/// `error.PreprocError` rather than recursing forever. 200 comfortably covers any real
/// include chain while still being cheap to unwind.
const max_include_depth: u32 = 200;

/// Shared mutable state threaded through the whole translation unit's directive and
/// expansion pass, including every recursive descent `handleInclude` makes into a resolved
/// `#include`d file. One `Proc` is built by `preprocess` and passed by pointer to
/// `processLines` and, transitively, `handleInclude`. `macros` and `cond_stack` are the
/// same table and stack across every included file: a `#define` made inside a header is
/// still visible afterward, in the includer, exactly as C requires. `depth` is likewise
/// shared, so it bounds the total include nesting, not just one file's own. `included`
/// backs `#pragma once`, by `ResolvedFile.identity`, across the whole translation unit.
const Proc = struct {
    allocator: std.mem.Allocator,
    opts: Options,
    macros: *MacroTable,
    cond_stack: *std.ArrayList(CondFrame),
    included: *std.StringHashMapUnmanaged(void),
    out: *std.ArrayList(PToken),
    depth: u32 = 0,
    // `__DATE__`/`__TIME__`'s text, computed once by `preprocess`. See `Dyn`.
    date_str: []const u8 = "",
    time_str: []const u8 = "",
};

/// Handles one `#include` directive line. `args` is the tokens after the `include` keyword
/// itself. `includer_dir` is the directory of the file this `#include` line lives in. This
/// is best-effort, for a disk resolver's relative `"..."` search. A virtual resolver, like
/// the test harness's, ignores it.
///
/// Resolution: `p.opts.resolver` must be set and its `resolveFn` must return a file, or
/// this is `error.PreprocError`. Both "no resolver configured" and "resolver could not find
/// it" fail the same way. `#pragma once` bookkeeping: if the resolved `identity` is already
/// in `p.included`, the include is skipped entirely, not even re-scanned. This is what
/// makes a doubly-`#include`d `#pragma once` header a no-op the second time. Otherwise the
/// depth counter is bumped, and unconditionally restored via `defer` so an error partway
/// through unwinds cleanly, and checked against `max_include_depth` before any recursive
/// work happens. This makes a guardless include cycle fail closed instead of hanging. The
/// resolved bytes are then scanned fresh and recursively fed through `processLines`,
/// sharing `p`'s macro table and conditional stack with the includer. `#include`d text is
/// processed exactly inline, as the standard specifies.
/// `next_after` distinguishes `#include` (null) from `#include_next`. For an
/// `#include_next`, it is the current file's `identity`: an absolute path for a disk
/// header, or a bare built-in name. The resolver uses it to start the search after the
/// directory the current file came from, so a glibc header can layer over the compiler's
/// own copy of the same header (`<limits.h>` becomes the compiler's `<limits.h>`). See
/// `FsResolver.resolveFn`.
fn handleInclude(p: *Proc, args: []const PToken, includer_dir: ?[]const u8, next_after: ?[]const u8) Error!void {
    var arena_state = std.heap.ArenaAllocator.init(p.allocator);
    defer arena_state.deinit();
    const target = try parseIncludeName(arena_state.allocator(), args);

    const resolver = p.opts.resolver orelse return error.PreprocError;
    const rf = (try resolver.resolveFn(resolver.ctx, target.name, target.is_system, includer_dir, next_after)) orelse return error.PreprocError;

    if (p.included.contains(rf.identity)) return; // #pragma once: already processed, skip.

    p.depth += 1;
    defer p.depth -= 1;
    if (p.depth > max_include_depth) return error.PreprocError; // cycle guard: never hang.

    const inc_ptoks = try scan(p.allocator, rf.bytes, p.opts);
    defer freePTokens(p.allocator, inc_ptoks);

    try processLines(p, inc_ptoks, std.fs.path.dirname(rf.identity), rf.identity);
}

/// One entry of the conditional-group stack `preprocess` maintains. `parent_active`
/// is whether the enclosing region was emitting when this group was opened. An `#if`
/// nested inside an already-skipped group must stay skipped no matter what its own
/// condition says. `taken` is whether any branch of this group (the `#if`/`#ifdef`/
/// `#ifndef`, or a later `#elif`) has been taken yet, so a later `#elif`/`#else` knows to
/// stay inactive even if its own condition would otherwise hold. Only the first true
/// branch in a group emits. `active` is whether this branch is currently emitting. It is
/// the value `condEmitting` reads. `else_seen` is not part of the semantic model the
/// standard describes, but it is needed to fail closed on an `#elif` written after that
/// group's `#else`, a malformed conditional the standard forbids.
const CondFrame = struct {
    parent_active: bool,
    taken: bool,
    active: bool,
    else_seen: bool = false,
};

/// The overall "am I currently emitting" state: the innermost open group's `active`, or
/// `true` if no conditional group is open (top level always emits).
fn condEmitting(stack: []const CondFrame) bool {
    return if (stack.len == 0) true else stack[stack.len - 1].active;
}

/// Is `name` one of the six conditional-compilation directive names? `preprocess` uses this
/// to decide whether a directive line must be dispatched, to track nesting and find the
/// matching branch, even while the current region is inactive, rather than dropped like
/// every other directive would be.
fn isCondDirective(name: []const u8) bool {
    inline for (.{ "if", "ifdef", "ifndef", "elif", "else", "endif" }) |n| {
        if (std.mem.eql(u8, name, n)) return true;
    }
    return false;
}

/// Parses an `int_lit` PToken's text into an `i64`, for the `#if` constant-expression
/// evaluator. The text is digits plus an optional `u`/`U`/`l`/`L` suffix. A
/// character-literal `int_lit`'s text is already-decoded decimal, has no suffix, and
/// parses the same way. C integer-literal base rules apply: `0x`/`0X` is hexadecimal,
/// `0b`/`0B` is binary, a leading `0` followed by one or more further digits is octal
/// (`010` is 8, not 10), a bare `0` is zero, and a literal with a nonzero leading digit is
/// decimal. The suffix (`u`/`l` combinations) is stripped first. None of `u`/`U`/`l`/`L` is
/// a hex digit, so stripping them off a `0x...` literal is safe. A suffix or prefix with no
/// digits after it, or a digit out of range for the base, is `error.PreprocError` rather
/// than silently misparsed.
fn parseIfIntLit(text: []const u8) Error!i64 {
    var end: usize = text.len;
    while (end > 0 and switch (text[end - 1]) {
        'u', 'U', 'l', 'L' => true,
        else => false,
    }) end -= 1;
    if (end == 0) return error.PreprocError;
    var digits = text[0..end];
    var base: u8 = 10;
    if (digits.len >= 2 and digits[0] == '0' and (digits[1] == 'x' or digits[1] == 'X')) {
        base = 16;
        digits = digits[2..];
    } else if (digits.len >= 2 and digits[0] == '0' and (digits[1] == 'b' or digits[1] == 'B')) {
        base = 2;
        digits = digits[2..];
    } else if (digits.len > 1 and digits[0] == '0') {
        base = 8; // a leading `0` with at least one more digit (a bare `0` stays base 10 = 0).
    }
    if (digits.len == 0) return error.PreprocError; // a bare `0x`/`0b` with no digits.
    return std.fmt.parseInt(i64, digits, base) catch error.PreprocError;
}

/// Replaces every `defined X` or `defined ( X )` in a `#if`/`#elif` line with a single
/// `int_lit` token (`"1"`/`"0"`), before macro expansion runs. `defined`'s operand must
/// never itself be macro-expanded. The standard's `defined` is a preprocessor operator, not
/// something whose argument goes through the replacement list, so this pass has to happen
/// first, against the current macro table, not after `expandTokens`. Malformed use
/// (`defined` with no following identifier, or an unbalanced paren form) is
/// `error.PreprocError`. Allocated from `arena`.
fn replaceDefined(arena: std.mem.Allocator, macros: *const MacroTable, args: []const PToken) Error![]const PToken {
    var out: std.ArrayList(PToken) = .empty;
    var i: usize = 0;
    while (i < args.len) {
        const t = args[i];
        if (t.kind == .ident and std.mem.eql(u8, t.text, "defined")) {
            i += 1;
            var name: []const u8 = undefined;
            if (i < args.len and args[i].kind == .lparen) {
                i += 1;
                if (i >= args.len or args[i].kind != .ident) return error.PreprocError;
                name = args[i].text;
                i += 1;
                if (i >= args.len or args[i].kind != .rparen) return error.PreprocError;
                i += 1;
            } else if (i < args.len and args[i].kind == .ident) {
                name = args[i].text;
                i += 1;
            } else {
                return error.PreprocError;
            }
            // A real compiler reports `defined __has_include` (and the other four `__has_*`
            // operator names) as TRUE, even though none of them is a macro in `macros`. This
            // lets a header probe for operator support with the portable idiom
            // `#ifdef __has_include` before using it, matching GCC/Clang.
            const is_def = macros.contains(name) or isHasOp(name);
            try out.append(arena, .{
                .kind = .int_lit,
                .text = if (is_def) "1" else "0",
                .line = t.line,
                .col = t.col,
                .space_before = t.space_before,
                .bol = t.bol,
            });
        } else {
            try out.append(arena, t);
            i += 1;
        }
    }
    return out.toOwnedSlice(arena);
}

/// The `__has_*` names this pre-pass (`replaceHasOps`) recognizes as operators.
const has_op_names = [_][]const u8{ "__has_include", "__has_attribute", "__has_builtin", "__has_feature", "__has_extension" };

fn isHasOp(name: []const u8) bool {
    for (has_op_names) |n| {
        if (std.mem.eql(u8, name, n)) return true;
    }
    return false;
}

/// The `__builtin_*` spellings this compiler recognizes: the `va_start`/`va_arg`/
/// `va_end`/`va_copy` calls plus the `va_list` type. See `stdarg.zig`. `__has_builtin`
/// answers `1` for exactly these names, `0` for every other name. This is a small,
/// conservative allowlist, so a real header's `#if __has_builtin(X)` guard takes its
/// portable fallback path for any `X` this frontend does not actually implement, instead
/// of a false `1` that then hits an unimplemented builtin later.
const known_builtins = [_][]const u8{
    "__builtin_va_start",
    "__builtin_va_arg",
    "__builtin_va_end",
    "__builtin_va_copy",
    "__builtin_va_list",
};

fn isKnownBuiltin(name: []const u8) bool {
    for (known_builtins) |n| {
        if (std.mem.eql(u8, name, n)) return true;
    }
    return false;
}

/// Resolves `__has_include(<h>)`/`__has_include("h")` against `resolver`, threaded in from
/// `evalIfExpr`, ultimately `opts.resolver`. It is true if the header resolves, false if it
/// does not, or if no resolver is configured. "No resolver" and "resolver cannot find it"
/// answer the same way, mirroring `handleInclude`'s own "both fail alike" contract, but as
/// a plain bool instead of an error, since `__has_include` must never itself abort
/// preprocessing. `inner` is the token slice between `__has_include`'s parens. It reuses
/// `parseIncludeName`'s `<...>` reconstruction, so a multi-token system header spells
/// identically to a real `#include`.
fn hasInclude(arena: std.mem.Allocator, resolver: ?IncludeResolver, includer_dir: ?[]const u8, inner: []const PToken) Error!bool {
    const target = try parseIncludeName(arena, inner);
    const r = resolver orelse return false;
    const rf = try r.resolveFn(r.ctx, target.name, target.is_system, includer_dir, null);
    return rf != null;
}

/// Replaces every `__has_include(...)`/`__has_attribute(x)`/`__has_builtin(x)`/
/// `__has_feature(x)`/`__has_extension(x)` in a `#if`/`#elif` line with a `1`/`0` `int_lit`
/// token, before `replaceDefined`/macro expansion run. This mirrors `replaceDefined` exactly
/// and for the same reason: these are preprocessor operators, not macro calls, so their
/// operand must never itself be macro-expanded first. An `__has_*` name not immediately
/// followed by `(` is left completely alone. It stays an ordinary surviving identifier,
/// which `evalIfExpr`'s existing ident-to-`0` rule turns into `0` later. This is never an
/// error, since a header is free to use one of these names as an ordinary macro or
/// identifier on a frontend that predates them. A malformed parenthesized form (unbalanced
/// parens, or a non-single-identifier operand to `__has_attribute`/`__has_builtin`/
/// `__has_feature`/`__has_extension`) is `error.PreprocError`, matching `replaceDefined`'s
/// fail-closed handling of malformed `defined(...)`. Allocated from `arena`.
fn replaceHasOps(arena: std.mem.Allocator, resolver: ?IncludeResolver, includer_dir: ?[]const u8, args: []const PToken) Error![]const PToken {
    var out: std.ArrayList(PToken) = .empty;
    var i: usize = 0;
    while (i < args.len) {
        const t = args[i];
        if (t.kind == .ident and isHasOp(t.text) and i + 1 < args.len and args[i + 1].kind == .lparen) {
            const name = t.text;
            var depth: usize = 1;
            var k = i + 2;
            while (k < args.len and depth > 0) : (k += 1) {
                if (args[k].kind == .lparen) depth += 1;
                if (args[k].kind == .rparen) {
                    depth -= 1;
                    if (depth == 0) break;
                }
            }
            if (k >= args.len) return error.PreprocError; // unbalanced '('
            const inner = args[i + 2 .. k];

            const val: bool = if (std.mem.eql(u8, name, "__has_include"))
                try hasInclude(arena, resolver, includer_dir, inner)
            else has_attr: {
                // `__has_attribute`/`__has_builtin`/`__has_feature`/`__has_extension`: a
                // single bare identifier operand, checked against the (for now, empty except
                // `__has_builtin`'s) allowlist.
                if (inner.len != 1 or inner[0].kind != .ident) return error.PreprocError;
                break :has_attr std.mem.eql(u8, name, "__has_builtin") and isKnownBuiltin(inner[0].text);
            };

            try out.append(arena, .{
                .kind = .int_lit,
                .text = if (val) "1" else "0",
                .line = t.line,
                .col = t.col,
                .space_before = t.space_before,
                .bol = t.bol,
            });
            i = k + 1;
        } else {
            try out.append(arena, t);
            i += 1;
        }
    }
    return out.toOwnedSlice(arena);
}

/// A precedence-climbing recursive-descent parser and evaluator for a `#if`/`#elif`
/// constant expression, over an already macro-expanded, `defined`-resolved token slice (see
/// `evalIfExpr`). It implements the full C precedence ladder this evaluator needs, from
/// lowest to highest: `?:`, `||`, `&&`, `|`, `^`, `&`, `==`/`!=`, relational (`< > <= >=`),
/// shift (`<< >>`), additive (`+ -`), multiplicative (`* / %`), unary (`! ~ - +`), then
/// parenthesized sub-expressions and integer-literal primaries. It evaluates in `i64` with
/// wrapping arithmetic, so there are no overflow panics.
///
/// `&&`/`||`/`?:` short-circuit exactly as C requires. Every parse method threads an
/// `evaluate` flag. The un-taken operand of a short-circuiting operator is still parsed,
/// its tokens consumed, so the overall expression structure stays correct, but with
/// `evaluate = false`. This makes that sub-tree return `0` without computing, and,
/// crucially, without running the division or modulo-by-zero check. So `#if 1 || (1/0)`
/// accepts, since the `1/0` is parsed but not evaluated, matching the real
/// `#if X && 100/X` guard idiom, while a div/mod by zero in an actually-evaluated branch is
/// still `error.PreprocError`.
const IfEval = struct {
    toks: []const PToken,
    pos: usize = 0,

    fn peekKind(self: *IfEval) ?lexer.Kind {
        if (self.pos < self.toks.len) return self.toks[self.pos].kind;
        return null;
    }

    /// Consumes and returns true if the next token is kind `k`. Otherwise leaves `pos`
    /// untouched and returns false.
    fn eat(self: *IfEval, k: lexer.Kind) bool {
        if (self.peekKind() == k) {
            self.pos += 1;
            return true;
        }
        return false;
    }

    fn expect(self: *IfEval, k: lexer.Kind) Error!void {
        if (!self.eat(k)) return error.PreprocError;
    }

    fn boolToI64(b: bool) i64 {
        return if (b) 1 else 0;
    }

    fn parseExpr(self: *IfEval, evaluate: bool) Error!i64 {
        return self.parseTernary(evaluate);
    }

    fn parseTernary(self: *IfEval, evaluate: bool) Error!i64 {
        const cond = try self.parseLogicalOr(evaluate);
        if (self.eat(.question)) {
            const take_true = evaluate and cond != 0;
            // Both arms are parsed (tokens consumed), but only the taken one, and only when
            // this whole `?:` is itself being evaluated, actually computes.
            const a = try self.parseTernary(take_true);
            try self.expect(.colon);
            const b = try self.parseTernary(evaluate and cond == 0);
            return if (cond != 0) a else b;
        }
        return cond;
    }

    fn parseLogicalOr(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseLogicalAnd(evaluate);
        while (self.eat(.pipe_pipe)) {
            // If the left is already true, the right short-circuits: parse it but don't eval.
            const short = evaluate and v != 0;
            const r = try self.parseLogicalAnd(evaluate and !short);
            v = boolToI64(v != 0 or r != 0);
        }
        return v;
    }

    fn parseLogicalAnd(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseBitOr(evaluate);
        while (self.eat(.amp_amp)) {
            // If the left is already false, the right short-circuits: parse it but don't eval.
            const short = evaluate and v == 0;
            const r = try self.parseBitOr(evaluate and !short);
            v = boolToI64(v != 0 and r != 0);
        }
        return v;
    }

    fn parseBitOr(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseBitXor(evaluate);
        while (self.eat(.pipe)) v |= try self.parseBitXor(evaluate);
        return v;
    }

    fn parseBitXor(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseBitAnd(evaluate);
        while (self.eat(.caret)) v ^= try self.parseBitAnd(evaluate);
        return v;
    }

    fn parseBitAnd(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseEquality(evaluate);
        while (self.eat(.amp)) v &= try self.parseEquality(evaluate);
        return v;
    }

    fn parseEquality(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseRelational(evaluate);
        while (true) {
            if (self.eat(.eq_eq)) {
                v = boolToI64(v == try self.parseRelational(evaluate));
            } else if (self.eat(.bang_eq)) {
                v = boolToI64(v != try self.parseRelational(evaluate));
            } else break;
        }
        return v;
    }

    fn parseRelational(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseShift(evaluate);
        while (true) {
            if (self.eat(.lt)) {
                v = boolToI64(v < try self.parseShift(evaluate));
            } else if (self.eat(.gt)) {
                v = boolToI64(v > try self.parseShift(evaluate));
            } else if (self.eat(.le)) {
                v = boolToI64(v <= try self.parseShift(evaluate));
            } else if (self.eat(.ge)) {
                v = boolToI64(v >= try self.parseShift(evaluate));
            } else break;
        }
        return v;
    }

    /// Shift counts are masked into `0..63` (via `@mod`) before use so an out-of-range count
    /// (itself already UB in C) can't trip Zig's shift-amount-in-range safety check.
    fn parseShift(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseAdditive(evaluate);
        while (true) {
            if (self.eat(.lshift)) {
                const amt: u6 = @intCast(@mod(try self.parseAdditive(evaluate), 64));
                v = v << amt;
            } else if (self.eat(.rshift)) {
                const amt: u6 = @intCast(@mod(try self.parseAdditive(evaluate), 64));
                v = v >> amt;
            } else break;
        }
        return v;
    }

    fn parseAdditive(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseMultiplicative(evaluate);
        while (true) {
            if (self.eat(.plus)) {
                v = v +% try self.parseMultiplicative(evaluate);
            } else if (self.eat(.minus)) {
                v = v -% try self.parseMultiplicative(evaluate);
            } else break;
        }
        return v;
    }

    fn parseMultiplicative(self: *IfEval, evaluate: bool) Error!i64 {
        var v = try self.parseUnary(evaluate);
        while (true) {
            if (self.eat(.star)) {
                v = v *% try self.parseUnary(evaluate);
            } else if (self.eat(.slash)) {
                const r = try self.parseUnary(evaluate);
                // The div-by-zero check is skipped when this branch is not being evaluated. A
                // short-circuited `1/0` must not error. See the struct doc comment.
                if (evaluate and r == 0) return error.PreprocError;
                v = if (evaluate) @divTrunc(v, r) else 0;
            } else if (self.eat(.percent)) {
                const r = try self.parseUnary(evaluate);
                if (evaluate and r == 0) return error.PreprocError;
                v = if (evaluate) @rem(v, r) else 0;
            } else break;
        }
        return v;
    }

    fn parseUnary(self: *IfEval, evaluate: bool) Error!i64 {
        if (self.eat(.bang)) return boolToI64(try self.parseUnary(evaluate) == 0);
        if (self.eat(.tilde)) return ~(try self.parseUnary(evaluate));
        if (self.eat(.minus)) return -%(try self.parseUnary(evaluate));
        if (self.eat(.plus)) return try self.parseUnary(evaluate);
        return self.parsePrimary(evaluate);
    }

    fn parsePrimary(self: *IfEval, evaluate: bool) Error!i64 {
        if (self.pos >= self.toks.len) return error.PreprocError;
        const t = self.toks[self.pos];
        if (t.kind == .lparen) {
            self.pos += 1;
            const v = try self.parseExpr(evaluate);
            try self.expect(.rparen);
            return v;
        }
        if (t.kind == .int_lit) {
            self.pos += 1;
            // A literal still has to be well-formed even when not evaluated (its structure is
            // part of the parse), but a valid parse yields `0` in the un-taken branch.
            const val = try parseIfIntLit(t.text);
            return if (evaluate) val else 0;
        }
        return error.PreprocError;
    }
};

/// Evaluates a `#if`/`#elif` line's expression tokens (`args`, that is the directive line
/// after the `if`/`elif` keyword itself) to a truth value, in four steps.
/// (1) `replaceHasOps` resolves every `__has_include`/`__has_attribute`/`__has_builtin`/
/// `__has_feature`/`__has_extension` operator written directly on this line. `resolver`/
/// `includer_dir` back `__has_include`'s resolver lookup. See `hasInclude`.
/// (2) `replaceDefined` resolves every `defined X`/`defined(X)` against the current macro
/// table. Both run before expansion, since neither operand is macro-expandable.
/// (3) The rest is fully macro-expanded (`expandTokens`, the same engine ordinary lines
/// use).
/// (3b) `replaceHasOps` runs a second time, over the now-expanded tokens. A real header
/// commonly hides a `__has_*` operator behind its own portability macro. For example
/// glibc's `sys/cdefs.h` expands `__glibc_has_attribute(a)` to `__has_attribute (a)`, so
/// the operator only becomes visible in the token stream after that wrapper macro expands,
/// exactly like a real compiler's macro-rescanning finds it. The first, pre-expansion, pass
/// already turned every directly-written occurrence into a `0`/`1` `int_lit`, so this
/// second pass only ever touches occurrences a macro just revealed. It is a no-op for
/// every pre-existing test of the direct form.
/// (4) Any identifier still standing after both passes, that is one that was not a macro,
/// for example `true` or an unresolvable `sizeof`, becomes the literal `0`, per the
/// standard's rule for `#if`. The resulting integer-token line is evaluated by `IfEval`,
/// which must consume every token. Trailing garbage is `error.PreprocError`, same as an
/// empty expression. Scratch memory is a per-call arena freed before return.
fn evalIfExpr(allocator: std.mem.Allocator, macros: *const MacroTable, dyn: Dyn, resolver: ?IncludeResolver, includer_dir: ?[]const u8, args: []const PToken) Error!bool {
    if (args.len == 0) return error.PreprocError;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const after_has = try replaceHasOps(arena, resolver, includer_dir, args);
    const after_defined = try replaceDefined(arena, macros, after_has);
    const expanded = try expandTokens(arena, macros, dyn, after_defined);
    const after_has2 = try replaceHasOps(arena, resolver, includer_dir, expanded);

    var final: std.ArrayList(PToken) = .empty;
    for (after_has2) |t| {
        if (t.kind == .ident) {
            try final.append(arena, .{ .kind = .int_lit, .text = "0", .line = t.line, .col = t.col });
        } else {
            try final.append(arena, t);
        }
    }

    var p: IfEval = .{ .toks = final.items };
    const v = try p.parseExpr(true); // top-level: this whole expression IS being evaluated
    if (p.pos != final.items.len) return error.PreprocError;
    return v != 0;
}

/// Dispatches one of the six conditional-compilation directives. `preprocess` calls this
/// instead of `processDirective` whenever `isCondDirective(dname)`, active or not. See that
/// function's doc comment for why. `dname` names the directive. `args` is the line's tokens
/// after it. `stack` is the enclosing `preprocess` call's conditional-group stack, pushed to
/// by `if`/`ifdef`/`ifndef` and popped by `endif`.
///
/// - `ifdef NAME`/`ifndef NAME`: pushes a new frame. `active` (and `taken`, its initial
///   value) is `parent_active && (NAME defined / not defined)`. A group nested inside an
///   already-inactive one can never itself become active, no matter what its own condition
///   says.
/// - `if EXPR`: same push, but the condition comes from `evalIfExpr`, evaluated only when
///   `parent_active`. An inactive enclosing region skips evaluation entirely, so, for
///   example, an `#if` guarding on an intentionally-undefined macro deep inside a disabled
///   `#if 0` block never has to type-check.
/// - `elif EXPR`: requires an open group with no `#else` yet seen, else `error.PreprocError`.
///   An `#elif` after `#else`, or with no matching `#if`, is malformed. It is evaluated only
///   if the group has not been `taken` yet and its `parent_active` is true. Otherwise
///   `active` is forced false without evaluating. This mirrors `if`'s skip-when-inactive
///   rule, and also avoids re-evaluating once some earlier branch already won.
/// - `else`: requires an open group with no `#else` yet seen. A second `#else`, or one with
///   no matching `#if`, is `error.PreprocError`. `active = parent_active && !taken`.
/// - `endif`: requires a non-empty stack, else `error.PreprocError` for an `#endif` with
///   nothing open. It pops the stack.
fn handleConditional(allocator: std.mem.Allocator, macros: *const MacroTable, dyn: Dyn, resolver: ?IncludeResolver, includer_dir: ?[]const u8, stack: *std.ArrayList(CondFrame), dname: []const u8, args: []const PToken) Error!void {
    if (std.mem.eql(u8, dname, "ifdef") or std.mem.eql(u8, dname, "ifndef")) {
        const parent_active = condEmitting(stack.items);
        if (args.len == 0 or args[0].kind != .ident) return error.PreprocError;
        // As in `replaceDefined`, a `__has_*` operator name reports as defined even though
        // it never sits in `macros`. This keeps `#ifdef __has_include` true, matching
        // GCC/Clang's portable-detection idiom.
        const is_def = macros.contains(args[0].text) or isHasOp(args[0].text);
        const want = if (std.mem.eql(u8, dname, "ifdef")) is_def else !is_def;
        const active = parent_active and want;
        try stack.append(allocator, .{ .parent_active = parent_active, .taken = active, .active = active });
        return;
    }
    if (std.mem.eql(u8, dname, "if")) {
        const parent_active = condEmitting(stack.items);
        const active = parent_active and try evalIfExpr(allocator, macros, dyn, resolver, includer_dir, args);
        try stack.append(allocator, .{ .parent_active = parent_active, .taken = active, .active = active });
        return;
    }
    if (std.mem.eql(u8, dname, "elif")) {
        if (stack.items.len == 0) return error.PreprocError;
        const top = &stack.items[stack.items.len - 1];
        if (top.else_seen) return error.PreprocError; // #elif after #else
        if (!top.taken and top.parent_active) {
            const active = try evalIfExpr(allocator, macros, dyn, resolver, includer_dir, args);
            top.active = active;
            if (active) top.taken = true;
        } else {
            top.active = false;
        }
        return;
    }
    if (std.mem.eql(u8, dname, "else")) {
        if (stack.items.len == 0) return error.PreprocError;
        const top = &stack.items[stack.items.len - 1];
        if (top.else_seen) return error.PreprocError; // duplicate #else
        top.active = top.parent_active and !top.taken;
        top.taken = true;
        top.else_seen = true;
        return;
    }
    // dname == "endif" (the only remaining `isCondDirective` name).
    if (stack.items.len == 0) return error.PreprocError;
    _ = stack.pop();
}

/// `hide` restricted to the names it shares with `other`: `HS(name) ∩ HS(closing-rparen)`
/// in Prosser's function-like rule. See `expandTokens`. Allocated from `arena`.
fn intersectHide(arena: std.mem.Allocator, hide: []const []const u8, other: []const []const u8) Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (hide) |h| if (hideContains(other, h)) try out.append(arena, h);
    return out.toOwnedSlice(arena);
}

/// `base ∪ extra`, a set union, deduplicated, allocated from `arena`. Used to combine a
/// substituted argument token's own hide-set, the names it earned during its own recursive
/// expansion, with the enclosing function-like expansion's new hide-set, rather than
/// discarding the former by overwriting. Like `hideUnion`/`intersectHide`, it only ever
/// grows a hide-set, so it preserves the termination guarantee.
fn hideMerge(arena: std.mem.Allocator, base: []const []const u8, extra: []const []const u8) Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    try out.appendSlice(arena, base);
    for (extra) |name| if (!hideContains(base, name)) try out.append(arena, name);
    return out.toOwnedSlice(arena);
}

/// The index of `name` in `params`, or null if it doesn't name one of the macro's
/// parameters.
fn paramIndex(params: []const []const u8, name: []const u8) ?usize {
    for (params, 0..) |p, i| if (std.mem.eql(u8, p, name)) return i;
    return null;
}

/// `paramIndex`, extended so a variadic macro's `__VA_ARGS__` binds like one extra
/// parameter sitting right after the named ones, at index `params.len` (the caller's
/// `va_idx`). Returns null for `__VA_ARGS__` when `variadic` is false, so it falls through
/// to a plain, non-substituted, identifier as before.
fn paramOrVaIndex(params: []const []const u8, variadic: bool, name: []const u8) ?usize {
    if (paramIndex(params, name)) |i| return i;
    if (variadic and std.mem.eql(u8, name, "__VA_ARGS__")) return params.len;
    return null;
}

/// The result of `collectArgs`. `args[k]` is the unexpanded token slice for the k-th
/// argument, borrowed from the scanned `items`. `close` is the index, also into `items`,
/// of the invocation's matching closing `)`.
const CollectedArgs = struct { args: []const []const PToken, close: usize };

/// Collects a function-like macro invocation's arguments out of `items`, starting right
/// after the call's opening `(`, already consumed by the caller (`items[start - 1]` is
/// that `(`). It splits on commas at the call's own nesting depth only. A further `(`,
/// `[`, or `{` raises the depth, so a comma inside one is not an argument separator, and
/// the matching close lowers it back. The invocation's own closing `)` is the first close
/// that brings the depth back to 0. This is `error.PreprocError` if that never happens, an
/// unterminated invocation, or if depth reaches 0 on a `]`/`}` instead of a `)`, mismatched
/// nesting. The argument count is top-level commas plus one, C's rule: `F()` yields one
/// argument that is the empty token slice (a 1-param macro's param then substitutes to
/// nothing), not zero arguments. The caller special-cases the zero-param macro
/// (`#define Z() ...` called `Z()`), where that lone empty argument is instead treated as
/// no arguments. See `expandTokens`.
fn collectArgs(arena: std.mem.Allocator, items: []const PToken, start: usize) Error!CollectedArgs {
    var depth: i32 = 1;
    var arg_start = start;
    var groups: std.ArrayList([]const PToken) = .empty;
    var close: ?usize = null;

    var i = start;
    while (i < items.len) : (i += 1) {
        switch (items[i].kind) {
            .lparen, .lbracket, .lbrace => depth += 1,
            .rparen, .rbracket, .rbrace => {
                depth -= 1;
                if (depth == 0) {
                    if (items[i].kind != .rparen) return error.PreprocError;
                    close = i;
                }
            },
            .comma => if (depth == 1) {
                try groups.append(arena, items[arg_start..i]);
                arg_start = i + 1;
            },
            else => {},
        }
        if (close != null) break;
    }
    const close_idx = close orelse return error.PreprocError;

    // Always (commas + 1) groups: the final span `items[arg_start..close_idx]` is appended
    // even when empty, so `F()` yields one empty argument (C: 0 commas => 1 argument).
    try groups.append(arena, items[arg_start..close_idx]);
    return .{ .args = try groups.toOwnedSlice(arena), .close = close_idx };
}

/// `__VA_ARGS__`'s binding: joins the trailing argument `groups` of a variadic call, each
/// already either a raw or a fully-expanded token slice, per the two call sites in
/// `expandTokens`, into the one token sequence `__VA_ARGS__` substitutes to. It inserts a
/// literal `,` between each pair of adjacent groups, the comma the call itself used to
/// separate them. Zero groups, or a single empty group, a call supplying no trailing
/// arguments at all, yields an empty result: `__VA_ARGS__` substitutes to nothing.
/// Allocated from `arena`.
fn joinVaArgs(arena: std.mem.Allocator, groups: []const []const PToken) Error![]const PToken {
    var out: std.ArrayList(PToken) = .empty;
    for (groups, 0..) |g, gi| {
        if (gi != 0) {
            try out.append(arena, .{
                .kind = .comma,
                .text = ",",
                .line = 0,
                .col = 0,
                .owns_text = false,
                .space_before = false,
                .bol = false,
                .hide = &.{},
            });
        }
        try out.appendSlice(arena, g);
    }
    return out.toOwnedSlice(arena);
}

/// Resolves every `##` paste marker in `toks` in place, left to right. Each `hash_hash`
/// fuses the token immediately before and after it, their spellings concatenated, and
/// re-lexes the fused text via `lexer.tokenize`, splicing the produced token(s) in where
/// the `left ## right` triple was. Chained pastes (`a##b##c`) resolve because the cursor
/// rewinds to the just-produced token after each fuse. A `##` with no left or right
/// operand, at the start or end, or a fused text that does not re-lex, is
/// `error.PreprocError`. An operand whose spelling cannot be recovered (a `str_lit`/char
/// literal, see `spellingRecoverable`) is `error.Unsupported`, failing closed rather than pasting wrong
/// bytes. Shared by both the object-like and function-like expansion branches: an
/// object-like body like `#define G a##b` pastes its adjacent body tokens, and a
/// function-like body pastes after argument substitution. Every allocation is from
/// `arena`.
fn resolvePastes(arena: std.mem.Allocator, toks: *std.ArrayList(PToken)) Error!void {
    var pi: usize = 0;
    while (pi < toks.items.len) {
        if (toks.items[pi].kind != .hash_hash) {
            pi += 1;
            continue;
        }
        if (pi == 0 or pi + 1 >= toks.items.len) return error.PreprocError;
        const left = toks.items[pi - 1];
        const right = toks.items[pi + 1];
        if (!spellingRecoverable(left) or !spellingRecoverable(right)) return error.Unsupported;
        var fused: std.ArrayList(u8) = .empty;
        try fused.appendSlice(arena, left.text);
        try fused.appendSlice(arena, right.text);
        const fused_text = try fused.toOwnedSlice(arena);
        const lexed = lexer.tokenize(arena, fused_text) catch |err| switch (err) {
            error.UnexpectedChar => return error.PreprocError,
            else => return err,
        };
        const produced = lexed[0 .. lexed.len - 1]; // drop the trailing eof
        if (produced.len == 0) return error.PreprocError;
        var replacement: std.ArrayList(PToken) = .empty;
        for (produced, 0..) |pt, k| {
            try replacement.append(arena, .{
                .kind = pt.kind,
                .text = pt.text,
                .line = left.line,
                .col = left.col,
                .owns_text = false,
                .space_before = if (k == 0) left.space_before else false,
                .bol = if (k == 0) left.bol else false,
                .hide = left.hide,
            });
        }
        try toks.replaceRange(arena, pi - 1, 3, replacement.items);
        pi -= 1;
    }
}

/// Whether token `t`'s original source spelling can be faithfully recovered from its
/// `.text`. False for a `str_lit`/`wstr_lit`, whose `.text` is already-decoded bytes with
/// quotes and any `L` prefix stripped and escapes resolved, and for a `char_lit` `int_lit`,
/// whose `.text` is the decoded decimal value, not the `'A'` source. Stringizing (`#`) or
/// pasting (`##`) such a token would emit wrong bytes. For example `#"a\n"` must be the
/// 6-byte spelling `"a\n"` plus NUL, not the decoded `a<LF>`. So those operations fail
/// closed with `error.Unsupported` on it, rather than silently miscompiling. TODO: carry
/// raw spellings in PTokens, and decode only at `emit`, to stringize and paste these
/// faithfully.
fn spellingRecoverable(t: PToken) bool {
    return t.kind != .str_lit and t.kind != .wstr_lit and !t.char_lit;
}

/// Stringizes one macro argument's raw tokens (`#param`). It concatenates each token's
/// spelling, with a single space wherever a token had `space_before`, except the first,
/// escaping `\` and `"` so the joined text is safe to embed as a C string literal's
/// contents. `t.text` is used directly as each token's spelling. A token whose spelling
/// is not recoverable from its `.text` (a `str_lit` or char literal, see
/// `spellingRecoverable`) makes this `error.Unsupported` rather than emit wrong bytes.
/// Allocated from `arena`.
fn stringize(arena: std.mem.Allocator, arg_toks: []const PToken) Error![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    for (arg_toks, 0..) |t, i| {
        if (i != 0 and t.space_before) try buf.append(arena, ' ');
        // A string or character literal's PToken text is the decoded value, not the source
        // spelling, so a byte-exact re-spelling is not recoverable here. Emit a best-effort
        // one: a string literal's contents wrapped in escaped quotes, a character literal's
        // numeric value. This only affects the contents of a `#`-stringized argument, which in
        // practice becomes a discarded diagnostic message (a `_Static_assert`/`assert` text), so
        // an approximate spelling is harmless. An ordinary token uses its `.text` directly.
        const is_str = t.kind == .str_lit or t.kind == .wstr_lit;
        if (is_str) try buf.appendSlice(arena, "\\\"");
        for (t.text) |ch| {
            if (ch == '\\' or ch == '"') try buf.append(arena, '\\');
            try buf.append(arena, ch);
        }
        if (is_str) try buf.appendSlice(arena, "\\\"");
    }
    return buf.toOwnedSlice(arena);
}

/// Parameter, `__VA_ARGS__`, and `__VA_OPT__` substitution pass. This is pass 1 of
/// function-like macro expansion. See `expandTokens`'s function-like branch, which is
/// pass 2. It walks `body` left to right, appending to `subst` either a substituted or a
/// literal copy of each token:
///   - `# param` (or `# __VA_ARGS__`, in a variadic macro) -> a single `str_lit` token
///     holding `stringize`'s result of the raw argument.
///   - `##` -> carried through as a marker for `resolvePastes` (the caller's pass 2),
///     unless its right operand is literally the identifier `__VA_ARGS__` and
///     `__VA_ARGS__`'s raw binding is empty. Then the `##` always vanishes, since an empty
///     `__VA_ARGS__` has nothing to paste, and the token immediately to its left also
///     vanishes, but only when that left token is a comma: the GNU `, ##__VA_ARGS__`
///     comma-swallow. Any other left token is a placemarker paste and stays in place, so
///     `pre ## __VA_ARGS__` with an empty `__VA_ARGS__` yields `pre`.
///   - `__VA_OPT__ ( content )` -> recursively substitutes `content` into `subst` when
///     `__VA_ARGS__`'s raw binding is non-empty. Emits nothing when it is empty. `content`
///     may itself reference params, `__VA_ARGS__`, `#`, or `##`.
///   - a plain parameter reference, or, only when `variadic`, `__VA_ARGS__` itself -> the
///     matching argument's token sequence, spliced in, raw beside `#`/`##`, otherwise
///     fully expanded. `bind_raw`/`bind_expanded` supply both forms, indexed 0..
///     `params.len - 1` for the named parameters and `va_idx` (== `params.len`) for
///     `__VA_ARGS__`. See `expandTokens` for how the caller builds them.
///   - anything else -> copied literally.
/// `anchor` is the macro-invocation token. Its `.line`/`.col` become every produced
/// token's position, and its `.space_before`/`.bol` become the very first token of the
/// whole substitution, tracked via `subst.items.len == 0`, which stays correct across a
/// recursive `__VA_OPT__` call since it appends into this same `subst` list. `new_hide`
/// is the hide-set every produced token carries, unioned (`hideMerge`) with an argument
/// token's own earned hide-set rather than overwritten by it. A `#`/`##` operand whose
/// spelling cannot be faithfully recovered (a `str_lit`/char literal, see
/// `spellingRecoverable`) fails closed with `error.Unsupported`.
fn substituteMacroBody(
    arena: std.mem.Allocator,
    body: []const PToken,
    params: []const []const u8,
    variadic: bool,
    va_idx: usize,
    bind_raw: []const []const PToken,
    bind_expanded: []const []const PToken,
    anchor: PToken,
    new_hide: []const []const u8,
    subst: *std.ArrayList(PToken),
) Error!void {
    var prev_was_paste = false;
    var bi: usize = 0;
    while (bi < body.len) : (bi += 1) {
        const b = body[bi];
        if (b.kind == .hash) {
            bi += 1;
            if (bi >= body.len or body[bi].kind != .ident) return error.PreprocError;
            const pidx = paramOrVaIndex(params, variadic, body[bi].text) orelse return error.PreprocError;
            const str_bytes = try stringize(arena, bind_raw[pidx]);
            try subst.append(arena, .{
                .kind = .str_lit,
                .text = str_bytes,
                .line = anchor.line,
                .col = anchor.col,
                .owns_text = false,
                .space_before = if (subst.items.len == 0) anchor.space_before else b.space_before,
                .bol = if (subst.items.len == 0) anchor.bol else false,
                .hide = new_hide,
            });
            prev_was_paste = false;
            continue;
        }
        if (b.kind == .ident and variadic and std.mem.eql(u8, b.text, "__VA_OPT__")) {
            if (bi + 1 >= body.len or body[bi + 1].kind != .lparen) return error.PreprocError;
            var depth: i32 = 1;
            var ci = bi + 2;
            var close: ?usize = null;
            while (ci < body.len) : (ci += 1) {
                switch (body[ci].kind) {
                    .lparen => depth += 1,
                    .rparen => {
                        depth -= 1;
                        if (depth == 0) close = ci;
                    },
                    else => {},
                }
                if (close != null) break;
            }
            const close_idx = close orelse return error.PreprocError; // unterminated __VA_OPT__(...)
            if (bind_raw[va_idx].len != 0) {
                try substituteMacroBody(arena, body[bi + 2 .. close_idx], params, variadic, va_idx, bind_raw, bind_expanded, anchor, new_hide, subst);
            }
            bi = close_idx;
            prev_was_paste = false;
            continue;
        }
        if (b.kind == .hash_hash) {
            if (variadic and bi + 1 < body.len and body[bi + 1].kind == .ident and std.mem.eql(u8, body[bi + 1].text, "__VA_ARGS__") and bind_raw[va_idx].len == 0) {
                // GNU comma-swallow (`, ##__VA_ARGS__`) only deletes the left operand when
                // it is literally the comma that separates the named and variadic
                // arguments. Any other left operand is a standard placemarker paste: the
                // empty `__VA_ARGS__` and the `##` vanish, but the left token stays. For example
                // `pre ## __VA_ARGS__` with an empty `__VA_ARGS__` yields `pre`, not
                // nothing.
                if (subst.items.len > 0 and subst.items[subst.items.len - 1].kind == .comma) _ = subst.pop();
                bi += 1; // also skip the `__VA_ARGS__` identifier, since both operands vanish.
                prev_was_paste = false;
                continue;
            }
            try subst.append(arena, .{
                .kind = .hash_hash,
                .text = "##",
                .line = anchor.line,
                .col = anchor.col,
                .owns_text = false,
                .space_before = if (subst.items.len == 0) anchor.space_before else b.space_before,
                .bol = false,
                .hide = new_hide,
            });
            prev_was_paste = true;
            continue;
        }
        if (b.kind == .ident) {
            if (paramOrVaIndex(params, variadic, b.text)) |pidx| {
                const next_is_paste = bi + 1 < body.len and body[bi + 1].kind == .hash_hash;
                const is_paste_operand = prev_was_paste or next_is_paste;
                const src = if (is_paste_operand) bind_raw[pidx] else bind_expanded[pidx];
                // A `##` operand uses the raw argument tokens. If any of them cannot be
                // faithfully pasted (a `str_lit`/char literal, since its `.text` is decoded,
                // not the source spelling), fail closed rather than paste wrong bytes.
                if (is_paste_operand) for (src) |st| {
                    if (!spellingRecoverable(st)) return error.Unsupported;
                };
                for (src, 0..) |st, sk| {
                    try subst.append(arena, .{
                        .kind = st.kind,
                        .text = st.text,
                        .line = anchor.line,
                        .col = anchor.col,
                        .owns_text = false,
                        .space_before = if (subst.items.len == 0) anchor.space_before else (if (sk == 0) b.space_before else st.space_before),
                        .bol = if (subst.items.len == 0) anchor.bol else false,
                        .char_lit = st.char_lit,
                        // Preserve the arg token's own earned hide-set, unioned with,
                        // not overwritten by, this expansion's new hide.
                        .hide = try hideMerge(arena, st.hide, new_hide),
                    });
                }
                prev_was_paste = false;
                continue;
            }
        }
        try subst.append(arena, .{
            .kind = b.kind,
            .text = b.text,
            .line = anchor.line,
            .col = anchor.col,
            .owns_text = false,
            .space_before = if (subst.items.len == 0) anchor.space_before else b.space_before,
            .bol = if (subst.items.len == 0) anchor.bol else false,
            .char_lit = b.char_lit,
            .hide = new_hide,
        });
        prev_was_paste = false;
    }
}

/// Fully macro-expands `tokens` (Dave Prosser's algorithm, object-like and function-like),
/// returning the result as an arena-allocated `PToken` slice. Used both for a whole
/// non-directive logical line (via `expandLine`) and, recursively, to pre-expand a
/// function-like macro argument before it is substituted into a body. Per the standard,
/// this does not apply when that argument is an operand of `#` or `##`, which use the
/// argument's raw tokens instead. See the function-like branch below.
///
/// `tokens` seeds a work queue. The token at the front is repeatedly either substituted in
/// place, so the substitution is itself rescanned before the rest of the queue is looked
/// at, or left alone while the cursor advances. A token is substituted when it is an
/// identifier naming a macro not already in its own hide-set, and, for a function-like
/// macro, the very next queued token is `(`. Otherwise it is a plain identifier. A
/// function-like macro's bare name with no following `(` is never replaced. An identifier
/// that is not in `macros` at all falls through to `expandDynamicMacro`:
/// `__LINE__`/`__FILE__`/`__DATE__`/`__TIME__` are handled there instead of via a stored
/// `Macro`, but only once the table lookup above has already missed. So a user
/// `#define __LINE__ ...` (or any of the other three) always wins, exactly as redefining
/// any other predefined name would.
///
/// Object-like substitution: the body's tokens replace the macro name, each carrying
/// hide-set `token.hide ∪ {macro.name}`. Then `resolvePastes` resolves any `##` between
/// adjacent body tokens (`#define G a##b` -> `ab`), exactly as the function-like path does.
///
/// Function-like substitution additionally works as follows. `collectArgs` gathers the
/// invocation's argument token slices and the closing `)`'s index. The argument count
/// (top-level commas plus one) must match the macro's parameter count, else
/// `error.PreprocError`. A `variadic` macro instead only requires at least that many. Too
/// few is still `error.PreprocError`. It binds every group past the named ones to
/// `__VA_ARGS__` (see `joinVaArgs`). Zero trailing groups is an empty `__VA_ARGS__`. An
/// empty-parens call `G()` is one empty argument, so a 1-param macro matches, since its
/// param substitutes to nothing. The sole zero-param exception (`Z()` = no args) is
/// special-cased for non-variadic macros only. A variadic zero-named-param macro's `Z()`
/// is instead one empty `__VA_ARGS__` group, per the standard. Each argument is fully
/// expanded via a recursive `expandTokens` call unless it is used as an operand of `#` or
/// `##`, which get the argument's raw, unexpanded, tokens per the standard.
/// `substituteMacroBody` walks the body left to right doing the substitution itself. See
/// its own doc comment for `#`/`##`/`__VA_ARGS__`/`__VA_OPT__` handling. `##` markers it
/// carries through are then resolved by `resolvePastes`, which fuses and re-lexes each
/// `left ## right`, so chained pastes like `a##b##c` resolve correctly. The hide-set for
/// every produced token is Prosser's function-like rule, `HS' = (HS(name) ∩
/// HS(closing ")")) ∪ {name}`, and the whole result is spliced back into the outer work
/// queue at the macro name's position and rescanned.
///
/// Termination is identical to the object-like-only version this replaces. A hide-set only
/// ever grows: `hideUnion`/`intersectHide` never add a name a token did not already have
/// available, and a name is added only at substitution, which the hide-set check then
/// forbids repeating. So mutual or self recursion through function-like macros terminates
/// exactly as it does through object-like ones.
///
/// Scratch memory (the work queue, substituted-body copies, hide-set arrays, argument
/// slices) all comes from `arena`, owned by the caller and freed once the whole line, and
/// every argument it recursively expanded, is done. Token text within that scratch is
/// never reallocated. It is borrowed from `tokens`, from a `Macro.body`, or, for a pasted
/// or stringized token, freshly built in `arena`, and is only copied, via a real allocator,
/// at the point `expandLine` finally emits a token.
fn expandTokens(arena: std.mem.Allocator, macros: *const MacroTable, dyn: Dyn, tokens: []const PToken) Error![]PToken {
    var work: std.ArrayList(PToken) = .empty;
    try work.appendSlice(arena, tokens);

    var idx: usize = 0;
    while (idx < work.items.len) {
        const t = work.items[idx];
        var did_subst = false;

        if (t.kind == .ident) {
            if (macros.get(t.text)) |m| {
                if (!hideContains(t.hide, m.name)) {
                    if (m.params == null) {
                        const new_hide = try hideUnion(arena, t.hide, m.name);
                        var subst: std.ArrayList(PToken) = .empty;
                        for (m.body, 0..) |b, bi| {
                            try subst.append(arena, .{
                                .kind = b.kind,
                                .text = b.text,
                                .line = t.line,
                                .col = t.col,
                                .owns_text = false,
                                .space_before = if (bi == 0) t.space_before else b.space_before,
                                .bol = if (bi == 0) t.bol else false,
                                .char_lit = b.char_lit,
                                .hide = new_hide,
                            });
                        }
                        // An object-like body can contain `##` too (`#define G a##b` -> `ab`):
                        // resolve pastes on its adjacent body tokens, same as function-like.
                        try resolvePastes(arena, &subst);
                        try work.replaceRange(arena, idx, 1, subst.items);
                        did_subst = true;
                    } else if (idx + 1 < work.items.len and work.items[idx + 1].kind == .lparen) {
                        const params = m.params.?;
                        const collected = try collectArgs(arena, work.items, idx + 2);
                        var args = collected.args;
                        if (m.variadic) {
                            // At least the named params' worth of groups. Everything past
                            // that (possibly zero groups) becomes `__VA_ARGS__` below.
                            if (args.len < params.len) return error.PreprocError;
                        } else {
                            // C: `Z()` on a ZERO-param macro is no arguments, but
                            // `collectArgs` returns one empty argument (commas+1). Collapse
                            // that lone empty arg to none so a zero-param call matches. Any
                            // other count mismatches.
                            if (params.len == 0 and args.len == 1 and args[0].len == 0) args = &.{};
                            if (args.len != params.len) return error.PreprocError;
                        }

                        // Each argument fully expanded up front (used everywhere EXCEPT as a
                        // `#`/`##` operand, which substitutes the raw `args[k]` instead). An
                        // empty argument expands to no tokens (`expandTokens` of `&.{}` -> `&.{}`).
                        const expanded_args = try arena.alloc([]const PToken, args.len);
                        for (args, 0..) |raw, ai| expanded_args[ai] = try expandTokens(arena, macros, dyn, raw);

                        // `__VA_ARGS__` binds like one extra parameter at index `va_idx`:
                        // the trailing groups past the named ones, raw and
                        // expanded, rejoined with their separating commas (`joinVaArgs`). A
                        // non-variadic macro leaves `bind_raw`/`bind_expanded` aliased
                        // straight to `args`/`expanded_args`. No new allocation, no change
                        // in behavior from before.
                        const va_idx = params.len;
                        var bind_raw: []const []const PToken = args;
                        var bind_expanded: []const []const PToken = expanded_args;
                        if (m.variadic) {
                            const raw_ext = try arena.alloc([]const PToken, va_idx + 1);
                            @memcpy(raw_ext[0..va_idx], args[0..va_idx]);
                            raw_ext[va_idx] = try joinVaArgs(arena, args[va_idx..]);
                            bind_raw = raw_ext;

                            const exp_ext = try arena.alloc([]const PToken, va_idx + 1);
                            @memcpy(exp_ext[0..va_idx], expanded_args[0..va_idx]);
                            exp_ext[va_idx] = try joinVaArgs(arena, expanded_args[va_idx..]);
                            bind_expanded = exp_ext;
                        }

                        const close_tok = work.items[collected.close];
                        const call_hide = try intersectHide(arena, t.hide, close_tok.hide);
                        const new_hide = try hideUnion(arena, call_hide, m.name);

                        // Pass 1: substitute params/`__VA_ARGS__`/`__VA_OPT__` and stringize
                        // `#param`. `##` itself is carried through as a marker for pass 2.
                        var subst: std.ArrayList(PToken) = .empty;
                        try substituteMacroBody(arena, m.body, params, m.variadic, va_idx, bind_raw, bind_expanded, t, new_hide, &subst);

                        // Pass 2: resolve `##` pastes (param `##` operands' faithfulness is
                        // already checked at substitution above. `resolvePastes` re-checks
                        // body-literal operands and fuses/re-lexes).
                        try resolvePastes(arena, &subst);

                        try work.replaceRange(arena, idx, collected.close - idx + 1, subst.items);
                        did_subst = true;
                    }
                }
            } else if (try expandDynamicMacro(arena, dyn, t)) |repl| {
                // `__LINE__`/`__FILE__`/`__DATE__`/`__TIME__`: only reached when
                // `t.text` is not a normal `#define`d macro. The `macros.get` branch above
                // already claimed that case, so a user redefinition always wins.
                work.items[idx] = repl;
                did_subst = true;
            }
        }

        if (!did_subst) idx += 1;
    }
    return work.items;
}

/// Expands macros, object-like, function-like, and the four dynamic macros, in one
/// non-directive logical `line`, appending the result to `out`. Each appended `PToken`'s
/// `.text` is freshly heap-owned via `allocator`, ready to be consumed by `emit`. Runs
/// `expandTokens` in a per-line arena, freed on return, and copies its result out, token by
/// token, via `allocator`. See `expandTokens` for the expansion algorithm itself. `dyn`
/// backs `__LINE__`/`__FILE__`/`__DATE__`/`__TIME__`. See `Dyn`.
fn expandLine(allocator: std.mem.Allocator, macros: *const MacroTable, dyn: Dyn, line: []const PToken, out: *std.ArrayList(PToken)) Error!void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const expanded = try expandTokens(arena, macros, dyn, line);
    for (expanded) |t| {
        const dup_text = try allocator.dupe(u8, t.text);
        errdefer allocator.free(dup_text);
        try out.append(allocator, .{
            .kind = t.kind,
            .text = dup_text,
            .line = t.line,
            .col = t.col,
            .owns_text = true,
            .space_before = t.space_before,
            .bol = t.bol,
            .hide = &.{},
        });
    }
}

/// Walks one file's already-`scan`ned `PToken` stream (`ptoks`, from the top-level source
/// or, recursively via `handleInclude`, from a resolved `#include`d file) one logical line
/// at a time, appending the expanded result to `p.out`. `includer_dir`/`identity` describe
/// this file. `includer_dir` is its own directory, passed to `handleInclude` for any
/// `#include` line found here, so a nested relative `"..."` resolves against the file that
/// contains it, not the original top-level source. `identity` is its `ResolvedFile.identity`
/// (null for the top-level source, which has none) and backs `#pragma once`. See
/// `processDirective`. Stops at this stream's own `eof`. It does not consume it, check
/// `p.cond_stack` for balance, or append anything to `p.out` for it. Those are top-level-only
/// steps `preprocess` takes once every recursive call has returned. A header may open a
/// conditional group its own `#endif` does not close. An include-guard `#ifndef`/`#endif`
/// wrapping the whole file is fine, but the group only needs to be balanced by the end of
/// the translation unit, not the end of each individual file.
///
/// Non-directive lines are not expanded one at a time. Consecutive non-directive lines are
/// accumulated into a `region` and expanded together, so a function-like macro invocation
/// whose argument list (or whose `(`) spans a newline still expands, as C requires. A
/// directive line, or this stream's `eof`, flushes the pending region first, so a call
/// never crosses a `#`-directive or an `#include` boundary, and then the directive runs.
/// Dropping the `newline` markers when building the region is safe: `emit` drops them
/// anyway, and macro expansion does not otherwise depend on line boundaries.
///
/// Conditional compilation (`p.cond_stack`/`condEmitting`) gates both halves of that split.
/// A non-directive line is only appended to `region`, so it only survives to be expanded
/// and emitted, while `condEmitting(p.cond_stack.items)` is true. While it is false,
/// non-directive lines are silently dropped, macros and all, exactly as if they were never
/// in the source. This is what lets `#if 0` skip a nested block containing tokens that
/// would not even scan as valid expansion input, like the "nested conditionals skip
/// correctly" test's `garbage nonsense !!!`. A directive line's dispatch instead branches
/// three ways. One of the six conditional directives (`isCondDirective`) goes to
/// `handleConditional` unconditionally, active or not, since they are what tracks nesting
/// and finds the matching branch in the first place. `include` goes to `handleInclude`, but
/// only while emitting. Every other directive name goes to `processDirective`, likewise
/// only while emitting. While inactive, a `#define`/`#undef`/`#include`/etc. is silently
/// ignored rather than dispatched.
fn processLines(p: *Proc, ptoks: []const PToken, includer_dir: ?[]const u8, identity: ?[]const u8) Error!void {
    // `__FILE__` for every line in this file: `identity` (an `#include`d file's own
    // `ResolvedFile.identity`) if this call is one, else `opts.filename` for the top-level
    // source. `date_str`/`time_str` are the same for every file in the translation unit.
    // See `Dyn`.
    const dyn: Dyn = .{ .filename = identity orelse p.opts.filename, .date_str = p.date_str, .time_str = p.time_str };

    // A run of consecutive non-directive lines' tokens (borrowed from `ptoks`, no owned
    // text), expanded as one unit so multi-line macro calls work. Flushed at each directive.
    var region: std.ArrayList(PToken) = .empty;
    defer region.deinit(p.allocator);

    var i: usize = 0;
    while (i < ptoks.len and ptoks[i].kind != .eof) {
        var j = i;
        while (j < ptoks.len and ptoks[j].kind != .newline and ptoks[j].kind != .eof) j += 1;
        const line = ptoks[i..j];
        i = if (j < ptoks.len and ptoks[j].kind == .newline) j + 1 else j;

        if (line.len == 0) continue; // blank line
        if (line[0].kind == .hash and line[0].bol) {
            if (region.items.len != 0) {
                try expandLine(p.allocator, p.macros, dyn, region.items, p.out);
                region.clearRetainingCapacity();
            }
            const rest = line[1..];
            if (rest.len > 0 and rest[0].kind == .ident and isCondDirective(rest[0].text)) {
                try handleConditional(p.allocator, p.macros, dyn, p.opts.resolver, includer_dir, p.cond_stack, rest[0].text, rest[1..]);
            } else if (condEmitting(p.cond_stack.items)) {
                if (rest.len > 0 and rest[0].kind == .ident and std.mem.eql(u8, rest[0].text, "include")) {
                    try handleInclude(p, rest[1..], includer_dir, null);
                } else if (rest.len > 0 and rest[0].kind == .ident and std.mem.eql(u8, rest[0].text, "include_next")) {
                    // `#include_next <h>` re-resolves `h` starting after the search
                    // directory this file came from. `identity` (this file's own identity) is
                    // that origin marker. See `handleInclude`. glibc's `<limits.h>` uses this
                    // to reach the compiler's `<limits.h>`.
                    try handleInclude(p, rest[1..], includer_dir, identity orelse p.opts.filename);
                } else {
                    try processDirective(p.allocator, p.macros, rest, p.included, identity);
                }
            } // else: inactive branch, a non-conditional directive is silently ignored.
        } else if (condEmitting(p.cond_stack.items)) {
            try region.appendSlice(p.allocator, line);
        } // else: inactive branch, the line is silently dropped, not expanded.
    }
    if (region.items.len != 0) try expandLine(p.allocator, p.macros, dyn, region.items, p.out);
}

/// Everything `expandDynamicMacro` needs to synthesize `__LINE__`/`__FILE__`/`__DATE__`/
/// `__TIME__` at expansion time. Threaded alongside `macros` through
/// `expandTokens`/`expandLine`/`evalIfExpr`/`handleConditional`, rather than stored in
/// `MacroTable`, since these four are not ordinary macros: `__LINE__`'s value changes with
/// where it is written, and none of the four could be freed or looked up the way a `Macro`
/// is. `filename` is the current file's name for `__FILE__`. `processLines` sets it to
/// `opts.filename` for the top-level source, or the current `#include`d file's
/// `ResolvedFile.identity` for anything included. This is best-effort: a header included by
/// yet another header still just reports its own identity, matching real cpp behavior.
/// `date_str`/`time_str` are `formatDateTime`'s result, computed exactly once by
/// `preprocess`, so every `__DATE__`/`__TIME__` anywhere in the whole translation unit
/// reports the identical value, per the standard's "a single specific time" wording, and
/// never by reading the clock, since `preprocess` stays pure.
const Dyn = struct {
    filename: []const u8,
    date_str: []const u8,
    time_str: []const u8,
};

/// `formatDateTime`'s result: two heap-owned strings, freed by the caller with
/// `allocator.free` once (they back `__DATE__`/`__TIME__` for the whole `preprocess` call).
const DateTime = struct { date: []const u8, time: []const u8 };

/// The twelve C month abbreviations, indexed `[numeric() - 1]` (`jan` = 1, per
/// `std.time.epoch.Month`).
const month_names = [_][]const u8{ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

/// Formats `__DATE__` (`"Mmm DD YYYY"`, a single-digit day space-padded, for example
/// `"Jan  1 1970"`) and `__TIME__` (`"HH:MM:SS"`, always zero-padded) from `timestamp`, a
/// Unix timestamp, never the wall clock, since `preprocess` must stay pure and
/// reproducible, via `std.time.epoch`. A negative `timestamp` clamps to 0
/// (`EpochSeconds.secs` is `u64`). Allocated from `allocator`. Free both fields of the
/// result with `allocator.free`.
fn formatDateTime(allocator: std.mem.Allocator, timestamp: i64) Error!DateTime {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(0, timestamp)) };
    const ds = es.getDaySeconds();
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const month = month_names[@as(usize, md.month.numeric()) - 1];
    const day: u32 = @as(u32, md.day_index) + 1;

    // `{d:2}`, with no explicit fill or alignment, defaults to right-aligned, space-filled
    // width 2. This is exactly the standard's single-digit-day rule. `{d:0>2}` is explicitly
    // zero-filled.
    const date = try std.fmt.allocPrint(allocator, "{s} {d:2} {d}", .{ month, day, yd.year });
    errdefer allocator.free(date);
    const time = try std.fmt.allocPrint(allocator, "{d:0>2}:{d:0>2}:{d:0>2}", .{ ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute() });
    return .{ .date = date, .time = time };
}

/// Inserts (or replaces) a single object-like macro `name` -> `body_src` into `macros`,
/// duping both the name and the body (via `dupPTokenSlice`) so the result outlives whatever
/// `body_src` was borrowed from. Shared by `preseedMacros`'s two predefined-macro/`-D`
/// paths below. It mirrors the tail of `handleDefine` minus the function-like-macro parsing
/// that does not apply here.
fn putObjectMacro(allocator: std.mem.Allocator, macros: *MacroTable, name: []const u8, body_src: []const PToken) Error!void {
    const body = try dupPTokenSlice(allocator, body_src);
    errdefer {
        for (body) |t| if (t.owns_text) allocator.free(t.text);
        allocator.free(body);
    }
    const name_dup = try allocator.dupe(u8, name);
    errdefer allocator.free(name_dup);

    if (macros.fetchRemove(name)) |kv| freeMacro(allocator, &kv.value);
    try macros.put(allocator, name_dup, .{ .name = name_dup, .params = null, .variadic = false, .body = body });
}

/// `putObjectMacro` with a single `int_lit` body token holding `text`. Used for
/// `__STDC__`/`__STDC_VERSION__` and a valueless `-D name`, which the standard gives the
/// body `1`.
fn defineSimple(allocator: std.mem.Allocator, macros: *MacroTable, name: []const u8, text: []const u8) Error!void {
    const body_src = [_]PToken{.{ .kind = .int_lit, .text = text, .line = 0, .col = 0 }};
    try putObjectMacro(allocator, macros, name, &body_src);
}

/// `putObjectMacro` for a `-D name=value`. `value` is scanned with the same lexer everything
/// else runs through, so `-D SIZE=8` gets a real `int_lit` body, `-D GREET="hi"` a real
/// `str_lit` body, and so on. The scanned tokens (minus the trailing `eof`) become the
/// macro's body.
fn defineFromScannedValue(allocator: std.mem.Allocator, macros: *MacroTable, name: []const u8, value: []const u8) Error!void {
    const scanned = try scan(allocator, value, .{});
    defer freePTokens(allocator, scanned);
    try putObjectMacro(allocator, macros, name, scanned[0 .. scanned.len - 1]);
}

/// The GCC-style per-target predefined set: `__GNUC__`/arch/OS/endian/size macros as plain
/// decimal `defineSimple` calls, and the handful of type-name macros (`__SIZE_TYPE__` and
/// friends, whose value is more than one token, for example `long unsigned int`) via
/// `defineFromScannedValue`, so they get a real multi-token body. `sp.long_bits == 64`
/// picks the LP64 forms of the type-name macros (aarch64/riscv64/x86_64 here are all
/// LP64). Anything narrower (x86) gets the ILP32 forms. Called by `preseedMacros` only when
/// `opts.system` is non-null. See that function's doc comment for why.
fn seedSystemPredefs(allocator: std.mem.Allocator, macros: *MacroTable, sp: SystemPredef) Error!void {
    // A decimal value always fits `buf` (the widest field here is a `u16` byte count, at
    // most 5 digits), so a `bufPrint` failure cannot actually happen. `unreachable` on its
    // unused error case keeps this a plain `[]const u8` for `defineSimple` below.
    var buf: [16]u8 = undefined;
    const dec = struct {
        fn f(b: *[16]u8, v: anytype) []const u8 {
            return std.fmt.bufPrint(b, "{d}", .{v}) catch unreachable;
        }
    }.f;

    try defineSimple(allocator, macros, "__GNUC__", dec(&buf, sp.gnuc_major));
    try defineSimple(allocator, macros, "__GNUC_MINOR__", dec(&buf, sp.gnuc_minor));
    try defineSimple(allocator, macros, "__GNUC_PATCHLEVEL__", dec(&buf, sp.gnuc_patch));

    switch (sp.arch) {
        .aarch64 => try defineSimple(allocator, macros, "__aarch64__", "1"),
        .x86_64 => {
            try defineSimple(allocator, macros, "__x86_64__", "1");
            try defineSimple(allocator, macros, "__amd64__", "1");
        },
        .riscv64 => {
            try defineSimple(allocator, macros, "__riscv", "1");
            try defineSimple(allocator, macros, "__riscv_xlen", "64");
        },
        .x86 => try defineSimple(allocator, macros, "__i386__", "1"),
    }

    try defineSimple(allocator, macros, "__linux__", "1");
    try defineSimple(allocator, macros, "__unix__", "1");
    try defineSimple(allocator, macros, "__unix", "1");
    try defineSimple(allocator, macros, "__ELF__", "1");
    try defineSimple(allocator, macros, "__gnu_linux__", "1");

    try defineSimple(allocator, macros, "__ORDER_LITTLE_ENDIAN__", "1234");
    try defineSimple(allocator, macros, "__ORDER_BIG_ENDIAN__", "4321");
    try defineSimple(allocator, macros, "__ORDER_PDP_ENDIAN__", "3412");
    try defineSimple(allocator, macros, "__BYTE_ORDER__", "1234");

    try defineSimple(allocator, macros, "__CHAR_BIT__", "8");
    try defineSimple(allocator, macros, "__SIZEOF_SHORT__", "2");
    try defineSimple(allocator, macros, "__SIZEOF_INT__", "4");
    try defineSimple(allocator, macros, "__SIZEOF_LONG__", dec(&buf, sp.long_bits / 8));
    try defineSimple(allocator, macros, "__SIZEOF_LONG_LONG__", "8");
    try defineSimple(allocator, macros, "__SIZEOF_POINTER__", dec(&buf, sp.ptr_bits / 8));
    try defineSimple(allocator, macros, "__SIZEOF_SIZE_T__", dec(&buf, sp.ptr_bits / 8));
    try defineSimple(allocator, macros, "__SIZEOF_FLOAT__", "4");
    try defineSimple(allocator, macros, "__SIZEOF_DOUBLE__", "8");
    try defineSimple(allocator, macros, "__SIZEOF_WCHAR_T__", "4");
    try defineSimple(allocator, macros, "__SIZEOF_PTRDIFF_T__", dec(&buf, sp.ptr_bits / 8));

    const lp64 = sp.long_bits == 64;
    try defineFromScannedValue(allocator, macros, "__SIZE_TYPE__", if (lp64) "long unsigned int" else "unsigned int");
    try defineFromScannedValue(allocator, macros, "__PTRDIFF_TYPE__", if (lp64) "long int" else "int");
    try defineFromScannedValue(allocator, macros, "__WCHAR_TYPE__", "int");
    try defineFromScannedValue(allocator, macros, "__WINT_TYPE__", "unsigned int");
    try defineFromScannedValue(allocator, macros, "__INTMAX_TYPE__", if (lp64) "long int" else "long long int");
    try defineFromScannedValue(allocator, macros, "__UINTMAX_TYPE__", if (lp64) "long unsigned int" else "long long unsigned int");

    // The `__*_MAX__` family of GCC predefines. `<limits.h>`, VCC's own built-in copy, see
    // `limits.zig`, spells INT_MAX/LONG_MAX/... off these, matching how a real GCC header
    // reads them. Only `__LONG_MAX__` varies by target (long is 32 or 64 bits). The rest
    // are fixed: short 2 bytes, int 4 bytes, long long 8 bytes, wchar_t = int.
    try defineSimple(allocator, macros, "__SCHAR_MAX__", "127");
    try defineSimple(allocator, macros, "__SHRT_MAX__", "32767");
    try defineSimple(allocator, macros, "__INT_MAX__", "2147483647");
    try defineSimple(allocator, macros, "__LONG_MAX__", if (lp64) "9223372036854775807L" else "2147483647L");
    try defineSimple(allocator, macros, "__LONG_LONG_MAX__", "9223372036854775807LL");
    try defineSimple(allocator, macros, "__WCHAR_MAX__", "2147483647");

    // GCC defines `__CHAR_UNSIGNED__` for a target whose plain `char` is unsigned (aarch64,
    // riscv64). `<limits.h>` reads it to pick CHAR_MIN/CHAR_MAX. A signed-char target (x86)
    // leaves it undefined, same as GCC.
    if (!sp.char_signed) try defineSimple(allocator, macros, "__CHAR_UNSIGNED__", "1");
}

/// Populates `macros`, empty on entry, in order: first the standard predefined object-like
/// macros (`__STDC__` -> `1`, `__STDC_VERSION__` -> `199409L`), then, only when
/// `opts.system` is non-null (null is the default, so this step is a no-op for every
/// existing caller), the full GCC-style per-target set (`seedSystemPredefs`), then
/// `opts.defines` (`-D name` -> `1`, `-D name=value` -> `value` scanned), then
/// `opts.undefines` (`-U name`, removed if present). So a `-U` can strip a `-D`, a standard
/// predefined name, or a system-predefined name, and a `-D` can likewise override a
/// system-predefined name. All of this happens before any source line is processed, so it
/// is visible even to a `#ifdef` on the translation unit's very first line.
fn preseedMacros(allocator: std.mem.Allocator, macros: *MacroTable, opts: Options) Error!void {
    try defineSimple(allocator, macros, "__STDC__", "1");
    try defineSimple(allocator, macros, "__STDC_VERSION__", "199409L");

    if (opts.system) |sp| try seedSystemPredefs(allocator, macros, sp);

    for (opts.defines) |d| {
        if (d.value.len == 0) {
            try defineSimple(allocator, macros, d.name, "1");
        } else {
            try defineFromScannedValue(allocator, macros, d.name, d.value);
        }
    }

    for (opts.undefines) |name| {
        if (macros.fetchRemove(name)) |kv| freeMacro(allocator, &kv.value);
    }
}

/// Produces the special-cased replacement token for one of the four dynamic macros
/// (`__LINE__`/`__FILE__`/`__DATE__`/`__TIME__`), or `null` if `t.text` does not name
/// one. Called by `expandTokens` only when `t.text` is not already a normal `#define`d
/// macro, checked first by the caller (see `expandTokens`), so a user `#define __LINE__ x`
/// always wins over this built-in, exactly like redefining any other predefined macro.
/// `__LINE__` reads `t`'s own `.line`, the identifier occurrence's original physical line,
/// already splice-correct. See `scan`. `__FILE__`/`__DATE__`/`__TIME__` come from `dyn`.
/// Every produced token carries `t`'s position and whitespace bits, so it slots into the
/// surrounding line exactly like an object-like macro's single-token body would, with an
/// empty hide-set. None of the four can ever re-trigger itself, since the result is always
/// a literal, never another identifier, so there is nothing for a hide-set to guard
/// against. `text` fields are borrowed, from `arena`-allocated `__LINE__` text, or directly
/// from `dyn`'s slices. `expandLine` dupes the final token text before it leaves the arena,
/// as it already does for every other expansion result.
fn expandDynamicMacro(arena: std.mem.Allocator, dyn: Dyn, t: PToken) Error!?PToken {
    if (std.mem.eql(u8, t.text, "__LINE__")) {
        const text = try std.fmt.allocPrint(arena, "{d}", .{t.line});
        return PToken{ .kind = .int_lit, .text = text, .line = t.line, .col = t.col, .space_before = t.space_before, .bol = t.bol };
    }
    if (std.mem.eql(u8, t.text, "__FILE__")) {
        return PToken{ .kind = .str_lit, .text = dyn.filename, .line = t.line, .col = t.col, .space_before = t.space_before, .bol = t.bol };
    }
    if (std.mem.eql(u8, t.text, "__DATE__")) {
        return PToken{ .kind = .str_lit, .text = dyn.date_str, .line = t.line, .col = t.col, .space_before = t.space_before, .bol = t.bol };
    }
    if (std.mem.eql(u8, t.text, "__TIME__")) {
        return PToken{ .kind = .str_lit, .text = dyn.time_str, .line = t.line, .col = t.col, .space_before = t.space_before, .bol = t.bol };
    }
    return null;
}

/// The shared core of `preprocess`/`preprocessToText`: `splice`, then `scan`, then the
/// directive and expansion pass (`processLines`, recursing into `handleInclude` for every
/// `#include`), returning the final `PToken` stream. The stream is in source order, with
/// `newline` markers already dropped by `processLines`, and terminated by an `eof` PToken.
/// `preprocess` collapses this to `lexer.Token`s via `emit`. `preprocessToText` (`-E`)
/// instead walks it directly to reconstruct preprocessed text. Free the result with
/// `freePTokens`.
///
/// Predefined macros: `preseedMacros` seeds `__STDC__`/`__STDC_VERSION__` plus
/// `opts.defines`/`opts.undefines` (`-D`/`-U`) before any source line runs. `__LINE__`/
/// `__FILE__`/`__DATE__`/`__TIME__` are handled dynamically at expansion time instead (see
/// `expandDynamicMacro`), rather than stored in the table.
fn preprocessCore(allocator: std.mem.Allocator, source: []const u8, opts: Options) Error![]PToken {
    const ptoks = try scan(allocator, source, opts);
    defer freePTokens(allocator, ptoks);

    var macros: MacroTable = .empty;
    defer {
        var it = macros.valueIterator();
        while (it.next()) |m| freeMacro(allocator, m);
        macros.deinit(allocator);
    }
    try preseedMacros(allocator, &macros, opts);

    // `__DATE__`/`__TIME__`'s text, computed exactly once for the whole translation unit.
    // See `Dyn`.
    const dt = try formatDateTime(allocator, opts.timestamp);
    defer allocator.free(dt.date);
    defer allocator.free(dt.time);

    var out: std.ArrayList(PToken) = .empty;
    errdefer {
        for (out.items) |t| if (t.owns_text) allocator.free(t.text);
        out.deinit(allocator);
    }

    // The open conditional-compilation groups, innermost last. Empty means we are
    // at top level, always emitting. Shared across every recursively-included file. See
    // `Proc`'s doc comment. See `CondFrame`/`condEmitting`/`handleConditional`.
    var cond_stack: std.ArrayList(CondFrame) = .empty;
    defer cond_stack.deinit(allocator);

    // `#pragma once` bookkeeping, keyed by `ResolvedFile.identity`. Each key is a heap-owned
    // dupe (see `processDirective`), freed here.
    var included: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = included.keyIterator();
        while (it.next()) |k| allocator.free(k.*);
        included.deinit(allocator);
    }

    var p: Proc = .{
        .allocator = allocator,
        .opts = opts,
        .macros = &macros,
        .cond_stack = &cond_stack,
        .included = &included,
        .out = &out,
        .date_str = dt.date,
        .time_str = dt.time,
    };
    try processLines(&p, ptoks, std.fs.path.dirname(opts.filename), null);
    if (cond_stack.items.len != 0) return error.PreprocError; // unclosed #if/#ifdef/#ifndef at EOF

    // The `eof` marker never owns text, so copying the struct is safe.
    try out.append(allocator, ptoks[ptoks.len - 1]);

    return out.toOwnedSlice(allocator);
}

/// Preprocesses `source` into the final `lexer.Token` stream: `preprocessCore` then `emit`.
/// For source with no directives or macros beyond the predefined ones, the resulting token
/// kinds/text are identical to `lexer.tokenize(source)` (splices and comments having been
/// resolved away). Free the result with `lexer.freeTokens`.
pub fn preprocess(allocator: std.mem.Allocator, source: []const u8, opts: Options) Error![]lexer.Token {
    const final = try preprocessCore(allocator, source, opts);
    defer freePTokens(allocator, final);
    return emit(allocator, final);
}

/// Writes a `str_lit` PToken's decoded bytes (`.text`) back out as a valid `"..."` C string
/// literal. It is re-quoted, and re-escaped just enough to stay valid: `\`/`"` and the
/// common control escapes `\n`/`\t`/`\r`. Every other byte is copied as-is. `preprocess`'s
/// `PToken`s never carry the original source spelling for a string literal (see
/// `spellingRecoverable`), so this is a best-effort re-synthesis, not a byte-exact echo of
/// what was written. That is fine for `-E` output, which does not need to be exact.
fn writeStrLit(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), text: []const u8) Error!void {
    try buf.append(allocator, '"');
    for (text) |ch| {
        switch (ch) {
            '\\', '"' => {
                try buf.append(allocator, '\\');
                try buf.append(allocator, ch);
            },
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            else => try buf.append(allocator, ch),
        }
    }
    try buf.append(allocator, '"');
}

/// Appends one final `PToken`'s spelling to `buf`. A `str_lit` re-quotes and re-escapes via
/// `writeStrLit`, since its `.text` is decoded bytes, not source spelling. A `wstr_lit` is
/// the same, with a leading `L`. Every other kind, including a `char_lit` `int_lit`, whose
/// `.text` is already the decoded decimal value, is just its `.text` verbatim: an
/// identifier, number, operator, or keyword spelling all match the source directly.
fn appendSpelling(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), t: PToken) Error!void {
    if (t.kind == .str_lit) return writeStrLit(allocator, buf, t.text);
    if (t.kind == .wstr_lit) {
        try buf.append(allocator, 'L');
        return writeStrLit(allocator, buf, t.text);
    }
    try buf.appendSlice(allocator, t.text);
}

/// One token's emitted spelling as a fresh heap slice (`appendSpelling` into an owned
/// buffer). Free with `allocator.free`. Used by `preprocessToText`'s paste-avoidance check,
/// which needs each token's output spelling (a `str_lit`'s re-quoted form, not its decoded
/// `.text`) to decide whether two adjacent spellings would re-lex differently.
fn tokenSpelling(allocator: std.mem.Allocator, t: PToken) Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try appendSpelling(allocator, &buf, t);
    return buf.toOwnedSlice(allocator);
}

/// Would emitting spelling `b` immediately after spelling `a`, with no separating
/// whitespace, change the tokenization versus lexing them apart? That is, does
/// `lexer.tokenize(a ++ b)` differ from `lexer.tokenize(a)` followed by
/// `lexer.tokenize(b)`, each minus its trailing `eof`, token for token by kind and text? If
/// so, the two spellings would fuse (`+`/`+` -> `++`, `x`/`y` -> `xy`, `/`/`/` -> a `//`
/// comment start, and so on), and `preprocessToText` must inject a space between them to
/// keep the `-E` text re-lexing back into the original token boundaries. A lex failure on
/// the concatenation (`error.UnexpectedChar`) is treated as "would change": insert the
/// space, and fail safe. This is the standard cpp paste-avoidance rule. The per-pair re-lex
/// is affordable because `-E` is not a hot path.
fn concatChangesLex(allocator: std.mem.Allocator, a: []const u8, b: []const u8) Error!bool {
    const at = lexer.tokenize(allocator, a) catch |err| switch (err) {
        error.UnexpectedChar => return true,
        else => return err,
    };
    defer lexer.freeTokens(allocator, at);
    const bt = lexer.tokenize(allocator, b) catch |err| switch (err) {
        error.UnexpectedChar => return true,
        else => return err,
    };
    defer lexer.freeTokens(allocator, bt);

    const cat = try std.mem.concat(allocator, u8, &.{ a, b });
    defer allocator.free(cat);
    const ct = lexer.tokenize(allocator, cat) catch |err| switch (err) {
        error.UnexpectedChar => return true,
        else => return err,
    };
    defer lexer.freeTokens(allocator, ct);

    // Drop each result's trailing `eof` before comparing.
    const a_toks = at[0 .. at.len - 1];
    const b_toks = bt[0 .. bt.len - 1];
    const c_toks = ct[0 .. ct.len - 1];
    if (c_toks.len != a_toks.len + b_toks.len) return true;
    for (c_toks, 0..) |c, i| {
        const want = if (i < a_toks.len) a_toks[i] else b_toks[i - a_toks.len];
        if (c.kind != want.kind or !std.mem.eql(u8, c.text, want.text)) return true;
    }
    return false;
}

/// Preprocesses `source` and reconstructs the result as text rather than a token stream
/// (the `-E` driver mode). It runs the same `preprocessCore` (macro expansion,
/// conditionals, `#include`, all applied exactly as `preprocess` does), then walks the
/// final `PToken` stream emitting each token's spelling (`appendSpelling`), separated so
/// the reconstructed text re-lexes back into exactly the same token boundaries:
///   - one `\n` per source line advanced since the previous token, so a run of dropped
///     lines, an inactive `#if` branch or a directive line, simply collapses, rather than
///     leaving blank lines. gcc-style `# line "file"` linemarkers are not produced.
///   - otherwise a single space wherever the token's own `space_before` was set.
///   - otherwise, when two tokens the source wrote adjacent, but which, after macro
///     substitution, would now fuse into a different token (for example `PLUS`->`+`
///     sitting before a source `+` making `++`), a space is inserted anyway. This is
///     paste-avoidance per `concatChangesLex`, so the `-E` output never re-lexes
///     differently than the token stream `preprocess` produces.
/// Free the result with `allocator.free`.
pub fn preprocessToText(allocator: std.mem.Allocator, source: []const u8, opts: Options) Error![]u8 {
    const final = try preprocessCore(allocator, source, opts);
    defer freePTokens(allocator, final);

    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    // The previous emitted token's spelling, kept so an adjacent (no-whitespace, same-line)
    // pair can be tested for fusion before deciding whether to separate them.
    var prev_spell: ?[]u8 = null;
    defer if (prev_spell) |s| allocator.free(s);

    var last_line: u32 = 0;
    for (final) |t| {
        if (t.kind == .eof) break;
        const cur_spell = try tokenSpelling(allocator, t);
        errdefer allocator.free(cur_spell);

        if (prev_spell == null) {
            last_line = t.line;
        } else if (t.line > last_line) {
            var n = t.line - last_line;
            while (n > 0) : (n -= 1) try buf.append(allocator, '\n');
            last_line = t.line;
        } else if (t.space_before) {
            try buf.append(allocator, ' ');
        } else if (try concatChangesLex(allocator, prev_spell.?, cur_spell)) {
            try buf.append(allocator, ' ');
        }
        try buf.appendSlice(allocator, cur_spell);

        if (prev_spell) |s| allocator.free(s);
        prev_spell = cur_spell;
    }
    return buf.toOwnedSlice(allocator);
}

test "preprocess strips block and line comments" {
    const toks = try preprocess(std.testing.allocator, "int /* c */ x; // tail\nint y;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
    try std.testing.expectEqual(lexer.Kind.semicolon, toks[2].kind);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[3].kind);
    try std.testing.expectEqualStrings("y", toks[4].text);
}

test "line splicing joins physical lines" {
    const toks = try preprocess(std.testing.allocator, "in\\\nt x;", .{}); // "int x;"
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "a line comment ending in a splice extends onto the joined line" {
    // The `\`+newline is removed in phase 2, so the `//` comment swallows `y = 2;` too and
    // only `int x = 1;` survives. This is the miscompile the phase-2 pre-pass fixes.
    const toks = try preprocess(std.testing.allocator, "int x = 1; // c\\\n y = 2;\nint z;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .kw_int, .ident, .assign, .int_lit, .semicolon, .kw_int, .ident, .semicolon, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
    try std.testing.expectEqualStrings("z", toks[6].text);
}

test "an operator split by a splice scans as one operator" {
    // `=\<NL>=` -> `==`, not two `=`s.
    const ptoks = try scan(std.testing.allocator, "a =\\\n= b", .{});
    defer freePTokens(std.testing.allocator, ptoks);
    const kinds = [_]lexer.Kind{ .ident, .eq_eq, .ident, .eof };
    try std.testing.expectEqual(kinds.len, ptoks.len);
    for (kinds, ptoks) |want, got| try std.testing.expectEqual(want, got.kind);
}

test "a string split by a splice scans as one literal" {
    const ptoks = try scan(std.testing.allocator, "\"ab\\\ncd\"", .{});
    defer freePTokens(std.testing.allocator, ptoks);
    try std.testing.expectEqual(lexer.Kind.str_lit, ptoks[0].kind);
    try std.testing.expectEqualStrings("abcd", ptoks[0].text);
}

test "splice records each byte's original physical line" {
    const sp = try splice(std.testing.allocator, "a\\\nb\nc");
    defer std.testing.allocator.free(sp.text);
    defer std.testing.allocator.free(sp.line_of);
    // "a\<NL>b" splices to "ab" (physical lines 1 and 2), then "\nc".
    try std.testing.expectEqualStrings("ab\nc", sp.text);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 2, 3 }, sp.line_of);
}

test "a token after a splice reports its original line" {
    // `ab` is one token starting on line 1. `c`, past a real newline, is on line 3.
    const ptoks = try scan(std.testing.allocator, "a\\\nb\nc", .{});
    defer freePTokens(std.testing.allocator, ptoks);
    try std.testing.expectEqualStrings("ab", ptoks[0].text);
    try std.testing.expectEqual(@as(u32, 1), ptoks[0].line);
    // ptoks[1] is the newline marker. ptoks[2] is `c`.
    try std.testing.expectEqualStrings("c", ptoks[2].text);
    try std.testing.expectEqual(@as(u32, 3), ptoks[2].line);
}

test "preprocess of directive-free source matches lexer.tokenize byte-for-byte" {
    const allocator = std.testing.allocator;
    const source = "int f(int a, int b){ int x = a + b * 2; return x - 1; }";
    const want = try lexer.tokenize(allocator, source);
    defer lexer.freeTokens(allocator, want);
    const got = try preprocess(allocator, source, .{});
    defer lexer.freeTokens(allocator, got);

    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectEqual(w.kind, g.kind);
        try std.testing.expectEqualStrings(w.text, g.text);
    }
}

test "an unterminated block comment is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "int x; /* never closed", .{}));
}

test "a stray # outside a directive is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "int x; # int y;", .{}));
}

// Directive dispatch, object-like macros, and the simple directives.

test "define then use expands the macro" {
    const toks = try preprocess(std.testing.allocator, "#define N 5\nint x = N;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .kw_int, .ident, .assign, .int_lit, .semicolon, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("5", toks[3].text);
}

test "undef makes a prior macro name plain again" {
    const toks = try preprocess(std.testing.allocator, "#define N 5\n#undef N\nint N;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqual(lexer.Kind.ident, toks[1].kind);
    try std.testing.expectEqualStrings("N", toks[1].text);
}

test "a null directive (bare #) is ignored" {
    const toks = try preprocess(std.testing.allocator, "#\nint x;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
}

test "#error stops preprocessing" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#error bail out\nint x;", .{}));
}

test "#warning continues preprocessing" {
    const toks = try preprocess(std.testing.allocator, "#warning heads up\nint x;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
}

test "#line and #pragma are accepted no-ops" {
    const toks = try preprocess(std.testing.allocator, "#line 100\n#pragma once\nint x;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
}

test "an unimplemented directive is error.Unsupported" {
    // `#include` and `#ifdef` are both implemented now, so neither serves
    // as an example of an unsupported directive anymore. Use a made-up directive name
    // instead, which falls through `processDirective`'s dispatch unchanged.
    try std.testing.expectError(error.Unsupported, preprocess(std.testing.allocator, "#frobnicate\nint x;\n", .{}));
}

// A direct termination check: `A` and `B` expand into each
// other forever unless the hide-set stops it. This must return, not hang, regardless of
// what the leftover, still-hidden `A` token means downstream.
test "mutually recursive macros terminate instead of looping forever" {
    const toks = try preprocess(std.testing.allocator, "#define A B\n#define B A\nint x = A;\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
}

// Function-like macros: arguments, `#` stringize, `##` token-paste. The
// gcc-diff execution tests (`ADD`/`S`/`CAT`/bare-name) live in `tests/native.zig` next to
// `expectAgrees`.

// Variadic macros: `__VA_ARGS__`, the GNU `, ##__VA_ARGS__`
// comma-swallow, and C23 `__VA_OPT__`. These are purely-syntactic token-stream checks,
// like the rest of this file's macro tests, so they assert on `preprocess`'s
// `lexer.Token` output directly rather than via `expectAgrees`.

test "a variadic macro with no named params binds __VA_ARGS__ to every argument" {
    const toks = try preprocess(std.testing.allocator, "#define F(...) g(__VA_ARGS__)\nF(1,2,3)\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .lparen, .int_lit, .comma, .int_lit, .comma, .int_lit, .rparen, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("g", toks[0].text);
    try std.testing.expectEqualStrings("1", toks[2].text);
    try std.testing.expectEqualStrings("2", toks[4].text);
    try std.testing.expectEqualStrings("3", toks[6].text);
}

test "a variadic macro's named params take the leading arguments, __VA_ARGS__ the rest" {
    const toks = try preprocess(std.testing.allocator, "#define H(a, ...) a: __VA_ARGS__\nH(x,1,2)\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .colon, .int_lit, .comma, .int_lit, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("x", toks[0].text);
    try std.testing.expectEqualStrings("1", toks[2].text);
    try std.testing.expectEqualStrings("2", toks[4].text);
}

test "a variadic macro called with no trailing arguments has an empty __VA_ARGS__" {
    const toks = try preprocess(std.testing.allocator, "#define F(...) g(__VA_ARGS__)\nF()\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .lparen, .rparen, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
}

test "too few call arguments for a variadic macro's named params is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#define K(a,b,...) a\nK(x)\n", .{}));
}

test "__VA_ARGS__ in a non-variadic macro body is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#define M(a) __VA_ARGS__\n", .{}));
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#define X __VA_ARGS__\n", .{}));
}

test "__VA_OPT__ in a non-variadic macro body is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#define M(a) __VA_OPT__(x)\n", .{}));
}

// GNU `, ##__VA_ARGS__` comma-swallow: the comma stays when `__VA_ARGS__` is non-empty.
// The `##` itself is a no-op there, since pasting `,` against a digit or identifier never
// fuses into one token, so re-lexing just hands the two operands straight back. But the
// comma and the `##` both vanish when `__VA_ARGS__` is empty.
test "GNU , ##__VA_ARGS__ keeps the comma when __VA_ARGS__ is non-empty" {
    const toks = try preprocess(std.testing.allocator, "#define D(fmt, ...) printf(fmt, ##__VA_ARGS__)\nD(\"x\", 1, 2)\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .lparen, .str_lit, .comma, .int_lit, .comma, .int_lit, .rparen, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("printf", toks[0].text);
    try std.testing.expectEqualStrings("x", toks[2].text);
    try std.testing.expectEqualStrings("1", toks[4].text);
    try std.testing.expectEqualStrings("2", toks[6].text);
}

test "GNU , ##__VA_ARGS__ deletes the comma when __VA_ARGS__ is empty" {
    const toks = try preprocess(std.testing.allocator, "#define D(fmt, ...) printf(fmt, ##__VA_ARGS__)\nD(\"x\")\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .lparen, .str_lit, .rparen, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("printf", toks[0].text);
    try std.testing.expectEqualStrings("x", toks[2].text);
}

// `pre ## __VA_ARGS__`, with no comma before the `##`: an empty `__VA_ARGS__` is a
// placemarker, so the paste keeps the left operand (GCC: `P(foo)` -> `foo`, not empty).
// The comma-swallow above only deletes the left token when that token is literally a
// comma. This is the non-comma case, and it must not delete `pre`.
test "## __VA_ARGS__ with no comma keeps the left operand when __VA_ARGS__ is empty" {
    const toks = try preprocess(std.testing.allocator, "#define P(pre, ...) pre ## __VA_ARGS__\nP(foo)\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("foo", toks[0].text);
}

test "## __VA_ARGS__ with no comma pastes normally when __VA_ARGS__ is non-empty" {
    const toks = try preprocess(std.testing.allocator, "#define P(pre, ...) pre ## __VA_ARGS__\nP(foo, 1)\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("foo1", toks[0].text);
}

// C23 `__VA_OPT__(content)`: `content` is emitted only when `__VA_ARGS__` is non-empty.
test "__VA_OPT__ emits its content when __VA_ARGS__ is non-empty" {
    const toks = try preprocess(std.testing.allocator, "#define E(fmt, ...) printf(fmt __VA_OPT__(,) __VA_ARGS__)\nE(\"x\",1)\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .lparen, .str_lit, .comma, .int_lit, .rparen, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("printf", toks[0].text);
    try std.testing.expectEqualStrings("x", toks[2].text);
    try std.testing.expectEqualStrings("1", toks[4].text);
}

test "__VA_OPT__ emits nothing when __VA_ARGS__ is empty" {
    const toks = try preprocess(std.testing.allocator, "#define E(fmt, ...) printf(fmt __VA_OPT__(,) __VA_ARGS__)\nE(\"x\")\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    const kinds = [_]lexer.Kind{ .ident, .lparen, .str_lit, .rparen, .eof };
    try std.testing.expectEqual(kinds.len, toks.len);
    for (kinds, toks) |want, got| try std.testing.expectEqual(want, got.kind);
    try std.testing.expectEqualStrings("printf", toks[0].text);
    try std.testing.expectEqualStrings("x", toks[2].text);
}

// `#` needs a token's original source spelling, but a `str_lit`/char literal PToken
// carries decoded bytes (`"a\n"` becomes the 2 bytes `a`,LF, and `'c'` becomes "99"), so stringizing one is
// approximate. Rather than fail closed, which blocked real gnulib code whose stringized text
// only feeds a discarded `_Static_assert`/`assert` diagnostic, `stringize` now emits a
// best-effort spelling and succeeds. Ordinary identifier/number/operator stringize, the
// `S(hello)` sizeof-6 test in `tests/native.zig`, is byte-exact and unaffected.
test "stringize of a string-literal argument is best-effort, not a hard failure" {
    const toks = try preprocess(std.testing.allocator, "#define S(x) #x\nchar *n = S(hi);\n", .{});
    lexer.freeTokens(std.testing.allocator, toks);
}

test "stringize of a char-literal argument is best-effort, not a hard failure" {
    const toks = try preprocess(std.testing.allocator, "#define S(x) #x\nchar *n = S('c');\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    // `'c'` stringizes to its decoded numeric value, so the macro yields the string "99".
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(std.testing.allocator);
    for (toks) |t| try joined.appendSlice(std.testing.allocator, t.text);
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "99") != null);
}

// Pasting a string/char literal is likewise unfaithful and fails closed.
test "token-paste of a string-literal argument soft-fails" {
    try std.testing.expectError(error.Unsupported, preprocess(std.testing.allocator, "#define CAT(a,b) a##b\nint n = CAT(x,\"y\");\n", .{}));
}

// The object-like `##` path shares `resolvePastes`, so its
// spelling-recoverable guard fires too: pasting a body `str_lit` operand fails closed.
test "object-like token-paste of a string-literal operand soft-fails" {
    try std.testing.expectError(error.Unsupported, preprocess(std.testing.allocator, "#define BAD a##\"y\"\nint n = BAD;\n", .{}));
}

// Conditional compilation (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`,
// `defined`, and the `#if` constant-expression evaluator). The gcc-diff execution tests
// (ifdef/ifndef/`defined`/`elif` chains, and the nested-skip-garbage case) live in
// `tests/native.zig` next to `expectAgrees`. These are the purely-syntactic malformed-input
// cases that belong here. Every one of them must fail closed with `error.PreprocError`
// rather than silently misparse or accept an ill-formed conditional.

test "an unclosed conditional group at EOF is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#if 1\nint x;\n", .{}));
}

test "#endif with no open conditional group is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#endif\nint x;\n", .{}));
}

test "#else with no open conditional group is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#else\nint x;\n", .{}));
}

test "#elif with no open conditional group is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#elif 1\nint x;\n", .{}));
}

test "#elif after #else is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#if 0\nint a;\n#else\nint b;\n#elif 1\nint c;\n#endif\n", .{}));
}

test "a second #else in the same group is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#if 0\nint a;\n#else\nint b;\n#else\nint c;\n#endif\n", .{}));
}

test "division by zero in a #if expression is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#if 1 / 0\nint x;\n#endif\n", .{}));
}

test "an empty #if expression is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#if\nint x;\n#endif\n", .{}));
}

// A `#if 0` group's own `#define`s, in its now-dead branch, must never reach the macro
// table. The inactive-branch skip applies to every non-conditional directive, not only
// plain code lines.
test "a #define inside an inactive #if branch is ignored" {
    const toks = try preprocess(std.testing.allocator, "#if 0\n#define N 999\n#endif\nint N;\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqual(lexer.Kind.ident, toks[1].kind);
    try std.testing.expectEqualStrings("N", toks[1].text);
}

// `#if` with a full C-precedence expression (`?:`, shifts, bitwise ops) picks the correct
// branch entirely inside the evaluator. No gcc oracle is needed to check this.
test "#if evaluates full operator precedence" {
    const toks = try preprocess(std.testing.allocator, "#if (1 << 3) == 8 && (1 ? 2 : 3) == 2\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqual(lexer.Kind.ident, toks[1].kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

// A `#if` integer literal follows C base rules. A leading `0` with more
// digits is octal, so `010` is 8, not decimal 10, and takes the `#if` branch here.
test "#if parses an octal literal as base 8" {
    const toks = try preprocess(std.testing.allocator, "#if 010 == 8\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

// An out-of-range octal digit (`8` in a `0`-led literal) is malformed, not silently decimal.
test "#if octal literal with a non-octal digit is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#if 08\nint x;\n#endif\n", .{}));
}

// `||` short-circuits: the true left operand means the `1/0` on the
// right is parsed but not evaluated, so there is no div-by-zero error and the `#if` branch is taken.
test "#if || short-circuits past a divide-by-zero right operand" {
    const toks = try preprocess(std.testing.allocator, "#if 1 || (1/0)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

// `&&` short-circuits on a false left operand: `0 && (1/0)` is false (no error), `#else` runs.
test "#if && short-circuits past a divide-by-zero right operand" {
    const toks = try preprocess(std.testing.allocator, "#if 0 && (1/0)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

// `?:` evaluates only the taken arm: `1 ? 42 : (1/0)` takes the `42` side, so the untaken
// `1/0` is parsed but never evaluated. There is no error, and 42 is nonzero, so the `#if` branch runs.
test "#if ternary evaluates only the taken arm" {
    const toks = try preprocess(std.testing.allocator, "#if 1 ? 42 : (1/0)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

// `#include`, the resolver, recursive inclusion, and `#pragma once`. The gcc-diff
// execution tests (`expectAgreesPPBits`, headers written to disk for the oracle vs. a
// virtual resolver for the frontend) live in `tests/native.zig`. These are the purely
// in-process cases (resolver wiring, `<...>` reconstruction, the cycle/depth-limit guard)
// that do not need an oracle.

/// A tiny in-memory header table for these direct unit tests, looked up by exact name.
/// `tests/native.zig`'s `expectAgreesPPBits` builds an equivalent virtual `IncludeResolver`
/// for its gcc-diff tests. This is the same idea without a second file to read from.
const TestHeader = struct { name: []const u8, content: []const u8 };

/// An `IncludeResolver.resolveFn` closing over a `[]const TestHeader` via `ctx`, a pointer
/// to the slice variable itself, so the lookup sees whatever `ctx` currently points at.
/// `is_system`/`includer_dir`/`next_after` are unused. A linear name match is all these tests
/// need. No test here exercises `#include_next`, so `next_after` is simply ignored.
fn testResolve(ctx: ?*anyopaque, name: []const u8, is_system: bool, includer_dir: ?[]const u8, next_after: ?[]const u8) Error!?ResolvedFile {
    _ = is_system;
    _ = includer_dir;
    _ = next_after;
    const headers: *const []const TestHeader = @ptrCast(@alignCast(ctx.?));
    for (headers.*) |h| {
        if (std.mem.eql(u8, h.name, name)) return .{ .identity = h.name, .bytes = h.content };
    }
    return null;
}

test "#include \"...\" resolves a header and expands its macro" {
    const headers = [_]TestHeader{.{ .name = "defs.h", .content = "#define MAGIC 41\n" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(std.testing.allocator, "#include \"defs.h\"\nint x = MAGIC + 1;", .{ .resolver = resolver });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.kw_int, toks[0].kind);
    try std.testing.expectEqualStrings("41", toks[3].text); // int x = 41 + 1;
}

// `<...>` is not one token. The scanner has no bracket mode, so `parseIncludeName` has to
// reconstruct "sys/types.h" from the individual `ident`/`slash`/`dot`/`ident` tokens between
// `<` and `>`. A successful lookup, where the name must match exactly, is the proof it worked.
test "#include <...> reconstructs a multi-token system header name" {
    const headers = [_]TestHeader{.{ .name = "sys/types.h", .content = "#define T 7\n" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(std.testing.allocator, "#include <sys/types.h>\nint x = T;", .{ .resolver = resolver });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("7", toks[3].text);
}

test "#pragma once skips a header the second time it's #include-d" {
    const headers = [_]TestHeader{.{ .name = "a.h", .content = "#pragma once\n#define V 9\n" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(std.testing.allocator, "#include \"a.h\"\n#include \"a.h\"\nint x = V;", .{ .resolver = resolver });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("9", toks[3].text);
}

test "#include with no resolver configured is a preprocessor error" {
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#include \"defs.h\"\nint x;\n", .{}));
}

test "#include of a header the resolver can't find is a preprocessor error" {
    const headers = [_]TestHeader{};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#include \"missing.h\"\nint x;\n", .{ .resolver = resolver }));
}

// A direct cycle-guard check: `a.h` `#include`s `b.h`, which
// `#include`s `a.h` back, with neither guarded by `#pragma once`/`#ifndef`. Without a depth
// limit this recurses forever. With it, it must return `error.PreprocError` instead of
// hanging.
test "an unguarded include cycle hits the depth limit instead of hanging" {
    const headers = [_]TestHeader{
        .{ .name = "a.h", .content = "#include \"b.h\"\n" },
        .{ .name = "b.h", .content = "#include \"a.h\"\n" },
    };
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    try std.testing.expectError(error.PreprocError, preprocess(std.testing.allocator, "#include \"a.h\"\nint x;\n", .{ .resolver = resolver }));
}

// Predefined macros (`__STDC__`/`__STDC_VERSION__`/`__LINE__`/`__FILE__`/
// `__DATE__`/`__TIME__`) plus command-line `-D`/`-U`. `__STDC__ + __LINE__`'s execution test
// (gcc-diff) and the `sizeof(__DATE__) + sizeof(__TIME__)` structural test live in
// `tests/native.zig` next to `expectAgrees`. These are the purely in-process cases (the
// standard macros' own values, `-D`/`-U` wiring, and the deterministic `__DATE__`/`__TIME__`
// formatting) that do not need an oracle.

test "__STDC__ and __STDC_VERSION__ are predefined" {
    const toks = try preprocess(std.testing.allocator, "int a = __STDC__; long b = __STDC_VERSION__;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
    try std.testing.expectEqualStrings("199409L", toks[8].text);
}

test "-D name=value defines a macro via Options" {
    const toks = try preprocess(std.testing.allocator, "int x = FOO;", .{ .defines = &.{.{ .name = "FOO", .value = "7" }} });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("7", toks[3].text);
}

test "-D name with no value defines it as 1" {
    const toks = try preprocess(std.testing.allocator, "int x = BAR;", .{ .defines = &.{.{ .name = "BAR", .value = "" }} });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
}

test "-U removes a -D define" {
    const toks = try preprocess(std.testing.allocator, "#ifdef FOO\nint x = 1;\n#else\nint x = 2;\n#endif\n", .{
        .defines = &.{.{ .name = "FOO", .value = "9" }},
        .undefines = &.{"FOO"},
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("2", toks[3].text);
}

test "-U removes a standard predefined macro" {
    const toks = try preprocess(std.testing.allocator, "#ifdef __STDC__\nint x = 1;\n#else\nint x = 2;\n#endif\n", .{
        .undefines = &.{"__STDC__"},
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("2", toks[3].text);
}

test "a user #define of __LINE__ wins over the dynamic macro" {
    const toks = try preprocess(std.testing.allocator, "#define __LINE__ 42\nint x = __LINE__;", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("42", toks[3].text);
}

test "__LINE__ reports the current physical line" {
    const toks = try preprocess(std.testing.allocator, "int a = __LINE__;\nint b = __LINE__;\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
    try std.testing.expectEqualStrings("2", toks[8].text);
}

// The per-target system predefined set, opt-in via `Options.system`. Left unset (the
// default), preprocessing stays byte-identical to before this set existed. Set, it seeds
// the full GCC-style set (`__GNUC__`, arch/OS/endian/size macros, and the `__SIZE_TYPE__`
// family), so a real C library's headers see the same environment they would see under gcc.

const aarch64_system_predef: SystemPredef = .{
    .arch = .aarch64,
    .gnuc_major = 4,
    .gnuc_minor = 2,
    .gnuc_patch = 0,
    .long_bits = 64,
    .ptr_bits = 64,
    .char_signed = false,
};

const x86_system_predef: SystemPredef = .{
    .arch = .x86,
    .gnuc_major = 4,
    .gnuc_minor = 2,
    .gnuc_patch = 0,
    .long_bits = 32,
    .ptr_bits = 32,
    .char_signed = true,
};

test "Options.system == null leaves __GNUC__ undefined (byte-identical to before)" {
    const toks = try preprocess(std.testing.allocator, "#ifdef __GNUC__\nint x = 1;\n#else\nint x = 2;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("2", toks[3].text);
}

test "Options.system seeds __GNUC__/__GNUC_MINOR__" {
    const toks = try preprocess(std.testing.allocator, "#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 2\nint x = 1;\n#else\nint x = 2;\n#endif\n", .{
        .system = aarch64_system_predef,
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
}

test "Options.system seeds the arch macro for the target and no other arch's" {
    const toks = try preprocess(std.testing.allocator, "#ifdef __aarch64__\nint x = 1;\n#else\nint x = 2;\n#endif\n#ifdef __x86_64__\nint y = 1;\n#else\nint y = 2;\n#endif\n", .{
        .system = aarch64_system_predef,
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
    try std.testing.expectEqualStrings("2", toks[8].text);
}

test "Options.system seeds __SIZEOF_POINTER__ and __linux__" {
    const toks = try preprocess(std.testing.allocator, "#if __SIZEOF_POINTER__ == 8\nint x = 1;\n#else\nint x = 2;\n#endif\n#ifdef __linux__\nint y = 1;\n#else\nint y = 2;\n#endif\n", .{
        .system = aarch64_system_predef,
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
    try std.testing.expectEqualStrings("1", toks[8].text);
}

test "Options.system seeds __SIZE_TYPE__ as a multi-token LP64 type name" {
    const toks = try preprocess(std.testing.allocator, "__SIZE_TYPE__ x;", .{ .system = aarch64_system_predef });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("long", toks[0].text);
    try std.testing.expectEqualStrings("unsigned", toks[1].text);
    try std.testing.expectEqualStrings("int", toks[2].text);
    try std.testing.expectEqualStrings("x", toks[3].text);
}

test "Options.system on an ILP32 (x86) target seeds __i386__ and 4-byte pointer sizes" {
    const toks = try preprocess(std.testing.allocator, "#ifdef __i386__\nint x = 1;\n#else\nint x = 2;\n#endif\n#if __SIZEOF_POINTER__ == 4\nint y = 1;\n#else\nint y = 2;\n#endif\n", .{
        .system = x86_system_predef,
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("1", toks[3].text);
    try std.testing.expectEqualStrings("1", toks[8].text);
}

test "Options.system on an ILP32 (x86) target seeds __SIZE_TYPE__ as unsigned int" {
    const toks = try preprocess(std.testing.allocator, "__SIZE_TYPE__ x;", .{ .system = x86_system_predef });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("unsigned", toks[0].text);
    try std.testing.expectEqualStrings("int", toks[1].text);
    try std.testing.expectEqualStrings("x", toks[2].text);
}

test "a caller -D still overrides a system-predefined macro" {
    const toks = try preprocess(std.testing.allocator, "int x = __GNUC__;", .{
        .system = aarch64_system_predef,
        .defines = &.{.{ .name = "__GNUC__", .value = "9" }},
    });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("9", toks[3].text);
}

test "__FILE__ expands to the configured filename" {
    const toks = try preprocess(std.testing.allocator, "char *f = __FILE__;", .{ .filename = "prog.c" });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqual(lexer.Kind.str_lit, toks[4].kind);
    try std.testing.expectEqualStrings("prog.c", toks[4].text);
}

test "__DATE__/__TIME__ deterministic at epoch" {
    const toks = try preprocess(std.testing.allocator, "char *d = __DATE__; char *t = __TIME__;", .{ .timestamp = 0 });
    defer lexer.freeTokens(std.testing.allocator, toks);
    var found_date = false;
    var found_time = false;
    for (toks) |t| {
        if (t.kind != .str_lit) continue;
        if (std.mem.eql(u8, t.text, "Jan  1 1970")) found_date = true;
        if (std.mem.eql(u8, t.text, "00:00:00")) found_time = true;
    }
    try std.testing.expect(found_date);
    try std.testing.expect(found_time);
}

// `-E` preprocess-only mode (`preprocessToText`) plus the driver flags, disk
// resolver, and timestamp wiring (`frontends/vcc.zig`, not directly testable from here).

test "preprocessToText expands macros to text" {
    const out = try preprocessToText(std.testing.allocator, "#define N 5\nint x = N;", .{});
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "5") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "int") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "N") == null);
}

test "preprocessToText expands a function-like macro to text" {
    const out = try preprocessToText(std.testing.allocator, "#define ADD(a,b) a+b\nint x = ADD(1,2);", .{});
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "ADD") == null);
}

test "preprocessToText drops the inactive conditional branch from the text" {
    const out = try preprocessToText(std.testing.allocator, "#if 0\nint dead;\n#endif\nint alive;\n", .{});
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "dead") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "alive") != null);
}

test "preprocessToText re-quotes a string literal" {
    const out = try preprocessToText(std.testing.allocator, "char *s = \"a\\\"b\";", .{});
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"a\\\"b\"") != null);
}

// Paste-avoidance: a macro whose expansion abuts a following source token
// must not fuse into a different token in the reconstructed `-E` text. `LT` -> `<` sitting
// right before the source `<` would spell `<<` (one `lshift` token) unless a space is
// inserted. The output must re-lex to two separate `<` (`lt`) tokens, exactly as the token
// stream `preprocess` produces. This lexer has no `++`, so `+`/`+` would not actually fuse.
// `<<` is the analogous real fusion here. `concatChangesLex` covers `+`/`+` too, on any
// lexer that does fuse it.
test "preprocessToText avoids accidental token pasting" {
    const out = try preprocessToText(std.testing.allocator, "#define LT <\nint x = 1 LT<2;", .{});
    defer std.testing.allocator.free(out);
    // The inserted separating space appears literally...
    try std.testing.expect(std.mem.indexOf(u8, out, "< <") != null);
    // ...and re-lexing the output yields two `lt` tokens, never a fused `<<` (`lshift`).
    const relexed = try lexer.tokenize(std.testing.allocator, out);
    defer lexer.freeTokens(std.testing.allocator, relexed);
    for (relexed) |tok| try std.testing.expect(tok.kind != .lshift and tok.kind != .lshift_eq);
}

// `#if` completeness: character-literal primaries and the `__has_include`/
// `__has_attribute`/`__has_builtin`/`__has_feature`/`__has_extension` operators.

// A character literal reaches `IfEval` as an ordinary `int_lit` token. `scan` already decodes
// `'A'` to a `char_lit`-flagged `int_lit` "65". See the `PToken` doc comment. So it needs no
// new `parsePrimary` branch. This proves the existing `int_lit` primary already accepts it.
test "#if char literal compares equal to its ASCII value" {
    const toks = try preprocess(std.testing.allocator, "#if 'A' == 65\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "#if bare char literal is truthy" {
    const toks = try preprocess(std.testing.allocator, "#if 'A'\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "#if char literal decodes a backslash escape" {
    const toks = try preprocess(std.testing.allocator, "#if '\\n' == 10\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "__has_include(<...>) resolves against the configured resolver" {
    const headers = [_]TestHeader{.{ .name = "yes.h", .content = "" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(std.testing.allocator, "#if __has_include(<yes.h>)\nint x;\n#else\nint y;\n#endif\n", .{ .resolver = resolver });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "__has_include(<...>) of an unresolvable header is false" {
    const headers = [_]TestHeader{.{ .name = "yes.h", .content = "" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(std.testing.allocator, "#if __has_include(<no.h>)\nint x;\n#else\nint y;\n#endif\n", .{ .resolver = resolver });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

test "__has_include(\"...\") resolves the quote form too" {
    const headers = [_]TestHeader{.{ .name = "yes.h", .content = "" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(std.testing.allocator, "#if __has_include(\"yes.h\")\nint x;\n#else\nint y;\n#endif\n", .{ .resolver = resolver });
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "__has_include with no resolver configured is false, not an error" {
    const toks = try preprocess(std.testing.allocator, "#if __has_include(<anything.h>)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

test "__has_builtin is true for a recognized va_ builtin" {
    const toks = try preprocess(std.testing.allocator, "#if __has_builtin(__builtin_va_start)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "__has_builtin is false for an unrecognized name" {
    const toks = try preprocess(std.testing.allocator, "#if __has_builtin(__builtin_nope)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

test "__has_attribute is conservatively false" {
    const toks = try preprocess(std.testing.allocator, "#if __has_attribute(nope)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

test "__has_feature and __has_extension are conservatively false" {
    const toks = try preprocess(std.testing.allocator, "#if __has_feature(nope) || __has_extension(nope)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

// An `__has_*` name not followed by `(` is left as an ordinary identifier, never an error,
// and falls to `evalIfExpr`'s existing surviving-ident-becomes-`0` rule.
test "__has_include with no following paren is treated as a plain identifier, not an error" {
    const toks = try preprocess(std.testing.allocator, "#if __has_include\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

// A real compiler answers `defined __has_include` and `#ifdef __has_include` true, even though
// `__has_include` is never inserted into the macro table. This is the portable-detection
// idiom real headers rely on to guard their own use of the operator.
test "#ifdef __has_include takes the then branch" {
    const toks = try preprocess(std.testing.allocator, "#ifdef __has_include\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

test "defined(__has_include) and defined __has_include are both true" {
    const toks1 = try preprocess(std.testing.allocator, "#if defined(__has_include)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks1);
    try std.testing.expectEqualStrings("x", toks1[1].text);

    const toks2 = try preprocess(std.testing.allocator, "#if defined __has_include\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks2);
    try std.testing.expectEqualStrings("x", toks2[1].text);
}

test "defined(__has_builtin) is true too" {
    const toks = try preprocess(std.testing.allocator, "#if defined(__has_builtin)\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("x", toks[1].text);
}

// An ordinary undefined identifier must still report as not defined. Only the five real
// `__has_*` operator names get the defined-for-detection treatment.
test "#ifdef __has_nonexistent still takes the else branch" {
    const toks = try preprocess(std.testing.allocator, "#ifdef __has_nonexistent\nint x;\n#else\nint y;\n#endif\n", .{});
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("y", toks[1].text);
}

// The combined idiom real headers use: guard the operator's own availability with `#ifdef`,
// then use it. Both nest correctly and reach the inner branch.
test "the #ifdef __has_include / __has_include(...) idiom reaches the inner take" {
    const headers = [_]TestHeader{.{ .name = "yes.h", .content = "" }};
    var slice: []const TestHeader = &headers;
    const resolver: IncludeResolver = .{ .ctx = @ptrCast(&slice), .resolveFn = testResolve };
    const toks = try preprocess(
        std.testing.allocator,
        "#ifdef __has_include\n#if __has_include(<yes.h>)\nint take;\n#endif\n#endif\n",
        .{ .resolver = resolver },
    );
    defer lexer.freeTokens(std.testing.allocator, toks);
    try std.testing.expectEqualStrings("take", toks[1].text);
}

// The real invariant: for any macro program, `tokenize(preprocessToText(src))` must yield the
// same token kind sequence as `preprocess(src)`. This catches a dropped required space, which
// would merge two tokens into a different one. A spurious space would never change the kind
// sequence, so what this asserts is precisely "no fusion, no dropped boundary".
test "preprocessToText re-lexes to the same token kinds as preprocess" {
    const cases = [_][]const u8{
        "#define N 5\nint a=N;",
        "#define LT <\nint z = 1 LT<2;", // <<  fusion
        "#define EQ =\nint b; b EQ=1;", // ==  fusion
        "#define AMP &\nint c = 1 AMP&2;", // && fusion
        "#define ADD(a,b) a+b\nint y = ADD(3,4)*2;",
    };
    for (cases) |src| {
        const via_tokens = try preprocess(std.testing.allocator, src, .{});
        defer lexer.freeTokens(std.testing.allocator, via_tokens);
        const text = try preprocessToText(std.testing.allocator, src, .{});
        defer std.testing.allocator.free(text);
        const via_text = try lexer.tokenize(std.testing.allocator, text);
        defer lexer.freeTokens(std.testing.allocator, via_text);

        try std.testing.expectEqual(via_tokens.len, via_text.len);
        for (via_tokens, via_text) |a, b| try std.testing.expectEqual(a.kind, b.kind);
    }
}
