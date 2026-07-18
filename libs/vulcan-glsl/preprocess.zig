//! A minimal GLSL ES preprocessor: object-like `#define` macros with expansion, the
//! conditional-compilation directives (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`),
//! and `#version`/`#extension`/`#line`/`#undef`/`#pragma` handling. This is deliberately a
//! SUBSET sized to real GLSL ES 1.00/3.00 shaders (e.g. glmark2's): it predefines `GL_ES`
//! and `GL_FRAGMENT_PRECISION_HIGH`, evaluates `defined(X)` / `&&` / `||` / `!` /
//! integer-literal `#if` expressions, and substitutes object-like macros token-by-token.
//! Function-like macros are NOT supported (none of the target shaders use them).
//!
//! The result is a fresh source string (allocated from the caller's allocator) with all
//! directives removed and every macro use expanded, ready for the existing lexer/parser.

const std = @import("std");

pub const Error = error{ PreprocessError, MacroRedefined, OutOfMemory };

/// A preprocessor macro. `params == null` is object-like; a non-null (possibly empty) list
/// is function-like (`NAME(p0, p1) body`). `owned` macros heap-allocate name/body/params
/// (freed via `freeMacro`); predefined macros point at static string literals.
const Macro = struct { name: []const u8, body: []const u8, params: ?[]const []const u8 = null, owned: bool = false };

fn freeMacro(allocator: std.mem.Allocator, m: Macro) void {
    if (!m.owned) return;
    allocator.free(m.name);
    allocator.free(m.body);
    if (m.params) |ps| {
        for (ps) |p| allocator.free(p);
        allocator.free(ps);
    }
}

/// Run the preprocessor over `src` for the given shader stage. `is_es` predefines `GL_ES`.
/// Returns a newly-allocated source string the caller owns.
pub fn run(allocator: std.mem.Allocator, src: []const u8) Error![]u8 {
    var macros: std.ArrayList(Macro) = .empty;
    defer {
        for (macros.items) |m| freeMacro(allocator, m);
        macros.deinit(allocator);
    }
    // Predefined macros (GLSL ES). Value "1" so `#if GL_ES` and `defined(GL_ES)` are true.
    try macros.append(allocator, .{ .name = "GL_ES", .body = "1" });
    try macros.append(allocator, .{ .name = "GL_FRAGMENT_PRECISION_HIGH", .body = "1" });
    try macros.append(allocator, .{ .name = "__VERSION__", .body = "100" });

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // The conditional stack: each entry says whether the current branch emits, and whether
    // any branch of this #if group has been taken yet (for #elif/#else).
    const Cond = struct { emitting: bool, taken: bool, parent_emitting: bool };
    var cond_stack: std.ArrayList(Cond) = .empty;
    defer cond_stack.deinit(allocator);

    var emitting = true;

    var line_iter = std.mem.splitScalar(u8, src, '\n');
    var joined_buf: std.ArrayList(u8) = .empty;
    defer joined_buf.deinit(allocator);
    while (line_iter.next()) |raw_line| {
        var line = raw_line;
        // Backslash line continuation: splice this and following physical lines (dropping
        // each trailing backslash) into one logical line. `extra_newlines` re-emits the
        // consumed line breaks afterward so error line numbers stay aligned.
        var extra_newlines: usize = 0;
        if (endsWithBackslash(line)) {
            joined_buf.clearRetainingCapacity();
            while (true) {
                try joined_buf.appendSlice(allocator, dropBackslash(line));
                extra_newlines += 1;
                const next = line_iter.next() orelse break;
                line = next;
                if (!endsWithBackslash(line)) {
                    try joined_buf.appendSlice(allocator, line);
                    break;
                }
            }
            line = joined_buf.items;
        }
        const trimmed = std.mem.trimStart(u8, line, " \t");
        if (trimmed.len > 0 and trimmed[0] == '#') {
            const directive = std.mem.trimStart(u8, trimmed[1..], " \t");
            // Directive name = leading identifier.
            var d_end: usize = 0;
            while (d_end < directive.len and (std.ascii.isAlphanumeric(directive[d_end]) or directive[d_end] == '_')) d_end += 1;
            const name = directive[0..d_end];
            const rest = std.mem.trim(u8, directive[d_end..], " \t");

            if (std.mem.eql(u8, name, "ifdef") or std.mem.eql(u8, name, "ifndef")) {
                const macro_name = firstIdent(rest);
                const is_def = lookup(macros.items, macro_name) != null;
                const want = if (std.mem.eql(u8, name, "ifdef")) is_def else !is_def;
                const branch = emitting and want;
                try cond_stack.append(allocator, .{ .emitting = branch, .taken = branch, .parent_emitting = emitting });
                emitting = branch;
            } else if (std.mem.eql(u8, name, "if")) {
                const v = try evalCondition(macros.items, rest);
                const branch = emitting and v;
                try cond_stack.append(allocator, .{ .emitting = branch, .taken = branch, .parent_emitting = emitting });
                emitting = branch;
            } else if (std.mem.eql(u8, name, "elif")) {
                if (cond_stack.items.len == 0) return error.PreprocessError;
                var top = &cond_stack.items[cond_stack.items.len - 1];
                if (!top.taken and top.parent_emitting) {
                    const v = try evalCondition(macros.items, rest);
                    top.emitting = v;
                    top.taken = v;
                } else top.emitting = false;
                emitting = top.emitting;
            } else if (std.mem.eql(u8, name, "else")) {
                if (cond_stack.items.len == 0) return error.PreprocessError;
                var top = &cond_stack.items[cond_stack.items.len - 1];
                top.emitting = !top.taken and top.parent_emitting;
                if (top.emitting) top.taken = true;
                emitting = top.emitting;
            } else if (std.mem.eql(u8, name, "endif")) {
                if (cond_stack.items.len == 0) return error.PreprocessError;
                _ = cond_stack.pop();
                emitting = if (cond_stack.items.len == 0) true else cond_stack.items[cond_stack.items.len - 1].emitting;
            } else if (emitting and std.mem.eql(u8, name, "define")) {
                const macro_name = firstIdent(rest);
                if (macro_name.len == 0) return error.PreprocessError;
                // A `(` immediately after the name (no space) marks a function-like macro.
                if (rest.len > macro_name.len and rest[macro_name.len] == '(') {
                    try defineFunctionMacro(allocator, &macros, macro_name, rest[macro_name.len..]);
                } else {
                    // `#define NAME body` (object-like).
                    const after = std.mem.trim(u8, rest[macro_name.len..], " \t");
                    // GLSL ES 3.4: redefining a macro is an error unless the new definition is
                    // identical to the existing one (the real NVIDIA driver rejects this with
                    // "error C7101: Macro <name> redefined" - several glmark2 scenes, e.g.
                    // conditionals/loop/function, assemble a shader that #defines
                    // HIGHP_OR_DEFAULT twice and so fail to compile on conformant hardware).
                    if (lookupMut(macros.items, macro_name)) |m| {
                        if (m.params != null or !std.mem.eql(u8, m.body, after)) return error.MacroRedefined;
                        // Identical redefinition is a legal no-op, keep the existing body.
                    } else {
                        try macros.append(allocator, .{ .name = try allocator.dupe(u8, macro_name), .body = try allocator.dupe(u8, after), .owned = true });
                    }
                }
            } else if (emitting and std.mem.eql(u8, name, "undef")) {
                const macro_name = firstIdent(rest);
                removeMacro(allocator, &macros, macro_name);
            } else if (emitting and std.mem.eql(u8, name, "version")) {
                // `#version N [es|core]` redefines __VERSION__ so `#if __VERSION__ >= 300`
                // guards resolve for real ES 3.0 shaders. The leading integer is the version.
                const num = firstNumber(rest);
                if (num.len > 0) {
                    removeMacro(allocator, &macros, "__VERSION__");
                    try macros.append(allocator, .{ .name = try allocator.dupe(u8, "__VERSION__"), .body = try allocator.dupe(u8, num), .owned = true });
                }
            }
            // #version / #extension / #line / #pragma (and any directive while !emitting): drop.
            // Emit a blank line to keep error line numbers roughly aligned.
            try out.append(allocator, '\n');
            for (0..extra_newlines) |_| try out.append(allocator, '\n');
            continue;
        }

        if (emitting) {
            try expandLine(allocator, &out, line, macros.items, &.{});
        }
        try out.append(allocator, '\n');
        for (0..extra_newlines) |_| try out.append(allocator, '\n');
    }
    return out.toOwnedSlice(allocator);
}

/// Expand object-like macros across one source line, token by token. Identifiers that name
/// a macro are replaced by the (recursively-expanded) body, everything else copies through.
/// `hide` names the macros currently mid-expansion: an identifier in the hide set is emitted
/// literally rather than re-expanded, which is the C/GLSL rule that stops a self-referential
/// (`#define A A`) or mutually-recursive (`#define A B` / `#define B A`) macro from looping.
fn expandLine(allocator: std.mem.Allocator, out: *std.ArrayList(u8), line: []const u8, macros: []const Macro, hide: []const []const u8) Error!void {
    var i: usize = 0;
    while (i < line.len) {
        const c = line[i];
        // Skip `//` line comments verbatim (no expansion inside).
        if (c == '/' and i + 1 < line.len and line[i + 1] == '/') {
            try out.appendSlice(allocator, line[i..]);
            return;
        }
        if (std.ascii.isAlphabetic(c) or c == '_') {
            var j = i;
            while (j < line.len and (std.ascii.isAlphanumeric(line[j]) or line[j] == '_')) j += 1;
            const ident = line[i..j];
            const m = if (inHide(hide, ident)) null else lookup(macros, ident);
            if (m) |macro| {
                if (macro.params) |params| {
                    // Function-like: expands only when followed by a `(` argument list.
                    var k = j;
                    while (k < line.len and (line[k] == ' ' or line[k] == '\t')) k += 1;
                    if (k < line.len and line[k] == '(') {
                        var args: std.ArrayList([]const u8) = .empty;
                        defer args.deinit(allocator);
                        if (parseArgs(line, k, &args, allocator)) |end| {
                            if (args.items.len == params.len) {
                                var sub: std.ArrayList(u8) = .empty;
                                defer sub.deinit(allocator);
                                try substituteParams(allocator, &sub, macro.body, params, args.items);
                                const child_hide = try appendHide(allocator, hide, ident);
                                defer allocator.free(child_hide);
                                try expandLine(allocator, out, sub.items, macros, child_hide);
                                i = end;
                                continue;
                            }
                        } else |_| {}
                    }
                    // No call, a bad arg list, or an arg-count mismatch: emit the name as-is.
                    try out.appendSlice(allocator, ident);
                } else {
                    // Object-like: recurse with `ident` hidden so it cannot re-expand itself.
                    const child_hide = try appendHide(allocator, hide, ident);
                    defer allocator.free(child_hide);
                    try expandLine(allocator, out, macro.body, macros, child_hide);
                }
            } else {
                try out.appendSlice(allocator, ident);
            }
            i = j;
        } else {
            try out.append(allocator, c);
            i += 1;
        }
    }
}

/// Whether a physical line ends with a `\` continuation (tolerating a trailing CR).
fn endsWithBackslash(line: []const u8) bool {
    const l = if (line.len > 0 and line[line.len - 1] == '\r') line[0 .. line.len - 1] else line;
    return l.len > 0 and l[l.len - 1] == '\\';
}

/// A continued line with its trailing CR (if any) and continuation backslash removed.
fn dropBackslash(line: []const u8) []const u8 {
    const l = if (line.len > 0 and line[line.len - 1] == '\r') line[0 .. line.len - 1] else line;
    return l[0 .. l.len - 1];
}

/// Whether `name` is in the hide set (a macro currently being expanded).
fn inHide(hide: []const []const u8, name: []const u8) bool {
    for (hide) |h| if (std.mem.eql(u8, h, name)) return true;
    return false;
}

/// A fresh hide set = `hide` plus `name`. The caller frees it (transient per expansion).
fn appendHide(allocator: std.mem.Allocator, hide: []const []const u8, name: []const u8) Error![]const []const u8 {
    const out = try allocator.alloc([]const u8, hide.len + 1);
    @memcpy(out[0..hide.len], hide);
    out[hide.len] = name;
    return out;
}

/// Parse a function-like macro's argument list. `line[open]` is `(`; arguments are split on
/// top-level commas (commas nested in parentheses belong to an argument, e.g.
/// `ID(clamp(t,0.0,1.0))`). Returns the index just past the matching `)`, or an error if the
/// list does not close on this line (a call spanning lines is unsupported).
fn parseArgs(line: []const u8, open: usize, args: *std.ArrayList([]const u8), allocator: std.mem.Allocator) Error!usize {
    var i = open + 1;
    var depth: usize = 1;
    var arg_start = i;
    while (i < line.len) : (i += 1) {
        switch (line[i]) {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if (depth == 0) {
                    const a = std.mem.trim(u8, line[arg_start..i], " \t");
                    // An empty `()` is zero arguments, not one empty argument.
                    if (a.len > 0 or args.items.len > 0) try args.append(allocator, a);
                    return i + 1;
                }
            },
            ',' => if (depth == 1) {
                try args.append(allocator, std.mem.trim(u8, line[arg_start..i], " \t"));
                arg_start = i + 1;
            },
            else => {},
        }
    }
    return error.PreprocessError;
}

/// Substitute a function-like macro's parameters with the call arguments, copying the body
/// token by token (identifiers matching a parameter are replaced by the argument text).
fn substituteParams(allocator: std.mem.Allocator, out: *std.ArrayList(u8), body: []const u8, params: []const []const u8, args: []const []const u8) Error!void {
    var i: usize = 0;
    while (i < body.len) {
        const c = body[i];
        if (std.ascii.isAlphabetic(c) or c == '_') {
            var j = i;
            while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
            const id = body[i..j];
            if (paramIndex(params, id)) |idx| {
                try out.appendSlice(allocator, args[idx]);
            } else {
                try out.appendSlice(allocator, id);
            }
            i = j;
        } else {
            try out.append(allocator, c);
            i += 1;
        }
    }
}

fn paramIndex(params: []const []const u8, id: []const u8) ?usize {
    for (params, 0..) |p, idx| if (std.mem.eql(u8, p, id)) return idx;
    return null;
}

/// Define a function-like macro from `NAME` and the text starting at its `(`.
fn defineFunctionMacro(allocator: std.mem.Allocator, macros: *std.ArrayList(Macro), mname: []const u8, from_paren: []const u8) Error!void {
    const close = std.mem.indexOfScalar(u8, from_paren, ')') orelse return error.PreprocessError;
    const params_str = from_paren[1..close];
    const body = std.mem.trim(u8, from_paren[close + 1 ..], " \t");

    var plist: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (plist.items) |p| allocator.free(p);
        plist.deinit(allocator);
    }
    var it = std.mem.splitScalar(u8, params_str, ',');
    while (it.next()) |raw| {
        const p = std.mem.trim(u8, raw, " \t");
        if (p.len == 0) continue; // a zero-parameter macro `F()` has an empty parameter list
        try plist.append(allocator, try allocator.dupe(u8, p));
    }

    // Redefinition is lenient for function-like macros: replace any prior definition.
    removeMacro(allocator, macros, mname);
    try macros.append(allocator, .{
        .name = try allocator.dupe(u8, mname),
        .body = try allocator.dupe(u8, body),
        .params = try plist.toOwnedSlice(allocator),
        .owned = true,
    });
}

fn firstIdent(s: []const u8) []const u8 {
    const t = std.mem.trimStart(u8, s, " \t");
    var k: usize = 0;
    while (k < t.len and (std.ascii.isAlphanumeric(t[k]) or t[k] == '_')) k += 1;
    return t[0..k];
}

/// The leading run of decimal digits in `s` (after whitespace), or an empty slice.
fn firstNumber(s: []const u8) []const u8 {
    const t = std.mem.trimStart(u8, s, " \t");
    var k: usize = 0;
    while (k < t.len and std.ascii.isDigit(t[k])) k += 1;
    return t[0..k];
}

fn lookup(macros: []const Macro, name: []const u8) ?Macro {
    if (name.len == 0) return null;
    for (macros) |m| if (std.mem.eql(u8, m.name, name)) return m;
    return null;
}
fn lookupMut(macros: []Macro, name: []const u8) ?*Macro {
    for (macros) |*m| if (std.mem.eql(u8, m.name, name)) return m;
    return null;
}
fn removeMacro(allocator: std.mem.Allocator, macros: *std.ArrayList(Macro), name: []const u8) void {
    var i: usize = 0;
    while (i < macros.items.len) : (i += 1) {
        if (std.mem.eql(u8, macros.items[i].name, name)) {
            freeMacro(allocator, macros.orderedRemove(i));
            return;
        }
    }
}

/// Evaluate a `#if`/`#elif` condition expression: `defined(X)`, `defined X`, an identifier
/// (expands to its macro value or 0), integer literals, `!`, `&&`, `||`, `==`, `!=`, and
/// parentheses. A non-zero result is true.
fn evalCondition(macros: []const Macro, expr: []const u8) Error!bool {
    var ev = Eval{ .macros = macros, .src = expr, .pos = 0 };
    const v = try ev.parseOr();
    return v != 0;
}

const Eval = struct {
    macros: []const Macro,
    src: []const u8,
    pos: usize,

    fn skipWs(self: *Eval) void {
        while (self.pos < self.src.len and (self.src[self.pos] == ' ' or self.src[self.pos] == '\t')) self.pos += 1;
    }
    fn peekTok(self: *Eval) u8 {
        self.skipWs();
        return if (self.pos < self.src.len) self.src[self.pos] else 0;
    }

    fn parseOr(self: *Eval) Error!i64 {
        var lhs = try self.parseAnd();
        while (true) {
            self.skipWs();
            if (self.matchOp("||")) {
                const rhs = try self.parseAnd();
                lhs = if (lhs != 0 or rhs != 0) 1 else 0;
            } else break;
        }
        return lhs;
    }
    fn parseAnd(self: *Eval) Error!i64 {
        var lhs = try self.parseEq();
        while (true) {
            self.skipWs();
            if (self.matchOp("&&")) {
                const rhs = try self.parseEq();
                lhs = if (lhs != 0 and rhs != 0) 1 else 0;
            } else break;
        }
        return lhs;
    }
    fn parseEq(self: *Eval) Error!i64 {
        var lhs = try self.parseRel();
        while (true) {
            self.skipWs();
            if (self.matchOp("==")) {
                const rhs = try self.parseRel();
                lhs = if (lhs == rhs) 1 else 0;
            } else if (self.matchOp("!=")) {
                const rhs = try self.parseRel();
                lhs = if (lhs != rhs) 1 else 0;
            } else break;
        }
        return lhs;
    }
    /// Relational: `<=`/`>=`/`<`/`>`. The two-char forms are matched first so `>=` is not
    /// mistaken for `>` (the common `#if __VERSION__ >= 300` guard).
    fn parseRel(self: *Eval) Error!i64 {
        var lhs = try self.parseAdd();
        while (true) {
            self.skipWs();
            if (self.matchOp("<=")) {
                const rhs = try self.parseAdd();
                lhs = if (lhs <= rhs) 1 else 0;
            } else if (self.matchOp(">=")) {
                const rhs = try self.parseAdd();
                lhs = if (lhs >= rhs) 1 else 0;
            } else if (self.matchOp("<")) {
                const rhs = try self.parseAdd();
                lhs = if (lhs < rhs) 1 else 0;
            } else if (self.matchOp(">")) {
                const rhs = try self.parseAdd();
                lhs = if (lhs > rhs) 1 else 0;
            } else break;
        }
        return lhs;
    }
    fn parseAdd(self: *Eval) Error!i64 {
        var lhs = try self.parseMul();
        while (true) {
            self.skipWs();
            if (self.matchOp("+")) {
                lhs +%= try self.parseMul();
            } else if (self.matchOp("-")) {
                lhs -%= try self.parseMul();
            } else break;
        }
        return lhs;
    }
    fn parseMul(self: *Eval) Error!i64 {
        var lhs = try self.parseUnary();
        while (true) {
            self.skipWs();
            if (self.matchOp("*")) {
                lhs *%= try self.parseUnary();
            } else if (self.matchOp("/")) {
                const rhs = try self.parseUnary();
                // Guard the minInt/-1 overflow (illegal behavior) as well as div-by-zero;
                // -1 divides to the wrapping negation, matching the wraparound used above.
                lhs = if (rhs == 0) 0 else if (rhs == -1) 0 -% lhs else @divTrunc(lhs, rhs);
            } else if (self.matchOp("%")) {
                const rhs = try self.parseUnary();
                // x % -1 is 0; guarding it also avoids the minInt/-1 @rem overflow.
                lhs = if (rhs == 0) 0 else if (rhs == -1) 0 else @rem(lhs, rhs);
            } else break;
        }
        return lhs;
    }
    fn parseUnary(self: *Eval) Error!i64 {
        self.skipWs();
        if (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c == '!') {
                // Not a `!=` (handled above), a unary not.
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '=') {} else {
                    self.pos += 1;
                    const v = try self.parseUnary();
                    return if (v == 0) 1 else 0;
                }
            } else if (c == '-') {
                self.pos += 1;
                return -%(try self.parseUnary());
            } else if (c == '+') {
                self.pos += 1;
                return self.parseUnary();
            } else if (c == '~') {
                self.pos += 1;
                return ~(try self.parseUnary());
            }
        }
        return self.parsePrimary();
    }
    fn parsePrimary(self: *Eval) Error!i64 {
        self.skipWs();
        if (self.pos >= self.src.len) return 0;
        const c = self.src[self.pos];
        if (c == '(') {
            self.pos += 1;
            const v = try self.parseOr();
            self.skipWs();
            if (self.pos < self.src.len and self.src[self.pos] == ')') self.pos += 1;
            return v;
        }
        if (std.ascii.isDigit(c)) {
            var j = self.pos;
            while (j < self.src.len and std.ascii.isDigit(self.src[j])) j += 1;
            const v = std.fmt.parseInt(i64, self.src[self.pos..j], 10) catch 0;
            self.pos = j;
            return v;
        }
        if (std.ascii.isAlphabetic(c) or c == '_') {
            var j = self.pos;
            while (j < self.src.len and (std.ascii.isAlphanumeric(self.src[j]) or self.src[j] == '_')) j += 1;
            const ident = self.src[self.pos..j];
            self.pos = j;
            if (std.mem.eql(u8, ident, "defined")) {
                self.skipWs();
                var paren = false;
                if (self.pos < self.src.len and self.src[self.pos] == '(') {
                    paren = true;
                    self.pos += 1;
                }
                const mname = firstIdent(self.src[self.pos..]);
                self.skipWs();
                self.pos += leadingWs(self.src[self.pos..]) + mname.len;
                if (paren) {
                    self.skipWs();
                    if (self.pos < self.src.len and self.src[self.pos] == ')') self.pos += 1;
                }
                return if (lookup(self.macros, mname) != null) 1 else 0;
            }
            // A bare identifier expands to its macro value (parsed as int) or 0.
            if (lookup(self.macros, ident)) |m| {
                return std.fmt.parseInt(i64, std.mem.trim(u8, m.body, " \t"), 10) catch 1;
            }
            return 0;
        }
        // Unknown char: consume to avoid an infinite loop.
        self.pos += 1;
        return 0;
    }
    fn matchOp(self: *Eval, op: []const u8) bool {
        self.skipWs();
        if (self.pos + op.len <= self.src.len and std.mem.eql(u8, self.src[self.pos .. self.pos + op.len], op)) {
            self.pos += op.len;
            return true;
        }
        return false;
    }
};

fn leadingWs(s: []const u8) usize {
    var k: usize = 0;
    while (k < s.len and (s[k] == ' ' or s[k] == '\t')) k += 1;
    return k;
}

test "object-like define expansion" {
    const src = "#define X 3\nfloat a = X;\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "float a = 3;") != null);
}

test "ifdef GL_ES emits the ES branch" {
    const src = "#ifdef GL_ES\nprecision mediump float;\n#else\nint x;\n#endif\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "precision mediump float;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "int x;") == null);
}

test "if defined(A) && defined(B)" {
    const src = "#if defined(GL_ES) && defined(GL_FRAGMENT_PRECISION_HIGH)\n#define P highp\n#else\n#define P\n#endif\nP vec2 v;\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "highp vec2 v;") != null);
}

test "macro used as precision qualifier (glmark2 function scene)" {
    const src =
        \\#if defined(GL_ES) && defined(GL_FRAGMENT_PRECISION_HIGH)
        \\#define HIGHP_OR_DEFAULT highp
        \\#else
        \\#define HIGHP_OR_DEFAULT
        \\#endif
        \\HIGHP_OR_DEFAULT vec2 FragCoord = gl_FragCoord.xy;
        \\
    ;
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "highp vec2 FragCoord") != null);
}

test "if with relational operators (version check)" {
    // __VERSION__ is predefined to 100, so `>= 300` is false: the else branch wins. Before
    // relational support the `>= 300` was silently dropped and `#if __VERSION__` (100, i.e.
    // truthy) wrongly took the first branch.
    const src = "#if __VERSION__ >= 300\nint es3;\n#else\nint es1;\n#endif\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int es1;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "int es3;") == null);
}

test "if with arithmetic and relational precedence" {
    // 2 * 3 == 6 (mul binds tighter than ==), and 5 - 3 < 4 (additive tighter than <).
    try std.testing.expect(try evalCondition(&.{}, "2 * 3 == 6"));
    try std.testing.expect(try evalCondition(&.{}, "5 - 3 < 4"));
    try std.testing.expect(try evalCondition(&.{}, "10 % 3 == 1"));
    try std.testing.expect(!try evalCondition(&.{}, "3 > 4"));
    try std.testing.expect(try evalCondition(&.{}, "-1 < 0"));
    try std.testing.expect(try evalCondition(&.{}, "7 >= 7 && 2 <= 2"));
}

test "line continuation joins physical lines" {
    // A trailing backslash splices the next physical line on, so a macro body can span lines.
    const out = try run(std.testing.allocator, "#define BIG one \\\ntwo\nint x = BIG;\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int x = one two;") != null);
}

test "line continuation in a function-like macro" {
    const src = "#define ADD(a, b) \\\n    ((a) + (b))\nfloat s = ADD(p, q);\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "float s = ((p) + (q));") != null);
}

test "function-like macro: single argument" {
    const out = try run(std.testing.allocator, "#define SQ(x) ((x)*(x))\nfloat a = SQ(3.0);\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "float a = ((3.0)*(3.0));") != null);
}

test "function-like macro: multiple arguments" {
    const out = try run(std.testing.allocator, "#define MX(a,b) ((a)>(b)?(a):(b))\nfloat m = MX(p, q);\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "float m = ((p)>(q)?(p):(q));") != null);
}

test "function-like macro: commas inside parens are not argument separators" {
    const out = try run(std.testing.allocator, "#define ID(x) x\nfloat v = ID(clamp(t,0.0,1.0));\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "float v = clamp(t,0.0,1.0);") != null);
}

test "function-like macro: name without a call is left alone" {
    // A function-like macro identifier not followed by `(` is not a macro invocation.
    const out = try run(std.testing.allocator, "#define F(x) (x)\nint F = 3;\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int F = 3;") != null);
}

test "function-like macro: nested expansion in arguments" {
    const out = try run(std.testing.allocator, "#define HALF 0.5\n#define SC(x) ((x)*HALF)\nfloat s = SC(2.0);\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "float s = ((2.0)*0.5);") != null);
}

test "self-referential macro expands once (no infinite loop)" {
    // `#define A A` must expand A to A and stop (the hide set blocks re-expansion). Before
    // the hide set this recursed forever and crashed the preprocessor.
    const out = try run(std.testing.allocator, "#define A A\nint x = A;\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int x = A;") != null);
}

test "mutually recursive macros terminate" {
    // A -> B -> A: each is hidden while expanding, so A expands to B expands to A and stops.
    const out = try run(std.testing.allocator, "#define A B\n#define B A\nint y = A;\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int y = A;") != null);
}

test "nested non-recursive macros still fully expand" {
    // A -> B -> 3: distinct names, no hiding conflict, expands all the way.
    const out = try run(std.testing.allocator, "#define B 3\n#define A B\nint z = A;\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int z = 3;") != null);
}

test "version directive sets __VERSION__" {
    // `#version 300 es` makes `__VERSION__` 300, so the >= 300 guard takes the ES3 branch.
    const src = "#version 300 es\n#if __VERSION__ >= 300\nint es3;\n#else\nint es1;\n#endif\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "int es3;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "int es1;") == null);
}

test "undef removes a macro" {
    const src = "#define X 1\n#undef X\n#ifdef X\nint defined_path;\n#else\nint undef_path;\n#endif\n";
    const out = try run(std.testing.allocator, src);
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "undef_path") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "defined_path") == null);
}

test "redefining a macro to a different value is an error (GLSL ES 3.4), identical is allowed" {
    // The real NVIDIA driver rejects this (error C7101). Several glmark2 scenes assemble a
    // shader that #defines HIGHP_OR_DEFAULT twice with different bodies.
    try std.testing.expectError(error.MacroRedefined, run(std.testing.allocator, "#define X highp\n#define X mediump\n"));
    // Identical redefinition is a legal no-op.
    const out = try run(std.testing.allocator, "#define X highp\n#define X highp\nX vec2 a;\n");
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "highp vec2 a") != null);
}
