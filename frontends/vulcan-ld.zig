//! `ld.vulcan`: a standalone linker frontend CLI over the shared static linker
//! (`vulcan-link`), so the linker is usable with any compiler's relocatable output,
//! not just Vulcan's own frontends. Reads `.o`/`.a` inputs (auto-detected via the
//! `!<arch>\n` archive magic), resolves `-l`/`-L` static-library references,
//! `linkInputs`s them into one image, and wraps the image in a runnable ELF via
//! `writeExecutable`. The target architecture is never asked for on the command
//! line: it is read straight out of the first object input's `e_machine`, exactly
//! like the shared linker library itself infers it per object.
//!
//! The core is `run`, which takes argv and returns a process exit status rather than
//! erroring or exiting directly, so a test can drive the whole CLI in-process (no
//! `std.process.run` spawn needed) and assert on both the returned status and any
//! produced file. `main` is a thin wrapper: gather argv, call `run`, exit with its
//! status.

const std = @import("std");
const builtin = @import("builtin");
const ld = @import("vulcan-link");

/// One positional/`-l` input, in command-line order. Resolved to an actual classified
/// input only after the full argument scan, so a `-l` can be searched against every
/// `-L` directory given anywhere on the line, while archive-pull order (which only
/// depends on the relative order of `.object`/`.archive`/`.shared` entries, not `-L`)
/// still matches the command line. A `.lib` spec also carries the `-Bdynamic`/
/// `-Bstatic` search preference in effect at the point it was scanned (SM10 P4d):
/// `prefer_dynamic` means "try `lib<name>.so` before `lib<name>.a`". This is only ONE
/// input to the actual resolution preference, though - see the classification loop in
/// `run`, which also folds in whole-link dynamic intent (`-shared`/`--dynamic-linker`)
/// before calling `resolveLib`.
const InputSpec = union(enum) {
    path: []const u8,
    lib: struct { name: []const u8, prefer_dynamic: bool },
};

fn diag(errw: *std.Io.Writer, comptime fmt: []const u8, args: anytype) u8 {
    errw.print("ld.vulcan: error: " ++ fmt ++ "\n", args) catch {};
    errw.flush() catch {};
    return 1;
}

fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));
}

/// Which kind of library `resolveLib` found: an `.a` archive (fed to the static/`-T`
/// linker unchanged) or a `.so` shared object (a `ld.DynInput.shared`, only reachable
/// when dynamic linking is in play).
const LibKind = enum { archive, shared };

const LibFound = struct { bytes: []u8, kind: LibKind };

/// Read `path`, turning "not found" into `null` (expected when probing a candidate
/// filename/kind that just isn't there - the caller tries the next one) rather than an
/// error. Any other read failure propagates.
fn tryReadLib(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !?[]u8 {
    return readFile(allocator, io, path) catch |e| switch (e) {
        error.FileNotFound => null,
        else => return e,
    };
}

/// Search `dirs` IN ORDER for `lib<name>`, and within EACH directory try the preferred
/// kind (`.so` when `prefer_dynamic`, `.a` otherwise) before the other kind, matching a
/// real `ld`'s own per-directory search: a directory containing only the non-preferred
/// kind still wins over a LATER directory that has the preferred one, since it is
/// checked first. (An earlier, buggier shape here searched every directory for one kind
/// before considering the other kind in any of them - wrong whenever an earlier `-L`
/// directory has only the non-preferred kind and a later one has both.)
///
/// `null` means not found in any of them (not an error - the caller turns that into a
/// clear diagnostic). Any read failure other than "not found" propagates.
fn resolveLib(allocator: std.mem.Allocator, io: std.Io, name: []const u8, dirs: []const []const u8, prefer_dynamic: bool) !?LibFound {
    const so_filename = try std.fmt.allocPrint(allocator, "lib{s}.so", .{name});
    defer allocator.free(so_filename);
    const a_filename = try std.fmt.allocPrint(allocator, "lib{s}.a", .{name});
    defer allocator.free(a_filename);

    for (dirs) |dir| {
        const so_path = try std.fs.path.join(allocator, &.{ dir, so_filename });
        defer allocator.free(so_path);
        const a_path = try std.fs.path.join(allocator, &.{ dir, a_filename });
        defer allocator.free(a_path);

        const first_path = if (prefer_dynamic) so_path else a_path;
        const first_kind: LibKind = if (prefer_dynamic) .shared else .archive;
        if (try tryReadLib(allocator, io, first_path)) |bytes| return .{ .bytes = bytes, .kind = first_kind };

        const second_path = if (prefer_dynamic) a_path else so_path;
        const second_kind: LibKind = if (prefer_dynamic) .archive else .shared;
        if (try tryReadLib(allocator, io, second_path)) |bytes| return .{ .bytes = bytes, .kind = second_kind };
    }
    return null;
}

/// The `e_machine` field lives at the same byte offset (18) in both the ELF32 and
/// ELF64 header layouts, so this reads straight off the raw bytes without needing to
/// know the class up front.
fn readEMachine(bytes: []const u8) ?ld.Arch {
    if (bytes.len < 20) return null;
    return ld.fromEMachine(std.mem.readInt(u16, bytes[18..20], .little));
}

/// `ET_DYN` (a shared object or a PIE), the `e_type` value that (together with a `.so`
/// filename suffix) marks a path input as a dynamic-linking `.shared` input rather than
/// a plain relocatable `.object`. `e_type` lives at byte offset 16 in both the ELF32 and
/// ELF64 header layouts, mirroring `readEMachine`'s offset-18 `e_machine` read.
const ET_DYN: u16 = 3;

fn readEType(bytes: []const u8) ?u16 {
    if (bytes.len < 18) return null;
    return std.mem.readInt(u16, bytes[16..18], .little);
}

/// A path input is a dynamic `.shared` input iff its name ends `.so` or its ELF header
/// says `ET_DYN` (a `.so` given under a non-standard name still round-trips correctly).
fn isSharedObjectPath(path: []const u8, bytes: []const u8) bool {
    if (std.mem.endsWith(u8, path, ".so")) return true;
    return readEType(bytes) == ET_DYN;
}

/// The architecture to link for and to pick the default image base for: the first
/// `.object` input's `e_machine`, in command-line order; if there is none, the first
/// `.archive` input's first member's `e_machine` (a `-l`-only link with no plain `.o`
/// on the line, e.g. a stub-free static-library-only invocation).
fn detectArch(allocator: std.mem.Allocator, inputs: []const ld.Input) ?ld.Arch {
    for (inputs) |in| switch (in) {
        .object => |bytes| {
            if (readEMachine(bytes)) |a| return a;
        },
        .archive => {},
    };
    for (inputs) |in| switch (in) {
        .object => {},
        .archive => |bytes| {
            const members = ld.archive.parseArchive(allocator, bytes) catch continue;
            defer allocator.free(members);
            for (members) |m| {
                if (readEMachine(m.bytes)) |a| return a;
            }
        },
    };
    return null;
}

/// The default image base per architecture, used unless `--image-base` overrides it.
/// aarch64/x86_64/riscv64 share the conventional `0x400000` static-executable base
/// (also what the in-tree link tests use); `x86` uses `0x08048000`, the classic i386
/// ELF base (matches `x86/tests/link_native.zig`).
fn defaultBase(arch: ld.Arch) u64 {
    return switch (arch) {
        .aarch64, .x86_64, .riscv64 => 0x400000,
        .x86 => 0x08048000,
    };
}

/// One human-readable line per `ld.Error` variant, for a diagnostic that doesn't leak
/// a bare Zig error name at the user.
fn linkErrorMessage(e: ld.Error) []const u8 {
    return switch (e) {
        error.OutOfMemory => "out of memory",
        error.MalformedObject => "malformed object or archive input",
        error.UndefinedSymbol => "undefined symbol reference could not be resolved",
        error.DuplicateSymbol => "duplicate symbol definition across inputs",
        error.RelocationOutOfRange => "a relocation target is out of range",
        error.UnsupportedReloc => "an input uses an unsupported relocation type",
    };
}

/// One human-readable line per `ld.ScriptError` variant (the `-T` link path), for the
/// same clean-diagnostic treatment as `linkErrorMessage`. `ld.ScriptError` is
/// `ld.Error || ld.ScriptParseError || {ScriptRegionOverflow, ScriptUndefinedRegion,
/// ScriptUndefinedSymbol, ScriptDivByZero}`; the generic-linker variants share
/// `linkErrorMessage`'s wording. `error.ScriptSyntax` cannot actually surface here
/// (the script already parsed successfully by the time `linkInputsScript` runs) but is
/// listed for switch exhaustiveness.
fn scriptErrorMessage(e: ld.ScriptError) []const u8 {
    return switch (e) {
        error.OutOfMemory => "out of memory",
        error.MalformedObject => "malformed object or archive input",
        error.UndefinedSymbol => "undefined symbol reference could not be resolved",
        error.DuplicateSymbol => "duplicate symbol definition across inputs",
        error.RelocationOutOfRange => "a relocation target is out of range",
        error.UnsupportedReloc => "an input uses an unsupported relocation type",
        error.ScriptSyntax => "linker script syntax error",
        error.ScriptRegionOverflow => "a linker-script MEMORY region overflowed",
        error.ScriptUndefinedRegion => "linker script references an undefined MEMORY region",
        error.ScriptUndefinedSymbol => "linker script references an undefined symbol",
        error.ScriptDivByZero => "linker script expression divides by zero",
    };
}

/// Parse `args` (no argv0), link, and write the executable. Returns the process exit
/// status: 0 on success, non-zero (with a diagnostic on stderr) on any failure. Only
/// `error.OutOfMemory` ever propagates as a real Zig error - every other failure mode
/// (bad flag, unreadable input, missing `-l` library, link error) is reported and
/// turned into a non-zero status here.
pub fn run(allocator: std.mem.Allocator, io: std.Io, errw: *std.Io.Writer, args: []const []const u8) !u8 {
    var output: []const u8 = "a.out";
    var entry: []const u8 = "_start";
    // Whether `-e` was actually given on the command line, as opposed to `entry` still
    // holding its `"_start"` default. Only the `-T` (script) path cares about this: a
    // script's own `ENTRY(...)` is the default entry there, and `-e` must OVERRIDE it
    // only when the user actually asked for that (see the `-T` branch below).
    var entry_explicit = false;
    var image_base: ?u64 = null;
    // `-T <script>`: when set, link through the script-driven layout engine
    // (`ld.linkInputsScript` + `ld.writeElfSegments`) instead of the default
    // `ld.linkInputs` + `ld.writeExecutable` path.
    var script_path: ?[]const u8 = null;

    // Dynamic-linking flags (SM10 P4d). `shared_flag` (`-shared`) and `dynamic_linker`
    // (`--dynamic-linker`) are the two ways to ASK for the dynamic route explicitly; a
    // plain `-l`/path resolving to a `.so` also triggers it (see `dynamic_mode` below).
    // `-Bdynamic`/`-Bstatic`/`--static` only steer `-l` search preference, captured onto
    // each `.lib` spec as it is scanned (`prefer_dynamic_pref`), matching GNU ld's own
    // positional `-B*` semantics; the default preference is STATIC-first (`.a` before
    // `.so`) - a deliberate divergence from a real `ld` (whose default IS dynamic-first),
    // because unlike a real `ld` this linker has no default dynamic interpreter of its
    // own. Dynamic-first-by-default would mean a plain `-lfoo` on a system that happens
    // to have `libfoo.so` next to `libfoo.a` silently diverts a plain static-intent link
    // into the dynamic route, where it then fails with "needs --dynamic-linker" - a
    // footgun for a link that never asked for dynamic linking at all. `-Bdynamic` (or an
    // explicit whole-link dynamic intent - see the classification loop below) opts back
    // into dynamic-first for the `-l`s it covers.
    var shared_flag = false;
    var dynamic_linker: ?[]const u8 = null;
    var soname: ?[]const u8 = null;
    var prefer_dynamic_pref = false;

    var search_dirs: std.ArrayList([]const u8) = .empty;
    defer search_dirs.deinit(allocator);

    var specs: std.ArrayList(InputSpec) = .empty;
    defer specs.deinit(allocator);

    var owned_bufs: std.ArrayList([]u8) = .empty;
    defer {
        for (owned_bufs.items) |b| allocator.free(b);
        owned_bufs.deinit(allocator);
    }

    // Owned strings allocated only for a `-l`-resolved `.so`'s DT_NEEDED fallback name
    // (`lib<name>.so`, when the `.so` itself carries no `DT_SONAME`); freed at the end.
    var owned_strs: std.ArrayList([]u8) = .empty;
    defer {
        for (owned_strs.items) |s| allocator.free(s);
        owned_strs.deinit(allocator);
    }

    var inputs: std.ArrayList(ld.Input) = .empty;
    defer inputs.deinit(allocator);

    // Every input, classified in command order as `ld.DynInput` (`.object`/`.archive`/
    // `.shared`) regardless of which route ends up handling the link: the static/`-T`
    // path only ever sees `.object`/`.archive` members (converted to `ld.Input` below)
    // unless `dynamic_mode` is set, in which case this feeds `ld.linkDynamic` directly.
    // `display_names` is a parallel array (same length/order): the input's basename, used
    // only as a `.shared` input's DT_NEEDED fallback name when its `.so` has no SONAME;
    // empty for `.object`/`.archive` entries.
    var dyn_inputs: std.ArrayList(ld.DynInput) = .empty;
    defer dyn_inputs.deinit(allocator);
    var display_names: std.ArrayList([]const u8) = .empty;
    defer display_names.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-o")) {
            i += 1;
            if (i >= args.len) return diag(errw, "-o requires an argument", .{});
            output = args[i];
        } else if (std.mem.eql(u8, arg, "-e")) {
            i += 1;
            if (i >= args.len) return diag(errw, "-e requires an argument", .{});
            entry = args[i];
            entry_explicit = true;
        } else if (std.mem.eql(u8, arg, "-T")) {
            i += 1;
            if (i >= args.len) return diag(errw, "-T requires an argument", .{});
            script_path = args[i];
        } else if (std.mem.startsWith(u8, arg, "-T") and arg.len > 2) {
            script_path = arg[2..];
        } else if (std.mem.eql(u8, arg, "--image-base")) {
            i += 1;
            if (i >= args.len) return diag(errw, "--image-base requires an argument", .{});
            image_base = std.fmt.parseInt(u64, args[i], 0) catch return diag(errw, "--image-base: invalid value '{s}'", .{args[i]});
        } else if (std.mem.eql(u8, arg, "--static") or std.mem.eql(u8, arg, "-Bstatic")) {
            prefer_dynamic_pref = false;
        } else if (std.mem.eql(u8, arg, "-Bdynamic")) {
            prefer_dynamic_pref = true;
        } else if (std.mem.eql(u8, arg, "-shared")) {
            shared_flag = true;
        } else if (std.mem.eql(u8, arg, "--dynamic-linker")) {
            i += 1;
            if (i >= args.len) return diag(errw, "--dynamic-linker requires an argument", .{});
            dynamic_linker = args[i];
        } else if (std.mem.startsWith(u8, arg, "--dynamic-linker=")) {
            dynamic_linker = arg["--dynamic-linker=".len..];
        } else if (std.mem.eql(u8, arg, "-soname") or std.mem.eql(u8, arg, "-h")) {
            i += 1;
            if (i >= args.len) return diag(errw, "-soname requires an argument", .{});
            soname = args[i];
        } else if (std.mem.eql(u8, arg, "-L")) {
            i += 1;
            if (i >= args.len) return diag(errw, "-L requires an argument", .{});
            try search_dirs.append(allocator, args[i]);
        } else if (std.mem.startsWith(u8, arg, "-L") and arg.len > 2) {
            try search_dirs.append(allocator, arg[2..]);
        } else if (std.mem.eql(u8, arg, "-l")) {
            i += 1;
            if (i >= args.len) return diag(errw, "-l requires an argument", .{});
            try specs.append(allocator, .{ .lib = .{ .name = args[i], .prefer_dynamic = prefer_dynamic_pref } });
        } else if (std.mem.startsWith(u8, arg, "-l") and arg.len > 2) {
            try specs.append(allocator, .{ .lib = .{ .name = arg[2..], .prefer_dynamic = prefer_dynamic_pref } });
        } else if (std.mem.startsWith(u8, arg, "-")) {
            return diag(errw, "unknown argument '{s}'", .{arg});
        } else {
            try specs.append(allocator, .{ .path = arg });
        }
    }

    if (specs.items.len == 0) return diag(errw, "no input files", .{});

    // Classify every input in command order into `ld.DynInput`. This is a superset of
    // the OLD (static-only) resolution: when nothing here ever turns out `.shared`, the
    // classification below is byte-for-byte the same decision the old code made
    // (`ld.isArchive` for a path, always `.archive` for a `-l`), so the static/`-T`
    // route stays unaffected by this refactor.
    for (specs.items) |spec| switch (spec) {
        .path => |p| {
            const bytes = readFile(allocator, io, p) catch |e| return diag(errw, "cannot read '{s}': {s}", .{ p, @errorName(e) });
            try owned_bufs.append(allocator, bytes);
            if (ld.isArchive(bytes)) {
                try dyn_inputs.append(allocator, .{ .archive = bytes });
                try display_names.append(allocator, "");
            } else if (isSharedObjectPath(p, bytes)) {
                try dyn_inputs.append(allocator, .{ .shared = bytes });
                try display_names.append(allocator, std.fs.path.basename(p));
            } else {
                try dyn_inputs.append(allocator, .{ .object = bytes });
                try display_names.append(allocator, "");
            }
        },
        .lib => |l| {
            // Whole-link dynamic intent (`-shared`/`--dynamic-linker`) can appear
            // ANYWHERE on the command line, including after this `-l` - but by the time
            // this classification loop runs, the full arg scan has already completed, so
            // `shared_flag`/`dynamic_linker` are fully known here. Folding them in (on
            // top of the `-Bdynamic`/`-Bstatic` state captured at scan time) means a
            // link that is dynamic ANYWAY still gets dynamic-first `-l` resolution by
            // default (matching the existing `--dynamic-linker ... -ladd` E2E below),
            // while a plain static-intent link keeps the conservative `.a`-first default.
            const prefer_dynamic = l.prefer_dynamic or shared_flag or dynamic_linker != null;
            const found = resolveLib(allocator, io, l.name, search_dirs.items, prefer_dynamic) catch |e|
                return diag(errw, "cannot search for -l{s}: {s}", .{ l.name, @errorName(e) });
            const res = found orelse {
                if (prefer_dynamic)
                    return diag(errw, "cannot find -l{s}: no 'lib{s}.so' or 'lib{s}.a' in any -L search directory", .{ l.name, l.name, l.name });
                return diag(errw, "cannot find -l{s}: no 'lib{s}.a' in any -L search directory", .{ l.name, l.name });
            };
            try owned_bufs.append(allocator, res.bytes);
            switch (res.kind) {
                .archive => {
                    try dyn_inputs.append(allocator, .{ .archive = res.bytes });
                    try display_names.append(allocator, "");
                },
                .shared => {
                    try dyn_inputs.append(allocator, .{ .shared = res.bytes });
                    const fname = try std.fmt.allocPrint(allocator, "lib{s}.so", .{l.name});
                    try owned_strs.append(allocator, fname);
                    try display_names.append(allocator, fname);
                },
            }
        },
    };

    // Dynamic-mode TRIGGER (SM10 P4d): `-shared`, `--dynamic-linker`, or any input that
    // actually resolved to a `.so`. Absent all three, every classified entry above is
    // `.object`/`.archive` and the conversion below reproduces the exact `ld.Input` array
    // the pre-P4d code built, so the static/`-T` route runs byte-unchanged.
    var dynamic_mode = shared_flag or dynamic_linker != null;
    for (dyn_inputs.items) |di| {
        if (di == .shared) dynamic_mode = true;
    }

    if (dynamic_mode and script_path != null)
        return diag(errw, "-T (linker scripts) does not support dynamic linking (-shared/--dynamic-linker/.so inputs)", .{});

    if (!dynamic_mode) {
        for (dyn_inputs.items) |di| switch (di) {
            .object => |b| try inputs.append(allocator, .{ .object = b }),
            .archive => |b| try inputs.append(allocator, .{ .archive = b }),
            .shared => unreachable, // dynamic_mode would be true otherwise
        };
    }

    // `-T <script>`: a self-contained flow, parse -> `linkInputsScript` ->
    // `writeElfSegments`. The architecture is inferred from the objects by
    // `linkInputsScript` itself (not `detectArch`/`--image-base`, which only matter to
    // the default contiguous layout below - a script's own location-counter commands
    // control every address instead).
    if (script_path) |sp| {
        const script_bytes = readFile(allocator, io, sp) catch |e| return diag(errw, "cannot read '{s}': {s}", .{ sp, @errorName(e) });
        defer allocator.free(script_bytes);

        var sdiag: ld.ScriptDiagnostic = .{ .line = 0, .col = 0, .msg = "" };
        var parsed_script = ld.parseScript(allocator, script_bytes, &sdiag) catch |e| {
            if (e == error.OutOfMemory) return e;
            return diag(errw, "{s}:{d}:{d}: {s}", .{ sp, sdiag.line, sdiag.col, sdiag.msg });
        };
        defer parsed_script.deinit();

        var linked = ld.linkInputsScript(allocator, inputs.items, &parsed_script, null) catch |e| {
            if (e == error.OutOfMemory) return e;
            return diag(errw, "{s}", .{scriptErrorMessage(e)});
        };
        defer linked.deinit(allocator);

        // `-e` overrides the script's own `ENTRY(...)` (`linked.entry`); absent `-e`,
        // the script's entry stands (0 if the script gave none - not this frontend's
        // problem to invent one).
        const entry_addr = if (entry_explicit)
            ld.elf.findSymbol(linked.placement.symbols, entry) orelse
                return diag(errw, "undefined entry symbol '{s}'", .{entry})
        else
            linked.entry;

        const elf_bytes = try ld.writeElfSegments(linked.arch, allocator, &linked.placement, entry_addr);
        defer allocator.free(elf_bytes);

        std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = output,
            .data = elf_bytes,
            .flags = .{ .permissions = .executable_file },
        }) catch |e| return diag(errw, "cannot write '{s}': {s}", .{ output, @errorName(e) });

        return 0;
    }

    // The dynamic route (SM10 P4d): `ld.linkDynamic` takes the classified `.object`/
    // `.archive`/`.shared` inputs directly (it resolves imports against `.shared`
    // exports itself - no PLT/GOT bookkeeping needed here) and returns the finished ELF
    // bytes (a `.so` for `-shared`, otherwise a dynamic executable).
    if (dynamic_mode) {
        const mode: ld.DynMode = if (shared_flag) .shared else .exec;
        if (mode == .exec and dynamic_linker == null)
            return diag(errw, "a dynamic executable needs --dynamic-linker <path>", .{});

        // DT_NEEDED: one entry per `.shared` input, preferring its own DT_SONAME and
        // falling back to its basename (the `.so` path's basename, or the `-l`-resolved
        // `lib<name>.so` filename) when it carries none.
        var needed_list: std.ArrayList([]const u8) = .empty;
        defer {
            for (needed_list.items) |n| allocator.free(n);
            needed_list.deinit(allocator);
        }
        for (dyn_inputs.items, display_names.items) |di, disp| switch (di) {
            .shared => |bytes| {
                var se = ld.readSharedExports(allocator, bytes) catch |e| {
                    if (e == error.OutOfMemory) return e;
                    return diag(errw, "cannot read a shared object's exports: {s}", .{@errorName(e)});
                };
                defer se.deinit(allocator);
                const name = if (se.soname) |s| try allocator.dupe(u8, s) else try allocator.dupe(u8, disp);
                try needed_list.append(allocator, name);
            },
            else => {},
        };

        const dyn_bytes = ld.linkDynamic(allocator, dyn_inputs.items, .{
            .mode = mode,
            .interp = dynamic_linker,
            .soname = soname,
            .needed = needed_list.items,
            .base = image_base orelse 0x400000,
            .entry = entry,
        }) catch |e| {
            if (e == error.OutOfMemory) return e;
            return diag(errw, "{s}", .{linkErrorMessage(e)});
        };
        defer allocator.free(dyn_bytes);

        // A `.so` gets exec permissions too: harmless (it is never invoked directly) and
        // conventional (matches what a real `ld -shared` produces).
        std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = output,
            .data = dyn_bytes,
            .flags = .{ .permissions = .executable_file },
        }) catch |e| return diag(errw, "cannot write '{s}': {s}", .{ output, @errorName(e) });

        return 0;
    }

    const arch = detectArch(allocator, inputs.items) orelse
        return diag(errw, "could not determine the target architecture (no valid object input found)", .{});
    const base = image_base orelse defaultBase(arch);

    var image = ld.linkInputs(allocator, inputs.items, base) catch |e| {
        if (e == error.OutOfMemory) return e;
        return diag(errw, "{s}", .{linkErrorMessage(e)});
    };
    defer image.deinit(allocator);

    const elf_bytes = ld.writeExecutable(arch, allocator, &image, entry) catch |e| {
        if (e == error.OutOfMemory) return e;
        return diag(errw, "{s}", .{linkErrorMessage(e)});
    };
    defer allocator.free(elf_bytes);

    std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = output,
        .data = elf_bytes,
        .flags = .{ .permissions = .executable_file },
    }) catch |e| return diag(errw, "cannot write '{s}': {s}", .{ output, @errorName(e) });

    return 0;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // argv0

    var args: std.ArrayList([]const u8) = .empty;
    while (it.next()) |a| try args.append(allocator, a);

    // The single diagnostic sink for this invocation: `diag` routes every error line
    // through here to the real stderr. In-process tests instead point their own
    // buffer-backed writer at `run`, so a negative test never leaks an `ld.vulcan:
    // error:` line to the test runner's stderr.
    var errbuf: [512]u8 = undefined;
    var errfile = std.Io.File.stderr().writer(io, &errbuf);
    const errw = &errfile.interface;

    const status = try run(allocator, io, errw, args.items);
    std.process.exit(status);
}

// ---------------------------------------------------------------------------------
// Tests. Build synthetic `.o`/`.a` inputs the same way the shared linker's own
// consumer tests do (`libs/vulcan-target/aarch64/tests/link_archive.zig`,
// `libs/vulcan-cc/tests/native.zig`), then drive `run` directly (no process spawn)
// and natively execute the produced ELF. Host-gated: the hand-assembled `_start`
// stub and the executed ELF are AArch64-specific.

const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const Function = ir.function.Function;

// A buffer-backed diagnostic sink for the in-process tests: `run` writes any `ld.vulcan:
// error:` line here instead of the real stderr, so a negative test asserts on the returned
// status without leaking error text that would make a passing test look failed. Each call
// resets the buffer, so a test may inspect `test_errbuf[0..testErrw().end]` if it wants.
var test_errbuf: [512]u8 = undefined;
var test_errw: std.Io.Writer = undefined;
fn testErrw() *std.Io.Writer {
    test_errw = std.Io.Writer.fixed(&test_errbuf);
    return &test_errw;
}

/// How many times `runExpectExit` attempts a run before giving up. Only a non-`.exited`
/// term consumes an attempt; a clean `.exited` result is checked and returned on
/// immediately. Mirrors `libs/vulcan-target/tests/run_helper.zig`'s own retry helper
/// (not importable here across the module boundary - `frontends/vulcan-ld.zig` is its
/// own root module - so this is a small, deliberate duplicate, not a divergent copy).
const run_max_attempts = 5;
const run_retry_backoff_ms = 20;

/// Run `options` up to `run_max_attempts` times and assert the child's exit code equals
/// `expected_exit`, retrying only when `run.term` is not `.exited` (a transient spawn
/// race, not a wrong-answer regression - the moment a run DOES exit, the code is
/// asserted immediately with no retry).
fn runExpectExit(
    allocator: std.mem.Allocator,
    io: std.Io,
    options: std.process.RunOptions,
    expected_exit: u8,
) !void {
    var attempt: usize = 0;
    while (true) {
        attempt += 1;
        const proc = try std.process.run(allocator, io, options);
        defer allocator.free(proc.stdout);
        defer allocator.free(proc.stderr);

        switch (proc.term) {
            .exited => |code| {
                try std.testing.expectEqual(expected_exit, code);
                return;
            },
            else => {
                if (attempt >= run_max_attempts) return error.TestUnexpectedResult;
                std.Io.sleep(io, .fromMilliseconds(run_retry_backoff_ms), .awake) catch {};
                continue;
            },
        }
    }
}

/// `helper() = 42` as its own one-function module -> ELF `.o` bytes.
fn buildHelperObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var helper = Function.init(allocator);
    defer helper.deinit();
    const t = try helper.types.intern(i32k);
    const b = try helper.appendBlock();
    const v = try helper.appendInst(b, t, .{ .iconst = 42 });
    helper.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    return target.native.writeObjectData(allocator, &.{.{ .name = "helper", .func = &helper }}, &.{});
}

/// `main() = helper()` as its own one-function module, referencing `helper` as an
/// external symbol only an archive member can satisfy.
fn buildMainObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var main_fn = Function.init(allocator);
    defer main_fn.deinit();
    const t = try main_fn.types.intern(i32k);
    const b = try main_fn.appendBlock();
    const r = try main_fn.appendCall(b, t, "helper", &.{});
    main_fn.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    return target.native.writeObjectData(allocator, &.{.{ .name = "main", .func = &main_fn }}, &.{});
}

/// `add(a, b) = a + b`, a leaf two-parameter function exported as a global - the SM10
/// P4d dynamic-CLI E2E's `.so` fixture (`libadd.so`'s single export).
fn buildAddObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var add_fn = Function.init(allocator);
    defer add_fn.deinit();
    const t = try add_fn.types.intern(i32k);
    const b = try add_fn.appendBlock();
    const a_param = try add_fn.appendBlockParam(b, t);
    const b_param = try add_fn.appendBlockParam(b, t);
    const sum = try add_fn.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a_param, .rhs = b_param } });
    add_fn.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });
    return target.native.writeObjectData(allocator, &.{.{ .name = "add", .func = &add_fn }}, &.{});
}

/// `unused() = 7`, a leaf function exported as a global that NOTHING in the SM10 P4d
/// DT_NEEDED-derivation test's link unit ever calls - the fixture that makes the CLI's
/// own DT_NEEDED/soname derivation loop (in `run`'s dynamic-mode branch) load-bearing:
/// `ld.linkDynamic` only auto-derives a `DT_NEEDED` entry from a `.shared` input whose
/// export some CALL relocation actually resolves against, so a `.so` built from this
/// function is never referenced by any import site and its soname would be silently
/// dropped from the produced dynexe if the CLI's OWN loop were buggy.
fn buildUnusedObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var unused_fn = Function.init(allocator);
    defer unused_fn.deinit();
    const t = try unused_fn.types.intern(i32k);
    const b = try unused_fn.appendBlock();
    const v = try unused_fn.appendInst(b, t, .{ .iconst = 7 });
    unused_fn.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    return target.native.writeObjectData(allocator, &.{.{ .name = "unused", .func = &unused_fn }}, &.{});
}

/// A `_start` that CALLS the imported `add(1, 41)` - an undefined, external symbol from
/// this object's own point of view, satisfied at dynamic-link time by `libadd.so` - and
/// exits with the result (already in `x0` per AAPCS64): the SM10 P4d dynamic-CLI E2E's
/// start stub. `bl add` at `.text` offset 8 carries a real `R_AARCH64_CALL26`
/// relocation, mirroring `startStubAArch64` above but with `add`'s two AAPCS64 integer
/// args loaded first.
fn startStubCallAddAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.native.backend.encode;
    const object = target.native.backend.object;

    const code = [_]u32{
        encode.movz(.x0, 1, 0), // x0 = 1  (add's first arg)   @0
        encode.movz(.x1, 41, 0), // x1 = 41 (add's second arg)  @4
        encode.bl(0), // bl add (offset 0) - patched by the linker's CALL26 reloc   @8
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall number)              @12
        encode.svc(0), // svc #0 -> exit(x0 = add's result)                       @16
    };
    var text: [code.len * 4]u8 = undefined;
    for (code, 0..) |w, idx| std.mem.writeInt(u32, text[idx * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .size = text.len, .kind = .func, .defined = true },
        .{ .name = "add", .size = 0, .kind = .notype, .defined = false },
    };
    const relocs = [_]object.Reloc{.{ .offset = 8, .symbol = 1, .type = .call26 }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// Locate a real glibc `ld-linux-aarch64.so.1` (the interpreter to embed in a dynamic
/// exe via `--dynamic-linker`). Returns an allocated absolute path, or `null` if none is
/// found (the E2E test then skips cleanly rather than failing on an environment gap).
///
/// Ask the C compiler for the target's loader instead of scanning `/nix/store`: fast,
/// cross-platform, and a target's test runs only where its toolchain exists. Native `cc`
/// answers for the host arch; a cross target needs its `<triple>-gcc`, absent here -> the
/// query fails or echoes the bare name -> skip. `cc -print-file-name=<name>` returns an
/// ABSOLUTE path (leading `/`) when the compiler has that glibc file, else it echoes the
/// bare `<name>` (no leading `/`), which means "not available" -> skip.
fn findGlibcInterpAArch64(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const host = @import("builtin").cpu.arch;
    const ccs: []const []const u8 = if (host == .aarch64)
        &.{ "cc", "gcc" }
    else
        &.{ "aarch64-unknown-linux-gnu-gcc", "aarch64-linux-gnu-gcc" };
    const arg = try std.fmt.allocPrint(allocator, "-print-file-name={s}", .{"ld-linux-aarch64.so.1"});
    defer allocator.free(arg);
    for (ccs) |ccname| {
        const proc = std.process.run(allocator, io, .{ .argv = &.{ ccname, arg } }) catch continue;
        defer allocator.free(proc.stdout);
        defer allocator.free(proc.stderr);
        const t = std.mem.trim(u8, proc.stdout, " \t\r\n");
        if (t.len > 0 and t[0] == '/') return try allocator.dupe(u8, t);
    }
    return null;
}

/// Read the `DT_NEEDED` soname list straight out of an ELF64 file's `.dynamic` section
/// (an `ET_EXEC` dynexe here, unlike `ld.readSharedExports` which only accepts `ET_DYN`)
/// - the SM10 P4d DT_NEEDED-derivation test's own ground truth, independent of the CLI
/// and linker code under test. Locates `PT_DYNAMIC`, walks its `Elf64_Dyn` array
/// collecting each `DT_NEEDED` value (a `DT_STRTAB`-relative string-table BYTE OFFSET,
/// not a VADDR - unlike `DT_SONAME` this is read straight off the tag, no `PT_LOAD`
/// translation needed for the value itself) and the `DT_STRTAB` VADDR, then translates
/// that VADDR to a file offset via the containing `PT_LOAD` (the same
/// `p_offset + (vaddr - p_vaddr)` rule `dynamic.zig`'s own `vaddrToOffset` uses, not
/// exported across the module boundary, so duplicated here as a small, deliberate
/// test-only helper). The returned names are slices INTO `bytes` (the caller must keep
/// `bytes` alive at least as long as the returned slice, and free the outer slice only).
fn readDtNeeded(allocator: std.mem.Allocator, bytes: []const u8) ![][]const u8 {
    const e_phoff = try ld.elf.rdInt(u64, bytes, 32);
    const e_phentsize = try ld.elf.rdInt(u16, bytes, 54);
    const e_phnum = try ld.elf.rdInt(u16, bytes, 56);

    var dyn_off: ?u64 = null;
    var dyn_size: u64 = 0;
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const ph = try ld.elf.tableOffset(e_phoff, i, e_phentsize);
        if (try ld.elf.rdInt(u32, bytes, ph) != ld.dynamic.PT_DYNAMIC) continue;
        dyn_off = try ld.elf.rdInt(u64, bytes, ph + 8); // p_offset
        dyn_size = try ld.elf.rdInt(u64, bytes, ph + 32); // p_filesz
    }
    const doff = dyn_off orelse return error.MalformedObject;

    var strtab_va: ?u64 = null;
    var needed_str_offs: std.ArrayList(u64) = .empty;
    defer needed_str_offs.deinit(allocator);
    const dyn_entsize: u64 = 16; // Elf64_Dyn: { d_tag: i64, d_val/d_ptr: u64 }
    const dyn_n = dyn_size / dyn_entsize;
    var n: u64 = 0;
    while (n < dyn_n) : (n += 1) {
        const e = try ld.elf.tableOffset(doff, n, dyn_entsize);
        const tag = try ld.elf.rdInt(i64, bytes, e);
        const val = try ld.elf.rdInt(u64, bytes, e + 8);
        if (tag == ld.dynamic.DT_NULL) break;
        if (tag == ld.dynamic.DT_STRTAB) strtab_va = val;
        if (tag == ld.dynamic.DT_NEEDED) try needed_str_offs.append(allocator, val);
    }
    const str_va = strtab_va orelse return error.MalformedObject;

    var str_file_off: ?u64 = null;
    i = 0;
    while (i < e_phnum) : (i += 1) {
        const ph = try ld.elf.tableOffset(e_phoff, i, e_phentsize);
        if (try ld.elf.rdInt(u32, bytes, ph) != ld.dynamic.PT_LOAD) continue;
        const p_offset = try ld.elf.rdInt(u64, bytes, ph + 8);
        const p_vaddr = try ld.elf.rdInt(u64, bytes, ph + 16);
        const p_filesz = try ld.elf.rdInt(u64, bytes, ph + 32);
        if (str_va < p_vaddr) continue;
        const rel = str_va - p_vaddr;
        if (rel >= p_filesz) continue;
        str_file_off = p_offset + rel;
    }
    const str_off = str_file_off orelse return error.MalformedObject;

    var names: std.ArrayList([]const u8) = .empty;
    errdefer names.deinit(allocator);
    for (needed_str_offs.items) |off| {
        const start_off = try ld.elf.tableOffset(str_off, off, 1);
        const start = std.math.cast(usize, start_off) orelse return error.MalformedObject;
        if (start > bytes.len) return error.MalformedObject;
        var end = start;
        while (end < bytes.len and bytes[end] != 0) : (end += 1) {}
        try names.append(allocator, bytes[start..end]);
    }
    return names.toOwnedSlice(allocator);
}

/// A minimal `ar` (common-format) archive: `!<arch>\n` magic, then one 60-byte header
/// per member (name blank-padded and `/`-terminated, decimal size, "`\n" terminator)
/// followed by the member's bytes, 2-byte aligned. Mirrors the archive builder in
/// `libs/vulcan-target/aarch64/tests/link_archive.zig`.
fn buildArchive(allocator: std.mem.Allocator, members: []const ld.Member) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, "!<arch>\n");
    for (members) |m| {
        var hdr: [60]u8 = undefined;
        @memset(&hdr, ' ');
        const name_field = try std.fmt.allocPrint(allocator, "{s}/", .{m.name});
        defer allocator.free(name_field);
        std.debug.assert(name_field.len <= 16);
        @memcpy(hdr[0..name_field.len], name_field);
        var size_buf: [10]u8 = undefined;
        const size_str = try std.fmt.bufPrint(&size_buf, "{d}", .{m.bytes.len});
        @memcpy(hdr[48..][0..size_str.len], size_str);
        hdr[58] = '`';
        hdr[59] = '\n';
        try buf.appendSlice(allocator, &hdr);
        try buf.appendSlice(allocator, m.bytes);
        if (m.bytes.len % 2 == 1) try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

/// Assemble a `_start` ELF relocatable object: `bl main; mov x8, #93; svc #0` - calls
/// whatever the linker resolves `main` to (an undefined, external symbol from this
/// object's point of view) and exits with its return value (already sitting in `x0`
/// per AAPCS64) via the `exit` syscall. The `bl` is a real `R_AARCH64_CALL26`
/// relocation, so this proves `run` links across separate objects rather than relying
/// on any in-process address patching. Mirrors `libs/vulcan-cc/tests/native.zig`'s
/// `startStubAArch64`.
fn startStubAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.aarch64.encode;
    const object = target.aarch64.object;

    const code = [_]u32{
        encode.bl(0), // bl main (offset 0) - patched by the linker's CALL26 reloc
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall number)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var text: [code.len * 4]u8 = undefined;
    for (code, 0..) |w, idx| std.mem.writeInt(u32, text[idx * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .size = text.len, .kind = .func, .defined = true },
        .{ .name = "main", .size = 0, .kind = .notype, .defined = false },
    };
    const relocs = [_]object.Reloc{.{ .offset = 0, .symbol = 1, .type = .call26 }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

test "ld.vulcan CLI: links main.o + libhelper.a + start.o into a runnable exe, natively runs to exit 42" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the produced AArch64 ELF directly
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const helper_o = try buildHelperObj(allocator);
    defer allocator.free(helper_o);
    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);
    const start_o = try startStubAArch64(allocator);
    defer allocator.free(start_o);

    const archive_bytes = try buildArchive(allocator, &.{.{ .name = "helper.o", .bytes = helper_o }});
    defer allocator.free(archive_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "main.o", .data = main_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "libhelper.a", .data = archive_bytes });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const main_path = try std.fmt.allocPrint(allocator, "{s}/main.o", .{tmpdir});
    defer allocator.free(main_path);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/out", .{tmpdir});
    defer allocator.free(out_path);

    const status = try run(allocator, io, testErrw(), &.{
        "-o", out_path, "-e", "_start", "-L", tmpdir, start_path, main_path, "-lhelper",
    });
    try std.testing.expectEqual(@as(u8, 0), status);

    const proc = try std.process.run(allocator, io, .{
        .argv = &.{"./out"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // helper() = 42
        else => return error.BackendFailed,
    }
}

test "ld.vulcan CLI: an undefined symbol (no library given) returns a non-zero status without crashing" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);
    const start_o = try startStubAArch64(allocator);
    defer allocator.free(start_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "main.o", .data = main_o });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const main_path = try std.fmt.allocPrint(allocator, "{s}/main.o", .{tmpdir});
    defer allocator.free(main_path);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/out", .{tmpdir});
    defer allocator.free(out_path);

    // No `-L`/`-l`: `helper` is never satisfied.
    const status = try run(allocator, io, testErrw(), &.{ "-o", out_path, "-e", "_start", start_path, main_path });
    try std.testing.expect(status != 0);
}

// ---------------------------------------------------------------------------------
// `-l` resolution preference (SM10 P4d review fix): a plain static-intent link (no
// `-shared`/`--dynamic-linker`/`-Bdynamic`) must default to `.a`-first and must never
// divert into the dynamic route just because a same-named `.so` also exists in a `-L`
// directory - there is no default dynamic interpreter to fall back on here, unlike a
// real `ld`, so a dynamic diversion on a plain link is always a hard failure, not a
// silent alternate path.

test "ld.vulcan CLI: a plain link (no -shared/--dynamic-linker) prefers a same-named .a over a colliding .so, no dynamic diversion" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the produced AArch64 ELF directly
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const helper_o = try buildHelperObj(allocator);
    defer allocator.free(helper_o);
    const main_o = try buildMainObj(allocator);
    defer allocator.free(main_o);
    const start_o = try startStubAArch64(allocator);
    defer allocator.free(start_o);

    const archive_bytes = try buildArchive(allocator, &.{.{ .name = "helper.o", .bytes = helper_o }});
    defer allocator.free(archive_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "main.o", .data = main_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "libhelper.a", .data = archive_bytes });
    // A same-named `.so` that would divert a plain static-intent link into (and then
    // fail in) the dynamic route if `-l` resolution were dynamic-first by default. It is
    // deliberately NOT a valid ELF: with the conservative `.a`-first default, this file
    // must never even be opened.
    try tmp.dir.writeFile(io, .{ .sub_path = "libhelper.so", .data = "not an elf - must never be read for a plain static link" });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const main_path = try std.fmt.allocPrint(allocator, "{s}/main.o", .{tmpdir});
    defer allocator.free(main_path);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/out", .{tmpdir});
    defer allocator.free(out_path);

    const status = try run(allocator, io, testErrw(), &.{
        "-o", out_path, "-e", "_start", "-L", tmpdir, start_path, main_path, "-lhelper",
    });
    try std.testing.expectEqual(@as(u8, 0), status);

    const proc = try std.process.run(allocator, io, .{
        .argv = &.{"./out"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // helper() = 42, static path taken
        else => return error.BackendFailed,
    }
}

test "resolveLib tries the preferred kind then the other kind WITHIN each -L dir, not every dir for one kind before the other" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // d1 has only `libx.a`; d2 has BOTH `libx.so` and `libx.a`. A per-directory-correct
    // search (try preferred, then the other kind, THEN move to the next dir) must return
    // d1's `.a` in both preference modes below - d1 already has a match (of some kind)
    // before d2 is even considered. The bug this guards against: searching every `-L`
    // dir for the preferred kind FIRST (across all dirs), only falling back to the other
    // kind afterward - which would wrongly return d2's `.so` when dynamic is preferred,
    // even though d1 (searched first) already had a usable `.a`.
    var tmp1 = std.testing.tmpDir(.{});
    defer tmp1.cleanup();
    var tmp2 = std.testing.tmpDir(.{});
    defer tmp2.cleanup();
    try tmp1.dir.writeFile(io, .{ .sub_path = "libx.a", .data = "A-in-d1" });
    try tmp2.dir.writeFile(io, .{ .sub_path = "libx.so", .data = "SO-in-d2" });
    try tmp2.dir.writeFile(io, .{ .sub_path = "libx.a", .data = "A-in-d2" });

    const d1 = try tmp1.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(d1);
    const d2 = try tmp2.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(d2);
    const dirs = [_][]const u8{ d1, d2 };

    // Static preference: `.a`-first anyway, d1's `.a` wins trivially.
    {
        const found = (try resolveLib(allocator, io, "x", &dirs, false)).?;
        defer allocator.free(found.bytes);
        try std.testing.expectEqual(LibKind.archive, found.kind);
        try std.testing.expectEqualStrings("A-in-d1", found.bytes);
    }

    // Dynamic preference: d1 has no `.so`, so the per-directory fallback to `.a` WITHIN
    // d1 must win before d2 (which DOES have a `.so`) is ever consulted.
    {
        const found = (try resolveLib(allocator, io, "x", &dirs, true)).?;
        defer allocator.free(found.bytes);
        try std.testing.expectEqual(LibKind.archive, found.kind);
        try std.testing.expectEqualStrings("A-in-d1", found.bytes);
    }
}

// ---------------------------------------------------------------------------------
// Dynamic linking (SM10 P4d Task 1): `-shared`, `--dynamic-linker`, `-soname`, and `.so`
// inputs routed through `ld.linkDynamic`. The E2E below drives the CLI twice (once to
// build a `.so`, once to link a dynexe against it) and then runs the produced binary
// through the REAL host glibc `ld.so` - not this repo's own linker resolving anything
// at runtime, an independent, off-the-shelf dynamic loader.

test "ld.vulcan CLI: -shared builds libadd.so, --dynamic-linker links a dynexe, the real host ld.so runs it to exit 42" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the produced AArch64 ELF directly
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const add_o = try buildAddObj(allocator);
    defer allocator.free(add_o);
    const start_o = try startStubCallAddAArch64(allocator);
    defer allocator.free(start_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "add.o", .data = add_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const add_path = try std.fmt.allocPrint(allocator, "{s}/add.o", .{tmpdir});
    defer allocator.free(add_path);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const libadd_path = try std.fmt.allocPrint(allocator, "{s}/libadd.so", .{tmpdir});
    defer allocator.free(libadd_path);
    const prog_path = try std.fmt.allocPrint(allocator, "{s}/prog", .{tmpdir});
    defer allocator.free(prog_path);

    // Step A: `-shared` builds `libadd.so`, exporting `add`.
    const shared_status = try run(allocator, io, testErrw(), &.{
        "-shared", add_path, "-o", libadd_path, "-soname", "libadd.so",
    });
    try std.testing.expectEqual(@as(u8, 0), shared_status);

    const so_bytes = try readFile(allocator, io, libadd_path);
    defer allocator.free(so_bytes);
    var exports = try ld.readSharedExports(allocator, so_bytes);
    defer exports.deinit(allocator);
    try std.testing.expect(exports.soname != null);
    try std.testing.expectEqualStrings("libadd.so", exports.soname.?);
    var found_add = false;
    for (exports.symbols) |s| if (std.mem.eql(u8, s, "add")) {
        found_add = true;
    };
    try std.testing.expect(found_add);

    // Step B: find the host glibc loader; skip cleanly if this environment has none.
    const ld_so_path = (try findGlibcInterpAArch64(allocator, io)) orelse return error.SkipZigTest;
    defer allocator.free(ld_so_path);

    const link_status = try run(allocator, io, testErrw(), &.{
        "--dynamic-linker", ld_so_path, "-L", tmpdir, "-ladd", "-e", "_start", "-o", prog_path, start_path,
    });
    try std.testing.expectEqual(@as(u8, 0), link_status);

    // Step C: the REAL host ld.so loads `prog` + `libadd.so`, binds `add`, runs
    // `_start` -> `add(1, 41)` -> exit 42.
    //
    // No `SkipZigTest` catch here: the loader's presence was ALREADY verified above by
    // `findGlibcInterpAArch64` (the only place absence of the environment's own dynamic
    // loader should skip this test). A `FileNotFound` surfacing from spawning `prog`
    // itself is `execve` failing to find the INTERPRETER named in `prog`'s own
    // `PT_INTERP` (a broken/missing/wrong interp path baked into the produced binary) -
    // that is a real linker regression and must FAIL the test, not silently skip it.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", tmpdir);

    try runExpectExit(allocator, io, .{
        .argv = &.{prog_path},
        .environ_map = &env,
    }, 42);
}

test "ld.vulcan CLI: a .exec dynamic link with no --dynamic-linker returns a clean non-zero status" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const add_o = try buildAddObj(allocator);
    defer allocator.free(add_o);
    const start_o = try startStubCallAddAArch64(allocator);
    defer allocator.free(start_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "add.o", .data = add_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const add_path = try std.fmt.allocPrint(allocator, "{s}/add.o", .{tmpdir});
    defer allocator.free(add_path);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const libadd_path = try std.fmt.allocPrint(allocator, "{s}/libadd.so", .{tmpdir});
    defer allocator.free(libadd_path);
    const prog_path = try std.fmt.allocPrint(allocator, "{s}/prog", .{tmpdir});
    defer allocator.free(prog_path);

    // Build a real `.so` first, so this is a genuine ".so input with no
    // --dynamic-linker" case, not just "the -shared flag is missing".
    const shared_status = try run(allocator, io, testErrw(), &.{ "-shared", add_path, "-o", libadd_path, "-soname", "libadd.so" });
    try std.testing.expectEqual(@as(u8, 0), shared_status);

    // A `.so` input alone (no `-shared`, no `--dynamic-linker`) still triggers the
    // dynamic route (a `.exec` dynamic link) - which then has no interpreter to embed.
    const status = try run(allocator, io, testErrw(), &.{
        "-L", tmpdir, "-ladd", "-e", "_start", "-o", prog_path, start_path,
    });
    try std.testing.expect(status != 0);
}

test "ld.vulcan CLI: DT_NEEDED includes a linked .so that is never CALLED, proving the CLI's own derivation (not linkDynamic's call-driven one)" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the produced AArch64 ELF directly
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const add_o = try buildAddObj(allocator);
    defer allocator.free(add_o);
    const unused_o = try buildUnusedObj(allocator);
    defer allocator.free(unused_o);
    // Calls `add` only - `unused` is never referenced by anything in this link unit, so
    // `ld.linkDynamic`'s own call-driven DT_NEEDED auto-derivation (from resolved import
    // sites) has nothing to derive `libunused.so`'s soname from. Only the CLI's own
    // ~20-line derivation loop (one DT_NEEDED per `.shared` input, `run`'s dynamic-mode
    // branch) can put it in the output.
    const start_o = try startStubCallAddAArch64(allocator);
    defer allocator.free(start_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "add.o", .data = add_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "unused.o", .data = unused_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const add_path = try std.fmt.allocPrint(allocator, "{s}/add.o", .{tmpdir});
    defer allocator.free(add_path);
    const unused_path = try std.fmt.allocPrint(allocator, "{s}/unused.o", .{tmpdir});
    defer allocator.free(unused_path);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const libadd_path = try std.fmt.allocPrint(allocator, "{s}/libadd.so", .{tmpdir});
    defer allocator.free(libadd_path);
    const libunused_path = try std.fmt.allocPrint(allocator, "{s}/libunused.so", .{tmpdir});
    defer allocator.free(libunused_path);
    const prog_path = try std.fmt.allocPrint(allocator, "{s}/prog", .{tmpdir});
    defer allocator.free(prog_path);

    // Build both `.so`s.
    const add_shared_status = try run(allocator, io, testErrw(), &.{ "-shared", add_path, "-o", libadd_path, "-soname", "libadd.so" });
    try std.testing.expectEqual(@as(u8, 0), add_shared_status);
    const unused_shared_status = try run(allocator, io, testErrw(), &.{ "-shared", unused_path, "-o", libunused_path, "-soname", "libunused.so" });
    try std.testing.expectEqual(@as(u8, 0), unused_shared_status);

    const ld_so_path = (try findGlibcInterpAArch64(allocator, io)) orelse return error.SkipZigTest;
    defer allocator.free(ld_so_path);

    // Link a dynexe against BOTH `.so`s - `-lunused` only ever contributes a `DT_NEEDED`
    // entry, no imports (nothing calls into it).
    const link_status = try run(allocator, io, testErrw(), &.{
        "--dynamic-linker", ld_so_path, "-L", tmpdir, "-ladd", "-lunused", "-e", "_start", "-o", prog_path, start_path,
    });
    try std.testing.expectEqual(@as(u8, 0), link_status);

    const prog_bytes = try readFile(allocator, io, prog_path);
    defer allocator.free(prog_bytes);
    const dt_needed = try readDtNeeded(allocator, prog_bytes);
    defer allocator.free(dt_needed);

    var found_add = false;
    var found_unused = false;
    for (dt_needed) |n| {
        if (std.mem.eql(u8, n, "libadd.so")) found_add = true;
        if (std.mem.eql(u8, n, "libunused.so")) found_unused = true;
    }
    try std.testing.expect(found_add);
    // The load-bearing assertion: without the CLI's own derivation loop, this would be
    // missing (linkDynamic's call-driven auto-derivation never sees `libunused.so`, since
    // nothing calls into it).
    try std.testing.expect(found_unused);

    // The produced dynexe still runs correctly under the REAL host `ld.so`: it eagerly
    // binds every `DT_NEEDED` entry at load time (including the never-called
    // `libunused.so`), so this also proves the extra dependency doesn't break loading.
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put("LD_LIBRARY_PATH", tmpdir);
    try runExpectExit(allocator, io, .{
        .argv = &.{prog_path},
        .environ_map = &env,
    }, 42);
}

// ---------------------------------------------------------------------------------
// `-T <script>`: script-driven layout (SM10 P3c Task 3). Mirrors
// `libs/vulcan-target/aarch64/tests/link_script.zig`'s script-A coverage, but driven
// through the CLI's own `run` rather than calling `ld.linkInputsScript` directly, so
// the `-T` flag parsing + diagnostic + `-e`-override wiring is what's under test.

/// `main() i32 { return *(&g) + 37; }` with a writable `.data` global `g = 5`, so a
/// successful link+run round-trips the script's VMA assignment for `.data` (not just
/// `.text`/relocations, as `buildMainObj`'s `helper()` call alone would).
fn buildScriptMainObj(allocator: std.mem.Allocator) ![]u8 {
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
    var main_fn = Function.init(allocator);
    defer main_fn.deinit();
    const t = try main_fn.types.intern(i32k);
    const ptr_t = try main_fn.types.intern(.ptr);
    const b = try main_fn.appendBlock();
    const g = try main_fn.appendGlobalAddr(b, ptr_t, "g");
    const gv = try main_fn.appendInst(b, t, .{ .load = .{ .ptr = g } });
    const c37 = try main_fn.appendInst(b, t, .{ .iconst = 37 });
    const sum = try main_fn.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = gv, .rhs = c37 } });
    main_fn.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    const g_bytes = [_]u8{ 5, 0, 0, 0 };
    const data = [_]target.native.ObjData{
        .{ .name = "g", .bytes = &g_bytes, .kind = .data, .size = g_bytes.len },
    };
    return target.native.writeObjectDataFor(allocator, .aarch64, &.{.{ .name = "main", .func = &main_fn }}, &data);
}

/// A representative GNU-ld-style script: `ENTRY(_start)`, a custom base
/// `. = 0x400000`, `ALIGN`, output sections gathering `*(.text*)`/`*(.rodata*)`/
/// `*(.data*)`/`*(.bss*)`, and `__bss_start`/`__bss_end` boundary symbols. Same shape
/// `libs/vulcan-target/aarch64/tests/link_script.zig` proves against `linkInputsScript`
/// directly.
const script_a_text =
    \\ENTRY(_start)
    \\SECTIONS {
    \\  . = 0x400000;
    \\  .text : { *(.text*) }
    \\  . = ALIGN(16);
    \\  .rodata : { *(.rodata*) }
    \\  .data : { *(.data*) }
    \\  __bss_start = .;
    \\  .bss : { *(.bss*) }
    \\  __bss_end = .;
    \\}
;

test "ld.vulcan CLI: -T links via a linker script, natively runs to exit 42" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest; // executes the produced AArch64 ELF directly
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const start_o = try startStubAArch64(allocator);
    defer allocator.free(start_o);
    const main_o = try buildScriptMainObj(allocator);
    defer allocator.free(main_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "main.o", .data = main_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "script.ld", .data = script_a_text });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const main_path = try std.fmt.allocPrint(allocator, "{s}/main.o", .{tmpdir});
    defer allocator.free(main_path);
    const script_path = try std.fmt.allocPrint(allocator, "{s}/script.ld", .{tmpdir});
    defer allocator.free(script_path);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/out", .{tmpdir});
    defer allocator.free(out_path);

    const status = try run(allocator, io, testErrw(), &.{
        "-T", script_path, "-o", out_path, start_path, main_path,
    });
    try std.testing.expectEqual(@as(u8, 0), status);

    const proc = try std.process.run(allocator, io, .{
        .argv = &.{"./out"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    switch (proc.term) {
        .exited => |code| try std.testing.expectEqual(@as(u8, 42), code), // *(&g) + 37 == 5 + 37
        else => return error.BackendFailed,
    }
}

test "ld.vulcan CLI: -T with a malformed script returns a non-zero status without crashing" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const start_o = try startStubAArch64(allocator);
    defer allocator.free(start_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });
    // Unterminated SECTIONS body - a real syntax error, not just an unresolved reference.
    try tmp.dir.writeFile(io, .{ .sub_path = "bad.ld", .data = "SECTIONS { .text : { *(.text* " });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const script_path = try std.fmt.allocPrint(allocator, "{s}/bad.ld", .{tmpdir});
    defer allocator.free(script_path);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/out", .{tmpdir});
    defer allocator.free(out_path);

    const status = try run(allocator, io, testErrw(), &.{ "-T", script_path, "-o", out_path, start_path });
    try std.testing.expect(status != 0);
}

test "ld.vulcan CLI: -T with -e overrides the script's ENTRY (smoke: links without crashing)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const start_o = try startStubAArch64(allocator);
    defer allocator.free(start_o);
    const main_o = try buildScriptMainObj(allocator);
    defer allocator.free(main_o);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "start.o", .data = start_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "main.o", .data = main_o });
    try tmp.dir.writeFile(io, .{ .sub_path = "script.ld", .data = script_a_text });

    const tmpdir = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(tmpdir);
    const start_path = try std.fmt.allocPrint(allocator, "{s}/start.o", .{tmpdir});
    defer allocator.free(start_path);
    const main_path = try std.fmt.allocPrint(allocator, "{s}/main.o", .{tmpdir});
    defer allocator.free(main_path);
    const script_path = try std.fmt.allocPrint(allocator, "{s}/script.ld", .{tmpdir});
    defer allocator.free(script_path);
    const out_path = try std.fmt.allocPrint(allocator, "{s}/out", .{tmpdir});
    defer allocator.free(out_path);

    // `-e main` overrides `ENTRY(_start)` from the script; `main` is a valid resolved
    // symbol (defined in main.o, placed by the script), so the link itself must still
    // succeed - not run natively (entering at `main` skips the `_start` exit-syscall
    // wrapper, so its behavior after `ret` is undefined and not meaningfully assertable).
    const status = try run(allocator, io, testErrw(), &.{
        "-T", script_path, "-e", "main", "-o", out_path, start_path, main_path,
    });
    try std.testing.expectEqual(@as(u8, 0), status);

    // An unresolved `-e` name is a clean error, not a crash.
    const bad_status = try run(allocator, io, testErrw(), &.{
        "-T", script_path, "-e", "no_such_symbol", "-o", out_path, start_path, main_path,
    });
    try std.testing.expect(bad_status != 0);
}
