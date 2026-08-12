//! vcc: the Vulcan C Compiler driver. It has a clang-compatible command-line surface.
//! It compiles C to Vulcan IR, and can dump the IR. Object, assembly, and link flags are
//! recognized. The driver rejects unimplemented flags with a clear "not yet" diagnostic,
//! so the flag surface stays stable for future work. The driver wires the preprocessor
//! into the compile flow: `-I`/`-D`/`-U` feed `preproc.Options`, `-E` runs preprocess-only
//! (`preprocessToText`) and prints the reconstructed text instead of compiling, a disk
//! `IncludeResolver` backs `#include`, and `-freproducible-compile` pins
//! `__DATE__`/`__TIME__` to the epoch instead of the wall clock. This makes two builds of
//! the same source byte-identical. System includes resolve by default:
//! `cc.fs_resolver.FsResolver` (it replaces the old disk-only resolver) serves this
//! compiler's own built-in `<stddef.h>`/`<stdarg.h>` first. `-isystem` is a real search
//! directory. The driver discovers the host or cross glibc dev tree automatically
//! (`-nostdinc` suppresses this). The driver seeds the per-arch GNU predefined-macro set
//! (`__GNUC__`, `__x86_64__`, ...). So `#include <stdio.h>` compiles with no `-I`.

const std = @import("std");
const cc = @import("vulcan-cc");
const target = @import("vulcan-target");
const link = @import("vulcan-link");
const mm = @import("vulcan-opt").microarch;
const preproc = cc.preproc;

/// One positional input, in command-line order. It is a bare path (a `.c`/`.i` source, or
/// an `.o`/`.a`/`.so` link input, classified by extension in `main`, see `classifyExt`), or
/// a `-l<name>` library reference. The driver resolves a library reference against `-L`
/// search directories only after it scans the full argument list. This matches
/// `ld.vulcan`'s own `InputSpec`: a `-l` can search against a `-L` given anywhere on the
/// line, while archive-pull order still matches the command line. A `.lib` spec also
/// carries the `-Bdynamic`/`-Bstatic` search preference in effect at the point it was
/// scanned, see `resolveLib` and the classification loop in `main`. That loop also folds
/// in whole-link dynamic intent (`-shared`/`--dynamic-linker`) before it resolves,
/// mirroring `ld.vulcan`'s own `InputSpec`.
const InputSpec = union(enum) {
    path: []const u8,
    lib: struct { name: []const u8, prefer_dynamic: bool },
};

const Options = struct {
    /// Every positional or `-l` input, in command-line order. `-c`/`-E` still require
    /// exactly one `.c`/`.i` entry. The LINK path (the default when neither is given)
    /// consumes all inputs in order.
    inputs: std.ArrayList(InputSpec) = .empty,
    /// `-L dir` / `-Ldir`: additional `lib<name>.a` search directories for `-l`.
    lib_dirs: std.ArrayList([]const u8) = .empty,
    /// `-T <script>`: link via a linker script instead of the default contiguous layout.
    /// See the LINK step in `main`.
    script_path: ?[]const u8 = null,
    output: ?[]const u8 = null,
    target_triple: ?[]const u8 = null,
    /// `-target <arch>` resolved to `link.Arch`, or `null` for the default (the host's
    /// own arch, see `hostLinkArch`). Both compilation (which backend emits the `.o`) and
    /// linking (`writeExecutable`'s `arch`) use this value.
    target_arch: ?link.Arch = null,
    emit_ir: bool = false,
    preprocess_only: bool = false,
    /// `-c`: compile to a relocatable ELF `.o` object instead of linking. See
    /// `target.native.writeObjectData`.
    compile_only: bool = false,
    /// `-I dir` / `-Idir`: additional directories in the `#include` search order.
    includes: std.ArrayList([]const u8) = .empty,
    /// `-isystem dir` / `-isystemdir`: additional SYSTEM `#include` search directories.
    /// This list stays separate from `-I`, even though `buildSystemDirs` currently folds
    /// both into one ordered `system_dirs` list. See that function's doc comment for why a
    /// combined list is a correct-enough first cut.
    isystem: std.ArrayList([]const u8) = .empty,
    /// `-D name` / `-Dname` / `-D name=value` / `-Dname=value`: predefined macros. The
    /// driver applies these before the source's own directives run. See
    /// `preproc.preseedMacros`.
    defines: std.ArrayList(preproc.Define) = .empty,
    /// `-U name` / `-Uname`: strips a `-D` or standard predefined macro.
    undefines: std.ArrayList([]const u8) = .empty,
    /// `-freproducible-compile`: pin `__DATE__`/`__TIME__` to the Unix epoch rather than the
    /// wall clock, so two compiles of identical source are byte-identical.
    reproducible: bool = false,
    /// `-shared`: emit a dynamic shared object (`link.DynMode.shared`) instead of a
    /// static executable. This is one of three dynamic-mode triggers. The other two are
    /// `--dynamic-linker` and any input that resolves to a `.so`. See the LINK step in
    /// `main`. `-shared` is mutually exclusive with `-c`.
    shared: bool = false,
    /// `--dynamic-linker <path>` / `=<path>`: the `PT_INTERP` path embedded in a dynamic
    /// EXECUTABLE (`link.DynOptions.interp`). It is required whenever the dynamic route
    /// produces a `.exec` image, not a `-shared` one. It is also a dynamic-mode trigger
    /// on its own.
    dynamic_linker: ?[]const u8 = null,
    /// `-soname <name>`: the `DT_SONAME` recorded in a `-shared` output
    /// (`link.DynOptions.soname`). Ignored for a dynamic executable.
    soname: ?[]const u8 = null,
    /// `-MD` / `-MMD`: emit a Make depfile alongside the compiled object. See the
    /// depfile-writing step in `main`'s `-c` branch.
    gen_depfile: bool = false,
    /// `-MF <file>` / `-MF<file>`: the depfile's own path. `null` means the default (the
    /// output's basename with `.d` in place of its extension). See `derivedDepfileName`.
    depfile: ?[]const u8 = null,
    /// `-MT <target>` / `-MQ <target>`: the depfile rule's target name. `null` means the
    /// default, the compiled object's own path.
    depfile_target: ?[]const u8 = null,
    /// `-nostdlib`: suppress both the C runtime startup objects and the default libraries
    /// at link time. Parsed here; the link step acts on it.
    no_stdlib: bool = false,
    /// `-nostartfiles`: suppress only the C runtime startup objects (`crt0` and related
    /// files), and keep the default libraries. Parsed here; the link step acts on it.
    no_startfiles: bool = false,
    /// `-nodefaultlibs`: suppress only the default libraries, and keep the C runtime
    /// startup objects. Parsed here; the link step acts on it.
    no_defaultlibs: bool = false,
    /// `-rdynamic`: export all symbols to the dynamic symbol table, so a `dlopen`ed
    /// module can resolve back into the executable. Parsed here; the link step acts on it.
    rdynamic: bool = false,
    /// `-pthread`: compile and link against the POSIX threads library. Parsed here; the
    /// link step acts on it, for example by linking `libpthread`.
    pthread: bool = false,
    /// `-nostdinc`: suppress the discovered DEFAULT system include directory (the host or
    /// cross glibc dev tree, see `findHostGlibcIncludeDir`), same as real gcc. This
    /// compiler's own built-in `<stddef.h>`/`<stdarg.h>` still resolve either way. Real
    /// gcc's own built-ins also survive `-nostdinc`; only the standard LIBRARY headers are
    /// suppressed. `-I`/`-isystem` are unaffected.
    no_stdinc: bool = false,
    /// `-B <prefix>` / `-B<prefix>`: extra directories the crt/libc autolink discovery
    /// searches FIRST for `crt1.o`, `libc.so.6`, and the dynamic linker. This search runs
    /// ahead of the `gcc -print-file-name` probe and the Nix-store glob. This flag is
    /// gcc-compatible; autotools sometimes passes `-B` to point at an alternate toolchain
    /// prefix.
    prefixes: std.ArrayList([]const u8) = .empty,
    /// `--sysroot <dir>` / `--sysroot=<dir>`: a root the autolink discovery searches under
    /// (`<dir>/lib`, `<dir>/usr/lib`) for the crt objects, `libc.so.6`, and the dynamic
    /// linker. This flag is gcc-compatible.
    sysroot: ?[]const u8 = null,
    /// `--version` / `-v` / `-dumpversion` / `-dumpmachine`: which autoconf-style version
    /// probe `main` should answer, bypassing the normal compile/link pipeline. `null`
    /// means none was requested, so the ordinary "no input file" check still applies.
    probe: ?Probe = null,
    /// `-mcpu=<name>` / `-mtune=<name>`: the raw value of the LAST such flag on the command
    /// line. `null` means neither flag was given. `resolveModel` reads it to pick the model.
    cpu_tune: ?[]const u8 = null,
};

/// Which autoconf-style version probe was requested. `main` answers these BEFORE running
/// the normal compile/link pipeline, since none of them take an input file. `AC_PROG_CC`
/// and similar macros run the compiler with only one of these flags.
const Probe = enum { version, dumpversion, dumpmachine };

/// The version string `--version` / `-dumpversion` report: a plain `MAJOR.MINOR.PATCH`
/// value. It does not pretend to be a particular GCC release.
const vcc_version = "15.0.0";

/// The GNU-style target triple `-dumpmachine` reports: the same arch names
/// `parseTargetArch` accepts, paired with a generic `unknown-linux-gnu` vendor, OS, and
/// ABI suffix. VCC only targets Linux ELF hosts today.
fn archTriple(arch: link.Arch) []const u8 {
    return switch (arch) {
        .aarch64 => "aarch64-unknown-linux-gnu",
        .x86_64 => "x86_64-unknown-linux-gnu",
        .riscv64 => "riscv64-unknown-linux-gnu",
        .x86 => "i686-unknown-linux-gnu",
    };
}

/// Classifies a positional path by its extension for the link step: `.c`/`.i` are
/// compiled, `.o` is a relocatable object, `.a` is a static archive, `.so` is a dynamic
/// shared object. Anything else is `.unknown`. The caller turns that into a clear
/// diagnostic rather than guessing.
const InputKind = enum { c, o, a, so, unknown };
fn classifyExt(path: []const u8) InputKind {
    if (std.mem.endsWith(u8, path, ".c") or std.mem.endsWith(u8, path, ".i")) return .c;
    if (std.mem.endsWith(u8, path, ".o")) return .o;
    if (std.mem.endsWith(u8, path, ".a")) return .a;
    if (std.mem.endsWith(u8, path, ".so")) return .so;
    return .unknown;
}

fn fail(comptime fmt: []const u8, args: anytype) error{Usage} {
    std.debug.print("vcc: error: " ++ fmt ++ "\n", args);
    return error.Usage;
}

/// Resolves a `-target` operand to `link.Arch`: a bare arch name (`aarch64`, `x86_64`,
/// `x86`, `riscv64`, plus the common aliases `arm64`, `amd64`, `i386`, `i686`), or a full
/// target triple (`x86_64-linux-gnu`, `riscv64-unknown-elf`, ...). Only the token before
/// the first `-` is inspected, matching clang's own triple convention. `null` means
/// unrecognized; the caller turns that into a clear diagnostic rather than silently
/// falling back to the host.
fn parseTargetArch(spec: []const u8) ?link.Arch {
    const arch_tok = if (std.mem.indexOfScalar(u8, spec, '-')) |dash| spec[0..dash] else spec;
    if (std.mem.eql(u8, arch_tok, "aarch64") or std.mem.eql(u8, arch_tok, "arm64")) return .aarch64;
    if (std.mem.eql(u8, arch_tok, "x86_64") or std.mem.eql(u8, arch_tok, "amd64")) return .x86_64;
    if (std.mem.eql(u8, arch_tok, "x86") or std.mem.eql(u8, arch_tok, "i386") or std.mem.eql(u8, arch_tok, "i686")) return .x86;
    if (std.mem.eql(u8, arch_tok, "riscv64")) return .riscv64;
    return null;
}

/// The default `-c` output name when `-o` is absent: `input`'s basename with its extension
/// (if any) replaced by `.o` (`foo.c` -> `foo.o`; `src/foo.c` -> `foo.o`, matching clang's
/// convention of dropping the input's directory too).
fn derivedOutputName(allocator: std.mem.Allocator, input: []const u8) std.mem.Allocator.Error![]const u8 {
    const base = std.fs.path.basename(input);
    const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
    return std.fmt.allocPrint(allocator, "{s}.o", .{stem});
}

/// The default depfile name when `-MF` is absent: `out`'s basename with its extension (if
/// any) replaced by `.d`. Mirrors `derivedOutputName`'s `.o` convention.
fn derivedDepfileName(allocator: std.mem.Allocator, out: []const u8) std.mem.Allocator.Error![]const u8 {
    const base = std.fs.path.basename(out);
    const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
    return std.fmt.allocPrint(allocator, "{s}.d", .{stem});
}

/// Splits a `-D`/`-Dname=value` operand on its FIRST `=` into a `preproc.Define`: no `=`
/// means a valueless `-D name` (`preseedMacros` gives it the body `1`); `name=value` gives it
/// `value` (scanned as real tokens - see `preproc.defineFromScannedValue`).
fn appendDefine(allocator: std.mem.Allocator, defines: *std.ArrayList(preproc.Define), spec: []const u8) std.mem.Allocator.Error!void {
    if (std.mem.indexOfScalar(u8, spec, '=')) |eq| {
        try defines.append(allocator, .{ .name = spec[0..eq], .value = spec[eq + 1 ..] });
    } else {
        try defines.append(allocator, .{ .name = spec, .value = "" });
    }
}

fn parseArgs(allocator: std.mem.Allocator, it: anytype) (error{Usage} || std.mem.Allocator.Error)!Options {
    var opts = Options{};
    // `-Bdynamic`/`-Bstatic` only steer `-l` search preference. The parser captures this
    // onto each `.lib` spec as it scans, matching `ld.vulcan`'s own scan-time state (see
    // that file's `prefer_dynamic_pref`). The default is STATIC-first (`.a` before `.so`),
    // the same conservative choice `ld.vulcan` makes. `resolveLib` in `main` also folds in
    // whole-link dynamic intent (`-shared`/`--dynamic-linker`), known only once the parser
    // has scanned the whole command line.
    var prefer_dynamic_pref = false;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-o")) {
            opts.output = it.next() orelse return fail("-o requires an argument", .{});
        } else if (std.mem.eql(u8, arg, "-target")) {
            const spec = it.next() orelse return fail("-target requires an argument", .{});
            opts.target_triple = spec;
            opts.target_arch = parseTargetArch(spec) orelse
                return fail("unknown -target architecture '{s}' (expected aarch64/x86_64/x86/riscv64, or a triple starting with one)", .{spec});
        } else if (std.mem.eql(u8, arg, "--emit-ir")) {
            opts.emit_ir = true;
        } else if (std.mem.eql(u8, arg, "-E")) {
            opts.preprocess_only = true;
        } else if (std.mem.eql(u8, arg, "-freproducible-compile")) {
            opts.reproducible = true;
        } else if (std.mem.eql(u8, arg, "-I")) {
            const dir = it.next() orelse return fail("-I requires an argument", .{});
            try opts.includes.append(allocator, dir);
        } else if (std.mem.startsWith(u8, arg, "-I")) {
            try opts.includes.append(allocator, arg[2..]);
        } else if (std.mem.eql(u8, arg, "-isystem")) {
            // `-isystem <dir>` was once an accept-and-ignore flag. It is now a real search
            // directory. See `buildSystemDirs`.
            const dir = it.next() orelse return fail("-isystem requires an argument", .{});
            try opts.isystem.append(allocator, dir);
        } else if (std.mem.startsWith(u8, arg, "-isystem") and arg.len > "-isystem".len) {
            try opts.isystem.append(allocator, arg["-isystem".len..]);
        } else if (std.mem.eql(u8, arg, "-nostdinc")) {
            opts.no_stdinc = true;
        } else if (std.mem.eql(u8, arg, "-D")) {
            const spec = it.next() orelse return fail("-D requires an argument", .{});
            try appendDefine(allocator, &opts.defines, spec);
        } else if (std.mem.startsWith(u8, arg, "-D")) {
            try appendDefine(allocator, &opts.defines, arg[2..]);
        } else if (std.mem.eql(u8, arg, "-U")) {
            const name = it.next() orelse return fail("-U requires an argument", .{});
            try opts.undefines.append(allocator, name);
        } else if (std.mem.startsWith(u8, arg, "-U")) {
            try opts.undefines.append(allocator, arg[2..]);
        } else if (std.mem.eql(u8, arg, "-c")) {
            opts.compile_only = true;
        } else if (std.mem.eql(u8, arg, "-shared")) {
            opts.shared = true;
        } else if (std.mem.eql(u8, arg, "--dynamic-linker")) {
            opts.dynamic_linker = it.next() orelse return fail("--dynamic-linker requires an argument", .{});
        } else if (std.mem.startsWith(u8, arg, "--dynamic-linker=")) {
            opts.dynamic_linker = arg["--dynamic-linker=".len..];
        } else if (std.mem.eql(u8, arg, "-soname")) {
            opts.soname = it.next() orelse return fail("-soname requires an argument", .{});
        } else if (std.mem.eql(u8, arg, "-Bdynamic")) {
            prefer_dynamic_pref = true;
        } else if (std.mem.eql(u8, arg, "-Bstatic") or std.mem.eql(u8, arg, "--static")) {
            prefer_dynamic_pref = false;
        } else if (std.mem.eql(u8, arg, "-B")) {
            // `-B <prefix>` / `-B<prefix>`: a toolchain search prefix for the crt/libc
            // autolink discovery. This arm sits AFTER the `-Bdynamic`/`-Bstatic` arms above,
            // so those exact spellings still steer `-l` search preference. They do not
            // become a prefix named "dynamic" or "static".
            const p = it.next() orelse return fail("-B requires an argument", .{});
            try opts.prefixes.append(allocator, p);
        } else if (std.mem.startsWith(u8, arg, "-B") and arg.len > 2) {
            try opts.prefixes.append(allocator, arg[2..]);
        } else if (std.mem.eql(u8, arg, "--sysroot")) {
            opts.sysroot = it.next() orelse return fail("--sysroot requires an argument", .{});
        } else if (std.mem.startsWith(u8, arg, "--sysroot=")) {
            opts.sysroot = arg["--sysroot=".len..];
        } else if (std.mem.eql(u8, arg, "-L")) {
            const dir = it.next() orelse return fail("-L requires an argument", .{});
            try opts.lib_dirs.append(allocator, dir);
        } else if (std.mem.startsWith(u8, arg, "-L") and arg.len > 2) {
            try opts.lib_dirs.append(allocator, arg[2..]);
        } else if (std.mem.eql(u8, arg, "-T")) {
            opts.script_path = it.next() orelse return fail("-T requires an argument", .{});
        } else if (std.mem.startsWith(u8, arg, "-T") and arg.len > 2) {
            opts.script_path = arg[2..];
        } else if (std.mem.eql(u8, arg, "-l")) {
            const name = it.next() orelse return fail("-l requires an argument", .{});
            try opts.inputs.append(allocator, .{ .lib = .{ .name = name, .prefer_dynamic = prefer_dynamic_pref } });
        } else if (std.mem.startsWith(u8, arg, "-l") and arg.len > 2) {
            try opts.inputs.append(allocator, .{ .lib = .{ .name = arg[2..], .prefer_dynamic = prefer_dynamic_pref } });
        } else if (std.mem.eql(u8, arg, "-MD") or std.mem.eql(u8, arg, "-MMD")) {
            // `-MD`/`-MMD`: ask for a Make depfile alongside the object. VCC does not yet
            // separate system headers from user headers, the real `-MMD` difference, so
            // both flags act the same way here.
            opts.gen_depfile = true;
        } else if (std.mem.eql(u8, arg, "-MF")) {
            opts.depfile = it.next() orelse return fail("-MF requires an argument", .{});
        } else if (std.mem.startsWith(u8, arg, "-MF") and arg.len > 3) {
            opts.depfile = arg[3..];
        } else if (std.mem.eql(u8, arg, "-MT") or std.mem.eql(u8, arg, "-MQ")) {
            opts.depfile_target = it.next() orelse return fail("'{s}' requires an argument", .{arg});
        } else if ((std.mem.startsWith(u8, arg, "-MT") or std.mem.startsWith(u8, arg, "-MQ")) and arg.len > 3) {
            opts.depfile_target = arg[3..];
        } else if (std.mem.eql(u8, arg, "-MP") or std.mem.eql(u8, arg, "-MG") or
            std.mem.eql(u8, arg, "-M") or std.mem.eql(u8, arg, "-MM"))
        {
            // `-MP`/`-MG`/`-M`/`-MM`: accepted, no effect yet. A real `-M`/`-MM` prints
            // dependency info instead of compiling; VCC still compiles. This is a minimal
            // accept, not the full behavior, and is tracked as a follow-up.
        } else if (std.mem.eql(u8, arg, "-nostdlib")) {
            opts.no_stdlib = true;
        } else if (std.mem.eql(u8, arg, "-nostartfiles")) {
            opts.no_startfiles = true;
        } else if (std.mem.eql(u8, arg, "-nodefaultlibs")) {
            opts.no_defaultlibs = true;
        } else if (std.mem.eql(u8, arg, "-rdynamic")) {
            opts.rdynamic = true;
        } else if (std.mem.eql(u8, arg, "-pthread")) {
            opts.pthread = true;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            opts.probe = .version;
        } else if (std.mem.eql(u8, arg, "-dumpversion")) {
            opts.probe = .dumpversion;
        } else if (std.mem.eql(u8, arg, "-dumpmachine")) {
            opts.probe = .dumpmachine;
        } else if (std.mem.eql(u8, arg, "-S")) {
            // `-S` asks for a textual assembly output, a codegen mode VCC does not have.
            // Ignoring it would silently produce the wrong output type (see the
            // `-m32`/`-m64`/`-mabi=` note below), so it stays a hard, fail-closed error
            // rather than joining the accept-ignore families.
            return fail("'{s}' is recognized but not implemented yet (needs the object/link phase)", .{arg});
        } else if (std.mem.eql(u8, arg, "-m32") or std.mem.eql(u8, arg, "-m64") or
            std.mem.startsWith(u8, arg, "-mabi="))
        {
            // DENYLIST: these flags change the ABI or word size. VCC's ABI is fixed by
            // `-target`. Silently dropping one of these risks a silent miscompile, so the
            // driver fails closed with a clear message instead of joining the
            // accept-ignore families below.
            return fail("'{s}' changes the ABI/word size and is not supported (use -target instead)", .{arg});
        } else if (std.mem.startsWith(u8, arg, "-mcpu=")) {
            opts.cpu_tune = arg["-mcpu=".len..]; // last wins
        } else if (std.mem.startsWith(u8, arg, "-mtune=")) {
            opts.cpu_tune = arg["-mtune=".len..];
        } else if (std.mem.startsWith(u8, arg, "-march=")) {
            // ISA selection. VCC's ISA is fixed by -target, so accept and ignore.
        } else if (std.mem.eql(u8, arg, "-Xlinker") or std.mem.eql(u8, arg, "-Xassembler")) {
            // `-Xlinker <arg>` / `-Xassembler <arg>`: accepted, along with the one
            // argument each carries, but not yet forwarded anywhere. There is no separate
            // linker or assembler subprocess here to forward them to.
            _ = it.next() orelse return fail("'{s}' requires an argument", .{arg});
        } else if (flagTakesSeparateArg(arg)) {
            // ACCEPT-AND-IGNORE a gcc flag that carries its value as the NEXT token:
            // `-include foo.h`, `-idirafter dir`, `-iquote dir`, `-imacros foo.h`, `-x c`,
            // and similar flags (`-isystem` has its own real arm above). Consume BOTH the
            // flag and its argument, so the argument is not mistaken for a positional
            // input file. Misreading it that way would fail the `./configure`/gnulib
            // builds that pass these flags.
            _ = it.next() orelse return fail("'{s}' requires an argument", .{arg});
        } else if (isIgnoredFlag(arg)) {
            // ACCEPT-AND-IGNORE: a warning, optimization, debug, or standard-selection
            // flag that `./configure`-style builds throw at the compiler. VCC has one
            // optimization level and does not implement most gcc warnings, so these are
            // recognized and dropped rather than failing the whole build.
        } else if (std.mem.startsWith(u8, arg, "-")) {
            // FALLBACK: an unrecognized flag outside every family above. `./configure`
            // probes many gcc flags VCC does not know about. Erroring here would fail the
            // whole `./configure` run over one unimplemented flag, so the safer default is
            // to accept and ignore it. A flag that would silently miscompile belongs in
            // the denylist above, not here.
        } else {
            try opts.inputs.append(allocator, .{ .path = arg });
        }
    }
    if (opts.probe == null and opts.inputs.items.len == 0) return fail("no input file", .{});
    return opts;
}

/// Whether `arg` is a warning, optimization, debug, or standard-selection flag VCC
/// recognizes but does not act on. `./configure`/`make` throw the full gcc flag surface at
/// the compiler. Most of it (which warnings to print, which optimization level, how much
/// debug info) has no effect on VCC's own single-level pipeline.
fn isIgnoredFlag(arg: []const u8) bool {
    if (std.mem.startsWith(u8, arg, "-std=")) return true; // -std=gnu11, -std=c99, ...
    if (std.mem.startsWith(u8, arg, "-O")) return true; // -O, -O0.. -Ofast, -Os, -Og
    if (std.mem.startsWith(u8, arg, "-g")) return true; // -g, -g0.., -ggdb, -gdwarf-*, ...
    if (std.mem.eql(u8, arg, "-w")) return true; // suppress all warnings
    if (std.mem.startsWith(u8, arg, "-W")) return true; // -Wall, -Wl,..., -Wp,..., -Wa,...
    if (std.mem.eql(u8, arg, "-pedantic") or std.mem.eql(u8, arg, "-pedantic-errors")) return true;
    if (std.mem.eql(u8, arg, "-pipe")) return true;
    if (std.mem.startsWith(u8, arg, "-f")) return true; // -fPIC, -fno-builtin, ... (denylist wins first)
    return false;
}

/// Whether `arg` is a gcc flag that carries its value as the SEPARATE NEXT token, so the
/// driver must consume that token too rather than let it be read as a positional input.
/// These are the common include, preprocessor, and language flags a `./configure`/gnulib
/// build passes. `-isystem` is NOT here; it is a real search directory, see `parseArgs`'s
/// own `-isystem` arm, ahead of this fallback.
fn flagTakesSeparateArg(arg: []const u8) bool {
    const with_arg = [_][]const u8{
        "-include",  "-imacros", "-idirafter",     "-iquote",
        "-isysroot", "-iprefix", "-iwithprefix",   "-iwithprefixbefore",
        "-MJ",       "-x",       "-Xpreprocessor", "-aux-info",
        "-dumpbase", "-dumpdir", "--param",
    };
    for (with_arg) |f| if (std.mem.eql(u8, arg, f)) return true;
    return false;
}

/// Maps the driver's own `link.Arch` to `cc.layout.Arch`. Both enums use the same four
/// names but NOT the same declaration order (`elf.zig`'s `Arch` is `riscv64, aarch64,
/// x86_64, x86`; `layout.zig`'s is `aarch64, riscv64, x86_64, x86`). A plain
/// `@enumFromInt` cast would silently pick the wrong arch, so this function switches on
/// the name explicitly.
fn ccLayoutArch(arch: link.Arch) cc.layout.Arch {
    return switch (arch) {
        .aarch64 => .aarch64,
        .x86_64 => .x86_64,
        .riscv64 => .riscv64,
        .x86 => .x86,
    };
}

/// The per-target GNU predefined-macro set a real system header's `__GNUC_PREREQ`/`#ifdef
/// __x86_64__`-style gates expect, so `#include <stdio.h>` takes the same GNU-extension
/// branches a real gcc at this posture would. `long_bits`/`ptr_bits`/`char_signed` come
/// from `arch`'s own `TargetLayout`, so `__SIZEOF_*__` and the LP64-vs-ILP32 type macros
/// match the compile's ACTUAL target, not the build host's.
///
/// `__GNUC__` is 4.6: earlier testing proved 4.2 sufficient for the glibc header chain, but
/// real gnulib code gates its `verify`/static-assert form on `__GNUC__` 4.6 or later. Below
/// that level it falls back to an unused function-returning-pointer-to-array declarator.
/// 4.6 makes gnulib emit the plain `_Static_assert` VCC now understands. It is also honest
/// about VCC's real level: VCC already accepts `_Generic`/`_Alignof`/`_Noreturn`/`typeof`,
/// all newer than gcc 4.2.
fn systemPredefFor(arch: link.Arch) preproc.SystemPredef {
    const cc_arch = ccLayoutArch(arch);
    const lay = cc.layout.forArch(cc_arch);
    return .{
        .arch = cc_arch,
        .gnuc_major = 4,
        .gnuc_minor = 6,
        .gnuc_patch = 0,
        .long_bits = lay.long_bits,
        .ptr_bits = lay.ptr_bits,
        .char_signed = lay.char_signed,
    };
}

/// Builds the ordered directory list `cc.fs_resolver.FsResolver` searches for a `<...>`
/// include. It is also the fallback list for a `"..."` include not found in the including
/// file's own directory. The order is `-I` dirs FIRST, then `-isystem` dirs, then
/// `default_dir` (the discovered host or cross glibc dev tree, or `null` when none was
/// found or `-nostdinc` suppressed it, see `main`). Real gcc keeps `-I` and `-isystem` in
/// TWO separate lists: a `-I` dir is also searched for a QUOTED include ahead of
/// `-isystem`, and `-isystem` sits strictly between `-I` and the compiler's own standard
/// system dirs. `FsResolver` has only ONE `system_dirs` list, so folding both into one
/// ordered list here is a correct-enough first cut. A `-I` dir still wins over an
/// `-isystem` one, matching gcc's own precedence, just without the quoted-vs-angle
/// distinction gcc draws between the two families. An `-isystem` dir also gets searched
/// for a QUOTED include here, where real gcc would not.
fn buildSystemDirs(allocator: std.mem.Allocator, opts: *const Options, default_dir: ?[]const u8) std.mem.Allocator.Error![]const []const u8 {
    var dirs: std.ArrayList([]const u8) = .empty;
    try dirs.appendSlice(allocator, opts.includes.items);
    try dirs.appendSlice(allocator, opts.isystem.items);
    if (default_dir) |d| try dirs.append(allocator, d);
    return dirs.items;
}

/// Ensures `path` has a directory component before it becomes `preproc.Options.filename`.
/// `preproc.preprocess` derives a `#include "..."` search's OWN starting directory from
/// `std.fs.path.dirname(opts.filename)`. A bare filename with no `/` (`foo.c`) makes that
/// `null`, which skips the "search the including file's own directory" step entirely
/// (`fs_resolver.zig`'s `FsResolver.resolveFn` only consults `includer_dir` when it is
/// non-null). So a same-directory `#include "local.h"` next to a bare-named source would
/// stop resolving. This is a regression against the old `diskResolve`, which defaulted a
/// `null` includer_dir to `.` itself. Prefixing with `./` fixes this without changing WHICH
/// file is read. The only visible effect is `__FILE__`'s exact spelling (`./foo.c` instead
/// of `foo.c`) for a bare-named source, a cosmetic difference nothing here tests for.
fn filenameForPreproc(allocator: std.mem.Allocator, path: []const u8) std.mem.Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, path, '/') != null) return path;
    return std.fmt.allocPrint(allocator, "./{s}", .{path});
}

/// Runs `sh -c script`, returning its trimmed stdout, or `null` if the process failed to
/// spawn or produced nothing. It NEVER returns an error, so system-include discovery
/// degrades to "no default system dir" instead of aborting the whole compile. Mirrors
/// `preproc_glibc.zig`'s own `shOutput`, generalized off the real `io` instead of
/// `std.testing.io`.
fn shOutput(allocator: std.mem.Allocator, io: std.Io, script: []const u8) !?[]u8 {
    const proc = std.process.run(allocator, io, .{ .argv = &.{ "sh", "-c", script } }) catch return null;
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    const trimmed = std.mem.trim(u8, proc.stdout, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

/// Locates the HOST glibc "dev" include directory, the one holding a real `stdio.h`,
/// `features.h`, `sys/cdefs.h`, and similar files, so `#include <stdio.h>` resolves with no
/// explicit `-I`/`-isystem`. The function tries candidates in order, and validates each one
/// by checking for a real `stdio.h` inside it before trusting it: (1) `VCC_GLIBC_INCLUDE`,
/// an explicit override for a host where the search below doesn't apply; (2) a real
/// `gcc`'s own `-E -v` system-include search list, whose LAST entry is always the glibc dev
/// tree gcc itself was built against; (3) any `/nix/store/*-glibc-*-dev/include` path. Step
/// 3 skips the arch-suffixed CROSS trees, see `findCrossGlibcIncludeDir`. Mirrors
/// `preproc_glibc.zig`'s own discovery, minus that test's hardcoded last-resort store path.
/// That path is already covered by step 3's glob; a store hash belongs in a test's
/// fallback, not production driver code, where it would only rot. This function never
/// fails: `null` means no default system dir is added, so `#include <stdio.h>` then fails
/// later, AT the `#include` itself, with a clear "not found" error (`fs_resolver.zig`'s
/// contract), not a hard driver error.
fn findHostGlibcIncludeDir(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const script =
        \\if [ -n "${VCC_GLIBC_INCLUDE:-}" ] && [ -f "$VCC_GLIBC_INCLUDE/stdio.h" ]; then
        \\  echo "$VCC_GLIBC_INCLUDE"; exit 0
        \\fi
        \\if command -v gcc >/dev/null 2>&1; then
        \\  d=$(echo | gcc -E -v -xc - 2>&1 | grep -E '^ .*/glibc-[0-9][^/]*-dev/include$' | tail -1 | sed -e 's/^ //')
        \\  if [ -n "$d" ] && [ -f "$d/stdio.h" ]; then echo "$d"; exit 0; fi
        \\fi
        \\for d in /nix/store/*-glibc-[0-9]*-dev/include; do
        \\  if [ -f "$d/stdio.h" ]; then echo "$d"; exit 0; fi
        \\done
        \\exit 1
    ;
    return shOutput(allocator, io, script);
}

/// Locates a CROSS glibc dev include dir for `triple` (for example `riscv64-unknown-linux-
/// gnu` from `archTriple`), for a `-target` that isn't the host's own arch. It skips when
/// absent: this Nix profile may simply not have a given arch's cross glibc installed, which
/// is not a VCC bug. Mirrors `findHostGlibcIncludeDir`'s "not found means null, never fail"
/// contract.
fn findCrossGlibcIncludeDir(allocator: std.mem.Allocator, io: std.Io, triple: []const u8) !?[]u8 {
    const script = try std.fmt.allocPrint(allocator,
        \\for d in /nix/store/*-glibc-{s}-*-dev/include; do
        \\  if [ -f "$d/stdio.h" ]; then echo "$d"; exit 0; fi
        \\done
        \\exit 1
    , .{triple});
    defer allocator.free(script);
    return shOutput(allocator, io, script);
}

/// The per-arch dynamic-linker soname the autolink embeds as `PT_INTERP`, so a linked
/// executable finds its loader with no explicit `--dynamic-linker`. These are the
/// canonical glibc loader names, one per target. `external_linkage.zig`/`variadic.zig`
/// embed the very same strings when they link against the real glibc `ld.so`.
fn interpSoname(arch: link.Arch) []const u8 {
    return switch (arch) {
        .aarch64 => "ld-linux-aarch64.so.1",
        .x86_64 => "ld-linux-x86-64.so.2",
        .riscv64 => "ld-linux-riscv64-lp64d.so.1",
        .x86 => "ld-linux.so.2",
    };
}

/// The host C runtime the default-executable autolink discovers: absolute paths to `crt1.o`
/// (the ELF entry `_start`, which calls `__libc_start_main(main, ...)`), `libc.so.6` (the
/// shared C library), and the per-arch dynamic linker (`PT_INTERP`). VCC's `_start` image
/// is a NON-PIE `ET_EXEC` (see `link.linkDynamic`'s `.exec` mode at base `0x400000`), so the
/// autolink uses the NON-S crt set. A plain printf hello needs only `crt1.o` here. A modern
/// aarch64 `crt1.o` passes NULL `init`/`fini` to `__libc_start_main`, so `crti.o`/`crtn.o`'s
/// `_init`/`_fini` framing is not required, and `crtbegin.o`/`crtend.o`'s `.init_array`/frame
/// registration is not needed by a program with no C++ static constructors. `crti.o`,
/// `crtn.o`, and `crtbegin.o` also pull in relocations and weak-undefined symbols VCC's
/// from-scratch linker does not yet model (`__gmon_start__`,
/// `R_AARCH64_LDST8_ABS_LO12_NC`), so the minimal set is both sufficient and the only one
/// that links today.
const Toolchain = struct {
    crt1: []const u8,
    libc: []const u8,
    interp: []const u8,
    /// The glibc `libc_nonshared.a` beside `libc.so.6`. The real `-lc` is a linker script
    /// (`GROUP ( libc.so.6 libc_nonshared.a ... )`). A handful of symbols (`atexit`, `stat`,
    /// `__stack_chk_fail_local`, ...) live ONLY in this static archive, not the shared
    /// object. VCC links `libc.so.6` directly, so it appends this archive too. This field
    /// is optional: `null` when the libdir has no such archive, for example a non-glibc
    /// libc, in which case the autolink skips it.
    nonshared: []const u8,
};

/// Discovers the host C runtime for a default-executable autolink, or `null` when the
/// essential set (`crt1.o`, `libc.so.6`, and the arch's dynamic linker, all in one
/// directory) is not found. The caller then errors CLEARLY rather than emitting an
/// unrunnable binary. The search is LAYERED, and it accepts each candidate directory only
/// when it holds all three files: (1) `-B` prefixes and `--sysroot`/`$VCC_SYSROOT`
/// (`<root>/lib`, `<root>/usr/lib`) and an explicit `$VCC_CRT_DIR`; (2) the directory of
/// `gcc -print-file-name=crt1.o` (the host glibc lib dir, which on this Nix host holds
/// `crt1.o`, `libc.so.6`, and `ld-linux-*.so.*` together); (3) any
/// `/nix/store/*-glibc-*/lib`. The per-arch loader soname (`interpSoname`) gates each
/// directory, so a host glibc dir is only accepted for the arch whose loader it actually
/// contains. A cross `-target` simply finds nothing here, and the caller reports it; cross
/// autolink is best-effort. This function never fails: a probe that cannot run degrades to
/// `null`, mirroring `findHostGlibcIncludeDir`'s contract.
fn discoverToolchain(allocator: std.mem.Allocator, io: std.Io, arch: link.Arch, opts: *const Options) !?Toolchain {
    const soname = interpSoname(arch);

    // The `-B`/`--sysroot`-derived candidate directories, space-joined into the `for` loop
    // below (Nix store paths never contain spaces, so shell word-splitting is safe here).
    var extra: std.ArrayList(u8) = .empty;
    for (opts.prefixes.items) |p| {
        try extra.appendSlice(allocator, p);
        try extra.append(allocator, ' ');
    }
    if (opts.sysroot) |s| {
        const roots = try std.fmt.allocPrint(allocator, "{s}/lib {s}/usr/lib ", .{ s, s });
        try extra.appendSlice(allocator, roots);
    }

    var script: std.ArrayList(u8) = .empty;
    try script.appendSlice(allocator, "soname=");
    try script.appendSlice(allocator, soname);
    try script.appendSlice(allocator,
        \\
        \\try() {
        \\  if [ -n "$1" ] && [ -f "$1/crt1.o" ] && [ -f "$1/libc.so.6" ] && [ -f "$1/$soname" ]; then
        \\    echo "$1"; exit 0
        \\  fi
        \\}
        \\for d in $VCC_CRT_DIR
    );
    try script.appendSlice(allocator, extra.items);
    try script.appendSlice(allocator,
        \\; do try "$d"; done
        \\if [ -n "$VCC_SYSROOT" ]; then try "$VCC_SYSROOT/lib"; try "$VCC_SYSROOT/usr/lib"; fi
        \\if command -v gcc >/dev/null 2>&1; then
        \\  c=$(gcc -print-file-name=crt1.o 2>/dev/null)
        \\  case "$c" in /*) try "$(dirname "$c")";; esac
        \\fi
        \\for d in /nix/store/*-glibc-[0-9]*/lib; do try "$d"; done
        \\exit 1
    );

    const libdir = (try shOutput(allocator, io, script.items)) orelse return null;
    return .{
        .crt1 = try std.fs.path.join(allocator, &.{ libdir, "crt1.o" }),
        .libc = try std.fs.path.join(allocator, &.{ libdir, "libc.so.6" }),
        .interp = try std.fs.path.join(allocator, &.{ libdir, soname }),
        .nonshared = try std.fs.path.join(allocator, &.{ libdir, "libc_nonshared.a" }),
    };
}

/// Which kind of library `resolveLib` found: an `.a` archive (fed to the static/`-T`
/// linker unchanged) or a `.so` shared object (a `link.DynInput.shared`, only reachable
/// when dynamic linking is in play). Mirrors `ld.vulcan`'s `LibKind`/`LibFound`.
const LibKind = enum { archive, shared };
const LibFound = struct { bytes: []u8, kind: LibKind };

/// Read `path`, turning "not found" into `null` (expected when probing a candidate
/// filename/kind that just isn't there - the caller tries the next one) rather than an
/// error. Any other read failure propagates. Mirrors `ld.vulcan`'s `tryReadLib`.
fn tryReadLib(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
}

/// Search `dirs` IN ORDER for `lib<name>`, and within EACH directory try the preferred
/// kind (`.so` when `prefer_dynamic`, `.a` otherwise) before the other kind. This matches
/// a real `ld`'s own per-directory search order. `null` means not found in any of them;
/// this is not an error, the caller turns that into a clear diagnostic. Any read failure
/// other than "not found" propagates. Mirrors `ld.vulcan`'s `resolveLib`.
fn resolveLib(allocator: std.mem.Allocator, io: std.Io, name: []const u8, dirs: []const []const u8, prefer_dynamic: bool) !?LibFound {
    const so_filename = try std.fmt.allocPrint(allocator, "lib{s}.so", .{name});
    const a_filename = try std.fmt.allocPrint(allocator, "lib{s}.a", .{name});

    for (dirs) |dir| {
        const so_path = try std.fs.path.join(allocator, &.{ dir, so_filename });
        const a_path = try std.fs.path.join(allocator, &.{ dir, a_filename });

        const first_path = if (prefer_dynamic) so_path else a_path;
        const first_kind: LibKind = if (prefer_dynamic) .shared else .archive;
        if (try tryReadLib(allocator, io, first_path)) |bytes| return .{ .bytes = bytes, .kind = first_kind };

        const second_path = if (prefer_dynamic) a_path else so_path;
        const second_kind: LibKind = if (prefer_dynamic) .archive else .shared;
        if (try tryReadLib(allocator, io, second_path)) |bytes| return .{ .bytes = bytes, .kind = second_kind };
    }
    return null;
}

/// One human-readable line per `link.Error` variant, for a diagnostic that doesn't leak a
/// bare Zig error name at the user. Mirrors `ld.vulcan`'s `linkErrorMessage`.
fn linkErrorMessage(e: link.Error) []const u8 {
    return switch (e) {
        error.OutOfMemory => "out of memory",
        error.MalformedObject => "malformed object or archive input",
        error.UndefinedSymbol => "undefined symbol reference could not be resolved (missing a crt0/_start object or a library?)",
        error.DuplicateSymbol => "duplicate symbol definition across inputs",
        error.RelocationOutOfRange => "a relocation target is out of range",
        error.UnsupportedReloc => "an input uses an unsupported relocation type",
    };
}

/// One human-readable line per `link.ScriptError` variant (the `-T` link path). Mirrors
/// `ld.vulcan`'s `scriptErrorMessage`. `error.ScriptSyntax` cannot actually surface here;
/// the script already parsed successfully by the time `linkInputsScript` runs. It is
/// listed for switch exhaustiveness.
fn scriptErrorMessage(e: link.ScriptError) []const u8 {
    return switch (e) {
        error.OutOfMemory => "out of memory",
        error.MalformedObject => "malformed object or archive input",
        error.UndefinedSymbol => "undefined symbol reference could not be resolved (missing a crt0/_start object or a library?)",
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

/// Maps the host CPU architecture to `vulcan-link`'s `Arch`, the DEFAULT target when
/// `-target` is absent. Mirrors `target.native.hostLinkArch`.
fn hostLinkArch() error{UnsupportedHostArch}!link.Arch {
    return switch (target.native.arch) {
        .aarch64 => .aarch64,
        .x86_64 => .x86_64,
        .riscv64 => .riscv64,
        .x86 => .x86,
        else => error.UnsupportedHostArch,
    };
}

/// The microarch model arch for a target, or null when the target has no model (32-bit x86
/// ships no microarch part).
fn modelArchOf(a: link.Arch) ?mm.Arch {
    return switch (a) {
        .aarch64 => .aarch64,
        .x86_64 => .x86_64,
        .riscv64 => .riscv64,
        .x86 => null,
    };
}

/// The result of the PURE `-mcpu`/`-mtune` decision: a resolved part, an opt-out (no model
/// selected), or a named part that does not exist for the target architecture.
const Pick = union(enum) { model: mm.Microarch, none, mismatch };

/// The pure `-mcpu`/`-mtune` decision. No host reads, no I/O, so this is fully unit-testable.
/// `selector` is `Options.cpu_tune` (null means the implicit native default). `host` is the
/// arch this compiler runs on. It is reserved for a future rule. The current rules read only
/// `detected`, the host's DETECTED part. `detected` is `mm.detectHost()`'s result, or null
/// when the host part is not recognized.
fn pickModel(selector: ?[]const u8, target_arch: link.Arch, host: link.Arch, detected: ?mm.Microarch) Pick {
    _ = host;

    const is_native = selector == null or std.mem.eql(u8, selector.?, "native");
    if (!is_native) {
        if (std.mem.eql(u8, selector.?, "generic") or std.mem.eql(u8, selector.?, "none")) return .none;
        const tag = mm.Microarch.parse(selector.?) orelse return .none;
        return if (mm.modelFor(tag).arch == modelArchOf(target_arch)) .{ .model = tag } else .mismatch;
    }

    if (detected) |d| {
        if (mm.modelFor(d).arch == modelArchOf(target_arch)) return .{ .model = d };
    }
    return .none;
}

/// Writes a `vcc: warning: ...` line to stderr. Failures to write are swallowed: a warning
/// that cannot print is not worth aborting the compile over.
fn warn(io: std.Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [256]u8 = undefined;
    var w = std.Io.File.stderr().writer(io, &buf);
    w.interface.print("vcc: warning: " ++ fmt ++ "\n", args) catch return;
    w.interface.flush() catch return;
}

/// Resolves the effective microarch model for this compile, or null when none applies.
/// Reads the host's detected part (`mm.detectHost`) and this build's own host arch, then
/// applies `pickModel`'s pure decision. An opt-out or an unresolved implicit default warns
/// (or stays silent) and returns null. A named part that does not exist for `arch` is a hard
/// error, except on a 32-bit x86 target, where it only warns (x86 ships no model at all).
fn resolveModel(opts: *const Options, arch: link.Arch, io: std.Io) !?*const mm.Model {
    const detected = mm.detectHost();
    const host = hostLinkArch() catch arch;
    switch (pickModel(opts.cpu_tune, arch, host, detected)) {
        .model => |tag| return mm.modelFor(tag),
        .none => {
            if (opts.cpu_tune) |v| {
                if (std.mem.eql(u8, v, "native")) {
                    warn(io, "-mcpu=native has no model for target {s}, ignoring", .{@tagName(arch)});
                } else if (!std.mem.eql(u8, v, "generic") and !std.mem.eql(u8, v, "none") and mm.Microarch.parse(v) == null) {
                    warn(io, "unknown -mcpu/-mtune value '{s}', ignoring", .{v});
                }
            }
            return null;
        },
        .mismatch => {
            if (arch == .x86) {
                warn(io, "-mcpu/-mtune has no model for a 32-bit x86 target, ignoring", .{});
                return null;
            }
            return fail("-mcpu={s} is not valid for target {s}", .{ opts.cpu_tune.?, @tagName(arch) });
        },
    }
}

/// The default static-executable image base per architecture - matches `ld.vulcan`'s
/// `defaultBase` (aarch64/x86_64/riscv64 share the conventional `0x400000`; `x86` uses the
/// classic i386 `0x08048000`).
fn defaultBase(arch: link.Arch) u64 {
    return switch (arch) {
        .aarch64, .x86_64, .riscv64 => 0x400000,
        .x86 => 0x08048000,
    };
}

/// Maps `objs` (the frontend's own `cc.DataObject`s) to `target.native.ObjData`, the
/// NEUTRAL, arch-independent data-object shape `writeObjectDataFor` consumes. It switches
/// the same-named `cc.DataKind` to `target.native.DataKind` and copies each internal reloc
/// across. This function is extracted out so `compileToObject`, now the single path both
/// `-c` and the link step compile through, doesn't duplicate this mapping.
fn buildModuleData(allocator: std.mem.Allocator, objs: []const cc.DataObject) std.mem.Allocator.Error!std.ArrayList(target.native.ObjData) {
    var list: std.ArrayList(target.native.ObjData) = .empty;
    for (objs) |d| {
        var relocs: []target.native.DataReloc = &.{};
        if (d.relocs.len != 0) {
            relocs = try allocator.alloc(target.native.DataReloc, d.relocs.len);
            for (d.relocs, relocs) |r, *o| o.* = .{ .off = r.off, .symbol = r.symbol };
        }
        try list.append(allocator, .{
            .name = d.name,
            .bytes = d.bytes,
            .kind = switch (d.kind) {
                .rodata => .rodata,
                .data => .data,
                .bss => .bss,
            },
            .size = d.size,
            .relocs = relocs,
        });
    }
    return list;
}

/// Reads and compiles one `.c`/`.i` source file (`path`) to a relocatable ELF object FOR
/// `arch`: `cc.compileWithOpts`, then `buildModuleData`, then
/// `target.native.writeObjectDataFor`. `pp_base` is the driver's shared preprocessor
/// options (defines, undefines, resolver, timestamp). Only `.filename` is overridden per
/// file here, so every `.c` input in a multi-input link shares the same
/// `-I`/`-D`/`-U`/`-freproducible-compile` behavior. `model`, when set, is the resolved
/// microarch model for `arch` (see `resolveModel`). It tunes both the IR layer and the
/// backend for every function this call compiles. Used by both the `-c` path (a single
/// source) and the link step (one call per `.c`/`.i` input, in command order).
fn compileToObject(allocator: std.mem.Allocator, io: std.Io, path: []const u8, pp_base: preproc.Options, arch: link.Arch, model: ?*const mm.Model) ![]u8 {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    var pp_opts = pp_base;
    pp_opts.filename = try filenameForPreproc(allocator, path);
    return compileSourceToObject(allocator, source, pp_opts, arch, model);
}

/// Compiles in-memory `source` (already-read text, `pp_opts` carrying its filename and
/// predefines) to a relocatable object. Split from `compileToObject` so the driver can
/// also compile a SYNTHETIC source string it never read from disk (see
/// `synthDsoHandleObject`). `model`, when set, runs `mm.optimize` over each function's IR
/// before codegen, then hands the same model to `writeObjectDataForModel` so the backend
/// tunes its own choices (instruction selection, scheduling) for it too.
fn compileSourceToObject(allocator: std.mem.Allocator, source: []const u8, pp_opts: preproc.Options, arch: link.Arch, model: ?*const mm.Model) ![]u8 {
    var mod = try cc.compileWithOpts(allocator, source, pp_opts);
    defer mod.deinit(allocator);

    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    for (mod.funcs) |*nf| {
        if (model) |m| _ = try mm.optimize(allocator, &nf.func, m);
        try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });
    }

    const data = try buildModuleData(allocator, mod.data);

    return target.native.writeObjectDataForModel(allocator, arch, mfs.items, data.items, model);
}

/// A synthetic object defining `__dso_handle` (a NULL `void *`), the per-DSO handle the
/// glibc `atexit`/`__cxa_atexit` wrapper (pulled from `libc_nonshared.a`) loads. `crtbegin.o`
/// normally supplies it. VCC links a minimal crt (`crt1.o` only), so the driver provides it
/// instead. A whole-program (main-executable) handle is NULL, so the stored value is 0. This
/// object holds data only, no functions, so it passes no microarch model through.
fn synthDsoHandleObject(allocator: std.mem.Allocator, pp_base: preproc.Options, arch: link.Arch) ![]u8 {
    var pp_opts = pp_base;
    pp_opts.filename = "<vcc-dso-handle>";
    return compileSourceToObject(allocator, "void *__dso_handle = 0;\n", pp_opts, arch, null);
}

/// The `vcc` driver entry point: parse argv, then dispatch on mode. `-E` preprocesses and
/// prints or writes text. `-c` compiles a single source to a relocatable `.o`.
/// `--emit-ir` compiles a single source and dumps its IR text. Otherwise, the default with
/// no flag, every input is compiled or read and LINKED: `.c`/`.i` sources are compiled,
/// `.o`/`.a`/`.so` inputs are read as-is, and `-l<name>` is resolved against `-L` search
/// directories, all in command-line order. The result is either a static executable
/// (`link.linkInputs`/`link.writeExecutable`, or `-T`'s script-driven layout), or, when
/// `-shared`/`--dynamic-linker` is given or any input resolves to a `.so`, a dynamic image
/// via `link.linkDynamic`.
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // argv0

    const opts = try parseArgs(allocator, &it);

    // Autoconf version probes: `--version`/`-v`/`-dumpversion`/`-dumpmachine` answer BEFORE
    // the normal compile/link pipeline runs. `AC_PROG_CC` and similar macros invoke the
    // compiler with only one of these flags and no input file, so `single_source` and the
    // link step below would otherwise reject the invocation outright.
    if (opts.probe) |p| {
        var pbuf: [256]u8 = undefined;
        var w = std.Io.File.stdout().writer(io, &pbuf);
        switch (p) {
            .version => try w.interface.print("vcc (Vulcan C Compiler) {s}\n", .{vcc_version}),
            .dumpversion => try w.interface.print("{s}\n", .{vcc_version}),
            .dumpmachine => {
                const probe_arch: link.Arch = opts.target_arch orelse
                    (hostLinkArch() catch return fail("no native linker support for this host architecture (pass -target)", .{}));
                try w.interface.print("{s}\n", .{archTriple(probe_arch)});
            },
        }
        try w.interface.flush();
        return;
    }

    // `-shared` (a dynamic .so output) and `-c` (a relocatable .o output) are mutually
    // exclusive end products. The check happens up front rather than deep inside the `-c`
    // branch, so the diagnostic fires before any compilation work happens.
    if (opts.shared and opts.compile_only)
        return fail("-shared and -c are mutually exclusive", .{});

    // The compile target for this whole invocation: `-target <arch>` when given, else the
    // host's own arch. Both `-c`/the link step's per-source `.o` emission
    // (`compileToObject`), and the link step's `writeExecutable`, use this SAME arch. A
    // cross-emitted object always links against a matching-arch executable format.
    const arch: link.Arch = opts.target_arch orelse
        (hostLinkArch() catch return fail("no native linker support for this host architecture (pass -target)", .{}));

    // The microarch model this whole invocation tunes for, or null when none applies (an
    // explicit opt-out, an unresolved implicit default, or a target with no model at all).
    // A native build with no `-mcpu`/`-mtune` resolves the detected host part BY DEFAULT,
    // so `compileToObject` below runs the IR microarch layer and hands the backend a model
    // even with no flags given. See `resolveModel`.
    const model = try resolveModel(&opts, arch, io);

    // Default system include directory: the host glibc dev tree when `arch` is the host's
    // own arch, else a CROSS glibc dev tree for `arch`'s triple. This lets `#include
    // <stdio.h>` resolve with no explicit `-I`/`-isystem`. `-nostdinc` suppresses this; the
    // built-in `<stddef.h>`/`<stdarg.h>` still resolve regardless, see `FsResolver`. `null`
    // (the host has no matching glibc dev tree on this machine, or none for a cross target)
    // just means no default dir is added. The compile only fails later, AT an actual
    // `#include <...>` that needed it, not here.
    const default_sysdir: ?[]const u8 = if (opts.no_stdinc) null else blk: {
        const host_arch: ?link.Arch = hostLinkArch() catch null;
        break :blk if (host_arch != null and arch == host_arch.?)
            try findHostGlibcIncludeDir(allocator, io)
        else
            try findCrossGlibcIncludeDir(allocator, io, archTriple(arch));
    };

    // The `FsResolver` wired into `pp_base.resolver` replaces the old disk-only
    // `diskResolve`. It serves this compiler's own built-in `<stddef.h>`/`<stdarg.h>` FIRST
    // (see `fs_resolver.zig`), then searches `system_dirs` (`-I` plus `-isystem` plus the
    // default system dir above, see `buildSystemDirs`). `fs_resolver` is a `main`-scoped
    // `var`, not a temporary inside a helper, backed by `allocator` (the process arena, see
    // `init.arena` above). So it, and every `ResolvedFile` it hands out, outlive the whole
    // compile, the same way `disk_ctx` did before it.
    const system_dirs = try buildSystemDirs(allocator, &opts, default_sysdir);
    var fs_resolver = cc.fs_resolver.FsResolver.init(allocator, io, system_dirs);

    // `-freproducible-compile` pins `__DATE__`/`__TIME__` to the epoch instead of reading
    // the wall clock, so `preprocess`/`preprocessToText`/`compileWithOpts` themselves stay
    // pure. Only the driver ever touches the clock or the filesystem. `.filename` is left
    // at its struct default here; every use site below overrides it with its own input path.
    const timestamp: i64 = if (opts.reproducible) 0 else std.Io.Clock.real.now(io).toSeconds();
    const pp_base: preproc.Options = .{
        .timestamp = timestamp,
        .resolver = fs_resolver.asResolver(),
        .defines = opts.defines.items,
        .undefines = opts.undefines.items,
        // The per-arch GNU predefined-macro set, see `systemPredefFor`. `-D`/`-U` still
        // layer on top afterward (`preseedMacros` applies `defines`/`undefines` AFTER the
        // system set, see `preproc.zig`), so for example `-U__GNUC__` still strips it
        // exactly as it did before this field existed.
        .system = systemPredefFor(arch),
    };

    var buf: [4096]u8 = undefined;

    // `-E`/`-c`/`--emit-ir` each require exactly one `.c`/`.i` source, expressed over
    // `opts.inputs`.
    const single_source: ?[]const u8 = blk: {
        if (!opts.preprocess_only and !opts.compile_only and !opts.emit_ir) break :blk null;
        if (opts.inputs.items.len != 1) return fail("this mode requires exactly one input file", .{});
        break :blk switch (opts.inputs.items[0]) {
            .path => |p| p,
            .lib => return fail("this mode requires a source file, not -l", .{}),
        };
    };

    if (opts.preprocess_only) {
        const input = single_source.?;
        const source = try std.Io.Dir.cwd().readFileAlloc(io, input, allocator, .limited(16 * 1024 * 1024));
        var pp_opts = pp_base;
        pp_opts.filename = try filenameForPreproc(allocator, input);
        const text = try preproc.preprocessToText(allocator, source, pp_opts);
        defer allocator.free(text);
        if (opts.output) |out_path| {
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = text });
        } else {
            var w = std.Io.File.stdout().writer(io, &buf);
            try w.interface.writeAll(text);
            try w.interface.flush();
        }
        return;
    }

    if (opts.compile_only) {
        const input = single_source.?;
        if (classifyExt(input) != .c) return fail("-c requires a .c/.i source file, got '{s}'", .{input});
        const obj = try compileToObject(allocator, io, input, pp_base, arch, model);
        const out = opts.output orelse try derivedOutputName(allocator, input);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = obj });
        if (opts.gen_depfile) {
            // `-MD`/`-MMD`: a minimal Make depfile, `target: source`. VCC does not yet
            // track which headers a compile pulled in, so the recorded prerequisite is the
            // source file alone. A header-aware depfile is a follow-up, not a correctness
            // gap for `make` itself; a missing prerequisite just means an edited header
            // does not trigger a rebuild.
            const dep_target = opts.depfile_target orelse out;
            const dep_path = opts.depfile orelse try derivedDepfileName(allocator, out);
            const dep_text = try std.fmt.allocPrint(allocator, "{s}: {s}\n", .{ dep_target, input });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dep_path, .data = dep_text });
        }
        return;
    }

    if (opts.emit_ir) {
        const input = single_source.?;
        if (classifyExt(input) != .c) return fail("--emit-ir requires a .c/.i source file, got '{s}'", .{input});
        const source = try std.Io.Dir.cwd().readFileAlloc(io, input, allocator, .limited(16 * 1024 * 1024));
        var pp_opts = pp_base;
        pp_opts.filename = try filenameForPreproc(allocator, input);
        var mod = try cc.compileWithOpts(allocator, source, pp_opts);
        defer mod.deinit(allocator);
        var w = std.Io.File.stdout().writer(io, &buf);
        for (mod.funcs) |nf| try w.interface.print("; function {s}\n{f}\n", .{ nf.name, nf.func });
        try w.interface.flush();
        return;
    }

    // LINK step: every input, in command order, becomes one `link.DynInput`.
    // `.c`/`.i` sources are compiled via `compileToObject` (for `arch`, composing with
    // `-target`). `.o`/`.a`/`.so` inputs are read straight off disk. `-l<name>` is resolved
    // against `-L` search directories, conservatively `.a`-first, unless dynamic intent
    // (`-shared`/`--dynamic-linker`/`-Bdynamic`) prefers `.so`; this mirrors `ld.vulcan`'s
    // own `-l` resolution. When nothing here ever turns out `.shared`, converting
    // `dyn_inputs` below reproduces the same `link.Input` array the static/`-T` route
    // expects, so that route stays byte-unchanged. `display_names` parallels `dyn_inputs`
    // in length and order: the input's basename, used only as a `.shared` input's
    // DT_NEEDED fallback name when it carries no `DT_SONAME` of its own. It is empty for
    // `.object`/`.archive` entries.
    var dyn_inputs: std.ArrayList(link.DynInput) = .empty;
    var display_names: std.ArrayList([]const u8) = .empty;
    for (opts.inputs.items) |spec| switch (spec) {
        .path => |p| switch (classifyExt(p)) {
            .c => {
                const obj = try compileToObject(allocator, io, p, pp_base, arch, model);
                try dyn_inputs.append(allocator, .{ .object = obj });
                try display_names.append(allocator, "");
            },
            .o => {
                const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, allocator, .limited(64 * 1024 * 1024));
                try dyn_inputs.append(allocator, .{ .object = bytes });
                try display_names.append(allocator, "");
            },
            .a => {
                const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, allocator, .limited(64 * 1024 * 1024));
                try dyn_inputs.append(allocator, .{ .archive = bytes });
                try display_names.append(allocator, "");
            },
            .so => {
                const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, allocator, .limited(64 * 1024 * 1024));
                try dyn_inputs.append(allocator, .{ .shared = bytes });
                try display_names.append(allocator, std.fs.path.basename(p));
            },
            .unknown => return fail("unrecognized input file type '{s}' (expected .c/.i/.o/.a/.so)", .{p}),
        },
        .lib => |l| {
            // Whole-link dynamic intent (`-shared`/`--dynamic-linker`) can appear anywhere
            // on the command line. By the time this loop runs, the full scan has already
            // completed, so `opts.shared`/`opts.dynamic_linker` are fully known here. A
            // link that is dynamic anyway gets dynamic-first `-l` resolution by default,
            // on top of any `-Bdynamic` already captured on `l.prefer_dynamic`.
            const prefer_dynamic = l.prefer_dynamic or opts.shared or opts.dynamic_linker != null;
            const found = (try resolveLib(allocator, io, l.name, opts.lib_dirs.items, prefer_dynamic)) orelse {
                if (prefer_dynamic)
                    return fail("cannot find -l{s}: no 'lib{s}.so' or 'lib{s}.a' in any -L search directory", .{ l.name, l.name, l.name });
                return fail("cannot find -l{s}: no 'lib{s}.a' in any -L search directory", .{ l.name, l.name });
            };
            switch (found.kind) {
                .archive => {
                    try dyn_inputs.append(allocator, .{ .archive = found.bytes });
                    try display_names.append(allocator, "");
                },
                .shared => {
                    try dyn_inputs.append(allocator, .{ .shared = found.bytes });
                    const fname = try std.fmt.allocPrint(allocator, "lib{s}.so", .{l.name});
                    try display_names.append(allocator, fname);
                },
            }
        },
    };

    // Default-executable AUTOLINK: a normal executable link (not `-shared`, not
    // `-c`/`-E`/`--emit-ir`, not `-T`) discovers the host C runtime and links `crt1.o`,
    // `libc.so.6`, and the dynamic linker around the user's own objects, in gcc's order. So
    // a plain `vcc hello.c -o hello` produces a RUNNING dynamic executable with no
    // hand-supplied crt, `-lc`, or `--dynamic-linker`. `-nostartfiles` drops the crt
    // objects, `-nodefaultlibs` drops `libc.so.6`, and `-nostdlib` drops both, matching gcc.
    // A program that supplies its own `_start`/libc uses those flags to opt out. The
    // discovered interp is embedded as `PT_INTERP` in the dynamic branch below; an explicit
    // `--dynamic-linker` still wins.
    var autolink_interp: ?[]const u8 = null;
    {
        const is_normal_exec = !opts.shared and !opts.compile_only and
            !opts.preprocess_only and !opts.emit_ir and opts.script_path == null;
        const add_crt = is_normal_exec and !opts.no_stdlib and !opts.no_startfiles;
        const add_libc = is_normal_exec and !opts.no_stdlib and !opts.no_defaultlibs;
        if (add_crt or add_libc) {
            const tc = (try discoverToolchain(allocator, io, arch, &opts)) orelse
                return fail("cannot find the C runtime (crt1.o/libc.so.6/the dynamic linker); set VCC_SYSROOT or pass -B <dir>, or use -nostdlib to link freestanding", .{});
            autolink_interp = tc.interp;

            // Rebuild the input list in gcc's order: `crt1.o` FIRST (it defines `_start`), then
            // every user object/lib in command order, then `libc.so.6` LAST (a `.shared` input,
            // so the dynamic branch records its `DT_SONAME` = `libc.so.6` as a `DT_NEEDED`).
            var autolinked: std.ArrayList(link.DynInput) = .empty;
            var autolinked_names: std.ArrayList([]const u8) = .empty;
            if (add_crt) {
                const crt1_bytes = std.Io.Dir.cwd().readFileAlloc(io, tc.crt1, allocator, .limited(64 * 1024 * 1024)) catch
                    return fail("cannot read the discovered crt1.o at '{s}'", .{tc.crt1});
                try autolinked.append(allocator, .{ .object = crt1_bytes });
                try autolinked_names.append(allocator, "");
            }
            // Define `__dso_handle` when autolinking glibc. The `atexit` wrapper in
            // `libc_nonshared.a` loads it, and the minimal crt (no crtbegin.o) never
            // supplies it. This is an always-included synthetic object, ahead of the user
            // inputs.
            if (add_libc) {
                const dso = try synthDsoHandleObject(allocator, pp_base, arch);
                try autolinked.append(allocator, .{ .object = dso });
                try autolinked_names.append(allocator, "<vcc-dso-handle>");
            }
            try autolinked.appendSlice(allocator, dyn_inputs.items);
            try autolinked_names.appendSlice(allocator, display_names.items);
            if (add_libc) {
                const libc_bytes = std.Io.Dir.cwd().readFileAlloc(io, tc.libc, allocator, .limited(256 * 1024 * 1024)) catch
                    return fail("cannot read the discovered libc.so.6 at '{s}'", .{tc.libc});
                try autolinked.append(allocator, .{ .shared = libc_bytes });
                try autolinked_names.append(allocator, "libc.so.6");
                // `libc_nonshared.a` comes after `libc.so.6`, matching the real `-lc`
                // linker script `GROUP ( libc.so.6 libc_nonshared.a ... )`. A few symbols
                // (`atexit`, `stat`, ...) live only here. It is a pull-on-demand `.archive`,
                // so it adds nothing to a link that never references those symbols. When
                // absent, for example a non-glibc libc, it is skipped.
                if (try tryReadLib(allocator, io, tc.nonshared)) |ns_bytes| {
                    try autolinked.append(allocator, .{ .archive = ns_bytes });
                    try autolinked_names.append(allocator, "libc_nonshared.a");
                }
            }
            dyn_inputs = autolinked;
            display_names = autolinked_names;
        }
    }

    // Dynamic-mode TRIGGER: `-shared`, `--dynamic-linker`, any input that resolved to a
    // `.so`, or the autolink step above (which appends `libc.so.6` and supplies
    // `autolink_interp`). Absent all of these, `dyn_inputs` above holds only
    // `.object`/`.archive` entries, so the conversion below reproduces the exact
    // `link.Input` array. The static/`-T` route is unaffected either way.
    var dynamic_mode = opts.shared or opts.dynamic_linker != null or autolink_interp != null;
    for (dyn_inputs.items) |di| {
        if (di == .shared) dynamic_mode = true;
    }

    if (dynamic_mode and opts.script_path != null)
        return fail("-T (linker scripts) does not support dynamic linking (-shared/--dynamic-linker/.so inputs)", .{});

    // The static/`-T` route's own input shape, built only when NOT dynamic. A `.shared`
    // entry can only appear in `dyn_inputs` when `dynamic_mode` is already true, so this
    // conversion never has to represent one as a `link.Input`.
    var link_inputs: std.ArrayList(link.Input) = .empty;
    if (!dynamic_mode) {
        for (dyn_inputs.items) |di| switch (di) {
            .object => |b| try link_inputs.append(allocator, .{ .object = b }),
            .archive => |b| try link_inputs.append(allocator, .{ .archive = b }),
            .shared => unreachable, // dynamic_mode would be true otherwise
        };
    }

    // `-T <script>` routes the link through the script-driven layout engine instead of
    // the default contiguous layout. It composes with `-target`: `arch` above already
    // reflects `-target` (or the host), and `linkInputsScript` re-derives the same arch
    // from the objects themselves. Both must agree, same as the default path, so
    // `applyRelocs`/`writeElfSegments` patch and emit for the right target either way. This
    // is a self-contained flow, mirroring `ld.vulcan`'s `-T` branch; the default path below
    // is unchanged. `dynamic_mode` was already rejected above when `-T` is also given, so
    // this branch only ever runs with a static `link_inputs`.
    if (opts.script_path) |sp| {
        const script_bytes = std.Io.Dir.cwd().readFileAlloc(io, sp, allocator, .limited(16 * 1024 * 1024)) catch |e|
            return fail("cannot read '{s}': {s}", .{ sp, @errorName(e) });

        var sdiag: link.ScriptDiagnostic = .{ .line = 0, .col = 0, .msg = "" };
        var parsed_script = link.parseScript(allocator, script_bytes, &sdiag) catch |e| {
            if (e == error.OutOfMemory) return e;
            return fail("{s}:{d}:{d}: {s}", .{ sp, sdiag.line, sdiag.col, sdiag.msg });
        };
        defer parsed_script.deinit();

        var linked = link.linkInputsScript(allocator, link_inputs.items, &parsed_script, null) catch |e| {
            if (e == error.OutOfMemory) return e;
            return fail("{s}", .{scriptErrorMessage(e)});
        };
        defer linked.deinit(allocator);

        const elf_bytes = try link.writeElfSegments(linked.arch, allocator, &linked.placement, linked.entry);

        const out = opts.output orelse "a.out";
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = out,
            .data = elf_bytes,
            .flags = .{ .permissions = .executable_file },
        });
        return;
    }

    // The dynamic route: `link.linkDynamic` takes the classified
    // `.object`/`.archive`/`.shared` inputs directly. It resolves imports against
    // `.shared` exports itself, so no PLT/GOT bookkeeping is needed here. It returns the
    // finished ELF bytes: a `.so` for `-shared`, otherwise a dynamic executable. It
    // composes with `-target`: every `.object` above was compiled or read for `arch`, and
    // `linkDynamic` infers the same arch from those objects, so the emitted image matches.
    // Mirrors `ld.vulcan`'s own dynamic branch.
    if (dynamic_mode) {
        const mode: link.DynMode = if (opts.shared) .shared else .exec;
        // The `PT_INTERP` for a dynamic executable: an explicit `--dynamic-linker` wins,
        // else the autolink step's discovered loader (`autolink_interp`). A dynamic exec
        // still needs one either way; a `-shared` object needs none.
        const interp = opts.dynamic_linker orelse autolink_interp;
        if (mode == .exec and interp == null)
            return fail("a dynamic executable needs --dynamic-linker <path>", .{});

        // DT_NEEDED: one entry per `.shared` input, preferring its own DT_SONAME and
        // falling back to its basename (the `.so` path's basename, or the `-l`-resolved
        // `lib<name>.so` filename) when it carries none.
        var needed_list: std.ArrayList([]const u8) = .empty;
        for (dyn_inputs.items, display_names.items) |di, disp| switch (di) {
            .shared => |bytes| {
                var se = link.readSharedExports(allocator, bytes) catch |e| {
                    if (e == error.OutOfMemory) return e;
                    return fail("cannot read a shared object's exports: {s}", .{@errorName(e)});
                };
                defer se.deinit(allocator);
                const name = if (se.soname) |s| try allocator.dupe(u8, s) else try allocator.dupe(u8, disp);
                try needed_list.append(allocator, name);
            },
            else => {},
        };

        const dyn_bytes = link.linkDynamic(allocator, dyn_inputs.items, .{
            .mode = mode,
            .interp = interp,
            .soname = opts.soname,
            .needed = needed_list.items,
            .base = defaultBase(arch),
            .entry = "_start",
        }) catch |e| {
            if (e == error.OutOfMemory) return e;
            return fail("{s}", .{linkErrorMessage(e)});
        };

        // A `.so` gets exec permissions too: harmless (it is never invoked directly) and
        // conventional (matches what a real `ld -shared` produces).
        const out = opts.output orelse "a.out";
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = out,
            .data = dyn_bytes,
            .flags = .{ .permissions = .executable_file },
        });
        return;
    }

    const base = defaultBase(arch);

    var image = link.linkInputs(allocator, link_inputs.items, base) catch |e| {
        if (e == error.OutOfMemory) return e;
        return fail("{s}", .{linkErrorMessage(e)});
    };
    defer image.deinit(allocator);

    const elf_bytes = link.writeExecutable(arch, allocator, &image, "_start") catch |e| {
        if (e == error.OutOfMemory) return e;
        return fail("{s}", .{linkErrorMessage(e)});
    };

    const out = opts.output orelse "a.out";
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = out,
        .data = elf_bytes,
        .flags = .{ .permissions = .executable_file },
    });
}

// ---------------------------------------------------------------------------------
// Tests: `parseArgs`' flag-classification and version-probe surface, driven IN-PROCESS
// (no process spawn) off a literal argument slice.

/// A `[]const []const u8`-backed stand-in for the real `std.process.Args` iterator:
/// `parseArgs` only ever calls `.next() ?[]const u8`, so a test drives it straight off a
/// literal slice instead of spawning a real process or reading real argv.
const SliceArgs = struct {
    items: []const []const u8,
    idx: usize = 0,

    fn next(self: *SliceArgs) ?[]const u8 {
        if (self.idx >= self.items.len) return null;
        const a = self.items[self.idx];
        self.idx += 1;
        return a;
    }
};

test "parseArgs: a ./configure-style flag soup does not fatal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{
        "-std=gnu11", "-Wall", "-Wextra", "-g",  "-O2", "-fPIC", "-pthread", "-rdynamic",
        "-MMD",       "-MF",   "x.d",     "-MT", "x.o", "-pipe", "-c",       "foo.c",
        "-o",         "foo.o",
    } };
    const opts = try parseArgs(allocator, &it);

    try std.testing.expect(opts.compile_only);
    try std.testing.expectEqualStrings("foo.o", opts.output.?);
    try std.testing.expectEqual(@as(usize, 1), opts.inputs.items.len);
    try std.testing.expectEqualStrings("foo.c", opts.inputs.items[0].path);
    try std.testing.expect(opts.gen_depfile);
    try std.testing.expectEqualStrings("x.d", opts.depfile.?);
    try std.testing.expectEqualStrings("x.o", opts.depfile_target.?);
    try std.testing.expect(opts.pthread);
    try std.testing.expect(opts.rdynamic);
    try std.testing.expect(opts.probe == null);
}

test "parseArgs: an arg-taking gcc flag consumes its value, not the next input" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // `-isystem /usr/include` and `-include prelude.h` each carry a SEPARATE-token argument.
    // Both the flag and its value must be consumed, so `foo.c` stays the only input file (a
    // regression against reading `/usr/include` or `prelude.h` as a positional source).
    var it = SliceArgs{ .items = &.{
        "-isystem", "/usr/include", "-include", "prelude.h", "-c", "foo.c", "-o", "foo.o",
    } };
    const opts = try parseArgs(allocator, &it);

    try std.testing.expectEqual(@as(usize, 1), opts.inputs.items.len);
    try std.testing.expectEqualStrings("foo.c", opts.inputs.items[0].path);
    try std.testing.expectEqualStrings("foo.o", opts.output.?);
}

test "parseArgs: --version and -v both set the version probe" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it1 = SliceArgs{ .items = &.{"--version"} };
    const o1 = try parseArgs(allocator, &it1);
    try std.testing.expectEqual(Probe.version, o1.probe.?);

    var it2 = SliceArgs{ .items = &.{"-v"} };
    const o2 = try parseArgs(allocator, &it2);
    try std.testing.expectEqual(Probe.version, o2.probe.?);
}

test "parseArgs: -dumpversion and -dumpmachine set their own probes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it1 = SliceArgs{ .items = &.{"-dumpversion"} };
    const o1 = try parseArgs(allocator, &it1);
    try std.testing.expectEqual(Probe.dumpversion, o1.probe.?);

    var it2 = SliceArgs{ .items = &.{"-dumpmachine"} };
    const o2 = try parseArgs(allocator, &it2);
    try std.testing.expectEqual(Probe.dumpmachine, o2.probe.?);
}

test "parseArgs: a version probe alone (no input file) is not an error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{"--version"} };
    _ = try parseArgs(allocator, &it); // used to return error.Usage ("no input file")
}

test "parseArgs: -m32/-m64/-mabi= are denylisted and fail closed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it1 = SliceArgs{ .items = &.{ "-m32", "foo.c" } };
    try std.testing.expectError(error.Usage, parseArgs(allocator, &it1));

    var it2 = SliceArgs{ .items = &.{ "-m64", "foo.c" } };
    try std.testing.expectError(error.Usage, parseArgs(allocator, &it2));

    var it3 = SliceArgs{ .items = &.{ "-mabi=lp64", "foo.c" } };
    try std.testing.expectError(error.Usage, parseArgs(allocator, &it3));
}

test "parseArgs: -mcpu= and -mtune= capture the value, last wins" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-mcpu=cascadelake-sp", "-c", "foo.c" } };
    const o = try parseArgs(allocator, &it);
    try std.testing.expectEqualStrings("cascadelake-sp", o.cpu_tune.?);

    var it2 = SliceArgs{ .items = &.{ "-mtune=native", "-mcpu=et-soc", "-c", "foo.c" } };
    const o2 = try parseArgs(allocator, &it2);
    try std.testing.expectEqualStrings("et-soc", o2.cpu_tune.?); // last wins
}

test "parseArgs: -march= is accepted and ignored (no model selector)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-march=armv8-a", "-c", "foo.c" } };
    const o = try parseArgs(allocator, &it);
    try std.testing.expect(o.cpu_tune == null);
}

test "pickModel: named part matching target -> model" {
    try std.testing.expectEqual(mm.Microarch.@"cascadelake-sp", pickModel("cascadelake-sp", .x86_64, .aarch64, null).model);
}

test "pickModel: named part not matching target -> mismatch" {
    try std.testing.expect(pickModel("cascadelake-sp", .riscv64, .riscv64, null) == .mismatch);
}

test "pickModel: implicit native on recognized matching host -> host model" {
    try std.testing.expectEqual(mm.Microarch.@"ampere-altra", pickModel(null, .aarch64, .aarch64, .@"ampere-altra").model);
}

test "pickModel: implicit native, cross target -> none" {
    try std.testing.expect(pickModel(null, .x86_64, .aarch64, .@"ampere-altra") == .none);
}

test "pickModel: generic opts out -> none" {
    try std.testing.expect(pickModel("generic", .aarch64, .aarch64, .@"ampere-altra") == .none);
}

test "pickModel: unknown value -> none" {
    try std.testing.expect(pickModel("frobnicate", .aarch64, .aarch64, .@"ampere-altra") == .none);
}

test "pickModel: named part on i386 target -> mismatch (no x86 model)" {
    try std.testing.expect(pickModel("cascadelake-sp", .x86, .x86, null) == .mismatch);
}

test "resolveModel: a named part mismatched with -target fails closed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-target", "riscv64", "-mcpu=cascadelake-sp", "foo.c" } };
    const o = try parseArgs(allocator, &it);
    try std.testing.expectError(error.Usage, resolveModel(&o, .riscv64, std.testing.io));
}

test "parseArgs: -S still fails closed, not silently ignored" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-S", "foo.c" } };
    try std.testing.expectError(error.Usage, parseArgs(allocator, &it));
}

test "parseArgs: crt-suppression flags are parsed for M5c to consume" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-nostdlib", "-nostartfiles", "-nodefaultlibs", "foo.c" } };
    const opts = try parseArgs(allocator, &it);
    try std.testing.expect(opts.no_stdlib);
    try std.testing.expect(opts.no_startfiles);
    try std.testing.expect(opts.no_defaultlibs);
}

test "parseArgs: every pre-M5a flag still parses exactly as before" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{
        "-target", "x86_64", "-I", "inc", "-Dfoo=bar", "-U", "baz", "-c", "-o", "out.o", "foo.c",
    } };
    const opts = try parseArgs(allocator, &it);

    try std.testing.expectEqual(link.Arch.x86_64, opts.target_arch.?);
    try std.testing.expectEqual(@as(usize, 1), opts.includes.items.len);
    try std.testing.expectEqualStrings("inc", opts.includes.items[0]);
    try std.testing.expectEqual(@as(usize, 1), opts.defines.items.len);
    try std.testing.expectEqualStrings("foo", opts.defines.items[0].name);
    try std.testing.expectEqualStrings("bar", opts.defines.items[0].value);
    try std.testing.expectEqual(@as(usize, 1), opts.undefines.items.len);
    try std.testing.expectEqualStrings("baz", opts.undefines.items[0]);
    try std.testing.expect(opts.compile_only);
    try std.testing.expectEqualStrings("out.o", opts.output.?);

    // An unknown -target arch is still a hard error (unrelated to the new flag families).
    var it2 = SliceArgs{ .items = &.{ "-target", "sparc64", "foo.c" } };
    try std.testing.expectError(error.Usage, parseArgs(allocator, &it2));

    // A bare compile with no input is still a hard error (no probe requested).
    var it3 = SliceArgs{ .items = &.{"-Wall"} };
    try std.testing.expectError(error.Usage, parseArgs(allocator, &it3));
}

// ---------------------------------------------------------------------------------
// Tests: default system-include resolution. This covers `-isystem` parsing,
// `buildSystemDirs`/`systemPredefFor`'s pure wiring logic, and the `FsResolver` this
// drives, exercised the same in-process way `preproc_glibc.zig`'s own tests do. There is
// no `vcc` binary spawn; that end-to-end proof is a manual run.

test "parseArgs: -isystem <dir> and -isystem<dir> both populate opts.isystem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-isystem", "/usr/include", "-isystem/opt/include", "-c", "foo.c" } };
    const opts = try parseArgs(allocator, &it);

    try std.testing.expectEqual(@as(usize, 2), opts.isystem.items.len);
    try std.testing.expectEqualStrings("/usr/include", opts.isystem.items[0]);
    try std.testing.expectEqualStrings("/opt/include", opts.isystem.items[1]);
    // Not read as a positional input, a regression this flag family already covers.
    try std.testing.expectEqual(@as(usize, 1), opts.inputs.items.len);
}

test "parseArgs: -nostdinc is parsed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var it = SliceArgs{ .items = &.{ "-nostdinc", "-c", "foo.c" } };
    const opts = try parseArgs(allocator, &it);
    try std.testing.expect(opts.no_stdinc);
}

test "buildSystemDirs: -I dirs come before -isystem dirs, which come before the default dir" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var opts = Options{};
    try opts.includes.append(allocator, "inc1");
    try opts.isystem.append(allocator, "sys1");
    const dirs = try buildSystemDirs(allocator, &opts, "default1");

    try std.testing.expectEqual(@as(usize, 3), dirs.len);
    try std.testing.expectEqualStrings("inc1", dirs[0]);
    try std.testing.expectEqualStrings("sys1", dirs[1]);
    try std.testing.expectEqualStrings("default1", dirs[2]);
}

test "systemPredefFor: long_bits/ptr_bits/char_signed match cc.layout.forArch per target" {
    const aarch64_sp = systemPredefFor(.aarch64);
    try std.testing.expectEqual(cc.layout.Arch.aarch64, aarch64_sp.arch);
    try std.testing.expectEqual(@as(u16, 64), aarch64_sp.long_bits);
    try std.testing.expectEqual(@as(u16, 64), aarch64_sp.ptr_bits);
    try std.testing.expect(!aarch64_sp.char_signed);
    try std.testing.expectEqual(@as(u32, 4), aarch64_sp.gnuc_major);
    try std.testing.expectEqual(@as(u32, 6), aarch64_sp.gnuc_minor);

    const x86_sp = systemPredefFor(.x86);
    try std.testing.expectEqual(cc.layout.Arch.x86, x86_sp.arch);
    try std.testing.expectEqual(@as(u16, 32), x86_sp.long_bits);
    try std.testing.expectEqual(@as(u16, 32), x86_sp.ptr_bits);
    try std.testing.expect(x86_sp.char_signed);

    const x86_64_sp = systemPredefFor(.x86_64);
    try std.testing.expectEqual(cc.layout.Arch.x86_64, x86_64_sp.arch);
    try std.testing.expect(x86_64_sp.char_signed);
}

test "filenameForPreproc: a bare filename gains a directory component, a pathed one is unchanged" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const bare = try filenameForPreproc(allocator, "foo.c");
    try std.testing.expectEqualStrings("./foo.c", bare);
    try std.testing.expect(std.fs.path.dirname(bare) != null);

    const pathed = try filenameForPreproc(allocator, "src/foo.c");
    try std.testing.expectEqualStrings("src/foo.c", pathed);
}

test "driver wiring: built-in stddef.h resolves through FsResolver+buildSystemDirs with no -I at all" {
    const allocator = std.testing.allocator;

    var opts = Options{};
    const system_dirs = try buildSystemDirs(allocator, &opts, null);
    defer allocator.free(system_dirs);

    var fsr = cc.fs_resolver.FsResolver.init(allocator, std.testing.io, system_dirs);
    const toks = try preproc.preprocess(allocator, "#include <stddef.h>\nsize_t x;\n", .{
        .resolver = fsr.asResolver(),
        .system = systemPredefFor(.aarch64),
    });
    defer cc.lexer.freeTokens(allocator, toks);

    // The built-in's own `typedef`s precede the source's own `size_t x;` line, unexpanded;
    // the preprocessor has no notion of typedefs. Only the LAST few tokens are checked, so
    // this doesn't pin the built-in's own exact token count. It proves the built-in served
    // with ZERO `-I`/`-isystem`/default dir, exactly like a bare `vcc -c hello.c` whose
    // `hello.c` does `#include <stddef.h>`.
    try std.testing.expect(toks.len >= 4);
    try std.testing.expectEqualStrings("size_t", toks[toks.len - 4].text);
    try std.testing.expectEqualStrings("x", toks[toks.len - 3].text);
    try std.testing.expectEqualStrings(";", toks[toks.len - 2].text);
    try std.testing.expectEqual(cc.lexer.Kind.eof, toks[toks.len - 1].kind);
}

test "driver wiring: -isystem resolves a <...> include from that directory" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "sysfoo.h", .data = "int from_sysfoo;\n" });
    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var opts = Options{};
    try opts.isystem.append(arena.allocator(), dir_path);
    const system_dirs = try buildSystemDirs(arena.allocator(), &opts, null);

    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, system_dirs);
    const toks = try preproc.preprocess(allocator, "#include <sysfoo.h>\nint y;\n", .{ .resolver = fsr.asResolver() });
    defer cc.lexer.freeTokens(allocator, toks);
    try std.testing.expectEqualStrings("from_sysfoo", toks[1].text);
    try std.testing.expectEqualStrings("y", toks[4].text);
}

test "driver wiring regression: an -I dir still resolves a quoted include" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "quoted.h", .data = "int from_quoted_h;\n" });
    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var opts = Options{};
    try opts.includes.append(arena.allocator(), dir_path);
    const system_dirs = try buildSystemDirs(arena.allocator(), &opts, null);

    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, system_dirs);
    const toks = try preproc.preprocess(allocator, "#include \"quoted.h\"\nint z;\n", .{ .resolver = fsr.asResolver() });
    defer cc.lexer.freeTokens(allocator, toks);
    try std.testing.expectEqualStrings("from_quoted_h", toks[1].text);
    try std.testing.expectEqualStrings("z", toks[4].text);
}

test "driver wiring regression: a quoted include next to a real (non-bare) source directory still resolves with no -I" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "local.h", .data = "int from_local_h;\n" });
    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer allocator.free(dir_path);
    const main_path = try std.fs.path.join(allocator, &.{ dir_path, "main.c" });
    defer allocator.free(main_path);

    // `fsr`'s OWN disk reads, the resolved `local.h` bytes and identity, come from an
    // arena. They free as one block once `preprocess`, which copies out everything it
    // needs into `toks`, has returned. This matches `fs_resolver.zig`'s own disk-reading
    // tests, since `FsResolver` never frees an individual read itself; see that module's
    // doc comment on lifetime.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var opts = Options{};
    const system_dirs = try buildSystemDirs(arena.allocator(), &opts, null);

    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, system_dirs);
    const toks = try preproc.preprocess(allocator, "#include \"local.h\"\nint w;\n", .{
        .resolver = fsr.asResolver(),
        .filename = try filenameForPreproc(arena.allocator(), main_path),
    });
    defer cc.lexer.freeTokens(allocator, toks);
    try std.testing.expectEqualStrings("from_local_h", toks[1].text);
    try std.testing.expectEqualStrings("w", toks[4].text);
}

/// Locates the host glibc dev include dir the same way `main` does, for the proof below.
/// This is a test-local re-declaration, not a call into the driver's own
/// `findHostGlibcIncludeDir`, which needs a real `std.Io`/`sh` subprocess. It mirrors
/// `preproc_glibc.zig`'s own discovery test, so this test SKIPs cleanly instead of failing
/// on a host with no glibc dev tree.
fn testFindHostGlibcIncludeDir(allocator: std.mem.Allocator) !?[]u8 {
    return findHostGlibcIncludeDir(allocator, std.testing.io);
}

test "driver wiring: default system dir (host glibc) resolves #include <stdio.h>, or skips cleanly" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const glibc_include = (try testFindHostGlibcIncludeDir(allocator)) orelse return error.SkipZigTest;
    defer allocator.free(glibc_include);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var opts = Options{};
    const system_dirs = try buildSystemDirs(arena.allocator(), &opts, glibc_include);

    var fsr = cc.fs_resolver.FsResolver.init(arena.allocator(), io, system_dirs);
    const toks = try preproc.preprocess(allocator, "#include <stdio.h>\n", .{
        .resolver = fsr.asResolver(),
        .system = systemPredefFor(.aarch64),
    });
    defer cc.lexer.freeTokens(allocator, toks);

    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(allocator);
    for (toks) |t| {
        try joined.appendSlice(allocator, t.text);
        try joined.append(allocator, ' ');
    }
    try std.testing.expect(std.mem.indexOf(u8, joined.items, "printf") != null);
}
