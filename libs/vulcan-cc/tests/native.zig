//! Execution tests for the C frontend. Each test compiles a C translation unit two ways and
//! checks that the results agree. The oracle path compiles the source with the host
//! `gcc`/`cc`, wrapped in a `main` that prints `f(args...)`'s return value. The value is
//! cast to the matching unsigned width first, so it prints the full bit pattern regardless
//! of sign. The frontend path lowers the same source to Vulcan IR and JIT-runs `f` on the
//! host, called with the same arguments. It reads the result at `f`'s actual IR return width
//! (8, 16, 32, or 64 bits) and zero-extends it to `u64`. A test skips when no C compiler is
//! on PATH.

const std = @import("std");
const builtin = @import("builtin");
const cc = @import("vulcan-cc");
const target = @import("vulcan-target");
const opt = @import("vulcan-opt");
const ir = @import("vulcan-ir");

/// The unsigned C type name that matches `ret_bits`. The oracle casts `f(args...)`'s result
/// to this type before printing it, so gcc prints the same full bit pattern the frontend
/// reads.
fn unsignedCTypeName(ret_bits: u16) ![]const u8 {
    return switch (ret_bits) {
        8 => "unsigned char",
        16 => "unsigned short",
        32 => "unsigned int",
        64 => "unsigned long long",
        else => error.Unsupported,
    };
}

/// One `#include`-able header for `expectAgreesPPBits`. `name` is the literal header
/// spelling both sides look it up by. gcc writes it to disk under that name. The frontend's
/// virtual `IncludeResolver` matches it by exact string. `content` is its raw, unprocessed
/// text.
const HeaderFile = struct { name: []const u8, content: []const u8 };

/// gcc-compile `source` (which defines `f(...)`, returning `ret_bits` wide) plus a printing
/// `main` that calls `f(args...)`, casts the result to the matching unsigned type (so the
/// full bit pattern prints, not a sign-extended default `%d`), and prints it with `%llu`.
/// `headers` are written into the same temp dir first, each under its own `name`. The `cc`
/// call below runs with `.cwd = tmp.dir`, so a `#include "name"` in `source` resolves
/// against them with no `-I` needed (the header list is empty for most callers). Runs the
/// program and returns its trimmed stdout. Returns `error.NoCompiler` if `cc` is absent, so
/// the caller can skip.
fn oracleRun(io: std.Io, allocator: std.mem.Allocator, source: []const u8, headers: []const HeaderFile, args: []const i32, ret_bits: u16) ![]u8 {
    return oracleRunStd(io, allocator, source, headers, args, ret_bits, "-std=c99");
}

/// `oracleRun`, but under `-std=gnu99` instead of `-std=c99`. The bare `typeof` spelling is
/// only a keyword in a GNU dialect. Its double-underscored `__typeof__` GNU-builtin twin
/// stays available under every `-std=`, but plain `typeof` does not. Under strict
/// `-std=c99`, gcc reads it as an ordinary, undeclared identifier and rejects
/// `typeof(x) y = x;` outright. This is the only difference from `oracleRun`. Both funnel
/// through the same `oracleRunStd`, so every existing `oracleRun` caller is untouched.
fn oracleRunGnu99(io: std.Io, allocator: std.mem.Allocator, source: []const u8, args: []const i32, ret_bits: u16) ![]u8 {
    return oracleRunStd(io, allocator, source, &.{}, args, ret_bits, "-std=gnu99");
}

fn oracleRunStd(io: std.Io, allocator: std.mem.Allocator, source: []const u8, headers: []const HeaderFile, args: []const i32, ret_bits: u16, std_flag: []const u8) ![]u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    for (headers) |h| try tmp.dir.writeFile(io, .{ .sub_path = h.name, .data = h.content });

    var call: std.ArrayList(u8) = .empty;
    defer call.deinit(allocator);
    try call.appendSlice(allocator, "f(");
    for (args, 0..) |a, i| {
        if (i != 0) try call.appendSlice(allocator, ", ");
        try call.print(allocator, "{d}", .{a});
    }
    try call.appendSlice(allocator, ")");

    const ut = try unsignedCTypeName(ret_bits);
    const program = try std.fmt.allocPrint(allocator,
        \\{s}
        \\#include <stdio.h>
        \\int main(void) {{ printf("%llu\n", (unsigned long long)({s})({s})); return 0; }}
        \\
    , .{ source, ut, call.items });
    defer allocator.free(program);
    try tmp.dir.writeFile(io, .{ .sub_path = "prog.c", .data = program });

    const compiled = std.process.run(allocator, io, .{
        .argv = &.{ "cc", std_flag, "-O0", "-w", "-o", "prog", "prog.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoCompiler,
        else => return err,
    };
    defer allocator.free(compiled.stdout);
    defer allocator.free(compiled.stderr);
    if (compiled.term != .exited or compiled.term.exited != 0) {
        std.debug.print("cc failed:\n{s}\n--- source ---\n{s}\n", .{ compiled.stderr, program });
        return error.CompileFailed;
    }

    const ran = try std.process.run(allocator, io, .{
        .argv = &.{"./prog"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);
    return allocator.dupe(u8, std.mem.trim(u8, ran.stdout, " \n\r\t"));
}

/// gcc-compile `source` (which defines a `float`- or `double`-returning `f(...)`) plus a
/// printing `main` that calls `f(args...)` and reinterprets the raw result bits through a
/// union: `union { float v; unsigned u; }` for float, or `union { double v; unsigned long
/// long u; }` for double. It prints the union's integer member as an unsigned decimal
/// (`%u`/`%llu`). This is the float-return counterpart to `oracleRun`. It uses a union
/// reinterpret rather than an integer cast, since `%f`/`%g` would round and lose bits. It is
/// kept as its own function, rather than folded into `oracleRun`, so the pre-existing int
/// path stays byte-identical. It mirrors `oracleRun`'s temp-dir/cc machinery exactly, minus
/// the `headers` parameter, since no caller here needs `#include`s. Returns
/// `error.NoCompiler` if `cc` is absent, so the caller can skip.
fn oracleRunFloat(io: std.Io, allocator: std.mem.Allocator, source: []const u8, args: []const i32, is_double: bool) ![]u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var call: std.ArrayList(u8) = .empty;
    defer call.deinit(allocator);
    try call.appendSlice(allocator, "f(");
    for (args, 0..) |a, i| {
        if (i != 0) try call.appendSlice(allocator, ", ");
        try call.print(allocator, "{d}", .{a});
    }
    try call.appendSlice(allocator, ")");

    const ct = if (is_double) "double" else "float";
    const ut = if (is_double) "unsigned long long" else "unsigned int";
    const fmt = if (is_double) "%llu" else "%u";
    const program = try std.fmt.allocPrint(allocator,
        \\{s}
        \\#include <stdio.h>
        \\int main(void) {{ union {{ {s} v; {s} u; }} x; x.v = ({s})({s}); printf("{s}\n", x.u); return 0; }}
        \\
    , .{ source, ct, ut, ct, call.items, fmt });
    defer allocator.free(program);
    try tmp.dir.writeFile(io, .{ .sub_path = "prog.c", .data = program });

    const compiled = std.process.run(allocator, io, .{
        .argv = &.{ "cc", "-std=c99", "-O0", "-w", "-o", "prog", "prog.c" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |err| switch (err) {
        error.FileNotFound => return error.NoCompiler,
        else => return err,
    };
    defer allocator.free(compiled.stdout);
    defer allocator.free(compiled.stderr);
    if (compiled.term != .exited or compiled.term.exited != 0) {
        std.debug.print("cc failed:\n{s}\n--- source ---\n{s}\n", .{ compiled.stderr, program });
        return error.CompileFailed;
    }

    const ran = try std.process.run(allocator, io, .{
        .argv = &.{"./prog"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(ran.stdout);
    defer allocator.free(ran.stderr);
    return allocator.dupe(u8, std.mem.trim(u8, ran.stdout, " \n\r\t"));
}

/// `f`'s return classification, read off whichever block's `ret` terminator carries a
/// value: either an integer at a given bit width, or a scalar float width (`f32`/`f64`).
/// `L.ret_ty` is uniform across every `return`/fall-off-the-end seal in a function, so any
/// one `ret` settles it.
const ReturnKind = union(enum) { int: u16, f32, f64 };

/// The bit width or kind of `func`'s return type. See `ReturnKind`. `.f16` has no harness
/// support yet, since no caller needs it, so it fails closed with `error.Unsupported` like
/// any other unhandled kind.
fn returnKind(func: *const ir.function.Function) !ReturnKind {
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: ir.function.Block = @enumFromInt(@as(u32, @intCast(bi)));
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| if (r.count > 0) {
                const v = r.values[0];
                return switch (func.types.type_kind(func.valueType(v))) {
                    .int => |i| .{ .int = i.bits },
                    .float => |fk| switch (fk) {
                        .f32 => .f32,
                        .f64 => .f64,
                        .f16 => error.Unsupported,
                    },
                    else => error.Unsupported,
                };
            },
            .jump => {},
        };
    }
    return error.Unsupported; // every lowered function has at least one `ret`
}

/// Call the JIT'd `f` with `args` (0 to 3 ints), reading its result at `ret_bits` width
/// (8, 16, 32, or 64) and zero-extending the exact bit pattern to `u64`. `f` always takes
/// `int` params. Only the return width varies.
fn callFBits(jitted: *const target.native.JittedModule, args: []const i32, ret_bits: u16) !u64 {
    return switch (ret_bits) {
        8 => switch (args.len) {
            0 => @as(u8, @bitCast((jitted.entry(*const fn () callconv(.c) i8, "f") orelse return error.NoEntry)())),
            1 => @as(u8, @bitCast((jitted.entry(*const fn (i32) callconv(.c) i8, "f") orelse return error.NoEntry)(args[0]))),
            2 => @as(u8, @bitCast((jitted.entry(*const fn (i32, i32) callconv(.c) i8, "f") orelse return error.NoEntry)(args[0], args[1]))),
            3 => @as(u8, @bitCast((jitted.entry(*const fn (i32, i32, i32) callconv(.c) i8, "f") orelse return error.NoEntry)(args[0], args[1], args[2]))),
            else => return error.Unsupported,
        },
        16 => switch (args.len) {
            0 => @as(u16, @bitCast((jitted.entry(*const fn () callconv(.c) i16, "f") orelse return error.NoEntry)())),
            1 => @as(u16, @bitCast((jitted.entry(*const fn (i32) callconv(.c) i16, "f") orelse return error.NoEntry)(args[0]))),
            2 => @as(u16, @bitCast((jitted.entry(*const fn (i32, i32) callconv(.c) i16, "f") orelse return error.NoEntry)(args[0], args[1]))),
            3 => @as(u16, @bitCast((jitted.entry(*const fn (i32, i32, i32) callconv(.c) i16, "f") orelse return error.NoEntry)(args[0], args[1], args[2]))),
            else => return error.Unsupported,
        },
        32 => switch (args.len) {
            0 => @as(u32, @bitCast((jitted.entry(*const fn () callconv(.c) i32, "f") orelse return error.NoEntry)())),
            1 => @as(u32, @bitCast((jitted.entry(*const fn (i32) callconv(.c) i32, "f") orelse return error.NoEntry)(args[0]))),
            2 => @as(u32, @bitCast((jitted.entry(*const fn (i32, i32) callconv(.c) i32, "f") orelse return error.NoEntry)(args[0], args[1]))),
            3 => @as(u32, @bitCast((jitted.entry(*const fn (i32, i32, i32) callconv(.c) i32, "f") orelse return error.NoEntry)(args[0], args[1], args[2]))),
            else => return error.Unsupported,
        },
        64 => switch (args.len) {
            0 => @as(u64, @bitCast((jitted.entry(*const fn () callconv(.c) i64, "f") orelse return error.NoEntry)())),
            1 => @as(u64, @bitCast((jitted.entry(*const fn (i32) callconv(.c) i64, "f") orelse return error.NoEntry)(args[0]))),
            2 => @as(u64, @bitCast((jitted.entry(*const fn (i32, i32) callconv(.c) i64, "f") orelse return error.NoEntry)(args[0], args[1]))),
            3 => @as(u64, @bitCast((jitted.entry(*const fn (i32, i32, i32) callconv(.c) i64, "f") orelse return error.NoEntry)(args[0], args[1], args[2]))),
            else => return error.Unsupported,
        },
        else => return error.Unsupported,
    };
}

/// Call the JIT'd `f` (int params, `float`-returning) and reinterpret its result's raw
/// bits as `u32`. This is the float-return counterpart to `callFBits`.
fn callFF32Bits(jitted: *const target.native.JittedModule, args: []const i32) !u32 {
    return switch (args.len) {
        0 => @bitCast((jitted.entry(*const fn () callconv(.c) f32, "f") orelse return error.NoEntry)()),
        1 => @bitCast((jitted.entry(*const fn (i32) callconv(.c) f32, "f") orelse return error.NoEntry)(args[0])),
        2 => @bitCast((jitted.entry(*const fn (i32, i32) callconv(.c) f32, "f") orelse return error.NoEntry)(args[0], args[1])),
        3 => @bitCast((jitted.entry(*const fn (i32, i32, i32) callconv(.c) f32, "f") orelse return error.NoEntry)(args[0], args[1], args[2])),
        else => return error.Unsupported,
    };
}

/// Like `callFF32Bits`, but for a `double`-returning `f`, reinterpreted as `u64`.
fn callFF64Bits(jitted: *const target.native.JittedModule, args: []const i32) !u64 {
    return switch (args.len) {
        0 => @bitCast((jitted.entry(*const fn () callconv(.c) f64, "f") orelse return error.NoEntry)()),
        1 => @bitCast((jitted.entry(*const fn (i32) callconv(.c) f64, "f") orelse return error.NoEntry)(args[0])),
        2 => @bitCast((jitted.entry(*const fn (i32, i32) callconv(.c) f64, "f") orelse return error.NoEntry)(args[0], args[1])),
        3 => @bitCast((jitted.entry(*const fn (i32, i32, i32) callconv(.c) f64, "f") orelse return error.NoEntry)(args[0], args[1], args[2])),
        else => return error.Unsupported,
    };
}

/// Verify every function in `mod` against the `low` profile (primitives only, which is
/// all the C frontend ever emits) and fail loudly on the first malformed one. A dominance
/// violation, arity mismatch, or bad terminator should be caught here, not as a JIT crash
/// or a silent wrong-answer mismatch further down the pipeline.
fn verifyModule(allocator: std.mem.Allocator, mod: *const cc.Module) !void {
    for (mod.funcs) |*nf| {
        var diags = try ir.verify.verify(allocator, &nf.func, .low);
        defer diags.deinit();
        if (!diags.ok()) {
            std.debug.print("verify failed for `{s}`: {any}\n", .{ nf.name, diags.items() });
        }
        try std.testing.expect(diags.ok());
    }
}

/// Map `mod.data`'s frontend-owned `DataObject`s to `target.native.ModuleData` for
/// `jitModuleData`. Shared by `frontendRun`/`frontendRunOpt`. `name`/`bytes` are borrowed
/// straight from `mod`, which outlives the JIT call in both callers. Each object's `relocs`
/// (empty for an ordinary global) is copied into a freshly allocated native-shaped slice.
/// Free the result with `freeModuleData`. A program with no globals builds an empty list, so
/// `jitModuleData` behaves exactly like `jitModule`'s own no-data path.
fn buildModuleData(allocator: std.mem.Allocator, objs: []const cc.DataObject) !std.ArrayList(target.native.ModuleData) {
    var list: std.ArrayList(target.native.ModuleData) = .empty;
    errdefer freeModuleData(allocator, &list);
    for (objs) |d| {
        var relocs: []target.native.backend.link.DataReloc = &.{};
        if (d.relocs.len != 0) {
            relocs = try allocator.alloc(target.native.backend.link.DataReloc, d.relocs.len);
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

/// Free the (possibly empty) reloc-slice allocations `buildModuleData` made, then the list
/// itself.
fn freeModuleData(allocator: std.mem.Allocator, list: *std.ArrayList(target.native.ModuleData)) void {
    for (list.items) |d| if (d.relocs.len != 0) allocator.free(d.relocs);
    list.deinit(allocator);
}

/// Lower `source` to IR through `cc.compileWithOpts`, so an `#include` resolver can be
/// threaded through. `pp = .{}` is the plain-C89 case most callers use. JIT every function
/// plus every global's data object, and format `f(args...)`'s return value (read at `f`'s
/// actual IR return width, zero-extended to `u64`) as decimal.
fn frontendRunPP(allocator: std.mem.Allocator, source: []const u8, args: []const i32, pp: cc.preproc.Options) ![]u8 {
    var mod = try cc.compileWithOpts(allocator, source, pp);
    defer mod.deinit(allocator);
    try verifyModule(allocator, &mod);
    const f_ir = mod.find("f") orelse return error.NoFunc;
    const rk = try returnKind(f_ir);

    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });

    var data = try buildModuleData(allocator, mod.data);
    defer freeModuleData(allocator, &data);

    var jitted = try target.native.jitModuleData(allocator, mfs.items, data.items);
    defer jitted.deinit();

    return switch (rk) {
        .int => |bits| std.fmt.allocPrint(allocator, "{d}", .{try callFBits(&jitted, args, bits)}),
        .f32 => std.fmt.allocPrint(allocator, "{d}", .{try callFF32Bits(&jitted, args)}),
        .f64 => std.fmt.allocPrint(allocator, "{d}", .{try callFF64Bits(&jitted, args)}),
    };
}

/// Like `frontendRunPP`, but runs the optimizer (mem2reg + more) on each function first.
fn frontendRunOptPP(allocator: std.mem.Allocator, source: []const u8, args: []const i32, pp: cc.preproc.Options) ![]u8 {
    var mod = try cc.compileWithOpts(allocator, source, pp);
    defer mod.deinit(allocator);
    try verifyModule(allocator, &mod);
    for (mod.funcs) |*nf| _ = try opt.optimize(allocator, &nf.func);
    try verifyModule(allocator, &mod);
    const f_ir = mod.find("f") orelse return error.NoFunc;
    const rk = try returnKind(f_ir);
    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });
    var data = try buildModuleData(allocator, mod.data);
    defer freeModuleData(allocator, &data);
    var jitted = try target.native.jitModuleData(allocator, mfs.items, data.items);
    defer jitted.deinit();
    return switch (rk) {
        .int => |bits| std.fmt.allocPrint(allocator, "{d}", .{try callFBits(&jitted, args, bits)}),
        .f32 => std.fmt.allocPrint(allocator, "{d}", .{try callFF32Bits(&jitted, args)}),
        .f64 => std.fmt.allocPrint(allocator, "{d}", .{try callFF64Bits(&jitted, args)}),
    };
}

/// `frontendRunPP`/`frontendRunOptPP` for the common case: plain C89, no `#include`
/// resolver. Kept as thin wrappers so every existing call site is untouched.
fn frontendRun(allocator: std.mem.Allocator, source: []const u8, args: []const i32) ![]u8 {
    return frontendRunPP(allocator, source, args, .{});
}
fn frontendRunOpt(allocator: std.mem.Allocator, source: []const u8, args: []const i32) ![]u8 {
    return frontendRunOptPP(allocator, source, args, .{});
}

/// Compile `source` both ways (gcc oracle and the frontend's JIT), call `f(args...)` on
/// each (an `int`-param, `ret_bits`-wide-return function), and assert their results agree
/// bit-for-bit at that width.
fn expectAgreesBits(source: []const u8, args: []const i32, ret_bits: u16) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const oracle = oracleRun(io, allocator, source, &.{}, args, ret_bits) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(oracle);

    const got = try frontendRun(allocator, source, args);
    defer allocator.free(got);

    std.testing.expectEqualStrings(oracle, got) catch |err| {
        std.debug.print("mismatch: source=\n{s}\noracle={s} got={s}\n", .{ source, oracle, got });
        return err;
    };

    const got_opt = try frontendRunOpt(allocator, source, args);
    defer allocator.free(got_opt);
    std.testing.expectEqualStrings(oracle, got_opt) catch |err| {
        std.debug.print("OPT mismatch: source=\n{s}\noracle={s} got_opt={s}\n", .{ source, oracle, got_opt });
        return err;
    };
}

/// `expectAgreesBits` for the common case: an `int` (32-bit) return, which most tests use.
/// Kept as a 2-arg wrapper so none of those call sites need editing.
fn expectAgrees(source: []const u8, args: []const i32) !void {
    return expectAgreesBits(source, args, 32);
}

/// `expectAgrees`, but oracle-compiled under `-std=gnu99`, for source using the bare
/// `typeof` spelling, which needs a GNU dialect on gcc's side (see `oracleRunGnu99`'s doc
/// comment). The frontend side is unaffected, since VCC has no `-std=` distinction at all.
fn expectAgreesGnu99(source: []const u8, args: []const i32) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const oracle = oracleRunGnu99(io, allocator, source, args, 32) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(oracle);

    const got = try frontendRun(allocator, source, args);
    defer allocator.free(got);
    std.testing.expectEqualStrings(oracle, got) catch |err| {
        std.debug.print("mismatch: source=\n{s}\noracle={s} got={s}\n", .{ source, oracle, got });
        return err;
    };

    const got_opt = try frontendRunOpt(allocator, source, args);
    defer allocator.free(got_opt);
    std.testing.expectEqualStrings(oracle, got_opt) catch |err| {
        std.debug.print("OPT mismatch: source=\n{s}\noracle={s} got_opt={s}\n", .{ source, oracle, got_opt });
        return err;
    };
}

/// Compile `source` (which defines a `float`- or `double`-returning `f(...)`) both ways and
/// assert their results agree bit-for-bit. It uses `oracleRunFloat`'s union-reinterpret
/// print on the oracle side, and `returnKind`/`callFF32Bits`/`callFF64Bits` (through
/// `frontendRun`/`frontendRunOpt`, unchanged from the int path apart from now branching on
/// `returnKind`) on the frontend side. This harness only has real callers once the frontend
/// parses float literals.
fn expectAgreesFloatKind(source: []const u8, args: []const i32, is_double: bool) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const oracle = oracleRunFloat(io, allocator, source, args, is_double) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(oracle);

    const got = try frontendRun(allocator, source, args);
    defer allocator.free(got);
    std.testing.expectEqualStrings(oracle, got) catch |err| {
        std.debug.print("mismatch: source=\n{s}\noracle={s} got={s}\n", .{ source, oracle, got });
        return err;
    };

    const got_opt = try frontendRunOpt(allocator, source, args);
    defer allocator.free(got_opt);
    std.testing.expectEqualStrings(oracle, got_opt) catch |err| {
        std.debug.print("OPT mismatch: source=\n{s}\noracle={s} got_opt={s}\n", .{ source, oracle, got_opt });
        return err;
    };
}

/// `expectAgreesFloatKind` for a `float`-returning `f`.
fn expectAgreesFloat(source: []const u8, args: []const i32) !void {
    return expectAgreesFloatKind(source, args, false);
}

/// `expectAgreesFloatKind` for a `double`-returning `f`.
fn expectAgreesDouble(source: []const u8, args: []const i32) !void {
    return expectAgreesFloatKind(source, args, true);
}

/// An `IncludeResolver.resolveFn` for `expectAgreesPPBits`. It closes over a
/// `[]const HeaderFile` via `ctx` (a pointer to the slice variable itself) and looks `name`
/// up by exact match, mirroring what the oracle side does on disk. `is_system` and
/// `includer_dir` are unused, since a flat header list needs neither `<...>` vs `"..."` nor
/// a relative-path search.
fn ppTestResolve(ctx: ?*anyopaque, name: []const u8, is_system: bool, includer_dir: ?[]const u8, next_after: ?[]const u8) cc.preproc.Error!?cc.preproc.ResolvedFile {
    _ = is_system;
    _ = includer_dir;
    _ = next_after;
    const headers: *const []const HeaderFile = @ptrCast(@alignCast(ctx.?));
    for (headers.*) |h| {
        if (std.mem.eql(u8, h.name, name)) return .{ .identity = h.name, .bytes = h.content };
    }
    return null;
}

/// Compile `source` (which `#include`s one or more of `headers`) both ways and require
/// agreement, like `expectAgreesBits`, but wiring an `#include` resolver on both sides. The
/// oracle gets every header written to its temp dir, so gcc's own `"..."` search resolves
/// them with no `-I`. The frontend gets a virtual `IncludeResolver` that looks `headers` up
/// by name in memory (see `ppTestResolve`).
fn expectAgreesPPBits(source: []const u8, headers: []const HeaderFile, args: []const i32, ret_bits: u16) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const oracle = oracleRun(io, allocator, source, headers, args, ret_bits) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(oracle);

    var headers_slice: []const HeaderFile = headers;
    const resolver: cc.preproc.IncludeResolver = .{ .ctx = @ptrCast(&headers_slice), .resolveFn = ppTestResolve };
    const pp: cc.preproc.Options = .{ .resolver = resolver };

    const got = try frontendRunPP(allocator, source, args, pp);
    defer allocator.free(got);
    std.testing.expectEqualStrings(oracle, got) catch |err| {
        std.debug.print("mismatch: source=\n{s}\noracle={s} got={s}\n", .{ source, oracle, got });
        return err;
    };

    const got_opt = try frontendRunOptPP(allocator, source, args, pp);
    defer allocator.free(got_opt);
    std.testing.expectEqualStrings(oracle, got_opt) catch |err| {
        std.debug.print("OPT mismatch: source=\n{s}\noracle={s} got_opt={s}\n", .{ source, oracle, got_opt });
        return err;
    };
}

test "constant return agrees with gcc" {
    try expectAgrees("int f(void){ return 42; }", &.{});
}
test "identity through a parameter agrees with gcc" {
    try expectAgrees("int f(int a){ return a; }", &.{5});
}
test "second parameter agrees with gcc" {
    try expectAgrees("int f(int a, int b){ return b; }", &.{ 3, 9 });
}
test "arithmetic precedence agrees with gcc" {
    try expectAgrees("int f(int a, int b){ return a + b * 2; }", &.{ 3, 4 });
}
test "parenthesized arithmetic agrees with gcc" {
    try expectAgrees("int f(int a, int b){ return (a + b) * 2; }", &.{ 3, 4 });
}
test "division and remainder agree with gcc" {
    try expectAgrees("int f(int a, int b){ return a / b + a % b; }", &.{ 17, 5 });
}
test "left-associative subtraction agrees with gcc" {
    try expectAgrees("int f(int a, int b, int c){ return a - b - c; }", &.{ 10, 3, 2 });
}
test "bitwise ops agree with gcc" {
    try expectAgrees("int f(int a, int b){ return (a & b) | (a ^ b); }", &.{ 12, 10 });
}
test "shifts agree with gcc" {
    try expectAgrees("int f(int a, int b){ return (a << 2) + (b >> 1); }", &.{ 3, 16 });
}
test "bitwise-and precedence below arithmetic agrees with gcc" {
    try expectAgrees("int f(int a, int b){ return a + b & 6; }", &.{ 5, 3 });
}
test "relational operators agree with gcc" {
    try expectAgrees("int f(int a, int b){ return a < b; }", &.{ 3, 9 });
    try expectAgrees("int f(int a, int b){ return a >= b; }", &.{ 9, 9 });
}
test "equality operators agree with gcc" {
    try expectAgrees("int f(int a, int b){ return (a == b) + (a != b); }", &.{ 4, 4 });
}
test "relational result is usable arithmetically" {
    try expectAgrees("int f(int a){ return (a > 0) * 10; }", &.{-5});
}
test "bitwise complement agrees with gcc" {
    try expectAgrees("int f(int a){ return ~a; }", &.{5});
}
test "logical not agrees with gcc" {
    try expectAgrees("int f(int a){ return !a + !!a; }", &.{0});
}
test "unary binds tighter than binary" {
    try expectAgrees("int f(int a, int b){ return -a * b; }", &.{ 3, 4 });
}
test "local variable and assignment agree with gcc" {
    try expectAgrees("int f(int a){ int b = a + 1; b = b * 2; return b; }", &.{5});
}
test "compound assignment agrees with gcc" {
    try expectAgrees("int f(int a){ int s = 0; s += a; s *= 3; return s; }", &.{7});
}
test "multiple locals agree with gcc" {
    try expectAgrees("int f(int a, int b){ int x = a * a; int y = b * b; return x + y; }", &.{ 3, 4 });
}
test "same-TU call agrees with gcc" {
    try expectAgrees("int sq(int x){ return x * x; } int f(int a){ return sq(a) + 1; }", &.{6});
}
test "call with multiple args agrees with gcc" {
    try expectAgrees("int add(int a, int b){ return a + b; } int f(int a, int b){ return add(a, b) * 2; }", &.{ 3, 4 });
}

// Function pointers. `add`'s address, taken via the bare function designator
// (`fp = add;`), and a call through the resulting pointer (`fp(1, 41)`), agree with gcc's
// own evaluation. This is the frontend lowering's gcc-differential oracle case. The host
// aarch64 JIT exercises the aarch64 `.call_indirect` isel. The x86/riscv64 isel is proven
// separately by `tests/external_linkage.zig`'s cross-arch end-to-end tests.
test "same-TU function pointer call agrees with gcc" {
    try expectAgrees("int add(int a, int b){ return a + b; } int f(void){ int (*fp)(int, int) = add; return fp(1, 41); }", &.{});
}
test "function pointer passed as a value and called through a local agrees with gcc" {
    try expectAgrees(
        "int mul(int a, int b){ return a * b; } int apply(int (*g)(int, int), int x, int y){ return g(x, y); } int f(int a, int b){ return apply(mul, a, b); }",
        &.{ 6, 7 },
    );
}

// A call-classification fix: a LOCAL function-pointer variable that shares its name with a
// file-scope FUNCTION must shadow it. This follows C's innermost-binding-wins rule, the same
// as for any other identifier. `f`'s `inner` local (pointing at `other`) shadows the
// file-scope `inner` function (which adds), so `inner(x, 3)` inside `f` must call through
// the local (`other`, a multiply: `5*3 == 15`), never the file-scope `inner` it shadows
// (which would wrongly add: `5+3 == 8`). Before this fix, the `.call` arm's `.name`
// classification checked `findFunc` before `l.lookup`, so the callee-position fast path
// direct-called the shadowed function instead. This test fails (returns 8) without the
// lookup-first fix and passes (returns 15) with it. See lower.zig's `.call` arm in
// `lowerExpr`/`typeOf`.
test "a local function-pointer variable shadowing a same-named file-scope function calls through the local, agrees with gcc" {
    try expectAgrees(
        "int inner(int a, int b){ return a + b; } int other(int a, int b){ return a * b; } int f(int x){ int (*inner)(int, int) = other; return inner(x, 3); }",
        &.{5},
    );
}

test "unreachable statement after return agrees with gcc" {
    // `lowerBlock` stops lowering a statement list once a statement terminates (returns
    // true). It silently drops the unreachable rest. It does not reject the program. This
    // matches C: gcc accepts dead code after `return` (a warning, suppressed here by -w)
    // and simply never runs it.
    try expectAgrees("int f(void){ return 1; return 2; }", &.{});
}

test "compound block agrees with gcc" {
    try expectAgrees("int f(int a){ { int b = a + 1; a = b * 2; } return a; }", &.{5});
}
test "nested-scope shadowing agrees with gcc" {
    try expectAgrees("int f(int a){ int x = a; { int x = a + 100; a = x; } return a + x; }", &.{7});
}

test "if/else agrees with gcc" {
    // The grammar requires an initializer on `int` decls (no bare `int r;`), so this uses
    // `int r = 0;` rather than the brief's `int r;`. Both branches unconditionally overwrite
    // `r` before it is read, so the result agrees with gcc either way.
    try expectAgrees("int f(int a){ int r = 0; if (a > 0) { r = 1; } else { r = -1; } return r; }", &.{-4});
    try expectAgrees("int f(int a){ int r = 0; if (a > 0) { r = 1; } else { r = -1; } return r; }", &.{9});
}
test "if without else agrees with gcc" {
    try expectAgrees("int f(int a){ int r = 0; if (a) { r = 100; } return r; }", &.{0});
}
test "if both branches return agrees with gcc" {
    try expectAgrees("int f(int a){ if (a > 0) { return a; } else { return -a; } }", &.{-6});
}

test "while loop agrees with gcc" {
    try expectAgrees("int f(int n){ int s = 0; int i = 0; while (i < n) { s += i; i += 1; } return s; }", &.{5});
}
test "while with break agrees with gcc" {
    try expectAgrees("int f(int n){ int i = 0; while (1) { if (i >= n) { break; } i += 1; } return i; }", &.{7});
}
test "while with continue agrees with gcc" {
    try expectAgrees("int f(int n){ int s = 0; int i = 0; while (i < n) { i += 1; if (i == 3) { continue; } s += i; } return s; }", &.{5});
}
test "decl inside loop body agrees with gcc" {
    // A local declared INSIDE the loop body: its alloca must be emitted in the (already-
    // terminated) entry block, not the loop body (which would grow the stack per iteration
    // and defeat mem2reg). If this fails or the optimized path diverges, allocSlot is
    // targeting the wrong block.
    try expectAgrees("int f(int n){ int s = 0; int i = 0; while (i < n) { int t = i * 2; s += t; i += 1; } return s; }", &.{5});
}

test "for loop agrees with gcc" {
    try expectAgrees("int f(int n){ int s = 0; for (int i = 0; i < n; i += 1) { s += i; } return s; }", &.{6});
}
test "for with continue agrees with gcc" {
    try expectAgrees("int f(int n){ int s = 0; for (int i = 0; i < n; i += 1) { if (i == 2) { continue; } s += i; } return s; }", &.{5});
}
test "for init var is scoped" {
    try expectAgrees("int f(int n){ int i = 100; for (int i = 0; i < n; i += 1) { } return i; }", &.{3});
}

test "do-while runs body once then tests" {
    try expectAgrees("int f(int n){ int s = 0; int i = 0; do { s += i; i += 1; } while (i < n); return s; }", &.{0});
    try expectAgrees("int f(int n){ int s = 0; int i = 0; do { s += i; i += 1; } while (i < n); return s; }", &.{4});
}

test "logical and short-circuits agree with gcc" {
    try expectAgrees("int f(int a, int b){ return a && b; }", &.{ 0, 5 });
    try expectAgrees("int f(int a, int b){ return a && b; }", &.{ 3, 0 });
    try expectAgrees("int f(int a, int b){ return a && b; }", &.{ 3, 5 });
}
test "logical or short-circuits agree with gcc" {
    try expectAgrees("int f(int a, int b){ return a || b; }", &.{ 0, 0 });
    try expectAgrees("int f(int a, int b){ return a || b; }", &.{ 0, 7 });
}
test "logical result is 0 or 1" {
    try expectAgrees("int f(int a){ return (a && 5) + (a || 2); }", &.{4});
}

test "ternary agrees with gcc" {
    try expectAgrees("int f(int a){ return a > 0 ? a : -a; }", &.{-8});
    try expectAgrees("int f(int a){ return a > 0 ? a : -a; }", &.{8});
}
test "ternary short-circuits the untaken arm" {
    try expectAgrees("int f(int a, int b){ return a ? b + 1 : b - 1; }", &.{ 0, 10 });
}

test "switch selects a case agrees with gcc" {
    try expectAgrees("int f(int a){ int r = 0; switch (a) { case 1: r = 10; break; case 2: r = 20; break; default: r = -1; } return r; }", &.{2});
    try expectAgrees("int f(int a){ int r = 0; switch (a) { case 1: r = 10; break; case 2: r = 20; break; default: r = -1; } return r; }", &.{9});
}
test "switch fallthrough agrees with gcc" {
    try expectAgrees("int f(int a){ int r = 0; switch (a) { case 1: r += 1; case 2: r += 2; break; case 3: r += 3; } return r; }", &.{1});
}
test "break in loop inside switch targets the loop-vs-switch correctly" {
    try expectAgrees("int f(int a){ int s = 0; for (int i = 0; i < 3; i += 1) { switch (a) { case 0: break; default: s += 1; } s += 10; } return s; }", &.{0});
}
test "continue inside a switch inside a loop targets the loop, not the switch" {
    try expectAgrees("int f(int a){ int s = 0; for (int i = 0; i < 3; i += 1) { switch (a) { case 0: continue; default: s += 1; } s += 10; } return s; }", &.{0});
    try expectAgrees("int f(int a){ int s = 0; for (int i = 0; i < 3; i += 1) { switch (a) { case 0: continue; default: s += 1; } s += 10; } return s; }", &.{7});
}

// Regression test: a local declared inside a bare `{ }` block nested directly under a
// `case` used to crash. The root cause (see `lower.zig`'s `allocSlot` doc) is that a
// `switch`'s dispatch ladder appends a non-terminating `if` instruction straight into the
// entry block, the same as a top-level `if` statement would. `allocSlot` unconditionally
// appended every later local's alloca to entry too, landing it after that `if`, which every
// backend treats as ending the block's live code. The case-body local's address was then
// never computed, since it was dead code after the branch, so any load or store through it
// read or wrote a garbage register.
test "switch case bare block local agrees with gcc" {
    try expectAgrees("int f(int n){ switch (n) { case 1: { int x = 42; return x; } default: return 0; } }", &.{1});
    try expectAgrees("int f(int n){ switch (n) { case 1: { int x = 42; return x; } default: return 0; } }", &.{0});
}
test "a local after a top-level if (the same allocSlot-past-if bug) agrees with gcc" {
    // The switch dispatch ladder is not the only way entry ends with a non-terminating .if:
    // a top-level `if` statement leaves one there too, so a sibling local declared after it
    // hits the identical allocSlot-appends-past-.if defect. This is the general regression.
    try expectAgrees("int f(int n){ if (n) return 1; int x = 5; return x; }", &.{0});
    try expectAgrees("int f(int n){ if (n) return 1; int x = 5; return x; }", &.{1});
}
test "switch case bare block local written then read across statements agrees with gcc" {
    try expectAgrees("int f(int n){ switch(n){ case 2: { int a = 10; int b = a + 5; return b; } default: return 0; } }", &.{2});
    try expectAgrees("int f(int n){ switch(n){ case 2: { int a = 10; int b = a + 5; return b; } default: return 0; } }", &.{0});
}
test "two switch cases each with their own block local don't alias" {
    try expectAgrees("int f(int n){ switch(n){ case 1: { int x = 7; return x; } case 2: { int y = 9; return y; } default: return 0; } }", &.{1});
    try expectAgrees("int f(int n){ switch(n){ case 1: { int x = 7; return x; } case 2: { int y = 9; return y; } default: return 0; } }", &.{2});
}
test "switch case bare block local with fall-through agrees with gcc" {
    try expectAgrees("int f(int n){ int r = 0; switch(n){ case 1: { int x = 3; r += x; } case 2: r += 100; break; default: r = -1; } return r; }", &.{1});
    try expectAgrees("int f(int n){ int r = 0; switch(n){ case 1: { int x = 3; r += x; } case 2: r += 100; break; default: r = -1; } return r; }", &.{2});
}

test "while with a short-circuit && condition agrees with gcc" {
    // Regression: lowerExpr of a short-circuit condition moves l.block to the merge block
    // where the condition value is defined. The loop's appendIf must land there, not in
    // the stale header block, or the branch uses a value its block does not dominate.
    try expectAgrees("int f(int a, int b){ int n=0; while (a && b) { n+=1; a-=1; } return n; }", &.{ 3, 5 });
    try expectAgrees("int f(int a, int b){ int n=0; while (a && b) { n+=1; a-=1; } return n; }", &.{ 0, 5 });
}
test "for with a short-circuit || condition agrees with gcc" {
    try expectAgrees("int f(int a, int b){ int s=0; for (int i=0; i<a || i<b; i+=1) { s+=1; } return s; }", &.{ 2, 4 });
}
test "do-while with a ternary condition agrees with gcc" {
    try expectAgrees("int f(int a, int b){ int n=0; do { n+=1; a-=1; } while (a > 0 ? b : 0); return n; }", &.{ 3, 1 });
    try expectAgrees("int f(int a, int b){ int n=0; do { n+=1; a-=1; } while (a > 0 ? b : 0); return n; }", &.{ 3, 0 });
}

test "mem2reg fires on a function with locals" {
    const allocator = std.testing.allocator;
    var mod = try cc.compile(allocator, "int f(int a){ int s = 0; s += a; s *= 3; return s; }");
    defer mod.deinit(allocator);
    const changed = try opt.optimize(allocator, &mod.funcs[0].func);
    try std.testing.expect(changed); // optimize reports it rewrote the IR (allocas promoted)
}

test "long return agrees with gcc" {
    // 2e9 + 1e9 + 1e9 = 4e9 overflows i32 but fits `long`/i64. This proves the typed-
    // lowering return path (and the harness's 64-bit reader) is genuinely 64-bit, not
    // truncating to i32 and wrapping around like the old untyped, always-int, lowering
    // would have.
    try expectAgreesBits("long f(int a){ long x = a; x = x + 1000000000; x = x + 1000000000; return x; }", &.{2000000000}, 64);
}

// Differential coverage for integer promotions and the usual arithmetic conversions.
// `lowerExpr` is typed via `commonType`. These tests pin that behavior against gcc for the
// mixed-type and edge-case arms.

test "char/short promote to int in arithmetic" {
    // `c + c` is done in `int` (both operands promote before the add), so 100+100 = 200
    // (fits int) rather than wrapping as a char add would (200 doesn't fit signed char).
    try expectAgrees("int f(int a){ signed char c = a; return c + c; }", &.{100});
}
test "int + long -> long" {
    // 100000 * 100000 = 1e10 overflows i32 but fits `long`: the multiply must run in the
    // common (long) type, not int, once one operand is `long`.
    try expectAgreesBits("long f(int a){ long b = a; return b * a; }", &.{100000}, 64);
}
test "unsigned comparison differs from signed" {
    // (unsigned)-1 > 5 is true (a huge unsigned value). The plain-int literal `5` still
    // converts to unsigned to match `u`'s rank under the usual arithmetic conversions, so
    // no `u`-suffixed literal is needed to exercise the signed/unsigned divergence. The
    // lexer has no integer-literal-suffix support yet, but that is a separate concern.
    try expectAgrees("int f(int a){ unsigned int u = a; return u > 5; }", &.{-1});
}
test "unsigned division uses unsigned semantics" {
    // (unsigned)-2 / 2 is a huge quotient, not -1: division must pick the unsigned IR op
    // once the common type is unsigned.
    try expectAgreesBits("unsigned int f(int a){ unsigned int u = a; return u / 2; }", &.{-2}, 32);
}
test "narrowing on assignment truncates" {
    try expectAgrees("int f(int a){ signed char c = a; return c; }", &.{300}); // 300 -> (signed char)300 = 44
}

test "wide-armed ternary widens correctly" {
    // Both ternary arms are `long`, so `commonType` is 64-bit and the result slot (which
    // `lowerExpr`'s ternary arm starts as `int_t` and widens in place once both arms are
    // known) must actually widen. Otherwise, storing `x` (a genuinely 64-bit-valued long,
    // built via `x + x` so the doubling itself runs in 64-bit space rather than relying on
    // an out-of-int-range literal, which the lexer/lowering don't type as `long` yet) into
    // a stale 32-bit slot truncates it and disagrees with gcc.
    try expectAgreesBits("long f(int a){ long x = 2000000000; x = x + x; long y = 3; long r = a > 0 ? x : y; return r; }", &.{1}, 64);
    try expectAgreesBits("long f(int a){ long x = 2000000000; x = x + x; long y = 3; long r = a > 0 ? x : y; return r; }", &.{0}, 64);
}

// Implicit conversions in the remaining contexts: compound assignment, return, and call
// arguments. These paths are already typed. These tests pin each context specifically
// against gcc, closing the one likely gap where call args or the result were assumed to be
// `int` for every callee.

test "compound assignment converts in the variable's type" {
    // `c += a` must compute in `c`'s common-with-`a` type (int, after promotion) and then
    // narrow the result back to `signed char` on store: 0 + 200 = 200, (signed char)200 =
    // -56, not 200 (which would disagree with gcc if the narrow-back-to-var-type store were
    // missing).
    try expectAgrees("int f(int a){ signed char c = 0; c += a; return c; }", &.{200});
}
test "return converts to the function type" {
    // `return a` must narrow to `signed char` (the function's declared return type), not
    // return the full `int` value: (signed char)130 = -126, ret_bits 8.
    try expectAgreesBits("signed char f(int a){ return a; }", &.{130}, 8);
}
test "argument converts to the parameter type via internal call" {
    // `g`'s parameter is `long`. `b + 1000000000` (an `int` local `b` widened to `long` by
    // the addition, then added to a genuinely `long`-valued expression) must be passed to
    // `g` as a full 64-bit value, not truncated to `int` at the call site. This is proven by
    // `g(x) = x * 2` overflowing `int` (2e9 * 2 = 4e9) if the call truncated the arg back to
    // 32 bits before `g` ever multiplies it.
    try expectAgreesBits("long g(long x){ return x * 2; } long f(int a){ long b = a; return g(b + 1000000000); }", &.{2000000000}, 64);
}

// Integer-literal suffixes (`u`/`U`/`l`/`L`/`ll`/`LL` combos) give the literal itself a
// type, which then flows through the usual arithmetic conversions like any other operand's
// type.

test "unsigned literal keeps the expression unsigned" {
    // `0u` is `unsigned int`. `a < 0u` converts `a` to unsigned per the usual arithmetic
    // conversions, so for `a = -1` it is false (a huge unsigned value is never < 0), while
    // `a < 0` (both signed) is true. The two sides of the `==` must disagree, proving the
    // literal's own suffix (not just an existing unsigned variable) drives the conversion.
    try expectAgrees("int f(int a){ return (a < 0) == (a < 0u); }", &.{-1});
}
test "long literal forces 64-bit arithmetic" {
    // `1000000000L` is `long`; `a * 1000000000L` converts `a` (int) to `long` via
    // `commonType`, so the multiply runs in 64-bit space: 5 * 1e9 = 5e9, which overflows
    // `int` (so a plain `1000000000` literal here would disagree with gcc).
    try expectAgreesBits("long f(int a){ return a * 1000000000L; }", &.{5}, 64);
}
test "unsigned long long suffix picks the widest unsigned rank" {
    try expectAgreesBits("unsigned long long f(int a){ return a + 1ull; }", &.{41}, 64);
}
test "a plain literal without a suffix still types as int" {
    // Guards against a regression where suffix parsing accidentally widens/unsigns a bare
    // literal: `100000 * 100000` (both bare `int` literals) overflows `int` and wraps,
    // exactly matching gcc's `int` overflow (UB, but both sides do the same wraparound).
    try expectAgrees("int f(int a){ return a + 100000 * 100000; }", &.{0});
}

// `sizeof`. Both `sizeof(type-name)` and `sizeof expr` yield a `size_t` (`unsigned long`)
// constant from the layout. On the LP64 host, int is 4 bytes, long is 8, char is 1, and
// short is 2.

test "sizeof basic types agrees with gcc" {
    try expectAgreesBits("unsigned long f(void){ return sizeof(int) + sizeof(long) + sizeof(char) + sizeof(short); }", &.{}, 64);
}
test "sizeof an expression uses its type not its value" {
    // `x + 1` has type `long` (x is long) regardless of `a`'s runtime value, and `sizeof` is
    // unevaluated. The expression itself is never computed, only its static type. So this
    // is `sizeof(long)` = 8 for every `a`.
    try expectAgreesBits("unsigned long f(int a){ long x = a; return sizeof(x + 1); }", &.{0}, 64);
}

// Uninitialized declarations. `int x;` (no initializer) allocates but does not store. A
// read-before-write is C undefined behavior. mem2reg's `undefZero` covers it.

test "uninitialized decl then assigned agrees with gcc" {
    try expectAgrees("int f(int a){ int x; x = a * 2; return x; }", &.{21});
}
test "uninitialized decl in a branch" {
    try expectAgrees("int f(int a){ int r; if (a > 0) { r = a; } else { r = -a; } return r; }", &.{-5});
}

// A fix for C11 6.5.7p3: `<<`/`>>` promote each operand independently, and the result type
// is the promoted left operand. The right operand's type must not leak into the left
// operand's conversion or the result. Previously `.binary` routed shifts through
// `commonType(lhs, rhs)` like every other binary op, so a signed left operand mixed with an
// unsigned right operand silently became a logical (unsigned) shift.

test "signed right shift stays arithmetic regardless of count type" {
    try expectAgreesBits("int f(int a){ unsigned int b = 1; return a >> b; }", &.{-1}, 32); // -1, not 2147483647
    try expectAgreesBits("int f(int a){ long b = 2; return a >> b; }", &.{-8}, 32); // -2
}
test "shift result width is the promoted left operand not the count" {
    try expectAgreesBits("int f(int a){ long n = 1; unsigned int x = a; return (x << n) != 0; }", &.{-1}, 32);
}

// `CType` becomes a union (int, ptr, or array). Declarators parse `*` and `[N]`. `sizeof`
// handles pointers and arrays via `sizeInBytes`. This is behavior-preserving for all-scalar
// programs, since every scalar CType is still the `.int` case, with byte-identical IR.

test "sizeof pointer and array" {
    try expectAgreesBits("unsigned long f(void){ return sizeof(int*) + sizeof(char*); }", &.{}, 64); // 8+8
}
test "declaring a pointer variable compiles and stays int-behavior" {
    try expectAgrees("int f(int a){ int x = a; int *p; return x; }", &.{5}); // p unused, x path unchanged
}

// A pointer or array operand reaching an arithmetic path (promote/commonType) used to
// panic (`std.debug.assert(self.isInt())`), since the parser accepts `int *p` declarators
// with no semantic gate. Guards in `lower.zig` now fail closed with `error.Unsupported`
// instead. This must be an error return, not a crash, since a panic would abort the test
// binary rather than surface as `expectError`.
test "pointer in arithmetic fails cleanly (not a panic)" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int *p; return -p; }"));
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int *p; int *q; return p + q + 0; }"));
}

// The lvalue/address model: `&`, `*`, and assignment through a pointer. `&x` forces `x`'s
// alloca to escape (address-taken), so mem2reg will not promote it. These tests still go
// through `expectAgreesBits`, which runs both the raw and mem2reg-optimized paths, so the
// load/store through the pointer itself, not just x's own alloca, is what these tests check
// survives optimization unchanged.

test "address-of and deref round-trip" {
    try expectAgreesBits("int f(int a){ int x = a; int *p = &x; return *p; }", &.{42}, 32);
}
test "assign through a pointer" {
    try expectAgreesBits("int f(int a){ int x = a; int *p = &x; *p = *p + 10; return x; }", &.{5}, 32); // 15
}
test "pointer to pointer" {
    try expectAgreesBits("int f(int a){ int x = a; int *p = &x; int **pp = &p; **pp = 99; return x; }", &.{0}, 32); // 99
}

// Pointer arithmetic (scaled by `sizeof(pointee)`) and pointer comparison, replacing the
// earlier `error.Unsupported` guards in `.binary`/`.compare` for pointer operands. `p + 0`
// and comparison prove `lowerPtrArith` and the pointer `icmp` path without needing an
// array. Meaningful `p + n` indexing and `q - p` need two adjacent objects, so those tests
// live further down, once arrays exist.

test "pointer plus zero derefs to the same object" {
    try expectAgreesBits("int f(int a){ int x = a; int *p = &x; return *(p + 0) + 1; }", &.{7}, 32); // 8
}
test "pointer equality" {
    try expectAgreesBits("int f(int a){ int x = a; int y = a; int *p = &x; int *q = &x; int *r = &y; return (p == q) + (p == r); }", &.{0}, 32); // 1+0 = 1
}
test "int plus pointer commutes and derefs to the same object" {
    try expectAgreesBits("int f(int a){ int x = a; int *p = &x; return *(0 + p) + 1; }", &.{7}, 32); // 8
}
test "pointer times int fails cleanly (not a panic)" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int *p; return p * 2; }"));
}

// A fix for an ARRAY-typed rvalue reaching `.binary`/`.compare` (via `*(&arr)`, which
// yields `ty=.array` since it derefs a pointer-to-array, not a pointer-to-element). This
// used to slip past the `.ptr`-only diversion and panic in the same
// `std.debug.assert(self.isInt())` as the pointer case above. Guards now fail closed with
// `error.Unsupported` for any non-int, non-pointer-handled operand. This must be an error
// return, not a crash.
//
// `(&arr)[0] + 1` used to be a third case here too, but it is not actually an array-typed
// rvalue reaching `.binary` at all. `(&arr)[0]`, unlike `*(&arr)`, DOES decay:
// `lowerExpr`'s `.index` arm explicitly decays an array-typed result to a pointer. So `+ 1`
// is ordinary pointer arithmetic (`ty=.ptr`), and only the function's implicit
// `return`-conversion of that pointer back to `int` was ever unsupported. A later fix
// (`L.convertTo` int<->pointer) closes exactly that gap, so this source now compiles
// cleanly. This matches real C, which allows, with a warning, an integer-from-pointer
// conversion on return. It is no longer a case this test can assert `error.Unsupported`
// for.
test "array rvalue in arithmetic fails cleanly (not a panic)" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int arr[2]; return *(&arr) + 1; }"));
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int arr[2]; return *(&arr) == 0; }"));
}

// One-dimensional arrays: declaration (`int arr[N]` allocas N*elemsize), array-to-pointer
// decay (an array name as an rvalue is its address, typed pointer-to-elem, though `&arr[i]`
// and `sizeof` do not decay, since they resolve the array's own address/type), `arr[i]` indexing
// (read via a load through `lowerAddr`'s `.index` arm, write via the same address through
// the generalized assignment), and array parameters. `int p[N]`, and bare `int p[]` too,
// decay to `int *p` at the declarator level, in `parseParams`. The sum-loop test is the real
// integration point: an address-taken array (mem2reg must keep it in memory) alongside
// scalar loop variables (which mem2reg does promote) in the same function.

test "array write then read" {
    try expectAgreesBits("int f(int a){ int arr[3]; arr[0] = a; arr[1] = a + 1; arr[2] = a + 2; return arr[0] + arr[1] + arr[2]; }", &.{10}, 32); // 33
}
test "array sum loop" {
    try expectAgreesBits("int f(int n){ int arr[5]; int i; for (i = 0; i < 5; i = i + 1) { arr[i] = i * i; } int s = 0; for (i = 0; i < n; i = i + 1) { s = s + arr[i]; } return s; }", &.{4}, 32); // 0+1+4+9=14
}
test "array parameter decays to pointer (write through it)" {
    try expectAgreesBits("int g(int *p){ p[0] = 100; return p[1]; } int f(int a){ int arr[2]; arr[1] = a; return g(arr) + arr[0]; }", &.{7}, 32); // g writes arr[0]=100, returns arr[1]=7; +arr[0]=100 -> 107
}

// Empty-bracket array parameters. `int p[]` parses as an array declarator with `len == 0`
// (the "unsized" sentinel), same as `int p[N]`. Param decay in `parseParams` does not care
// about `len`, so it turns into `int *p` with no new decay code. A LOCAL array with
// `len == 0` (`int a[];`) has no sensible storage to reserve and is invalid C, so it must
// fail closed with `error.Unsupported` in `lowerStmt`'s `.decl` arm.
test "empty-bracket array parameter decays to pointer" {
    // Same as the int *p param test above, but declared int p[].
    try expectAgreesBits("int g(int p[]){ p[0] = 100; return p[1]; } int f(int a){ int arr[2]; arr[1] = a; return g(arr) + arr[0]; }", &.{7}, 32); // 107
}
test "unsized local array is rejected cleanly" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int a[]; a[0] = 1; return a[0]; }"));
}
test "pointer difference is element-scaled (long result, no cast)" {
    // q - p over adjacent array elements = 1 (element-scaled), returned as long (ptrdiff_t).
    try expectAgreesBits("long f(int a){ int arr[2]; int *p = &arr[0]; int *q = &arr[1]; return q - p; }", &.{0}, 64); // 1
}
test "p + n indexes the next element" {
    try expectAgreesBits("int f(int a){ int arr[3]; arr[0]=a; arr[1]=a+5; int *p = &arr[0]; return *(p + 1); }", &.{10}, 32); // 15
}

// Multi-dimensional arrays. `int m[2][3]` wraps innermost-first at the declarator
// (`array{len=2, elem=array{len=3, elem=int}}`). `m[i][j]` chains two `.index` nodes in the
// parser. `lowerExpr`'s `.index` rvalue arm decays `m[i]` (an array-typed element, since `m`
// is 2-D) to a pointer instead of loading it, exactly like the `.name` array-decay, so the
// outer `[j]` scales by `sizeof(int)` against `&m[i]` rather than loading garbage.
// `lowerAddr`'s existing `.index` arm and `CType.sizeInBytes`/`irType` are already recursive
// and needed no change for this to fall out.
test "2-D array write then read" {
    try expectAgreesBits("int f(int a){ int m[2][3]; m[0][0] = a; m[1][2] = a + 5; return m[0][0] + m[1][2]; }", &.{10}, 32); // 25
}
// The real integration test: row-major addressing. `m[i][j]`'s address is
// `base(m) + i*sizeof(int[3]) + j*sizeof(int)` = `base + i*12 + j*4`. This can only agree
// with gcc if that scaling is exactly right. A wrong wrap order or a missing decay would
// scale `i` by 4 instead of 12, or read through a bad address for the outer index.
test "2-D array nested loops (row-major)" {
    try expectAgreesBits("int f(int n){ int m[3][3]; int i; int j; for (i = 0; i < 3; i = i + 1) { for (j = 0; j < 3; j = j + 1) { m[i][j] = i * 10 + j; } } return m[n][n] + m[0][2]; }", &.{2}, 32); // m[2][2]=22 + m[0][2]=2 = 24
}
test "sizeof a 2-D array does not decay" {
    // sizeof(int[2][3]) = 2*3*4 = 24 bytes, not 8 (a decayed pointer). sizeof never decays
    // its operand, whether that operand is a type-name or, as here, an array variable.
    try expectAgreesBits("unsigned long f(void){ int m[2][3]; return sizeof(m); }", &.{}, 64); // 24
}
test "sizeof(int[2][3]) type-name form does not decay" {
    try expectAgreesBits("unsigned long f(void){ return sizeof(int[2][3]); }", &.{}, 64); // 24
}

// A constant-expression array dimension, including `sizeof`. `void*` is now a first-class
// type (previously a gap here), so it is used directly rather than `char*` standing in for
// it.
test "constant-expression array dimension with sizeof agrees with gcc" {
    try expectAgrees("int f(void){ char x[12*sizeof(int) - 5*sizeof(void*)]; return sizeof(x); }", &.{}); // 12*4 - 5*8 = 8
}
test "a simple arithmetic array dimension agrees with gcc" {
    try expectAgrees("int f(void){ int a[2*3]; return sizeof(a)/sizeof(int); }", &.{}); // 6
}
test "a shift in an array dimension agrees with gcc" {
    try expectAgrees("int f(void){ int a[1<<3]; return sizeof(a)/sizeof(int); }", &.{}); // 8
}
test "constant-expression dimensions on both axes of a 2-D array agree with gcc" {
    try expectAgrees("int f(void){ int a[2+1][4/2]; return sizeof(a)/sizeof(int); }", &.{}); // 6
}
test "a non-constant array dimension is rejected, not silently accepted as a VLA" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(int n){ int a[n]; return 0; }"));
}
test "3-D array write then read (dimension loop generalizes)" {
    try expectAgreesBits("int f(int a){ int c[2][2][2]; c[0][0][0] = a; c[1][1][1] = a + 3; return c[0][0][0] + c[1][1][1]; }", &.{4}, 32); // 11
}
test "unsized local 2-D array is rejected cleanly" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "int f(void){ int a[][3]; a[0][0] = 1; return a[0][0]; }"));
}

// Struct/union definitions, C-exact layout, struct variables, and sizeof. No member access
// yet. These only exercise definition, layout, sizeof, and a struct local declared
// uninitialized, without disturbing ordinary scalar lowering.
test "sizeof a struct with padding" {
    // struct { char c; int i; } -> c@0, i@4 (aligned up to int's 4-byte alignment,
    // introducing 3 bytes of padding), size 8. Matches gcc's layout exactly.
    try expectAgreesBits("struct S { char c; int i; }; unsigned long f(void){ return sizeof(struct S); }", &.{}, 64); // 8
}
test "sizeof a struct of two ints" {
    // No padding needed (both fields already 4-aligned): x@0, y@4, size 8.
    try expectAgreesBits("struct P { int x; int y; }; unsigned long f(void){ return sizeof(struct P); }", &.{}, 64); // 8
}
test "struct variable declares (unused) without disturbing scalars" {
    try expectAgreesBits("struct P { int x; int y; }; int f(int a){ struct P p; int r = a; return r; }", &.{5}, 32);
}

// Member access (`.`/`->`), struct pointers, and whole-struct assignment: the core usable
// struct feature. `lowerAddr`'s `.member` arm resolves a field's address exactly like
// `.index` resolves an element's, so these exercise it end to end: plain `.` read/write,
// `->` through a struct pointer parameter, offset correctness against gcc's own padding
// layout, a whole-struct blob copy, and `[]`/`.` chaining (array of structs).
test "struct member read/write" {
    try expectAgreesBits("struct P { int x; int y; }; int f(int a){ struct P p; p.x = a; p.y = a + 10; return p.x + p.y; }", &.{5}, 32); // 20
}
test "struct pointer arrow" {
    try expectAgreesBits("struct P { int x; int y; }; int g(struct P *q){ q->x = 100; return q->y; } int f(int a){ struct P p; p.y = a; return g(&p) + p.x; }", &.{7}, 32); // 107
}
test "struct padding: char then int member" {
    try expectAgreesBits("struct S { char c; int i; }; int f(int a){ struct S s; s.c = a; s.i = a * 100; return s.c + s.i; }", &.{3}, 32); // 303
}
test "whole struct copy" {
    try expectAgreesBits("struct P { int x; int y; }; int f(int a){ struct P p; p.x = a; p.y = a + 1; struct P q; q = p; q.x = q.x + 100; return q.x + q.y + p.x; }", &.{5}, 32); // 116
}
test "whole-struct copy of nested struct field does not clobber the following field" {
    // struct S is 4 bytes. O = { struct S a; int b; } packs b at offset 4. Copying s into o.a
    // must write only o.a's 4 bytes, leaving o.b intact.
    try expectAgreesBits("struct S { int x; }; struct O { struct S a; int b; }; int f(int v){ struct S s; struct O o; s.x = v; o.b = 99; o.a = s; return o.b + o.a.x; }", &.{7}, 32); // 99 + 7 = 106
}
test "whole-struct copy of an odd-sized struct fills exactly its bytes" {
    // struct T is 12 bytes. The copy leaves a trailing sentinel int right after it untouched.
    try expectAgreesBits("struct T { int a; int b; int c; }; struct W { struct T t; int guard; }; int f(int v){ struct T src; struct W w; src.a = v; src.b = v + 1; src.c = v + 2; w.guard = 77; w.t = src; return w.guard + w.t.a + w.t.b + w.t.c; }", &.{3}, 32); // 77 + 3 + 4 + 5 = 89
}
test "array of structs" {
    try expectAgreesBits("struct P { int x; int y; }; int f(int a){ struct P arr[2]; arr[0].x = a; arr[1].y = a + 1; return arr[0].x + arr[1].y; }", &.{10}, 32); // 21
}
// A fix for address striding over a struct array or pointer: it must use the storage size
// (`ceil(size/8)*8`, what `irType`'s alloca actually reserves per element), not the exact C
// `sizeInBytes`. The two diverge whenever a struct's exact size is not a multiple of 8.
// `struct P` above is exactly 8 bytes, so it never exposed the bug. These use 4- and 12-byte
// structs, rounding to 8 and 16, to catch element overlap and whole-struct-copy clobber.
test "array of odd-sized structs does not overlap elements" {
    // struct is 4 bytes (blob rounds to 8). arr[i] must be spaced by the storage size, not 4.
    try expectAgreesBits("struct S { int x; }; int f(int a){ struct S arr[3]; arr[0].x = a; arr[1].x = a + 1; arr[2].x = a + 2; return arr[0].x + arr[1].x + arr[2].x; }", &.{10}, 32); // 33
}
test "whole-struct copy between odd-sized array elements does not clobber neighbor" {
    // struct is 12 bytes (blob rounds to 16). Copying arr[0] into arr[1] must stay within arr[1].
    try expectAgreesBits("struct T { int a; int b; int c; }; int f(int v){ struct T arr[2]; arr[0].a = v; arr[0].b = v + 1; arr[0].c = v + 2; arr[1].a = 0; arr[1].b = 0; arr[1].c = 0; arr[1] = arr[0]; return arr[1].a + arr[1].b + arr[1].c + arr[0].a; }", &.{5}, 32); // (5+6+7)+5 = 23
}
test "pointer walk over odd-sized structs strides by storage size" {
    try expectAgreesBits("struct S { int x; }; int f(int a){ struct S arr[3]; struct S *p; p = arr; p[0].x = a; p[1].x = a + 4; p[2].x = a + 8; return arr[0].x + arr[1].x + arr[2].x; }", &.{1}, 32); // 1+5+9 = 15
}
test "bare struct rvalue fails closed" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "struct P { int x; }; int f(void){ struct P p; p; return 0; }"));
    try std.testing.expectError(error.Unsupported, cc.compile(allocator, "struct P { int x; }; int f(void){ struct P p; return !p; }"));
}

// Incomplete or forward struct types. A tag can be forward-declared (`struct N;`),
// referenced through a pointer before, or without ever, completing it, and completed in
// place by a later body. Every earlier pointer or alias to it then sees the completion.
// `struct N a; struct N b;` below, not the comma-list `struct N a, b;`, since a
// comma-separated LOCAL declarator list is a separate, unrelated parser gap. Only global
// decls support a declarator list today.
test "forward-declared struct, self-referential via pointer, completed in place" {
    try expectAgrees(
        \\struct N;
        \\struct N { int v; struct N *next; };
        \\int f(void){
        \\  struct N a;
        \\  struct N b;
        \\  a.v = 40;
        \\  b.v = 2;
        \\  a.next = &b;
        \\  return a.v + a.next->v;
        \\}
    , &.{});
}
test "pointer to a never-completed forward-declared struct compiles and runs" {
    // `p != 0` (a pointer-vs-int-literal compare) is a separate, unrelated gap this
    // frontend does not support yet, since `lowerExpr`'s `.compare` arm only handles
    // ptr-vs-ptr or int/float-vs-int/float. `!p` (logical not, which zeroes against the
    // operand's own type) exercises the same "pointer to Opaque, never completed, still a
    // valid object" point without hitting that gap.
    try expectAgreesBits(
        \\struct Opaque;
        \\int f(struct Opaque *p){ return !p; }
    , &.{0}, 32);
}
test "struct completed in place is visible through an earlier typedef alias" {
    try expectAgreesBits(
        \\struct S;
        \\typedef struct S Alias;
        \\struct S { int x; };
        \\int f(void){ Alias a; a.x = 42; return sizeof(a); }
    , &.{}, 64);
}
test "by-value use of an incomplete struct fails closed" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.IncompleteType, cc.compile(allocator, "struct Inc; int f(void){ struct Inc x; return 0; }"));
}
test "redefining an already-complete struct tag fails closed" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.StructRedefinition, cc.compile(allocator, "struct S { int x; }; struct S { int x; int y; }; int f(void){ return 0; }"));
}

// Anonymous struct/union. A tagless `struct { ... }`/`union { ... }` is itself a complete
// type, most commonly seen bound straight to a name by a typedef, for example glibc's
// `typedef struct { int __val[2]; } __fsid_t;`. The definition has no tag, so it is never
// registered in the by-name tag table. It is reachable only through the `CType` this parse
// produced. (Named `my_fsid_t`, not glibc's own `__fsid_t`, since `expectAgreesBits`'s gcc
// oracle also `#include`s `<stdio.h>` for `printf`, which pulls in glibc's own `__fsid_t`
// typedef. Reusing that exact reserved name would make gcc reject the second, unrelated
// declaration as conflicting. This is a test-harness artifact, not anything about VCC's
// anonymous-struct support.)
test "anonymous struct typedef" {
    try expectAgreesBits(
        \\typedef struct { int __val[2]; } my_fsid_t;
        \\int f(void){ my_fsid_t x; x.__val[0] = 42; return x.__val[0]; }
    , &.{}, 32);
}
test "anonymous union typedef" {
    try expectAgreesBits(
        \\typedef union { int i; char c[4]; } U;
        \\int f(void){ U u; u.i = 42; return u.i; }
    , &.{}, 32);
}
// Two anonymous struct typedefs with different fields must not collide with each other.
// Each gets its own, independently laid-out `StructDef`, since there is no shared tag to
// confuse them.
test "two different anonymous struct typedefs do not collide" {
    try expectAgreesBits(
        \\typedef struct { int x; } A;
        \\typedef struct { int x; int y; int z; } B;
        \\int f(void){ A a; B b; a.x = 1; b.x = 2; b.y = 3; b.z = 4; return a.x + b.x + b.y + b.z; }
    , &.{}, 32); // 1+2+3+4 = 10
}
// `sizeof(S)` naming a bare typedef (not a `struct`/`union`-prefixed type-name) hits a
// separate, unrelated parser gap: `sizeof(`'s type-vs-expr disambiguation only recognizes a
// following type-specifier keyword (`startsTypeSpec`), not a typedef-name identifier. The
// same gap would hit `sizeof(Point)` for `typedef struct P Point;` today, nothing specific
// to anonymous structs. `sizeof(a-variable-of-that-type)` (`sizeof_expr`, not
// `sizeof_type`) sidesteps it while still proving the anonymous struct's layout matches
// gcc's.
test "sizeof an anonymous struct typedef" {
    try expectAgreesBits(
        \\typedef struct { char c; int i; } S;
        \\unsigned long f(void){ S s; return sizeof(s); }
    , &.{}, 64); // 8 (padding after char)
}

// `typedef`. A file-scope `typedef <type> <name>;` registers `name` as an alias for
// `<type>`. The parser then resolves `name` as a type wherever a type-spec is expected (the
// classic typedef-name-vs-identifier ambiguity, resolved by `parser.zig`'s `typedefs`
// table). One test aliases a builtin (64-bit math through the alias). One aliases a struct
// tag (member access through the alias, proving it resolves to the full struct type, not
// just a name).
test "typedef of a builtin type" {
    try expectAgreesBits("typedef unsigned long u64; u64 f(int a){ u64 x = a; return x * 1000000000; }", &.{5}, 64); // 5e9
}
test "typedef of a struct" {
    try expectAgreesBits("struct P { int x; int y; }; typedef struct P Point; int f(int a){ Point p; p.x = a; p.y = a + 1; return p.x + p.y; }", &.{10}, 32); // 21
}
// A fix: a parameter or local can shadow a file-scope typedef-name. `parseStmt`'s
// typedef-name precheck must not route a shadowing use to declaration parsing just because
// the name is a registered typedef (see the two-token-lookahead comment in `parseStmt`).
test "variable shadowing a typedef name parses as an expression" {
    // `myint` is a typedef, but also a parameter name; `myint = 5;` must parse as an assignment.
    try expectAgreesBits("typedef int myint; int f(int myint){ myint = myint + 5; return myint; }", &.{10}, 32); // 15
}
// `enum`: named int constants, auto-incrementing from 0, plus `enum Tag` as an int type. An
// ordinary parameter (`a`) alongside enum constants (`RED`/`GREEN`/`BLUE`) checks that the
// parse-time substitution does not hijack unrelated identifiers.
test "enum constants" {
    try expectAgreesBits("enum Color { RED, GREEN, BLUE }; int f(int a){ return RED + GREEN + BLUE + a; }", &.{10}, 32); // 0+1+2+10=13
}
// Explicit values (`A = 5`) restart the auto-increment from THAT value (`B` = 6, not 1), and a
// later explicit value (`C = 20`) overrides the running count again.
test "enum with explicit values" {
    try expectAgreesBits("enum E { A = 5, B, C = 20 }; int f(void){ return A + B + C; }", &.{}, 32); // 5+6+20=31
}
// A fix: a parameter or local can shadow an `enum` constant of the same name. `parseAtom`'s
// parse-time substitution must check `local_names` first (mirrors the existing
// typedef-shadowing fix above) rather than always replacing the identifier with the
// constant.
test "parameter shadows enum constant of the same name" {
    try expectAgreesBits("enum Color { RED, GREEN, BLUE }; int f(int GREEN){ return GREEN + BLUE; }", &.{5}, 32); // param GREEN=5, enum BLUE=2 -> 7
}
test "local shadows enum constant of the same name" {
    try expectAgreesBits("enum E { X, Y }; int f(int a){ int X; X = a + 3; return X + Y; }", &.{4}, 32); // local X=7, enum Y=1 -> 8
}
test "unshadowed enum constant still substitutes" {
    try expectAgreesBits("enum E { P = 10, Q }; int f(void){ return P + Q; }", &.{}, 32); // 10 + 11 = 21
}

// `union`: all members at offset 0, size is the max member size. Member access (`.`/`->`)
// and struct pointers work identically, since unions reuse the struct machinery with
// is_union=true and offset-0 fields. Two tests verify the defining feature: aliasing (write
// one member, read another, same bytes) and sizeof (union size is the max member).
test "union members alias the same storage" {
    try expectAgreesBits("union U { int i; int j; }; int f(int a){ union U u; u.i = a; return u.j; }", &.{42}, 32); // 42 (same storage)
}
test "sizeof a union is its largest member" {
    try expectAgreesBits("union U { char c; long l; }; unsigned long f(void){ return sizeof(union U); }", &.{}, 64); // 8
}

// Bitfields: parse `:width`, ABI-correct LSB-first layout, and shift/mask read plus
// read-modify-write write codegen. ABI correctness is the gate. Every `sizeof`/offset test
// below is gcc-differential (the oracle compiles the same struct), so a wrong bit layout
// that changes the struct size is caught, not silently accepted.
test "sizeof struct of three adjacent bitfields matches gcc" {
    // a:5, b:3, c:24 all pack into one 32-bit int unit -> sizeof 4.
    try expectAgreesBits("struct S { int a:5; int b:3; int c:24; }; unsigned long f(void){ return sizeof(struct S); }", &.{}, 64);
}
test "sizeof single 24-bit bitfield matches gcc (glibc _flags2 case)" {
    try expectAgreesBits("struct T { int f:24; }; unsigned long f(void){ return sizeof(struct T); }", &.{}, 64);
}
test "mixed bitfield and plain fields sizeof matches gcc" {
    try expectAgreesBits("struct M { char x; int b:4; char y; }; unsigned long f(void){ return sizeof(struct M); }", &.{}, 64);
}
test "bitfield offset of trailing plain field matches gcc" {
    // offsetof(struct M, y) == 2: the int bitfield `b` lives in byte 1, `y` resumes at byte 2.
    try expectAgreesBits("struct M { char x; int b:4; char y; }; unsigned long f(void){ struct M m; return (unsigned long)((char*)&m.y - (char*)&m); }", &.{}, 64);
}
test "zero-width bitfield forces next unit, sizeof matches gcc" {
    try expectAgreesBits("struct Z { int a:5; int :0; int b:3; }; unsigned long f(void){ return sizeof(struct Z); }", &.{}, 64); // 8
}
test "unnamed padding bitfield sizeof matches gcc" {
    try expectAgreesBits("struct P { int a:5; int :3; int b:4; }; unsigned long f(void){ return sizeof(struct P); }", &.{}, 64); // 4
}
test "short bitfield that straddles a unit boundary matches gcc" {
    try expectAgreesBits("struct U { char c; short s:9; }; unsigned long f(void){ return sizeof(struct U); }", &.{}, 64); // 4
}
test "unsigned bitfield read/write round-trip matches gcc" {
    // a:5=21, b:3=5 -> 26, and neither corrupts the other.
    try expectAgreesBits("struct S { unsigned a:5; unsigned b:3; }; int f(void){ struct S s; s.a = 21; s.b = 5; return s.a + s.b; }", &.{}, 32); // 26
}
test "signed bitfield sign-extends on read (15 -> -1) matches gcc" {
    try expectAgreesBits("struct S { int x:4; }; int f(void){ struct S s; s.x = 15; return s.x; }", &.{}, 32); // -1
}
test "adjacent bitfields written independently do not corrupt each other" {
    try expectAgreesBits("struct S { unsigned a:5; unsigned b:3; }; int f(void){ struct S s; s.a = 1; s.b = 0; s.a = 30; return s.a * 10 + s.b; }", &.{}, 32); // 300
}
test "wider signed bitfield truncates to its width matches gcc" {
    // 200 into a 4-bit field keeps the low 4 bits (0b1000 = 8, signed -> -8).
    try expectAgreesBits("struct S { int x:4; }; int f(int v){ struct S s; s.x = v; return s.x; }", &.{200}, 32);
}
test "bitfield mixed with a full-width neighbor int field matches gcc" {
    try expectAgreesBits("struct S { int a:5; int b:3; int full; }; int f(int v){ struct S s; s.a = 7; s.b = 2; s.full = v; return s.a + s.b + s.full; }", &.{100}, 32); // 109
}
test "bitfield compound assign matches gcc" {
    try expectAgreesBits("struct S { unsigned a:5; unsigned b:3; }; int f(void){ struct S s; s.a = 4; s.b = 1; s.a += 3; return s.a * 10 + s.b; }", &.{}, 32); // 71
}
test "bitfield increment wraps within its width matches gcc" {
    try expectAgreesBits("struct S { unsigned a:2; }; int f(void){ struct S s; s.a = 3; s.a++; return s.a; }", &.{}, 32); // 0 (wraps mod 4)
}

// A fix: `&s.bf` on a bitfield has no address of its own. C forbids it (gcc: "cannot take
// address of bit-field"). Fails closed with `error.BitfieldAddress` instead of silently
// yielding the storage unit's address. A non-bitfield member on the same struct must still
// take its address normally, since the guard is per-field, not per-struct.
test "address-of a bitfield member is rejected" {
    try std.testing.expectError(error.BitfieldAddress, cc.compile(std.testing.allocator, "struct S { int a:5; }; int f(void){ struct S s; int *p = &s.a; return 0; }"));
}
test "address-of a plain member on a struct that also has a bitfield still works" {
    try expectAgreesBits("struct S { int a:5; int plain; }; int f(void){ struct S s; s.plain = 7; int *p = &s.plain; return *p; }", &.{}, 32); // 7
}
test "address-of a plain member on an ordinary struct still works" {
    try expectAgreesBits("struct S { int x; int y; }; int f(void){ struct S s; s.y = 9; int *p = &s.y; return *p; }", &.{}, 32); // 9
}

// A fix: a NAMED zero-width bitfield (`int a:0;`) is invalid C. A zero-width bitfield must
// be unnamed, since it is purely an alignment break, not an accessible field. Fails closed
// with `error.BitfieldZeroNamed`. The unnamed form (`int :0;`) is unaffected and must still
// work as the alignment break it always was.
test "named zero-width bitfield is rejected" {
    try std.testing.expectError(error.BitfieldZeroNamed, cc.compile(std.testing.allocator, "struct S { int a:0; }; int f(void){ return 0; }"));
}
test "unnamed zero-width bitfield alignment break still works" {
    try expectAgreesBits("struct S { int a:5; int :0; int b:5; }; unsigned long f(void){ return sizeof(struct S); }", &.{}, 64); // 8
}

// `_Bool`: a 1-byte integer type whose only values are 0 and 1. Converting any scalar to
// `_Bool` is a compare-nonzero (C11 6.3.1.2), never a truncation. `_Bool b = 5;` gives
// `b == 1`, not `b == (5 & 0xff)`. Every case here is gcc-differential, so a wrong
// truncating conversion would be caught, not silently accepted.
test "_Bool from a nonzero int is 1" {
    try expectAgreesBits("int f(void){ _Bool b = 5; return b; }", &.{}, 32); // 1
}
test "_Bool from zero is 0" {
    try expectAgreesBits("int f(void){ _Bool b = 0; return b; }", &.{}, 32); // 0
}
test "_Bool from a negative int is 1" {
    try expectAgreesBits("int f(void){ _Bool b = -3; return b; }", &.{}, 32); // 1
}
test "_Bool from 256 is 1 (proves a nonzero test, not a byte truncation)" {
    try expectAgreesBits("int f(void){ _Bool b = 256; return b; }", &.{}, 32); // 1, not 0
}
test "sizeof(_Bool) is 1" {
    try expectAgreesBits("int f(void){ return sizeof(_Bool); }", &.{}, 32); // 1
}
test "_Bool from a non-null pointer is 1" {
    try expectAgreesBits("int f(void){ int x; int *p = &x; _Bool b = p; return b; }", &.{}, 32); // 1
}
test "_Bool struct field round-trips a nonzero write as 1" {
    try expectAgreesBits("struct S { _Bool flag; }; int f(void){ struct S s; s.flag = 7; return s.flag; }", &.{}, 32); // 1
}
test "_Bool composes with const/volatile and as a pointer target" {
    try expectAgreesBits("int f(void){ const _Bool b = 9; _Bool *p = 0; return b; }", &.{}, 32); // 1
}

// This is a hand-written mirror of the real, reduced glibc `struct _IO_FILE` neighborhood.
// It proves that every earlier struct feature above composes in one translation unit. Tags
// are renamed away from glibc's own reserved spellings (`myio_file` not `_IO_FILE`, and so
// on), because `expectAgreesBits`'s gcc oracle wraps the source in `#include <stdio.h>`,
// which already defines `struct _IO_FILE`, `FILE`, and every other glibc-reserved name
// below. Reusing them would make gcc reject the second, unrelated declaration as
// conflicting. This is a test-harness artifact, not anything about VCC's own support (see
// the `my_fsid_t` precedent above). `void *` fields become `char *`, and `sizeof(void*)`
// becomes `sizeof(char*)`, for the same reason as an earlier test above: this frontend had
// no `void` type at the time this test was written, and on the LP64 host both are 8 bytes,
// so the sizeof comparison against gcc still holds.
//
// Five features, all in the one struct:
//   Incomplete/forward types: `struct myio_marker;`/`myio_codecvt;`/`myio_wide_data;` are
//     forward-declared and never completed, only ever referenced through a pointer. `struct
//     myio_file;` is forward-declared, aliased by `typedef struct myio_file myio_FILE;`
//     before its body, then completed in place. The alias and the self-pointer `struct
//     myio_file *chain;` both resolve to the completed definition.
//   Anonymous struct/union: `mbstate_alt`/`fsid_alt` are tagless typedefs, declared and used
//     alongside the named-tag struct.
//   Constant-expression array dimension: `char unused2[15*sizeof(int) - 4*sizeof(char*) -
//     sizeof(char*)]` (the real glibc `_unused2` formula, `void*` swapped for `char*`).
//   Bitfields: `int flags2:24;` sits between plain `int`/pointer fields, mirroring the real
//     `_flags2:24`.
//   `_Bool`: not itself a glibc `_IO_FILE` field, so exercised separately above. This test's
//     job is proving the other four features compose, since `_Bool` does not appear in the
//     real struct.
test "IO_FILE mirror: forward decls, self-pointer, bitfield, const-expr array dim, and anonymous typedefs all parse; sizeof matches gcc" {
    try expectAgreesBits(
        \\struct myio_file;
        \\struct myio_marker;
        \\struct myio_codecvt;
        \\struct myio_wide_data;
        \\typedef struct myio_file myio_FILE;
        \\typedef union { char c8[8]; long l; } mbstate_alt;
        \\typedef struct { int val[2]; } fsid_alt;
        \\struct myio_file {
        \\  int flags;
        \\  char *read_ptr;
        \\  char *read_end;
        \\  char *read_base;
        \\  char *write_base;
        \\  char *write_ptr;
        \\  char *write_end;
        \\  char *buf_base;
        \\  char *buf_end;
        \\  struct myio_file *chain;
        \\  int fileno;
        \\  int flags2:24;
        \\  struct myio_marker *markers;
        \\  struct myio_codecvt *codecvt_ptr;
        \\  struct myio_wide_data *wide_data;
        \\  char unused2[15 * sizeof(int) - 4 * sizeof(char*) - sizeof(char*)];
        \\};
        \\unsigned long f(void){
        \\  mbstate_alt m;
        \\  fsid_alt s;
        \\  m.l = 0;
        \\  s.val[0] = 0;
        \\  return sizeof(struct myio_file);
        \\}
    , &.{}, 64);
}
// Runtime companion to the parse-and-sizeof test above: the alias (`myio_FILE`), the
// self-pointer (`chain`), the never-completed pointer fields (`markers`/`codecvt_ptr`/
// `wide_data`, each set to a null literal and read back through `!`, mirroring the earlier
// "never-completed" test's `!p` idiom), and the bitfield (`flags2`) all read and write
// correctly together, not just in isolation.
test "IO_FILE mirror: alias, self-pointer, incomplete-pointer fields, and bitfield read/write compose" {
    try expectAgreesBits(
        \\struct myio_file;
        \\struct myio_marker;
        \\struct myio_codecvt;
        \\struct myio_wide_data;
        \\typedef struct myio_file myio_FILE;
        \\struct myio_file {
        \\  int flags;
        \\  char *read_ptr;
        \\  struct myio_file *chain;
        \\  int fileno;
        \\  int flags2:24;
        \\  struct myio_marker *markers;
        \\  struct myio_codecvt *codecvt_ptr;
        \\  struct myio_wide_data *wide_data;
        \\  char unused2[15 * sizeof(int) - 4 * sizeof(char*) - sizeof(char*)];
        \\};
        \\int f(void){
        \\  myio_FILE a;
        \\  struct myio_file b;
        \\  a.flags = 1;
        \\  a.fileno = 3;
        \\  a.flags2 = 42;
        \\  a.chain = &b;
        \\  b.flags = 2;
        \\  a.markers = 0;
        \\  a.codecvt_ptr = 0;
        \\  a.wide_data = 0;
        \\  return a.flags + a.fileno + a.flags2 + a.chain->flags + !a.markers;
        \\}
    , &.{}, 32); // 1+3+42+2+1 = 49
}

// The globals-walking skeleton. A bare `int g;` becomes a zero-initialized `.bss`
// `DataObject`. A `.name` reference to it, read or write, from any function in the TU,
// resolves through `global_addr` rather than an alloca slot.
test "global int read/write within a function" {
    try expectAgreesBits("int g; int f(int a){ g = a; return g; }", &.{7}, 32); // 7
}
test "global int across functions" {
    // The global is written by `setg` and read back by `f`. This proves the same `.bss`
    // object is resolved, by symbol name, from two different functions' IR, not two
    // independent slots.
    try expectAgreesBits("int g; void setg(int x){ g = x; } int f(int a){ setg(a); return g + 1; }", &.{41}, 32); // 42
}

// Scalar global initializers: a compile-time constant evaluator plus section routing.
// `int g = 5;` goes to `.data`, `const int g = 5;` goes to `.rodata`, `int g;`/`int g = 0;`
// go to `.bss`, and `int g = 2 + 3*4;` is constant-folded. A non-constant initializer is
// rejected at compile time (`error.Unsupported`), never lowered as a runtime store.
test "initialized global" {
    try expectAgreesBits("int g = 5; int f(void){ return g; }", &.{}, 32); // 5
}
test "const-folded global initializer" {
    try expectAgreesBits("int g = 2 + 3 * 4; int f(void){ return g; }", &.{}, 32); // 14
}
test "const global reads correctly" {
    try expectAgreesBits("const int k = 99; int f(void){ return k; }", &.{}, 32); // 99
}
test "global initialized to zero and bss agree" {
    try expectAgreesBits("int a = 0; int b; int f(void){ return a + b + 1; }", &.{}, 32); // 1
}
test "non-constant global init is rejected" {
    try std.testing.expectError(error.Unsupported, cc.compile(std.testing.allocator, "int x; int g = x; int f(void){ return g; }"));
}
// A constant initializer whose fold would TRAP on the evaluator's host i64 arithmetic must
// fail closed with `error.Unsupported`, never crash the compiler process: INT_MIN/-1 (two's-
// complement overflow, C UB anyway), and div/mod by zero.
test "overflowing constant global init is rejected, not a crash" {
    try std.testing.expectError(error.Unsupported, cc.compile(std.testing.allocator, "int g = (-9223372036854775807 - 1) / -1; int f(void){ return g; }"));
}
test "div-by-zero constant global init is rejected, not a crash" {
    try std.testing.expectError(error.Unsupported, cc.compile(std.testing.allocator, "int g = 5 / 0; int f(void){ return g; }"));
}
test "mod-by-zero constant global init is rejected, not a crash" {
    try std.testing.expectError(error.Unsupported, cc.compile(std.testing.allocator, "int g = 5 % 0; int f(void){ return g; }"));
}

// `static` local variables. A `static int n = 0;` inside a function persists across calls,
// backed by a uniquely-named data object, not a stack alloca. Its compile-time-constant
// initializer runs once, baked into the object's bytes.
test "static local persists across calls" {
    try expectAgreesBits("int next(void){ static int n = 0; n = n + 1; return n; } int f(int c){ int r = 0; int i; for (i = 0; i < c; i = i + 1) r = next(); return r; }", &.{5}, 32); // 5
}
test "static local initialized once, not reset" {
    try expectAgreesBits("int acc(int x){ static int s = 100; s = s + x; return s; } int f(int a){ acc(a); return acc(a); }", &.{3}, 32); // 106
}

// String literals in expression context. `"abc"` becomes an anonymous `.rodata` char array
// (bytes plus a NUL), decaying to `char*` as an rvalue. Indexing and `sizeof` both work.
test "index a string literal" {
    try expectAgreesBits("int f(void){ return \"abc\"[1]; }", &.{}, 32); // 'b' = 98
}
test "string literal via char pointer" {
    try expectAgreesBits("int f(void){ char *s = \"hi\"; return s[0] + s[1]; }", &.{}, 32); // 'h'+'i' = 104+105 = 209
}
test "sizeof string literal includes NUL" {
    try expectAgreesBits("unsigned long f(void){ return sizeof(\"abc\"); }", &.{}, 64); // 4
}

// Aggregate initializers: `int a[3] = {1,2,3};`, struct `{...}`, a partial init (zero-fills
// the rest), an unsized array sized from its initializer, and a `char[]` sized and filled
// from a string literal. File-scope (`consteval.evalInit`, `lower.compile`) and
// `static`-local (`lower.lowerStmt`'s `.decl` arm) share the same fold.
test "array aggregate init" {
    try expectAgreesBits("int a[3] = {10, 20, 30}; int f(int i){ return a[i]; }", &.{2}, 32); // 30
}
test "partial array init zero-fills" {
    try expectAgreesBits("int a[4] = {7}; int f(void){ return a[0] + a[3]; }", &.{}, 32); // 7
}
test "unsized array from initializer" {
    try expectAgreesBits("int a[] = {1,2,3,4}; unsigned long f(void){ return sizeof(a); }", &.{}, 64); // 16
}
test "struct aggregate init" {
    try expectAgreesBits("struct P { int x; int y; }; struct P p = {3, 4}; int f(void){ return p.x * 10 + p.y; }", &.{}, 32); // 34
}
test "char array from string" {
    try expectAgreesBits("char s[] = \"hi\"; int f(void){ return s[0] + s[1] + s[2]; }", &.{}, 32); // 'h'+'i'+0 = 209
}

// Pointer-valued static initializers, plus `extern`. A file-scope `int *p = &g;`/
// `char *s = "hi";` folds to a `DataObject` whose pointer field carries a `DataReloc`,
// patched to the target's runtime address once the JIT links the module
// (`consteval.tryAddressConst`). `extern int g;` registers `g` in the `GlobalTable` with no
// `DataObject` of its own, since the defining decl elsewhere in the TU supplies it.
test "pointer global to another global" {
    try expectAgreesBits("int g = 42; int *p = &g; int f(void){ return *p; }", &.{}, 32); // 42 (backward ref)
}
test "pointer global to a later-defined global, declared extern first (forward ref)" {
    // `&g` before `g`'s definition, but with a prior `extern int g;` declaration, is valid C
    // (gcc accepts it). The two-pass `compile` registers every global name before folding
    // any initializer, so `&g` resolves regardless of where the definition falls.
    try expectAgreesBits("extern int g; int *p = &g; int g = 42; int f(void){ return *p; }", &.{}, 32); // 42
}
test "pointer global to a later-defined global with NO prior declaration (frontend extension)" {
    // `int *p = &g; int g = 42;` with no prior `extern int g;` is rejected by gcc, since `g`
    // is undeclared at the point of `&g`, so it cannot be a gcc-differential test. The
    // two-pass `compile`, which registers all global names before folding any initializer,
    // accepts it and resolves `&g` to `g`'s object. Asserted directly against the frontend's
    // own JIT (raw and mem2reg), not gcc, since there is no valid oracle for a program gcc
    // will not build. This pins the two-pass behavior. A single forward pass would give
    // `error.Unsupported`.
    const allocator = std.testing.allocator;
    const src = "int *p = &g; int g = 42; int f(void){ return *p; }";
    const raw = try frontendRun(allocator, src, &.{});
    defer allocator.free(raw);
    try std.testing.expectEqualStrings("42", raw);
    const optimized = try frontendRunOpt(allocator, src, &.{});
    defer allocator.free(optimized);
    try std.testing.expectEqualStrings("42", optimized);
}
test "file-scope char pointer to string" {
    try expectAgreesBits("char *msg = \"hi\"; int f(void){ return msg[0] + msg[1]; }", &.{}, 32); // 'h'+'i' = 209
}
// A tentative definition (`int g;`, no initializer) plus a real one (`int g = 5;`) name the
// same object, defined with value 5. This is valid C in either order. Pass 1's
// definition-selection must pick the initialized decl, not emit `.bss` zero from the
// tentative and drop the `= 5`.
test "tentative definition before the real one" {
    try expectAgreesBits("int g; int g = 5; int f(void){ return g; }", &.{}, 32); // 5
}
test "tentative definition after the real one does not clobber it" {
    try expectAgreesBits("int g = 5; int g; int f(void){ return g; }", &.{}, 32); // 5
}
test "extern references a later definition in the same TU" {
    try expectAgreesBits("extern int g; int f(void){ return g; } int g = 7;", &.{}, 32); // 7
}
// Fails closed: an address constant that is not `&<file-scope global>` or a string literal.
// Here, `&nonexistent` (consteval has no local scope at all, so this is indistinguishable
// from, and exactly as unsupported as, `&<local>`) must never be silently treated as null
// or some other placeholder value. It fails the whole compile.
test "address of an unresolvable name in a static initializer is rejected" {
    try std.testing.expectError(error.Unsupported, cc.compile(std.testing.allocator, "int *p = &nonexistent; int f(void){ return *p; }"));
}
// A wholly-undefined `extern` (no definition anywhere in the TU) still compiles and lowers.
// `g` is registered in the `GlobalTable` (owned by `Module.extern_syms`, backing no
// `DataObject`), so `f`'s reference to it still lowers to a `global_addr`, and fails only
// once the JIT tries to link the missing symbol, per this function's and
// `Module.extern_syms`'s doc. `g`'s address now resolves GOT-indirectly (`via_got`, since
// `g` has no TU definition; see `resolveName`), the data-import path meant for the real
// object-emit and dynamic linker. This simplified in-process JIT linker deliberately never
// carries a GOT (every backend's own `link.zig` rejects
// `.got_pg`/`.got_lo12`/`.got_pcrel`/`.got_abs`/`.got_hi20` outright), so it now fails with
// `error.UnsupportedReloc` rather than `error.UndefinedSymbol`. It still "fails only at JIT
// link", just for the GOT-indirect reason.
test "extern with no definition in the TU compiles, fails only at JIT link" {
    const allocator = std.testing.allocator;
    var mod = try cc.compile(allocator, "extern int g; int f(void){ return g; }");
    defer mod.deinit(allocator);
    const f_ir = mod.find("f") orelse return error.NoFunc;
    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    try mfs.append(allocator, .{ .name = "f", .func = f_ir });
    var data = try buildModuleData(allocator, mod.data);
    defer freeModuleData(allocator, &data);
    try std.testing.expectError(error.UnsupportedReloc, target.native.jitModuleData(allocator, mfs.items, data.items));
}

// The preprocessor (line splicing and comment stripping) is now wired ahead of the lexer
// for every `cc.compile` call. A block comment mid-expression and a `//` line comment
// before the closing brace must both disappear without affecting the result.
test "comments are stripped by the preprocessor, agrees with gcc" {
    try expectAgrees("int f(int a){ /* add */ return a + 1; // done\n }", &.{7});
}

// A `//` comment whose line ends in a backslash-newline splice: phase-2 splicing removes
// the `\` plus the newline, so the comment runs on into the joined line and swallows
// `r = 99;`. gcc returns 1. Before the phase-2 pre-pass, this frontend mis-scanned
// `r = 99;` as live code and returned 99, a silent miscompile.
test "line comment extended by a splice agrees with gcc" {
    try expectAgrees("int f(void){ int r = 1; // c\\\n r = 99;\n return r; }", &.{});
}

// An operator split across a backslash-newline splice (`=\<NL>=`) must scan as one `==`,
// not two `=`s: `2 == 2` is 1.
test "spliced operator agrees with gcc" {
    try expectAgrees("int f(void){ return 2 =\\\n= 2; }", &.{});
}

// Directive dispatch, object-like `#define`/`#undef`, and macro expansion with rescanning
// and a Prosser hide-set.

// `M` expands to `N`, which itself expands to `5`: the rescan has to chase through both.
test "object-like macro expands and rescans, agrees with gcc" {
    try expectAgrees("#define N 5\n#define M N\nint f(void){ return M + 1; }", &.{});
}

// `#define A A`: `A` is its own replacement. It expands exactly once (to `A`, now hidden),
// so it terminates instead of looping, and the declaration/use of local `A` is unaffected.
test "self-referential macro terminates, agrees with gcc" {
    try expectAgrees("#define A A\nint f(void){ int A = 3; return A; }", &.{});
}

// `#undef` removes a macro so a later identically-named use is untouched.
test "undef removes a macro definition, agrees with gcc" {
    try expectAgrees("#define N 5\n#undef N\nint f(void){ int N = 9; return N; }", &.{});
}

// Function-like macros: arguments, `#` stringize, `##` token-paste, and recognizing, but
// soft-failing on, variadic macros.

// A plain function-like macro call: arguments are substituted (and, since neither is used
// with `#`/`##` here, fully macro-expanded first, though this test doesn't exercise that).
test "function-like macro" {
    try expectAgrees("#define ADD(a,b) ((a)+(b))\nint f(int x){ return ADD(x, 3); }", &.{4});
}

// `#x` stringizes the RAW argument token(s) into a string literal; `sizeof` on it counts
// the bytes plus the trailing NUL ("hello\0" is 6).
test "stringize length" {
    try expectAgreesBits("#define S(x) #x\nunsigned long f(void){ return sizeof(S(hello)); }", &.{}, 64);
}

// `a##b` pastes the two argument spellings into one re-lexed identifier `xy`, which then
// names the local variable declared just above.
test "token paste" {
    try expectAgrees("#define CAT(a,b) a##b\nint f(void){ int xy = 5; return CAT(x,y); }", &.{});
}

// A function-like macro's bare name, NOT immediately followed by `(`, is never expanded -
// so `int F = 8;` declares an ordinary variable named `F`, unaffected by `#define F(x) x`.
test "bare function-like name not followed by paren is literal" {
    try expectAgrees("#define F(x) x\nint F = 8;\nint f(void){ return F; }", &.{});
}

// A fix: a function-like invocation whose argument list spans a newline still expands.
// Whitespace, including newlines, is allowed inside the arg list, giving `7`.
test "function-like macro invocation spanning a newline" {
    try expectAgrees("#define ADD(a,b) ((a)+(b))\nint f(int x){ return ADD(x,\n3); }", &.{4});
}

// A fix: the `(` may be on the line after the macro name and still be a call.
test "function-like macro name and paren on separate lines" {
    try expectAgrees("#define ADD(a,b) ((a)+(b))\nint f(int x){ return ADD\n(x, 3); }", &.{4});
}

// A fix: an object-like macro body can contain `##` too. `a##b` pastes to the identifier
// `ab`, which names the local declared above, giving 5.
test "object-like macro token paste" {
    try expectAgrees("#define CAT a##b\nint f(void){ int ab = 5; return CAT; }", &.{});
}

// A fix: a 1-param function-like macro called with empty parens `NOTHING()` passes one
// empty argument (C: 0 commas means 1 arg), which substitutes to nothing, giving `5` -> 5.
test "empty function-like macro argument substitutes to nothing" {
    try expectAgrees("#define NOTHING(x) 5 x\nint f(void){ return NOTHING(); }", &.{});
}

// The same fix: a non-empty argument to the same shape still works, giving 7.
test "non-empty single argument still works" {
    try expectAgrees("#define E(a) a\nint f(void){ return E(7); }", &.{});
}

// Conditional compilation: `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, `defined`,
// and a dedicated `#if` constant-expression evaluator.

// `#ifdef` takes the `#if` branch when the name IS defined; the `#else` branch never runs.
test "ifdef takes the true branch when the macro is defined" {
    try expectAgrees("#define ON\n#ifdef ON\nint f(void){ return 1; }\n#else\nint f(void){ return 0; }\n#endif", &.{});
}

// `#ifndef` takes its branch when the name is NOT defined.
test "ifndef takes its branch when the macro is undefined" {
    try expectAgrees("#ifndef OFF\nint f(void){ return 7; }\n#endif", &.{});
}

// `defined(V)` and `defined X` are both resolved against the macro table BEFORE the rest of
// the `#if` line is macro-expanded, so `V` (a macro) still expands to `3` afterward.
test "if expression combines defined() with a macro-valued comparison" {
    try expectAgrees("#define V 3\n#if defined(V) && V > 2\nint f(void){ return 42; }\n#else\nint f(void){ return 0; }\n#endif", &.{});
}

// An `#elif` chain: only the first true branch (`X==2`) emits.
test "elif chain picks the first true branch" {
    try expectAgrees("#define X 2\n#if X==1\nint f(void){return 10;}\n#elif X==2\nint f(void){return 20;}\n#else\nint f(void){return 30;}\n#endif", &.{});
}

// A `#if 0` group's dead branch is skipped without ever tokenizing or expanding its
// contents. `garbage nonsense !!!`, which is not even valid C, let alone a valid
// macro-expansion input, must never reach the region/expansion machinery, nested
// conditional and all.
test "nested conditionals skip correctly, garbage in a dead branch never parsed" {
    try expectAgrees("#if 0\n#if 1\ngarbage nonsense !!!\n#endif\n#endif\nint f(void){return 5;}", &.{});
}

// A fix: a `#if` integer literal obeys C base rules. `010` is octal 8, so `010 == 8` is
// true and the `#if` branch (return 1) is taken, agreeing with gcc.
test "if with an octal literal parses base 8, agrees with gcc" {
    try expectAgrees("#if 010 == 8\nint f(void){return 1;}\n#else\nint f(void){return 0;}\n#endif", &.{});
}

// A larger octal literal: `0777` is 511.
test "if with a multi-digit octal literal, agrees with gcc" {
    try expectAgrees("#if 0777 == 511\nint f(void){return 1;}\n#else\nint f(void){return 0;}\n#endif", &.{});
}

// A fix: `||` short-circuits, so a true left operand means the `1/0` right operand is
// parsed but never evaluated. No error occurs, and the `#if` branch is taken, matching gcc.
test "if || short-circuits past a divide-by-zero, agrees with gcc" {
    try expectAgrees("#if 1 || (1/0)\nint f(void){return 1;}\n#else\nint f(void){return 0;}\n#endif", &.{});
}

// `&&` short-circuits on a false left operand: `0 && (1/0)` is false (no error), `#else` runs.
test "if && short-circuits past a divide-by-zero, agrees with gcc" {
    try expectAgrees("#if 0 && (1/0)\nint f(void){return 1;}\n#else\nint f(void){return 0;}\n#endif", &.{});
}

// `?:` evaluates only the taken arm: `1 ? 42 : (1/0)` takes the 42 side (nonzero, no error).
test "if ternary evaluates only the taken arm, agrees with gcc" {
    try expectAgrees("#if 1 ? 42 : (1/0)\nint f(void){return 1;}\n#else\nint f(void){return 0;}\n#endif", &.{});
}

// `#include`, a resolver, recursive inclusion, and `#pragma once`. The purely in-process
// cases (resolver wiring, `<...>` reconstruction, the include-cycle depth-limit guard) live
// in `preproc.zig` next to the rest of that file's direct unit tests. These two are the
// gcc-diff execution tests `expectAgreesPPBits` exists for.

// `defs.h` defines `MAGIC`; `f` `#include`s it and uses the macro in an expression gcc has
// to agree with bit-for-bit.
test "#include a header, agrees with gcc" {
    try expectAgreesPPBits("#include \"defs.h\"\nint f(void){ return MAGIC + 1; }", &.{
        .{ .name = "defs.h", .content = "#define MAGIC 41\n" },
    }, &.{}, 32);
}

// `a.h` is `#include`d twice. Its own `#pragma once` makes the second inclusion a no-op, so
// `V` is defined exactly once. A `#define` with no `#pragma once` guard would still agree
// here, since this frontend's `#define` allows redefinition, but a real header full of
// declarations, not just one macro, would double-declare without the guard. This test
// proves the guard itself fires, not merely that redefinition happens to be harmless.
test "#pragma once prevents redefinition errors, agrees with gcc" {
    try expectAgreesPPBits("#include \"a.h\"\n#include \"a.h\"\nint f(void){ return V; }", &.{
        .{ .name = "a.h", .content = "#pragma once\n#define V 9\n" },
    }, &.{}, 32);
}

// Predefined macros, plus `-D`/`-U`. `__STDC__` and `__LINE__` are both live from the very
// first line of the TU, and `f`'s body is entirely on that line, so gcc and the frontend
// agree on `__LINE__ == 1` here just as readily as on `__STDC__ == 1`.
test "__STDC__ and __LINE__ agree with gcc" {
    try expectAgrees("int f(void){ return __STDC__ + __LINE__; }", &.{}); // 1 + 1 = 2
}

// `__DATE__`/`__TIME__`'s STRING LENGTH is format-invariant (`"Mmm DD YYYY"` is always 11
// bytes, `"HH:MM:SS"` always 8) regardless of what moment either side actually formats -
// gcc's real wall-clock date and this frontend's `timestamp = 0` (the `Options` default)
// produce different bytes but the identical `sizeof`, so this agrees with gcc without either
// side needing to know the other's clock.
test "sizeof(__DATE__) + sizeof(__TIME__) agrees with gcc" {
    try expectAgreesBits("unsigned long f(void){ return sizeof(__DATE__) + sizeof(__TIME__); }", &.{}, 64); // 12 + 9 = 21
}

// `-D FOO=7`, through `Options.defines`, threaded through `compileWithOpts` exactly like an
// `#include` resolver is. No oracle is needed, since gcc is not invoked with a matching `-D`
// here. This just proves the frontend's own `-D` wiring produces the defined value.
test "-D via compileWithOpts defines a macro" {
    const allocator = std.testing.allocator;
    const pp: cc.preproc.Options = .{ .defines = &.{.{ .name = "FOO", .value = "7" }} };
    const got = try frontendRunPP(allocator, "int f(void){ return FOO; }", &.{}, pp);
    defer allocator.free(got);
    try std.testing.expectEqualStrings("7", got);
}

// The parser now reads the lexer's preserved `0x`/`0b`/leading-zero prefix and parses
// `int_lit` text at the right radix, instead of always assuming base 10.
test "hex literal" {
    try expectAgrees("int f(void){ return 0x1F; }", &.{}); // 31
}

test "octal literal" {
    try expectAgrees("int f(void){ return 0755; }", &.{}); // 493
}

test "binary literal" {
    try expectAgrees("int f(void){ return 0b101; }", &.{}); // 5
}

test "hex in enum and array" {
    try expectAgrees("enum E { X = 0x10 }; int f(void){ int a[0x4]; a[3]=X; return a[3]; }", &.{}); // 16
}

test "hex unsigned suffix" {
    try expectAgreesBits("unsigned long f(void){ return 0xFFFFFFFFUL; }", &.{}, 64); // 4294967295
}

// The float-return harness: `oracleRunFloat`'s union-reinterpret bit-compare (never
// `%f`/`%g`, which would round), `returnKind` recognizing a `.float` return type, and the
// `callFF32Bits`/`callFF64Bits` JIT entries. `expectAgreesFloat`/`expectAgreesDouble` (which
// run real C `source` through `cc.compileWithOpts`, like every other `expectAgrees*`) are
// what this harness delivers, but the frontend does not yet parse `float`/`double`
// type-specifiers or float literals. A later test's `double f(void){ return 1.5; }` is
// their first real caller. This regression test instead hand-builds the tiny IR function
// the parser will eventually produce, and drives `returnKind`, `callFF64Bits`, and
// `oracleRunFloat` directly against it, proving the harness machinery end to end without
// depending on the parser support.
test "float-return harness: hand-built double-returning IR agrees with the gcc oracle bit-for-bit" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const oracle = oracleRunFloat(io, allocator, "double f(void){ return 1.5; }", &.{}, true) catch |err| switch (err) {
        error.NoCompiler => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(oracle);

    var func = ir.function.Function.init(allocator);
    defer func.deinit();
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const c = try func.appendInst(b, f64_t, .{ .fconst = 1.5 });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(c) });

    // `returnKind` must classify this function's return as `.f64`, not fall through to
    // `error.Unsupported` (the earlier behavior for any non-`.int` return type).
    try std.testing.expectEqual(std.meta.Tag(ReturnKind).f64, std.meta.activeTag(try returnKind(&func)));

    var jitted = try target.native.jitModule(allocator, &.{.{ .name = "f", .func = &func }});
    defer jitted.deinit();
    const bits = try callFF64Bits(&jitted, &.{});
    const got = try std.fmt.allocPrint(allocator, "{d}", .{bits});
    defer allocator.free(got);

    try std.testing.expectEqualStrings(oracle, got);
}

// The frontend now parses `float`/`double` type-specifiers and float literals, emits float
// arithmetic, and inserts int<->float/float<->float `convert`s. These are the first real
// callers of `expectAgreesFloat`/`expectAgreesDouble` (the harness itself proved out above
// via a hand-built IR function).
test "float literal return" {
    try expectAgreesDouble("double f(void){ return 1.5; }", &.{});
}
test "float arithmetic" {
    try expectAgreesDouble("double f(void){ return 1.5 + 2.25 * 2.0; }", &.{}); // 6.0
}
test "float f32" {
    try expectAgreesFloat("float f(void){ return 1.5f + 0.5f; }", &.{}); // 2.0f
}
test "int plus double promotes" {
    try expectAgreesDouble("double f(int a){ return a + 0.5; }", &.{3}); // 3.5
}
test "double to int truncates on return" {
    try expectAgrees("int f(void){ return 3.9; }", &.{}); // 3 (double->int convert)
}
// A follow-up: unary negate on a float operand. `-d` lowers to `0.0 - d` (fsub), a very
// common valid C shape that was previously fail-closed `error.Unsupported`.
test "negate double literal" {
    try expectAgreesDouble("double f(void){ return -1.5; }", &.{}); // -1.5
}
test "negate double variable" {
    try expectAgreesDouble("double f(int a){ double d = a; return -d; }", &.{3}); // -3.0
}
test "negate double in arithmetic" {
    try expectAgreesDouble("double f(void){ double d = 2.5; return -d + 1.0; }", &.{}); // -1.5
}
test "negate float literal" {
    try expectAgreesFloat("float f(void){ return -0.5f; }", &.{}); // -0.5f
}
// A follow-up fix: a zero of a FLOAT type must be `fconst 0.0`, never a float-typed
// `iconst 0` (invalid IR, only accidentally right on aarch64, and rejected by the riscv64
// isel). These lock in the value (unchanged on host) at the fall-off-end/return-seal,
// `.lognot`, and `truthy` (if-condition) sites where a float zero is materialized.
test "float return with fall-through seal, taken false" {
    try expectAgreesDouble("double f(int c){ if (c) return 1.0; return 0.0; }", &.{0}); // 0.0
}
test "float return with fall-through seal, taken true" {
    try expectAgreesDouble("double f(int c){ if (c) return 1.0; return 0.0; }", &.{1}); // 1.0
}
test "lognot of a zero double is 1" {
    try expectAgrees("int f(void){ double d = 0.0; return !d; }", &.{}); // 1
}
test "lognot of a nonzero double is 0" {
    try expectAgrees("int f(void){ double d = 2.5; return !d; }", &.{}); // 0
}
test "float truthiness in an if condition" {
    try expectAgrees("int f(int c){ double d = 3.0; if (d) c = c + 1; return c; }", &.{5}); // 6
}

// `++`/`--` (prefix and postfix, for int/float/pointer), and generalizing compound
// assignment to floats and pointers. `.incdec` resolves `target`'s lvalue address once
// (`lowerAddr`, the same model `.assign` uses), loads the old value, stores new = old +/- 1
// (int/float) or old scaled by `sizeof(pointee)` (pointer, via the same `lowerPtrArith`
// path ordinary `p + n` takes) back through it, and yields new (prefix) or old (postfix).
test "prefix and postfix int" {
    // y=5 (old x), x=6. Then z=7 (new x), x=7. So 5*100+7+7=514.
    try expectAgrees("int f(int a){ int x = a; int y = x++; int z = ++x; return y*100 + z + x; }", &.{5});
}
test "pointer postfix increment (a.c pattern)" {
    // *p++ = a stores through p THEN advances p (postfix yields the old address) -> arr = {10,11,12}
    try expectAgrees("int f(int a){ int arr[3]; int *p = arr; *p++ = a; *p++ = a+1; *p = a+2; return arr[0]+arr[1]+arr[2]; }", &.{10}); // 33
}
test "while n-- loop" {
    try expectAgrees("int f(int n){ int c = 0; while (n--) c = c + 1; return c; }", &.{5}); // 5
}
test "float increment" {
    try expectAgreesDouble("double f(void){ double d = 1.5; d++; return d; }", &.{}); // 2.5
}
// Compound assignment generalized beyond int: a float target routes through the same
// `commonType`/`convertTo` path `.binary` uses. A pointer target routes through
// `lowerPtrArith`, scaled by `sizeof(pointee)`, the same as `p + n`.
test "float compound assign" {
    try expectAgreesDouble("double f(int a){ double d = a; d += 0.5; d *= 2.0; return d; }", &.{3}); // 7.0
}
test "pointer compound assign" {
    try expectAgrees("int f(int a){ int arr[3]; int *p = arr; p += 2; *p = a; return arr[2]; }", &.{9}); // 9
}

// Cast expressions `(type)expr`: a parser disambiguation of `(` as cast-vs-parenthesized
// (only a clear type-name right after `(` starts a cast; a plain `(a + b)` or `(x)` over a
// variable stays a paren-expr), plus `Expr.cast` lowering through `convertTo`, which is
// also extended here to cover int<->pointer.
test "double to int cast truncates" {
    try expectAgrees("int f(void){ return (int)3.9; }", &.{}); // 3
}
test "int to double cast" {
    try expectAgreesDouble("double f(void){ return (double)5 / 2; }", &.{}); // 2.5
}
test "char truncation cast" {
    try expectAgrees("int f(void){ return (unsigned char)300; }", &.{}); // 44
}
test "unsigned cast" {
    try expectAgreesBits("unsigned f(void){ return (unsigned)-1; }", &.{}, 32); // 4294967295
}
test "pointer to int roundtrip" {
    try expectAgrees("int f(int a){ int x = a; int *p = &x; long n = (long)p; int *q = (int*)n; return *q; }", &.{42}); // 42
}

// `void` and `void*` as first-class types. `void*` implicitly converts to and from any
// object-pointer type with no cast (C11 6.3.2.3p1). `lower.convertTo`'s pointer-pointer arm
// is pointee-agnostic for every `T* <-> U*` pair, not something added specifically for
// `void*`, so this exercises that same machinery.
test "void* implicitly converts to and from an object pointer, both ways" {
    try expectAgrees("int f(void){ int x = 42; void *p = &x; int *q = p; return *q; }", &.{}); // 42
}
test "cast through void*" {
    try expectAgrees("int f(void){ int x = 42; void *p = &x; return *(int*)p; }", &.{}); // 42
}
test "sizeof(void*) agrees with gcc" {
    try expectAgreesBits("unsigned long f(void){ return sizeof(void*); }", &.{}, 64); // 8 on an LP64 host
}
test "void* through char*, both ways" {
    try expectAgrees("int f(void){ char c = 7; void *p = &c; char *q = (char*)p; return *q; }", &.{}); // 7
}
test "void* compares against the null pointer constant" {
    try expectAgrees("int f(void){ void *p = 0; return p == 0; }", &.{}); // 1
}
test "the (void)expr discard idiom still parses and runs" {
    try expectAgrees("int f(void){ int a = 5; (void)a; return a; }", &.{}); // 5
}
test "a void-parameter-list function still works" {
    try expectAgrees("int f(void){ return 5; }", &.{}); // 5
}

// `__restrict`/`__restrict__`/`restrict` are ignored qualifiers. glibc puts `__restrict` on
// pointer params 78 times, for example `const char *__restrict __filename`. VCC used to
// consume `__restrict` as the declared name, corrupting the parse. `parseQualRun` now
// recognizes all three spellings by identifier text (no lexer keyword added) and drops
// them. VCC does no restrict-based optimization, so the qualifier carries no semantics
// here, it only parses.
test "restrict-qualified pointer param parses and runs" {
    // The GNU `__restrict` spelling, on a helper's pointer param, called from `f` (the
    // harness's fixed entry point) with the address of a local.
    try expectAgrees("int helper(char *__restrict p){ *p = 42; return *p; } int f(int a){ char c = (char)a; return helper(&c); }", &.{0}); // 42
}
test "restrict composes with void*" {
    try expectAgrees("int g(void *__restrict p){ return p != 0; } int f(int a){ int x = a; return g(&x); }", &.{5}); // 1
}
test "C99 restrict spelling parses and runs" {
    try expectAgrees("int h(int *restrict p){ return *p; } int f(int a){ int x = a; return h(&x); }", &.{9}); // 9
}
test "__restrict__ spelling parses and runs" {
    try expectAgrees("int k(int *__restrict__ p){ return *p; } int f(int a){ int x = a; return k(&x); }", &.{11}); // 11
}
test "restrict on a local pointer, not just a param" {
    try expectAgrees("int f(void){ int x = 5; int *__restrict p = &x; return *p; }", &.{}); // 5
}

// `__attribute__((...))` parses and is ignored at every position glibc's full-fat
// `stdio.h` chain uses (111 occurrences, dominantly `__THROW`/`__nonnull`/`__wur` right
// after a prototype's `)`). VCC records nothing from an attribute. It is a pure
// parse-and-skip, so every case here still behaves exactly like its attribute-free
// equivalent. A `__warn_unused_result__`/`noreturn`-style attribute does not change
// control flow here. Only real backend logic would, and no attribute drives any.
test "__attribute__ after a function prototype (glibc's __THROW position) parses and is ignored" {
    try expectAgrees("int f(int x) __attribute__((__nothrow__)); int f(int x){ return x; }", &.{42}); // 42
}
test "__attribute__ in decl-spec position (leading the whole declaration) parses and is ignored" {
    try expectAgrees("__attribute__((__pure__)) int f(void){ return 3; }", &.{}); // 3
}
test "__attribute__ after a declarator (on a global) parses and is ignored" {
    try expectAgrees("int y __attribute__((__unused__)) = 7; int f(void){ return y; }", &.{}); // 7
}
test "__attribute__ on a struct field parses and is ignored" {
    try expectAgrees("struct S { int a __attribute__((__deprecated__)); }; int f(void){ struct S s; s.a = 9; return s.a; }", &.{}); // 9
}
test "__attribute__ on a typedef parses and is ignored" {
    try expectAgrees("typedef int myint __attribute__((__aligned__(4))); int f(void){ myint x = 5; return x; }", &.{}); // 5
}
test "stacked, argument-bearing __attribute__ specifiers all parse and are ignored" {
    // `__format__(__printf__, ...)` demands a genuinely variadic printf-shaped function to
    // type-check under real gcc (its own semantic check, unrelated to attribute parsing),
    // so `__aligned__(8)` stands in here as the arg-bearing attribute alongside
    // `__nonnull__`. VCC ignores every attribute's argument list identically, regardless of
    // which one it is.
    try expectAgrees(
        "int f(int x) __attribute__((__nothrow__)) __attribute__((__nonnull__(1))) __attribute__((__aligned__(8))); int f(int x){ return x; }",
        &.{42},
    ); // 42
}
test "a noreturn-style __attribute__ is ignored - the function still returns normally" {
    try expectAgrees("int f(void) __attribute__((__warn_unused_result__)); int f(void){ return 42; }", &.{}); // 42
}

// `inline`/`__inline`/`__inline__`, `_Noreturn`, `__extension__`, and `__asm__`/`asm`
// labels all parse as no-op decorations. VCC compiles an `inline` function as an ordinary
// definition (no inlining, no linkage change), records nothing for
// `_Noreturn`/`__extension__`, and ignores an `__asm__` symbol-rename label entirely. The
// declared C name is what a caller resolves against. See
// `skipAsmLabelAndAttributes`'s doc comment in parser.zig for the `__REDIRECT`-style
// limitation this leaves.
test "static inline function composes with static/extern and runs as an ordinary definition" {
    try expectAgrees("static inline int f(void){ return 7; } int g(void){ return f(); }", &.{}); // 7
}
// Plain `__inline` (no `static`) is paired with `static` here. Under real gcc's C99
// semantics, a lone `inline`/`__inline` definition (no `static`, no `extern`) provides only
// an inline definition. At `-O0` gcc does not actually inline the call, and with nothing
// else providing an out-of-line copy, the oracle side fails to link (`undefined reference to
// h`). This is a genuine C99 linkage subtlety, unrelated to whether VCC's parser recognizes
// `__inline` (VCC always compiles a definition as an ordinary global regardless of any
// inline-ness, so it has no such gap itself). `static __inline` sidesteps the oracle-side
// linkage issue while still exercising the GNU `__inline` spelling composing with `static`.
test "__inline (GNU spelling) composes with static, parses, and runs" {
    try expectAgrees("static __inline int h(void){ return 1; } int f(void){ return h(); }", &.{}); // 1
}
test "_Noreturn on a void function still parses, and the function still falls off the end" {
    try expectAgrees("_Noreturn void n(void){ } int f(void){ return 5; }", &.{}); // 5
}
// A `static void` function's return type (the real `void` `CType`, not the bare-`void`
// special case's `int_t` placeholder) reaches the fall-off-the-end seal at `lower.zig` as
// an actual `void_`. An empty body falls off with no `return` at all. The seal used to call
// `zeroInto(void_)`, which reports `error.VoidValue`, since there is no zero value of type
// void. It must instead emit a valueless `ret` (`.ret = null`), the same as an ordinary
// non-void function's `return <expr>;` just carries no operand.
test "static void function with an empty body falls off the end" {
    try expectAgrees("static void s(void){ } int f(void){ s(); return 42; }", &.{}); // 42
}
// Same seal, but the body has statements (no `return`) before falling off. This proves the
// seal fires after real control flow, not just for a literally-empty block.
test "static void function with statements falls off the end (no return statement)" {
    try expectAgrees("static void s2(void){ int x = 1; x = x + 1; } int f(void){ s2(); return 7; }", &.{}); // 7
}
// A void function called purely for its side effect on a global. This proves the void
// fall-off seal does not corrupt the caller's state, for example by writing a stray zero
// return value somewhere live.
test "static void function called for its side effect on a global, twice" {
    try expectAgrees("static int counter; static void inc(void){ counter = counter + 1; } int f(void){ inc(); inc(); return counter; }", &.{}); // 2
}
// A valueless `return;` inside a void function. The parser's `.kw_return` arm used to
// always read an operand expression, so a bare `return;` failed to parse. `sealRet` already
// emits a valueless IR `ret` for a `void` return type, so lowering a valueless `return;`
// reuses that same helper.
test "valueless return in a void function agrees with gcc" {
    try expectAgrees("void g(int *p){ *p = 7; return; } int f(void){ int x = 0; g(&x); return x; }", &.{}); // 7
}
test "valueless return as an early exit inside an if agrees with gcc" {
    try expectAgrees("void g(int *p, int c){ if (c) { *p = 1; return; } *p = 2; } int f(void){ int x = 0; g(&x, 1); return x; }", &.{}); // 1
}
test "a void function whose only statement is a valueless return agrees with gcc" {
    try expectAgrees("void g(void){ return; } int f(void){ g(); return 3; }", &.{}); // 3
}
test "__extension__ as a no-op declaration prefix" {
    try expectAgrees("__extension__ int e = 3; int f(void){ return e; }", &.{}); // 3
}
test "__extension__ as a no-op expression prefix" {
    try expectAgrees("int f(void){ return __extension__ (2 + 40); }", &.{}); // 42
}
test "an __asm__ decl label after a prototype is ignored - the C name still resolves" {
    try expectAgrees("int p(void) __asm__(\"myp\"); int p(void){ return 8; } int f(void){ return p(); }", &.{}); // 8
}
test "an __asm__ decl label interleaves with __attribute__ in either order" {
    try expectAgrees(
        "int q(void) __asm__(\"qq\") __attribute__((__nothrow__)); int q(void){ return 9; } int f(void){ return q(); }",
        &.{},
    ); // 9
}
test "a statement-level __asm__(...) parses and is ignored (empty statement effect)" {
    try expectAgrees("int f(void){ __asm__(\"\"); return 3; }", &.{}); // 3
}

// `typeof`/`__typeof__` (a type-specifier resolving to an operand's type), `_Generic` (a
// controlling-expression-driven type-selection expression), and GNU statement expressions
// `({ ... })` (a compound statement used as an expression, valued by its last
// expression-statement).
// `typeof` (bare spelling) needs `-std=gnu99` on gcc's side to even parse. See
// `expectAgreesGnu99`'s doc comment. This is unlike `__typeof__`, which gcc keeps available
// under any `-std=`.
test "typeof(x) resolves to x's declared type" {
    try expectAgreesGnu99("int f(void){ int x = 42; typeof(x) y = x; return y; }", &.{}); // 42
}
test "typeof(int) resolves to a type-name operand directly" {
    try expectAgreesGnu99("int f(void){ typeof(int) z = 5; return z; }", &.{}); // 5
}
test "__typeof__ spelling works the same as typeof" {
    try expectAgrees("int f(void){ int x = 42; __typeof__(x) y = x; return y; }", &.{}); // 42
}
test "typeof(x) composes with a pointer declarator" {
    try expectAgreesGnu99("int f(void){ int x = 3; typeof(x) *p = &x; return *p; }", &.{}); // 3
}
test "_Generic selects the matching int association" {
    try expectAgrees("int f(void){ int x = 0; return _Generic(x, int: 42, char: 1, default: 0); }", &.{}); // 42
}
test "_Generic selects the matching char* association over a cast controlling expression" {
    try expectAgrees("int f(void){ return _Generic((char*)0, char*: 7, int: 1, default: 0); }", &.{}); // 7
}
test "_Generic falls back to default when no association matches" {
    try expectAgrees("int f(void){ double d = 0; return _Generic(d, int: 1, default: 42); }", &.{}); // 42
}
test "_Generic never evaluates a non-selected association" {
    // If the `char:` arm were (wrongly) evaluated, `bump()` would run and `counter` would be
    // 1, making the result 43 instead of 42. Both the oracle and the frontend must agree it
    // stays 42 (the standard's "not evaluated" rule for a losing association).
    try expectAgrees(
        \\int counter = 0;
        \\int bump(void) { counter = counter + 1; return 99; }
        \\int f(void) {
        \\  int x = 0;
        \\  int r = _Generic(x, int: 42, char: bump(), default: 0);
        \\  return r + counter;
        \\}
    , &.{}); // 42
}
test "a GNU statement expression's value is its last expression-statement" {
    try expectAgrees("int f(void){ return ({ int a = 40; a + 2; }); }", &.{}); // 42
}
test "a statement expression containing a loop" {
    try expectAgrees(
        "int f(void){ int s = ({ int t = 0; int i; for (i = 0; i < 5; i = i + 1) t = t + i; t; }); return s; }",
        &.{},
    ); // 10
}

// A fix: `locals` (the table `typeOfExpr`/`_Generic` consult for a name's static type) is
// now genuinely block-scoped. A shadowing declaration inside a nested block no longer leaks
// its type past that block's closing `}`. Before this fix, all three cases below disagreed
// with gcc.
test "_Generic after a shadowing block resolves to the OUTER type, not the shadow's" {
    // `x` is `int` again once the inner block closes, so gcc picks the `int:` association
    // (1). The bug this guards against: the parser kept the inner block's `double x`'s type
    // live, wrongly picking `double:` (2) instead.
    try expectAgrees(
        "int f(void){ int x = 5; { double x = 1.5; } return _Generic(x, int: 1, double: 2, default: 0); }",
        &.{},
    ); // 1
}
test "typeof(x) after a shadowing block resolves to the OUTER type, not the shadow's" {
    // Same shadow shape as above, but through `typeof`/`sizeof` instead of `_Generic`: `x` is
    // `int` (4 bytes) again once the inner block closes, not the shadow's `double` (8 bytes).
    try expectAgreesGnu99(
        "int f(void){ int x = 5; { double x = 1.5; } typeof(x) y = 10; return sizeof(y); }",
        &.{},
    ); // 4
}
test "_Generic INSIDE a shadowing block still resolves to the shadow's type" {
    // The positive counterpart of the two tests above: while still lexically inside the
    // inner block, `x` genuinely is the shadow's type (`long`). Block scoping must not hide
    // the shadow from code that is still within its own scope, only from code after it
    // closes.
    try expectAgrees(
        "int f(void){ int x = 5; { long x = 7; return _Generic(x, int: 1, long: 42, default: 0); } }",
        &.{},
    ); // 42
}

// `const`/`volatile` qualifiers parse, representation and parsing only. Const-correctness
// checking comes later. Every case here still lowers and behaves exactly as its unqualified
// equivalent would. `const`/`volatile` change nothing observable here, only what
// `ctype.CType`/`parser.Param`/`Stmt.decl`/`GlobalDecl` carry alongside the type (see
// `ctype.Quals`).
test "const and volatile qualifiers parse" {
    // A `const` global still folds to `.rodata` and reads back correctly, unchanged from
    // before. `is_const` routing, not `quals`, drives that.
    try expectAgrees("const int g = 5; int f(void){ return g; }", &.{}); // 5
    // `volatile` on a local changes nothing observable yet (no volatile IR flag until later).
    try expectAgrees("int f(int a){ volatile int x = a; return x + 1; }", &.{7}); // 8
    // `const int *p` (pointee const): reading through it works fine. No enforcement yet.
    try expectAgrees("int f(int a){ int x = a; const int *p = &x; return *p; }", &.{9}); // 9
    // `int *const p` (the pointer itself const, pointee plain `int`): writing through it to a
    // non-const int is fine either way. A later check will add the actual const-correctness
    // enforcement.
    try expectAgrees("int f(int a){ int x = a; int *const p = &x; *p = 3; return x; }", &.{1}); // 3
}

// The `volatile` IR flag: a read or write through a `volatile`-qualified lvalue now marks
// the emitted `load`/`store` `.volatile = true` (earlier work only parsed and checked
// `quals`. This is what makes it observable in the IR). Scans every block and instruction
// of the compiled function (mirrors `Function.blockInsts`/`opcode`) rather than asserting on
// a specific instruction index, since the exact instruction shape is an implementation
// detail.
fn hasVolatileAccess(func: *const ir.function.Function, want_load: bool, want_store: bool) bool {
    var has_load = false;
    var has_store = false;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .load => |l| if (l.@"volatile") {
                    has_load = true;
                },
                .store => |s| if (s.@"volatile") {
                    has_store = true;
                },
                else => {},
            }
        }
    }
    return (!want_load or has_load) and (!want_store or has_store);
}
test "volatile access sets the IR volatile flag" {
    const allocator = std.testing.allocator;
    var mod = try cc.compile(allocator, "int f(int a){ volatile int x = a; x = x + 1; return x; }");
    defer mod.deinit(allocator);
    try std.testing.expect(hasVolatileAccess(&mod.funcs[0].func, true, true));
}
test "non-volatile access leaves the flag false" {
    const allocator = std.testing.allocator;
    var mod = try cc.compile(allocator, "int f(int a){ int x = a; x = x + 1; return x; }");
    defer mod.deinit(allocator);
    try std.testing.expect(!hasVolatileAccess(&mod.funcs[0].func, true, false));
    try std.testing.expect(!hasVolatileAccess(&mod.funcs[0].func, false, true));
}

// The optimizer honors the volatile flag: mem2reg will not promote an alloca touched by a
// volatile load or store, and loadfwd will not forward or record a volatile access. Counts
// loads and stores of an `alloca` result anywhere in the function (mirrors
// `hasVolatileAccess`'s block/inst walk). A fully-promoted scalar leaves none.
fn countAllocaMemAccesses(func: *const ir.function.Function) usize {
    var count: usize = 0;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .load => |l| if (func.definingInst(l.ptr) != null and func.opcode(func.definingInst(l.ptr).?) == .alloca) {
                    count += 1;
                },
                .store => |s| if (func.definingInst(s.ptr) != null and func.opcode(func.definingInst(s.ptr).?) == .alloca) {
                    count += 1;
                },
                else => {},
            }
        }
    }
    return count;
}
test "volatile alloca not promoted by mem2reg" {
    const allocator = std.testing.allocator;

    var vmod = try cc.compile(allocator, "int f(int a){ volatile int x = a; x = x + 1; return x; }");
    defer vmod.deinit(allocator);
    _ = try opt.optimize(allocator, &vmod.funcs[0].func);
    // The volatile slot's load(s)/store(s) must survive optimization: neither promoted away by
    // mem2reg nor forwarded/dropped by loadfwd.
    try std.testing.expect(hasVolatileAccess(&vmod.funcs[0].func, true, true));
    try std.testing.expect(countAllocaMemAccesses(&vmod.funcs[0].func) > 0);

    var nmod = try cc.compile(allocator, "int f(int a){ int x = a; x = x + 1; return x; }");
    defer nmod.deinit(allocator);
    _ = try opt.optimize(allocator, &nmod.funcs[0].func);
    // The non-volatile twin is fully promoted: no loads/stores of the slot remain.
    try std.testing.expectEqual(@as(usize, 0), countAllocaMemAccesses(&nmod.funcs[0].func));
}
test "volatile value still correct" {
    try expectAgrees("int f(int a){ volatile int x = a; x = x + 1; return x; }", &.{5}); // 6
}

// Const-correctness checking: a write through a `const`-qualified lvalue is now
// `error.ConstAssign` instead of silently lowering (earlier work only parsed and
// represented `quals`. This is what gives them teeth). Non-const writes are entirely
// unaffected, byte-identical to before.
test "write to const local rejected" {
    try std.testing.expectError(error.ConstAssign, cc.compile(std.testing.allocator, "int f(void){ const int x = 5; x = 6; return x; }"));
}
test "write through pointer to const rejected" {
    try std.testing.expectError(error.ConstAssign, cc.compile(std.testing.allocator, "int f(int a){ int y = a; const int *p = &y; *p = 3; return *p; }"));
}
test "increment of const rejected" {
    try std.testing.expectError(error.ConstAssign, cc.compile(std.testing.allocator, "int f(void){ const int x = 5; x++; return x; }"));
}
test "non-const writes still work" {
    try expectAgrees("int f(int a){ int x = a; x = x + 1; return x; }", &.{5}); // 6
    try expectAgrees("int f(int a){ int x = a; int *const p = &x; *p = 9; return x; }", &.{1}); // 9 (pointee non-const)
}

// The `.c -> .o` object-emit glue (`target.native.writeObjectData`) proven end to end:
// compile C to a real ELF relocatable object, link it (with a hand-assembled `_start`
// stub) via the host's own `ld.zig`, and natively run the result, not just inspect its
// bytes. Host-gated to aarch64: the stub below is a hand-assembled AArch64 `.o`, and
// `ld.zig`/execution are aarch64-only today.

/// Assemble a `_start` ELF relocatable object: `bl main; mov x8, #93; svc #0`. It calls
/// whatever the linker resolves `main` to (an undefined, external symbol from this
/// object's point of view) and exits with its return value (already sitting in `x0`
/// per the AAPCS64 return-value register) via the `exit` syscall. Built with the host
/// backend's own `object.write` so the `bl` is a real `R_AARCH64_CALL26` relocation,
/// mirroring the hand-assembled stub in `aarch64/tests/native.zig`'s "object+ld+exec"
/// test, but linked as its own object rather than prepended to raw code, since this
/// test exercises `ld.linkObjects` across two real `.o`s.
fn startStubAArch64(allocator: std.mem.Allocator) ![]u8 {
    const encode = target.native.backend.encode;
    const object = target.native.backend.object;

    const code = [_]u32{
        encode.bl(0), // bl main (offset 0), patched by the linker's CALL26 reloc
        encode.movz(.x8, 93, 0), // x8 = 93 (the exit syscall number)
        encode.svc(0), // svc #0 -> exit(x0)
    };
    var text: [code.len * 4]u8 = undefined;
    for (code, 0..) |w, i| std.mem.writeInt(u32, text[i * 4 ..][0..4], w, .little);

    const symbols = [_]object.Symbol{
        .{ .name = "_start", .value = 0, .size = text.len, .kind = .func, .defined = true },
        .{ .name = "main", .size = 0, .kind = .notype, .defined = false },
    };
    const relocs = [_]object.Reloc{.{ .offset = 0, .symbol = 1, .type = .call26 }};
    return object.write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
}

/// Link `objs` into one executable image entering at `_start` (`ld.linkObjects` +
/// `ld.writeExecutable`), write it to a fresh temp file with executable permissions,
/// run it natively, and return its exit code. Mirrors `aarch64/tests/native.zig`'s
/// "object+ld+exec" test's link-and-run tail.
fn linkAndRun(allocator: std.mem.Allocator, io: std.Io, objs: []const []const u8) !u8 {
    const ld = @import("vulcan-link");
    const base: u64 = 0x400000;
    var image = try ld.linkObjects(allocator, objs, base);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(.aarch64, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });
    const proc = try std.process.run(allocator, io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    return switch (proc.term) {
        .exited => |code| code,
        else => error.BackendFailed,
    };
}

test "compile to object, link, and natively run" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // host-only
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // 1. compile source -> cc.Module
    var mod = try cc.compileWithOpts(allocator, "int add(int a, int b){ return a + b; } int main(void){ return add(1, 41); }", .{});
    defer mod.deinit(allocator);

    // 2. build ModuleFunction[] + ModuleData[] (reuse the buildModuleData pattern)
    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });
    var data = try buildModuleData(allocator, mod.data);
    defer freeModuleData(allocator, &data);

    // 3. emit the .o
    const obj = try target.native.writeObjectData(allocator, mfs.items, data.items);
    defer allocator.free(obj);
    try std.testing.expect(std.mem.eql(u8, obj[0..4], "\x7fELF")); // real ELF

    // 4. link .o + a tiny _start stub, run natively, expect exit 42.
    const stub = try startStubAArch64(allocator);
    defer allocator.free(stub);
    const exit_code = try linkAndRun(allocator, io, &.{ stub, obj });
    try std.testing.expectEqual(@as(u8, 42), exit_code);
}

// A parenthesized function designator as a call's callee. `(add)(1, 41)` only parses once
// `Expr.call.callee` is a full expression, rather than a bare identifier captured at the
// call site. Same compile/link/native-run shape as the test just above, swapping only the
// call site's spelling. It still resolves through the same `findFunc` path, since the
// callee expression is still a `.name` underneath the parens.
test "parenthesized callee (add)(1, 41) agrees with a plain call, natively runs to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // host-only
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var mod = try cc.compileWithOpts(allocator, "int add(int a, int b){ return a + b; } int main(void){ return (add)(1, 41); }", .{});
    defer mod.deinit(allocator);

    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });
    var data = try buildModuleData(allocator, mod.data);
    defer freeModuleData(allocator, &data);

    const obj = try target.native.writeObjectData(allocator, mfs.items, data.items);
    defer allocator.free(obj);
    try std.testing.expect(std.mem.eql(u8, obj[0..4], "\x7fELF"));

    const stub = try startStubAArch64(allocator);
    defer allocator.free(stub);
    const exit_code = try linkAndRun(allocator, io, &.{ stub, obj });
    try std.testing.expectEqual(@as(u8, 42), exit_code);
}

// The `vcc` driver's non-`-c` LINK step goes through `ld.linkInputs`/`ld.Input` (a
// `.o`/`.a`-tagged union, command-line order), not the older bare-byte-slice
// `ld.linkObjects` the earlier test above exercises. This test proves that exact call path
// end to end: compile C to a real `.o`, link it against a hand-assembled `_start.o` via
// `linkInputs`, and natively run the result.

/// Like `linkAndRun`, but through `ld.linkInputs`/`ld.Input`, the multi-object link API
/// the `vcc` driver's link step actually calls, rather than `linkAndRun`'s
/// `ld.linkObjects([]const []const u8)`. Mirrors its link-and-run tail otherwise.
fn linkInputsAndRun(allocator: std.mem.Allocator, io: std.Io, inputs: []const @import("vulcan-link").Input) !u8 {
    const ld = @import("vulcan-link");
    const base: u64 = 0x400000;
    var image = try ld.linkInputs(allocator, inputs, base);
    defer image.deinit(allocator);
    const elf = try ld.writeExecutable(.aarch64, allocator, &image, "_start");
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.out", .data = elf, .flags = .{ .permissions = .executable_file } });
    const proc = try std.process.run(allocator, io, .{
        .argv = &.{"./a.out"},
        .cwd = .{ .dir = tmp.dir },
    });
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    return switch (proc.term) {
        .exited => |code| code,
        else => error.BackendFailed,
    };
}

test "vcc driver link path: linkInputs multi-object links start.o + main.o, natively runs to exit 42" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest; // hand-assembled AArch64 stub + native exec

    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // 1. compile source -> cc.Module (the same `int main(void){ return 42; }` the vcc
    // link-step smoke test also compiles, so the two exercise the same expectation).
    var mod = try cc.compileWithOpts(allocator, "int main(void){ return 42; }", .{});
    defer mod.deinit(allocator);

    // 2. build ModuleFunction[] + ModuleData[] and emit the .o, identical to the vcc
    // driver's own `-c`/link-step path (`frontends/vcc.zig`'s `buildLinkObject`).
    var mfs: std.ArrayList(target.native.ModuleFunction) = .empty;
    defer mfs.deinit(allocator);
    for (mod.funcs) |*nf| try mfs.append(allocator, .{ .name = nf.name, .func = &nf.func });
    var data = try buildModuleData(allocator, mod.data);
    defer freeModuleData(allocator, &data);

    const main_obj = try target.native.writeObjectData(allocator, mfs.items, data.items);
    defer allocator.free(main_obj);
    try std.testing.expect(std.mem.eql(u8, main_obj[0..4], "\x7fELF"));

    // 3. link start.o + main.o (in command order, as `.object` inputs) via `linkInputs`,
    // run natively, expect exit 42.
    const start_obj = try startStubAArch64(allocator);
    defer allocator.free(start_obj);
    const exit_code = try linkInputsAndRun(allocator, io, &.{ .{ .object = start_obj }, .{ .object = main_obj } });
    try std.testing.expectEqual(@as(u8, 42), exit_code);
}

// An automatic (stack) local's brace-list aggregate initializer, lowered as a runtime
// element-wise store sequence (`lowerAggregateInit`). Previously this was
// `error.Unsupported`, since only a `static` local or a file-scope global could fold one,
// at compile time.

test "local array aggregate initializer, fully listed" {
    try expectAgrees("int f(void){ int a[3] = {1,2,3}; return a[0]+a[1]+a[2]; }", &.{}); // 6
}

test "local array aggregate initializer, partial list zero-fills the rest" {
    try expectAgrees("int f(void){ int a[3] = {1}; return a[0]*100 + a[1]*10 + a[2]; }", &.{}); // 100
}

test "local struct aggregate initializer, fully listed" {
    // NOTE: the struct tag is defined at file scope, and only the automatic-local
    // variable is inside `f`. A bare `struct S{ ... };` tag-only declaration as its own
    // local statement is a separate, pre-existing `parseDeclStmt` gap. It always demands a
    // declarator name, unlike the file-scope item loop's own bare-`;`-after-type-spec
    // check. This is unrelated to the runtime aggregate-init engine under test. This still
    // exercises exactly what is targeted here: a local variable's runtime brace-list init.
    try expectAgrees("struct S{int x; int y;}; int f(void){ struct S s = {10,32}; return s.x+s.y; }", &.{}); // 42
}

test "local nested array aggregate initializer" {
    try expectAgrees("int f(void){ int a[2][2] = {{1,2},{3,4}}; return a[0][0]*1000+a[0][1]*100+a[1][0]*10+a[1][1]; }", &.{}); // 1234
}

test "local nested struct aggregate initializer" {
    try expectAgrees("struct P{int x; int y;}; struct Line{struct P a; struct P b;}; int f(void){ struct Line L = {{1,2},{3,4}}; return L.a.x+L.a.y+L.b.x+L.b.y; }", &.{}); // 10
}

test "local struct aggregate initializer, partial list zero-fills the rest" {
    try expectAgrees("struct S{int a; int b; int c;}; int f(void){ struct S s = {7}; return s.a*100+s.b*10+s.c; }", &.{}); // 700
}

// Designated initializers (`.field = ...`, `[index] = ...`, and a chain of either) on both
// the runtime-local path (`lower.lowerAggregateInit`) and the compile-time const-fold path
// (`consteval.evalArrayInit`/`evalStructInit`). Struct tags stay at FILE scope in every
// test below, the same pre-existing `parseDeclStmt` workaround the tests above already use.
// A bare LOCAL `struct S{...};` tag-only statement is a different, unrelated parser gap.

test "local struct designated initializer" {
    try expectAgrees("struct S{int a; int b; int c;}; int f(void){ struct S s = {.a=1,.c=3}; return s.a*100+s.b*10+s.c; }", &.{}); // 103 (b zero-filled)
}

test "local array designated initializer" {
    try expectAgrees("int f(void){ int a[5] = {[1]=5,[3]=7}; return a[1]+a[3]+a[0]; }", &.{}); // 12
}

test "local struct designator-then-positional" {
    try expectAgrees("struct S{int a; int b; int c;}; int f(void){ struct S s = {.b=2, 30}; return s.b*100+s.c; }", &.{}); // 230 (30 lands in c)
}

test "local array designated initializer, last-wins" {
    try expectAgrees("int f(void){ int a[3] = {[0]=1, [1]=2, [0]=9}; return a[0]*100+a[1]*10+a[2]; }", &.{}); // 920 (a[0] overwritten to 9)
}

test "global array designated initializer" {
    try expectAgrees("int g[3] = {[0]=9, [2]=3}; int f(void){ return g[0]*10+g[2]; }", &.{}); // 93 (consteval path)
}

test "global struct designated initializer" {
    try expectAgrees("struct S{int a; int b;}; struct S g = {.b=7}; int f(void){ return g.a+g.b; }", &.{}); // 7 (consteval path)
}

test "local nested designated initializer" {
    try expectAgrees("struct P{int x; int y;}; struct L{struct P a; struct P b;}; int f(void){ struct L v = {.b.x=5}; return v.b.x; }", &.{}); // 5 (chained .b.x)
}

// A fix for a critical bug: a chained designator split across separate top-level list items
// (`.b.x=5, .b.y=6`) used to lose data. Each top-level item re-zeroed the whole `.b`
// sub-object before writing its own one field, so the second item's write wiped out the
// first's. The gcc-differential tests prove both siblings now survive, on both the
// runtime-local path and the compile-time (global/`consteval`) path, plus an
// array-of-struct variant and a mix with the single-nested case that already worked.

test "local nested designated initializer, sibling fields survive" {
    try expectAgrees("struct P{int x; int y;}; struct L{struct P a; struct P b;}; int f(void){ struct L v = {.b.x=5, .b.y=6}; return v.b.x*10+v.b.y; }", &.{}); // 56
}

test "global nested designated initializer, sibling fields survive" {
    try expectAgrees("struct P{int x; int y;}; struct L{struct P a; struct P b;}; struct L gv = {.b.x=5, .b.y=6}; int f(void){ return gv.b.x*10+gv.b.y; }", &.{}); // 56 (consteval path)
}

test "local array-of-struct nested designated initializer, sibling fields survive" {
    try expectAgrees("struct P{int x; int y;}; int f(void){ struct P a[2] = {[1].x=3, [1].y=4}; return a[1].x*10+a[1].y; }", &.{}); // 34
}

test "local nested designated initializer, all-siblings mix" {
    try expectAgrees("struct P{int x; int y;}; struct L{struct P a; struct P b;}; int f(void){ struct L v = {.a.x=1, .a.y=2, .b.x=3, .b.y=4}; return v.a.x*1000+v.a.y*100+v.b.x*10+v.b.y; }", &.{}); // 1234
}

// More designated-initializer coverage: a struct-with-array field addressed through an `.arr[i]`
// designator (local + global consteval), and aggregate-leaf-vs-scalar last-wins in both orders.

test "struct-with-array designator (.arr[i]) local" {
    try expectAgrees("struct S{int arr[3]; int tag;}; int f(void){ struct S s = {.arr[1]=5, .tag=9}; return s.arr[1]*10+s.tag; }", &.{});
}

test "struct-with-array designator (.arr[i]) global consteval" {
    try expectAgrees("struct S{int arr[3]; int tag;}; struct S g = {.arr[1]=5, .tag=9}; int f(void){ return g.arr[1]*10+g.tag; }", &.{});
}

test "aggregate-leaf then sibling scalar designator, last-wins" {
    try expectAgrees("struct P{int x; int y;}; struct L{struct P b;}; int f(void){ struct L v = {.b = {5,6}, .b.x = 99}; return v.b.x*100+v.b.y; }", &.{});
}

test "scalar-then-aggregate-leaf designator (aggregate wins whole subobject)" {
    try expectAgrees("struct P{int x; int y;}; struct L{struct P b;}; int f(void){ struct L v = {.b.x = 99, .b = {5,6}}; return v.b.x*100+v.b.y; }", &.{});
}

// A fix: a BITFIELD struct initializer must fold LSB-first the same way the runtime local
// path stores it. Before the fix, the const-fold (global) path overwrote the shared storage
// unit un-shifted, so two bitfields in one unit clobbered each other, and a global
// disagreed with the identical initializer lowered as a local. Both paths are proven
// against gcc here, positional and designated.

test "global bitfield struct initializer, positional (consteval path)" {
    try expectAgrees("struct S{unsigned a:4; unsigned b:4;}; struct S g = {1,2}; int f(void){ return g.a*16+g.b; }", &.{}); // 18
}

test "global bitfield struct initializer, designated (consteval path)" {
    try expectAgrees("struct S{unsigned a:4; unsigned b:4;}; struct S g = {.b=2, .a=1}; int f(void){ return g.a*16+g.b; }", &.{}); // 18
}

test "local bitfield struct initializer agrees with the global path" {
    try expectAgrees("struct S{unsigned a:4; unsigned b:4;}; int f(void){ struct S s = {1,2}; return s.a*16+s.b; }", &.{}); // 18
}

// Compound literals `(type-name){ initializer-list }` (C99 6.5.2.5): an unnamed automatic
// object, initialized via the same `lowerAggregateInit` engine used above, and a genuine
// lvalue, so `&`/`.field`/`[index]` on one all work through the ordinary postfix loop. The
// decl-init case (`struct S s = (struct S){...};`) folds the literal straight into `s`'s
// own slot rather than materializing-then-copying (no struct-by-value support yet, deferred
// to later work). Every other use (address-of, index, member, as a bare expression
// statement) materializes its own fresh temp slot. Struct tags stay at FILE scope in every
// test below, the same pre-existing `parseDeclStmt` workaround the tests above already use.
// A bare LOCAL `struct S{...};` tag-only statement is a different, unrelated parser gap.

test "compound literal decl-init folds directly into the target slot" {
    try expectAgrees("struct S{int x; int y;}; int f(void){ struct S s = (struct S){10,32}; return s.x+s.y; }", &.{}); // 42
}

test "compound literal address-of" {
    try expectAgrees("int f(void){ int *p = &(int){5}; return *p; }", &.{}); // 5
}

test "compound literal array index" {
    try expectAgrees("int f(void){ return (int[3]){1,2,3}[1]; }", &.{}); // 2
}

test "compound literal designated struct field" {
    try expectAgrees("struct S{int a; int b;}; int f(void){ return (struct S){.b=7}.b; }", &.{}); // 7
}

test "compound literal member access, positional init" {
    try expectAgrees("struct S{int x; int y;}; int f(void){ return (struct S){1,2}.x; }", &.{}); // 1
}

// The comma operator (`a, b`: evaluate `a`, discard its value, `b`'s value and type are the
// whole expression's) at every C `expression` context: `for` clauses, a parenthesized
// group, a compound-assignment chain, and an array subscript. Each of these is a grammar
// context routed through `parseComma`, distinct from every comma-separator context (call
// args, initializer elements, declarator lists) below, which must keep working exactly as
// before. Those are the guard against a mis-route.

// NOTE: `int i, j;` (a comma-separated LOCAL declarator list) is a separate, unrelated
// parser gap (`parseDeclStmt` has no declarator-list loop, only file scope does), and a bare
// `;` empty statement (the `for`'s body here) is also unsupported. Neither is something this
// grammar split touches. Two separate declarations plus an empty `{}` body sidestep both
// while still exercising the for-loop's init/incr clauses, the actual thing under test here.
test "comma operator in for-loop init/incr clauses" {
    try expectAgrees("int f(void){ int i = 0; int j = 0; for (i = 0, j = 10; i < 3; i++, j--) {} return i * 100 + j; }", &.{}); // 307
}

test "comma operator in a parenthesized group" {
    try expectAgrees("int f(void){ int x = (1, 2, 3); return x; }", &.{}); // 3
}

test "comma operator chaining compound assignments" {
    try expectAgrees("int f(void){ int a = 5; return (a += 1, a += 2, a); }", &.{}); // 8
}

test "comma operator inside an array subscript" {
    try expectAgrees("int f(void){ int a[3] = {0}; a[0, 1] = 4; return a[1]; }", &.{}); // 4 (a[0,1] == a[1])
}

test "comma operator as the ternary middle operand" {
    // The `?:` middle is a full C `expression` (comma allowed), so `b, c` is the middle and
    // `d` is the third operand. A true condition selects the middle, whose value is `c`.
    try expectAgrees("int f(void){ int a = 5; int b = 2; int c = 9; int d = 100; return a ? b, c : d; }", &.{}); // 9
}

test "comma as a call-argument separator still gives a 2-arg call" {
    try expectAgrees("int add(int a, int b){ return a + b; } int f(void){ return add(1, 2); }", &.{}); // 3, NOT add((1,2))
}

test "comma as an initializer-list separator still gives 3 elements" {
    try expectAgrees("int f(void){ int a[3] = {1, 2, 3}; return a[0] + a[1] + a[2]; }", &.{}); // 6
}

test "wide char literal L'A' is a plain int code point" {
    try expectAgrees("typedef int wchar_t; int f(void){ wchar_t c = L'A'; return c; }", &.{}); // 65
}

test "sizeof a wide string literal counts 4-byte elements" {
    try expectAgrees("int f(void){ return sizeof(L\"ab\"); }", &.{}); // 12 (3 wchar_t elements * 4 bytes)
}

test "wide string literal decays to wchar_t* and indexes its elements" {
    try expectAgrees("typedef int wchar_t; int f(void){ wchar_t *s = L\"Z\"; return s[0]; }", &.{}); // 90
}

test "a lone identifier named L stays an identifier, not a wide prefix" {
    try expectAgrees("int f(void){ int L = 7; return L; }", &.{}); // 7
}

test "an identifier starting with L stays an identifier, not a wide prefix" {
    try expectAgrees("int f(void){ int Label = 3; return Label; }", &.{}); // 3
}

test "goto forward over dead code to a label" {
    try expectAgrees("int f(void){ int x = 0; goto skip; x = 99; skip: return x; }", &.{}); // 0
}

test "goto backward re-enters a labeled if for a manual loop" {
    try expectAgrees("int f(void){ int i = 0; int s = 0; loop: if (i < 5) { s += i; i++; goto loop; } return s; }", &.{}); // 10
}

test "goto out of a nested block reaches an outer label" {
    try expectAgrees("int f(void){ int i = 0; { i = 5; goto out; } out: return i; }", &.{}); // 5
}

test "a label reached by fall-through, not by goto, still runs" {
    try expectAgrees("int f(void){ int x = 1; here: x = x + 1; if (x < 3) goto here; return x; }", &.{}); // 3
}

test "forward goto to a label nested inside dead code reaches it" {
    // The target label sits inside a block that follows the goto, so it is only reachable
    // through that otherwise-dead block. The lowering must still populate and terminate the
    // label's block, or the IR is invalid.
    try expectAgrees("int f(void){ goto skip; { skip: return 1; } }", &.{}); // 1
}

// A small all-integer struct passed and received by value, decomposed into 1 or 2 scalar
// `i64` register arguments per the real ABI (`abi.classify`'s `.registers` plan, every
// eightbyte `.integer`). `expectAgrees`/`expectAgreesBits` always call the entry point
// named `f` with zero int args, so each test below names the struct-by-value-taking
// function `g` and calls it from `f`. Struct tags stay at FILE scope, the same
// pre-existing `parseDeclStmt` workaround the tests above already use.

test "struct-by-value: one eightbyte (two int fields packed into one i64 arg)" {
    try expectAgrees("struct P{int x; int y;}; int g(struct P p){ return p.x + p.y; } int f(void){ struct P p = {3, 4}; return g(p); }", &.{}); // 7
}

test "struct-by-value: two eightbytes (two long fields, two i64 args)" {
    try expectAgreesBits("struct Q{long a; long b;}; long g(struct Q q){ return q.a*10 + q.b; } long f(void){ struct Q q = {3, 4}; return g(q); }", &.{}, 64); // 34
}

test "struct-by-value: the callee gets its OWN copy (mutating it never leaks to the caller)" {
    try expectAgrees("struct P{int x; int y;}; int g(struct P p){ p.x = 99; return p.x; } int f(void){ struct P p = {1, 2}; int r = g(p); return r*1000 + p.x; }", &.{}); // 99001
}

// A large struct (more than 16 bytes) passed and received by value. The real ABI
// (`abi.classify`'s `.memory_ref` plan on aarch64/riscv64) can no longer fit the struct
// into 1 or 2 integer registers, so the caller copies it into a fresh temp and passes the
// temp's address as a single pointer argument, and the callee copies from that pointer into
// its own slot. Same naming convention as the tests above: `f` is the zero-arg entry point,
// `g` is the struct-by-value-taking function.

test "struct-by-value: a large (more than 16 byte) struct passed by reference" {
    try expectAgreesBits("struct Big{long a; long b; long c;}; long g(struct Big b){ return b.a + b.b + b.c; } long f(void){ struct Big b = {10, 20, 30}; return g(b); }", &.{}, 64); // 60
}

test "struct-by-value: a large struct's by-reference pass still copies (callee mutation never leaks)" {
    try expectAgreesBits("struct Big{long a; long b; long c;}; long g(struct Big b){ b.a = 99; return b.a; } long f(void){ struct Big b = {1, 2, 3}; long r = g(b); return r*1000 + b.a; }", &.{}, 64); // 99001
}

// A small all-integer struct returned by value in a register pair (the real ABI
// `.registers` return plan). The callee loads its eightbytes and returns them in x0(/x1)
// via the multi-value `Ret.many`. The caller allocates a destination slot, threads it to
// the call as `ret_dest`, and the aarch64 backend stores the return registers into it after
// the `bl`. The frontend yields that slot as the call's lvalue, so a struct-returning call
// works as an assignment RHS, as a `.member` base, and passed onward as a by-value argument.
// Struct tags stay at FILE scope (the same pre-existing workaround the tests above use),
// and `mk` is the struct-returning function called from the zero-arg entry `f`.

test "struct-return: one integer eightbyte returned by value" {
    try expectAgrees("struct P{int x; int y;}; struct P mk(void){ struct P p; p.x = 3; p.y = 4; return p; } int f(void){ struct P q = mk(); return q.x + q.y; }", &.{}); // 7
}

test "struct-return: two integer eightbytes returned by value (x0:x1)" {
    try expectAgreesBits("struct Q{long a; long b;}; struct Q mk(void){ struct Q q; q.a = 3; q.b = 4; return q; } long f(void){ struct Q r = mk(); return r.a*10 + r.b; }", &.{}, 64); // 34
}

test "struct-return: a member read straight off the returning call" {
    try expectAgrees("struct P{int x; int y;}; struct P mk(void){ struct P p; p.x = 5; p.y = 6; return p; } int f(void){ return mk().x; }", &.{}); // 5
}

test "struct-return: the returned struct passed straight onward by value" {
    try expectAgrees("struct P{int x; int y;}; struct P mk(void){ struct P p; p.x = 5; p.y = 6; return p; } int g(struct P p){ return p.x*10 + p.y; } int f(void){ return g(mk()); }", &.{}); // 56
}

// A struct larger than 16 bytes returned by value through a hidden result pointer (the ABI
// `.sret` return plan). The caller allocates the destination slot and passes its address as
// the hidden pointer, and the callee copies its return value through that pointer. On
// aarch64 the hidden pointer travels in the AAPCS64 indirect-result register `x8` (outside
// the x0-x7 argument pool), so these exercise the one aarch64-only backend change.
// `struct Big` is 24 bytes (three longs), so it is `.sret` on aarch64.

test "struct-return: a 24-byte struct returned by value through the hidden pointer" {
    try expectAgreesBits("struct Big{long a; long b; long c;}; struct Big mk(void){ struct Big b; b.a = 10; b.b = 20; b.c = 30; return b; } long f(void){ struct Big r = mk(); return r.a + r.b + r.c; }", &.{}, 64); // 60
}

// A struct with FLOAT-class eightbytes returned by value. The callee loads each eightbyte
// at its own class (an integer eightbyte becomes `i64`, an `.sse` one becomes `f32`/`f64`)
// and returns the mix through `Ret.many`. Each backend places the return values into two
// register banks (integer x0/x1, floating v0..v3 on aarch64) by each value's own type, and
// the caller stores each return register of its bank into the destination slot at the
// eightbyte's offset. On aarch64 only an HFA (all-float, up to 4 members) uses the FP
// return registers. A mixed int/float struct is not an HFA and returns in the integer pair,
// so the mixed case here still exercises the integer path on the host, and the GP+FP mix
// runs cross-arch (external_linkage). Same file-scope struct-tag convention and
// `mk`-returns-the-struct shape as the tests above.

test "struct-return: an all-double HFA returned by value (v0/v1)" {
    try expectAgreesDouble("struct F{double a; double b;}; struct F mk(void){ struct F v; v.a = 1.5; v.b = 2.5; return v; } double f(void){ struct F r = mk(); return r.a + r.b; }", &.{}); // 4.0
}

test "struct-return: a single-double struct returned by value (one FP register)" {
    try expectAgreesDouble("struct F{double a;}; struct F mk(void){ struct F v; v.a = 3.25; return v; } double f(void){ struct F r = mk(); return r.a; }", &.{}); // 3.25
}

test "struct-return: an all-float HFA of four members returned by value (v0..v3)" {
    try expectAgreesFloat("struct H{float a; float b; float c; float d;}; struct H mk(void){ struct H v; v.a = 1.0f; v.b = 2.0f; v.c = 4.0f; v.d = 8.0f; return v; } float f(void){ struct H r = mk(); return r.a + r.b + r.c + r.d; }", &.{}); // 15.0
}

test "struct-return: a two-float HFA returned by value (v0/v1)" {
    try expectAgreesFloat("struct G{float a; float b;}; struct G mk(void){ struct G v; v.a = 1.25f; v.b = 2.5f; return v; } float f(void){ struct G r = mk(); return r.a + r.b; }", &.{}); // 3.75
}

test "struct-return: a mixed int/double struct returned by value" {
    try expectAgreesDouble("struct M{int i; double d;}; struct M mk(void){ struct M v; v.i = 3; v.d = 1.5; return v; } double f(void){ struct M r = mk(); return r.i + r.d; }", &.{}); // 4.5
}

test "struct-return sret: a member read straight off the returning call" {
    try expectAgreesBits("struct Big{long a; long b; long c;}; struct Big mk(void){ struct Big b; b.a = 10; b.b = 20; b.c = 30; return b; } long f(void){ return mk().b; }", &.{}, 64); // 20
}

test "struct-return sret: the returned struct passed straight onward by value" {
    try expectAgreesBits("struct Big{long a; long b; long c;}; struct Big mk(void){ struct Big b; b.a = 10; b.b = 20; b.c = 30; return b; } long g(struct Big b){ return b.a + b.b + b.c; } long f(void){ return g(mk()); }", &.{}, 64); // 60
}

test "struct-return sret: an sret function that also takes a register argument" {
    // The sret hidden pointer and a real scalar argument are live at the same call. On aarch64
    // the hidden pointer must go to x8 while the real argument `z` goes to x0, so this exercises
    // the shift-by-one that mk(void) cannot. 7 + 14 + 21 = 42.
    try expectAgreesBits("struct Big{long a; long b; long c;}; struct Big mk(long z){ struct Big b; b.a = z; b.b = z*2; b.c = z*3; return b; } long f(void){ struct Big r = mk(7); return r.a + r.b + r.c; }", &.{}, 64); // 42
}

// A struct with a FLOAT-class (`.sse`) eightbyte passed and received by value. On this
// aarch64 host `struct F`/`struct M`/`struct H` below are the AAPCS64 Homogeneous Float
// Aggregate case (`abi.classify`'s `.sse` eightbytes). Each element loads and stores as its
// own `f32`/`f64` IR value at the eightbyte's offset, and the aarch64 backend places it in
// the next `v` register positionally with no backend change. Same `f`/`g` naming
// convention as the tests above: `f` is the zero-arg entry point, `g` is the
// struct-by-value-taking function.

test "struct-by-value: two double fields (HFA of two sse eightbytes, d0/d1)" {
    try expectAgreesDouble("struct F{double a; double b;}; double g(struct F v){ return v.a + v.b; } double f(void){ struct F v; v.a = 1.5; v.b = 2.5; return g(v); }", &.{}); // 4.0
}

test "struct-by-value: a mixed integer+double struct (not an HFA, whole struct in GPRs)" {
    // NOTE: `(double)v.i` (an explicit cast applied straight to a struct member access) hits
    // a separate, pre-existing `error.Unsupported` unrelated to this test. It also fails for
    // `(long)v.i`, with no struct-by-value arg/param involved at all, and reproduces on a
    // struct with a single `int` field and no function call. Worked around here with an
    // implicit int-to-double conversion through a plain local (`double di = v.i;`), which is
    // unaffected and still exercises the mixed-eightbyte-classes struct-by-value arg this
    // test is actually about.
    try expectAgreesDouble("struct M{int i; double d;}; double g(struct M v){ double di = v.i; return di + v.d; } double f(void){ struct M v; v.i = 3; v.d = 1.5; return g(v); }", &.{}); // 4.5
}

test "struct-by-value: four float fields (HFA of four sse eightbytes, s0..s3, 4-byte offsets)" {
    try expectAgreesFloat("struct H{float a; float b; float c; float d;}; float g(struct H v){ return v.a+v.b+v.c+v.d; } float f(void){ struct H v; v.a = 1.0f; v.b = 2.0f; v.c = 3.0f; v.d = 4.0f; return g(v); }", &.{}); // 10.0f
}

test "struct-by-value: the sse-eightbyte callee gets its OWN copy (mutating it never leaks)" {
    try expectAgreesDouble("struct F{double a; double b;}; double g(struct F v){ v.a = 99.0; return v.a; } double f(void){ struct F v; v.a = 1.0; v.b = 2.0; double r = g(v); return r + v.a; }", &.{}); // 100.0
}

// A float-struct cross-arch differential. Every test above exercises struct-by-value
// arguments or returns separately. This one composes both in a single program: a
// `struct {double x, y;}` is returned by one function and passed to another, so the
// `.sse`-eightbyte arg path and the `.sse`-eightbyte return path share one destination slot
// and must agree on its layout. `external_linkage.zig` runs the same two-function shape
// cross-arch under qemu (x86_64 xmm/riscv64 fa registers). This is its aarch64-native
// gcc-diff twin.

test "struct-by-value + struct-return: an sse-eightbyte struct made by one call and consumed by another" {
    try expectAgreesDouble(
        "struct F{double x; double y;};" ++
            " struct F make(double x, double y){ struct F v; v.x = x; v.y = y; return v; }" ++
            " double addv(struct F v){ return v.x + v.y; }" ++
            " double f(void){ struct F v = make(20.0, 22.0); return addv(v); }",
        &.{},
    ); // 42.0
}

// A whole-milestone integration test: a small all-integer struct, a large (over-16-byte,
// by-reference/sret) struct, an all-double (HFA) struct, and a mixed int+double struct,
// each made by a struct-returning call and consumed by a struct-by-value argument, all in
// one program. Every piece was proven individually by the tests above. This proves they
// compose: four different `abi.classify` plans (`.registers` int, `.memory_ref`/`.sret`,
// `.registers` sse HFA, `.registers` mixed) live in the same function bodies and the same
// call sequence without one plan's destination-slot handling corrupting another's.
//
// Each piece is wrapped in its own zero-arg `doX` helper (`doSmall`/`doBig`/`doF`/
// `doMixed`), and the entry point `f` just sums their four `int` results. `f`'s own body
// never holds a struct or `double` local at all. This dodges a separate, pre-existing,
// unrelated register-allocator limit this integration test surfaced while bisecting an
// unrelated failure: combining two float-class (`.sse`) struct-by-value round trips (the
// HFA `struct F` and the mixed `struct Mixed`) in the same function body hits
// `wimmer.zig`'s `spillCurrent` (`error.Unsupported` on a same-position must-have demand
// exceeding the register pool, the same "too many live params" limit already documented on
// i386's variadic path), on the optimized (`mem2reg`) path only. This is a real, but
// unrelated, register-allocator capacity gap, not fixed here, since it is out of scope: a
// general regalloc capacity issue, not a struct-by-value ABI concern. Every `doX` helper
// individually matches an already-proven shape, so composing them behind one entry point
// still proves the four ABI plans coexist correctly in one compiled program, just not all
// crammed into one function's frame.

test "struct-by-value integration: small int + large by-ref/sret + float HFA + mixed structs, args and returns together" {
    try expectAgrees(
        "struct Small{int x; int y;};" ++
            " struct Big{long a; long b; long c;};" ++
            " struct F{double x; double y;};" ++
            " struct Mixed{int i; double d;};" ++
            " struct Small mkSmall(int x, int y){ struct Small s; s.x=x; s.y=y; return s; }" ++
            " int sumSmall(struct Small s){ return s.x + s.y; }" ++
            " int doSmall(void){ return sumSmall(mkSmall(1, 2)); }" ++
            " struct Big mkBig(long a, long b, long c){ struct Big g; g.a=a; g.b=b; g.c=c; return g; }" ++
            " long sumBig(struct Big g){ return g.a + g.b + g.c; }" ++
            " int doBig(void){ long b = sumBig(mkBig(10, 10, 11)); return (int)b; }" ++
            // `double s = sumF(...); return (int)s;` (a cast of a plain local, not of the
            // call itself) works around another pre-existing, unrelated parser gap this
            // integration test surfaced: a cast's operand parses through `parseAtom` (see
            // `parser.zig`'s `.lparen`/cast-vs-parenthesized-expr case), which stops before
            // the postfix chain (`parsePrimary`'s `()`/`[]`/`.`/`->` loop). So `(int)g()`
            // misparses as `((int)g)()` (cast `g` first, then call the cast result) instead
            // of C's `(int)(g())`. This is the same root cause as the already-documented
            // "cast of a struct member access" gap the tests above route around the
            // identical way (`(double)v.i` misparses as `((double)v).i`). It is confirmed
            // broader here, since it reproduces with no structs at all, on a bare `(int)g()`.
            // Not fixed here, since it is out of scope: a general parser precedence fix, not
            // a struct-by-value ABI concern.
            " struct F mkF(double x, double y){ struct F v; v.x=x; v.y=y; return v; }" ++
            " double sumF(struct F v){ return v.x + v.y; }" ++
            " int doF(void){ double s = sumF(mkF(1.5, 2.5)); return (int)s; }" ++
            " struct Mixed mkMixed(int i, double d){ struct Mixed m; m.i=i; m.d=d; return m; }" ++
            " double sumMixed(struct Mixed m){ double di = m.i; return di + m.d; }" ++
            " int doMixed(void){ double s = sumMixed(mkMixed(3, 1.5)); return (int)s; }" ++
            " int f(void){ return doSmall() + doBig() + doF() + doMixed(); }",
        &.{},
    ); // 3 + 31 + 4 + 4 = 42
}

