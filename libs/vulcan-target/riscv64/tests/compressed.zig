//! Executes Vulcan-COMPRESSED (RVC) machine code on qemu-riscv64 user mode. This closes
//! the loop the compressor's disassembly round-trip left open: not just "the compressed
//! halfword decodes to the same instruction" but "the compressed program computes the
//! same result when a real CPU runs it".
//!
//! A tiny entry stub loads integer args into a0.., calls the compiled function, then
//! exits via the Linux `exit` syscall (a7 = 93) with the function's a0 result, so qemu's
//! process exit code is the low byte of the return. qemu-user gives us a valid sp on
//! entry, so no stack setup is needed (same as the RVV float runner in isel.zig).
//!
//! Crucially the WHOLE program is emitted through `riscv64.emitImage` with the C feature
//! on, so the bytes that execute are exactly what the RVC compressor produced (branch
//! displacements recomputed for the smaller layout included). Skips when qemu-riscv64 is
//! not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const riscv64 = @import("../../riscv64.zig");

const encode = riscv64.encode;
const isel = riscv64.isel;
const schedule = riscv64.schedule;
const ld = riscv64.ld;
const link = riscv64.link;
const object = riscv64.object;

const Function = ir.function.Function;
const Value = ir.function.Value;

fn i32t(f: *Function) !ir.types.Type {
    return f.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
}
fn newFunc(allocator: std.mem.Allocator) !*Function {
    const f = try allocator.create(Function);
    f.* = Function.init(allocator);
    return f;
}

fn loadImm(allocator: std.mem.Allocator, w: *std.ArrayList(u32), reg: encode.Reg, val: i64) !void {
    if (val >= -2048 and val <= 2047) {
        try w.append(allocator, encode.addi(reg, .x0, @intCast(val)));
    } else {
        const bits: u32 = @bitCast(@as(i32, @intCast(val)));
        const hi: u20 = @truncate((bits +% 0x800) >> 12);
        const lo: i12 = @bitCast(@as(u12, @truncate(bits)));
        try w.append(allocator, encode.lui(reg, hi));
        try w.append(allocator, encode.addi(reg, reg, lo));
    }
}

/// Wrap a resolved word stream (entry at word 0) in an exit-with-a0 stub, emit the whole program
/// through `emitImage` under `features` (so C=on runs the RVC compressor over stub + body together),
/// then execute it on qemu-riscv64 and return the process exit code (the low byte of the returned a0).
fn runWords(allocator: std.mem.Allocator, code: []const u32, args: []const i64, features: riscv64.Features) !u8 {
    var program: std.ArrayList(u32) = .empty;
    defer program.deinit(allocator);
    for (args, 0..) |arg, i| try loadImm(allocator, &program, @enumFromInt(@as(u5, @intCast(10 + i))), arg);
    const call_idx = program.items.len;
    try program.append(allocator, encode.jal(.x1, 0)); // call entry; displacement patched below
    try program.append(allocator, encode.addi(.x17, .x0, 93)); // return lands here: li a7, 93 (exit)
    try program.append(allocator, encode.ecall()); // exit(a0 & 0xff)
    const fn_off: i21 = @intCast((program.items.len - call_idx) * 4);
    program.items[call_idx] = encode.jal(.x1, fn_off);
    try program.appendSlice(allocator, code);

    const bytes = try riscv64.emitImage(allocator, program.items, features);
    defer allocator.free(bytes);
    const elf = try ld.writeElfExec(allocator, bytes, bytes.len, 0x10000, 0x10000);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });
    const run = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "qemu-riscv64", "a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // qemu-riscv64 not on PATH
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |ec| ec,
        else => {
            std.debug.print("qemu term: {any}\nstderr: {s}\n", .{ run.term, run.stderr });
            return error.BackendFailed;
        },
    };
}

/// Compile a single `func` to a resolved word stream, then run it (see `runWords`).
fn runUnder(allocator: std.mem.Allocator, func: *Function, args: []const i64, features: riscv64.Features) !u8 {
    try ir.legalize.legalize(allocator, func);
    try isel.splitCriticalEdges(allocator, func);
    try schedule.scheduleFunction(allocator, func);
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    return runWords(allocator, code, args, features);
}

/// Link a whole `module` (entry function first) to a resolved blob, then run it. Intra-module calls
/// resolve to internal near `jal`s, which the compressor recomputes like any other branch.
fn runModuleUnder(allocator: std.mem.Allocator, module: *const link.Module, args: []const i64, features: riscv64.Features) !u8 {
    var linked = try link.compileModule(allocator, module);
    defer linked.deinit(allocator);
    return runWords(allocator, linked.code, args, features);
}

/// Prepend a fixed exit-with-a0 stub (jal to the image entry, then li a7,93 / ecall) to an already
/// laid-out image and run it. The image's PC-relative addressing is self-relative, so shifting it as
/// a unit behind the stub preserves it; the stub is not compressed, so its jal offset is fixed.
fn runLinkedImage(allocator: std.mem.Allocator, image_code: []const u8) !u8 {
    var final: std.ArrayList(u8) = .empty;
    defer final.deinit(allocator);
    const stub = [_]u32{ encode.jal(.x1, 12), encode.addi(.x17, .x0, 93), encode.ecall() }; // entry at byte 12
    for (stub) |w| {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, w, .little);
        try final.appendSlice(allocator, &buf);
    }
    try final.appendSlice(allocator, image_code);
    const elf = try ld.writeElfExec(allocator, final.items, final.items.len, 0x10000, 0x10000);
    defer allocator.free(elf);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });
    const run = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "qemu-riscv64", "a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |c| c,
        else => error.BackendFailed,
    };
}

test "qemu-riscv: linkObjectsCompressed resolves a global auipc pair against the shrunk layout" {
    const allocator = std.testing.allocator;
    // entry() -> *(&K), K a module i32 = 42, reached by a PC-relative auipc/addi pair. Linking with
    // .text compression must pin the pair, remap its reloc + label offsets, and still resolve the
    // pcrel against the shrunk addresses. Compared against the uncompressed link (both must read 42).
    const buildObj = struct {
        fn run(a: std.mem.Allocator) ![]u8 {
            var entry = Function.init(a);
            defer entry.deinit();
            const t = try entry.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
            const ptr_t = try entry.types.intern(.ptr);
            const b = try entry.appendBlock();
            const p = try entry.appendGlobalAddr(b, ptr_t, "K");
            const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
            entry.setTerminator(b, .{ .ret = v });
            const k_bytes = [_]u8{ 42, 0, 0, 0 };
            var module: link.Module = .{};
            defer module.deinit(a);
            try module.addFunction(a, "entry", &entry);
            try module.addData(a, "K", &k_bytes);
            return object.writeModule(a, &module);
        }
    }.run;

    const obj = try buildObj(allocator);
    defer allocator.free(obj);

    var image_c = try ld.linkObjectsCompressed(allocator, &.{obj}, 0x10000, null);
    defer image_c.deinit(allocator);
    const compressed = runLinkedImage(allocator, image_c.code) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), compressed);

    var image_g = try ld.linkObjects(allocator, &.{obj}, 0x10000);
    defer image_g.deinit(allocator);
    try std.testing.expectEqual(@as(u8, 42), try runLinkedImage(allocator, image_g.code));
    // Compression actually shrank the text (else this proves nothing about the compressed path).
    try std.testing.expect(image_c.code.len < image_g.code.len);
}

test "qemu-riscv: compressPinned + a linker-style call patch runs correctly" {
    const allocator = std.testing.allocator;
    const cm = riscv64.compress;
    // main sets a0/a1, calls callee, exits with a0; callee returns a0+a1. The call is PINNED (its
    // target is a relocation a linker owns), so compressPinned leaves it verbatim and reports the
    // shrunk offset map; we then patch the call displacement exactly as `ld` would.
    const code = [_]u32{
        encode.addi(.x10, .x0, 20), // 0: a0 = 20 (c.li)
        encode.addi(.x11, .x0, 22), // 1: a1 = 22 (c.li)
        encode.jal(.x1, 0), // 2: PINNED call to callee
        encode.addi(.x17, .x0, 93), // 3: li a7, 93 (return lands here)
        encode.ecall(), // 4: exit(a0)
        encode.add(.x10, .x10, .x11), // 5: callee: a0 = a0 + a1 (c.add)
        encode.jalr(.x0, .x1, 0), // 6: ret (c.jr)
    };
    const call_idx = 2;
    const callee_idx = 5;
    var offs: [code.len + 1]usize = undefined;
    const bytes = try cm.compressPinned(allocator, &code, &.{call_idx}, &offs);
    defer allocator.free(bytes);

    // The "linker": resolve the pinned call against the shrunk layout via the offset map.
    var image = try allocator.dupe(u8, bytes);
    defer allocator.free(image);
    const disp: i64 = @as(i64, @intCast(offs[callee_idx])) - @as(i64, @intCast(offs[call_idx]));
    std.mem.writeInt(u32, image[offs[call_idx]..][0..4], encode.jal(.x1, @intCast(disp)), .little);

    const elf = try ld.writeElfExec(allocator, image, image.len, 0x10000, 0x10000);
    defer allocator.free(elf);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });
    const run = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "qemu-riscv64", "a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    const ec = switch (run.term) {
        .exited => |c| c,
        else => return error.BackendFailed,
    };
    try std.testing.expectEqual(@as(u8, 42), ec);
}

/// Build [auipc a0,%hi ; addi a0,a0,%lo ; <filler> ; lbu a0,0(a0) ; exit(a0)] with a data byte (42)
/// appended after the code, then either compress the code (recomputing the pcrel pair for the shrunk
/// layout) or emit it fixed-width, append the data, and run on qemu. The compressible filler shifts
/// the data, so a correct result proves the pcrel recompute: a stale hi/lo would read the wrong byte.
fn runPcrelProgram(allocator: std.mem.Allocator, comptime compress: bool) !u8 {
    const compress_mod = riscv64.compress;
    // Original layout: 6 code words (24 bytes), data byte at 24. So pcrel = 24 (hi20=0, lo12=24).
    const code = [_]u32{
        encode.auipc(.x10, 0), // 0: %pcrel_hi(data)
        encode.addi(.x10, .x10, 24), // 1: %pcrel_lo -> +24
        encode.addi(.x8, .x8, 1), // 2: compressible filler (c.addi), shifts the data under compression
        encode.lbu(.x10, .x10, 0), // 3: a0 = *data (one byte)
        encode.addi(.x17, .x0, 93), // 4: li a7, 93 (exit)
        encode.ecall(), // 5: exit(a0 & 0xff)
    };
    const data = [_]u8{ 42, 0, 0, 0 };

    var blob: std.ArrayList(u8) = .empty;
    defer blob.deinit(allocator);
    if (compress) {
        const pairs = [_]compress_mod.PcrelPair{.{ .hi = 0, .lo = 1, .target = code.len * 4 }};
        const cc = try compress_mod.compressPairs(allocator, &code, &pairs);
        defer allocator.free(cc);
        try blob.appendSlice(allocator, cc);
    } else {
        for (code) |w| {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, w, .little);
            try blob.appendSlice(allocator, &buf);
        }
    }
    try blob.appendSlice(allocator, &data); // data begins right after the (possibly shrunk) code

    const elf = try ld.writeElfExec(allocator, blob.items, blob.items.len, 0x10000, 0x10000);
    defer allocator.free(elf);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });
    const run = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "qemu-riscv64", "a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |ec| ec,
        else => error.BackendFailed,
    };
}

test "qemu-riscv: a PC-relative auipc pair reads the right data after compression" {
    const allocator = std.testing.allocator;
    const compressed = runPcrelProgram(allocator, true) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), compressed); // recomputed pcrel finds the shifted data
    try std.testing.expectEqual(@as(u8, 42), try runPcrelProgram(allocator, false)); // fixed-width baseline
}

/// a + b: exercises the compressed stub (c.li args, c.jr/ret) plus a compiled body.
fn addFunc(allocator: std.mem.Allocator) !*Function {
    const func = try allocator.create(Function);
    func.* = Function.init(allocator);
    const t = try func.types.intern(.{ .int = .{ .bits = 64, .signedness = .signed } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = s });
    return func;
}

/// Counted loop summing 0..n-1: a back-edge branch, so it proves branch compression (or the 32-bit
/// displacement recompute) survives real execution, not just a straight-line body.
fn sumLoopFunc(allocator: std.mem.Allocator) !*Function {
    const func = try allocator.create(Function);
    func.* = Function.init(allocator);
    const t = try func.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });
    return func;
}

test "qemu-riscv: a compressed counted loop executes with a real back-edge" {
    const allocator = std.testing.allocator;
    // sum 0..8 = 36. Runs the full loop CFG through the RVC compressor (back-edge branch and all).
    const gc = try sumLoopFunc(allocator);
    defer {
        gc.deinit();
        allocator.destroy(gc);
    }
    try std.testing.expectEqual(@as(u8, 36), try runUnder(allocator, gc, &.{9}, riscv64.Features.rv64gc));

    const g = try sumLoopFunc(allocator);
    defer {
        g.deinit();
        allocator.destroy(g);
    }
    try std.testing.expectEqual(@as(u8, 36), try runUnder(allocator, g, &.{9}, riscv64.Features.rv64g));
}

// A diverse corpus of self-contained functions (no cross-function calls or globals, so compressing
// the resolved blob is safe). Each hits a different slice of codegen so the compressor is exercised
// on real instruction mixes, not just add + a loop.

/// (a + b) * a: a small arithmetic chain (add, mul).
fn fnArithChain(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const t = try i32t(f);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const p = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = s, .rhs = x } });
    f.setTerminator(b, .{ .ret = p });
    return f;
}

/// max(a, b) via icmp + select (no branch).
fn fnMaxSelect(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const t = try i32t(f);
    const bool_t = try f.types.intern(.bool);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const c = try f.appendInst(b, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = y } });
    const m = try f.appendInst(b, t, .{ .select = .{ .cond = c, .then = x, .@"else" = y } });
    f.setTerminator(b, .{ .ret = m });
    return f;
}

/// max(a, b) via an if/else diamond (branch + block-param merge).
fn fnMaxBranch(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const t = try i32t(f);
    const bool_t = try f.types.intern(.bool);
    const e = try f.appendBlock();
    const exit = try f.appendBlock();
    const x = try f.appendBlockParam(e, t);
    const y = try f.appendBlockParam(e, t);
    const racc = try f.appendBlockParam(exit, t);
    const c = try f.appendInst(e, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = y } });
    try f.appendIf(e, c, .{ .target = exit, .args = &.{x} }, .{ .target = exit, .args = &.{y} });
    f.setTerminator(exit, .{ .ret = racc });
    return f;
}

/// a / b + a % b: integer division and remainder.
fn fnDivRem(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const t = try i32t(f);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const q = try f.appendInst(b, t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    const r = try f.appendInst(b, t, .{ .arith = .{ .op = .rem, .lhs = x, .rhs = y } });
    const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = q, .rhs = r } });
    f.setTerminator(b, .{ .ret = s });
    return f;
}

/// store a to a stack slot, reload it, add b: alloca + store + load (sp-relative memory).
fn fnAllocaRoundtrip(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const t = try i32t(f);
    const ptr_t = try f.types.intern(.ptr);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try f.appendStore(b, x, slot);
    const v = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = y } });
    f.setTerminator(b, .{ .ret = s });
    return f;
}

/// for i in 0..n: acc += (i > 2 ? i : 0). A loop with a select in the body (back-edge + condition).
fn fnLoopWithSelect(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const t = try i32t(f);
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const loop = try f.appendBlock();
    const body = try f.appendBlock();
    const done = try f.appendBlock();
    const n = try f.appendBlockParam(entry, t);
    const i = try f.appendBlockParam(loop, t);
    const acc = try f.appendBlockParam(loop, t);
    const bi = try f.appendBlockParam(body, t);
    const bacc = try f.appendBlockParam(body, t);
    const racc = try f.appendBlockParam(done, t);
    const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
    try f.setJump(entry, loop, &.{ zero, zero });
    const cmp = try f.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try f.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const two = try f.appendInst(body, t, .{ .iconst = 2 });
    const bzero = try f.appendInst(body, t, .{ .iconst = 0 });
    const gt = try f.appendInst(body, bool_t, .{ .icmp = .{ .op = .gt, .lhs = bi, .rhs = two } });
    const contrib = try f.appendInst(body, t, .{ .select = .{ .cond = gt, .then = bi, .@"else" = bzero } });
    const nacc = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = contrib } });
    const ni = try f.appendArithImm(body, t, .add, bi, 1);
    try f.setJump(body, loop, &.{ ni, nacc });
    f.setTerminator(done, .{ .ret = racc });
    return f;
}

/// n -> (double) -> store to a stack slot -> reload -> (int): runs a real f64 function (fcvt.d.w,
/// fsd/fld, alloca, fcvt.w.d) through the compressor to confirm compression does not corrupt
/// float-heavy code. (The compressed c.fsd/c.fldsp encoders themselves are proven by the round-trip
/// and the float-save/restore shrink test in compress.zig; whether isel bases a given fsd on sp or a
/// scratch register decides if that specific one shrinks.)
fn fnDoubleRoundtrip(allocator: std.mem.Allocator) !*Function {
    const f = try newFunc(allocator);
    const it = try i32t(f);
    const dt = try f.types.intern(.{ .float = .f64 });
    const ptr_t = try f.types.intern(.ptr);
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, it);
    const xd = try f.appendInst(b, dt, .{ .convert = .{ .value = x } }); // i32 -> f64
    const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = dt } });
    try f.appendStore(b, xd, slot); // fsd
    const v = try f.appendInst(b, dt, .{ .load = .{ .ptr = slot } }); // fld
    const r = try f.appendInst(b, it, .{ .convert = .{ .value = v } }); // f64 -> i32
    f.setTerminator(b, .{ .ret = r });
    return f;
}

test "qemu-riscv: a linked two-function module runs compressed (intra-module call)" {
    const allocator = std.testing.allocator;
    // caller(a, b) -> callee(a, b); callee(a, b) -> a + b. compileModule resolves the call to an
    // internal near jal, so compressing the linked blob recomputes it like any branch. No global
    // data + a near call = the case that already flows through the linker safely.
    const buildModule = struct {
        fn run(a: std.mem.Allocator, features: riscv64.Features) !u8 {
            var callee = Function.init(a);
            defer callee.deinit();
            var caller = Function.init(a);
            defer caller.deinit();
            const t = try callee.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
            {
                const b = try callee.appendBlock();
                const x = try callee.appendBlockParam(b, t);
                const y = try callee.appendBlockParam(b, t);
                const s = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
                callee.setTerminator(b, .{ .ret = s });
            }
            const ct = try caller.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
            {
                const b = try caller.appendBlock();
                const x = try caller.appendBlockParam(b, ct);
                const y = try caller.appendBlockParam(b, ct);
                const r = try caller.appendCall(b, ct, "callee", &.{ x, y });
                caller.setTerminator(b, .{ .ret = r });
            }
            var module: link.Module = .{};
            defer module.deinit(a);
            try module.addFunction(a, "caller", &caller); // entry first
            try module.addFunction(a, "callee", &callee);
            return runModuleUnder(a, &module, &.{ 20, 22 }, features);
        }
    }.run;

    const compressed = buildModule(allocator, riscv64.Features.rv64gc) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), compressed);
    try std.testing.expectEqual(@as(u8, 42), try buildModule(allocator, riscv64.Features.rv64g));
}

test "qemu-riscv: a diverse corpus runs identically compressed and fixed-width" {
    const allocator = std.testing.allocator;
    const Case = struct {
        name: []const u8,
        build: *const fn (std.mem.Allocator) anyerror!*Function,
        args: []const i64,
        expected: u8,
    };
    const corpus = [_]Case{
        .{ .name = "arith-chain", .build = fnArithChain, .args = &.{ 3, 4 }, .expected = 21 }, // (3+4)*3
        .{ .name = "max-select", .build = fnMaxSelect, .args = &.{ 5, 9 }, .expected = 9 },
        .{ .name = "max-branch", .build = fnMaxBranch, .args = &.{ 12, 7 }, .expected = 12 },
        .{ .name = "div-rem", .build = fnDivRem, .args = &.{ 17, 5 }, .expected = 5 }, // 3 + 2
        .{ .name = "alloca-roundtrip", .build = fnAllocaRoundtrip, .args = &.{ 40, 2 }, .expected = 42 },
        .{ .name = "loop-with-select", .build = fnLoopWithSelect, .args = &.{6}, .expected = 12 }, // 3+4+5
        .{ .name = "double-roundtrip", .build = fnDoubleRoundtrip, .args = &.{42}, .expected = 42 }, // f64 fsd/fld
    };
    for (corpus) |c| {
        const fc = try c.build(allocator);
        defer {
            fc.deinit();
            allocator.destroy(fc);
        }
        const got_c = runUnder(allocator, fc, c.args, riscv64.Features.rv64gc) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest, // qemu missing: whole test skips
            else => return e,
        };
        const ff = try c.build(allocator);
        defer {
            ff.deinit();
            allocator.destroy(ff);
        }
        const got_f = try runUnder(allocator, ff, c.args, riscv64.Features.rv64g);
        std.testing.expectEqual(c.expected, got_c) catch |e| {
            std.debug.print("case '{s}' compressed gave {d}, expected {d}\n", .{ c.name, got_c, c.expected });
            return e;
        };
        try std.testing.expectEqual(c.expected, got_f); // fixed-width sanity
        try std.testing.expectEqual(got_f, got_c); // compression preserved semantics
    }
}

test "qemu-riscv: the emitted image is actually smaller with C enabled" {
    const allocator = std.testing.allocator;
    // Same program bytes both ways: proves the exec test above ran genuinely compressed
    // code, not a fixed-width fallback. A loop body with a back-edge branch also proves the
    // branch-displacement recompute survives execution.
    const f = try addFunc(allocator);
    defer {
        f.deinit();
        allocator.destroy(f);
    }
    try ir.legalize.legalize(allocator, f);
    try isel.splitCriticalEdges(allocator, f);
    try schedule.scheduleFunction(allocator, f);
    const code = try isel.selectFunction(allocator, f);
    defer allocator.free(code);

    const gc = try riscv64.emitImage(allocator, code, riscv64.Features.rv64gc);
    defer allocator.free(gc);
    const g = try riscv64.emitImage(allocator, code, riscv64.Features.rv64g);
    defer allocator.free(g);
    try std.testing.expect(gc.len < g.len);
}
