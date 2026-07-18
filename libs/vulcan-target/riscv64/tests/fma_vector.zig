//! RVV vector float FMA fusion, executed on qemu-riscv64 (the oracle). Builds a <4 x f32>
//! `a*b+c` / `a*b-c` / `c-a*b` - the vector mul immediately preceding its single-use vector
//! add/sub, exactly the shape `fusesIntoNextArith` (isel.zig) now also fuses for RVV into
//! `vfmacc`/`vfmsac`/`vfnmsac` - and checks EACH lane, bit-identically, against the host's
//! `@mulAdd`. Mirrors fma.zig's scalar version; the difference is the harness's user-mode float
//! ABI returns one scalar in fa0, not a whole vector register, so each lane is checked by
//! compiling and running a variant of the function that extracts and returns that lane (the
//! vector computation itself, and its operands, are identical across the four variants).
//!
//! Compilation here calls `isel.selectFunction` directly rather than `harness.runFuncFloat`
//! (which additionally runs `schedule.scheduleFunction`). This function is a single straight-
//! line block with no control flow, so neither legalize's constant-folding (it only folds
//! `iconst`, never `fconst` - see legalize.zig's `constValue`) nor critical-edge splitting apply;
//! skipping them changes nothing. Skipping the SCHEDULER, though, is deliberate: it is free to
//! reorder independent instructions to hide a high-latency op's latency, and it will slot `vc`'s
//! own pack instructions into the gap between the mul and the add (both otherwise adjacent, and
//! `vc` is not needed until the add) - which breaks `fusesIntoNextArith`'s immediately-preceding
//! requirement and silently falls back to the unfused vfmul+vfadd/vfsub path (a fine, safe
//! fallback in general, but exactly what this test must NOT hit to prove the fused path fires
//! and computes the right answer). Splitting `vc`'s construction into a second block reached by
//! a jump - avoiding the scheduler that way instead - would need the jump-edge block-argument
//! lowering (now cycle-correct for permuted edges, see parallel_move.zig), but is unrelated to what
//! this test proves. Compiling this single block directly, with no scheduler and no jump edge,
//! keeps the mul and add adjacent so the fusion this test targets actually fires.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const emit = @import("../emit.zig");
const ld = @import("../ld.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// The three fused shapes `fusesIntoNextArith` recognizes for a vector mul (see isel.zig's
/// `.arith` vector add/sub branch): `add`: a*b+c -> vfmacc. `sub`: a*b-c -> vfmsac. `csub`:
/// c-a*b -> vfnmsac. Unlike aarch64's NEON (which has no instruction for the `sub` shape), RVV's
/// OPFVV fused family covers all three.
const FmaShape = enum { add, sub, csub };

/// Build a <4 x f32> function computing `shape` on constant lanes `a`, `b`, `c` - the vector mul
/// immediately preceding its single-use consuming vector add/sub, exactly the shape
/// `fusesIntoNextArith` fuses into one vfmacc/vfmsac/vfnmsac - returning lane `ret_lane` as a
/// scalar f32. The lane values are baked in as `fconst`s rather than passed as block params (RVV
/// has no float-vector ABI to carry a whole vector in, and this backend has no >8-argument float
/// ABI either): nothing in this backend's lowering constant-folds float arithmetic (see this
/// file's top doc comment), so this still exercises the real vfmacc/vfmsac/vfnmsac hardware at
/// qemu-execution time, not a compile-time-folded shortcut.
fn buildVecFmaFunc(allocator: std.mem.Allocator, shape: FmaShape, a: [4]f32, b: [4]f32, c: [4]f32, ret_lane: u32) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ft = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = ft } });
    const blk = try func.appendBlock();
    var ap: [4]Value = undefined;
    var bp: [4]Value = undefined;
    var cp: [4]Value = undefined;
    for (0..4) |i| ap[i] = try func.appendInst(blk, ft, .{ .fconst = a[i] });
    for (0..4) |i| bp[i] = try func.appendInst(blk, ft, .{ .fconst = b[i] });
    for (0..4) |i| cp[i] = try func.appendInst(blk, ft, .{ .fconst = c[i] });
    const va = try func.appendInst(blk, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const vb = try func.appendInst(blk, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
    const vc = try func.appendInst(blk, v4, .{ .struct_new = .{ .fields = try func.internValueList(&cp) } });
    const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
    const r = switch (shape) {
        .add => try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = vc } }),
        .sub => try func.appendInst(blk, v4, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = vc } }),
        .csub => try func.appendInst(blk, v4, .{ .arith = .{ .op = .sub, .lhs = vc, .rhs = prod } }),
    };
    const lane = try func.appendInst(blk, ft, .{ .extract = .{ .aggregate = r, .index = ret_lane } });
    func.setTerminator(blk, .{ .ret = lane });
    return func;
}

/// Like `harness.runFuncFloat`, but selects `func` directly (no legalize, no scheduler) - see
/// this file's top doc comment for why the scheduler specifically must be bypassed here.
fn runSelectedFloat(io: std.Io, allocator: std.mem.Allocator, func: *Function) !u64 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);

    const stub = try harness.buildUserStubFloat(allocator, false, &.{});
    defer allocator.free(stub);
    const program = try allocator.alloc(u32, stub.len + code.len);
    defer allocator.free(program);
    @memcpy(program[0..stub.len], stub);
    @memcpy(program[stub.len..], code);

    const bytes = try emit.emitBytes(allocator, program);
    defer allocator.free(bytes);
    const user_base: u64 = 0x10000;
    const elf = try ld.writeElfExec(allocator, bytes, bytes.len, user_base, user_base);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "firmware.elf", .data = elf, .flags = .{ .permissions = .executable_file } });

    const argv = try harness.qemu_user.buildArgv(allocator, "firmware.elf");
    defer allocator.free(argv);
    const result = std.process.run(allocator, io, .{ .argv = argv, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.stdout.len < 8) {
        std.debug.print("qemu-riscv64: stdout too short ({d} bytes):\nstdout: {s}\nstderr: {s}\n", .{ result.stdout.len, result.stdout, result.stderr });
        return error.BackendFailed;
    }
    const tail = result.stdout[result.stdout.len - 8 ..];
    return std.mem.readInt(u64, tail[0..8], .little);
}

/// Build, compile, and check the vector FMA `shape` on the 4-lane operands `a`/`b`/`c`:
///   1. Structural (checked once, lane choice does not affect the emitted arithmetic): the
///      emitted code contains `mnemonic` and no separate vfmul/vfadd/vfsub - proves the fusion
///      actually fired for this shape.
///   2. For every lane, the qemu-executed result is bit-identical to the hardware FMA reference
///      `@mulAdd` (proves the fused instruction's semantics and single-rounding are correct, not
///      just that one lane happens to work).
///   3. For every lane, that result DIFFERS from the naive separately-rounded computation - the
///      operands are chosen so fused != separate, so a fusion that silently didn't fire (or fired
///      with the wrong instruction, landing back on the separate-rounding value by coincidence)
///      would be caught here.
fn checkVecFma(io: std.Io, shape: FmaShape, a: [4]f32, b: [4]f32, c: [4]f32, mnemonic: []const u8) !void {
    const allocator = std.testing.allocator;

    var struct_func = try buildVecFmaFunc(allocator, shape, a, b, c, 0);
    defer struct_func.deinit();
    const code = try isel.selectFunction(allocator, &struct_func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, mnemonic) != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "vfmul") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "vfadd") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "vfsub") == null);

    for (0..4) |i| {
        var run_func = try buildVecFmaFunc(allocator, shape, a, b, c, @intCast(i));
        defer run_func.deinit();
        const got_bits = try runSelectedFloat(io, allocator, &run_func);
        const got: f32 = @bitCast(@as(u32, @truncate(got_bits)));

        const want = switch (shape) {
            .add => @mulAdd(f32, a[i], b[i], c[i]),
            .sub => @mulAdd(f32, a[i], b[i], -c[i]),
            .csub => @mulAdd(f32, -a[i], b[i], c[i]),
        };
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));

        const naive = switch (shape) {
            .add => a[i] * b[i] + c[i],
            .sub => a[i] * b[i] - c[i],
            .csub => c[i] - a[i] * b[i],
        };
        try std.testing.expect(@as(u32, @bitCast(naive)) != @as(u32, @bitCast(got)));
    }
}

// Each lane's (a, b, c) triple below was found by random search specifically because the
// separately-rounded computation (vfmul then +/- vfadd/vfsub, two roundings) differs from the
// fused one (one rounding), so the test would fail if fusion did not fire, or fired with the
// wrong instruction. Lanes are distinct per test so a lane-mixing bug (e.g. a swapped extract
// index) would also show up as a mismatch, not pass by coincidence.

test "vfma: <4 x f32> a*b+c matches @mulAdd bit-exactly per lane and fuses to vfmacc (qemu-riscv64)" {
    const a = [4]f32{ @bitCast(@as(u32, 0xbf4ce3b4)), @bitCast(@as(u32, 0x3ef89cf3)), @bitCast(@as(u32, 0x40d81d0b)), @bitCast(@as(u32, 0xc231bbe7)) };
    const b = [4]f32{ @bitCast(@as(u32, 0x431bd56f)), @bitCast(@as(u32, 0xbaf6fc73)), @bitCast(@as(u32, 0xc10fee8c)), @bitCast(@as(u32, 0x43490765)) };
    const c = [4]f32{ @bitCast(@as(u32, 0x40a36717)), @bitCast(@as(u32, 0x3aaf8468)), @bitCast(@as(u32, 0xbd79ae05)), @bitCast(@as(u32, 0xc2c4dbca)) };
    try checkVecFma(std.testing.io, .add, a, b, c, "vfmacc");
}

test "vfma: <4 x f32> a*b-c matches @mulAdd bit-exactly per lane and fuses to vfmsac (qemu-riscv64)" {
    const a = [4]f32{ @bitCast(@as(u32, 0x3b9208a1)), @bitCast(@as(u32, 0xbfd8ce59)), @bitCast(@as(u32, 0xc479ca67)), @bitCast(@as(u32, 0xbfc5340e)) };
    const b = [4]f32{ @bitCast(@as(u32, 0x40ff1a17)), @bitCast(@as(u32, 0x42dcabd2)), @bitCast(@as(u32, 0x3bb44236)), @bitCast(@as(u32, 0xc15b3e30)) };
    const c = [4]f32{ @bitCast(@as(u32, 0xbb0357c8)), @bitCast(@as(u32, 0xbdf8f417)), @bitCast(@as(u32, 0x3f0b022a)), @bitCast(@as(u32, 0x3e252c8c)) };
    try checkVecFma(std.testing.io, .sub, a, b, c, "vfmsac");
}

test "vfma: <4 x f32> c-a*b matches @mulAdd bit-exactly per lane and fuses to vfnmsac (qemu-riscv64)" {
    const a = [4]f32{ @bitCast(@as(u32, 0x3d240cf6)), @bitCast(@as(u32, 0x418a1fe2)), @bitCast(@as(u32, 0x3dbf1dc2)), @bitCast(@as(u32, 0x3db0893f)) };
    const b = [4]f32{ @bitCast(@as(u32, 0x42d5e05a)), @bitCast(@as(u32, 0xbce38852)), @bitCast(@as(u32, 0x43c673eb)), @bitCast(@as(u32, 0x4118d7e0)) };
    const c = [4]f32{ @bitCast(@as(u32, 0xc14b1340)), @bitCast(@as(u32, 0xbe0d2ee5)), @bitCast(@as(u32, 0x3c0ae5e3)), @bitCast(@as(u32, 0xbe7db49b)) };
    try checkVecFma(std.testing.io, .csub, a, b, c, "vfnmsac");
}

test "vfma: a multi-use vector mul does NOT fuse (separate vfmul+vfadd, correct result, qemu-riscv64)" {
    // The vector product feeds the fusible add AND is read again afterward, so it is not
    // single-use: fusion must be declined, and the multiply stays materialized.
    const allocator = std.testing.allocator;
    const a = [4]f32{ 2.5, 1.25, -3.0, 0.5 };
    const b = [4]f32{ 3.25, 4.0, 2.0, -1.5 };
    const c = [4]f32{ 1.5, -0.75, 0.25, 2.0 };

    const build = struct {
        fn f(alloc: std.mem.Allocator, ret_lane: u32) !Function {
            var func = Function.init(alloc);
            errdefer func.deinit();
            const ft = try func.types.intern(.{ .float = .f32 });
            const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = ft } });
            const blk = try func.appendBlock();
            var ap: [4]Value = undefined;
            var bp: [4]Value = undefined;
            var cp: [4]Value = undefined;
            for (0..4) |i| ap[i] = try func.appendInst(blk, ft, .{ .fconst = a[i] });
            for (0..4) |i| bp[i] = try func.appendInst(blk, ft, .{ .fconst = b[i] });
            for (0..4) |i| cp[i] = try func.appendInst(blk, ft, .{ .fconst = c[i] });
            const va = try func.appendInst(blk, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
            const vb = try func.appendInst(blk, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
            const vc = try func.appendInst(blk, v4, .{ .struct_new = .{ .fields = try func.internValueList(&cp) } });
            const prod = try func.appendInst(blk, v4, .{ .arith = .{ .op = .mul, .lhs = va, .rhs = vb } });
            const s = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = vc } }); // a*b+c, fusible shape...
            const r = try func.appendInst(blk, v4, .{ .arith = .{ .op = .add, .lhs = s, .rhs = prod } }); // ...but prod is reused here
            const lane = try func.appendInst(blk, ft, .{ .extract = .{ .aggregate = r, .index = ret_lane } });
            func.setTerminator(blk, .{ .ret = lane });
            return func;
        }
    }.f;

    var struct_func = try build(allocator, 0);
    defer struct_func.deinit();
    const code = try isel.selectFunction(allocator, &struct_func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "vfmul") != null); // the multiply is still materialized
    try std.testing.expect(std.mem.indexOf(u8, text, "vfadd") != null); // and added separately, twice
    try std.testing.expect(std.mem.indexOf(u8, text, "vfmacc") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "vfmsac") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "vfnmsac") == null);

    for (0..4) |i| {
        var run_func = try build(allocator, @intCast(i));
        defer run_func.deinit();
        const got_bits = try runSelectedFloat(std.testing.io, allocator, &run_func);
        const got: f32 = @bitCast(@as(u32, @truncate(got_bits)));
        const prod_v = a[i] * b[i];
        const expected = (prod_v + c[i]) + prod_v;
        try std.testing.expectEqual(expected, got);
    }
}
