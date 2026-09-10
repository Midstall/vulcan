//! Frontend-to-GPU path: SPIR-V binary lowered to Vulcan IR, selected to a SASS
//! compute kernel. Validation is structural (the emitted instruction stream).
//! live GPU execution runs from prism's compute dispatch, not this repo.

const std = @import("std");
const spirv = @import("vulcan-spirv");
const opt = @import("vulcan-opt");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");

const op = spirv.opcodes;

/// Opcodes present in `kernel.code`, one per 4-dword instruction.
fn hasOpcode(code: []const u32, opcode: u32) bool {
    var i: usize = 0;
    while (i < code.len) : (i += 4) {
        if (code[i] & 0xfff == opcode) return true;
    }
    return false;
}

fn countOpcode(code: []const u32, opcode: u32) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < code.len) : (i += 4) {
        if (code[i] & 0xfff == opcode) n += 1;
    }
    return n;
}

test "SPIR-V compute function -> IR -> SASS kernel (x*y - x)" {
    const allocator = std.testing.allocator;

    // The function f(int x, int y) returning x*y - x exercises IMAD and ISUB.
    // ids: int=1, fnty=2, f=3, x=4, y=5, entry=6, prod=7, diff=8.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(allocator, op.ISub, &.{ 1, 8, 7, 4 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // SPIR-V -> IR -> SASS compute kernel.
    var kernel = try isel.compileKernel(allocator, &func, isel.nvidia_abi);
    defer kernel.deinit(allocator);

    // The kernel loads the output pointer + two inputs (4x LDC), multiplies,
    // subtracts, stores, and exits.
    try std.testing.expectEqual(@as(usize, 4), countOpcode(kernel.code, 0xb82)); // LDC x4
    try std.testing.expect(hasOpcode(kernel.code, 0x224)); // IMAD
    try std.testing.expect(hasOpcode(kernel.code, 0x210)); // IADD3 (the subtract)
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT

    // The subtract's srcB negate bit (bit 63 -> word offset +1, bit 31) is set on
    // the IADD3 produced from OpISub.
    var i: usize = 0;
    var saw_negated_iadd3 = false;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0x210 and (kernel.code[i + 1] >> 31) & 1 == 1) saw_negated_iadd3 = true;
    }
    try std.testing.expect(saw_negated_iadd3);
}

test "SPIR-V function composes with the optimizer before SASS codegen" {
    const allocator = std.testing.allocator;

    // The function f(int x) returning (x + 7) * x folds a constant into the stream.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 1, 3, 7 });
    try b.emit(allocator, op.Function, &.{ 1, 4, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IAdd, &.{ 1, 7, 5, 3 });
    try b.emit(allocator, op.IMul, &.{ 1, 8, 7, 5 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    _ = try opt.optimize(allocator, &func);

    var kernel = try isel.compileKernel(allocator, &func, isel.nvidia_abi);
    defer kernel.deinit(allocator);
    // Still a well-formed kernel: it loads the input, computes, stores, and exits.
    try std.testing.expect(hasOpcode(kernel.code, 0xb82)); // LDC
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT
}

test "SPIR-V conversions -> SASS I2F/F2I" {
    const allocator = std.testing.allocator;

    // The function f(int x) returning int(float(x) * float(x)) squares via the float path.
    // ids: int=1, float=2, fnty=3, f=4, x=5, entry=6, fx=7, sq=8, r=9.
    var b = try spirv.binary.Builder.init(allocator, 10);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeFunction, &.{ 3, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 4, 0, 3 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.ConvertSToF, &.{ 2, 7, 5 });
    try b.emit(allocator, op.FMul, &.{ 2, 8, 7, 7 });
    try b.emit(allocator, op.ConvertFToS, &.{ 1, 9, 8 });
    try b.emit(allocator, op.ReturnValue, &.{9});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    var kernel = try isel.compileKernel(allocator, &func, isel.nvidia_abi);
    defer kernel.deinit(allocator);

    try std.testing.expect(hasOpcode(kernel.code, 0x306)); // I2F (base 0x106 | reg form)
    try std.testing.expect(hasOpcode(kernel.code, 0x220)); // FMUL
    try std.testing.expect(hasOpcode(kernel.code, 0x305)); // F2I
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG
}

test "SPIR-V compute shader -> SASS kernel (buffer load/store + thread id)" {
    const allocator = std.testing.allocator;

    // The shader main() assigns data[gl_GlobalInvocationID.x] = data[gid] * 2.
    var b = try spirv.binary.Builder.init(allocator, 23);
    defer b.deinit(allocator);
    try b.emit(allocator, op.ExecutionMode, &.{ 16, op.ExecutionModeKind.local_size, 64, 1, 1 }); // local_size_x = 64
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.global_invocation_id });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 1 });
    try b.emit(allocator, op.TypeInt, &.{ 3, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, op.Constant, &.{ 3, 12, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 2 });
    try b.emit(allocator, op.Variable, &.{ 5, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 15, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, op.Label, &.{17});
    try b.emit(allocator, op.AccessChain, &.{ 6, 18, 14, 12 });
    try b.emit(allocator, op.Load, &.{ 3, 19, 18 });
    try b.emit(allocator, op.AccessChain, &.{ 10, 20, 15, 12, 19 });
    try b.emit(allocator, op.Load, &.{ 2, 21, 20 });
    try b.emit(allocator, op.IMul, &.{ 2, 22, 21, 13 });
    try b.emit(allocator, op.Store, &.{ 20, 22 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    var kernel = try isel.compileKernel(allocator, &func, isel.nvidia_abi);
    defer kernel.deinit(allocator);

    // The invocation id is blockIdx.x * local_size_x + threadIdx.x: two S2R reads
    // (tid + ctaid) and a MOV of the local size (64). The buffer base comes from the
    // constant bank (a 64-bit pair = two LDC), the element address from a 64-bit
    // IADD3 carry add, then LDG, the multiply, STG, and EXIT (no output pointer).
    try std.testing.expectEqual(@as(usize, 2), countOpcode(kernel.code, 0x919)); // S2R x2 (tid + ctaid)
    var saw_localsize_mov = false;
    var k: usize = 0;
    while (k < kernel.code.len) : (k += 4) {
        if (kernel.code[k] & 0xfff == 0x802 and kernel.code[k + 1] == 64) saw_localsize_mov = true; // MOV imm 64
    }
    try std.testing.expect(saw_localsize_mov); // local_size_x folded in
    try std.testing.expectEqual(@as(usize, 2), countOpcode(kernel.code, 0xb82)); // LDC x2 (buffer ptr)
    try std.testing.expect(hasOpcode(kernel.code, 0x981)); // LDG (load data[i])
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG (store data[i])
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT

    // The carry-add pair: an IADD3 writing a carry-out predicate (P6) at bits 81-83,
    // and an IADD3 reading a carry-in predicate at bits 87-89.
    var saw_cout = false;
    var saw_cin = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff != 0x210) continue;
        if ((kernel.code[i + 2] >> 17) & 0x7 == 6) saw_cout = true; // carry-out -> P6 at bit 81
        if ((kernel.code[i + 2] >> 23) & 0x7 == 6) saw_cin = true; // carry-in <- P6 at bit 87
    }
    try std.testing.expect(saw_cout);
    try std.testing.expect(saw_cin);

    // The scoreboard scheduler ran: the LDG (variable latency) carries a write
    // barrier (wr_bar at bits 110-112 is a real scoreboard, not 7 = none), and some
    // later instruction waits on a scoreboard (a non-zero wait mask at 116-121).
    var ldg_bar: u32 = 7;
    var saw_wait = false;
    i = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0x981) ldg_bar = (kernel.code[i + 3] >> 14) & 0x7; // LDG wr_bar
        if ((kernel.code[i + 3] >> 20) & 0x3f != 0) saw_wait = true; // some wait mask set
    }
    try std.testing.expect(ldg_bar < 6); // a real scoreboard was assigned to the load
    try std.testing.expect(saw_wait);
}

test "SASS: a lowered division compiles to a kernel (register reuse)" {
    const allocator = std.testing.allocator;
    const Function = ir.function.Function;

    // f(x, y) = x / y (unsigned). The GPU has no integer divide, so opt.lowerdiv
    // expands it to ~256 instructions with 32 short-lived compares. The linear-scan
    // allocator reuses registers so it fits the 6 predicates / 250 GPRs.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const q = try func.appendInst(b, t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(q) });

    try std.testing.expect(try opt.lowerdiv.run(allocator, &func));
    var kernel = try isel.compileKernel(allocator, &func, isel.nvidia_abi);
    defer kernel.deinit(allocator);

    // The expansion lowers to shifts, compares, and selects, ending in STG + EXIT.
    try std.testing.expect(hasOpcode(kernel.code, 0x219)); // SHF (shift)
    try std.testing.expect(hasOpcode(kernel.code, 0x20c)); // ISETP (compare)
    try std.testing.expect(hasOpcode(kernel.code, 0x207)); // SEL
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG (output)
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT
    // Reuse kept the register count modest despite ~256 instructions.
    try std.testing.expect(kernel.reg_count <= 32);
}

// ---------------------------------------------------------------------------
// Grid-builtin differential tests (M3).
//
// Each test below builds ONE kernel IR function and sends it down two paths.
//
//   1. `vulcan-gpu.lowerToLoopNest` rewrites it into a host loop nest, the native JIT
//      compiles that, and the host runs the whole grid. The buffer it writes is the
//      reference answer: it says what the builtin MEANS, one value per thread, in nest
//      order. That is the CPU offload oracle.
//   2. `isel.compileKernel` selects the same function to SASS. This repository has no GPU,
//      so the GPU side stays structural. The assertion is not "an S2R appears" but the exact
//      special-register index in the exact operand field of the exact instruction, plus the
//      register wiring around it.
//
// Together they close the loop: the oracle fixes the quantity, and the structural check
// fixes the register the hardware holds that quantity in. A swapped axis fails path 1, a
// wrong SR number fails path 2, and a lowering that reads the right register into the wrong
// place fails the wiring check.

const gpu = @import("vulcan-gpu");
const encode = @import("../encode.zig");
const native = @import("../../native.zig");
const host_builtin = @import("builtin");

/// SASS opcodes the decoders below match on, in the low 12 bits of word 0.
const s2r_opcode: u32 = 0x919;
const mov_imm_opcode: u32 = 0x802;
const imad_opcode: u32 = 0x224;
const ldc_opcode: u32 = 0xb82;

/// The workgroup size every probe kernel declares. The three axes differ, so a lowering that
/// reads the wrong axis writes different numbers instead of the same ones.
const probe_block = [3]u32{ 5, 3, 2 };

/// The grid the oracle runs. Again all three axes differ.
const probe_grid = [3]i32{ 2, 3, 4 };

/// The number of threads `probe_grid` x `probe_block` starts, which is the length of the
/// trace the oracle writes.
const probe_threads: usize =
    @as(usize, probe_block[0]) * probe_block[1] * probe_block[2] *
    @as(usize, @intCast(probe_grid[0] * probe_grid[1] * probe_grid[2]));

/// Whether the native JIT has a backend for the host architecture. The structural half of
/// each test runs everywhere; only the execution half needs this.
fn hasJit() bool {
    return switch (host_builtin.cpu.arch) {
        .aarch64, .x86_64, .x86, .riscv64 => true,
        else => false,
    };
}

/// The signature of the lowered probe kernel: the grid size in workgroups on all three axes,
/// then the kernel's real parameters. The builtin parameter is gone, because the nest
/// computes it.
const ProbeFn = *const fn (
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    out: [*]i32,
    counter: *i32,
) callconv(.c) void;

/// The kernel `out[counter[0]] = v; counter[0] += 1`, where `v` is the one builtin parameter
/// under test.
///
/// The counter makes the buffer hold the value of `v` for every thread of the launch, in the
/// order the nest visits them, rather than one value per grid point. So the same kernel reads
/// back a full trace whichever builtin it is tagged with, including the uniform ones. The
/// nest runs one thread at a time, so the counter needs no atomic.
///
/// `v` is the FIRST entry parameter, so the prologue emits its hardware read before anything
/// else and the structural checks can index from instruction 0.
fn probeKernel(
    allocator: std.mem.Allocator,
    tag: gpu.Builtin,
    block: [3]u32,
) !ir.function.Function {
    var func = ir.function.Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const v = try func.appendBlockParam(entry, i32_t);
    const out = try func.appendBlockParam(entry, ptr_t);
    const counter = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, v, tag);
    try gpu.attrs.setLocalSize(&func, block);

    const seq = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = counter } });
    const off = try func.appendArithImm(entry, i32_t, .mul, seq, 4);
    const slot = try func.appendInst(entry, ptr_t, .{
        .arith = .{ .op = .add, .lhs = out, .rhs = off },
    });
    try func.appendStore(entry, v, slot);
    const next = try func.appendArithImm(entry, i32_t, .add, seq, 1);
    try func.appendStore(entry, next, counter);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// Run the probe kernel for `tag` through the CPU offload oracle and fill `out` with the
/// value every thread of the launch read, in nest order.
fn oracleTrace(allocator: std.mem.Allocator, tag: gpu.Builtin, out: []i32) !void {
    var kernel = try probeKernel(allocator, tag, probe_block);
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try native.jitFunction(allocator, &lowered);
    defer code.deinit();

    @memset(out, -1);
    var counter: i32 = 0;
    code.entry(ProbeFn, 0)(probe_grid[0], probe_grid[1], probe_grid[2], out.ptr, &counter);
    // Every thread wrote exactly once, so the trace has no hole and no overrun.
    try std.testing.expectEqual(@as(i32, @intCast(out.len)), counter);
}

/// Select the probe kernel for `tag` to SASS. The caller owns the result.
fn probeSass(allocator: std.mem.Allocator, tag: gpu.Builtin) !isel.Kernel {
    var kernel = try probeKernel(allocator, tag, probe_block);
    defer kernel.deinit();
    return isel.compileKernel(allocator, &kernel, isel.nvidia_abi);
}

/// One decoded `S2R`: the GPR it writes (bits 16..23, word 0) and the special-register index
/// it reads (bits 72..79, which is word 2 bits 8..15).
const S2RRead = struct { dst: u8, sysval: u8 };

fn decodeS2R(w: []const u32) S2RRead {
    return .{ .dst = @truncate(w[0] >> 16), .sysval = @truncate(w[2] >> 8) };
}

/// One decoded `MOV dst, imm32`: the GPR it writes and the 32-bit immediate (word 1).
const MovImm = struct { dst: u8, imm: u32 };

fn decodeMovImm(w: []const u32) MovImm {
    return .{ .dst = @truncate(w[0] >> 16), .imm = w[1] };
}

/// One decoded `LDC dst, c[0][offset]`: the GPR it writes and the static byte offset it reads
/// from, which lives in bits 38..53 (word 1 bits 6..21).
const LdcRead = struct { dst: u8, offset: u32 };

fn decodeLdc(w: []const u32) LdcRead {
    return .{ .dst = @truncate(w[0] >> 16), .offset = @as(u16, @truncate(w[1] >> 6)) & 0xffff };
}

/// One decoded ALU `IMAD dst, a, b, c`: `dst = a * b + c`. The operand fields are the shared
/// ALU ones: dst at 16..23, srcA at 24..31, srcB at 32..39 (word 1 bits 0..7), and srcC at
/// 64..71 (word 2 bits 0..7).
const Imad = struct { dst: u8, a: u8, b: u8, c: u8 };

fn decodeImad(w: []const u32) Imad {
    return .{
        .dst = @truncate(w[0] >> 16),
        .a = @truncate(w[0] >> 24),
        .b = @truncate(w[1]),
        .c = @truncate(w[2]),
    };
}

test "grid builtins: thread_id on each axis reads its own SR_TID and traces that axis" {
    const allocator = std.testing.allocator;

    const cases = [3]struct { tag: gpu.Builtin, axis: u2, sysval: u8 }{
        .{ .tag = .thread_id_x, .axis = 0, .sysval = 0x21 },
        .{ .tag = .thread_id_y, .axis = 1, .sysval = 0x22 },
        .{ .tag = .thread_id_z, .axis = 2, .sysval = 0x23 },
    };

    const trace = try allocator.alloc(i32, probe_threads);
    defer allocator.free(trace);

    for (cases) |case| {
        // The SASS path. The builtin is the first entry parameter, so its hardware read is
        // instruction 0, and one S2R is the whole lowering.
        var kernel = try probeSass(allocator, case.tag);
        defer kernel.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 1), countOpcode(kernel.code, s2r_opcode));
        try std.testing.expectEqual(s2r_opcode, kernel.code[0] & 0xfff);
        const read = decodeS2R(kernel.code[0..4]);
        try std.testing.expectEqual(case.sysval, read.sysval);
        try std.testing.expectEqual(encode.sr_tid[case.axis], read.sysval);

        // The oracle. The thread index counts 0..block[axis]-1 inside every workgroup, so the
        // trace is that axis's induction variable and nothing else.
        if (!hasJit()) continue;
        try oracleTrace(allocator, case.tag, trace);

        var want = try allocator.alloc(i32, probe_threads);
        defer allocator.free(want);
        var n: usize = 0;
        for (0..@intCast(probe_grid[2] * probe_grid[1] * probe_grid[0])) |_| {
            for (0..probe_block[2]) |tz| {
                for (0..probe_block[1]) |ty| {
                    for (0..probe_block[0]) |tx| {
                        const t = [3]usize{ tx, ty, tz };
                        want[n] = @intCast(t[case.axis]);
                        n += 1;
                    }
                }
            }
        }
        try std.testing.expectEqualSlices(i32, want, trace);
    }
}

test "grid builtins: block_id on each axis reads its own SR_CTAID and traces that axis" {
    const allocator = std.testing.allocator;

    const cases = [3]struct { tag: gpu.Builtin, axis: u2, sysval: u8 }{
        .{ .tag = .block_id_x, .axis = 0, .sysval = 0x25 },
        .{ .tag = .block_id_y, .axis = 1, .sysval = 0x26 },
        .{ .tag = .block_id_z, .axis = 2, .sysval = 0x27 },
    };

    const trace = try allocator.alloc(i32, probe_threads);
    defer allocator.free(trace);

    for (cases) |case| {
        var kernel = try probeSass(allocator, case.tag);
        defer kernel.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 1), countOpcode(kernel.code, s2r_opcode));
        try std.testing.expectEqual(s2r_opcode, kernel.code[0] & 0xfff);
        const read = decodeS2R(kernel.code[0..4]);
        try std.testing.expectEqual(case.sysval, read.sysval);
        try std.testing.expectEqual(encode.sr_ctaid[case.axis], read.sysval);

        // The oracle. The workgroup index holds still for a whole workgroup and then steps,
        // so the trace repeats each value block[0]*block[1]*block[2] times.
        if (!hasJit()) continue;
        try oracleTrace(allocator, case.tag, trace);

        var want = try allocator.alloc(i32, probe_threads);
        defer allocator.free(want);
        var n: usize = 0;
        var bz: i32 = 0;
        while (bz < probe_grid[2]) : (bz += 1) {
            var by: i32 = 0;
            while (by < probe_grid[1]) : (by += 1) {
                var bx: i32 = 0;
                while (bx < probe_grid[0]) : (bx += 1) {
                    const b = [3]i32{ bx, by, bz };
                    for (0..probe_block[2] * probe_block[1] * probe_block[0]) |_| {
                        want[n] = b[case.axis];
                        n += 1;
                    }
                }
            }
        }
        try std.testing.expectEqualSlices(i32, want, trace);
    }
}

test "grid builtins: global_id on each axis fuses ctaid * ntid + tid on that same axis" {
    const allocator = std.testing.allocator;

    const cases = [3]struct { tag: gpu.Builtin, axis: u2, tid: u8, ctaid: u8 }{
        .{ .tag = .global_id_x, .axis = 0, .tid = 0x21, .ctaid = 0x25 },
        .{ .tag = .global_id_y, .axis = 1, .tid = 0x22, .ctaid = 0x26 },
        .{ .tag = .global_id_z, .axis = 2, .tid = 0x23, .ctaid = 0x27 },
    };

    const trace = try allocator.alloc(i32, probe_threads);
    defer allocator.free(trace);

    for (cases) |case| {
        // The SASS path. The lowering is four instructions: the declared workgroup size as an
        // immediate, the two hardware reads, and the multiply-add that fuses them. Both
        // special-register indices and the whole register wiring are checked, so a read of
        // the right register into the wrong operand fails here.
        var kernel = try probeSass(allocator, case.tag);
        defer kernel.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), countOpcode(kernel.code, s2r_opcode));
        try std.testing.expectEqual(mov_imm_opcode, kernel.code[0] & 0xfff);
        try std.testing.expectEqual(s2r_opcode, kernel.code[4] & 0xfff);
        try std.testing.expectEqual(s2r_opcode, kernel.code[8] & 0xfff);
        try std.testing.expectEqual(imad_opcode, kernel.code[12] & 0xfff);

        const size = decodeMovImm(kernel.code[0..4]);
        const tid = decodeS2R(kernel.code[4..8]);
        const ctaid = decodeS2R(kernel.code[8..12]);
        const fuse = decodeImad(kernel.code[12..16]);

        try std.testing.expectEqual(case.tid, tid.sysval);
        try std.testing.expectEqual(encode.sr_tid[case.axis], tid.sysval);
        try std.testing.expectEqual(case.ctaid, ctaid.sysval);
        try std.testing.expectEqual(encode.sr_ctaid[case.axis], ctaid.sysval);
        try std.testing.expectEqual(probe_block[case.axis], size.imm);
        // dst = ctaid * ntid + tid, with each operand coming from the instruction that
        // produced it.
        try std.testing.expectEqual(ctaid.dst, fuse.a);
        try std.testing.expectEqual(size.dst, fuse.b);
        try std.testing.expectEqual(tid.dst, fuse.c);
        try std.testing.expectEqual(size.dst, fuse.dst);
        // The two hardware reads land in different registers, or the multiply-add would fold
        // one axis onto the other.
        try std.testing.expect(tid.dst != ctaid.dst);

        // The oracle. Every thread reads block_id * block_dim + thread_id on its own axis.
        if (!hasJit()) continue;
        try oracleTrace(allocator, case.tag, trace);

        var want = try allocator.alloc(i32, probe_threads);
        defer allocator.free(want);
        var n: usize = 0;
        var bz: i32 = 0;
        while (bz < probe_grid[2]) : (bz += 1) {
            var by: i32 = 0;
            while (by < probe_grid[1]) : (by += 1) {
                var bx: i32 = 0;
                while (bx < probe_grid[0]) : (bx += 1) {
                    for (0..probe_block[2]) |tz| {
                        for (0..probe_block[1]) |ty| {
                            for (0..probe_block[0]) |tx| {
                                const b = [3]i32{ bx, by, bz };
                                const t = [3]usize{ tx, ty, tz };
                                const size_a: i32 = @intCast(probe_block[case.axis]);
                                want[n] = b[case.axis] * size_a + @as(i32, @intCast(t[case.axis]));
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
        try std.testing.expectEqualSlices(i32, want, trace);
    }
}

test "grid builtins: block_dim on each axis is the declared size, with no hardware read" {
    const allocator = std.testing.allocator;

    const cases = [3]struct { tag: gpu.Builtin, axis: u2 }{
        .{ .tag = .block_dim_x, .axis = 0 },
        .{ .tag = .block_dim_y, .axis = 1 },
        .{ .tag = .block_dim_z, .axis = 2 },
    };

    const trace = try allocator.alloc(i32, probe_threads);
    defer allocator.free(trace);

    for (cases) |case| {
        // The SASS path. The workgroup size is not a special register on this hardware. It is
        // the size the kernel declares, so the lowering is one immediate and the kernel reads
        // no special register at all.
        var kernel = try probeSass(allocator, case.tag);
        defer kernel.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 0), countOpcode(kernel.code, s2r_opcode));
        try std.testing.expectEqual(mov_imm_opcode, kernel.code[0] & 0xfff);
        try std.testing.expectEqual(probe_block[case.axis], decodeMovImm(kernel.code[0..4]).imm);
        // The launch descriptor carries the same number, so the runtime cannot launch a
        // workgroup shape the folded immediate disagrees with.
        try std.testing.expectEqual(probe_block[case.axis], kernel.launch.block[case.axis]);

        // The oracle. A size is uniform over the launch, so every thread reads the same value.
        if (!hasJit()) continue;
        try oracleTrace(allocator, case.tag, trace);
        for (trace) |got| try std.testing.expectEqual(@as(i32, @intCast(probe_block[case.axis])), got);
    }
}

test "grid builtins: grid_dim on each axis loads its own launch-shape slot and traces it" {
    const allocator = std.testing.allocator;

    const cases = [3]struct { tag: gpu.Builtin, axis: u2 }{
        .{ .tag = .grid_dim_x, .axis = 0 },
        .{ .tag = .grid_dim_y, .axis = 1 },
        .{ .tag = .grid_dim_z, .axis = 2 },
    };

    const trace = try allocator.alloc(i32, probe_threads);
    defer allocator.free(trace);

    for (cases) |case| {
        // The SASS path. The grid size has no special register, so the lowering is one LDC
        // from the launch-shape region and no hardware read at all. The offset comes back out
        // of the instruction and has to be the one `LaunchInfo` tells the runtime to write.
        var kernel = try probeSass(allocator, case.tag);
        defer kernel.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 0), countOpcode(kernel.code, s2r_opcode));
        try std.testing.expectEqual(ldc_opcode, kernel.code[0] & 0xfff);

        const shape = kernel.launch.launch_shape.?;
        const want = isel.nvidia_abi.param_base + shape.axisOffset(case.axis);
        try std.testing.expectEqual(want, decodeLdc(kernel.code[0..4]).offset);
        // The region sits after both explicit pointers, which keep the offsets they would
        // have had with no grid builtin at all.
        try std.testing.expectEqual(@as(usize, 2), kernel.launch.params.len);
        try std.testing.expectEqual(@as(u32, 0), kernel.launch.params[0].offset);
        try std.testing.expectEqual(@as(u32, 8), kernel.launch.params[1].offset);
        try std.testing.expectEqual(@as(u32, 16), shape.offset);
        try std.testing.expectEqual(@as(u32, 28), kernel.launch.param_bytes);

        // The oracle. The grid size is uniform over the launch, so every thread reads the
        // same value, and that value is the grid the host nest ran.
        if (!hasJit()) continue;
        try oracleTrace(allocator, case.tag, trace);
        for (trace) |got| try std.testing.expectEqual(probe_grid[case.axis], got);
    }
}

test "grid builtins: the subgroup builtins still refuse to lower" {
    // These have no correct lowering on this backend yet: the subgroup builtins have no
    // oracle to check them against, because the host nest runs one thread at a time and has
    // no subgroup. A refusal is the honest answer: a kernel that read the wrong register
    // would look right and compute garbage.
    const allocator = std.testing.allocator;

    for ([2]gpu.Builtin{ .warp_id, .subgroup_size }) |tag| {
        var kernel = try probeKernel(allocator, tag, probe_block);
        defer kernel.deinit();
        try std.testing.expectError(
            error.Unsupported,
            isel.compileKernel(allocator, &kernel, isel.nvidia_abi),
        );
    }
}

test "grid builtins: the oracle's nest order is the contract's linearization" {
    // The cross-target check. `gpu.kernel.axisIndex` is the rule a LINEAR-ID backend follows
    // to split one identifier into three axes, and the CPU oracle is the reference every
    // target is diffed against. If the two disagreed, the ET-SoC prologue would put a hart in
    // a slot the oracle never visits, and each side would still look self-consistent.
    //
    // The oracle writes one value per thread in nest order, so the nest position IS the
    // linear index: the thread part counts inside a workgroup and the workgroup part counts
    // outside it.
    const allocator = std.testing.allocator;
    if (!hasJit()) return error.SkipZigTest;

    const threads_per_block: usize = @as(usize, probe_block[0]) * probe_block[1] * probe_block[2];
    const grid: [3]u32 = .{
        @intCast(probe_grid[0]),
        @intCast(probe_grid[1]),
        @intCast(probe_grid[2]),
    };

    const trace = try allocator.alloc(i32, probe_threads);
    defer allocator.free(trace);

    const thread_tags = [3]gpu.Builtin{ .thread_id_x, .thread_id_y, .thread_id_z };
    for (thread_tags, 0..) |tag, axis| {
        try oracleTrace(allocator, tag, trace);
        for (trace, 0..) |got, i| {
            const local: u32 = @intCast(i % threads_per_block);
            const want = gpu.kernel.axisIndex(local, probe_block)[axis];
            try std.testing.expectEqual(@as(i32, @intCast(want)), got);
        }
    }

    const block_tags = [3]gpu.Builtin{ .block_id_x, .block_id_y, .block_id_z };
    for (block_tags, 0..) |tag, axis| {
        try oracleTrace(allocator, tag, trace);
        for (trace, 0..) |got, i| {
            const workgroup: u32 = @intCast(i / threads_per_block);
            const want = gpu.kernel.axisIndex(workgroup, grid)[axis];
            try std.testing.expectEqual(@as(i32, @intCast(want)), got);
        }
    }
}
