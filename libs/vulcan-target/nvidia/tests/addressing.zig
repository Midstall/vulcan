//! NVIDIA address-forming and immediate-operand execution tests.
//!
//! The instruction selector puts a constant in the instruction that reads it: an ALU
//! operand goes in the 32-bit immediate field, and a constant byte offset goes in the
//! 24-bit address displacement of LDG, STG, LDS and STS. Both replace a MOV plus a
//! register, and the address fold also replaces a whole IADD3 carry chain.
//!
//! EVERY TEST HERE RUNS ON THE GPU AND CHECKS THE NUMBERS. A shorter instruction
//! sequence that computes the wrong answer is worth nothing, and an addressing change
//! is exactly the kind that reads or writes the wrong place while every structural
//! test stays green.
//!
//! THE BUFFERS SIT ABOVE 4 GiB on purpose. A 64-bit pointer plus a constant offset used
//! to write only the LOW half of the address pair and leave the high half unwritten, and
//! a buffer in the low 4 GiB hides that: the high half is zero either way.
//!
//! `compute.Runner.init` gives `error.SkipZigTest` when no GPU answers, so a machine
//! with no NVIDIA hardware skips every test in this file.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu_abi = @import("vulcan-gpu");
const target = @import("vulcan-target");
const nvidia = @import("nvidia");

const isel = target.nvidia.isel;
const compute = nvidia.compute;
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const testing = std.testing;

/// The parameter ABI of the nvidia.zig dispatch: the parameter buffer is the base of
/// constant bank 0, so the first parameter sits at offset 0. See `tests/execute.zig`.
const runner_abi: gpu_abi.Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = false,
};

/// The first GPU virtual address this file maps a buffer at. The driver keeps
/// 0x0000_0000_FFE0_0000 up to 0x0000_0002_0000_0000 for itself, and a map inside that
/// window silently swallows every store. See `tests/carry.zig`.
const high_va: u64 = 0x0000_0002_0000_0000;

/// Whether `err` means "this machine has no usable NVIDIA GPU". A DISPATCH failure such
/// as `error.GridTimeout` is deliberately absent: a kernel that hangs must fail.
fn noGpu(err: anyerror) bool {
    return switch (err) {
        error.SkipZigTest,
        error.FileNotFound,
        error.AccessDenied,
        error.PermissionDenied,
        error.DeviceBusy,
        error.NoDevice,
        => true,
        else => false,
    };
}

const Harness = struct {
    runner: compute.Runner,
    params: compute.Buffer,

    fn open() !Harness {
        var runner = compute.Runner.init() catch |err| {
            if (noGpu(err)) return error.SkipZigTest;
            return err;
        };
        errdefer runner.deinit();
        const params = try runner.alloc(.system, 0x1000);
        runner.next_va = high_va;
        return .{ .runner = runner, .params = params };
    }

    fn deinit(self: *Harness) void {
        self.runner.deinit();
    }

    /// Allocate a zeroed buffer ABOVE 4 GiB, so the high half of its address is nonzero.
    fn alloc(self: *Harness, size: u64) !compute.Buffer {
        const buf = try self.runner.alloc(.system, size);
        std.debug.assert(buf.va >= high_va);
        return buf;
    }

    fn compile(self: *Harness, func: *Function) !Launch {
        const kernel = try isel.compileKernel(testing.allocator, func, runner_abi);
        return .{ .harness = self, .kernel = kernel };
    }
};

const Launch = struct {
    harness: *Harness,
    kernel: isel.Kernel,

    fn deinit(self: *Launch) void {
        self.kernel.deinit(testing.allocator);
    }

    fn setPtr(self: *Launch, off: u32, va: u64) void {
        std.mem.writeInt(u64, self.harness.params.bytes[off..][0..8], va, .little);
    }

    fn setU32(self: *Launch, off: u32, v: u32) void {
        std.mem.writeInt(u32, self.harness.params.bytes[off..][0..4], v, .little);
    }

    fn setF32(self: *Launch, off: u32, v: f32) void {
        self.setU32(off, @bitCast(v));
    }

    fn run(self: *Launch, grid: [3]u32) !void {
        try self.harness.runner.run(self.kernel.code, .{
            .grid = grid,
            .block = self.kernel.launch.block,
            .register_count = self.kernel.launch.reg_count,
            .cbuf0_va = self.harness.params.va,
            .cbuf0_size = @max(self.kernel.launch.param_bytes, 16),
            .shared_mem_bytes = self.kernel.launch.shared_bytes,
        });
    }

    fn paramOffset(self: *const Launch, i: usize) u32 {
        return self.kernel.launch.params[i].offset;
    }
};

/// How many instructions the kernel holds, not counting the trailing NOP padding the
/// emitter adds. A test that names a count states what the change bought.
fn instructionCount(kernel: *const isel.Kernel) usize {
    return kernel.code.len / 4;
}

fn ptrAddImm(func: *Function, b: Block, ptr_t: ir.types.Type, base: Value, imm: i64) !Value {
    return func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = imm } });
}

fn ptrAddVal(func: *Function, b: Block, ptr_t: ir.types.Type, base: Value, off: Value) !Value {
    return func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
}

fn binImm(func: *Function, b: Block, ty: ir.types.Type, op: ir.function.BinOp, v: Value, imm: i64) !Value {
    return func.appendInst(b, ty, .{ .arith_imm = .{ .op = op, .lhs = v, .imm = imm } });
}

test "live: a constant array index folds into the load and the store, and reads the right slots" {
    // `out[4] = in[2] + 7`, with both constant offsets in the access's own displacement
    // field. Nothing computes an address at all, so a fold that picked the wrong base or
    // the wrong sign reads or writes a neighbouring slot and this test says so.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const src = try ptrAddImm(&func, b, ptr_t, in, 2 * 4);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = src } });
    const r = try binImm(&func, b, i32_t, .add, v, 7);
    const dst = try ptrAddImm(&func, b, ptr_t, out, 4 * 4);
    try func.appendStore(b, r, dst);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();

    // Two LDC.64s (one per pointer), the load, the add, the store, EXIT. No address
    // arithmetic at all.
    try testing.expectEqual(@as(usize, 6), instructionCount(&launch.kernel));

    const dstbuf = try h.alloc(0x1000);
    const inbuf = try h.alloc(0x1000);
    for (0..8) |i| inbuf.slice(i32)[i] = @intCast(100 + i);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setPtr(launch.paramOffset(1), inbuf.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 102 + 7), dstbuf.read(i32, 4));
    // The neighbours stay zero, so a displacement off by one slot fails here.
    try testing.expectEqual(@as(i32, 0), dstbuf.read(i32, 3));
    try testing.expectEqual(@as(i32, 0), dstbuf.read(i32, 5));
}

test "live: a NEGATIVE constant index folds, and reads the slot BEFORE the base" {
    // The displacement field is SIGNED. A fold that wrote the value unsigned addresses
    // 16 MiB past the base instead of 8 bytes before it, which faults or reads garbage.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const src = try ptrAddImm(&func, b, ptr_t, in, -8);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = src } });
    try func.appendStore(b, v, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();
    const dstbuf = try h.alloc(0x1000);
    const inbuf = try h.alloc(0x1000);
    for (0..8) |i| inbuf.slice(i32)[i] = @intCast(500 + i);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    // The pointer parameter names slot 4, so -8 bytes is slot 2.
    launch.setPtr(launch.paramOffset(1), inbuf.va + 4 * 4);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 502), dstbuf.read(i32, 0));
}

test "live: a chain of constant offsets folds into ONE displacement" {
    // `((in + 4) + 8)` is slot 3. A walk that stopped at the first step would read slot
    // 1, and one that added the steps wrongly would read anything else.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const step1 = try ptrAddImm(&func, b, ptr_t, in, 4);
    const step2 = try ptrAddImm(&func, b, ptr_t, step1, 8);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = step2 } });
    try func.appendStore(b, v, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();

    // Two LDC.64s, the load, the store, EXIT. Both address adds disappear, and the inner
    // one only because the dead scan repeats.
    try testing.expectEqual(@as(usize, 5), instructionCount(&launch.kernel));

    const dstbuf = try h.alloc(0x1000);
    const inbuf = try h.alloc(0x1000);
    for (0..8) |i| inbuf.slice(i32)[i] = @intCast(900 + i);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setPtr(launch.paramOffset(1), inbuf.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 903), dstbuf.read(i32, 0));
}

test "live: an offset TOO WIDE for the displacement field falls back to a 64-bit add" {
    // 1 << 23 is one past the signed 24-bit range, so `fitsAddrOffset` refuses it and the
    // address becomes a real IADD3 plus IADD3.X pair. THE HIGH HALF MUST BE WRITTEN: the
    // constant-offset path used to write the low half alone and leave the high half
    // holding whatever the register allocator last put there, which with a buffer above
    // 4 GiB addresses another place entirely.
    const allocator = testing.allocator;
    const wide: i64 = 1 << 23;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const src = try ptrAddImm(&func, b, ptr_t, in, wide);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = src } });
    const dst = try ptrAddImm(&func, b, ptr_t, out, wide);
    try func.appendStore(b, v, dst);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();

    const size = @as(u64, wide) + 0x1000;
    const dstbuf = try h.alloc(size);
    const inbuf = try h.alloc(size);
    const slot: usize = @intCast(@divExact(wide, 4));
    inbuf.slice(i32)[slot] = 0x5a5a_1234;
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setPtr(launch.paramOffset(1), inbuf.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 0x5a5a_1234), dstbuf.read(i32, slot));
}

test "live: a shared-tile slot reached by a constant displacement carries the right value" {
    // The LDS and STS displacement field, and the shared address space, which is a
    // 32-bit window offset in ONE register and not a pointer pair.
    const allocator = testing.allocator;
    const threads = 32;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const tile = try func.appendBlockParam(b, shared_t);
    const tid = try func.appendBlockParam(b, i32_t);
    try gpu_abi.attrs.setBuiltin(&func, tid, .thread_id_x);
    try gpu_abi.attrs.setLocalSize(&func, .{ threads, 1, 1 });
    try gpu_abi.attrs.setSharedBytes(&func, threads * 4);

    const byte = try binImm(&func, b, i32_t, .shl, tid, 2);
    // Every thread writes its own slot: tile[tid] = tid * 10.
    const mine = try binImm(&func, b, i32_t, .mul, tid, 10);
    try func.appendStore(b, mine, try ptrAddVal(&func, b, shared_t, tile, byte));
    try func.appendBarrier(b, .workgroup);
    // Then reads the slot 3 places up, through the displacement field alone:
    // tile[(tid & 15) + 3]. The mask keeps the read inside the tile.
    const masked = try binImm(&func, b, i32_t, .bit_and, tid, 15);
    const mbyte = try binImm(&func, b, i32_t, .shl, masked, 2);
    const at = try ptrAddVal(&func, b, shared_t, tile, mbyte);
    const at3 = try ptrAddImm(&func, b, shared_t, at, 3 * 4);
    const got = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = at3 } });
    try func.appendStore(b, got, try ptrAddVal(&func, b, ptr_t, out, byte));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();
    const dstbuf = try h.alloc(0x1000);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setU32(launch.paramOffset(1), 0); // the tile starts at window offset 0
    try launch.run(.{ 1, 1, 1 });

    for (0..threads) |i| {
        const expected: i32 = @intCast(((i & 15) + 3) * 10);
        try testing.expectEqual(expected, dstbuf.read(i32, i));
    }
}

test "live: every integer ALU op computes the right answer with a CONSTANT operand" {
    // One kernel per operand form would hide an op whose immediate lands in the wrong
    // field, so this runs them all against one input and checks each result. `sub` and
    // `shr` are the ones to watch: subtraction has no negate bit in the immediate form,
    // so the VALUE is negated, and a right shift keeps its value in the third operand
    // slot while the count goes in the immediate.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);

    const ops = [_]struct { op: ir.function.BinOp, imm: i64 }{
        .{ .op = .add, .imm = 7 },
        .{ .op = .sub, .imm = 9 },
        .{ .op = .mul, .imm = 3 },
        .{ .op = .bit_and, .imm = 0xf0 },
        .{ .op = .bit_or, .imm = 0x101 },
        .{ .op = .bit_xor, .imm = 0x0ff },
        .{ .op = .shl, .imm = 5 },
        .{ .op = .shr, .imm = 3 },
        .{ .op = .add, .imm = -11 },
        .{ .op = .mul, .imm = -2 },
    };
    inline for (ops, 0..) |o, i| {
        const r = try binImm(&func, b, i32_t, o.op, x, o.imm);
        try func.appendStore(b, r, try ptrAddImm(&func, b, ptr_t, out, i * 4));
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();
    const dstbuf = try h.alloc(0x1000);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    const x_in: i32 = 0x1234;
    launch.setU32(launch.paramOffset(1), @bitCast(x_in));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(x_in + 7, dstbuf.read(i32, 0));
    try testing.expectEqual(x_in - 9, dstbuf.read(i32, 1));
    try testing.expectEqual(x_in * 3, dstbuf.read(i32, 2));
    try testing.expectEqual(x_in & 0xf0, dstbuf.read(i32, 3));
    try testing.expectEqual(x_in | 0x101, dstbuf.read(i32, 4));
    try testing.expectEqual(x_in ^ 0x0ff, dstbuf.read(i32, 5));
    try testing.expectEqual(x_in << 5, dstbuf.read(i32, 6));
    try testing.expectEqual(x_in >> 3, dstbuf.read(i32, 7));
    try testing.expectEqual(x_in - 11, dstbuf.read(i32, 8));
    try testing.expectEqual(x_in * -2, dstbuf.read(i32, 9));
}

test "live: every float ALU op computes the right answer with a CONSTANT operand" {
    // FADD, FMUL and the subtract that flips the immediate's sign bit. FMUL is the one
    // with a field the register form also needs (PDIV at 84..86): dropping it makes the
    // product saturate instead of scaling.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, f32_t);

    const add = try func.appendInst(b, f32_t, .{ .arith_imm = .{
        .op = .add,
        .lhs = x,
        .imm = @as(i64, @as(u32, @bitCast(@as(f32, 0.25)))),
    } });
    const sub = try func.appendInst(b, f32_t, .{ .arith_imm = .{
        .op = .sub,
        .lhs = x,
        .imm = @as(i64, @as(u32, @bitCast(@as(f32, 1.5)))),
    } });
    const mul = try func.appendInst(b, f32_t, .{ .arith_imm = .{
        .op = .mul,
        .lhs = x,
        .imm = @as(i64, @as(u32, @bitCast(@as(f32, 0.5)))),
    } });
    try func.appendStore(b, add, try ptrAddImm(&func, b, ptr_t, out, 0));
    try func.appendStore(b, sub, try ptrAddImm(&func, b, ptr_t, out, 4));
    try func.appendStore(b, mul, try ptrAddImm(&func, b, ptr_t, out, 8));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();
    const dstbuf = try h.alloc(0x1000);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setF32(launch.paramOffset(1), 2.0);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(f32, 2.25), dstbuf.read(f32, 0));
    try testing.expectEqual(@as(f32, 0.5), dstbuf.read(f32, 1));
    try testing.expectEqual(@as(f32, 1.0), dstbuf.read(f32, 2));
}

test "live: the fused global index reads the real hardware with the workgroup size folded" {
    // `gid = ctaid * ntid + tid` is now ONE IMAD with the workgroup size in its immediate
    // field. A wrong field would give every thread the same index, or scale by zero, and
    // this writes one slot per thread so either shows up as a hole in the output.
    const allocator = testing.allocator;
    const threads = 8;
    const groups = 4;
    const total = threads * groups;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const gid = try func.appendBlockParam(b, i32_t);
    try gpu_abi.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu_abi.attrs.setLocalSize(&func, .{ threads, 1, 1 });

    const byte = try binImm(&func, b, i32_t, .shl, gid, 2);
    const val = try binImm(&func, b, i32_t, .add, gid, 1000);
    try func.appendStore(b, val, try ptrAddVal(&func, b, ptr_t, out, byte));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();
    const dstbuf = try h.alloc(0x1000);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    try launch.run(.{ groups, 1, 1 });

    for (0..total) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1000)), dstbuf.read(i32, i));
    }
}

test "live: a computed array index reaches the right slot through IMAD.WIDE" {
    // `out[gid] = in[gid] * 7`, where each address is ONE IMAD.WIDE.U32 that scales,
    // widens and adds the 64-bit base together. It replaced an IADD3 with a carry-out and
    // an IADD3.X that read it, so a wrong operand slot or a lost carry lands the access on
    // another slot, or on another buffer. The reversal in the input makes a thread that
    // reads its own slot instead of the indexed one give the wrong number.
    const allocator = testing.allocator;
    const threads = 16;
    const groups = 4;
    const total = threads * groups;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const gid = try func.appendBlockParam(b, i32_t);
    try gpu_abi.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu_abi.attrs.setLocalSize(&func, .{ threads, 1, 1 });

    const byte = try binImm(&func, b, i32_t, .shl, gid, 2);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, b, ptr_t, in, byte) } });
    const r = try binImm(&func, b, i32_t, .mul, v, 7);
    try func.appendStore(b, r, try ptrAddVal(&func, b, ptr_t, out, byte));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();

    // Two LDC.64s, two S2Rs, the fused index, the byte shift, two IMAD.WIDEs, the load,
    // the multiply, the store, EXIT.
    try testing.expectEqual(@as(usize, 12), instructionCount(&launch.kernel));

    const dstbuf = try h.alloc(0x1000);
    const inbuf = try h.alloc(0x1000);
    for (0..total) |i| inbuf.slice(i32)[i] = @intCast(total - 1 - i);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setPtr(launch.paramOffset(1), inbuf.va);
    try launch.run(.{ groups, 1, 1 });

    for (0..total) |i| {
        try testing.expectEqual(@as(i32, @intCast((total - 1 - i) * 7)), dstbuf.read(i32, i));
    }
}

test "live: an LDC.64 loads BOTH halves of a pointer parameter" {
    // The prologue reads a 64-bit pointer with one LDC.64 into a register pair. A load
    // that filled only the low half would address the low 4 GiB, and every buffer here
    // sits above it, so the store would fault or land nowhere. Three pointers make sure
    // the widened load reads the right slot of the parameter block for each of them.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const a_ptr = try func.appendBlockParam(b, ptr_t);
    const b_ptr = try func.appendBlockParam(b, ptr_t);
    const c_ptr = try func.appendBlockParam(b, ptr_t);
    const va = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = a_ptr } });
    const vb = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = b_ptr } });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    try func.appendStore(b, sum, c_ptr);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();

    // Three LDC.64s, two loads, the add, the store, EXIT. Six LDCs before.
    try testing.expectEqual(@as(usize, 8), instructionCount(&launch.kernel));

    const abuf = try h.alloc(0x1000);
    const bbuf = try h.alloc(0x1000);
    const cbuf = try h.alloc(0x1000);
    abuf.slice(i32)[0] = 111;
    bbuf.slice(i32)[0] = 222;
    launch.setPtr(launch.paramOffset(0), abuf.va);
    launch.setPtr(launch.paramOffset(1), bbuf.va);
    launch.setPtr(launch.paramOffset(2), cbuf.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 333), cbuf.read(i32, 0));
}

test "live: a NEGATIVE offset too wide to fold sign-extends into the high address word" {
    // The fallback path for a constant the 24-bit displacement field cannot hold. The
    // constant is added as a 64-bit value, so its high word is 0xFFFFFFFF when it is
    // negative. Added as zero instead, the address moves 4 GiB UP rather than 8 MiB down,
    // which reads unmapped memory or another buffer.
    const allocator = testing.allocator;
    const back: i64 = -((1 << 23) + 4); // one word past the field's negative range
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const src = try ptrAddImm(&func, b, ptr_t, in, back);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = src } });
    try func.appendStore(b, v, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();

    const at: i64 = 1 << 24; // the pointer parameter names a word 16 MiB into the buffer
    const dstbuf = try h.alloc(0x1000);
    const inbuf = try h.alloc(@as(u64, at) + 0x1000);
    // `back` therefore lands a little under 8 MiB in, well inside the buffer.
    const slot: usize = @intCast(@divExact(at + back, 4));
    inbuf.slice(i32)[slot] = 0x1234_5678;
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setPtr(launch.paramOffset(1), inbuf.va + @as(u64, at));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 0x1234_5678), dstbuf.read(i32, 0));
}

test "live: a register-register float add is EXACT, not a bfloat16 add" {
    // `encode.fadd` used to write PT into bits 81, 84 and 87, which turned the instruction
    // into `FHADD.BF16 Rd, Ra.H1, Rb`: a half-precision add that truncates source A to
    // bfloat16. `fsub` is built on `fadd`, so subtraction carried the same defect, and a
    // naive matmul returned 4486 where 12276 was right.
    //
    // THE OPERANDS MATTER. Every earlier float test used values whose low 16 mantissa bits
    // are zero (0.5, 1.0, 2.0), and truncating those to bfloat16 changes nothing, so they
    // all passed with the bug present. These have every low bit set.
    const allocator = testing.allocator;
    const a_bits: u32 = 0x3eff_fffc;
    const b_bits: u32 = 0x3e7f_fff9;
    const a: f32 = @bitCast(a_bits);
    const b: f32 = @bitCast(b_bits);

    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const bl = try func.appendBlock();
    const out = try func.appendBlockParam(bl, ptr_t);
    const x = try func.appendBlockParam(bl, f32_t);
    const y = try func.appendBlockParam(bl, f32_t);
    const sum = try func.appendInst(bl, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const diff = try func.appendInst(bl, f32_t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = y } });
    const prod = try func.appendInst(bl, f32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    // `x + x` reads the SAME register in both operand slots, which is the shape the matmul
    // accumulator has and the one that made the truncation of source A visible.
    const dbl = try func.appendInst(bl, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    try func.appendStore(bl, sum, out);
    try func.appendStore(bl, diff, try ptrAddImm(&func, bl, ptr_t, out, 4));
    try func.appendStore(bl, prod, try ptrAddImm(&func, bl, ptr_t, out, 8));
    try func.appendStore(bl, dbl, try ptrAddImm(&func, bl, ptr_t, out, 12));
    func.setTerminator(bl, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func);
    defer launch.deinit();
    const dstbuf = try h.alloc(0x1000);
    launch.setPtr(launch.paramOffset(0), dstbuf.va);
    launch.setU32(launch.paramOffset(1), a_bits);
    launch.setU32(launch.paramOffset(2), b_bits);
    try launch.run(.{ 1, 1, 1 });

    // The oracle is the host's own f32 arithmetic, compared as BIT PATTERNS so a result
    // that is merely close still fails.
    try testing.expectEqual(@as(u32, @bitCast(a + b)), dstbuf.read(u32, 0));
    try testing.expectEqual(@as(u32, @bitCast(a - b)), dstbuf.read(u32, 1));
    try testing.expectEqual(@as(u32, @bitCast(a * b)), dstbuf.read(u32, 2));
    try testing.expectEqual(@as(u32, @bitCast(a + a)), dstbuf.read(u32, 3));
}
