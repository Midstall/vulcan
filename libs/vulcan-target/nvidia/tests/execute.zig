//! NVIDIA hardware execution. Every other NVIDIA test in this repository reads the
//! STRUCTURE of the emitted instruction stream: opcode counts and decoded bit fields.
//! This one compiles a kernel with `isel.compileKernel`, uploads the SASS to a real
//! GPU through the nvidia.zig compute dispatch, runs it, reads the buffers back, and
//! compares the ACTUAL NUMBERS. It is the only test that proves the silicon agrees.
//!
//! ## The parameter base
//!
//! nvidia.zig binds the caller's parameter buffer as the BASE of constant bank 0, so
//! its kernels read parameters at offset 0. The CUDA driver instead puts a block of
//! its own in front, which is why `isel.nvidia_abi` uses 0x160. The two conventions
//! are both correct, and `Abi.param_base` is data for exactly this reason. Every
//! kernel here compiles under `runner_abi`, whose `param_base` is 0. A kernel built
//! with 0x160 and dispatched by this runner would read 352 bytes past its parameters,
//! and "the same kernel under both bases" below proves the base is honoured as data.
//!
//! ## Skipping
//!
//! `compute.Runner.init` gives `error.SkipZigTest` when no GPU answers, and `gpu()`
//! below turns a permission error or a missing driver node into the same result. A
//! machine with no NVIDIA hardware, such as CI, skips every test in this file.

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

/// The parameter ABI of the nvidia.zig dispatch. It differs from `isel.nvidia_abi` in
/// `param_base` alone: this runner binds the parameter buffer as the base of constant
/// bank 0, so the first parameter sits at offset 0 of the bank. See the module comment.
const runner_abi: gpu_abi.Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = false,
};

/// Whether `err` means "this machine has no usable NVIDIA GPU". `Runner.init` already
/// answers `error.SkipZigTest` when `/dev/nvidiactl` is missing or the RM refuses the
/// device, and these cover the rest: no driver node, no permission on it, or the device
/// held by something else. A DISPATCH failure such as `error.GridTimeout` is not in this
/// list, because a kernel that hangs on hardware must fail the test and not skip it.
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

/// Open the GPU, or skip the test when there is none. See `noGpu`.
fn gpu() !compute.Runner {
    return compute.Runner.init() catch |err| {
        if (noGpu(err)) return error.SkipZigTest;
        return err;
    };
}

/// An open GPU context: the compute channel and the one parameter buffer every kernel
/// of a test binds as constant bank 0.
///
/// A `Runner` owns its own GPU address space, so a buffer allocated from one runner has
/// no address in another. A test that runs two kernels therefore compiles both against
/// ONE harness. That is what "the same kernel under both parameter bases" needs.
const Harness = struct {
    runner: compute.Runner,
    params: compute.Buffer,

    /// Open GPU 0, or skip the test when there is none. See `noGpu`.
    fn open() !Harness {
        var runner = try gpu();
        errdefer runner.deinit();
        const params = try runner.alloc(.system, 0x1000);
        return .{ .runner = runner, .params = params };
    }

    fn deinit(self: *Harness) void {
        self.runner.deinit();
    }

    /// Allocate a zeroed buffer visible to both the CPU and the GPU.
    fn alloc(self: *Harness, size: u64) !compute.Buffer {
        return self.runner.alloc(.system, size);
    }

    /// Compile `func` under `abi` and bind it to this context.
    fn compile(self: *Harness, func: *Function, abi: gpu_abi.Abi) !Launch {
        const kernel = try isel.compileKernel(testing.allocator, func, abi);
        return .{ .harness = self, .kernel = kernel, .base = abi.param_base };
    }
};

/// One compiled kernel bound to a `Harness`.
const Launch = struct {
    harness: *Harness,
    kernel: isel.Kernel,
    /// The byte offset inside the parameter buffer where the block starts. It is the
    /// ABI's `param_base`, because the runner binds that buffer at the base of the bank.
    base: u32,

    fn deinit(self: *Launch) void {
        self.kernel.deinit(testing.allocator);
    }

    /// Write a 64-bit global address at block offset `off`.
    fn setPtr(self: *Launch, off: u32, va: u64) void {
        std.mem.writeInt(u64, self.harness.params.bytes[self.base + off ..][0..8], va, .little);
    }

    /// Write a 32-bit scalar at block offset `off`. A shared pointer parameter is one of
    /// these: it carries a byte offset into the workgroup's shared window.
    fn setU32(self: *Launch, off: u32, v: u32) void {
        std.mem.writeInt(u32, self.harness.params.bytes[self.base + off ..][0..4], v, .little);
    }

    /// Write the grid extents into the launch-shape region. The kernel must read a grid
    /// builtin, or `layoutParams` reserves no region and this is a programming error.
    fn setGridShape(self: *Launch, grid: [3]u32) void {
        const region = self.kernel.launch.launch_shape.?;
        for (0..3) |axis| self.setU32(region.axisOffset(@intCast(axis)), grid[axis]);
    }

    /// Dispatch the kernel over `grid` workgroups and wait for it.
    fn run(self: *Launch, grid: [3]u32) !void {
        try self.harness.runner.run(self.kernel.code, .{
            .grid = grid,
            .block = self.kernel.launch.block,
            .register_count = self.kernel.launch.reg_count,
            .cbuf0_va = self.harness.params.va,
            // The bank must cover the parameter block wherever the ABI put it. A bank
            // smaller than the highest offset the kernel reads gives that read zero.
            .cbuf0_size = @max(self.base + self.kernel.launch.param_bytes, 16),
            .shared_mem_bytes = self.kernel.launch.shared_bytes,
        });
    }
};

/// The highest GPR the instruction stream names, read back through the disassembler so
/// an immediate operand is never mistaken for a register number. The register-count test
/// needs this to prove it landed in the window it aims at. RZ is a fixed zero, not an
/// allocation, so it does not count.
fn maxRegisterUsed(code: []const u32) !u8 {
    const decoded = try target.nvidia.disasm.decode(testing.allocator, code);
    defer testing.allocator.free(decoded);
    var top: u8 = 0;
    for (decoded) |d| {
        if (d.dst != 255 and d.dst > top) top = d.dst;
        for (d.srcs) |s| {
            if (s != 255 and s > top) top = s;
        }
    }
    return top;
}

/// `base + imm` as a fresh pointer value. Pointer arithmetic in the IR counts BYTES.
fn ptrAdd(func: *Function, b: Block, ptr_t: ir.types.Type, base: Value, imm: i64) !Value {
    return func.appendInst(b, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = imm } });
}

/// `base + off` as a fresh pointer value, where `off` is a byte count in a register.
fn ptrAddVal(func: *Function, b: Block, ptr_t: ir.types.Type, base: Value, off: Value) !Value {
    return func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
}

/// `v <op> imm` as a fresh value of `v`'s own type.
fn binImm(func: *Function, b: Block, ty: ir.types.Type, op: ir.function.BinOp, v: Value, imm: i64) !Value {
    return func.appendInst(b, ty, .{ .arith_imm = .{ .op = op, .lhs = v, .imm = imm } });
}

/// `lhs <op> rhs` as a fresh value of type `ty`.
fn bin(func: *Function, b: Block, ty: ir.types.Type, op: ir.function.BinOp, lhs: Value, rhs: Value) !Value {
    return func.appendInst(b, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
}

/// `(idx.z * extent.y + idx.y) * extent.x + idx.x` as IR, which is the launch contract's
/// own linearization: x fastest and z slowest. See `gpu.kernel.linearIndex`.
fn linear3(func: *Function, b: Block, ty: ir.types.Type, idx: [3]Value, extent: [3]u32) !Value {
    const zy = try bin(func, b, ty, .add, try binImm(func, b, ty, .mul, idx[2], @intCast(extent[1])), idx[1]);
    return bin(func, b, ty, .add, try binImm(func, b, ty, .mul, zy, @intCast(extent[0])), idx[0]);
}

test "live: a void kernel stores a computed value through a global pointer" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);
    const y = try func.appendBlockParam(b, i32_t);
    const prod = try bin(&func, b, i32_t, .mul, x, y);
    const sum = try bin(&func, b, i32_t, .add, prod, x);
    try func.appendStore(b, sum, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const buf = try h.alloc(0x1000);

    launch.setPtr(launch.kernel.launch.params[0].offset, buf.va);
    launch.setU32(launch.kernel.launch.params[1].offset, 7);
    launch.setU32(launch.kernel.launch.params[2].offset, 5);
    try launch.run(.{ 1, 1, 1 });

    // 7 * 5 + 7.
    try testing.expectEqual(@as(i32, 42), buf.read(i32, 0));
}

test "live: a value-returning kernel writes through the implicit output pointer" {
    // The out-pointer is placed FIRST in the block, before every explicit parameter,
    // and the kernel's `ret` stores through it. Nothing but hardware proves that the
    // emitted STG really targets the address the runtime wrote into slot zero.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const y = try func.appendBlockParam(b, i32_t);
    const diff = try bin(&func, b, i32_t, .sub, x, y);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(diff) });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const buf = try h.alloc(0x1000);

    // The layout is reproduced from LaunchInfo: pointer first at 0, then the scalars.
    launch.setPtr(0, buf.va);
    launch.setU32(launch.kernel.launch.params[0].offset, 900);
    launch.setU32(launch.kernel.launch.params[1].offset, 258);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 642), buf.read(i32, 0));
}

test "live: the same kernel runs under BOTH parameter bases, from the ABI alone" {
    // The reason `param_base` is data and not a constant. The CUDA driver keeps a block
    // of its own at the front of constant bank 0 and the kernel parameters follow it at
    // 0x160. A runtime that binds its own buffer as the bank, as this one does, has no
    // such block and starts at 0. Both kernels are compiled from the same IR, differing
    // only in the ABI, and the parameter block is written twice into one buffer, once at
    // each base. Both must give the same answer, and the emitted LDC offsets must differ
    // by exactly 0x160, or the base is being ignored somewhere.
    const allocator = testing.allocator;

    const build = struct {
        fn f(alloc: std.mem.Allocator) !Function {
            var func = Function.init(alloc);
            errdefer func.deinit();
            const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
            const ptr_t = try func.types.ptrGlobal();
            const b = try func.appendBlock();
            const out = try func.appendBlockParam(b, ptr_t);
            const x = try func.appendBlockParam(b, i32_t);
            const y = try func.appendBlockParam(b, i32_t);
            const prod = try bin(&func, b, i32_t, .mul, x, y);
            const sum = try bin(&func, b, i32_t, .add, prod, x);
            try func.appendStore(b, sum, out);
            func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
            return func;
        }
    }.f;

    var func_zero = try build(allocator);
    defer func_zero.deinit();
    var func_cuda = try build(allocator);
    defer func_cuda.deinit();

    // Both kernels share ONE harness, so both read the same bank and reach the same
    // buffers. `Launch.base` shifts every parameter write to its own base.
    var h = try Harness.open();
    defer h.deinit();
    var zero = try h.compile(&func_zero, runner_abi);
    defer zero.deinit();
    var cuda = try h.compile(&func_cuda, isel.nvidia_abi);
    defer cuda.deinit();

    // The two streams must read the same parameters 0x160 bytes apart.
    try testing.expectEqual(@as(u32, 0), firstLdcOffset(zero.kernel.code).?);
    try testing.expectEqual(@as(u32, 0x160), firstLdcOffset(cuda.kernel.code).?);

    const buf = try h.alloc(0x1000);
    for ([2]*Launch{ &zero, &cuda }) |launch| {
        buf.slice(u32)[0] = 0;
        launch.setPtr(launch.kernel.launch.params[0].offset, buf.va);
        launch.setU32(launch.kernel.launch.params[1].offset, 6);
        launch.setU32(launch.kernel.launch.params[2].offset, 9);
        try launch.run(.{ 1, 1, 1 });
        // 6 * 9 + 6.
        try testing.expectEqual(@as(i32, 60), buf.read(i32, 0));
    }
}

/// The constant-bank byte offset the first LDC in `code` reads, or null when there is
/// none. LDC is opcode 0xb82 and the offset field starts at bit 38 (word 1, bit 6).
fn firstLdcOffset(code: []const u32) ?u32 {
    var i: usize = 0;
    while (i < code.len) : (i += 4) {
        if (code[i] & 0xfff == 0xb82) return @as(u16, @truncate(code[i + 1] >> 6)) & 0xffff;
    }
    return null;
}

test "live: a kernel whose top register the hardware would otherwise keep computes correctly" {
    // The hardware RESERVES the top two GPRs of each thread's allocation: a write to one
    // is dropped and a read gives zero, with no fault. `regCount` covers them. Before
    // that fix a kernel whose highest register sat in the top two of its own rounded
    // allocation lost those registers SILENTLY.
    //
    // Eight live parameters put the accumulator at R14. Rounding `14 + 1` to the granule
    // alone gives 16, which makes R14 and R15 hardware property, so the accumulator would
    // read back as zero and the kernel would store 0 instead of the sum. The right count
    // is one granule higher.
    const allocator = testing.allocator;
    const count = 8;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    var vals: [count]Value = undefined;
    for (&vals) |*v| v.* = try func.appendBlockParam(b, i32_t);
    // Sum in a tree, so no parameter dies before the last one is read.
    var acc = vals[0];
    for (vals[1..]) |v| acc = try bin(&func, b, i32_t, .add, acc, v);
    try func.appendStore(b, acc, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();

    // The window this test exists for. `naive` is what rounding the register the kernel
    // uses to the granule gives, with no room for the two the hardware keeps. The kernel
    // must reach into the top two of that count, or the test proves nothing, and the
    // declared count must then be one granule higher.
    const max_reg = try maxRegisterUsed(launch.kernel.code);
    const naive = (@as(u32, max_reg) + 1 + 7) & ~@as(u32, 7);
    try testing.expect(max_reg + 2 >= naive);
    try testing.expectEqual(naive + 8, launch.kernel.launch.reg_count);

    const buf = try h.alloc(0x1000);
    launch.setPtr(launch.kernel.launch.params[0].offset, buf.va);
    var expected: i32 = 0;
    for (0..count) |i| {
        const v: i32 = @intCast((i + 1) * 100 + i);
        launch.setU32(launch.kernel.launch.params[i + 1].offset, @bitCast(v));
        expected += v;
    }
    try launch.run(.{ 1, 1, 1 });
    try testing.expectEqual(expected, buf.read(i32, 0));

    // The negative control, and the reason the fix exists. The SAME instruction stream
    // dispatched with the naive register count gives the hardware the top two registers
    // the kernel is using, so the answer changes. This runs the mutation on the silicon
    // instead of in the compiler, so no source change can make it pass by accident.
    buf.slice(i32)[0] = 0;
    try h.runner.run(launch.kernel.code, .{
        .grid = .{ 1, 1, 1 },
        .block = launch.kernel.launch.block,
        .register_count = naive,
        .cbuf0_va = h.params.va,
        .cbuf0_size = @max(launch.kernel.launch.param_bytes, 16),
    });
    try testing.expect(buf.read(i32, 0) != expected);
}

test "live: a byte store writes ONE byte and leaves its neighbours intact" {
    // Before the access width came from the IR value type, a byte store emitted the B32
    // encoder and destroyed the three bytes beside its own. Nothing in the instruction
    // stream says which bytes memory actually kept, so only a real store proves it.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendInst(b, i8_t, .{ .load = .{ .ptr = in } });
    const at = try ptrAdd(&func, b, ptr_t, out, 1);
    try func.appendStore(b, v, at);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const dst = try h.alloc(0x1000);
    const src = try h.alloc(0x1000);
    @memset(dst.bytes[0..8], 0xaa);
    src.bytes[0] = 0x5a;

    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, src.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(u8, 0xaa), dst.read(u8, 0));
    try testing.expectEqual(@as(u8, 0x5a), dst.read(u8, 1));
    try testing.expectEqual(@as(u8, 0xaa), dst.read(u8, 2));
    try testing.expectEqual(@as(u8, 0xaa), dst.read(u8, 3));
}

test "live: a signed byte load sign-extends, where a 32-bit load would read a positive number" {
    // The load half of the same width fix. The source byte is 0xff with three zero bytes
    // after it, so a B32 load reads 255 and an I8 load reads -1. A compare against zero
    // tells the two apart, and the answer travels back as a 32-bit word.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendInst(b, i8_t, .{ .load = .{ .ptr = in } });
    const zero8 = try func.appendInst(b, i8_t, .{ .iconst = 0 });
    const neg = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = v, .rhs = zero8 } });
    const yes = try func.appendInst(b, i32_t, .{ .iconst = 111 });
    const no = try func.appendInst(b, i32_t, .{ .iconst = 222 });
    const pick = try func.appendInst(b, i32_t, .{ .select = .{ .cond = neg, .then = yes, .@"else" = no } });
    try func.appendStore(b, pick, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const dst = try h.alloc(0x1000);
    const src = try h.alloc(0x1000);
    src.bytes[0] = 0xff;
    src.bytes[1] = 0;
    src.bytes[2] = 0;
    src.bytes[3] = 0;

    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, src.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 111), dst.read(i32, 0));
}

test "live: a pointer load and store move BOTH halves of the address pair" {
    // A pointer is 64 bits in an aligned register pair. A B32 load would keep the low
    // dword and leave the high one holding whatever the register had, which turns the
    // next access through that pointer into a wild address. Loading a pointer-shaped
    // word with a nonzero HIGH half and storing it back proves both halves travelled.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendInst(b, ptr_t, .{ .load = .{ .ptr = in } });
    try func.appendStore(b, p, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const dst = try h.alloc(0x1000);
    const src = try h.alloc(0x1000);
    const pattern: u64 = 0xdead_beef_cafe_f00d;
    std.mem.writeInt(u64, src.bytes[0..8], pattern, .little);

    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, src.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(pattern, dst.read(u64, 0));
}

test "live: a dependent LDG-to-LDG-to-LDC chain gives the right answer in every thread" {
    // What a stolen scoreboard breaks. The address of the second load comes out of the
    // first load, the workgroup index comes from S2R, and the grid extent comes from an
    // LDC, so every variable-latency class feeds the chain. A consumer that issues before
    // its producer lands reads a stale register, which shows up as a wrong number here
    // and as nothing at all in a structural test.
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
    const index = try func.appendBlockParam(b, ptr_t);
    const table = try func.appendBlockParam(b, ptr_t);
    const gid = try func.appendBlockParam(b, i32_t);
    const gdim = try func.appendBlockParam(b, i32_t);
    try gpu_abi.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu_abi.attrs.setBuiltin(&func, gdim, .grid_dim_x);
    try gpu_abi.attrs.setLocalSize(&func, .{ threads, 1, 1 });

    const byte = try binImm(&func, b, i32_t, .shl, gid, 2);
    // j = index[gid], then v = table[j]: the second address depends on the first load.
    const j = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, b, ptr_t, index, byte) } });
    const jbyte = try binImm(&func, b, i32_t, .shl, j, 2);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, b, ptr_t, table, jbyte) } });
    // Fold in the LDC-sourced grid extent and the S2R-sourced index.
    const scaled = try binImm(&func, b, i32_t, .mul, gdim, 1000);
    const acc = try bin(&func, b, i32_t, .add, v, scaled);
    const result = try bin(&func, b, i32_t, .add, acc, gid);
    try func.appendStore(b, result, try ptrAddVal(&func, b, ptr_t, out, byte));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const dst = try h.alloc(0x1000);
    const idx = try h.alloc(0x1000);
    const tab = try h.alloc(0x1000);
    // A reversal, so a thread that reads its own slot instead of the indexed one is wrong.
    for (0..total) |i| idx.slice(i32)[i] = @intCast(total - 1 - i);
    for (0..total) |i| tab.slice(i32)[i] = @intCast(i * 7 + 3);

    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, idx.va);
    launch.setPtr(launch.kernel.launch.params[2].offset, tab.va);
    launch.setGridShape(.{ groups, 1, 1 });
    try launch.run(.{ groups, 1, 1 });

    for (0..total) |i| {
        const expected: i32 = @intCast((total - 1 - i) * 7 + 3 + groups * 1000 + i);
        try testing.expectEqual(expected, dst.read(i32, i));
    }
}

test "live: shared memory carries a value between threads across a workgroup barrier" {
    // STS, BAR.SYNC and LDS together. Each thread writes its own slot, the barrier makes
    // every write visible, and each thread then reads the slot of the thread at the other
    // end of the workgroup. A missing barrier, a lost shared window, or an LDS that
    // reached global memory all give the wrong number, and none of them is visible in the
    // opcode counts. The kernel is straight-line, so no divergent branch crosses the
    // barrier: on this hardware a branch around BAR.SYNC corrupts a staged tile.
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
    const mine = try binImm(&func, b, i32_t, .add, try binImm(&func, b, i32_t, .mul, tid, 3), 1);
    try func.appendStore(b, mine, try ptrAddVal(&func, b, shared_t, tile, byte));
    try func.appendBarrier(b, .workgroup);
    // The mirror slot: threads - 1 - tid, as (-tid) + (threads - 1).
    const mirror = try binImm(&func, b, i32_t, .add, try binImm(&func, b, i32_t, .mul, tid, -1), threads - 1);
    const mbyte = try binImm(&func, b, i32_t, .shl, mirror, 2);
    const got = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, b, shared_t, tile, mbyte) } });
    try func.appendStore(b, got, try ptrAddVal(&func, b, ptr_t, out, byte));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    try testing.expectEqual(@as(u32, 1), launch.kernel.launch.barrier_count);
    try testing.expectEqual(@as(u32, threads * 4), launch.kernel.launch.shared_bytes);

    const dst = try h.alloc(0x1000);
    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    // A shared pointer is a byte offset into the workgroup's own window, not an address.
    launch.setU32(launch.kernel.launch.params[1].offset, 0);
    try launch.run(.{ 1, 1, 1 });

    for (0..threads) |i| {
        const expected: i32 = @intCast((threads - 1 - i) * 3 + 1);
        try testing.expectEqual(expected, dst.read(i32, i));
    }
}

test "live: the three-dimensional thread and workgroup builtins read the real hardware" {
    // Six special registers, three launch-shape extents and three declared workgroup
    // extents, over a grid that is not square on any axis, so a swapped axis cannot pass.
    // Each thread writes its own ten answers into its own row.
    const allocator = testing.allocator;
    const block = [3]u32{ 4, 3, 2 };
    const grid = [3]u32{ 3, 2, 4 };
    const per_thread = 10;
    const threads_per_group = block[0] * block[1] * block[2];
    const groups = grid[0] * grid[1] * grid[2];

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);

    const order = [per_thread]gpu_abi.Builtin{
        .thread_id_x, .thread_id_y, .thread_id_z,
        .block_id_x,  .block_id_y,  .block_id_z,
        .grid_dim_x,  .grid_dim_y,  .grid_dim_z,
        .block_dim_y,
    };
    var read: [per_thread]Value = undefined;
    for (order, 0..) |bi, i| {
        read[i] = try func.appendBlockParam(b, i32_t);
        try gpu_abi.attrs.setBuiltin(&func, read[i], bi);
    }
    try gpu_abi.attrs.setLocalSize(&func, block);

    // row = linearGroup * threads_per_group + linearThread, x fastest and z slowest.
    const group_lin = try linear3(&func, b, i32_t, .{ read[3], read[4], read[5] }, grid);
    const thread_lin = try linear3(&func, b, i32_t, .{ read[0], read[1], read[2] }, block);
    const rows_before = try binImm(&func, b, i32_t, .mul, group_lin, threads_per_group);
    const row = try bin(&func, b, i32_t, .add, rows_before, thread_lin);
    const row_byte = try binImm(&func, b, i32_t, .mul, row, per_thread * 4);
    const base = try ptrAddVal(&func, b, ptr_t, out, row_byte);
    for (read, 0..) |v, i| {
        try func.appendStore(b, v, try ptrAdd(&func, b, ptr_t, base, @intCast(i * 4)));
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const rows = groups * threads_per_group;
    const dst = try h.alloc(rows * per_thread * 4 + 0x1000);
    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setGridShape(grid);
    try launch.run(grid);

    const got = dst.slice(i32);
    for (0..grid[2]) |bz| {
        for (0..grid[1]) |by| {
            for (0..grid[0]) |bx| {
                const g = (bz * grid[1] + by) * grid[0] + bx;
                for (0..block[2]) |tz| {
                    for (0..block[1]) |ty| {
                        for (0..block[0]) |tx| {
                            const t = (tz * block[1] + ty) * block[0] + tx;
                            const row_at = (g * threads_per_group + t) * per_thread;
                            const want = [per_thread]i32{
                                @intCast(tx),       @intCast(ty),      @intCast(tz),
                                @intCast(bx),       @intCast(by),      @intCast(bz),
                                @intCast(grid[0]),  @intCast(grid[1]), @intCast(grid[2]),
                                @intCast(block[1]),
                            };
                            for (want, 0..) |w, k| {
                                try testing.expectEqual(w, got[row_at + k]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "live: a kernel's OWN shared tile carries a value between threads across a barrier" {
    // The shared ALLOCA, on silicon. The kernel declares its own tile instead of taking a
    // `ptr(shared)` parameter, so the address of the tile is a frame offset the isel assigned
    // and the host writes nothing for it. Each thread stages its own value, the barrier makes
    // every write visible, and each thread then reads the slot of the thread at the other end
    // of the workgroup. The answer is only right if the staging really crossed threads.
    const allocator = testing.allocator;
    const threads = 32;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const tile_t = try func.types.intern(.{ .array = .{ .len = threads, .elem = i32_t } });
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const tid = try func.appendBlockParam(b, i32_t);
    try gpu_abi.attrs.setBuiltin(&func, tid, .thread_id_x);
    try gpu_abi.attrs.setLocalSize(&func, .{ threads, 1, 1 });

    const tile = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = tile_t } });
    const byte = try binImm(&func, b, i32_t, .shl, tid, 2);
    const mine = try binImm(&func, b, i32_t, .add, try binImm(&func, b, i32_t, .mul, tid, 3), 1);
    try func.appendStore(b, mine, try ptrAddVal(&func, b, shared_t, tile, byte));
    try func.appendBarrier(b, .workgroup);
    // The mirror slot: threads - 1 - tid, as (-tid) + (threads - 1).
    const mirror = try binImm(&func, b, i32_t, .add, try binImm(&func, b, i32_t, .mul, tid, -1), threads - 1);
    const mbyte = try binImm(&func, b, i32_t, .shl, mirror, 2);
    const got = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, b, shared_t, tile, mbyte) } });
    try func.appendStore(b, got, try ptrAddVal(&func, b, ptr_t, out, byte));
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    try testing.expectEqual(@as(u32, 1), launch.kernel.launch.barrier_count);
    // The frontend declared no total, so the frame the isel placed is what the runtime gets.
    try testing.expectEqual(@as(u32, threads * 4), launch.kernel.launch.shared_bytes);
    // The tile takes no room in the parameter block: `out` is the only parameter.
    try testing.expectEqual(@as(usize, 1), launch.kernel.launch.params.len);

    const dst = try h.alloc(0x1000);
    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    try launch.run(.{ 1, 1, 1 });

    for (0..threads) |i| {
        const expected: i32 = @intCast((threads - 1 - i) * 3 + 1);
        try testing.expectEqual(expected, dst.read(i32, i));
    }
}

test "live: the tiled shape runs on hardware, staging a shared tile in a uniform loop" {
    // The kernel this milestone was built for, end to end on silicon:
    //
    //     for (t in 0..tiles) {          // uniform trip count, from a scalar parameter
    //         tile[tid] = src[t * threads + tid];
    //         barrier;
    //         acc += tile[threads - 1 - tid];
    //         barrier;
    //     }
    //     out[tid] = acc;
    //
    // Five things have to hold at once: the uniformity analysis has to admit the two barriers
    // inside the loop, the shared frame has to give the tile an address, the global staging
    // load has to reach the right element, the staging has to cross threads, and the
    // loop-carried accumulator has to survive the back edge. Each thread reads the MIRROR
    // slot, so a run in which the staging never crossed threads gives a different number
    // rather than the same one, and the second barrier is what stops the next trip from
    // overwriting a slot another thread still reads.
    const allocator = testing.allocator;
    const threads = 32;
    const tiles = 4;

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const tile_t = try func.types.intern(.{ .array = .{ .len = threads, .elem = i32_t } });

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const out = try func.appendBlockParam(entry, ptr_t);
    const src = try func.appendBlockParam(entry, ptr_t);
    const trips = try func.appendBlockParam(entry, i32_t);
    const tid = try func.appendBlockParam(entry, i32_t);
    try gpu_abi.attrs.setBuiltin(&func, tid, .thread_id_x);
    try gpu_abi.attrs.setLocalSize(&func, .{ threads, 1, 1 });

    // The kernel's own tile, plus the two slot addresses every trip reuses.
    const tile = try func.appendInst(entry, shared_t, .{ .alloca = .{ .elem = tile_t } });
    const byte = try binImm(&func, entry, i32_t, .shl, tid, 2);
    const mirror = try binImm(&func, entry, i32_t, .add, try binImm(&func, entry, i32_t, .mul, tid, -1), threads - 1);
    const mbyte = try binImm(&func, entry, i32_t, .shl, mirror, 2);
    const mine = try ptrAddVal(&func, entry, shared_t, tile, byte);
    const theirs = try ptrAddVal(&func, entry, shared_t, tile, mbyte);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{ zero, zero }) } });

    // head(t, acc): the loop test. `trips` is a scalar parameter, so every thread of the
    // workgroup makes the same number of trips and the body's barriers are legal.
    const t = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, i32_t);
    const more = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = t, .rhs = trips } });
    try func.appendIf(head, more, .{ .target = body }, .{ .target = done, .args = &.{acc} });

    // body: stage this thread's element of tile t, wait, accumulate the MIRROR slot, wait.
    const row = try binImm(&func, body, i32_t, .mul, t, threads * 4);
    const at = try bin(&func, body, i32_t, .add, row, byte);
    const staged = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, body, ptr_t, src, at) } });
    try func.appendStore(body, staged, mine);
    try func.appendBarrier(body, .workgroup);
    const got = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = theirs } });
    const sum = try bin(&func, body, i32_t, .add, acc, got);
    try func.appendBarrier(body, .workgroup);
    const next = try binImm(&func, body, i32_t, .add, t, 1);
    func.setTerminator(body, .{ .jump = .{ .target = head, .args = try func.internValues(&.{ next, sum }) } });

    const total = try func.appendBlockParam(done, i32_t);
    try func.appendStore(done, total, try ptrAddVal(&func, done, ptr_t, out, byte));
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    try testing.expectEqual(@as(u32, 1), launch.kernel.launch.barrier_count);
    try testing.expectEqual(@as(u32, threads * 4), launch.kernel.launch.shared_bytes);

    const source = try h.alloc(threads * tiles * 4 + 0x1000);
    const feed = source.slice(i32);
    for (0..threads * tiles) |i| feed[i] = @intCast(i + 1); // src[i] = i + 1
    const dst = try h.alloc(0x1000);
    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, source.va);
    launch.setU32(launch.kernel.launch.params[2].offset, tiles);
    try launch.run(.{ 1, 1, 1 });

    for (0..threads) |i| {
        // acc[tid] = sum over t of src[t * threads + (threads - 1 - tid)], and src[i] = i + 1.
        var want: i32 = 0;
        for (0..tiles) |k| want += @intCast(k * threads + (threads - 1 - i) + 1);
        try testing.expectEqual(want, dst.read(i32, i));
    }
}
