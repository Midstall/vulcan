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

    /// Allocate a zeroed buffer at the GPU virtual address `va`. The runner hands
    /// addresses out of a bump pointer, so moving the pointer first places the mapping.
    /// The read-barrier test needs this: a clobbered address must land in MAPPED memory,
    /// or a missing barrier faults the channel instead of reporting a value.
    ///
    /// The driver keeps 0x0000_0000_FFE0_0000 up to 0x0000_0002_0000_0000 for itself, so
    /// every address passed here sits at or above 0x2_0000_0000. See `tests/carry.zig`.
    fn allocAt(self: *Harness, va: u64, size: u64) !compute.Buffer {
        self.runner.next_va = va;
        const buf = try self.runner.alloc(.system, size);
        std.debug.assert(buf.va == va);
        return buf;
    }

    /// Compile `func` under `abi` and bind it to this context.
    fn compile(self: *Harness, func: *Function, abi: gpu_abi.Abi) !Launch {
        return self.compileOpts(func, abi, .{});
    }

    /// `compile` with explicit code-generation options, for a test that needs the same IR
    /// built two ways. See `isel.Options`.
    fn compileOpts(self: *Harness, func: *Function, abi: gpu_abi.Abi, options: isel.Options) !Launch {
        const kernel = try isel.compileKernelOpts(testing.allocator, func, abi, options);
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

/// The buffer the load is TOLD to read. Placed, not bump-allocated, so its low half is
/// known and differs from the poison buffer's. Above the driver's reserved window.
const rb_good_va: u64 = 0x2_8000_0000;
/// The buffer the load reaches when the clobber wins. Its HIGH half is the same as
/// `rb_good_va`, and its low half is 0x4000_0000, which is `rb_clobber` squared. Both
/// buffers are mapped, so a lost read barrier reports a value and does not fault.
const rb_poison_va: u64 = 0x2_4000_0000;
/// The kernel computes `v * v` into the register that held the load's address low half.
/// 0x8000 squared is 0x4000_0000, the low half of `rb_poison_va`.
const rb_clobber: u32 = 0x8000;
const rb_good_magic: i32 = 0x1111_1111;
const rb_poison_magic: i32 = 0x7777_7777;
/// How many one-thread workgroups the dispatch needs before the memory pipe is busy
/// enough for the race to show. Measured on an RTX 5070: 8192 was intermittent and
/// 16384 was every run. This is four times the reliable figure.
const rb_grid: u32 = 65535;

test "live: a load's address register survives being reused one instruction later" {
    // The WRITE-AFTER-READ hazard the read scoreboards exist for. `assignLocs` ends the
    // address value's live range at the load and gives its register to the next value, so
    // the emitted stream is:
    //
    //     LDG R9, [R4:R5]
    //     IMAD R4, R8, R8      <- the address LOW half, one instruction later
    //
    // An LDG is decoupled: it reads its address when its pipe reaches it, not when it
    // issues. Without a read barrier on the LDG the IMAD lands first under load, and the
    // LDG reads the CLOBBERED address. Both addresses are mapped here, so the wrong answer
    // is a value rather than an Xid.
    //
    // Measured with the barrier removed: every run at this grid returned the poison value.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const out = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, i32_t);
    // `p` dies at the load, so the multiply below takes its register.
    const x = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    const y = try bin(&func, b, i32_t, .mul, v, v);
    const z = try bin(&func, b, i32_t, .add, x, y);
    try func.appendStore(b, z, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();

    const good = try h.allocAt(rb_good_va, 0x1000);
    const poison = try h.allocAt(rb_poison_va, 0x1000);
    const out_buf = try h.alloc(0x1000);
    good.slice(i32)[0] = rb_good_magic;
    poison.slice(i32)[0] = rb_poison_magic;

    launch.setPtr(launch.kernel.launch.params[0].offset, good.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, out_buf.va);
    launch.setU32(launch.kernel.launch.params[2].offset, rb_clobber);
    try launch.run(.{ rb_grid, 1, 1 });

    // Every thread writes the same slot, so one surviving thread is enough to show it.
    const squared: i32 = @bitCast(rb_clobber * rb_clobber);
    try testing.expectEqual(rb_good_magic +% squared, out_buf.read(i32, 0));
    try testing.expect(out_buf.read(i32, 0) != rb_poison_magic +% squared);
}

/// The two integers the convert test squares. They differ, so the poison a missed
/// scoreboard produces (the SECOND convert's consumer reading the FIRST convert's result)
/// is a different number from the right answer.
const conv_v: u32 = 3;
const conv_w: u32 = 5;
/// `v * v + w * w`, the only right answer.
const conv_want: f32 = 34.0;
/// How many one-thread workgroups the dispatch needs. Measured on an RTX 5070 with the
/// write barriers removed, over eight runs at each size: one block gave a wrong answer six
/// times out of eight, and 1024 blocks gave one every run. The wrong answers were 9, 25,
/// 50, 650, 1181, 1394786 and `inf`, so this hazard does not need occupancy the way the
/// read-barrier one did. 1024 is the size that was wrong every time.
const conv_grid: u32 = 1024;

test "live: a converted value reaches its consumer, and not the register's old contents" {
    // I2F and F2I are DECOUPLED on sm120 (NAK sm120_instr_latencies: `Op::I2F(_) =>
    // Decoupled`), so the converted value lands an unknown number of cycles after issue.
    // NAK's own RAW table gives a decoupled write "1 & sb" against every consumer class:
    // one cycle of delay AND a scoreboard, because no fixed number covers it. The emitted
    // stream is
    //
    //     I2F  R8, R7      <- (float) w
    //     FMUL R7, R8, R8
    //     I2F  R8, R6      <- (float) v, the SAME destination register
    //     FMUL R6, R8, R8
    //     FADD R8, R6, R7
    //
    // so if the second convert has not landed, the second FMUL squares the FIRST convert's
    // result and the kernel answers `2 * w * w`. That is the tidy failure. The others were
    // arbitrary, because the register held whatever the hardware left in it.
    //
    // CONTRACTION IS OFF HERE ON PURPOSE. `v * v + w * w` is a multiply-add, so with
    // contraction on the second multiply folds into the add, `w`'s converted value stays
    // live to the fused instruction, and the two I2Fs land in DIFFERENT registers. The
    // stream above is then no longer what the kernel emits, and the shape this test pins
    // stops existing. The scoreboard rule it guards belongs to I2F and not to
    // contraction, so the test keeps the unfused shape and states why.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, i32_t);
    const w = try func.appendBlockParam(b, i32_t);
    const wf = try func.appendInst(b, f32_t, .{ .convert = .{ .value = w } });
    const w2 = try bin(&func, b, f32_t, .mul, wf, wf);
    const vf = try func.appendInst(b, f32_t, .{ .convert = .{ .value = v } });
    const v2 = try bin(&func, b, f32_t, .mul, vf, vf);
    const sum = try bin(&func, b, f32_t, .add, v2, w2);
    try func.appendStore(b, sum, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compileOpts(&func, runner_abi, .{ .contract_fma = false });
    defer launch.deinit();

    const out_buf = try h.alloc(0x1000);
    launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
    launch.setU32(launch.kernel.launch.params[1].offset, conv_v);
    launch.setU32(launch.kernel.launch.params[2].offset, conv_w);
    try launch.run(.{ conv_grid, 1, 1 });

    // Every thread writes the same slot, so one thread that lost the race shows it.
    try testing.expectEqual(conv_want, out_buf.read(f32, 0));
}

// The operand triple every float contraction test below uses, as IEEE-754 binary32 bit
// patterns. ALL THREE HAVE DIRTY LOW MANTISSA BITS, which is a deliberate choice: a
// register FADD that silently truncated its source A to bfloat16 once survived every float
// test in this repository, because those tests used values such as `1.0 + 0.001` whose low
// 16 mantissa bits are zero. These do not survive that.
//
// The triple also SEPARATES THE TWO ROUNDINGS. `a * b + c` rounds once fused and twice
// unfused, and here the two answers differ, so a test that expects the fused answer really
// does prove the FFMA fired. `fmaRefs` re-checks that separation at the point of use, and
// does not take it on trust from this comment.
const fma_a: u32 = 0x3F7FFFFD;
const fma_b: u32 = 0x3F7FFFFB;
const fma_c: u32 = 0xBF7FFFF9;

/// The two answers `a * b + c` has: fused, rounding once, and unfused, rounding twice.
/// They must differ, or a test built on them cannot tell an FFMA from an FMUL plus an FADD.
fn fmaRefs(a: f32, b: f32, c: f32) !struct { fused: f32, unfused: f32 } {
    const fused = @mulAdd(f32, a, b, c);
    const unfused = a * b + c;
    // The host compiler must not fuse this itself, or both references would be the fused
    // answer and every assertion below would pass for the wrong reason.
    try testing.expect(fused != unfused);
    return .{ .fused = fused, .unfused = unfused };
}

/// A void kernel `out[0] = a * b + c` over three f32 parameters. `commuted` writes the add
/// as `c + a * b`, which is the same arithmetic in the other operand order.
fn buildFmaKernel(allocator: std.mem.Allocator, commuted: bool) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, f32_t);
    const y = try func.appendBlockParam(b, f32_t);
    const z = try func.appendBlockParam(b, f32_t);
    const prod = try bin(&func, b, f32_t, .mul, x, y);
    const sum = if (commuted)
        try bin(&func, b, f32_t, .add, z, prod)
    else
        try bin(&func, b, f32_t, .add, prod, z);
    try func.appendStore(b, sum, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    return func;
}

test "live: a float multiply-add FUSES on the silicon, and rounds once instead of twice" {
    // THE ONLY TEST THAT CAN TELL THE TWO APART IS A TEST OVER VALUES. An FFMA and an
    // FMUL plus an FADD compute the same thing to within a rounding, so an opcode count
    // proves the shorter stream and nothing about the answer. Here the operands are
    // chosen so single rounding and double rounding give DIFFERENT bits, and the same IR
    // is compiled both ways against the same hardware: contraction on must give the
    // once-rounded answer, and contraction off the twice-rounded one.
    const allocator = testing.allocator;
    const a: f32 = @bitCast(fma_a);
    const b: f32 = @bitCast(fma_b);
    const c: f32 = @bitCast(fma_c);
    const want = try fmaRefs(a, b, c);

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    for ([_]bool{ false, true }) |commuted| {
        for ([_]bool{ true, false }) |contract| {
            var func = try buildFmaKernel(allocator, commuted);
            defer func.deinit();
            var launch = try h.compileOpts(&func, runner_abi, .{ .contract_fma = contract });
            defer launch.deinit();

            out_buf.slice(u32)[0] = 0;
            launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
            launch.setU32(launch.kernel.launch.params[1].offset, fma_a);
            launch.setU32(launch.kernel.launch.params[2].offset, fma_b);
            launch.setU32(launch.kernel.launch.params[3].offset, fma_c);
            try launch.run(.{ 1, 1, 1 });

            // The commuted form `c + a * b` must fuse exactly as `a * b + c` does, so both
            // rows expect the same number.
            try testing.expectEqual(
                if (contract) want.fused else want.unfused,
                out_buf.read(f32, 0),
            );
        }
    }
}

test "live: a CONSTANT-scale multiply-add fuses through the immediate FFMA field" {
    // `x * k + y`, where the scale is a compile-time constant that `foldConstantsToImm`
    // has already moved into the multiply's immediate field. The fused form is
    // `FFMA dst, x, k, y`, which is a source form this encoder had never emitted before:
    // the immediate is at src1 and the addend is a register at src2. `nvdisasm -b SM120
    // -c` reads the word back as `FFMA R6, R4, 0.25, R7`, and this test is the other half
    // of that proof, because a decoder agreeing on the operand names still does not say
    // the hardware multiplies by the right one.
    const allocator = testing.allocator;
    const x: f32 = @bitCast(fma_a);
    const k: f32 = @bitCast(fma_b);
    const y: f32 = @bitCast(fma_c);
    const want = try fmaRefs(x, k, y);

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    for ([_]bool{ true, false }) |contract| {
        var func = Function.init(allocator);
        defer func.deinit();
        const f32_t = try func.types.intern(.{ .float = .f32 });
        const ptr_t = try func.types.ptrGlobal();
        const blk = try func.appendBlock();
        const out = try func.appendBlockParam(blk, ptr_t);
        const xp = try func.appendBlockParam(blk, f32_t);
        const yp = try func.appendBlockParam(blk, f32_t);
        const kc = try func.appendInst(blk, f32_t, .{ .fconst = k });
        const scaled = try bin(&func, blk, f32_t, .mul, xp, kc);
        const sum = try bin(&func, blk, f32_t, .add, scaled, yp);
        try func.appendStore(blk, sum, out);
        func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });

        var launch = try h.compileOpts(&func, runner_abi, .{ .contract_fma = contract });
        defer launch.deinit();

        out_buf.slice(u32)[0] = 0;
        launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
        launch.setU32(launch.kernel.launch.params[1].offset, fma_a);
        launch.setU32(launch.kernel.launch.params[2].offset, fma_c);
        try launch.run(.{ 1, 1, 1 });

        try testing.expectEqual(if (contract) want.fused else want.unfused, out_buf.read(f32, 0));
    }
}

/// A second multiplicand for the hoist test, with nonzero low mantissa bits of its own.
/// `fmaRefs` re-checks that it separates single from double rounding, so the second site
/// cannot quietly decay into one that passes whichever answer the hardware gives.
const fma_a2: u32 = 0x3F7FFFF7;

test "live: `x * k1 + k2` fuses against a HOISTED constant addend, and rounds once" {
    // BOTH THE MULTIPLIER AND THE ADDEND ARE CONSTANTS, which is the shape a weight
    // dequantization and a scale-and-bias take. FFMA holds exactly one immediate, and its
    // third source must be a register, so the addend reaches the instruction through a
    // register the prologue loaded once. TWO SITES SHARE THAT ONE CONSTANT here, because a
    // lone site in straight-line code is refused: the prologue MOV would only break even.
    //
    // THE ANSWER IS THE ONLY THING THAT SAYS THE RIGHT REGISTER REACHED THE RIGHT SOURCE.
    // An FFMA that read the wrong register is shorter AND wrong, and the operands below
    // make single rounding and double rounding differ, which `fmaRefs` re-checks at the
    // point of use.
    const allocator = testing.allocator;
    const k1: f32 = @bitCast(fma_b);
    const k2: f32 = @bitCast(fma_c);
    const want0 = try fmaRefs(@bitCast(fma_a), k1, k2);
    const want1 = try fmaRefs(@bitCast(fma_a2), k1, k2);

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    for ([_]bool{ true, false }) |contract| {
        var func = Function.init(allocator);
        defer func.deinit();
        const f32_t = try func.types.intern(.{ .float = .f32 });
        const ptr_t = try func.types.ptrGlobal();
        const blk = try func.appendBlock();
        const out = try func.appendBlockParam(blk, ptr_t);
        const x0 = try func.appendBlockParam(blk, f32_t);
        const x1 = try func.appendBlockParam(blk, f32_t);
        const kc1 = try func.appendInst(blk, f32_t, .{ .fconst = k1 });
        const kc2 = try func.appendInst(blk, f32_t, .{ .fconst = k2 });
        const m0 = try bin(&func, blk, f32_t, .mul, x0, kc1);
        const s0 = try bin(&func, blk, f32_t, .add, m0, kc2);
        try func.appendStore(blk, s0, out);
        const m1 = try bin(&func, blk, f32_t, .mul, x1, kc1);
        const s1 = try bin(&func, blk, f32_t, .add, m1, kc2);
        try func.appendStore(blk, s1, try ptrAdd(&func, blk, ptr_t, out, 4));
        func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });

        var launch = try h.compileOpts(&func, runner_abi, .{ .contract_fma = contract });
        defer launch.deinit();

        out_buf.slice(u32)[0] = 0;
        out_buf.slice(u32)[1] = 0;
        launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
        launch.setU32(launch.kernel.launch.params[1].offset, fma_a);
        launch.setU32(launch.kernel.launch.params[2].offset, fma_a2);
        try launch.run(.{ 1, 1, 1 });

        try testing.expectEqual(if (contract) want0.fused else want0.unfused, out_buf.read(f32, 0));
        try testing.expectEqual(if (contract) want1.fused else want1.unfused, out_buf.read(f32, 1));
    }
}

test "live: a hoisted constant addend survives every trip of a loop" {
    // THE HOISTED REGISTER IS WRITTEN ONCE AND READ FOREVER, so anything that reused it
    // inside the loop would give a wrong answer only from the second trip on. A single
    // dispatch of the straight-line shape cannot see that. This runs the recurrence
    // `acc = acc * 0.9 + 0.05` for sixteen trips and compares every bit against a host
    // reference built from the same single-rounded step.
    const allocator = testing.allocator;
    const trips: i32 = 16;
    const seed: f32 = 0.125;
    var want: f32 = seed;
    for (0..@intCast(trips)) |_| want = @mulAdd(f32, want, 0.9, 0.05);

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const out = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const s = try func.appendBlockParam(entry, f32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, s });

    const t = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, f32_t);
    const more = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = t, .rhs = n } });
    try func.appendIf(head, more, .{ .target = body }, .{ .target = done, .args = &.{acc} });

    const kc1 = try func.appendInst(body, f32_t, .{ .fconst = 0.9 });
    const kc2 = try func.appendInst(body, f32_t, .{ .fconst = 0.05 });
    const scaled = try bin(&func, body, f32_t, .mul, acc, kc1);
    const stepped = try bin(&func, body, f32_t, .add, scaled, kc2);
    const one = try func.appendInst(body, i32_t, .{ .iconst = 1 });
    const next = try bin(&func, body, i32_t, .add, t, one);
    try func.setJump(body, head, &.{ next, stepped });

    const result = try func.appendBlockParam(done, f32_t);
    try func.appendStore(done, result, out);
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    var launch = try h.compileOpts(&func, runner_abi, .{});
    defer launch.deinit();

    out_buf.slice(u32)[0] = 0;
    launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
    launch.setU32(launch.kernel.launch.params[1].offset, @bitCast(trips));
    launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(seed));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(want, out_buf.read(f32, 0));
}

test "live: a counted loop runs one main trip of 8 and a remainder of 5" {
    // THE MAIN BLOCK AND THE TAIL MUST AGREE ON THE COUNTER. The unroller
    // splits a counted SIMT loop into a main block of 8 unguarded copies plus
    // the original loop as the remainder, entered through one guard that tests
    // `counter + 7` against the bound. A guard that over-runs takes trips past
    // the bound, a remainder that loses its counter starts the tail at the
    // wrong value, and neither shows up in a single dispatch of the
    // straight-line shape. This sums the induction variable for 13 trips: one
    // main trip of 8 plus a remainder of 5, which exercises the main loop, the
    // tail, and the exit, and compares the exact total.
    const allocator = testing.allocator;
    const trips: i32 = 13;
    const want: i32 = 13 * 12 / 2; // 0 + 1 + ... + 12

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const out = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, zero });

    const i = try func.appendBlockParam(head, i32_t);
    const s = try func.appendBlockParam(head, i32_t);
    const more = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(head, more, .{ .target = body, .args = &.{ i, s } }, .{ .target = done, .args = &.{s} });

    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, i32_t);
    const sum = try bin(&func, body, i32_t, .add, bs, bi);
    const next = try binImm(&func, body, i32_t, .add, bi, 1);
    try func.setJump(body, head, &.{ next, sum });

    const result = try func.appendBlockParam(done, i32_t);
    try func.appendStore(done, result, out);
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    // Unroll for sm_120 first. The body budget is 4 (two header instructions
    // plus the body's two), which the SIMT rule answers with the factor 8.
    const g = target.opt.microarch.modelFor(.sm_120);
    const changed = try target.opt.microarch.unroll.run(allocator, &func, g);
    try testing.expect(changed);

    var launch = try h.compileOpts(&func, runner_abi, .{});
    defer launch.deinit();

    out_buf.slice(u32)[0] = 0;
    launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
    launch.setU32(launch.kernel.launch.params[1].offset, @bitCast(trips));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(want, out_buf.read(i32, 0));
}

test "live: an integer multiply-add through IMAD gives the same bits fused or not" {
    // Integer contraction is EXACT either way. `a * b + c` in 32-bit wrapping arithmetic
    // has one answer, so unlike the float case the switch must not change the number. It
    // still changes the instruction count, and the operands below make a wrong addend
    // field visible: the register form `base + i * stride` and the immediate form
    // `base + i * 4` both run, and each product wraps past 32 bits so a wide multiply
    // would answer differently from a narrow one.
    const allocator = testing.allocator;
    const base: i32 = -1234567;
    const index: i32 = 0x0001_3579;
    const stride: i32 = 0x0002_4680;
    const want_reg: i32 = base +% index *% stride;
    const want_imm: i32 = base +% index *% 4;

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    for ([_]bool{ true, false }) |contract| {
        var func = Function.init(allocator);
        defer func.deinit();
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const ptr_t = try func.types.ptrGlobal();
        const blk = try func.appendBlock();
        const out = try func.appendBlockParam(blk, ptr_t);
        const bp = try func.appendBlockParam(blk, i32_t);
        const ip = try func.appendBlockParam(blk, i32_t);
        const sp = try func.appendBlockParam(blk, i32_t);
        // The register form: the add takes the multiply on its LEFT.
        const prod = try bin(&func, blk, i32_t, .mul, ip, sp);
        const reg_sum = try bin(&func, blk, i32_t, .add, prod, bp);
        try func.appendStore(blk, reg_sum, out);
        // The immediate form, stored one slot along: the scale is a constant.
        const four = try func.appendInst(blk, i32_t, .{ .iconst = 4 });
        const scaled = try bin(&func, blk, i32_t, .mul, ip, four);
        const imm_sum = try bin(&func, blk, i32_t, .add, bp, scaled);
        const slot1 = try ptrAdd(&func, blk, ptr_t, out, 4);
        try func.appendStore(blk, imm_sum, slot1);
        func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });

        var launch = try h.compileOpts(&func, runner_abi, .{ .contract_fma = contract });
        defer launch.deinit();

        out_buf.slice(i32)[0] = 0;
        out_buf.slice(i32)[1] = 0;
        launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
        launch.setU32(launch.kernel.launch.params[1].offset, @bitCast(base));
        launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(index));
        launch.setU32(launch.kernel.launch.params[3].offset, @bitCast(stride));
        try launch.run(.{ 1, 1, 1 });

        try testing.expectEqual(want_reg, out_buf.read(i32, 0));
        try testing.expectEqual(want_imm, out_buf.read(i32, 1));
    }
}

test "live: a dot product loop carries its counter and accumulator across the closing branch" {
    // THE CLOSING BRANCH MUST LAND ON ITS EDGE MOVES. The rotated main block
    // of the counted unroll ends in a guarded branch whose taken edge carries
    // the counter and the accumulator back to the block's own params. When the
    // allocator gives a carried value a register other than the param's, the
    // taken path must run the move first: a branch that lands past it loses
    // the value, and the loop reads the wrong elements. Two loads per trip
    // make the shape the same as the matmul bench that caught this.
    const allocator = testing.allocator;
    const trips: i32 = 21;

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);
    const a_buf = try h.alloc(0x1000);
    const b_buf = try h.alloc(0x1000);

    // Every product is exactly 2, so every partial sum is exact and the
    // total is exact no matter how the additions group.
    for (a_buf.slice(f32)) |*x| x.* = 1.0;
    for (b_buf.slice(f32)) |*x| x.* = 2.0;
    const want: f32 = 2.0 * @as(f32, @floatFromInt(trips));

    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const out = try func.appendBlockParam(entry, ptr_t);
    const ap = try func.appendBlockParam(entry, ptr_t);
    const bp = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const fzero = try func.appendInst(entry, f32_t, .{ .fconst = 0.0 });
    try func.setJump(entry, head, &.{ zero, fzero });

    const i = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, f32_t);
    const more = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(head, more, .{ .target = body }, .{ .target = done, .args = &.{acc} });

    const byte = try binImm(&func, body, i32_t, .shl, i, 2);
    const av = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, body, ptr_t, ap, byte) } });
    const bv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = try ptrAddVal(&func, body, ptr_t, bp, byte) } });
    const prod = try bin(&func, body, f32_t, .mul, av, bv);
    const sum = try bin(&func, body, f32_t, .add, acc, prod);
    const next = try binImm(&func, body, i32_t, .add, i, 1);
    try func.setJump(body, head, &.{ next, sum });

    const result = try func.appendBlockParam(done, f32_t);
    try func.appendStore(done, result, out);
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    const g = target.opt.microarch.modelFor(.sm_120);
    _ = try target.opt.microarch.unroll.run(allocator, &func, g);

    var launch = try h.compileOpts(&func, runner_abi, .{});
    defer launch.deinit();

    out_buf.slice(u32)[0] = 0;
    launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, a_buf.va);
    launch.setPtr(launch.kernel.launch.params[2].offset, b_buf.va);
    launch.setU32(launch.kernel.launch.params[3].offset, @bitCast(trips));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(want, out_buf.read(f32, 0));
}

test "live: a taken branch lands on its then-edge moves" {
    // The guarded branch of an `if` must land ON the moves of its taken
    // edge, never past them. Passing one live value to three parameters of
    // the target forces two copies that only the taken path runs: a branch
    // that skips them leaves the parameters reading whatever sat in their
    // registers. The else edge keeps the value, so both paths are checkable.
    const allocator = testing.allocator;
    const x: i32 = 1234567;

    var h = try Harness.open();
    defer h.deinit();
    const out_buf = try h.alloc(0x1000);

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const then_block = try func.appendBlock();
    const done = try func.appendBlock();

    const out = try func.appendBlockParam(entry, ptr_t);
    const xv = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const more = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ne, .lhs = xv, .rhs = zero } });
    try func.appendIf(entry, more, .{ .target = then_block, .args = &.{ xv, xv, xv } }, .{ .target = done, .args = &.{xv} });

    const a = try func.appendBlockParam(then_block, i32_t);
    const b = try func.appendBlockParam(then_block, i32_t);
    const c = try func.appendBlockParam(then_block, i32_t);
    const ab = try bin(&func, then_block, i32_t, .add, a, b);
    const abc = try bin(&func, then_block, i32_t, .add, ab, c);
    try func.appendStore(then_block, abc, out);
    func.setTerminator(then_block, .{ .ret = ir.function.Ret.none() });

    const kept = try func.appendBlockParam(done, i32_t);
    try func.appendStore(done, kept, out);
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();

    out_buf.slice(u32)[0] = 0;
    launch.setPtr(launch.kernel.launch.params[0].offset, out_buf.va);
    launch.setU32(launch.kernel.launch.params[1].offset, @bitCast(x));
    try launch.run(.{ 1, 1, 1 });
    try testing.expectEqual(3 * x, out_buf.read(i32, 0));
}
