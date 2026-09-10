//! The 64-bit address carry chain, proved on the silicon.
//!
//! A global pointer lives in a GPR PAIR, so `ptr + offset` is two instructions:
//! an `IADD3` that adds the low halves and writes a carry-out predicate, then an
//! `IADD3.X` that adds the high halves and READS that predicate. The `.X` flag
//! is bit 74 of the second instruction, and it is the only thing that makes the
//! hardware look at the carry at all. Without it the pair still assembles, still
//! disassembles as two plain adds, and computes an address 4 GiB too low
//! whenever the low halves overflow.
//!
//! Every other pointer test in this repository uses buffers whose low halves
//! never overflow, so all of them pass with the carry dropped. This file exists
//! to make the carry the ONLY difference between a right answer and a wrong one.
//!
//! ## How the address is arranged
//!
//! `compute.Runner` hands out GPU virtual addresses from a bump pointer, and
//! `next_va` is data, so a test can put a buffer where it wants one. Two buffers
//! are placed EXACTLY 4 GiB apart. The kernel then adds a base and an offset
//! whose low halves sum past 2^32:
//!
//!     base   0x0000_0002_FFF0_0000   (a bare address, never read or written)
//!     offset 0x0000_0000_FFF0_0000
//!     sum    0x0000_0003_FFE0_0000   with the carry, the HIGH buffer
//!     sum    0x0000_0002_FFE0_0000   without it, the LOW buffer
//!
//! Both outcomes are mapped memory, so a dropped carry writes the wrong buffer
//! instead of faulting the channel, and the test reports a value rather than a
//! timeout.
//!
//! ## The reserved address window
//!
//! The driver keeps a range of virtual addresses for itself. A mapping at
//! 0x1_0000_0000 is REFUSED outright, and a mapping anywhere else in the range is
//! accepted and then silently drops every store the GPU sends to it. No error, no
//! fault, the writes simply vanish.
//!
//! The nvidia.zig session measured the range more carefully than the first pass
//! here did, sweeping with a FRESH CHANNEL PER ADDRESS because a hung dispatch
//! poisons the channel and makes a shared-channel sweep report a false result.
//! Their figures, on a GB10: 0xFE00_0000 works, everything from 0xFF00_0000 up to
//! 0x1_FFF0_0000 swallows stores, and 0x2_0000_0000 works again. So the lower
//! bound is 0xFF00_0000, lower than the 0xFFE0_0000 this file first recorded, and
//! real code mapping at 0xFF00_0000 would have vanished.
//!
//! TREAT THE EXACT EDGES AS DRIVER AND PART DEPENDENT. That sweep ran on a GB10;
//! this machine is an RTX 5070. Both agree the window exists and that
//! 0x2_0000_0000 and above is safe, which is what the addresses below rely on.
//! Every address in this file sits well clear of it, which is why the pair is at
//! 0x2_FFE0_0000 and 0x3_FFE0_0000 rather than the more obvious 0xFFE0_0000.
//!
//! nvidia.zig now refuses the whole range in `rm.Client.mapToGpu` with
//! `error.ReservedGpuAddress`, so a consumer of that allocator is protected.
//!
//! ## Skipping
//!
//! `compute.Runner.init` gives `error.SkipZigTest` when no GPU answers, and
//! `gpu()` turns a permission error or a missing driver node into the same
//! result. A machine with no NVIDIA hardware skips every live test here. The
//! structural test at the end needs no GPU and always runs.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu_abi = @import("vulcan-gpu");
const target = @import("vulcan-target");
const nvidia = @import("nvidia");

const isel = target.nvidia.isel;
const encode = target.nvidia.encode;
const compute = nvidia.compute;
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const testing = std.testing;

/// The parameter ABI of the nvidia.zig dispatch: it binds the caller's buffer as
/// the BASE of constant bank 0, so the first parameter sits at offset 0.
const runner_abi: gpu_abi.Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = false,
};

/// The low buffer's address. Its own low half is large, so a second buffer 4 GiB
/// above it has the SAME low half and a high half one greater. That is what lets
/// one add reach either buffer depending on the carry alone.
///
/// The value is 2 MiB aligned, because that is the granule `Runner` maps at, it
/// is far above the runner's own `VA_BASE` allocations, and it is above the
/// driver's reserved window. See the module comment.
const low_va: u64 = 0x2_FFE0_0000;
/// The high buffer, exactly 4 GiB above the low one.
const high_va: u64 = low_va + (1 << 32);
/// The base the kernel adds to. It is never dereferenced.
const carry_base: u64 = 0x2_FFF0_0000;
/// The offset the kernel adds. `carry_base + carry_offset` is `high_va`, and the
/// two low halves sum past 2^32, so the sum needs the carry.
const carry_offset: u32 = 0xFFF0_0000;

/// The chained-add test starts at this address and adds it three more times.
/// Every one of the three adds overflows the low half.
const chain_step: u64 = 0xF000_0000;
/// Where four times `chain_step` lands when all three carries propagate.
const chain_carried_va: u64 = chain_step * 4;
/// Where it lands when none of them does: the same low half, high half still 0.
const chain_dropped_va: u64 = (chain_step * 4) & 0xFFFF_FFFF;

comptime {
    std.debug.assert(carry_base + carry_offset == high_va);
    std.debug.assert(@as(u64, @as(u32, @truncate(carry_base))) + carry_offset > 0xFFFF_FFFF);
    std.debug.assert(chain_carried_va >> 32 == 3);
    std.debug.assert(chain_dropped_va >> 32 == 0);
    std.debug.assert(chain_carried_va & 0xFFFF_FFFF == chain_dropped_va);
}

/// Whether `err` means "this machine has no usable NVIDIA GPU". A DISPATCH
/// failure such as `error.GridTimeout` is deliberately absent: a kernel that
/// hangs must fail the test and not skip it.
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

/// An open GPU context: the compute channel plus the one parameter buffer every
/// kernel binds as constant bank 0.
const Harness = struct {
    runner: compute.Runner,
    params: compute.Buffer,

    fn open() !Harness {
        var runner = try gpu();
        errdefer runner.deinit();
        const params = try runner.alloc(.system, 0x1000);
        return .{ .runner = runner, .params = params };
    }

    fn deinit(self: *Harness) void {
        self.runner.deinit();
    }

    /// Allocate a zeroed buffer at the GPU virtual address `va`. The runner hands
    /// addresses out of a bump pointer, so moving the pointer first places the
    /// mapping. Every call here asks for an address above everything the runner
    /// allocates for itself.
    fn allocAt(self: *Harness, va: u64, size: u64) !compute.Buffer {
        self.runner.next_va = va;
        const buf = try self.runner.alloc(.system, size);
        std.debug.assert(buf.va == va);
        return buf;
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
    base: u32,

    fn deinit(self: *Launch) void {
        self.kernel.deinit(testing.allocator);
    }

    fn setPtr(self: *Launch, off: u32, va: u64) void {
        std.mem.writeInt(u64, self.harness.params.bytes[self.base + off ..][0..8], va, .little);
    }

    fn setU32(self: *Launch, off: u32, v: u32) void {
        std.mem.writeInt(u32, self.harness.params.bytes[self.base + off ..][0..4], v, .little);
    }

    fn run(self: *Launch, grid: [3]u32) !void {
        try self.harness.runner.run(self.kernel.code, .{
            .grid = grid,
            .block = self.kernel.launch.block,
            .register_count = self.kernel.launch.reg_count,
            .cbuf0_va = self.harness.params.va,
            .cbuf0_size = @max(self.base + self.kernel.launch.param_bytes, 16),
            .shared_mem_bytes = self.kernel.launch.shared_bytes,
        });
    }
};

/// `base + off` as a fresh pointer value, where `off` is a byte count held in a
/// register. This is the instruction pair the whole file is about.
fn ptrAddVal(func: *Function, b: Block, ptr_t: ir.types.Type, base: Value, off: Value) !Value {
    return func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
}

/// A kernel that stores `value` at `base + offset`, where all three are
/// parameters. The address arithmetic is the only work it does.
fn buildOffsetStore(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const offset = try func.appendBlockParam(b, i32_t);
    const value = try func.appendBlockParam(b, i32_t);
    const at = try ptrAddVal(&func, b, ptr_t, base, offset);
    try func.appendStore(b, value, at);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    return func;
}

test "live: a 64-bit pointer add carries out of the low half" {
    // The one test in this repository whose answer depends on the `.X` bit. With
    // the carry the store lands in the HIGH buffer; without it, 4 GiB lower, in
    // the LOW buffer. Both are mapped, so the wrong answer is a value and not a
    // fault.
    const allocator = testing.allocator;
    var func = try buildOffsetStore(allocator);
    defer func.deinit();

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();

    const low = try h.allocAt(low_va, 0x1000);
    const high = try h.allocAt(high_va, 0x1000);

    const magic: i32 = 0x5EED_C0DE;
    launch.setPtr(launch.kernel.launch.params[0].offset, carry_base);
    launch.setU32(launch.kernel.launch.params[1].offset, carry_offset);
    launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(magic));
    try launch.run(.{ 1, 1, 1 });

    // 0xFFF0_0000 + 0xFFF0_0000 = 0x1_FFE0_0000. The high half of the base is
    // zero, so ONLY the carry can put a 1 in the high half of the address.
    try testing.expectEqual(magic, high.read(i32, 0));
    try testing.expectEqual(@as(i32, 0), low.read(i32, 0));
}

test "live: an add that does NOT carry still leaves the high half alone" {
    // The other direction, and the reason the fix is not "always add one". The
    // same kernel with an offset small enough to stay inside the low half must
    // reach the LOW buffer. An `IADD3.X` that read a stale or always-set carry
    // would fail here and pass the test above.
    const allocator = testing.allocator;
    var func = try buildOffsetStore(allocator);
    defer func.deinit();

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();

    const low = try h.allocAt(low_va, 0x1000);
    const high = try h.allocAt(high_va, 0x1000);

    const magic: i32 = 0x0BAD_F00D;
    launch.setPtr(launch.kernel.launch.params[0].offset, low_va);
    launch.setU32(launch.kernel.launch.params[1].offset, 0x40);
    launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(magic));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(magic, low.read(i32, 16)); // byte 0x40 is word 16
    try testing.expectEqual(@as(i32, 0), high.read(i32, 0));
}

test "live: a chain of carrying adds propagates every carry, not just the first" {
    // Three adds in a row, each of which overflows the low half, so the high
    // half must be incremented three times. A carry predicate that is written
    // once and never refreshed gives the wrong buffer here.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const step = try func.appendBlockParam(b, i32_t);
    const value = try func.appendBlockParam(b, i32_t);
    var at = base;
    for (0..3) |_| at = try ptrAddVal(&func, b, ptr_t, at, step);
    try func.appendStore(b, value, at);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();

    // Start and step are both 0xF000_0000, so each of the three adds overflows
    // the low half and the high half ends at 3. With no carry at all the low
    // half is the same and the high half stays 0, which is `chain_dropped_va`.
    // Both are mapped, so neither outcome faults the channel.
    const step_bytes: u32 = chain_step;
    const start: u64 = chain_step;

    const dropped = try h.allocAt(chain_dropped_va, 0x1000);
    const carried = try h.allocAt(chain_carried_va, 0x1000);

    const magic: i32 = 0x1234_5678;
    launch.setPtr(launch.kernel.launch.params[0].offset, start);
    launch.setU32(launch.kernel.launch.params[1].offset, step_bytes);
    launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(magic));
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(magic, carried.read(i32, 0));
    try testing.expectEqual(@as(i32, 0), dropped.read(i32, 0));
}

test "live: a weak global load and store still reach the host after the grid ends" {
    // LDG and STG used to ask for `STRONG.SYS` on EVERY access, which is
    // system-scope coherence on every load and store a kernel makes. ptxas and
    // NAK both emit the weak default. The launch's end-of-grid release is what
    // makes a weak write visible to the host, so this reads a value the host
    // wrote and writes one the host reads, with nothing but that release
    // between them.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = in } });
    const doubled = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = v } });
    try func.appendStore(b, doubled, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const dst = try h.runner.alloc(.system, 0x1000);
    const src = try h.runner.alloc(.system, 0x1000);
    src.slice(i32)[0] = 21;

    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, src.va);
    try launch.run(.{ 1, 1, 1 });

    try testing.expectEqual(@as(i32, 42), dst.read(i32, 0));
}

test "live: a boolean in P0 survives a global load" {
    // LDG used to write its fault predicate into P0 and to guard itself on
    // `!P0`, because the uniform base was written at bit 64 and pushed both
    // fields off their places. P0 is the FIRST predicate the boolean allocator
    // hands out, so a bool held across a load was destroyed, and a load reached
    // by a true bool did not execute at all.
    //
    // The kernel compares two parameters into a bool, loads through a pointer,
    // and then selects on the bool it computed BEFORE the load.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const in = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);
    const y = try func.appendBlockParam(b, i32_t);
    const flag = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = y } });
    const loaded = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = in } });
    const other = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = loaded, .rhs = loaded } });
    const picked = try func.appendInst(b, i32_t, .{ .select = .{ .cond = flag, .then = loaded, .@"else" = other } });
    try func.appendStore(b, picked, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var h = try Harness.open();
    defer h.deinit();
    var launch = try h.compile(&func, runner_abi);
    defer launch.deinit();
    const dst = try h.runner.alloc(.system, 0x1000);
    const src = try h.runner.alloc(.system, 0x1000);
    src.slice(i32)[0] = 7;

    launch.setPtr(launch.kernel.launch.params[0].offset, dst.va);
    launch.setPtr(launch.kernel.launch.params[1].offset, src.va);

    // flag = true: the load must still happen, and the bool must still be true
    // afterwards, so the answer is the loaded value itself.
    launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(@as(i32, 9)));
    launch.setU32(launch.kernel.launch.params[3].offset, @bitCast(@as(i32, 4)));
    try launch.run(.{ 1, 1, 1 });
    try testing.expectEqual(@as(i32, 7), dst.read(i32, 0));

    // flag = false: the doubled value.
    dst.slice(i32)[0] = 0;
    launch.setU32(launch.kernel.launch.params[2].offset, @bitCast(@as(i32, 4)));
    launch.setU32(launch.kernel.launch.params[3].offset, @bitCast(@as(i32, 9)));
    try launch.run(.{ 1, 1, 1 });
    try testing.expectEqual(@as(i32, 14), dst.read(i32, 0));
}

test "a compiled pointer add is ONE IMAD.WIDE, and it names no carry predicate" {
    // The structural half of the proof, which needs no GPU.
    //
    // A 64-bit pointer add is now a single `IMAD.WIDE.U32 dst, off, 0x1, base`: the wide
    // multiply-add widens the 32-bit byte offset and adds the 64-bit base in one
    // instruction. It replaced an IADD3 that wrote a carry-out predicate plus an IADD3.X
    // that read it, and the three live tests above prove on silicon that the carry still
    // reaches the high half.
    //
    // THE POINT OF THE OLD TEST SURVIVES, INVERTED. It watched a carry that travelled
    // between two instructions through a predicate register, which is where a carry can be
    // dropped. The hardware never splits this sum, so the check is that NO carry predicate
    // is named at all: no `.X` bit anywhere, and no IADD3 writing a carry-out.
    const allocator = testing.allocator;
    var func = try buildOffsetStore(allocator);
    defer func.deinit();
    var kernel = try isel.compileKernel(allocator, &func, runner_abi);
    defer kernel.deinit(allocator);

    var wide: u32 = 0;
    var extended: u32 = 0;
    var carry_out: u32 = 0;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        const op = kernel.code[i] & 0xfff;
        const hi = kernel.code[i + 2];
        if (op == encode.IMAD_WIDE_IMM_OPCODE) {
            wide += 1;
            // The scale is 1, so the sum is exactly `base + zext(offset)`. See the isel.
            try testing.expectEqual(@as(u32, 1), kernel.code[i + 1]);
            // Unsigned: bit 73 clear, so the offset is zero-extended and not sign-extended.
            try testing.expectEqual(@as(u32, 0), (hi >> (73 - 64)) & 0x1);
            // The destination pair is aligned, as a 64-bit result must be.
            try testing.expectEqual(@as(u32, 0), ((kernel.code[i] >> 16) & 0xff) % 2);
        }
        if (op != 0x210) continue; // IADD3, register form
        if ((hi >> (74 - 64)) & 0x1 == 1) extended += 1;
        if ((hi >> (81 - 64)) & 0x7 != encode.PT) carry_out += 1;
    }

    try testing.expectEqual(@as(u32, 1), wide);
    try testing.expectEqual(@as(u32, 0), extended);
    try testing.expectEqual(@as(u32, 0), carry_out);
}
