//! The ET-SoC-1 kernel launch model: the parameter ABI, the hardware builtins, and the
//! `LaunchInfo` a runtime reads. This is the ET-SoC half of what `nvidia/isel.zig` does for
//! SASS, and it uses the same `vulcan-gpu` contract, so a kernel written once compiles for
//! both.
//!
//! It sits beside `isel.zig` instead of inside it because it is a different concern. `isel`
//! lowers a function body under the ordinary RISC-V calling convention. This file adds only
//! the ENTRY convention: a prologue that reads the parameter block and the hardware
//! identifiers into the argument registers the body already expects. The body compiles
//! through `isel.selectFunctionForModel` unchanged.
//!
//! ET-SoC-1 is MIMD. A minion holds two harts, each hart runs its own instruction stream, and
//! branches never reconverge. So `vulcan-gpu`'s SIMT vocabulary maps as follows, and the
//! mapping is grounded in the vendor SDK rather than in prose:
//!
//! | vulcan-gpu | ET-SoC-1 | source |
//! |---|---|---|
//! | thread | hart | `csrr hartid`, `et-common-libs/include/etsoc/isa/hart.h` |
//! | workgroup | shire, 64 harts | `HARTS_PER_SHIRE`, `et-common-libs/include/system/layout.h` |
//! | shared memory | shire L2 scratchpad | `ETSOC_SCP_GET_SHIRE_ADDR`, same header |
//! | `subgroup_size` | VPU width, 8 lanes | `encode.zig` packed-single and packed-integer ops |
//! | divergence | independent branches | no reconvergence, and none is emitted |
//!
//! The roadmap expected a thread to be a MINION. It is a HART. A minion runs two harts with
//! separate register files and separate program counters, so a thread identifier taken from
//! the minion index would give both harts of a minion the same identifier. `hartid >> 1` is
//! the minion index when a kernel needs it, and the ET-SoC tensor unit needs it: the hardware
//! accepts a tensor operation only from hart 0 of a minion (`require_feature_ml_on_thread0`
//! in the sw-sysemu interpreter), so a `matmul` kernel must run one hart per minion.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const mm = @import("vulcan-opt").microarch;
const encode = @import("encode.zig");
const isel = @import("isel.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Reg = encode.Reg;

pub const Error = std.mem.Allocator.Error || gpu.abi.Error || error{Unsupported};

/// Harts in one shire. A workgroup is a shire, so this is the largest workgroup the hardware
/// can run. `HARTS_PER_SHIRE` in `et-common-libs/include/system/layout.h` and in
/// `etsoc/common/common_defs.h`, and `EMU_THREADS_PER_SHIRE` in the sw-sysemu interpreter.
pub const harts_per_shire: u32 = 64;
/// Harts in one minion. `HARTS_PER_MINION` in `etsoc/common/common_defs.h`, and
/// `EMU_THREADS_PER_MINION` in the sw-sysemu interpreter.
pub const harts_per_minion: u32 = 2;
/// Minions in one neighbourhood. `EMU_MINIONS_PER_NEIGH` in the sw-sysemu interpreter.
pub const minions_per_neigh: u32 = 8;
/// Neighbourhoods in one shire. `NEIGH_PER_SHIRE` in `et-common-libs/include/system/layout.h`.
pub const neigh_per_shire: u32 = 4;
/// Lanes in one VPU operation. The packed-single and packed-integer encoders in `encode.zig`
/// all move 256 bits as 8 lanes of 32 bits, and the et-soc microarchitecture model describes
/// the same eight-wide unit.
pub const vpu_lanes: u32 = 8;

/// The `hartid` CSR, 0xcd0. This is the register the vendor's own `get_hart_id()` reads
/// (`et-common-libs/include/etsoc/isa/hart.h`), so a vulcan kernel and a vendor kernel read
/// the same identifier in the same privilege mode. The machine-mode-only `mhartid` (0xf14)
/// holds the same value, but a kernel does not run in machine mode on a loaded system.
pub const csr_hartid: u12 = 0xcd0;

/// The base of the shire scratchpad window. `ETSOC_SCP_REGION_BASEADDR` in
/// `et-common-libs/include/system/layout.h`.
pub const scratchpad_base: u64 = 0x8000_0000;
/// The address stride between one shire's scratchpad and the next. The vendor's
/// `ETSOC_SCP_GET_SHIRE_ADDR` shifts the shire identifier left by 23 bits.
pub const scratchpad_shire_stride: u64 = 1 << 23;
/// The scratchpad one shire holds, in bytes. `ETSOC_SCP_GET_SHIRE_SIZE` in
/// `et-common-libs/include/system/layout.h` calls this 2.5 MB. The sw-sysemu interpreter
/// backs a larger 4 MiB window per shire (`L2_SCP_SIZE`), so this is the smaller of the two
/// numbers: a kernel that fits here fits on both.
pub const scratchpad_shire_bytes: u32 = 0x28_0000;

/// The address of `offset` inside shire `shire`'s scratchpad. A runtime that binds a kernel's
/// shared memory writes this address into the kernel's shared pointer parameter. ET-SoC has
/// ONE flat address space, so a shared pointer needs no separate load instruction, unlike the
/// NVIDIA LDS window.
pub fn shireScratchpadAddr(shire: u32, offset: u32) u64 {
    return scratchpad_base + @as(u64, shire) * scratchpad_shire_stride + offset;
}

/// The ET-SoC-1 kernel parameter ABI.
///
/// `param_base` is 0 because a kernel takes the address of its parameter block as its ONLY
/// argument, in `a0`. The vendor's own compute kernels have exactly that signature, for
/// example `int entry_point(const MyVectors *)` in
/// `test-compute-kernels/src/add_vector/add_vector.c`, where `MyVectors` is the parameter
/// block. There is no constant bank and no driver-owned area in front of the block, so the
/// first parameter sits at offset 0.
///
/// `param_align` is 1 because the block is an ordinary RISC-V LP64D structure, whose members
/// align to their own size. `gpu.abi.layoutParams` raises each parameter to `max(param_align,
/// size)`, so a floor of 1 reproduces the LP64D layout exactly and any larger floor would
/// place the block differently from the C structure a runtime writes.
///
/// `linear_thread_id` is true because a hart reads ONE `hartid`, not one identifier per axis.
/// The prologue splits it with `gpu.kernel.axisIndex`, and the grid half of that split needs
/// the grid extents, so `layoutParams` reserves the launch-shape region for every workgroup
/// builtin on this target and not for `grid_dim_*` alone.
pub const etsoc_abi: gpu.Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 1,
    .max_shared_bytes = scratchpad_shire_bytes,
    .linear_thread_id = true,
};

/// The register the prologue keeps the parameter block address in. `t0`, a caller-saved
/// temporary, so the compiled body may reuse it freely once the prologue has finished.
const r_block: Reg = .x5;
/// The prologue's first scratch register, `t1`. Holds the hart identifier.
const r_id: Reg = .x6;
/// The prologue's second scratch register, `t2`.
const r_tmp: Reg = .x7;
/// The prologue's third scratch register, `t3`. The grid split needs two live intermediates
/// at once, so `r_tmp` alone is not enough.
const r_tmp2: Reg = .x28;

/// The integer argument registers, a0 through a7. The prologue fills these in declaration
/// order, which is the order `isel.riscv64RegDescription` pins the entry parameters to.
const int_arg_regs = [_]Reg{ .x10, .x11, .x12, .x13, .x14, .x15, .x16, .x17 };
/// The float argument registers a kernel may use. `isel` refuses a seventh float argument in
/// VPU mode, because fa6 and fa7 lie inside the VPU vector partition, so this stops at six.
const float_arg_regs = [_]encode.FReg{ .f10, .f11, .f12, .f13, .f14, .f15 };

/// A compiled ET-SoC kernel: the machine words, and what a runtime needs to launch them.
///
/// The entry point is word 0. Call it as `void kernel(const void *param_block)` with the
/// block address in `a0` and a valid stack pointer, once per hart.
pub const Kernel = struct {
    code: []u32,
    /// How many words of `code` are the entry prologue. The kernel body starts after them.
    /// A test reads this to check the prologue alone.
    prologue_words: u32,
    /// What a runtime needs to launch this kernel. `params` belongs to this Kernel.
    launch: gpu.LaunchInfo,

    pub fn deinit(self: *Kernel, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.launch.params);
    }
};

/// Whether any block returns a value.
fn returnsValue(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |r| if (r.count != 0) return true,
            .jump => {},
        }
    }
    return false;
}

/// Append `csrr r_id, hartid` when the prologue has not read it yet. One read serves every
/// builtin in the prologue, and the value cannot change while the hart runs.
fn appendHartId(allocator: std.mem.Allocator, code: *std.ArrayList(u32), read: *bool) Error!void {
    if (read.*) return;
    try code.append(allocator, encode.csrrs(r_id, csr_hartid, .x0));
    read.* = true;
}

/// Append `li dst, value`. Every immediate this file emits is a workgroup extent, a product
/// of two workgroup extents, or the VPU width. `checkWorkgroupShape` bounds the product of all
/// three extents by `harts_per_shire`, so the twelve-bit form is always enough.
fn appendSmallImm(allocator: std.mem.Allocator, code: *std.ArrayList(u32), dst: Reg, value: u32) Error!void {
    std.debug.assert(value <= harts_per_shire);
    try code.append(allocator, encode.addi(dst, .x0, @intCast(value)));
}

/// Append `lwu dst, grid[axis](r_block)`: one grid extent from the launch-shape region.
///
/// `grid_at` is the displacement of the x extent from the parameter block pointer, with
/// `Abi.param_base` already added, or null when the layout reserved no region. The load is
/// unsigned because an extent is a count, and `divu` and `remu` below are 64-bit operations
/// that a sign-extended word would feed a negative divisor.
fn appendGridExtent(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    dst: Reg,
    grid_at: ?i12,
    axis: u2,
) Error!void {
    const at = grid_at orelse return error.Unsupported;
    const disp: i32 = @as(i32, at) + @as(i32, axis) * @as(i32, @intCast(gpu.LaunchShape.axis_bytes));
    try code.append(allocator, encode.lwu(dst, r_block, @intCast(disp)));
}

/// Emit the thread index on `axis` into `dst`. Clobbers `r_tmp` and reads `r_id`.
///
/// A hart holds ONE linear identifier, so the three axes come out of it by division. The
/// shire-local hart index is
///
///     t = hartid & (harts_per_shire - 1)
///
/// which is the vendor's `get_hart_id() % 64`. The launch contract starts exactly
/// `bx * by * bz` harts in each enabled shire, so `t` is the linear thread index and
/// `0 <= t < bx * by * bz`. The contract linearizes x fastest (`gpu.kernel.linearIndex`), so
///
///     t = (z * by + y) * bx + x
///
/// and `gpu.kernel.axisIndex` inverts it:
///
///     x = t % bx
///     y = (t / bx) % by
///     z = t / (bx * by)
///
/// Three foldings drop an instruction where the bound `t < bx * by * bz` makes it a no-op:
///
///   - `bx == bx * by * bz` means `t < bx`, so `t % bx` is `t` and the mask is the whole
///     lowering. This is the flat one-dimensional workgroup every launch used before.
///   - `by == 1` forces y to 0, and `bz == 1` forces z to 0.
///   - `bx * by == bx * by * bz` means `t / bx < by`, so the outer `% by` is a no-op.
fn emitThreadAxis(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    block: [3]u32,
    dst: Reg,
    axis: u2,
    read_id: *bool,
) Error!void {
    std.debug.assert(dst != r_id and dst != r_tmp and dst != r_block);
    const threads = block[0] * block[1] * block[2];
    const mask: i12 = @intCast(harts_per_shire - 1);
    switch (axis) {
        0 => {
            try appendHartId(allocator, code, read_id);
            if (block[0] == threads) {
                try code.append(allocator, encode.andi(dst, r_id, mask));
                return;
            }
            try code.append(allocator, encode.andi(r_tmp, r_id, mask));
            try appendSmallImm(allocator, code, dst, block[0]);
            try code.append(allocator, encode.remu(dst, r_tmp, dst));
        },
        1 => {
            if (block[1] == 1) {
                try code.append(allocator, encode.addi(dst, .x0, 0));
                return;
            }
            try appendHartId(allocator, code, read_id);
            try code.append(allocator, encode.andi(r_tmp, r_id, mask));
            try appendSmallImm(allocator, code, dst, block[0]);
            try code.append(allocator, encode.divu(dst, r_tmp, dst));
            if (block[0] * block[1] == threads) return;
            try code.append(allocator, encode.addi(r_tmp, dst, 0));
            try appendSmallImm(allocator, code, dst, block[1]);
            try code.append(allocator, encode.remu(dst, r_tmp, dst));
        },
        2 => {
            if (block[2] == 1) {
                try code.append(allocator, encode.addi(dst, .x0, 0));
                return;
            }
            try appendHartId(allocator, code, read_id);
            try code.append(allocator, encode.andi(r_tmp, r_id, mask));
            try appendSmallImm(allocator, code, dst, block[0] * block[1]);
            try code.append(allocator, encode.divu(dst, r_tmp, dst));
        },
        // `Builtin.axis` yields 0, 1 or 2, so a fourth value never reaches this switch.
        3 => unreachable,
    }
}

/// Emit the workgroup index on `axis` into `dst`. Clobbers `r_tmp` and `r_tmp2`, and reads
/// `r_id`.
///
/// A workgroup is a SHIRE, so the linear workgroup index is
///
///     s = hartid >> 6
///
/// which is the vendor's `get_shire_id()`. The runtime enables exactly `gx * gy * gz` shires,
/// so `0 <= s < gx * gy * gz` and `gpu.kernel.axisIndex` inverts the same linearization the
/// thread split uses:
///
///     x = s % gx
///     y = (s / gx) % gy
///     z = s / (gx * gy)
///
/// No folding is possible here. The grid extents are LAUNCH-time values read from the
/// launch-shape region, so nothing at compile time proves `gy` or `gz` is 1. In particular
/// `x` keeps its `% gx`: a two-dimensional grid numbers shire `gx` as workgroup `(0, 1)`, and
/// dropping the remainder would report `(gx, 1)` instead.
fn emitBlockAxis(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    dst: Reg,
    axis: u2,
    grid_at: ?i12,
    read_id: *bool,
) Error!void {
    std.debug.assert(dst != r_id and dst != r_tmp and dst != r_tmp2 and dst != r_block);
    try appendHartId(allocator, code, read_id);
    const shire_shift: u6 = @intCast(std.math.log2_int(u32, harts_per_shire));
    try code.append(allocator, encode.srli(r_tmp, r_id, shire_shift));
    switch (axis) {
        0 => {
            try appendGridExtent(allocator, code, dst, grid_at, 0);
            try code.append(allocator, encode.remu(dst, r_tmp, dst));
        },
        1 => {
            try appendGridExtent(allocator, code, dst, grid_at, 0);
            try code.append(allocator, encode.divu(r_tmp, r_tmp, dst));
            try appendGridExtent(allocator, code, dst, grid_at, 1);
            try code.append(allocator, encode.remu(dst, r_tmp, dst));
        },
        2 => {
            try appendGridExtent(allocator, code, dst, grid_at, 0);
            try appendGridExtent(allocator, code, r_tmp2, grid_at, 1);
            try code.append(allocator, encode.mul(dst, dst, r_tmp2));
            try code.append(allocator, encode.divu(dst, r_tmp, dst));
        },
        // `Builtin.axis` yields 0, 1 or 2, so a fourth value never reaches this switch.
        3 => unreachable,
    }
}

/// Emit the fused global index on `axis` into `dst`:
///
///     global_id[axis] = block_id[axis] * block_dim[axis] + thread_id[axis]
///
/// This is the same fusion the NVIDIA backend emits and the same one the host loop nest
/// computes, with the declared workgroup extent as an immediate. The workgroup index lands in
/// `dst` first, because `emitBlockAxis` needs both scratch registers; the thread index then
/// lands in `r_tmp2`, which `emitThreadAxis` does not touch.
fn emitGlobalAxis(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    block: [3]u32,
    dst: Reg,
    axis: u2,
    grid_at: ?i12,
    read_id: *bool,
) Error!void {
    try emitBlockAxis(allocator, code, dst, axis, grid_at, read_id);
    try appendSmallImm(allocator, code, r_tmp2, block[axis]);
    try code.append(allocator, encode.mul(dst, dst, r_tmp2));
    try emitThreadAxis(allocator, code, block, r_tmp2, axis, read_id);
    try code.append(allocator, encode.add(dst, dst, r_tmp2));
}

/// Emit the hardware read for one compute builtin into `dst`.
///
/// What lowers, and why:
///
///   - `thread_id_*` splits the shire-local hart index with the DECLARED workgroup extents.
///     See `emitThreadAxis` for the arithmetic.
///   - `block_id_*` splits the shire identifier with the grid extents the runtime wrote into
///     the launch-shape region. See `emitBlockAxis`.
///   - `global_id_*` fuses the two. See `emitGlobalAxis`.
///   - `block_dim_*` is the workgroup size the kernel DECLARES. `LaunchInfo.block` carries the
///     same number to the runtime, so the kernel and the launch agree by construction.
///   - `grid_dim_*` is one unsigned word from the launch-shape region, which is the only place
///     a launch-time value can come from on hardware that has no register for it.
///   - `subgroup_size` is the VPU width, 8. The VPU moves 256 bits as eight 32-bit lanes in
///     every packed encoder in `encode.zig`.
///
/// What refuses, and why. A refusal is deliberate. A builtin that reads the wrong number
/// produces a kernel that looks right and indexes the wrong element, which no amount of
/// structural checking finds.
fn emitBuiltin(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    func: *const Function,
    dst: Reg,
    b: gpu.Builtin,
    read_id: *bool,
    grid_at: ?i12,
) Error!void {
    const block = gpu.attrs.localSize(func);
    switch (b) {
        .thread_id_x, .thread_id_y, .thread_id_z => {
            try emitThreadAxis(allocator, code, block, dst, b.axis().?, read_id);
        },
        .block_id_x, .block_id_y, .block_id_z => {
            try emitBlockAxis(allocator, code, dst, b.axis().?, grid_at, read_id);
        },
        .block_dim_x, .block_dim_y, .block_dim_z => {
            try appendSmallImm(allocator, code, dst, block[b.axis().?]);
        },
        .grid_dim_x, .grid_dim_y, .grid_dim_z => {
            try appendGridExtent(allocator, code, dst, grid_at, b.axis().?);
        },
        .global_id_x, .global_id_y, .global_id_z => {
            try emitGlobalAxis(allocator, code, block, dst, b.axis().?, grid_at, read_id);
        },
        // The VPU is not a set of independently sequenced lanes. One hart issues one packed
        // instruction that moves all eight lanes, so no lane has a program counter and there
        // is no hardware lane index to read. A kernel that wants per-lane work uses the packed
        // operations, not a lane identifier.
        .lane_id => return error.Unsupported,
        // A shire is not divided into subgroups by the hardware. Every hart in the shire runs
        // its own stream, so a warp index would be a software partition this file invents.
        .warp_id => return error.Unsupported,
        .subgroup_size => try appendSmallImm(allocator, code, dst, vpu_lanes),
        // A graphics builtin has no meaning in a compute kernel, and ET-SoC has no raster.
        .vertex_index, .instance_index, .frag_coord, .point_coord, .front_facing => return error.Unsupported,
    }
}

/// Append the load of one explicit parameter from the block into its argument register.
fn emitParamLoad(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    func: *const Function,
    p: Value,
    at: i12,
    int_idx: *usize,
    float_idx: *usize,
) Error!void {
    switch (func.types.type_kind(func.valueType(p))) {
        .ptr => {
            if (int_idx.* >= int_arg_regs.len) return error.Unsupported;
            try code.append(allocator, encode.ld(int_arg_regs[int_idx.*], r_block, at));
            int_idx.* += 1;
        },
        .bool => {
            if (int_idx.* >= int_arg_regs.len) return error.Unsupported;
            try code.append(allocator, encode.lbu(int_arg_regs[int_idx.*], r_block, at));
            int_idx.* += 1;
        },
        // The RISC-V psABI widens a narrow scalar to its own signedness up to 32 bits, then
        // sign-extends that to 64. So a 32-bit parameter uses `lw` whatever its signedness,
        // and only 8-bit and 16-bit parameters split on it.
        .int => |i| {
            if (int_idx.* >= int_arg_regs.len) return error.Unsupported;
            const dst = int_arg_regs[int_idx.*];
            const signed = i.signedness == .signed;
            try code.append(allocator, switch (i.bits) {
                8 => if (signed) encode.lb(dst, r_block, at) else encode.lbu(dst, r_block, at),
                16 => if (signed) encode.lh(dst, r_block, at) else encode.lhu(dst, r_block, at),
                32 => encode.lw(dst, r_block, at),
                64 => encode.ld(dst, r_block, at),
                else => return error.Unsupported,
            });
            int_idx.* += 1;
        },
        // f16 arrives in an integer register under this backend's software emulation and
        // f128 in an even-aligned integer PAIR. Neither has been executed through this entry
        // convention, so both refuse rather than guess.
        .float => |f| {
            if (float_idx.* >= float_arg_regs.len) return error.Unsupported;
            const dst = float_arg_regs[float_idx.*];
            try code.append(allocator, switch (f) {
                .f32 => encode.flw(dst, r_block, at),
                .f64 => encode.fld(dst, r_block, at),
                .f16, .f128 => return error.Unsupported,
            });
            float_idx.* += 1;
        },
        // `layoutParams` has already refused these, so this arm is unreachable in practice.
        // It stays explicit because an exhaustive switch is what makes a new type kind a
        // compile error here.
        .vector, .@"struct", .array, .slice => return error.Unsupported,
    }
}

/// Refuse a workgroup the hardware cannot run. A workgroup is a shire and a shire holds
/// `harts_per_shire` harts, so a declared workgroup larger than that has no launch.
fn checkWorkgroupShape(func: *const Function) Error!void {
    const block = gpu.attrs.localSize(func);
    var threads: u64 = 1;
    for (block) |n| {
        if (n == 0) return error.Unsupported;
        threads *= n;
    }
    if (threads > harts_per_shire) return error.Unsupported;
}

/// Lower `func` to an ET-SoC-1 compute kernel under the parameter ABI `a`. The caller owns the
/// result and releases it with `Kernel.deinit`.
///
/// The emitted entry point is `void kernel(const void *param_block)`. The prologue moves the
/// block address out of `a0`, then fills every entry-block parameter in declaration order: a
/// builtin from the hardware, and every other parameter from the block at its placed offset.
/// The prologue then FALLS THROUGH into the body, which `isel` compiled under the ordinary
/// calling convention, so the body's own `ret` returns to the caller of the kernel.
pub fn compileKernel(allocator: std.mem.Allocator, func: *const Function, a: gpu.Abi) Error!Kernel {
    // The ET-SoC entry convention has one argument and no result register a runtime reads, so
    // a kernel returns its answer through a pointer parameter. A value-returning kernel would
    // need an implicit output pointer and an epilogue that stores through it, which this entry
    // convention does not define.
    if (returnsValue(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;
    // The prologue walks the entry parameters positionally against the placed layout, so the
    // two must see the same list. `mem2reg` adds and removes block parameters on any block
    // that has a predecessor, so an entry a branch reaches has no stable list to walk.
    if (gpu.abi.entryIsBranchTarget(func)) return error.Unsupported;
    try checkWorkgroupShape(func);

    var layout = try gpu.layoutParams(allocator, func, a, false);
    errdefer layout.deinit(allocator);

    // The prologue addresses the launch-shape region with the same signed twelve-bit
    // displacement it addresses a parameter with, so the LAST extent has to stay in range.
    const grid_at: ?i12 = if (layout.launch_shape) |shape| at: {
        const last = a.param_base + shape.axisOffset(2);
        if (last > std.math.maxInt(i12)) return error.Unsupported;
        break :at @intCast(a.param_base + shape.offset);
    } else null;

    var code: std.ArrayList(u32) = .empty;
    defer code.deinit(allocator);

    // `a0` holds the block address on entry and is also the first parameter's destination, so
    // the address moves to a scratch register before anything overwrites it.
    try code.append(allocator, encode.addi(r_block, int_arg_regs[0], 0));

    var int_idx: usize = 0;
    var float_idx: usize = 0;
    var placed: usize = 0;
    var read_id = false;
    for (func.blockParams(@enumFromInt(0))) |p| {
        if (gpu.attrs.builtinOf(func, p)) |b| {
            // A builtin is an integer the hardware supplies, so it takes an integer argument
            // register. A frontend that typed one as a float has tagged the wrong parameter.
            if (func.types.type_kind(func.valueType(p)) != .int) return error.Unsupported;
            if (int_idx >= int_arg_regs.len) return error.Unsupported;
            try emitBuiltin(allocator, &code, func, int_arg_regs[int_idx], b, &read_id, grid_at);
            int_idx += 1;
            continue;
        }
        const slot = layout.params[placed];
        placed += 1;
        const off = a.param_base + slot.offset;
        // The prologue addresses the block with one signed twelve-bit displacement, so a block
        // larger than that would need an address computation this prologue does not emit.
        if (off > std.math.maxInt(i12)) return error.Unsupported;
        try emitParamLoad(allocator, &code, func, p, @intCast(off), &int_idx, &float_idx);
    }

    const prologue_words: u32 = @intCast(code.items.len);
    const body = try isel.selectFunctionForModel(allocator, func, mm.modelFor(.@"et-soc"));
    defer allocator.free(body);
    try code.appendSlice(allocator, body);

    const words = try code.toOwnedSlice(allocator);
    return .{
        .code = words,
        .prologue_words = prologue_words,
        .launch = .{
            .params = layout.params,
            .param_bytes = layout.bytes,
            .launch_shape = layout.launch_shape,
            .block = gpu.attrs.localSize(func),
            .shared_bytes = gpu.attrs.sharedBytes(func),
            // ET-SoC has no launch descriptor that declares a register budget. A hart owns its
            // whole register file for the life of the kernel.
            .reg_count = 0,
            // ET-SoC has no hardware barrier the launch descriptor counts. Harts of a shire
            // synchronize through memory.
            .barrier_count = 0,
        },
    };
}

// --- tests ---

const testing = std.testing;

/// Build `void k(u32 *out)` with `out[0] = builtin`, tagged with `b`. The smallest kernel that
/// exercises one builtin end to end.
fn buildOneBuiltinKernel(func: *Function, b: gpu.Builtin) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const v = try func.appendBlockParam(blk, i32_t);
    try gpu.attrs.setBuiltin(func, v, b);
    try func.appendStore(blk, v, out);
    func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });
}

test "the et-soc abi places the block at offset zero with LP64D alignment" {
    try testing.expectEqual(@as(u32, 0), etsoc_abi.param_base);
    try testing.expectEqual(@as(u8, 8), etsoc_abi.pointer_bytes);
    try testing.expectEqual(@as(u8, 1), etsoc_abi.param_align);
    try testing.expectEqual(@as(u32, 0x28_0000), etsoc_abi.max_shared_bytes);
}

test "the topology constants agree with each other" {
    // A shire is four neighbourhoods of eight minions of two harts. If any one of these ever
    // changes, the hart-identifier arithmetic in `emitBuiltin` changes with it.
    try testing.expectEqual(harts_per_shire, harts_per_minion * minions_per_neigh * neigh_per_shire);
    // `thread_id_x` masks with `harts_per_shire - 1`, and `block_id_x` shifts by the same
    // number of bits, so the count must be a power of two.
    try testing.expectEqual(@as(u32, 1), @popCount(harts_per_shire));
}

test "a shire scratchpad address decodes to the vendor formula" {
    try testing.expectEqual(@as(u64, 0x8000_0000), shireScratchpadAddr(0, 0));
    try testing.expectEqual(@as(u64, 0x8080_0000), shireScratchpadAddr(1, 0));
    try testing.expectEqual(@as(u64, 0x9000_0040), shireScratchpadAddr(32, 0x40));
}

test "compileKernel reports a launch a runtime can use" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .global_id_x);
    try gpu.attrs.setLocalSize(&func, .{ 64, 1, 1 });
    try gpu.attrs.setSharedBytes(&func, 1024);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    // One explicit parameter, the output pointer, eight bytes at offset 0.
    try testing.expectEqual(@as(usize, 1), kernel.launch.params.len);
    try testing.expectEqual(@as(u32, 0), kernel.launch.params[0].offset);
    try testing.expectEqual(@as(u32, 8), kernel.launch.params[0].size);
    try testing.expectEqual(gpu.AddressSpace.global, kernel.launch.params[0].kind.pointer);
    // `global_id_x` splits the shire identifier, so the launch-shape region follows the
    // pointer and the runtime writes the grid there.
    try testing.expectEqual(@as(u32, 8), kernel.launch.launch_shape.?.offset);
    try testing.expectEqual(@as(u32, 20), kernel.launch.param_bytes);
    try testing.expectEqual([3]u32{ 64, 1, 1 }, kernel.launch.block);
    try testing.expectEqual(@as(u32, 1024), kernel.launch.shared_bytes);
    // ET-SoC declares neither a register budget nor a barrier count.
    try testing.expectEqual(@as(u32, 0), kernel.launch.reg_count);
    try testing.expectEqual(@as(u32, 0), kernel.launch.barrier_count);
}

test "the prologue moves the block pointer out of a0 before it loads a0" {
    // Suspicious case: the first parameter's destination is the register the block address
    // arrives in, so a prologue that loaded before it copied would read a garbage address.
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .subgroup_size);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    try testing.expectEqual(encode.addi(r_block, .x10, 0), kernel.code[0]);
    try testing.expectEqual(encode.ld(.x10, r_block, 0), kernel.code[1]);
}

test "subgroup_size lowers to the VPU width" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .subgroup_size);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 3), kernel.prologue_words);
    try testing.expectEqual(encode.addi(.x11, .x0, 8), kernel.code[2]);
}

test "thread_id_x masks the hart identifier to its shire" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .thread_id_x);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    try testing.expectEqual(encode.csrrs(r_id, csr_hartid, .x0), kernel.code[2]);
    try testing.expectEqual(encode.andi(.x11, r_id, 63), kernel.code[3]);
}

test "block_id_x takes the shire identifier modulo the grid width" {
    // The shire identifier is the LINEAR workgroup index. On a grid wider than one row the x
    // axis is that index modulo the grid width, so the remainder is load-bearing and not an
    // optimization the backend may drop.
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .block_id_x);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    const grid_at: i12 = @intCast(kernel.launch.launch_shape.?.offset);
    try testing.expectEqual(encode.csrrs(r_id, csr_hartid, .x0), kernel.code[2]);
    try testing.expectEqual(encode.srli(r_tmp, r_id, 6), kernel.code[3]);
    try testing.expectEqual(encode.lwu(.x11, r_block, grid_at), kernel.code[4]);
    try testing.expectEqual(encode.remu(.x11, r_tmp, .x11), kernel.code[5]);
}

test "block_id_y divides by the grid width and then wraps on the grid height" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .block_id_y);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    const shape = kernel.launch.launch_shape.?;
    const gx: i12 = @intCast(shape.axisOffset(0));
    const gy: i12 = @intCast(shape.axisOffset(1));
    try testing.expectEqual(encode.srli(r_tmp, r_id, 6), kernel.code[3]);
    try testing.expectEqual(encode.lwu(.x11, r_block, gx), kernel.code[4]);
    try testing.expectEqual(encode.divu(r_tmp, r_tmp, .x11), kernel.code[5]);
    try testing.expectEqual(encode.lwu(.x11, r_block, gy), kernel.code[6]);
    try testing.expectEqual(encode.remu(.x11, r_tmp, .x11), kernel.code[7]);
}

test "block_id_z divides by the area of one grid layer" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .block_id_z);

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    const shape = kernel.launch.launch_shape.?;
    const gx: i12 = @intCast(shape.axisOffset(0));
    const gy: i12 = @intCast(shape.axisOffset(1));
    try testing.expectEqual(encode.srli(r_tmp, r_id, 6), kernel.code[3]);
    try testing.expectEqual(encode.lwu(.x11, r_block, gx), kernel.code[4]);
    try testing.expectEqual(encode.lwu(r_tmp2, r_block, gy), kernel.code[5]);
    try testing.expectEqual(encode.mul(.x11, .x11, r_tmp2), kernel.code[6]);
    try testing.expectEqual(encode.divu(.x11, r_tmp, .x11), kernel.code[7]);
}

test "grid_dim reads one unsigned word per axis from the launch-shape region" {
    for ([_]gpu.Builtin{ .grid_dim_x, .grid_dim_y, .grid_dim_z }) |b| {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, b);

        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);

        const at: i12 = @intCast(kernel.launch.launch_shape.?.axisOffset(b.axis().?));
        // One load and nothing else: the grid size needs no hart identifier.
        try testing.expectEqual(@as(u32, 3), kernel.prologue_words);
        try testing.expectEqual(encode.lwu(.x11, r_block, at), kernel.code[2]);
    }
}

test "block_dim_x lowers to the declared workgroup width" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .block_dim_x);
    try gpu.attrs.setLocalSize(&func, .{ 32, 1, 1 });

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    try testing.expectEqual(encode.addi(.x11, .x0, 32), kernel.code[2]);
}

test "one hartid read serves two builtins" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const tid = try func.appendBlockParam(blk, i32_t);
    const bid = try func.appendBlockParam(blk, i32_t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    try gpu.attrs.setBuiltin(&func, bid, .block_id_x);
    const sum = try func.appendInst(blk, i32_t, .{ .arith = .{ .op = .add, .lhs = tid, .rhs = bid } });
    try func.appendStore(blk, sum, out);
    func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    var reads: u32 = 0;
    for (kernel.code[0..kernel.prologue_words]) |w| {
        if (w == encode.csrrs(r_id, csr_hartid, .x0)) reads += 1;
    }
    try testing.expectEqual(@as(u32, 1), reads);
}

test "the subgroup index and the lane index refuse" {
    for ([_]gpu.Builtin{ .lane_id, .warp_id }) |b| {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, b);
        try testing.expectError(error.Unsupported, compileKernel(testing.allocator, &func, etsoc_abi));
    }
}

test "a graphics builtin refuses in a compute kernel" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .vertex_index);
    try testing.expectError(error.Unsupported, compileKernel(testing.allocator, &func, etsoc_abi));
}

test "thread_id_y is zero on a flat workgroup and divides on a tall one" {
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_y);
        try gpu.attrs.setLocalSize(&func, .{ 8, 1, 1 });
        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);
        // A height of one leaves y no other value, so the whole lowering is that constant and
        // the prologue never reads the hart identifier.
        try testing.expectEqual(@as(u32, 3), kernel.prologue_words);
        try testing.expectEqual(encode.addi(.x11, .x0, 0), kernel.code[2]);
    }
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_y);
        try gpu.attrs.setLocalSize(&func, .{ 8, 4, 1 });
        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);
        // `y = (t / 8) % 4`, and `8 * 4` is the whole workgroup, so `t / 8 < 4` and the outer
        // remainder folds away.
        try testing.expectEqual(encode.csrrs(r_id, csr_hartid, .x0), kernel.code[2]);
        try testing.expectEqual(encode.andi(r_tmp, r_id, 63), kernel.code[3]);
        try testing.expectEqual(encode.addi(.x11, .x0, 8), kernel.code[4]);
        try testing.expectEqual(encode.divu(.x11, r_tmp, .x11), kernel.code[5]);
        try testing.expectEqual(@as(u32, 6), kernel.prologue_words);
    }
}

test "a three-dimensional workgroup splits the hart identifier on every axis" {
    // The general case, where no folding applies: `t % 2`, `(t / 2) % 2` and `t / 4`. The
    // outer remainder on y survives here because `2 * 2` is not the whole workgroup.
    const cases = [3]struct { b: gpu.Builtin, words: u32 }{
        .{ .b = .thread_id_x, .words = 6 },
        .{ .b = .thread_id_y, .words = 9 },
        .{ .b = .thread_id_z, .words = 6 },
    };
    for (cases) |case| {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, case.b);
        try gpu.attrs.setLocalSize(&func, .{ 2, 2, 2 });

        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);

        // A thread index needs no grid, so no kernel here reserves the region.
        try testing.expectEqual(@as(?gpu.LaunchShape, null), kernel.launch.launch_shape);
        try testing.expectEqual(case.words, kernel.prologue_words);
        try testing.expectEqual(encode.andi(r_tmp, r_id, 63), kernel.code[3]);
    }
    // x takes the remainder by the width, z divides by the area of one layer.
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_x);
        try gpu.attrs.setLocalSize(&func, .{ 2, 2, 2 });
        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);
        try testing.expectEqual(encode.addi(.x11, .x0, 2), kernel.code[4]);
        try testing.expectEqual(encode.remu(.x11, r_tmp, .x11), kernel.code[5]);
    }
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_z);
        try gpu.attrs.setLocalSize(&func, .{ 2, 2, 2 });
        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);
        try testing.expectEqual(encode.addi(.x11, .x0, 4), kernel.code[4]);
        try testing.expectEqual(encode.divu(.x11, r_tmp, .x11), kernel.code[5]);
    }
}

test "global_id fuses the split workgroup index with the split thread index" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .global_id_y);
    try gpu.attrs.setLocalSize(&func, .{ 4, 4, 1 });

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    const shape = kernel.launch.launch_shape.?;
    const gx: i12 = @intCast(shape.axisOffset(0));
    const gy: i12 = @intCast(shape.axisOffset(1));
    // block_id_y into a1, then the declared height, then the product.
    try testing.expectEqual(encode.csrrs(r_id, csr_hartid, .x0), kernel.code[2]);
    try testing.expectEqual(encode.srli(r_tmp, r_id, 6), kernel.code[3]);
    try testing.expectEqual(encode.lwu(.x11, r_block, gx), kernel.code[4]);
    try testing.expectEqual(encode.divu(r_tmp, r_tmp, .x11), kernel.code[5]);
    try testing.expectEqual(encode.lwu(.x11, r_block, gy), kernel.code[6]);
    try testing.expectEqual(encode.remu(.x11, r_tmp, .x11), kernel.code[7]);
    try testing.expectEqual(encode.addi(r_tmp2, .x0, 4), kernel.code[8]);
    try testing.expectEqual(encode.mul(.x11, .x11, r_tmp2), kernel.code[9]);
    // thread_id_y into the scratch, then the sum. `4 * 4` is the whole workgroup, so the
    // outer remainder folds away here too.
    try testing.expectEqual(encode.andi(r_tmp, r_id, 63), kernel.code[10]);
    try testing.expectEqual(encode.addi(r_tmp2, .x0, 4), kernel.code[11]);
    try testing.expectEqual(encode.divu(r_tmp2, r_tmp, r_tmp2), kernel.code[12]);
    try testing.expectEqual(encode.add(.x11, .x11, r_tmp2), kernel.code[13]);
}

test "the split registers stay apart, so no step overwrites its own input" {
    // Suspicious case: the whole grid split runs in three scratch registers while the answer
    // lands in an argument register that earlier parameters already use. If any two of them
    // aliased, a later step would read a value a earlier step had already replaced.
    try testing.expect(r_block != r_id);
    try testing.expect(r_block != r_tmp and r_block != r_tmp2);
    try testing.expect(r_id != r_tmp and r_id != r_tmp2);
    try testing.expect(r_tmp != r_tmp2);
    for (int_arg_regs) |arg| {
        try testing.expect(arg != r_block and arg != r_id);
        try testing.expect(arg != r_tmp and arg != r_tmp2);
    }
}

test "a workgroup larger than a shire refuses" {
    // Suspicious case: the boundary. 64 harts is a whole shire and launches, 65 cannot.
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_x);
        try gpu.attrs.setLocalSize(&func, .{ 64, 1, 1 });
        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        defer kernel.deinit(testing.allocator);
        try testing.expectEqual([3]u32{ 64, 1, 1 }, kernel.launch.block);
    }
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_x);
        try gpu.attrs.setLocalSize(&func, .{ 65, 1, 1 });
        try testing.expectError(error.Unsupported, compileKernel(testing.allocator, &func, etsoc_abi));
    }
}

test "a shared memory request larger than a shire scratchpad refuses" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    try buildOneBuiltinKernel(&func, .thread_id_x);
    try gpu.attrs.setSharedBytes(&func, scratchpad_shire_bytes + 1);
    try testing.expectError(error.SharedMemoryTooLarge, compileKernel(testing.allocator, &func, etsoc_abi));
}

test "a value-returning kernel refuses" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const blk = try func.appendBlock();
    const v = try func.appendBlockParam(blk, i32_t);
    func.setTerminator(blk, .{ .ret = ir.function.Ret.one(v) });
    try testing.expectError(error.Unsupported, compileKernel(testing.allocator, &func, etsoc_abi));
}

test "a mixed parameter block places a float in its own argument register" {
    var func = Function.init(testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const blk = try func.appendBlock();
    const out = try func.appendBlockParam(blk, ptr_t);
    const n = try func.appendBlockParam(blk, i32_t);
    const s = try func.appendBlockParam(blk, f32_t);
    const c = try func.appendInst(blk, f32_t, .{ .convert = .{ .value = n } });
    const prod = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .mul, .lhs = c, .rhs = s } });
    try func.appendStore(blk, prod, out);
    func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
    defer kernel.deinit(testing.allocator);

    // LP64D layout: the pointer at 0, the i32 at 8, the f32 at 12.
    try testing.expectEqual(@as(u32, 0), kernel.launch.params[0].offset);
    try testing.expectEqual(@as(u32, 8), kernel.launch.params[1].offset);
    try testing.expectEqual(@as(u32, 12), kernel.launch.params[2].offset);
    try testing.expectEqual(@as(u32, 16), kernel.launch.param_bytes);
    // The pointer and the integer take a0 and a1, the float takes fa0.
    try testing.expectEqual(encode.ld(.x10, r_block, 0), kernel.code[1]);
    try testing.expectEqual(encode.lw(.x11, r_block, 8), kernel.code[2]);
    try testing.expectEqual(encode.flw(.f10, r_block, 12), kernel.code[3]);
}

test "a kernel whose entry a branch reaches refuses, and the same shape without the edge compiles" {
    // The parameter contract is positional, and `mem2reg` edits the parameter list of any
    // block that has a predecessor. The negative control is the identical two-block kernel
    // with the edge pointing forward, which still compiles.
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_x);
        const back = try func.appendBlock();
        try func.setJump(back, @enumFromInt(0), &.{});
        try testing.expectError(error.Unsupported, compileKernel(testing.allocator, &func, etsoc_abi));
    }
    {
        var func = Function.init(testing.allocator);
        defer func.deinit();
        try buildOneBuiltinKernel(&func, .thread_id_x);
        const tail = try func.appendBlock();
        try func.setJump(@enumFromInt(0), tail, &.{});
        func.setTerminator(tail, .{ .ret = ir.function.Ret.none() });
        var kernel = try compileKernel(testing.allocator, &func, etsoc_abi);
        kernel.deinit(testing.allocator);
    }
}
