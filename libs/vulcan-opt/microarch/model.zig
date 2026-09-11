//! Microarchitecture model: the target-independent data a microarch-aware pass reads. A `Model`
//! describes one part's execution mode, latencies and ISA extensions. A CPU part adds its issue
//! width, functional units, vector width and cache geometry. A SIMT part (a GPU streaming
//! multiprocessor) has none of those and adds a `Simt` block instead, because what bounds it is the
//! register budget and the resident warp count. It is public and user-constructible, so a caller
//! can hand a model for a part Vulcan does not ship. There is no generic model, code with no
//! microarch simply does not run the optimizer.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");

/// Which Vulcan backend a model targets.
pub const Arch = enum { aarch64, riscv64, x86_64, nvidia };

/// How the target issues instructions.
///
/// `in_order` and `out_of_order` both describe a CPU core: there is one instruction stream, and the
/// only question is whether the hardware reorders it. `simt` describes a streaming multiprocessor,
/// which is a different machine. Many warps share one instruction stream, and the hardware hides a
/// long latency by switching to another resident warp instead of by reordering. A `simt` model
/// therefore carries a `Simt` block and leaves every CPU-only field at zero. `Model.validate`
/// refuses a `simt` model that sets one, so `issue_width` and its neighbours cannot later acquire a
/// value that is true only as an input to one formula.
pub const ExecMode = enum { in_order, out_of_order, simt };

/// The execution resources of ONE streaming multiprocessor, for a `simt` model.
///
/// An SM hides a memory access by switching to another resident warp, so the count of RESIDENT
/// warps is the latency-hiding mechanism. That count is set by the register file: it is a fixed
/// budget and every resident warp takes a share of it, so the more registers a thread holds the
/// fewer warps stay resident. This is why a SIMT model bounds loop unrolling by the register
/// budget and not by an issue width, which an SM does not have.
pub const Simt = struct {
    /// Lanes in one warp. The register file is charged per warp, so a per-thread register count is
    /// multiplied by this to get what one warp costs.
    warp_size: u8,
    /// The most warps that are resident on one SM at once. This is the ceiling on latency hiding:
    /// no register budget makes more warps than this available.
    warps_per_sm: u16,
    /// 32-bit registers in one SM's register file. This is the budget the resident warps divide.
    regfile_per_sm: u32,
    /// The most registers one thread may hold. A kernel that needs more spills to local memory.
    max_regs_per_thread: u16,
    /// Registers are given to a thread in multiples of this. A kernel that asks for one register
    /// more than a multiple pays for the whole next multiple, which is why occupancy falls in steps
    /// and not smoothly.
    reg_alloc_granularity: u8,
    /// The smallest allocation a thread gets, whatever it asks for.
    min_regs_per_thread: u16,
    /// Cycles from issue to result for a load that misses to device memory.
    global_latency: u32,
    /// Cycles from issue to result for a load from the SM's own shared memory.
    shared_latency: u32,

    /// What a thread that asks for `regs` registers actually costs: rounded up to
    /// `reg_alloc_granularity`, and never below `min_regs_per_thread`. The rounding is the whole
    /// reason occupancy is a step function, so a caller must price a thread through this and never
    /// with the raw count.
    pub fn allocFor(self: Simt, regs: u32) u32 {
        const g: u32 = self.reg_alloc_granularity;
        const rounded = (regs + g - 1) / g * g;
        return @max(@as(u32, self.min_regs_per_thread), rounded);
    }

    /// How many warps stay resident on one SM when each thread asks for `regs` registers: the
    /// register file divided by what one warp costs, capped by the hardware's own resident-warp
    /// ceiling. `regs` is the RAW request, rounded through `allocFor` here.
    pub fn residentWarps(self: Simt, regs: u32) u32 {
        const per_warp = self.allocFor(regs) * @as(u32, self.warp_size);
        const by_budget = self.regfile_per_sm / per_warp;
        return @min(@as(u32, self.warps_per_sm), by_budget);
    }
};

/// The functional-unit class an IR op binds to, for port-pressure modeling.
pub const UnitClass = enum { alu, muldiv, mem, branch, fpsimd, none };

/// Simultaneous issue slots per functional-unit class (not pipeline depth).
pub const Units = struct {
    alu: u8 = 1,
    muldiv: u8 = 1,
    mem: u8 = 1,
    branch: u8 = 1,
    fpsimd: u8 = 0,
};

/// ISA extensions a model targets, so codegen can enable them. Keyed by arch.
pub const Features = union(Arch) {
    aarch64: struct { neon: bool = false, dotprod: bool = false, fp16: bool = false, lse: bool = false, rcpc: bool = false },
    riscv64: struct {
        m: bool = false,
        a: bool = false,
        f: bool = false,
        d: bool = false,
        c: bool = false,
        v: bool = false,
        zba: bool = false,
        zbb: bool = false,
        /// CORE-ET Erbium packed-single VPU: a custom 256-bit / 8-lane f32 SIMD unit,
        /// NOT RVV. Set only for et-soc. Gates both `vectorize.runModel` (so the SLP
        /// pass targets 8 lanes) and the riscv64 backend's VPU lowering path.
        vpu: bool = false,
        /// Zicbop: cache-block prefetch hints (`prefetch.r/w/i`). Gates the riscv64 backend's
        /// `.prefetch` lowering (see `isel.zig`'s `ModelCaps.zicbop`) and, transitively,
        /// `Model.prefetches()` below.
        zicbop: bool = false,
        /// Zfh: native IEEE half-precision (f16) instructions (fadd.h/flh/fcvt.s.h/...). Gates the
        /// riscv64 backend's NATIVE f16 lowering (see `isel.zig`'s `ModelCaps.zfh`): when set, an
        /// f16 is held natively in a float register instead of emulated as its f32 widening. When
        /// clear (every model that does not set it), the software emulation is unchanged.
        zfh: bool = false,
    },
    x86_64: struct {
        avx2: bool = false,
        fma: bool = false,
        avx512f: bool = false,
        avx512vl: bool = false,
        avx512dq: bool = false,
        avx512bw: bool = false,
        avx512vnni: bool = false,
        bmi2: bool = false,
    },
    /// NVIDIA ships no feature bit yet. Every capability the NVIDIA backend gates on today is a
    /// property of one instruction encoder and not of the model, and the tensor-core dtype set is
    /// already described by `vulcan-gpu.tensor`. An empty struct says that honestly. A bit belongs
    /// here only when a pass reads it to make a different choice.
    ///
    /// The field ORDER of this union must match `Arch`'s, so `nvidia` is last in both.
    nvidia: struct {},
};

/// An abstract fusible pair category. A backend maps each to its concrete instruction pattern.
pub const FuseKind = enum { cmp_branch, arith_branch, addr_hi_lo, shift_add };
pub const FusionRule = struct { kind: FuseKind };

/// The predefined parts Vulcan ships a model for. The tags are the canonical names, quoted, so the
/// display name is the tag name and std.meta.stringToEnum parses them with no side table.
pub const Microarch = enum {
    @"ampere-altra",
    @"et-soc",
    @"river-rc1.n",
    @"river-rc1.mi",
    @"river-rc1.s",
    @"river-rc1.f",
    @"river-rc1.ma",
    @"cascadelake-sp",
    sm_120,

    pub fn parse(name_: []const u8) ?Microarch {
        return std.meta.stringToEnum(Microarch, name_);
    }
    pub fn name(self: Microarch) []const u8 {
        return @tagName(self);
    }
};

/// A microarchitecture description. Comptime-constructed per part in registry.zig, or hand-built by
/// a caller for a part Vulcan does not ship.
pub const Model = struct {
    tag: Microarch,
    arch: Arch,
    exec: ExecMode,
    /// The SM description, set for a `simt` model and null for every CPU model. `validate` ties it
    /// to `exec` in both directions, so `exec == .simt` and `simt != null` can never disagree.
    simt: ?Simt = null,
    issue_width: u8,
    rob_size: u16,
    units: Units,
    vector_bits: u16,
    cache_line: u16,
    fetch_align: u16,
    features: Features,
    /// Per-IR-opcode issue latency in cycles: the time from an op's issue to its result being
    /// available to a DEPENDENT op. This is what the list scheduler (schedule.zig) uses to hide
    /// latency across a dependency chain, and it is the RIGHT metric there.
    latency: *const fn (op: ir.function.Opcode) u32,
    /// Per-IR-opcode reciprocal throughput in cycles: the cycles between two back-to-back INDEPENDENT
    /// issues of this op on ONE port of its functional unit. This, not latency, is the cost of a set
    /// of independent SLP lanes, which is why the profitability cost model (cost.zig) weights by it
    /// (and divides by the class's port count for cross-port parallelism).
    ///
    /// The `elem_float` flag is the ELEMENT type of the op's SLP group: true routes to the FP unit's
    /// throughput, false to the integer unit's. The same BinOp can have very different reciprocal
    /// throughput per type on one core: on the Neoverse N1 the f32 multiplier is fully pipelined
    /// (fmul throughput ~1, the on-host probe measures it) while the 64-bit integer multiplier is only
    /// PARTIALLY pipelined (~3); on et-soc the integer MulDiv is async/multicycle (throughput ==
    /// latency 8) while the VPU FP multiply-add is pipelined (throughput 1). A model that has no type
    /// split simply ignores the flag. Only mul/div/rem ever diverge here; add/sub/logic/shift and the
    /// non-arith ops are single-issue on both paths.
    ///
    ///   - FULLY-PIPELINED ops = 1: a new independent instance issues every cycle per port even
    ///     though its result is not ready for `latency` cycles (a pipelined multiply, an add, a
    ///     load-to-use, a NEON/VPU lane op).
    ///   - NON-PIPELINED ops = their `latency`: the unit is busy for the whole operation and cannot
    ///     accept a new one until it finishes (integer divide everywhere; an integer multiply on a
    ///     simple in-order core whose MulDiv is an async/multicycle block, e.g. et-soc).
    ///
    /// Invariant, asserted by `validate`: throughput(op, f) <= latency(op) for BOTH values of `f` and
    /// every op. You cannot issue independent instances faster than a non-pipelined op completes
    /// (throughput == latency), and a pipelined op issues at 1, which is <= its (>= 1) latency.
    throughput: *const fn (op: ir.function.Opcode, elem_float: bool) u32,
    /// Which functional-unit class an IR op binds to.
    unitOf: *const fn (op: ir.function.Opcode) UnitClass,
    /// Macro-op fusion rules a backend hook reads, empty when the part has none.
    fusion: []const FusionRule,

    pub fn superscalar(self: *const Model) bool {
        return self.issue_width > 1;
    }
    pub fn reorders(self: *const Model) bool {
        return self.exec == .out_of_order;
    }

    /// Whether Vulcan's prefetch-insertion pass gains anything by targeting this model: true when
    /// the target backend actually lowers the `.prefetch` hint to a real instruction, rather than
    /// dropping it (in which case inserting one is pure overhead: extra IR, extra address
    /// arithmetic, for zero benefit). aarch64 always qualifies (PRFM is in the base ISA). riscv64
    /// qualifies only with the Zicbop extension (`prefetch.r`, see `riscv64/encode.zig`); without
    /// it the riscv64 backend still drops the hint. x86_64 has no backend isel support for it yet.
    pub fn prefetches(self: *const Model) bool {
        return switch (self.arch) {
            .aarch64 => true,
            .riscv64 => self.features.riscv64.zicbop,
            .x86_64 => false,
            // The NVIDIA backend drops a `.prefetch` at emission. There is no CPU-style prefetch
            // instruction on this GPU to lower one to (see nvidia/isel.zig), so inserting one is
            // pure overhead: extra IR and extra address arithmetic for nothing.
            .nvidia => false,
        };
    }

    /// Whether this model's riscv64 backend should lower vectorized f32 arithmetic to the
    /// CORE-ET Erbium packed-single VPU (et-soc's custom 8-lane unit) instead of RVV. False for
    /// every non-riscv64 arch and every riscv64 model without the `vpu` feature bit.
    pub fn vpu(self: *const Model) bool {
        return switch (self.features) {
            .riscv64 => |f| f.vpu,
            .aarch64, .x86_64, .nvidia => false,
        };
    }

    /// What this model's target can do with a `matmul`, or null when it lowers none.
    ///
    /// The et-soc tensor unit comes with the CORE-ET packed-single VPU, and the riscv64 backend
    /// gates its `.matmul` lowering on exactly that feature bit (see `isel.zig`: `if (!vpu) return
    /// error.Unsupported`). So the descriptor follows `vpu()` and the two cannot drift apart. No
    /// other part in the registry has a tensor unit.
    ///
    /// A pass must ask this BEFORE it builds a `matmul`, because the op does not lower everywhere.
    /// The descriptor answers which dtypes, tiles and epilogues the target takes, and it records
    /// the alignment and the register ownership that no query over the IR can decide. See
    /// `vulcan-gpu.tensor`.
    ///
    /// An NVIDIA model gets null, which is the same answer `gpu.tensor.nvidia` gives: that
    /// descriptor lowers no dtype at all (`lowersAny()` is false) because the NVIDIA backend has no
    /// HMMA and no IMMA case yet. Both say "raise no matmul for this target".
    pub fn tensor(self: *const Model) ?*const gpu.tensor.Tensor {
        if (self.vpu()) return &gpu.tensor.et_soc;
        return null;
    }

    /// Whether this model's macro-op fusion table declares `kind`.
    pub fn fuses(self: *const Model, kind: FuseKind) bool {
        for (self.fusion) |r| if (r.kind == kind) return true;
        return false;
    }

    /// Compile-time consistency check. Call from a `comptime` block on every model constant so a
    /// malformed model fails the build, not a device.
    pub fn validate(comptime m: Model) void {
        @setEvalBranchQuota(4000); // the per-BinOp throughput<=latency sweep below grows with the enum
        if (m.exec == .in_order and m.rob_size != 0)
            @compileError("in-order model must have rob_size 0");
        // A SIMT model and its `Simt` block are tied in BOTH directions, so no model can claim an
        // SM without describing one, and no CPU model can carry SM numbers that nothing reads.
        if ((m.exec == .simt) != (m.simt != null))
            @compileError("exec == .simt and a non-null simt block must agree");
        if (m.simt) |s| {
            // An SM has no issue width, no reorder buffer, no vector register width, no
            // functional-unit port table and no macro-op fusion. Zero is NOT APPLICABLE here, and
            // this refusal is what stops one of them being filled in later with a number that is
            // true only because it makes some formula produce a wanted answer.
            if (m.issue_width != 0 or m.rob_size != 0 or m.vector_bits != 0)
                @compileError("simt model must leave issue_width, rob_size and vector_bits 0: an SM has none of them");
            if (m.units.alu != 0 or m.units.muldiv != 0 or m.units.mem != 0 or m.units.branch != 0 or m.units.fpsimd != 0)
                @compileError("simt model must leave every Units port count 0: an SM has no port table");
            if (m.fusion.len != 0)
                @compileError("simt model must declare no macro-op fusion");
            // The occupancy arithmetic divides by all four of these, so none may be zero.
            if (s.warp_size == 0 or s.warps_per_sm == 0 or s.regfile_per_sm == 0 or s.max_regs_per_thread == 0)
                @compileError("simt model needs a nonzero warp_size, warps_per_sm, regfile_per_sm and max_regs_per_thread");
            if (s.reg_alloc_granularity == 0 or @popCount(s.reg_alloc_granularity) != 1)
                @compileError("reg_alloc_granularity must be a power of two");
            if (s.min_regs_per_thread == 0 or s.min_regs_per_thread % s.reg_alloc_granularity != 0)
                @compileError("min_regs_per_thread must be a nonzero multiple of reg_alloc_granularity");
            // The smallest allocation must leave room for the full warp ceiling, or the part could
            // never reach its own stated occupancy and `warps_per_sm` would be fiction.
            if (s.residentWarps(0) != s.warps_per_sm)
                @compileError("a thread at the minimum allocation must reach warps_per_sm: the stated ceiling is unreachable");
            // A shared-memory access is served inside the SM and a global one leaves it, so the
            // shared figure is the smaller of the two on every part that has both.
            if (s.shared_latency >= s.global_latency)
                @compileError("shared_latency must be below global_latency");
        }
        if (m.units.fpsimd == 0 and m.vector_bits != 0)
            @compileError("no fpsimd ports but nonzero vector_bits");
        if (m.fetch_align != 0 and @popCount(m.fetch_align) != 1)
            @compileError("fetch_align must be 0 or a power of two");
        if (std.meta.activeTag(m.features) != m.arch)
            @compileError("features union tag must match arch");
        switch (m.features) {
            .aarch64 => |f| if (m.units.fpsimd == 0 and (f.neon or f.fp16))
                @compileError("aarch64 model claims neon/fp16 but has no fpsimd ports"),
            .riscv64 => |f| {
                if (m.units.fpsimd == 0 and (f.f or f.d or f.v or f.vpu))
                    @compileError("riscv64 model claims f/d/v/vpu but has no fpsimd ports");
                // sh1add/sh2add/sh3add (the shift_add fusion's concrete instructions) are Zba,
                // so a riscv64 model cannot declare the fusion without the extension bit.
                if (m.fuses(.shift_add) and !f.zba)
                    @compileError("riscv64 model declares shift_add fusion but lacks Zba");
            },
            .x86_64, .nvidia => {},
        }
        // The throughput <= latency invariant, checked over the ops the cost model actually weights:
        // every arith BinOp (including the mul/div a model may mark non-pipelined) plus a load, and for
        // BOTH element types (the flag routes mul to the FP or the integer path, which can differ). Other
        // ops (convert, unary, dot) may also have throughput != latency, but the cost model never
        // weights them, so they are not covered here; add them if that changes. Both functions are
        // pure switches on the opcode tag, so the placeholder operand handles are never dereferenced
        // and this evaluates at comptime.
        inline for (comptime std.meta.tags(ir.function.BinOp)) |bop| {
            const oc: ir.function.Opcode = .{ .arith = .{ .op = bop, .lhs = @enumFromInt(0), .rhs = @enumFromInt(0) } };
            inline for (.{ true, false }) |ef| {
                if (m.throughput(oc, ef) > m.latency(oc))
                    @compileError("throughput(op) must be <= latency(op): a model marked op '" ++ @tagName(bop) ++ "' as issuing faster than it completes");
            }
        }
        const load_oc: ir.function.Opcode = .{ .load = .{ .ptr = @enumFromInt(0) } };
        inline for (.{ true, false }) |ef| {
            if (m.throughput(load_oc, ef) > m.latency(load_oc))
                @compileError("throughput(load) must be <= latency(load)");
        }
    }
};

test "Microarch.parse round-trips the canonical dotted names and rejects junk" {
    try std.testing.expectEqual(Microarch.@"river-rc1.ma", Microarch.parse("river-rc1.ma").?);
    try std.testing.expectEqual(Microarch.@"ampere-altra", Microarch.parse("ampere-altra").?);
    try std.testing.expectEqualStrings("et-soc", Microarch.@"et-soc".name());
    try std.testing.expect(Microarch.parse("nonsense") == null);
}

test "Model helpers report width and reordering" {
    const m = Model{
        .tag = .@"ampere-altra",
        .arch = .aarch64,
        .exec = .out_of_order,
        .issue_width = 4,
        .rob_size = 128,
        .units = .{ .alu = 3, .muldiv = 1, .mem = 2, .branch = 1, .fpsimd = 2 },
        .vector_bits = 128,
        .cache_line = 64,
        .fetch_align = 32,
        .features = .{ .aarch64 = .{ .neon = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(m.superscalar());
    try std.testing.expect(m.reorders());

    const s = Model{
        .tag = .@"river-rc1.n",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 0 },
        .vector_bits = 0,
        .cache_line = 64,
        .fetch_align = 4,
        .features = .{ .riscv64 = .{ .c = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(!s.superscalar());
    try std.testing.expect(!s.reorders());
}

test "prefetches is true for aarch64, and for riscv64 only with the zicbop feature bit" {
    const altra_like = Model{
        .tag = .@"ampere-altra",
        .arch = .aarch64,
        .exec = .out_of_order,
        .issue_width = 4,
        .rob_size = 128,
        .units = .{ .alu = 3, .muldiv = 1, .mem = 2, .branch = 1, .fpsimd = 2 },
        .vector_bits = 128,
        .cache_line = 64,
        .fetch_align = 32,
        .features = .{ .aarch64 = .{ .neon = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(altra_like.prefetches());

    const etsoc_like = Model{
        .tag = .@"et-soc",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
        .vector_bits = 256,
        .cache_line = 64,
        .fetch_align = 8,
        .features = .{ .riscv64 = .{ .m = true, .f = true, .c = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(!etsoc_like.prefetches());

    const river_like = Model{
        .tag = .@"river-rc1.n",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 0 },
        .vector_bits = 0,
        .cache_line = 64,
        .fetch_align = 4,
        .features = .{ .riscv64 = .{ .c = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(!river_like.prefetches());

    const river_zicbop_like = Model{
        .tag = .@"river-rc1.f",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
        .vector_bits = 0,
        .cache_line = 64,
        .fetch_align = 4,
        .features = .{ .riscv64 = .{ .m = true, .a = true, .f = true, .d = true, .c = true, .zicbop = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(river_zicbop_like.prefetches());
}

test "Model.vpu is true only for a riscv64 model with the vpu feature bit set" {
    const etsoc_like = Model{
        .tag = .@"et-soc",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
        .vector_bits = 256,
        .cache_line = 64,
        .fetch_align = 8,
        .features = .{ .riscv64 = .{ .m = true, .f = true, .c = true, .vpu = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(etsoc_like.vpu());

    const river_like = Model{
        .tag = .@"river-rc1.n",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 0 },
        .vector_bits = 0,
        .cache_line = 64,
        .fetch_align = 4,
        .features = .{ .riscv64 = .{ .c = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(!river_like.vpu());

    const altra_like = Model{
        .tag = .@"ampere-altra",
        .arch = .aarch64,
        .exec = .out_of_order,
        .issue_width = 4,
        .rob_size = 128,
        .units = .{ .alu = 3, .muldiv = 1, .mem = 2, .branch = 1, .fpsimd = 2 },
        .vector_bits = 128,
        .cache_line = 64,
        .fetch_align = 32,
        .features = .{ .aarch64 = .{ .neon = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(!altra_like.vpu());
}

test "Model.tensor gives the et-soc descriptor to a vpu model and nothing to any other" {
    const etsoc_like = Model{
        .tag = .@"et-soc",
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
        .vector_bits = 256,
        .cache_line = 64,
        .fetch_align = 8,
        .features = .{ .riscv64 = .{ .m = true, .f = true, .c = true, .vpu = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    const caps = etsoc_like.tensor().?;
    // The et-soc descriptor, not some other one: 64-byte operands and a real register clobber set.
    try std.testing.expectEqual(@as(u32, 64), caps.operand_align);
    try std.testing.expect(caps.clobbers.len != 0);
    try std.testing.expect(caps.lowersAny());

    // Drop the vpu bit and the same part has no tensor unit to lower a matmul on.
    var no_vpu = etsoc_like;
    no_vpu.features = .{ .riscv64 = .{ .m = true, .f = true, .c = true } };
    try std.testing.expect(no_vpu.tensor() == null);

    const altra_like = Model{
        .tag = .@"ampere-altra",
        .arch = .aarch64,
        .exec = .out_of_order,
        .issue_width = 4,
        .rob_size = 128,
        .units = .{ .alu = 3, .muldiv = 1, .mem = 2, .branch = 1, .fpsimd = 2 },
        .vector_bits = 128,
        .cache_line = 64,
        .fetch_align = 32,
        .features = .{ .aarch64 = .{ .neon = true } },
        .latency = testLatency,
        .throughput = testThroughput,
        .unitOf = testUnit,
        .fusion = &.{},
    };
    try std.testing.expect(altra_like.tensor() == null);
}

test "cascadelake-sp tag parses and x86_64 Features carries the avx512 gating flags" {
    try std.testing.expectEqual(Microarch.@"cascadelake-sp", Microarch.parse("cascadelake-sp").?);
    const f: Features = .{ .x86_64 = .{ .avx512vnni = true, .fma = true } };
    try std.testing.expect(f.x86_64.avx512vnni and f.x86_64.fma);
    try std.testing.expect(!f.x86_64.avx2); // defaults false
}

/// An SM block with round numbers, for the arithmetic tests below. Not a real part: 64 warps, a
/// 65536-register file, 32 lanes, granularity 8, floor 16.
const test_simt = Simt{
    .warp_size = 32,
    .warps_per_sm = 64,
    .regfile_per_sm = 65536,
    .max_regs_per_thread = 255,
    .reg_alloc_granularity = 8,
    .min_regs_per_thread = 16,
    .global_latency = 500,
    .shared_latency = 30,
};

test "Simt.allocFor rounds a request up to the granularity and never below the minimum" {
    // The rounding is the reason occupancy falls in steps: one register past a multiple costs the
    // whole next multiple.
    try std.testing.expectEqual(@as(u32, 16), test_simt.allocFor(0));
    try std.testing.expectEqual(@as(u32, 16), test_simt.allocFor(1));
    try std.testing.expectEqual(@as(u32, 16), test_simt.allocFor(16));
    try std.testing.expectEqual(@as(u32, 24), test_simt.allocFor(17));
    try std.testing.expectEqual(@as(u32, 32), test_simt.allocFor(32));
    try std.testing.expectEqual(@as(u32, 40), test_simt.allocFor(33));
}

test "Simt.residentWarps divides the register file and is capped by the warp ceiling" {
    // 65536 registers / (32 lanes * regs) warps, capped at 64.
    try std.testing.expectEqual(@as(u32, 64), test_simt.residentWarps(16)); // budget would allow 128
    try std.testing.expectEqual(@as(u32, 64), test_simt.residentWarps(32)); // budget allows exactly 64
    try std.testing.expectEqual(@as(u32, 32), test_simt.residentWarps(64));
    try std.testing.expectEqual(@as(u32, 16), test_simt.residentWarps(128));
    // The step: 64 registers holds 32 warps, and asking for ONE more drops it to 28, because the
    // request rounds up to 72.
    try std.testing.expectEqual(@as(u32, 28), test_simt.residentWarps(65));
}

fn testLatency(op: ir.function.Opcode) u32 {
    _ = op;
    return 1;
}
fn testThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    _ = op;
    _ = elem_float;
    return 1;
}
fn testUnit(op: ir.function.Opcode) UnitClass {
    _ = op;
    return .alu;
}
