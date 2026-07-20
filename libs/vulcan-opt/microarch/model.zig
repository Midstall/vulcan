//! Microarchitecture model: the target-independent data a microarch-aware pass reads. A `Model`
//! describes one CPU part's execution mode, issue width, functional units, latencies, vector width,
//! cache geometry, and ISA extensions. It is public and user-constructible, so a caller can hand a
//! model for a part Vulcan does not ship. There is no generic model, code with no microarch simply
//! does not run the optimizer.

const std = @import("std");
const ir = @import("vulcan-ir");

/// Which Vulcan backend a model targets.
pub const Arch = enum { aarch64, riscv64, x86_64 };

/// Whether the core reorders instructions in hardware.
pub const ExecMode = enum { in_order, out_of_order };

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
        };
    }

    /// Whether this model's riscv64 backend should lower vectorized f32 arithmetic to the
    /// CORE-ET Erbium packed-single VPU (et-soc's custom 8-lane unit) instead of RVV. False for
    /// every non-riscv64 arch and every riscv64 model without the `vpu` feature bit.
    pub fn vpu(self: *const Model) bool {
        return switch (self.features) {
            .riscv64 => |f| f.vpu,
            .aarch64, .x86_64 => false,
        };
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
            .x86_64 => {},
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

test "cascadelake-sp tag parses and x86_64 Features carries the avx512 gating flags" {
    try std.testing.expectEqual(Microarch.@"cascadelake-sp", Microarch.parse("cascadelake-sp").?);
    const f: Features = .{ .x86_64 = .{ .avx512vnni = true, .fma = true } };
    try std.testing.expect(f.x86_64.avx512vnni and f.x86_64.fma);
    try std.testing.expect(!f.x86_64.avx2); // defaults false
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
