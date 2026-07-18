//! The predefined microarchitecture models Vulcan ships, and host detection. Each `Model` is a
//! comptime constant validated at build time. See the design spec for the provenance of every
//! latency: Ampere (Neoverse N1) measured on an Altra M128, ET-SOC from the core-et Erbium docs, River from the
//! river_hdl implementation.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const model = @import("model.zig");

const Model = model.Model;
const Microarch = model.Microarch;
const UnitClass = model.UnitClass;

// Latency functions, type-agnostic per opcode. Values in issue cycles.
//
// `arith` and `arith_imm` carry distinct payload structs (Arith vs ArithImm), so a shared switch
// capture over both is not possible (the capture types would have to unify). Each latency function
// factors its BinOp table into a helper and calls it from both arms instead.

fn altraArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul => 4,
        .div, .rem => 18,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn altraLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| altraArith(a.op),
        .arith_imm => |a| altraArith(a.op),
        .load => 4,
        .convert, .unary => 3,
        // A dot is a multiply-class op (4-way multiply-accumulate), grouped with `mul`.
        .dot => 4,
        // A matmul (et-soc tensor tile) is a big multicycle op, priced well above a
        // scalar mul/dot: not native to this arch, a placeholder pending real timing.
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// Ampere Altra is a Neoverse N1: a wide out-of-order core with a PER-TYPE multiplier split. Its
// FP/SIMD (NEON) multiplier is fully pipelined, so an independent f32/vector mul issues at well under
// one cycle even though its result takes ~4 cycles (latency): reciprocal throughput 1 for the FP path
// (rounded up from the ~0.5-0.9 the on-host probe measures, see altra_measure_test.zig's
// measureF32MulThroughputCycles). The f32 path is the multiply the SLP cost model prices on ampere
// today, because a NEON core only ever vectorizes f32 groups (no <N x i32> lowering; the integer SLP
// path is gated off). The scalar 64-bit INTEGER multiplier is only PARTIALLY pipelined on this part:
// the on-host probe (measureIntMulThroughputCycles) measures ~3 cycles/mul, so the integer path is
// priced at 3. Weighting an f32 mul by that 3 would leave a register-input f32 mul SLP group wrongly
// profitable (the flagged bug), which is exactly why the price is now type-aware rather than a single
// value. The int-mul weight is not used for ampere's SLP decisions yet, but it is set to the measured
// value both to be accurate and to future-proof a NEON <N x i32> path. Only the integer DIVIDE is
// fully non-pipelined: the iterative divider holds the unit for its whole (~18-cycle) run, so its
// throughput equals its latency for both types; rem shares the divider.
fn altraArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    return switch (op) {
        // Pipelined FP multiplier (1, measured ~0.5-0.9) vs partially-pipelined integer multiplier
        // (3, measured), see the on-host probes in altra_measure_test.zig.
        .mul => if (elem_float) 1 else 3,
        .div, .rem => 18, // non-pipelined iterative divider: throughput == latency for both types
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn altraThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| altraArithThroughput(a.op, elem_float),
        .arith_imm => |a| altraArithThroughput(a.op, elem_float),
        // Pipelined load-to-use: latency 4, but one independent load issues per mem port per cycle.
        .load => 1,
        // Pipelined FP/SIMD converts and unary FP ops.
        .convert, .unary => 1,
        // NEON dotprod is a pipelined multiply-accumulate: one issues per cycle.
        .dot => 1,
        // A matmul is not native here; a placeholder, non-pipelined (== latency).
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

fn etsocArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul => 8,
        .div, .rem => 65,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn etsocLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| etsocArith(a.op),
        .arith_imm => |a| etsocArith(a.op),
        .load => 4,
        .convert, .unary => 7,
        // A dot is a multiply-class op (4-way multiply-accumulate), grouped with `mul`.
        .dot => 8,
        // The et-soc fixed-tile matmul: the real tensor CSR-write sequence (load, wait,
        // fma, wait, store) is many times an arith latency; 64 is a placeholder pending
        // a cycle-accurate model of the CSR protocol (isel lowering is a later task).
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// ET-SOC (CORE-ET Erbium Minion): a single-issue in-order core with a PER-TYPE multiplier split. Its
// integer MulDiv is an ASYNC, MULTICYCLE block: "The MulDiv is a multicycle unit and operates
// asynchronously" and computes a 32/64-bit multiply in 4/8 cycles, held busy the whole time (FE-Intpipe
// Description, 3.4 EX stage / 3.4.2 MulDiv Unit). So a second independent integer mul cannot start
// until the first finishes: reciprocal throughput equals latency (int-mul 8, div/rem 65). The FP path
// is different: FP arithmetic is computed on the VPU's per-lane TXFMA, a FULLY PIPELINED 8-stage
// (F0..F8) unit doing "eight operations per cycle" of FP multiply-add (Minion VPU Specification, 2
// Microarchitecture / 2.1 Pipeline Description), so an independent FP multiply issues every cycle:
// throughput 1. (Only the VPU's ML/tensor macro-ops (dot) are the async multicycle u-sequenced ops,
// and the SLP path never emits those.) The int-mul == latency price is exactly why 8-lane SLP is a
// big win here even for cheap element types, and a bigger one for an i32 mul group.
fn etsocArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    return switch (op) {
        // Async multicycle MulDiv for integers (8 = latency, non-pipelined, FE-Intpipe 3.4.2) vs the
        // pipelined VPU TXFMA for FP (1, Minion VPU Spec 2.1).
        .mul => if (elem_float) 1 else 8,
        .div, .rem => 65, // same async MulDiv block, iterative non-restoring divide (33/65 cyc): both types
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn etsocThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| etsocArithThroughput(a.op, elem_float),
        .arith_imm => |a| etsocArithThroughput(a.op, elem_float),
        // Pipelined load-to-use (latency 4, throughput 1).
        .load => 1,
        // Pipelined VPU converts / unary FP.
        .convert, .unary => 1,
        // The dot macro is an async multicycle MulDiv-class op: throughput == latency (never used by SLP).
        .dot => 8,
        // Matmul is the async CSR-write tensor sequence: non-pipelined, throughput == latency.
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// River in-order profiles (nano, micro, small, full): microcoded. Seeded from
// riscv64/schedule.zig riverLatency, refine against river_hdl later.
fn riverInorderArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul => 3,
        .div, .rem => 6,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn riverInorderLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| riverInorderArith(a.op),
        .arith_imm => |a| riverInorderArith(a.op),
        .load => 2,
        .convert, .unary => 2,
        // A dot is a multiply-class op (4-way multiply-accumulate), grouped with `mul`.
        // River carries no dotprod feature today; this is a placeholder in case one is added.
        .dot => 3,
        // River carries no tensor unit; a placeholder in case one is added.
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// River macro: dual-issue OoO, functional units overlap so effective latencies are a touch shorter.
//
// The river in-order and macro cost tables are seeded identical and are refined per profile
// against river_hdl later.
fn riverMacroArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul => 3,
        .div, .rem => 6,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn riverMacroLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| riverMacroArith(a.op),
        .arith_imm => |a| riverMacroArith(a.op),
        .load => 3,
        .convert, .unary => 2,
        // A dot is a multiply-class op (4-way multiply-accumulate), grouped with `mul`.
        // River carries no dotprod feature today; this is a placeholder in case one is added.
        .dot => 3,
        // River carries no tensor unit; a placeholder in case one is added.
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// River throughput. The embedded in-order tiers (n/mi/s) are simple cores: the multiplier is NOT
// pipelined (a small in-order core reuses one multicycle MulDiv), so mul/div throughput == latency,
// the conservative default the spec asks for when the config does not evidence a pipelined
// multiplier. These tiers carry NO hardware FPU (f = false, fpsimd = 0 in their models), so there is
// no pipelined FP multiplier either: the conservative fp-mul price is the same non-pipelined latency
// as the integer one (elem_float is ignored here). add/logic/shift are single-cycle for both types.
fn riverInorderArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    _ = elem_float; // no FPU on these tiers: FP and integer mul are both the non-pipelined MulDiv
    return switch (op) {
        .mul => 3, // non-pipelined multiplier: throughput == latency (no evidence of a pipelined FPU)
        .div, .rem => 6, // non-pipelined divide: throughput == latency
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn riverInorderThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| riverInorderArithThroughput(a.op, elem_float),
        .arith_imm => |a| riverInorderArithThroughput(a.op, elem_float),
        .load => 1, // pipelined load-to-use (latency 2)
        .convert, .unary => 1,
        .dot => 3, // mul-class placeholder, non-pipelined here
        .matmul => 64, // no tensor unit here; non-pipelined placeholder
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// The wider application-class River profiles (river-rc1.f, river-rc1.ma) carry a PIPELINED
// multiplier, like the Neoverse N1: an independent mul issues every cycle (throughput 1) even though
// its result takes `latency` cycles. This holds for BOTH the integer multiplier and the hardware FPU
// (both f and ma carry the RV64GC F/D extensions with a pipelined FP unit), so the price is 1 for
// either element type; elem_float does not diverge here. Divide stays non-pipelined (throughput ==
// latency 6). Shared by both f (in-order but pipelined-mul) and ma (dual-issue OoO); their load/dot
// latencies differ but a throughput of 1 is <= either, so one function serves both.
fn riverPipelinedArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    _ = elem_float; // pipelined for both the integer multiplier and the FPU on these tiers
    return switch (op) {
        .mul => 1, // pipelined multiplier (int and FP): one independent mul per cycle
        .div, .rem => 6, // non-pipelined divide: throughput == latency
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn riverPipelinedThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| riverPipelinedArithThroughput(a.op, elem_float),
        .arith_imm => |a| riverPipelinedArithThroughput(a.op, elem_float),
        .load => 1,
        .convert, .unary => 1,
        .dot => 1, // pipelined mul-accumulate on the wider profile
        .matmul => 64, // no tensor unit here; non-pipelined placeholder
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

fn sharedArithUnit(op: ir.function.BinOp) UnitClass {
    return switch (op) {
        .mul, .div, .rem => .muldiv,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => .alu,
    };
}

// Unit binding, shared across all models (the classes are ISA-neutral).
fn unitOfShared(op: ir.function.Opcode) UnitClass {
    return switch (op) {
        .arith => |a| sharedArithUnit(a.op),
        .arith_imm => |a| sharedArithUnit(a.op),
        .icmp, .select, .iconst, .fconst, .global_addr => .alu,
        .convert => .fpsimd,
        .unary => |u| switch (u.op) {
            .reinterpret => .alu,
            .sqrt, .ceil, .floor, .trunc, .nearest => .fpsimd,
        },
        .load, .store, .prefetch, .alloca => .mem,
        .@"if" => .branch,
        .call, .call_indirect => .branch,
        .struct_new, .extract => .none,
        // dot runs on the SIMD/vector unit, like the vector-shaped fpsimd ops above.
        .dot => .fpsimd,
        // matmul runs on the tensor/VPU unit, modeled as fpsimd like dot.
        .matmul => .fpsimd,
    };
}

const altra = Model{
    .tag = .@"ampere-altra",
    .arch = .aarch64,
    .exec = .out_of_order,
    .issue_width = 4,
    .rob_size = 128,
    .units = .{ .alu = 3, .muldiv = 1, .mem = 2, .branch = 1, .fpsimd = 2 },
    .vector_bits = 128,
    .cache_line = 64,
    .fetch_align = 32,
    .features = .{ .aarch64 = .{ .neon = true, .dotprod = true, .fp16 = true, .lse = true, .rcpc = true } },
    .latency = altraLatency,
    .throughput = altraThroughput,
    .unitOf = unitOfShared,
    .fusion = &.{ .{ .kind = .cmp_branch }, .{ .kind = .arith_branch } },
};

const etsoc = Model{
    .tag = .@"et-soc",
    .arch = .riscv64,
    .exec = .in_order,
    .issue_width = 1,
    .rob_size = 0,
    .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
    .vector_bits = 256,
    .cache_line = 64,
    .fetch_align = 8,
    .features = .{ .riscv64 = .{ .m = true, .a = false, .f = true, .d = false, .c = true, .vpu = true } },
    .latency = etsocLatency,
    .throughput = etsocThroughput,
    .unitOf = unitOfShared,
    .fusion = &.{},
};

fn riverInorder(comptime tag: Microarch, comptime feats: @FieldType(model.Features, "riscv64")) Model {
    return .{
        .tag = tag,
        .arch = .riscv64,
        .exec = .in_order,
        .issue_width = 1,
        .rob_size = 0,
        .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 0 },
        .vector_bits = 0,
        .cache_line = 64,
        .fetch_align = 4,
        .features = .{ .riscv64 = feats },
        .latency = riverInorderLatency,
        .throughput = riverInorderThroughput,
        .unitOf = unitOfShared,
        .fusion = &.{},
    };
}

// Nano is RV32IC but modeled here as riscv64 with placeholder latencies, pending an xlen field
// and a river_hdl derivation.
const river_n = riverInorder(.@"river-rc1.n", .{ .c = true });
// Micro and small are seeded identical pending the river_hdl re-derivation.
const river_mi = riverInorder(.@"river-rc1.mi", .{ .m = true, .a = true, .c = true });
const river_s = riverInorder(.@"river-rc1.s", .{ .m = true, .a = true, .c = true });
// river_f and river_ma are the RV64GC application-class tiers (full in-order, macro
// dual-issue), the ones River's RVA22/RVA23 profile actually carries Zicbop for
// (river/packages/river/lib/src/profiles.dart: `rvZicbop`, part of kRva22U64Extensions).
// The embedded tiers below (n/mi/s) target a narrower, non-application profile and do not
// carry it.

// river_f (RV64GC) has hardware float, so unlike the other in-order profiles it needs its own
// literal with a nonzero fpsimd port count instead of riverInorder's fpsimd = 0 default.
const river_f = Model{
    .tag = .@"river-rc1.f",
    .arch = .riscv64,
    .exec = .in_order,
    .issue_width = 1,
    .rob_size = 0,
    .units = .{ .alu = 1, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
    .vector_bits = 0,
    .cache_line = 64,
    .fetch_align = 4,
    // Zfh (native f16) rides with the application-class FP profile (per River's RVA23 baseline,
    // where Zfh is mandatory): it flips the riscv64 backend to native half instructions instead of
    // the software f32-widening emulation. The embedded tiers (n/mi/s) have no float at all.
    .features = .{ .riscv64 = .{ .m = true, .a = true, .f = true, .d = true, .c = true, .zicbop = true, .zfh = true } },
    .latency = riverInorderLatency,
    .throughput = riverPipelinedThroughput,
    .unitOf = unitOfShared,
    .fusion = &.{},
};
const river_ma = Model{
    .tag = .@"river-rc1.ma",
    .arch = .riscv64,
    .exec = .out_of_order,
    .issue_width = 2,
    .rob_size = 32,
    .units = .{ .alu = 2, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 1 },
    .vector_bits = 0,
    .cache_line = 64,
    .fetch_align = 8,
    // See river_f: the application-class FP profile carries Zfh (native f16) too.
    .features = .{ .riscv64 = .{ .m = true, .a = true, .f = true, .d = true, .c = true, .zicbop = true, .zfh = true } },
    .latency = riverMacroLatency,
    .throughput = riverPipelinedThroughput,
    .unitOf = unitOfShared,
    .fusion = &.{ .{ .kind = .addr_hi_lo }, .{ .kind = .shift_add } },
};

comptime {
    Model.validate(altra);
    Model.validate(etsoc);
    Model.validate(river_n);
    Model.validate(river_mi);
    Model.validate(river_s);
    Model.validate(river_f);
    Model.validate(river_ma);
}

/// The model for a predefined part. Total, one arm per Microarch.
pub fn modelFor(tag: Microarch) *const Model {
    return switch (tag) {
        .@"ampere-altra" => &altra,
        .@"et-soc" => &etsoc,
        .@"river-rc1.n" => &river_n,
        .@"river-rc1.mi" => &river_mi,
        .@"river-rc1.s" => &river_s,
        .@"river-rc1.f" => &river_f,
        .@"river-rc1.ma" => &river_ma,
    };
}

/// True when the aarch64 MIDR_EL1 names an ARM Neoverse N1 (implementer 0x41, part 0xd0c). Reads the
/// register directly, no I/O. On arm64 Linux the mrs read is emulated for EL0 and does not trap on
/// any supported kernel, so detectHost stays a pure query with no injected dependencies.
fn midrPartIsN1() bool {
    if (builtin.cpu.arch != .aarch64) return false;
    const midr = asm volatile ("mrs %[out], MIDR_EL1"
        : [out] "=r" (-> u64),
    );
    const implementer: u8 = @truncate(midr >> 24);
    const part: u12 = @truncate(midr >> 4);
    return implementer == 0x41 and part == 0xd0c;
}

/// Identify the host part, or null when Vulcan does not recognize it. On riscv there is no
/// architectural part register, so this returns null and the caller selects by name.
pub fn detectHost() ?Microarch {
    switch (builtin.cpu.arch) {
        .aarch64 => if (midrPartIsN1()) return .@"ampere-altra",
        else => {},
    }
    return null;
}

test "modelFor returns a self-consistent model for every Microarch" {
    inline for (std.meta.tags(model.Microarch)) |t| {
        const m = modelFor(t);
        try std.testing.expectEqual(t, m.tag);
    }
}

test "Model.prefetches: ampere and the RV64GC application tiers (river-rc1.f/.ma) are true, the embedded tiers and et-soc are false" {
    try std.testing.expect(modelFor(.@"ampere-altra").prefetches());
    try std.testing.expect(modelFor(.@"river-rc1.f").prefetches());
    try std.testing.expect(modelFor(.@"river-rc1.ma").prefetches());

    try std.testing.expect(!modelFor(.@"river-rc1.n").prefetches());
    try std.testing.expect(!modelFor(.@"river-rc1.mi").prefetches());
    try std.testing.expect(!modelFor(.@"river-rc1.s").prefetches());
    try std.testing.expect(!modelFor(.@"et-soc").prefetches());
}

test "the Ampere and ET-SOC models carry the measured and documented shape" {
    const a = modelFor(.@"ampere-altra");
    try std.testing.expectEqual(model.Arch.aarch64, a.arch);
    try std.testing.expectEqual(model.ExecMode.out_of_order, a.exec);
    try std.testing.expectEqual(@as(u8, 4), a.issue_width);
    try std.testing.expectEqual(@as(u16, 128), a.vector_bits);
    try std.testing.expectEqual(@as(u32, 4), a.latency(.{ .arith = .{ .op = .mul, .lhs = undefined, .rhs = undefined } }));

    const e = modelFor(.@"et-soc");
    try std.testing.expectEqual(model.Arch.riscv64, e.arch);
    try std.testing.expectEqual(model.ExecMode.in_order, e.exec);
    try std.testing.expectEqual(@as(u16, 256), e.vector_bits);
    try std.testing.expectEqual(@as(u32, 8), e.latency(.{ .arith = .{ .op = .mul, .lhs = undefined, .rhs = undefined } }));
    // et-soc's vpu capability is what lets it reach the CORE-ET packed-single unit:
    // vectorize.runModel and the riscv64 backend both gate on this.
    try std.testing.expect(e.vpu());
}

test "detectHost identifies this box when it is a Neoverse N1, else null or a matching-arch tag" {
    const got = detectHost();
    if (got) |t| {
        // Never claim a tag whose arch does not match the host.
        try std.testing.expectEqual(switch (builtin.cpu.arch) {
            .aarch64 => model.Arch.aarch64,
            .riscv64 => model.Arch.riscv64,
            else => modelFor(t).arch,
        }, modelFor(t).arch);
    }
    if (builtin.cpu.arch == .aarch64) {
        if (midrPartIsN1()) try std.testing.expectEqual(Microarch.@"ampere-altra", got.?);
    }
}
