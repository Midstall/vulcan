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
        .mul, .mulh => 4,
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
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
        .mul, .mulh => if (elem_float) 1 else 3,
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
    };
}

fn cascadelakeArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul, .mulh => 3, // imul, measured ~2.79
        .div, .rem => 26, // idiv, unmeasured public estimate (doc's probe was defeated)
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn cascadelakeLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| cascadelakeArith(a.op),
        .arith_imm => |a| cascadelakeArith(a.op),
        .load => 5, // L1 ~4.7 corrected
        .convert, .unary => 4,
        .dot => 3, // mul-class
        .matmul => 64, // non-native placeholder
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
    };
}

// Cascade Lake-SP imul is fully pipelined (~1 per cycle on the single multiplier port), and the FP
// mul/fma path is likewise ~1 per cycle. The divider is non-pipelined, so its throughput approaches
// its latency: fp fdiv about 8, integer idiv about 6, rem shares the divider.
fn cascadelakeArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    return switch (op) {
        // imul is fully pipelined (~1/cycle, single multiplier port), and fp mul/fma is also ~1/cycle.
        .mul, .mulh => 1,
        // Divide is non-pipelined: throughput approaches latency. fp fdiv ~8, integer idiv ~6.
        .div, .rem => if (elem_float) 8 else 6,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn cascadelakeThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| cascadelakeArithThroughput(a.op, elem_float),
        .arith_imm => |a| cascadelakeArithThroughput(a.op, elem_float),
        .load => 1,
        .convert, .unary => 1,
        .dot => 1,
        .matmul => 64,
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
    };
}

fn etsocArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul, .mulh => 8,
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
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
        .mul, .mulh => if (elem_float) 1 else 8,
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
    };
}

// River in-order profiles (nano, micro, small, full): microcoded. Seeded from
// riscv64/schedule.zig riverLatency, refine against river_hdl later.
fn riverInorderArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul, .mulh => 3,
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
    };
}

// River macro: dual-issue OoO, functional units overlap so effective latencies are a touch shorter.
//
// The river in-order and macro cost tables are seeded identical and are refined per profile
// against river_hdl later.
fn riverMacroArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul, .mulh => 3,
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
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
        .mul, .mulh => 3, // non-pipelined multiplier: throughput == latency (no evidence of a pipelined FPU)
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
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
        .mul, .mulh => 1, // pipelined multiplier (int and FP): one independent mul per cycle
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
        .iconst, .fconst, .fconst128, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // SM12 T3: no backend expands these yet (Tasks 4a-d), so the microarch optimizer
        // never actually schedules one today - priced like any other cheap bookkeeping op
        // (`iconst`/`store`/...) so a future backend expansion doesn't silently under-model.
        .va_start, .va_arg, .va_end => 1,
        // No CPU backend lowers an atomic read-modify-write: every one of them refuses it,
        // so this model can never price a real one. Priced as cheap bookkeeping beside the
        // `va_list` ops rather than left to a wrong default, exactly like the barrier below.
        .atomic_rmw => 1,
        // No CPU backend lowers a barrier: every one of them refuses it. This model can
        // therefore never price a real one, so it is priced as cheap bookkeeping like the
        // `va_list` ops above rather than left to a wrong default.
        .barrier => 1,
    };
}

// NVIDIA sm_120 (Blackwell consumer, the RTX 5070 class part vulcan's NVIDIA backend targets).
//
// PROVENANCE WARNING, read before trusting a number in this table. Nothing reads this latency table
// today: `microarch.schedule` refuses a SIMT model outright (the NVIDIA backend does its own
// scoreboard scheduling in nvidia/schedule.zig), and `microarch.cost` is reached only from the
// vectorizer, which a SIMT model turns off. `Model` requires the two functions, so they exist and
// they are as accurate as the sources allow, but a pass that starts reading them must re-derive
// them first. The numbers that ARE load-bearing for a SIMT model live in `sm120_simt` below.
//
// Sourcing: NVIDIA publishes no SASS latency table. The fixed-pipe figures are the ones the public
// microbenchmark literature reports for recent NVIDIA parts (Volta and later put a dependent
// integer or FMA ALU result about 4 to 5 cycles after issue). They are NOT measured on sm_120 here,
// and every arm that is a placeholder rather than a sourced figure says so.
fn sm120Arith(op: ir.function.BinOp) u32 {
    return switch (op) {
        // IMAD and the FP multiply both run on the FMA pipe.
        .mul, .mulh => 5,
        // sm_120 HAS NO INTEGER DIVIDE INSTRUCTION. The backend expands one into a reciprocal
        // sequence, so this prices a whole expansion and not an instruction. 64 is an
        // order-of-magnitude placeholder, matching what the CPU models use for an op their target
        // does not have natively. It is not a measured figure.
        .div, .rem => 64,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 4,
    };
}

fn sm120Latency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| sm120Arith(a.op),
        .arith_imm => |a| sm120Arith(a.op),
        // A load is priced at the GLOBAL figure, which is the larger of the two the model carries.
        // `latency` is given only an `Opcode`, and an `Opcode` does not carry the pointer's address
        // space, so this function cannot tell a global load from a shared one. The conservative
        // choice is the global figure. See `sm120_simt.shared_latency` for what is missing.
        .load => sm120_simt.global_latency,
        // An atomic read-modify-write is DECOUPLED on sm_120, the same latency class as a load
        // (NAK's sm120_instr_latencies classes `Op::Atom(_) => DecoupledAgu`, cited in
        // nvidia/schedule.zig), so it is priced the same way and for the same reason.
        .atomic_rmw => sm120_simt.global_latency,
        // I2F, F2I and MUFU are all DECOUPLED on sm_120 (NAK's sm120_instr_latencies, cited in
        // nvidia/schedule.zig), which means there IS no fixed number: the result lands an unknown
        // number of cycles after issue and the hardware needs a scoreboard, not a stall count. 8 is
        // a placeholder that says "more than an ALU op" and nothing more.
        .convert, .unary => 8,
        // ISETP and SEL are ordinary fixed-pipe ALU instructions, unlike the bookkeeping group.
        .icmp, .select => 4,
        // A BAR.SYNC waits for the slowest warp of the workgroup to arrive. That is a property of
        // the program and not of the instruction, so no per-opcode number can be right. 32 is a
        // placeholder, kept above the ALU figures so nothing treats a barrier as free.
        .barrier => 32,
        // Priced like a multiply because a dot is a multiply-accumulate, but the NVIDIA backend
        // lowers no `dot`, so this can never price a real one.
        .dot => 5,
        // The NVIDIA backend lowers no `matmul`: `gpu.tensor.nvidia` declares no dtype at all, so
        // `Model.tensor` refuses the op before it is ever built. Placeholder, as on every other
        // model with no tensor unit.
        .matmul => 64,
        // Cheap bookkeeping, or an op no NVIDIA backend lowers. `call`/`call_indirect` are in this
        // group because the NVIDIA target has NO CALL STACK (nvidia/isel.zig) and refuses both, and
        // `prefetch` because the backend drops it (see `Model.prefetches`).
        .iconst, .fconst, .fconst128, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        // No backend expands these, and a GPU kernel has no C variadic call anyway.
        .va_start, .va_arg, .va_end => 1,
    };
}

// An SM is throughput hardware: each pipe accepts a new INDEPENDENT instruction on its own schedule
// while earlier ones are still in flight. This is exactly the property unrolling exploits, and it is
// why a load prices at 1 here and at hundreds of cycles in `sm120Latency`.
fn sm120ArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    _ = elem_float; // the integer and FP pipes are both pipelined, so the element type does not split
    return switch (op) {
        .mul, .mulh => 1,
        // The divide expansion is many instructions, so a second independent divide cannot start
        // before the first finishes: throughput equals latency, as on every other model.
        .div, .rem => 64,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn sm120Throughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| sm120ArithThroughput(a.op, elem_float),
        .arith_imm => |a| sm120ArithThroughput(a.op, elem_float),
        // The load/store unit accepts a new request while earlier ones are outstanding. THIS is the
        // number the unroll rule is built on: independent loads cost issue slots, not latencies.
        .load, .atomic_rmw => 1,
        .convert, .unary => 1,
        .icmp, .select => 1,
        // A barrier is not pipelined: a second one cannot start before the first releases.
        .barrier => 32,
        .dot => 1,
        .matmul => 64, // no lowering. Non-pipelined placeholder, as in `sm120Latency`
        .iconst, .fconst, .fconst128, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
        .va_start, .va_arg, .va_end => 1,
    };
}

fn sharedArithUnit(op: ir.function.BinOp) UnitClass {
    return switch (op) {
        .mul, .mulh, .div, .rem => .muldiv,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => .alu,
    };
}

// Unit binding, shared across all models (the classes are ISA-neutral).
fn unitOfShared(op: ir.function.Opcode) UnitClass {
    return switch (op) {
        .arith => |a| sharedArithUnit(a.op),
        .arith_imm => |a| sharedArithUnit(a.op),
        .icmp, .select, .iconst, .fconst, .fconst128, .global_addr => .alu,
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
        // SM12 T3: no backend expands these yet. `va_arg` reads through `list` (a memory
        // access, like `load`); `va_start`/`va_end` are pure bookkeeping, like `struct_new`/
        // `extract` above.
        .va_arg => .mem,
        .va_start, .va_end => .none,
        // An atomic is a memory access, so it binds the memory unit like `load`/`store`
        // above. No CPU backend lowers one today, so this is never priced for real, but the
        // memory class is the honest answer if one ever does.
        .atomic_rmw => .mem,
        // A barrier binds no execution unit on any CPU this models, because no CPU backend
        // lowers one. See the latency arms above.
        .barrier => .none,
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
    // add-shift (ADD Xd, Xn, Xm, LSL #imm) is base-ISA on every aarch64 core, so shift_add is
    // always available here, unlike the riscv64 tiers where it needs Zba.
    .fusion = &.{ .{ .kind = .cmp_branch }, .{ .kind = .arith_branch }, .{ .kind = .shift_add } },
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
    // cmp_branch (compare fused into the branch) needs no extension. Shift_add is withheld
    // because et-soc has no Zba.
    .fusion = &.{.{ .kind = .cmp_branch }},
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
        // cmp_branch needs no extension. These embedded tiers have no Zba, so no shift_add.
        .fusion = &.{.{ .kind = .cmp_branch }},
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
    // Zba (sh1/2/3add) is also mandatory in River's RVA22 profile, which is what makes shift_add
    // fusion available on this tier.
    .features = .{ .riscv64 = .{ .m = true, .a = true, .f = true, .d = true, .c = true, .zicbop = true, .zfh = true, .zba = true } },
    .latency = riverInorderLatency,
    .throughput = riverPipelinedThroughput,
    .unitOf = unitOfShared,
    .fusion = &.{ .{ .kind = .cmp_branch }, .{ .kind = .shift_add } },
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
    // See river_f: the application-class FP profile carries Zfh (native f16) and Zba (shift_add) too.
    .features = .{ .riscv64 = .{ .m = true, .a = true, .f = true, .d = true, .c = true, .zicbop = true, .zfh = true, .zba = true } },
    .latency = riverMacroLatency,
    .throughput = riverPipelinedThroughput,
    .unitOf = unitOfShared,
    .fusion = &.{ .{ .kind = .cmp_branch }, .{ .kind = .addr_hi_lo }, .{ .kind = .shift_add } },
};

const cascadelake_fusion = [_]model.FusionRule{
    .{ .kind = .cmp_branch },
    .{ .kind = .arith_branch },
};

// Intel Cascade Lake-SP (Skylake-SP family, stepping 7). Numbers are ASSEMBLY-benchmarked inside a
// Claude-web sandbox (KVM guest, 1 vCPU, host model masked): the corrected core-cycle figures from
// the uArch doc, so they describe the microarchitecture, not a specific Midstall CI part. rob_size,
// mem/fpsimd port counts, and the divide latencies are public Skylake-SP values, not measured here.
// fetch_align and vector_bits are truthful data but NOT yet consumed: there is no x86_64 loop-align
// hook and vectorize.runModel gates x86_64 off, so today only the fusion table (Layer B) and the
// scalar optimize() passes act on this model.
const cascadelake = Model{
    .tag = .@"cascadelake-sp",
    .arch = .x86_64,
    .exec = .out_of_order,
    .issue_width = 4,
    .rob_size = 224,
    .units = .{ .alu = 4, .muldiv = 1, .mem = 2, .branch = 1, .fpsimd = 2 },
    .vector_bits = 512,
    .cache_line = 64,
    .fetch_align = 32,
    .features = .{ .x86_64 = .{
        .avx2 = true,
        .fma = true,
        .avx512f = true,
        .avx512vl = true,
        .avx512dq = true,
        .avx512bw = true,
        .avx512vnni = true,
        .bmi2 = true,
    } },
    .latency = cascadelakeLatency,
    .throughput = cascadelakeThroughput,
    .unitOf = unitOfShared,
    .fusion = &cascadelake_fusion,
};

/// The sm_120 SM description. Every number carries its source.
const sm120_simt = model.Simt{
    // A warp is 32 lanes on every NVIDIA part ever shipped. CUDA C Programming Guide, "Hardware
    // Implementation". It is also what `vulcan-gpu` already assumes for `lane_id` and for the
    // warp-collective tensor fragment layout.
    .warp_size = 32,
    // Maximum resident warps per SM for compute capability 12.0: 48 (equivalently 1536 resident
    // threads). CUDA C Programming Guide, "Technical Specifications per Compute Capability". Ada
    // (8.9) carries the same 48. Hopper and datacenter Blackwell carry 64, so this is a per-part
    // number and not a family constant.
    .warps_per_sm = 48,
    // 64 Ki 32-bit registers per SM. Same table, the "Maximum number of 32-bit registers per SM"
    // row, which has read 64 K for every compute capability since 7.0.
    .regfile_per_sm = 65536,
    // 255 registers per thread. Same table, unchanged since compute capability 3.5. Note that the
    // allocation granularity below means the largest reachable allocation is 248, not 255.
    .max_regs_per_thread = 255,
    // Registers are handed out in multiples of 8 per thread, floor 16. MEASURED on Blackwell
    // silicon and already relied on by the NVIDIA backend: see `regCount` in nvidia/isel.zig, whose
    // comment records that a kernel using R14 silently loses it unless the allocation is rounded
    // this way. Recomputed against that rule by the comptime block below rather than trusted.
    .reg_alloc_granularity = 8,
    .min_regs_per_thread = 16,
    // NOT MEASURED ON THIS PART. NVIDIA publishes no memory latency figure, and this repo has no
    // sm_120 measurement of one. 500 cycles is the round trip to device memory that the public
    // microbenchmark literature reports for recent NVIDIA parts (an L2 hit lands near 200 cycles
    // and a DRAM access near 400 to 600). The unroll rule uses it only through
    // `unroll.coversLatency`, and on this part that clause never fires, which a comptime block in
    // unroll.zig asserts rather than assumes: so an error in this figure cannot change a factor
    // here. It WOULD matter to a part whose warp ceiling is large relative to it.
    .global_latency = 500,
    // NOT MEASURED ON THIS PART either. About 30 cycles is what the same literature reports for a
    // shared-memory load-to-use on Volta and later. It has no consumer today: `Model.latency` is
    // given an `Opcode`, which does not carry the pointer's address space, so nothing can ask for
    // the shared figure rather than the global one. It is recorded because it is the number a
    // shared-memory-aware cost model needs first, and because `validate` uses the two together to
    // check that they are the right way round.
    .shared_latency = 30,
};

comptime {
    // Assertions that RECOMPUTE, not comments that record. Each line below is a published CUDA
    // occupancy figure for compute capability 12.0, re-derived from the four fields above. A typo
    // in the register file size or the warp ceiling fails the build here instead of quietly moving
    // every unroll factor.
    if (sm120_simt.residentWarps(32) != 48) @compileError("32 regs/thread must still reach the full 48 resident warps");
    if (sm120_simt.residentWarps(64) != 32) @compileError("64 regs/thread must give 32 resident warps");
    if (sm120_simt.residentWarps(128) != 16) @compileError("128 regs/thread must give 16 resident warps");
    if (sm120_simt.residentWarps(248) != 8) @compileError("the largest reachable allocation must give 8 resident warps");
    // The granularity is a step, not a rounding detail: asking for one register past a multiple of
    // 8 costs a whole further multiple, and that is what makes occupancy fall in tiers.
    if (sm120_simt.allocFor(65) != 72) @compileError("one register past a multiple of 8 must cost the next multiple of 8");
    if (sm120_simt.allocFor(1) != 16) @compileError("the minimum allocation is 16 registers");
}

// NVIDIA sm_120. A `simt` model: it describes a streaming multiprocessor, so it carries a `Simt`
// block and leaves every CPU-only field at zero. `Model.validate` refuses a simt model that sets
// one, so none of those zeros can later be given a value that only exists to feed a formula.
//
// `unitOf` is `unitOfShared` because the unit classes are ISA-neutral (an SM does have ALU, memory
// and branch pipes), but note that the PORT COUNTS are all zero, so nothing can read a width off
// this model. The classes are descriptive only here.
const sm_120 = Model{
    .tag = .sm_120,
    .arch = .nvidia,
    .exec = .simt,
    .simt = sm120_simt,
    .issue_width = 0,
    .rob_size = 0,
    .units = .{ .alu = 0, .muldiv = 0, .mem = 0, .branch = 0, .fpsimd = 0 },
    .vector_bits = 0,
    // The L1 line is 128 bytes, made of four 32-byte sectors which are the granularity a global
    // access actually moves. CUDA C Programming Guide, "Device Memory Accesses". No pass reads it
    // for this model today: the only consumer is prefetch insertion, and `Model.prefetches` is
    // false for NVIDIA.
    .cache_line = 128,
    // No loop-header alignment hint: the NVIDIA backend emits no padding and has no alignment hook.
    .fetch_align = 0,
    .features = .{ .nvidia = .{} },
    .latency = sm120Latency,
    .throughput = sm120Throughput,
    .unitOf = unitOfShared,
    .fusion = &.{},
};

comptime {
    Model.validate(altra);
    Model.validate(etsoc);
    Model.validate(river_n);
    Model.validate(river_mi);
    Model.validate(river_s);
    Model.validate(river_f);
    Model.validate(river_ma);
    Model.validate(cascadelake);
    Model.validate(sm_120);
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
        .@"cascadelake-sp" => &cascadelake,
        .sm_120 => &sm_120,
    };
}

/// True when the aarch64 MIDR_EL1 names an ARM Neoverse N1 (implementer 0x41, part 0xd0c). Reads the
/// register directly, no I/O. On arm64 Linux the mrs read is emulated for EL0 and does not trap on
/// any supported kernel, so detectHost stays a pure query with no injected dependencies. macOS traps
/// the read as an illegal instruction, which killed the darwin CI as a SIGILL; a macOS host is Apple
/// Silicon and never a Neoverse part, so false there is the right answer and not a fallback.
fn midrPartIsN1() bool {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return false;
    const midr = asm volatile ("mrs %[out], MIDR_EL1"
        : [out] "=r" (-> u64),
    );
    const implementer: u8 = @truncate(midr >> 24);
    const part: u12 = @truncate(midr >> 4);
    return implementer == 0x41 and part == 0xd0c;
}

/// The Cascade Lake-SP discriminator: family 6, model 85 (0x55), and AVX512-VNNI present. VNNI is the
/// clean separator from plain Skylake-SP, which shares family 6 / model 85 but lacks VNNI. Split out
/// from the CPUID read so it is unit-testable without a real CPU.
fn cascadelakeDiscriminator(family: u32, disp_model: u32, vnni: bool) bool {
    return family == 6 and disp_model == 85 and vnni;
}

/// Read CPUID leaf 1 (family/model, with the extended-model combine) and leaf 7 subleaf 0 (ECX bit 11
/// = AVX512-VNNI). x86_64-only, a pure query with no I/O, mirroring midrPartIsN1's shape. The x86
/// named-register asm below is only ever semantically analyzed under `comptime builtin.cpu.arch ==
/// .x86_64`, so this function still compiles cleanly on a non-x86_64 host (e.g. aarch64), where it
/// just returns false without the compiler ever looking at the `"={eax}"`-style constraints.
fn hostIsCascadeLake() bool {
    if (comptime builtin.cpu.arch == .x86_64) {
        // leaf 1 -> eax = version info
        var eax: u32 = undefined;
        var ebx: u32 = undefined;
        var ecx: u32 = undefined;
        var edx: u32 = undefined;
        asm volatile ("cpuid"
            : [a] "={eax}" (eax),
              [b] "={ebx}" (ebx),
              [c] "={ecx}" (ecx),
              [d] "={edx}" (edx),
            : [leaf] "{eax}" (@as(u32, 1)),
              [sub] "{ecx}" (@as(u32, 0)),
        );
        const base_family = (eax >> 8) & 0xF;
        const base_model = (eax >> 4) & 0xF;
        const ext_model = (eax >> 16) & 0xF;
        // Intel display-model combine: when base family == 6 or 15, model = (ext_model << 4) | base_model.
        const disp_model = if (base_family == 6 or base_family == 0xF) (ext_model << 4) | base_model else base_model;
        const disp_family = base_family; // ext_family adds only for base_family == 15, irrelevant to family 6
        // leaf 7 subleaf 0 -> ecx bit 11 = AVX512-VNNI
        asm volatile ("cpuid"
            : [a] "={eax}" (eax),
              [b] "={ebx}" (ebx),
              [c] "={ecx}" (ecx),
              [d] "={edx}" (edx),
            : [leaf] "{eax}" (@as(u32, 7)),
              [sub] "{ecx}" (@as(u32, 0)),
        );
        const vnni = (ecx & (1 << 11)) != 0;
        return cascadelakeDiscriminator(disp_family, disp_model, vnni);
    } else {
        return false;
    }
}

/// Identify the host part, or null when Vulcan does not recognize it. On riscv there is no
/// architectural part register, so this returns null and the caller selects by name.
pub fn detectHost() ?Microarch {
    switch (builtin.cpu.arch) {
        .aarch64 => if (midrPartIsN1()) return .@"ampere-altra",
        .x86_64 => if (hostIsCascadeLake()) return .@"cascadelake-sp",
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

test "cascadelake-sp model carries the measured x86_64 shape and declares cmp/arith fusion" {
    const c = modelFor(.@"cascadelake-sp");
    try std.testing.expectEqual(model.Arch.x86_64, c.arch);
    try std.testing.expectEqual(model.ExecMode.out_of_order, c.exec);
    try std.testing.expectEqual(@as(u8, 4), c.issue_width);
    try std.testing.expectEqual(@as(u16, 512), c.vector_bits);
    try std.testing.expectEqual(@as(u8, 4), c.units.alu);
    // integer imul latency 3 (per-BinOp, integer path)
    try std.testing.expectEqual(@as(u32, 3), c.latency(.{ .arith = .{ .op = .mul, .lhs = undefined, .rhs = undefined } }));
    try std.testing.expect(c.fuses(.cmp_branch));
    try std.testing.expect(c.fuses(.arith_branch));
    try std.testing.expect(!c.fuses(.shift_add));
    try std.testing.expect(c.features.x86_64.avx512vnni);
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

test "sm_120 is a SIMT model: it carries an SM block and leaves every CPU-only field at zero" {
    const g = modelFor(.sm_120);
    try std.testing.expectEqual(model.Arch.nvidia, g.arch);
    try std.testing.expectEqual(model.ExecMode.simt, g.exec);
    // Zero here means NOT APPLICABLE, and `Model.validate` refuses a simt model that sets one of
    // them. These assertions are the runtime half of that refusal.
    try std.testing.expectEqual(@as(u8, 0), g.issue_width);
    try std.testing.expectEqual(@as(u16, 0), g.rob_size);
    try std.testing.expectEqual(@as(u16, 0), g.vector_bits);
    try std.testing.expectEqual(@as(u8, 0), g.units.alu);
    try std.testing.expectEqual(@as(usize, 0), g.fusion.len);
    // So the CPU-shaped queries all answer "no" rather than something derived from a zero.
    try std.testing.expect(!g.superscalar());
    try std.testing.expect(!g.reorders());
    try std.testing.expect(!g.prefetches()); // the NVIDIA backend drops a .prefetch
    try std.testing.expect(!g.vpu());
    try std.testing.expect(g.tensor() == null); // no HMMA or IMMA lowering yet

    const s = g.simt.?;
    try std.testing.expectEqual(@as(u8, 32), s.warp_size);
    try std.testing.expectEqual(@as(u16, 48), s.warps_per_sm);
    try std.testing.expectEqual(@as(u32, 65536), s.regfile_per_sm);
    try std.testing.expectEqual(@as(u16, 255), s.max_regs_per_thread);
    // The published CUDA occupancy tiers for compute capability 12.0, re-derived from those four.
    try std.testing.expectEqual(@as(u32, 48), s.residentWarps(32));
    try std.testing.expectEqual(@as(u32, 32), s.residentWarps(64));
    try std.testing.expectEqual(@as(u32, 16), s.residentWarps(128));
    // A load is priced at the global figure, because `latency` is given only an Opcode and an
    // Opcode does not carry the pointer's address space.
    try std.testing.expectEqual(s.global_latency, g.latency(.{ .load = .{ .ptr = undefined } }));
}

test "every CPU model carries no SM block, and only sm_120 does" {
    // The tie `Model.validate` enforces at compile time, checked once more over the whole registry
    // so a new model cannot quietly arrive with SM numbers nothing reads.
    inline for (std.meta.tags(model.Microarch)) |t| {
        const m = modelFor(t);
        try std.testing.expectEqual(m.exec == .simt, m.simt != null);
        if (t != .sm_120) try std.testing.expect(m.simt == null);
    }
}

test "cascadelakeDiscriminator matches model 85 with VNNI, rejects Skylake-SP (no VNNI) and other models" {
    try std.testing.expect(cascadelakeDiscriminator(6, 85, true)); // Cascade Lake-SP
    try std.testing.expect(!cascadelakeDiscriminator(6, 85, false)); // Skylake-SP: model 85 but no VNNI
    try std.testing.expect(!cascadelakeDiscriminator(6, 94, true)); // different model
    try std.testing.expect(!cascadelakeDiscriminator(15, 85, true)); // different family
}

test "midrPartIsN1 never traps off linux" {
    // On darwin the MRS would trap as a SIGILL, which is what the darwin CI died
    // with. The gate must answer false there without ever reading the register.
    if (builtin.os.tag != .linux) {
        try std.testing.expect(!midrPartIsN1());
    }
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
