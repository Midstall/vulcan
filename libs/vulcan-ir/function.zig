//! Owns an IR function's blocks, instructions, and values as entity references
//! into dense pools off a caller-supplied allocator. SSA values cross block
//! boundaries as block parameters, not phi nodes. The control-flow graph is
//! derived from terminators, never stored separately, so it cannot drift.

const std = @import("std");
const types = @import("types.zig");
const attribute = @import("attribute.zig");

const Type = types.Type;
const TypeTable = types.TypeTable;
pub const Attribute = attribute.Attribute;

/// A handle to a basic block within a function.
pub const Block = enum(u32) { _ };

/// A handle to an SSA value: a block parameter or an instruction result.
pub const Value = enum(u32) { _ };

/// A handle to an instruction within a function.
pub const Inst = enum(u32) { _ };

/// A binary arithmetic/bitwise relation. Signedness for division comes from the
/// operand types, so the relation stays signedness-agnostic.
pub const BinOp = enum {
    add,
    sub,
    mul,
    div,
    rem,
    bit_and,
    bit_or,
    bit_xor,
    shl,
    shr,
    /// High half of the full-width product (`(lhs *widen rhs) >> bits`). Signedness comes from
    /// the operand type, like `div`/`rem`: signed operands take the signed high multiply, unsigned
    /// the unsigned one. The magic-number divide lowering (`strength.zig`) is its only producer.
    mulh,

    /// The symbolic operator this relation prints as.
    pub fn symbol(self: BinOp) []const u8 {
        return switch (self) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .rem => "%",
            .bit_and => "&",
            .bit_or => "|",
            .bit_xor => "^",
            .shl => "<<",
            .shr => ">>",
            .mulh => "*h",
        };
    }
};

/// A binary arithmetic/bitwise operation. The result type is the operand type.
pub const Arith = struct { op: BinOp, lhs: Value, rhs: Value };

/// Arithmetic against a constant operand: `lhs <op> imm`. Lowers to an immediate
/// instruction (addi/andi/.../slli) instead of materializing the constant.
pub const ArithImm = struct { op: BinOp, lhs: Value, imm: i64 };

/// A comparison relation. Signedness comes from the operand types, not the
/// relation, so the relation stays signedness-agnostic.
pub const CmpOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,

    /// The symbolic operator this relation prints as.
    pub fn symbol(self: CmpOp) []const u8 {
        return switch (self) {
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
        };
    }
};

/// An integer comparison: produces a `bool`.
pub const Compare = struct { op: CmpOp, lhs: Value, rhs: Value };

/// A value-producing conditional: `then` when `cond` is true, else `@"else"`.
/// The value form of `if` (`c := if {} else {}`).
pub const Select = struct { cond: Value, then: Value, @"else": Value };

/// Construct an aggregate value from field values. High profile only.
pub const StructNew = struct { fields: ValueList };

/// Extract field `index` from an aggregate value. High profile only.
pub const Extract = struct { aggregate: Value, index: u32 };

/// Convert a value to the instruction's result type (int<->float). The
/// direction is read from the source value's type versus the result type.
pub const Convert = struct { value: Value };

/// A single-operand operation. `reinterpret` reinterprets the bits as the result
/// type (int<->float, same width), the rest are floating-point math on the result
/// type.
pub const UnaryOp = enum { reinterpret, sqrt, ceil, floor, trunc, nearest };

/// A unary operation on `value`, producing the instruction's result type.
pub const Unary = struct { op: UnaryOp, value: Value };

/// Reserve a stack slot sized for `elem` and yield its address (a `ptr`).
pub const Alloca = struct { elem: Type };

/// Call a function named by symbol index, passing `args`. The result type is the
/// instruction's result type. Use a zero-width result for a void call. `is_variadic`
/// and `num_fixed` mark a variadic C call, for example `printf("%d", n)`. `num_fixed`
/// holds the callee's own declared parameter count. `args[0..num_fixed]` are the
/// callee's declared parameters, `args[num_fixed..]` are the variadic arguments (the
/// frontend already applies default argument promotion to them). Both fields default
/// to the non-variadic values, so every existing construction site stays byte-identical
/// (this mirrors the defaulted-bool pattern of `Load.@"volatile"`). IR construction,
/// clone, and print carry these bits through. A later change makes the backend read them.
/// One return-register eightbyte of a struct-by-value `.registers` return. `fp`
/// selects the register bank the eightbyte comes back in (`false` an integer register,
/// `true` a floating-point register). `offset` is the eightbyte's byte offset in the
/// struct, and `bytes` its width (4 or 8). The caller's post-call store reads each
/// return register of its bank and writes `bytes` of it into `[ret_dest + offset]`.
/// An integer eightbyte defaults to `fp=false`, `offset=0`, `bytes=8` (a full-register
/// store), which reproduces the plain integer-return store, so a pure-integer return
/// stays byte-identical. This lives at the IR level, with no dependency on the
/// frontend's `abi` classifier, so a backend can read it without that import.
pub const RetPiece = struct { fp: bool = false, offset: u8 = 0, bytes: u8 = 8 };

/// `ret_dest`/`ret_regs`/`ret_pieces`/`sret` carry a struct-by-value return. A small
/// struct returned in registers (`abi.classify`'s `.registers` plan) sets `ret_dest`
/// to a frontend-allocated destination slot (a frame-relative `alloca` address), and
/// `ret_regs` to how many eightbytes the return occupies (1 to 4). The call itself
/// produces no ordinary scalar result (its `result` is null, like a void call), and
/// the backend stores the return registers into `ret_dest` after the call.
/// `ret_pieces[0..ret_regs]` describe each return register's bank, offset, and width
/// for that store (see `RetPiece`), so a return mixing integer and floating eightbytes
/// lands the right register in the right dest slot. `ret_dest` is therefore an operand
/// (a use) of the call, so every operand-scanning pass counts it and keeps the
/// destination live, and its `alloca` is never dropped. `sret` marks the greater-than-
/// 16-byte hidden-pointer return: `args[0]` is the caller's result-slot address. All
/// fields default to the non-struct-return values, so every existing construction site
/// stays byte-identical (this mirrors the defaulted-field pattern of `is_variadic`).
pub const Call = struct { symbol: u32, args: ValueList, is_variadic: bool = false, num_fixed: u32 = 0, ret_dest: ?Value = null, ret_regs: u8 = 0, ret_pieces: [4]RetPiece = @splat(.{}), sret: bool = false };

/// Call the function whose address is `target` (a `ptr`), passing `args`. The result
/// type is the instruction's result type. `is_variadic`/`num_fixed` mirror `Call`'s own
/// fields, see there. `ret_dest`/`ret_regs`/`ret_pieces`/`sret` also mirror `Call`'s.
pub const CallIndirect = struct { target: Value, args: ValueList, is_variadic: bool = false, num_fixed: u32 = 0, ret_dest: ?Value = null, ret_regs: u8 = 0, ret_pieces: [4]RetPiece = @splat(.{}), sret: bool = false };

/// The address of a named global symbol (a module-level constant or variable),
/// yielding a `ptr`. The symbol is resolved at link time. `via_got` requests
/// GOT-indirect addressing: the code loads the symbol's address from the Global
/// Offset Table (a data import from another shared object) rather than forming
/// it directly. Defaults to `false` (an ordinary direct address), so every
/// existing construction site keeps the byte-identical direct lowering.
pub const GlobalAddr = struct { symbol: u32, via_got: bool = false };

/// `volatile` marks an access to a C `volatile`-qualified lvalue. The optimizer must
/// treat it as having an observable side effect: it must not eliminate it, reorder it
/// across another volatile access, or coalesce it with another load or store, even
/// though a plain `load` is otherwise pure. Defaults to `false` (an ordinary load), so
/// every existing named-field construction site is unaffected.
///
/// The optimizer HONORS this today. Elimination is refused by `loadfwd`, `mem2reg` and
/// `jumpthread`. Coalescing is refused by SLP load and store fusion (`vectorize`), by
/// `loopvec` map and reduction, by `dotprod`, and by `matmul_recog`. Reordering holds
/// structurally, because no pass moves a memory operation at all: if `licm`,
/// `microarch/schedule`, `blocklayout` or `loadfwd` is ever loosened to move a plain
/// load, that pass MUST gain a volatile-versus-volatile ordering check at the same time.
///
/// Two gaps remain, both recorded in the project memory. `microarch/prefetch` emits a
/// hint for a volatile address, which is a question about what `prefetch` may name
/// rather than a violation of the three rules above. `vulcan-spirv/widen`'s sampler
/// gather path would drop the flag, and is unreachable from any current frontend.
pub const Load = struct { ptr: Value, @"volatile": bool = false };

/// A store to memory. Produces no result. `volatile` mirrors `Load.volatile` (see its doc).
pub const Store = struct { value: Value, ptr: Value, @"volatile": bool = false };

/// A software prefetch hint for the address at `ptr`. Produces no result and has
/// no observable effect on the function's result, it only hints the backend to
/// warm the cache ahead of a later access.
pub const Prefetch = struct { ptr: Value };

/// Initialize a `va_list` object for reading a variadic call's extra arguments. This is
/// C11 7.15's `va_start`. `list` is the address (a `ptr`) of the `va_list` object (the
/// frontend's `__builtin_va_start(ap, last)` resolves `ap` to its address the same way any
/// other lvalue does, see `ctype.builtinVaList` for the object's per-target shape). No
/// result. A later backend expansion gives this its actual meaning, recording the first
/// unnamed argument's location. This only carries it through IR construction, clone,
/// print, and verify.
pub const VaStart = struct { list: Value };

/// Fetch the next variadic argument of the instruction's result type from the `va_list`
/// object at `list`. This is C11 7.15's `va_arg`, advancing `list` past it. `list` is a
/// `ptr` (mirrors `VaStart.list`). `ty` restates the result type on the op itself, rather
/// than relying solely on the instruction's own result type the way `Convert` does, since
/// a later backend expansion needs it to pick the right load width or register class
/// without re-deriving it from the surrounding `InstData`. A later backend gives this its
/// actual meaning; this only carries it through IR construction, clone, print, and verify.
pub const VaArg = struct { list: Value, ty: Type };

/// Finalize a `va_list` object at `list`. This is C11 7.15's `va_end`. `list` is a `ptr`
/// (mirrors `VaStart.list`). No result. On every target this compiler lowers, `va_end` is
/// a no-op at run time, since there is no allocated resource to release. It exists so the
/// IR shape mirrors the C source exactly, and a future target that does need cleanup has
/// somewhere to hang it. A later backend gives this its actual meaning; this only carries
/// it through IR construction, clone, print, and verify.
pub const VaEnd = struct { list: Value };

/// An INT8 4-way dot-product accumulate: `result = acc + dot(a, b)`. Pure (no
/// memory effect), like `arith`. `acc` and the result are `<4 x i32>`; `a` and
/// `b` are the same `<16 x i8>` (signed) or `<16 x u8>` (unsigned) type.
pub const Dot = struct { acc: Value, a: Value, b: Value };

/// A fixed-tile matrix multiply: `c := a * b` (or `c += a * b` when `accumulate`),
/// an `m x k` by `k x n` tile written to `c`. `a`, `b`, and `c` are `ptr` values;
/// `a` and `b` are read from memory, `c` is where the result is written. Produces
/// no result. EFFECTFUL (writes memory at `c`).
///
/// PRECONDITIONS ARE PER TARGET and live in `vulcan-gpu.tensor`, one `Tensor` descriptor per
/// target, because they are hardware facts and not properties of this operation. The three real
/// cases do not agree, so no single set can be stated here:
///   - The et-soc tensor unit is cache-line addressed. It needs `a`, `b` and `c` 64-byte aligned,
///     it accepts only the tiles its pass descriptor can encode, and it owns x5, x6, x7, x28, x31
///     and the TenC registers f0 to f(2 * @min(16, m) - 1) for the whole op, so a matmul must not
///     share a function with a live value in one of those registers. `tensor.et_soc` holds each of
///     those facts with the reason for it.
///   - The scalar loop nest `vulcan-ir.expand.expandMatmul` writes has NONE of them. It reads
///     tightly packed row-major memory at any alignment, it takes any tile, and it owns no
///     register. See `tensor.scalar`.
///   - An NVIDIA tensor core is different again: one instruction is warp-collective, and each of
///     the 32 lanes holds a fragment of the tile in its own registers. See `tensor.nvidia`.
///
/// Nothing verifies a precondition. `vulcan-ir` cannot import `vulcan-gpu`, and an IR `ptr` carries
/// no alignment, so alignment stays a contract on whoever builds the op. `Tensor.rejects` answers
/// what a `MatMul` alone can decide: the dtype, the tile and the epilogue. A pass that raises a
/// loop nest to a matmul must ask that query first, and must prove or repack for the alignment the
/// descriptor names. `vulcan-opt.microarch.matmul_recog` does both.
/// Element dtype of a `matmul`'s A/B inputs (its output C is ALWAYS 32-bit: fp32
/// accumulators for `fp32`/`fp16`, int32 accumulators for `int8`/`uint8`). This is the
/// op's own metadata (A/B/C are opaque `ptr` values, so `verify` gains no new type check);
/// it drives the et-soc tensor_fma `type` field and element packing in the backend. The
/// et-soc tensor unit natively supports these three hardware dtypes (tensors.h `TensorType`:
/// fp32=0, fp16->fp32=1, int8->int32=3); `int8` and `uint8` share the int8 hardware type and
/// differ only by the fma `tena`/`tenb_unsigned` bits (uint8 sets them). Encoded as a `u3` in
/// bitcode. Backends other than the et-soc VPU reject `matmul` regardless of dtype.
pub const MatMulType = enum(u3) { fp32, fp16, int8, uint8 };

/// The requantize scale for a `matmul` quant epilogue: either one fp32 scalar broadcast to every
/// output column, or one fp32 scale per output column (per-channel requantization). Both cases are
/// compile-time constant data, never an IR Value operand: `scalar` stores the fp32 bits directly,
/// `per_column` stores a handle into the function's `scale_pool` (see `internScales`/`scaleList`).
pub const MatMulScale = union(enum) {
    scalar: u32, // fp32 scale reinterpreted as u32, broadcast to every output column
    per_column: ScaleList, // n fp32-bit scales (one per output column), interned constant data
};

/// The requantized output element type of a `matmul` quant epilogue: `i8` saturates to signed
/// int8 (`-128..127`), `u8` saturates to unsigned uint8 (`0..255`). Independent of the matmul's
/// own `dtype` (the A/B input element type); the packed-byte store is identical either way, only
/// the saturating transform in the requantize chain differs (see isel.zig's `.matmul` case).
pub const MatMulQuantOut = enum { i8, u8 };

/// int8-requantize epilogue fused into a `matmul`: after the int32 tile is computed, add `bias`
/// (per-column, optional), scale it (`scale`, either a scalar or per-column), optionally relu,
/// saturate to `out` (signed int8 or unsigned uint8), add `zero_point`, and pack. Only valid when
/// `dtype == .int8` (verify rejects it otherwise); when set, the matmul's C output is one byte per
/// element (n bytes per row) rather than 32-bit. `bias` and `zero_point` are compile-time constant
/// data, never IR Value operands, mirroring `scale`'s `per_column` handle: this is what lets
/// asymmetric-uint8 requantization (bias-correct then re-center on a non-zero output zero-point)
/// be expressed without adding a new SSA operand that every operand-use pass would need to scan.
pub const MatMulQuant = struct {
    scale: MatMulScale,
    relu: bool,
    out: MatMulQuantOut = .i8,
    bias: ?BiasList = null, // optional per-column int32 bias (interned), null = no bias
    zero_point: i32 = 0, // per-tensor output zero-point, 0 = symmetric (existing behavior)
};

/// Per-operand signedness override for a `matmul`'s two int8 inputs. When a matmul's
/// `input_signs` is null (the default), signedness comes from `dtype` (int8 -> both signed,
/// uint8 -> both unsigned). When non-null it is AUTHORITATIVE per operand (dtype then only
/// selects the hardware element type), enabling mixed signedness such as uint8 activations
/// times int8 weights. verify requires `dtype == .int8` when this is set, so each config has one
/// spelling: symmetric-signed = int8+null, symmetric-unsigned = uint8+null, mixed = int8 + this.
pub const InputSigns = struct { a_unsigned: bool, b_unsigned: bool };

/// When `embedded` is set, the matmul is NOT the whole reachable function: it sits inside a larger
/// function with values live across it, so the backend must lower it self-contained (save every
/// register it clobbers on entry, restore on exit, and hold a/b/c in dedicated registers so the
/// allocator's placement of them cannot be clobbered). When false (every matmul built today, e.g.
/// the whole-function recognizer output and every standalone kernel) the backend lowers it with the
/// zero-overhead standalone-kernel path, byte-identical to before this field existed. Only the
/// et-soc VPU (riscv64) backend honors `embedded`; no other backend supports matmul at all.
pub const MatMul = struct { a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool, embedded: bool = false, quant: ?MatMulQuant = null, input_signs: ?InputSigns = null };

/// The set of threads a `barrier` synchronizes.
///
/// The scope is part of the operation from its first version on purpose. A scope-less barrier
/// needs a breaking change to every layer the moment a second scope arrives, and the two scopes
/// are not interchangeable: a subgroup barrier synchronizes one warp, a workgroup barrier
/// synchronizes the whole workgroup. A backend that lowers only one of them must refuse the
/// other, and it can only do that if the operation says which one it is.
///
/// Encoded as one byte in bitcode, so the tag values are pinned there.
pub const BarrierScope = enum {
    /// Every thread of the workgroup (the CUDA block, the NVIDIA CTA). This is what a
    /// `__syncthreads()` in a compute kernel means.
    workgroup,
    /// Every thread of the subgroup (the warp, the wave). No backend lowers this yet.
    subgroup,
};

/// An execution and memory barrier over `scope`. Every thread in the scope waits here, and
/// memory a thread wrote before the barrier is visible to the other threads after it. Produces
/// no result. EFFECTFUL, like `store` and `matmul`.
///
/// This is a first-class operation and not a recognised call for one reason: a barrier is a
/// constraint the optimizer must read. `licm`, `gvn`, `dce` and `loadfwd` must not move a load
/// or a store across it, must not common two of them, and must not delete one. A call is opaque
/// to those passes only by accident of how side effects are modelled, so a call that a later
/// pass learns to treat as pure becomes a SILENTLY DELETED barrier. A call also has no callee
/// here, which makes it a lie in the IR.
///
/// It carries no memory-order operand. Each scope that lowers today is a full fence in the
/// machine code, so a weaker order has no spelling to lower to.
pub const Barrier = struct { scope: BarrierScope };

/// The read-modify-write an `atomic_rmw` applies to the value already in memory.
///
/// Each name maps onto one NVIDIA `encode.AtomOp` with no gaps, so a backend needs no
/// table of exceptions. `inc` and `dec` (the wrapping counters) are deliberately absent:
/// no frontend spells them, and an operation the IR cannot build is an operation no
/// backend must lower.
///
/// Signedness and width are NOT part of this enum. They come from the value operand's
/// type, which is the only place they can stay consistent with the result type.
///
/// Encoded as one byte in bitcode, so the tag values are pinned there.
pub const AtomicOp = enum {
    /// Add the value operand.
    add,
    /// Keep the smaller of the two, compared as the value operand's type.
    min,
    /// Keep the larger of the two, compared as the value operand's type.
    max,
    bit_and,
    bit_or,
    bit_xor,
    /// Write the value operand and give back the old value.
    exchange,
    /// Write the value operand only if memory holds `compare`, and give back the old
    /// value either way. This is the ONE operation that reads the `compare` operand.
    compare_exchange,
};

/// How strongly an `atomic_rmw` orders the memory operations around it.
///
/// A backend may lower a weak order with a STRONGER instruction, because more ordering
/// than the program asked for is always correct. A backend must never lower a strong
/// order with a weaker instruction.
///
/// Encoded as one byte in bitcode, so the tag values are pinned there.
pub const AtomicOrdering = enum {
    /// Atomic, and nothing more. No memory operation around it is ordered.
    relaxed,
    /// No later memory operation moves before this one.
    acquire,
    /// No earlier memory operation moves after this one.
    release,
    /// Both of the two above.
    acq_rel,
    /// Both of the two above, plus a single total order over every `seq_cst` operation.
    seq_cst,
};

/// The set of threads an `atomic_rmw` is atomic with respect to.
///
/// The scope is part of the operation from its first version, for the reason
/// `BarrierScope` states: a scope-less operation needs a breaking change to every layer
/// the moment a second scope arrives, and a backend that lowers only one of them can
/// refuse the others only if the operation says which one it is.
///
/// Encoded as one byte in bitcode, so the tag values are pinned there.
pub const AtomicScope = enum {
    /// The threads of one workgroup (the CUDA block, the NVIDIA CTA).
    workgroup,
    /// Every thread on the device.
    device,
    /// Every thread on the device, plus the host and the peer devices.
    system,
};

/// An atomic read-modify-write of the memory at `ptr`: read the value there, apply `op`
/// to it with `value` (and `compare`, for `compare_exchange`), write the answer back, and
/// give back the OLD value. EFFECTFUL, and BOTH a load and a store of `ptr`.
///
/// This is a first-class operation and not a recognised call for a stronger reason than
/// the barrier has. Every pass that could break it reads opcode STRUCTURE, not call
/// side-effect metadata: `gvn` must not common two atomics on one address, `licm` must
/// not hoist one out of a loop, `dce` must not delete one whose result nobody reads,
/// `loadfwd` must not forward a store across one, and `addrfold` must not fold an offset
/// into one while the encoders leave their immediate offset field at zero.
///
/// The RESULT IS OPTIONAL, and this is the design decision the barrier did not have. An
/// atomic whose old value nobody reads lowers to a reduction (NVIDIA `RED`), which writes
/// no register and so claims no scoreboard; one whose old value is read lowers to the
/// full form (`ATOMG`), which claims one. A GPU has six scoreboards, so forcing a result
/// and trusting `dce` to notice nobody reads it would lose one on every fire-and-forget
/// counter increment. The optional result needs no new field here: `InstData.result` is
/// already `?Value`, so `appendAtomicRmw` builds the reading form and
/// `appendAtomicRmwStmt` the reduction form, and `instResult` tells them apart.
///
/// The address space rides in `ptr`'s type, exactly as it does for `load` and `store`, so
/// a backend picks its shared-memory form from the pointer and needs no new attribute.
pub const AtomicRmw = struct {
    op: AtomicOp,
    /// The address. Must be a `ptr`.
    ptr: Value,
    /// The operand the read-modify-write applies. Must be an integer, and the result (when
    /// there is one) has this same type.
    value: Value,
    /// The expected value, for `compare_exchange` ONLY. Null for every other operation,
    /// and non-null for that one. `verify` enforces both halves.
    compare: ?Value = null,
    ordering: AtomicOrdering,
    scope: AtomicScope,
};
/// A run of values in the function's value-list pool, used for variadic operands
/// like the arguments passed across a control-flow edge.
pub const ValueList = struct { start: u32, len: u32 };

/// A run of interned fp32-bit scales in the function's scale pool, used for a `matmul` quant
/// epilogue's `per_column` scale. Mirrors `ValueList`, but the pool holds constant data (u32),
/// never Values, so this is not scanned by any operand-use pass.
pub const ScaleList = struct { start: u32, len: u32 };

/// A run of interned int32 biases in the function's bias pool, used for a `matmul` quant
/// epilogue's per-column bias. Mirrors `ScaleList`, but the pool holds signed int32 constant
/// data, never Values, so this is not scanned by any operand-use pass.
pub const BiasList = struct { start: u32, len: u32 };

/// An edge to a block, passing arguments to its parameters. Used both as an
/// unconditional jump and as each side of a conditional branch.
pub const Jump = struct { target: Block, args: ValueList };

/// A conditional: take `then` when `cond` is true, `else` otherwise. In the high
/// profile this is a non-terminating instruction. Control continues to the
/// block's terminator afterward. Legalization lowers it to a flat
/// conditional-branch terminator in the low profile.
pub const If = struct { cond: Value, then: Jump, @"else": Jump };

/// The value list of a `ret` terminator. Holds up to 4 values inline: a void
/// return has a count of 0, a scalar return has a count of 1 (both lower
/// identically to today), and a count of 2 or more represents a future
/// register-pair or HFA struct return. No backend lowers a count above 1 yet.
pub const Ret = struct {
    values: [4]Value = undefined,
    count: u8 = 0,

    /// A void return.
    pub fn none() Ret {
        return .{ .count = 0 };
    }

    /// A single-value return. The common case, byte-identical to today's
    /// `?Value` path.
    pub fn one(v: Value) Ret {
        return .{ .values = .{ v, undefined, undefined, undefined }, .count = 1 };
    }

    /// A multi-value return. `vals` must hold at most 4 values.
    pub fn many(vals: []const Value) Ret {
        std.debug.assert(vals.len <= 4);
        var r: Ret = .{ .count = @intCast(vals.len) };
        for (vals, 0..) |v, i| r.values[i] = v;
        return r;
    }

    /// The live values, in return order.
    pub fn slice(self: *const Ret) []const Value {
        return self.values[0..self.count];
    }
};

/// How a block ends. The terminator transfers control out of the block. An unset
/// terminator is an implicit `ret void`.
pub const Terminator = union(enum) {
    /// Return from the function, with 0 to 4 values.
    ret: Ret,
    /// Branch unconditionally to a block, passing arguments to its parameters.
    jump: Jump,
};

/// A caller-facing description of a control-flow edge: a target block and the
/// arguments to pass it. The arguments are copied into the value-list pool.
pub const EdgeDesc = struct { target: Block, args: []const Value = &.{} };

/// What an attribute is attached to. Never a type (types are interned and shared).
pub const AttrTarget = union(enum) {
    func,
    block: Block,
    inst: Inst,
    value: Value,
};

/// One attribute attached to one target.
pub const AttrEntry = struct { target: AttrTarget, attr: Attribute };

/// Iterates the attributes attached to a single target.
pub const AttrIterator = struct {
    entries: []const AttrEntry,
    target: AttrTarget,
    index: usize = 0,

    pub fn next(self: *AttrIterator) ?Attribute {
        while (self.index < self.entries.len) {
            const entry = self.entries[self.index];
            self.index += 1;
            if (std.meta.eql(entry.target, self.target)) return entry.attr;
        }
        return null;
    }
};

/// The closed set of IR operations. Target-independent only, machine ops live in
/// codegen.
pub const Opcode = union(enum) {
    /// An integer constant of the instruction's result type.
    iconst: i64,
    /// A floating-point constant of the instruction's result type.
    fconst: f64,
    /// An f128 constant of the instruction's result type, as its binary128 bit
    /// pattern. A separate opcode because `fconst`'s f64 carrier cannot hold 128
    /// bits, and widening every existing fconst reader to a 128-bit carrier would
    /// churn them for a type they never produce.
    fconst128: u128,
    /// A binary arithmetic/bitwise operation.
    arith: Arith,
    /// Arithmetic against a constant operand (`lhs <op> imm`).
    arith_imm: ArithImm,
    /// Integer comparison of two operands, producing a `bool`.
    icmp: Compare,
    /// A value-producing conditional (`c := if {} else {}`).
    select: Select,
    /// Construct an aggregate from field values. High profile only.
    struct_new: StructNew,
    /// Extract a field from an aggregate. High profile only.
    extract: Extract,
    /// Convert a value to the result type (int<->float numeric conversion).
    convert: Convert,
    /// A single-operand op (bit reinterpret, or floating-point math) on the result type.
    unary: Unary,
    /// Reserve a stack slot and produce its address. Result type is `ptr`.
    alloca: Alloca,
    /// Call a function by symbol, passing arguments and producing its result.
    call: Call,
    /// Call a function through a computed address, passing arguments.
    call_indirect: CallIndirect,
    /// The address of a named global symbol, resolved at link time. Result `ptr`.
    global_addr: GlobalAddr,
    /// A load from memory, producing a value of the result type.
    load: Load,
    /// A store to memory. Produces no result.
    store: Store,
    /// A software prefetch hint for an address. Produces no result and has no
    /// observable effect.
    prefetch: Prefetch,
    /// Initialize a `va_list` object for reading a variadic call's extra arguments.
    /// Produces no result.
    va_start: VaStart,
    /// Fetch the next variadic argument from a `va_list` object, advancing it.
    /// Result type is the op's own `ty`.
    va_arg: VaArg,
    /// Finalize a `va_list` object. Produces no result.
    va_end: VaEnd,
    /// An INT8 4-way dot-product accumulate. Pure, like `arith`.
    dot: Dot,
    /// A fixed-tile matrix multiply. Produces no result. EFFECTFUL (writes memory
    /// at `c`). Its preconditions are per target, see `MatMul`.
    matmul: MatMul,
    /// An execution and memory barrier over a scope. Produces no result. EFFECTFUL.
    /// See `Barrier` for why this is an opcode and not a call.
    barrier: Barrier,
    /// An atomic read-modify-write. Produces the OLD value, or no result at all when
    /// nobody reads it. EFFECTFUL, and both a load and a store of `ptr`. See `AtomicRmw`.
    atomic_rmw: AtomicRmw,
    /// A non-terminating conditional. Produces no result in its statement form.
    @"if": If,
};

/// Per-instruction storage: its opcode and the value it defines, if any.
const InstData = struct {
    op: Opcode,
    result: ?Value,
};

/// What defines a value.
const ValueDef = union(enum) {
    /// The `index`-th parameter of `block`.
    block_param: struct { block: Block, index: u32 },
    /// The result of an instruction.
    inst_result: Inst,
};

/// Per-value storage: its type and what defines it.
const ValueData = struct {
    ty: Type,
    def: ValueDef,
};

/// Per-block storage.
const BlockData = struct {
    params: std.ArrayList(Value),
    insts: std.ArrayList(Inst),
    term: ?Terminator = null,
};

/// Owns an IR function's blocks, instructions, and values.
pub const Function = struct {
    allocator: std.mem.Allocator,
    types: TypeTable,
    blocks: std.ArrayList(BlockData),
    insts: std.ArrayList(InstData),
    values: std.ArrayList(ValueData),
    value_lists: std.ArrayList(Value),
    /// Interned fp32-bit scales for `matmul` quant `per_column` epilogues. Constant data, not
    /// Values, so it is never scanned by dce/licm/gvn/schedule/vectorize/remap.
    scale_pool: std.ArrayList(u32),
    /// Interned int32 biases for `matmul` quant per-column bias. Constant data, not Values, so
    /// it is never scanned by dce/licm/gvn/schedule/vectorize/remap, same as `scale_pool`.
    bias_pool: std.ArrayList(i32),
    attributes: std.ArrayList(AttrEntry),
    /// Callee names referenced by `call` instructions, owned by the function.
    symbols: std.ArrayList([]const u8),
    /// True when this function itself is a variadic definition, for example `int sum(int
    /// n, ...) { ... }`, as opposed to `Call.is_variadic`, which marks a variadic call site
    /// inside some function. The two are independent: a non-variadic function can still call
    /// a variadic one. Defaults to `false`, so every existing construction site stays byte-
    /// identical. A later backend reads it to decide whether to spill the incoming unnamed
    /// register or stack arguments, so `va_start`/`va_arg` can walk them.
    is_variadic: bool = false,
    /// This function's own fixed (declared, named) parameter count when `is_variadic` is
    /// set. Mirrors `Call.num_fixed`'s meaning but for the callee side. Meaningless when
    /// `is_variadic` is false, and then stays at its default `0`, matching every other
    /// defaulted-bool-paired field in this file, for example `Call.num_fixed`.
    num_fixed_params: u32 = 0,
    /// True when this function returns a struct through a hidden result pointer
    /// (`abi.classify`'s `.sret` plan for an aggregate over 16 bytes). The first entry
    /// parameter is that hidden pointer, and the function copies its result there and
    /// returns the same address in the ABI return register. Defaults to `false` (an
    /// ordinary return), so every existing construction site stays byte-identical. A
    /// register return sets it nowhere, since it needs no hidden pointer. It exists now
    /// so the marker round-trips through `clone`.
    sret: bool = false,
    /// True when this function has internal linkage (a C `static` function), so its object
    /// symbol is emitted with local binding. A local symbol does not participate in
    /// cross-object resolution, so two translation units that each define their own copy of a
    /// `static inline` helper (glibc's `__bswap_16`) do not collide at link time. Defaults
    /// to `false` (external/global), so every existing construction site stays byte-identical.
    is_local: bool = false,

    pub fn init(allocator: std.mem.Allocator) Function {
        return .{
            .allocator = allocator,
            .types = TypeTable.init(allocator),
            .blocks = .empty,
            .insts = .empty,
            .values = .empty,
            .value_lists = .empty,
            .scale_pool = .empty,
            .bias_pool = .empty,
            .attributes = .empty,
            .symbols = .empty,
        };
    }

    pub fn deinit(self: *Function) void {
        for (self.blocks.items) |*block| {
            block.params.deinit(self.allocator);
            block.insts.deinit(self.allocator);
        }
        self.blocks.deinit(self.allocator);
        self.insts.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.value_lists.deinit(self.allocator);
        self.scale_pool.deinit(self.allocator);
        self.bias_pool.deinit(self.allocator);
        for (self.attributes.items) |entry| self.freeAttr(entry.attr);
        self.attributes.deinit(self.allocator);
        for (self.symbols.items) |s| self.allocator.free(s);
        self.symbols.deinit(self.allocator);
        self.types.deinit();
    }

    /// A deep, independently-owned copy of `self` under `allocator`. The clone shares NO backing
    /// memory with the original: every array is duplicated, the type table is rebuilt by re-interning
    /// each kind in order (so Type handles are preserved), attribute string payloads are re-owned, and
    /// each block's parameter and instruction sub-lists are duplicated. All IR references are index or
    /// enum handles (Value/Inst/Block, ValueList `{start,len}`, scale/bias pool indices), never heap
    /// pointers, so the raw arrays copy verbatim and stay valid. Both the clone and the original can be
    /// `deinit`'d independently. Codegen uses this to run a MUTATING pass (critical-edge splitting)
    /// on a working copy without disturbing a caller's function, which may be shared across backends.
    pub fn clone(self: *const Function, allocator: std.mem.Allocator) std.mem.Allocator.Error!Function {
        var out = Function.init(allocator);
        errdefer out.deinit();

        // Whole-function metadata is plain data, copied straight across. `is_local` belongs
        // here with the rest: a clone of a C `static` function that came back global would
        // emit an object symbol two translation units can collide on.
        out.is_variadic = self.is_variadic;
        out.num_fixed_params = self.num_fixed_params;
        out.sret = self.sret;
        out.is_local = self.is_local;

        // Types: re-intern each kind in the original's order. The original table is deduped (every
        // kind unique), and interning is order-preserving, so the n-th kind receives handle n exactly,
        // keeping every Type handle in the copied arrays valid. `intern` re-owns slice-bearing kinds
        // (struct fields) into the clone's storage.
        for (self.types.kinds.items) |kind| _ = try out.types.intern(kind);

        // Interned callee names: re-intern in order (they are already unique, so indices match).
        for (self.symbols.items) |s| _ = try out.internSymbol(s);

        // Constant pools and the index-based reference arrays copy verbatim (plain data, no pointers).
        try out.scale_pool.appendSlice(allocator, self.scale_pool.items);
        try out.bias_pool.appendSlice(allocator, self.bias_pool.items);
        try out.value_lists.appendSlice(allocator, self.value_lists.items);
        try out.values.appendSlice(allocator, self.values.items);
        try out.insts.appendSlice(allocator, self.insts.items);

        // Blocks: duplicate each block's own parameter and instruction sub-lists (the terminator is
        // plain data and copies by value).
        try out.blocks.ensureTotalCapacity(allocator, self.blocks.items.len);
        for (self.blocks.items) |blk| {
            var nb: BlockData = .{ .params = .empty, .insts = .empty, .term = blk.term };
            errdefer {
                nb.params.deinit(allocator);
                nb.insts.deinit(allocator);
            }
            try nb.params.appendSlice(allocator, blk.params.items);
            try nb.insts.appendSlice(allocator, blk.insts.items);
            try out.blocks.append(allocator, nb);
        }

        // Attributes: `addAttr` re-owns any string payloads into the clone's storage.
        for (self.attributes.items) |entry| try out.addAttr(entry.target, entry.attr);

        return out;
    }

    /// Intern a callee name, returning its symbol index. Equal names share an
    /// index, the function owns the copied string.
    pub fn internSymbol(self: *Function, name: []const u8) std.mem.Allocator.Error!u32 {
        for (self.symbols.items, 0..) |s, i| {
            if (std.mem.eql(u8, s, name)) return @intCast(i);
        }
        const owned = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned);
        try self.symbols.append(self.allocator, owned);
        return @intCast(self.symbols.items.len - 1);
    }

    /// The name of an interned callee symbol.
    pub fn symbolName(self: *const Function, symbol: u32) []const u8 {
        return self.symbols.items[symbol];
    }

    /// The number of interned symbols (indices are `0..symbolCount`).
    pub fn symbolCount(self: *const Function) usize {
        return self.symbols.items.len;
    }

    /// Attach an attribute to a target. Namespaced string payloads are copied
    /// into function-owned storage so they stay valid.
    pub fn addAttr(self: *Function, target: AttrTarget, attr: Attribute) std.mem.Allocator.Error!void {
        const owned = try self.ownAttr(attr);
        errdefer self.freeAttr(owned);
        try self.attributes.append(self.allocator, .{ .target = target, .attr = owned });
    }

    /// Iterate the attributes attached to a target.
    pub fn attributesOf(self: *const Function, target: AttrTarget) AttrIterator {
        return .{ .entries = self.attributes.items, .target = target };
    }

    /// All attribute entries, for whole-function checks like verification.
    pub fn attributeEntries(self: *const Function) []const AttrEntry {
        return self.attributes.items;
    }

    /// True when the memory operation `inst` carries an `endian` attribute, in either of the
    /// two places the verifier accepts it: on the instruction, or on the instruction's result
    /// value. A load is tagged on its result, a store on the instruction, and this answers for
    /// both so a caller does not have to know which.
    ///
    /// ANY `endian` tag counts, `native` included. The attribute names the byte order of the
    /// DATA, and codegen resolves that against the target's native order (see
    /// `attribute.Endianness`). This layer does not know the target order, so it cannot tell a
    /// tag that needs a byte swap from one that does not: `endian(little)` needs a swap on a
    /// big-endian target exactly as `endian(big)` needs one on a little-endian target.
    ///
    /// The optimizer uses this to refuse a transform. See the doc on `Attribute.endian` for
    /// which transforms must refuse and which may carry the tag onto a copy.
    pub fn isByteOrderTagged(self: *const Function, inst: Inst) bool {
        var on_inst = self.attributesOf(.{ .inst = inst });
        while (on_inst.next()) |attr| switch (attr) {
            .endian => return true,
            .@"inline", .noreturn, .cold, .@"align", .custom => {},
        };
        if (self.instResult(inst)) |result| {
            var on_result = self.attributesOf(.{ .value = result });
            while (on_result.next()) |attr| switch (attr) {
                .endian => return true,
                .@"inline", .noreturn, .cold, .@"align", .custom => {},
            };
        }
        return false;
    }

    /// One `original -> clone` correspondence, for `cloneAttrs`.
    pub const ValuePair = struct { old: Value, new: Value };

    /// One `original -> clone` correspondence, for `cloneAttrs`.
    pub const InstPair = struct { old: Inst, new: Inst };

    /// Copy every attribute of a cloned entity onto its clone, inside ONE function.
    ///
    /// An attribute is not decoration. `endian` selects a byte-swapping load, and
    /// `vulcan.gpu.builtin` says a parameter comes from the hardware and takes no room in the
    /// parameter block. A pass that duplicates instructions without duplicating their
    /// attributes gives the copy a different meaning from the original, and the backend then
    /// emits the wrong code with no diagnostic. Every pass that duplicates instructions inside
    /// a function must call this.
    ///
    /// `values` and `insts` hold ONLY what the caller itself cloned. A pass whose value map is
    /// pre-seeded with an outside substitution must not put that substitution here, or it would
    /// pull an attribute onto a value the clone does not own.
    ///
    /// A `func` attribute is left alone: it already describes the function the clone sits in.
    /// A `block` attribute is left alone as well, because the structured control-flow
    /// attributes hold merge and continue block ids that name the ORIGINAL region, and a second
    /// block must not claim to be that region's merge point.
    pub fn cloneAttrs(
        self: *Function,
        values: []const ValuePair,
        insts: []const InstPair,
    ) std.mem.Allocator.Error!void {
        // Read the end ONCE. `addAttr` appends to this same list, so a copied entry would
        // otherwise be seen again and copied forever.
        const end = self.attributes.items.len;
        if (end == 0) return;

        var i: usize = 0;
        while (i < end) : (i += 1) {
            // Re-index each time: the append below can have moved the backing array. The
            // attribute's own string payloads are function-owned and do not move, so the
            // copy taken here stays valid across `addAttr`.
            const entry = self.attributes.items[i];
            // EVERY matching pair, not the first. One original can have several copies: a
            // loop unrolled by four gives one instruction four clones, and each needs its own
            // entry.
            switch (entry.target) {
                .value => |v| for (values) |pair| {
                    if (pair.old == v) try self.addAttr(.{ .value = pair.new }, entry.attr);
                },
                .inst => |n| for (insts) |pair| {
                    if (pair.old == n) try self.addAttr(.{ .inst = pair.new }, entry.attr);
                },
                .func, .block => {},
            }
        }
    }

    /// Copy every attribute of a cloned entity from `src` onto its clone in `self`. The
    /// cross-function form of `cloneAttrs`, for a pass that copies one function's body into
    /// another one (inlining). `addAttr` re-owns any string payload, so the copy shares no
    /// memory with `src`.
    ///
    /// A `func` attribute does NOT travel: it describes the SOURCE function (`noreturn`,
    /// `inline`), which says nothing about the function the body is spliced into.
    pub fn cloneAttrsFrom(
        self: *Function,
        src: *const Function,
        values: []const ValuePair,
        insts: []const InstPair,
    ) std.mem.Allocator.Error!void {
        // Two different functions. Appending to the list being read would move it.
        std.debug.assert(self != src);

        for (src.attributes.items) |entry| {
            // EVERY matching pair, for the same reason as `cloneAttrs`.
            switch (entry.target) {
                .value => |v| for (values) |pair| {
                    if (pair.old == v) try self.addAttr(.{ .value = pair.new }, entry.attr);
                },
                .inst => |n| for (insts) |pair| {
                    if (pair.old == n) try self.addAttr(.{ .inst = pair.new }, entry.attr);
                },
                .func, .block => {},
            }
        }
    }

    /// Copy any string payloads of an attribute into function-owned memory.
    fn ownAttr(self: *Function, attr: Attribute) std.mem.Allocator.Error!Attribute {
        switch (attr) {
            .custom => |c| {
                const namespace = try self.allocator.dupe(u8, c.namespace);
                errdefer self.allocator.free(namespace);
                const key = try self.allocator.dupe(u8, c.key);
                errdefer self.allocator.free(key);
                const value: attribute.AttrValue = switch (c.value) {
                    .string => |s| .{ .string = try self.allocator.dupe(u8, s) },
                    else => c.value,
                };
                return .{ .custom = .{ .namespace = namespace, .key = key, .value = value } };
            },
            else => return attr,
        }
    }

    /// Free any function-owned string payloads of an attribute.
    fn freeAttr(self: *Function, attr: Attribute) void {
        switch (attr) {
            .custom => |c| {
                self.allocator.free(c.namespace);
                self.allocator.free(c.key);
                switch (c.value) {
                    .string => |s| self.allocator.free(s),
                    else => {},
                }
            },
            else => {},
        }
    }

    /// Append a fresh, empty block and return its handle.
    pub fn appendBlock(self: *Function) std.mem.Allocator.Error!Block {
        const index: u32 = @intCast(self.blocks.items.len);
        try self.blocks.append(self.allocator, .{ .params = .empty, .insts = .empty });
        return @enumFromInt(index);
    }

    /// Append an instruction to a block, returning the result value it defines.
    pub fn appendInst(self: *Function, block: Block, ty: Type, op: Opcode) std.mem.Allocator.Error!Value {
        const inst: Inst = @enumFromInt(@as(u32, @intCast(self.insts.items.len)));
        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));

        try self.values.append(self.allocator, .{ .ty = ty, .def = .{ .inst_result = inst } });
        try self.insts.append(self.allocator, .{ .op = op, .result = value });
        try self.blocks.items[@intFromEnum(block)].insts.append(self.allocator, inst);
        return value;
    }

    /// Append a typed parameter to a block, returning the value it introduces.
    pub fn appendBlockParam(self: *Function, block: Block, ty: Type) std.mem.Allocator.Error!Value {
        const data = &self.blocks.items[@intFromEnum(block)];
        const param_index: u32 = @intCast(data.params.items.len);

        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));
        try self.values.append(self.allocator, .{
            .ty = ty,
            .def = .{ .block_param = .{ .block = block, .index = param_index } },
        });
        try data.params.append(self.allocator, value);
        return value;
    }

    /// The type of a value.
    pub fn valueType(self: *const Function, value: Value) Type {
        return self.values.items[@intFromEnum(value)].ty;
    }

    /// Retype a value in place (its defining instruction / param is unchanged). Used by
    /// the SIMD widener to re-type a scalar value to a packed vector without rebuilding
    /// the SSA graph. The caller is responsible for keeping the def consistent (e.g.
    /// splatting a scalar constant that becomes a vector).
    pub fn setValueType(self: *Function, value: Value, ty: Type) void {
        self.values.items[@intFromEnum(value)].ty = ty;
    }

    /// The textual name number of a value: its position in a deterministic walk
    /// (block by block, parameters then instruction results). A pure function of
    /// structure, so the text round-trips.
    fn valueName(self: *const Function, value: Value) u32 {
        var n: u32 = 0;
        for (self.blocks.items) |block| {
            for (block.params.items) |param| {
                if (param == value) return n;
                n += 1;
            }
            for (block.insts.items) |inst| {
                if (self.insts.items[@intFromEnum(inst)].result) |result| {
                    if (result == value) return n;
                    n += 1;
                }
            }
        }
        return n;
    }

    /// Set the block's terminator.
    pub fn setTerminator(self: *Function, block: Block, term: Terminator) void {
        self.blocks.items[@intFromEnum(block)].term = term;
    }

    /// Terminate a block with an unconditional jump, passing `args` to the
    /// target's block parameters.
    pub fn setJump(self: *Function, block: Block, target: Block, args: []const Value) std.mem.Allocator.Error!void {
        const list = try self.internValues(args);
        self.setTerminator(block, .{ .jump = .{ .target = target, .args = list } });
    }

    /// Append a result-less instruction (a statement) to a block, returning the
    /// instruction handle. Primitive behind the typed statement builders, also
    /// used to reconstruct instructions during deserialization.
    pub fn appendStmtRaw(self: *Function, block: Block, op: Opcode) std.mem.Allocator.Error!Inst {
        const inst: Inst = @enumFromInt(@as(u32, @intCast(self.insts.items.len)));
        try self.insts.append(self.allocator, .{ .op = op, .result = null });
        try self.blocks.items[@intFromEnum(block)].insts.append(self.allocator, inst);
        return inst;
    }

    /// Append a result-less instruction (a statement) to a block.
    fn appendStmt(self: *Function, block: Block, op: Opcode) std.mem.Allocator.Error!void {
        _ = try self.appendStmtRaw(block, op);
    }

    /// Append a non-terminating conditional to a block. Control continues to the
    /// block's terminator afterward.
    pub fn appendIf(self: *Function, block: Block, cond: Value, then_edge: EdgeDesc, else_edge: EdgeDesc) std.mem.Allocator.Error!void {
        const then_jump: Jump = .{ .target = then_edge.target, .args = try self.internValues(then_edge.args) };
        const else_jump: Jump = .{ .target = else_edge.target, .args = try self.internValues(else_edge.args) };
        try self.appendStmt(block, .{ .@"if" = .{ .cond = cond, .then = then_jump, .@"else" = else_jump } });
    }

    /// Append a store to a block.
    pub fn appendStore(self: *Function, block: Block, value: Value, ptr: Value) std.mem.Allocator.Error!void {
        try self.appendStoreVol(block, value, ptr, false);
    }

    /// Append a store to a block, marking it `volatile` when `is_volatile` is set.
    /// The frontend uses this when writing through a `volatile`-qualified lvalue.
    /// `appendStore` is this with `is_volatile = false`.
    pub fn appendStoreVol(self: *Function, block: Block, value: Value, ptr: Value, is_volatile: bool) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .store = .{ .value = value, .ptr = ptr, .@"volatile" = is_volatile } });
    }

    /// Append a software prefetch hint for `ptr` to a block. No observable effect.
    pub fn appendPrefetch(self: *Function, block: Block, ptr: Value) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .prefetch = .{ .ptr = ptr } });
    }

    /// Append a `va_start`: initialize the `va_list` object at `list` (its
    /// address, a `ptr`). No result.
    pub fn appendVaStart(self: *Function, block: Block, list: Value) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .va_start = .{ .list = list } });
    }

    /// Append a `va_arg`: fetch the next variadic argument of type `ty` from the
    /// `va_list` object at `list` (its address, a `ptr`), returning the fetched value.
    pub fn appendVaArg(self: *Function, block: Block, list: Value, ty: Type) std.mem.Allocator.Error!Value {
        return self.appendInst(block, ty, .{ .va_arg = .{ .list = list, .ty = ty } });
    }

    /// Append a `va_end`: finalize the `va_list` object at `list` (its address, a
    /// `ptr`). No result.
    pub fn appendVaEnd(self: *Function, block: Block, list: Value) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .va_end = .{ .list = list } });
    }

    /// Append an execution and memory barrier over `scope`. No result. EFFECTFUL: every
    /// pass must keep it, and must not move a memory operation across it. See `Barrier`.
    pub fn appendBarrier(self: *Function, block: Block, scope: BarrierScope) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .barrier = .{ .scope = scope } });
    }

    /// Append an atomic read-modify-write whose OLD value is read, returning that value.
    /// The result type is the value operand's type, so the two can never disagree.
    /// EFFECTFUL, and both a load and a store of `rmw.ptr`. See `AtomicRmw`.
    pub fn appendAtomicRmw(self: *Function, block: Block, rmw: AtomicRmw) std.mem.Allocator.Error!Value {
        return self.appendInst(block, self.valueType(rmw.value), .{ .atomic_rmw = rmw });
    }

    /// Append an atomic read-modify-write whose old value NOBODY reads: the reduction
    /// form. No result. Build this one, and not `appendAtomicRmw` plus a dead result, when
    /// the frontend knows the old value is unused: a GPU backend lowers it to an
    /// instruction that writes no register and so claims no scoreboard. See `AtomicRmw`.
    pub fn appendAtomicRmwStmt(self: *Function, block: Block, rmw: AtomicRmw) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .atomic_rmw = rmw });
    }

    /// Append an INT8 4-way dot-product accumulate: `result = acc + dot(a, b)`.
    /// Pure, like `arith`. The result type is `acc`'s type.
    pub fn appendDot(self: *Function, block: Block, acc: Value, a: Value, b: Value) std.mem.Allocator.Error!Value {
        return self.appendInst(block, self.valueType(acc), .{ .dot = .{ .acc = acc, .a = a, .b = b } });
    }

    /// Append an et-soc fixed-tile matrix multiply to a block: `c := a * b`
    /// (or `c += a * b` when `accumulate`), an `m x k` by `k x n` tile. No
    /// result. EFFECTFUL (writes memory at `c`).
    pub fn appendMatmul(self: *Function, block: Block, a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .matmul = .{ .a = a, .b = b, .c = c, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate } });
    }

    /// Append a `matmul` with an explicit per-operand signedness override (see `InputSigns`),
    /// e.g. uint8 activations times int8 weights. Only meaningful when `dtype == .int8`; verify
    /// rejects any other dtype paired with a non-null `input_signs`.
    pub fn appendMatmulSigned(self: *Function, block: Block, a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool, input_signs: InputSigns) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .matmul = .{ .a = a, .b = b, .c = c, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate, .input_signs = input_signs } });
    }

    /// Append a self-contained (`embedded`) `matmul`: identical to `appendMatmul`/`appendMatmulSigned`
    /// except the op is marked `embedded`, so the backend saves/restores every register it clobbers
    /// and holds a/b/c in dedicated registers. This is the builder a non-whole-function recognizer
    /// uses when it raises a matmul into a function that has code (and live values) around it. Pass a
    /// non-null `input_signs` for a mixed-signedness int8 matmul, else null (the symmetric cases).
    pub fn appendMatmulEmbedded(self: *Function, block: Block, a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool, input_signs: ?InputSigns) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .matmul = .{ .a = a, .b = b, .c = c, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate, .embedded = true, .input_signs = input_signs } });
    }

    /// Append a `matmul` with a fused int8-requantize epilogue (see `MatMulQuant`). Only meaningful
    /// when `dtype == .int8`; verify rejects any other dtype paired with a non-null `quant`.
    pub fn appendMatmulQuant(self: *Function, block: Block, a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool, quant: MatMulQuant) std.mem.Allocator.Error!void {
        try self.appendStmt(block, .{ .matmul = .{ .a = a, .b = b, .c = c, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate, .quant = quant } });
    }

    /// Append a `matmul` with a fused int8-requantize epilogue using a `per_column` scale: `scales`
    /// (fp32 bits, one per output column, `scales.len == n`) is interned as compile-time constant
    /// data into the scale pool, not an IR Value operand. Only meaningful when `dtype == .int8`;
    /// verify rejects any other dtype paired with a non-null `quant`, and rejects a per_column
    /// scale whose length does not equal `n`.
    pub fn appendMatmulQuantPerColumn(self: *Function, block: Block, a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool, relu: bool, out: MatMulQuantOut, scales: []const u32) std.mem.Allocator.Error!void {
        const h = try self.internScales(scales);
        try self.appendStmt(block, .{ .matmul = .{ .a = a, .b = b, .c = c, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate, .quant = .{ .scale = .{ .per_column = h }, .relu = relu, .out = out } } });
    }

    /// A `MatMulQuant` builder in un-interned, caller-friendly form: `bias`/`scale_per_column` are
    /// plain slices, interned into this function's pools by `appendMatmulQuantSpec`. Exists to tame
    /// the growing knob count on the quant epilogue (scale kind, bias, zero-point, relu, out) behind
    /// one call instead of a builder-per-combination; `appendMatmulQuant`/`appendMatmulQuantPerColumn`
    /// remain for the two original simple cases.
    pub const MatMulQuantSpec = struct {
        scale_scalar: ?u32 = null, // set EXACTLY one of scale_scalar / scale_per_column
        scale_per_column: ?[]const u32 = null,
        bias: ?[]const i32 = null, // per-column int32 bias, len must be n (verify enforces)
        zero_point: i32 = 0,
        relu: bool = false,
        out: MatMulQuantOut = .i8,
        input_signs: ?InputSigns = null, // optional per-operand signedness override, see `InputSigns`
    };

    /// Append a `matmul` with a fused quant epilogue built from a `MatMulQuantSpec`: interns
    /// `scale_per_column`/`bias` (whichever is set) into this function's pools and appends the
    /// resulting `MatMulQuant`. See `MatMulQuantSpec` and `MatMulQuant` for field semantics.
    pub fn appendMatmulQuantSpec(self: *Function, block: Block, a: Value, b: Value, c: Value, m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool, spec: MatMulQuantSpec) std.mem.Allocator.Error!void {
        std.debug.assert((spec.scale_scalar == null) != (spec.scale_per_column == null)); // exactly one
        const scale: MatMulScale = if (spec.scale_scalar) |sb|
            .{ .scalar = sb }
        else
            .{ .per_column = try self.internScales(spec.scale_per_column.?) };
        const bias: ?BiasList = if (spec.bias) |bb| try self.internBias(bb) else null;
        try self.appendStmt(block, .{ .matmul = .{ .a = a, .b = b, .c = c, .m = m, .n = n, .k = k, .dtype = dtype, .accumulate = accumulate, .quant = .{ .scale = scale, .relu = spec.relu, .out = spec.out, .bias = bias, .zero_point = spec.zero_point }, .input_signs = spec.input_signs } });
    }

    /// Append `lhs <op> imm`, returning the result value of type `ty`.
    pub fn appendArithImm(self: *Function, block: Block, ty: Type, op: BinOp, lhs: Value, imm: i64) std.mem.Allocator.Error!Value {
        return self.appendInst(block, ty, .{ .arith_imm = .{ .op = op, .lhs = lhs, .imm = imm } });
    }

    /// Append a call to `name` with `args`, returning the result value of type `ty`.
    pub fn appendCall(self: *Function, block: Block, ty: Type, name: []const u8, args: []const Value) std.mem.Allocator.Error!Value {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        return self.appendInst(block, ty, .{ .call = .{ .symbol = symbol, .args = list } });
    }

    /// Append an indirect call through `target` with `args`, returning the result.
    pub fn appendCallIndirect(self: *Function, block: Block, ty: Type, target: Value, args: []const Value) std.mem.Allocator.Error!Value {
        const list = try self.internValues(args);
        return self.appendInst(block, ty, .{ .call_indirect = .{ .target = target, .args = list } });
    }

    /// Append a variadic call to `name` with `args`, returning the result of type
    /// `ty`. `num_fixed` is the callee's own fixed (declared) parameter count, see `Call`'s
    /// doc comment. `appendCall` above is this with `is_variadic = false`, `num_fixed = 0`.
    pub fn appendCallV(self: *Function, block: Block, ty: Type, name: []const u8, args: []const Value, num_fixed: u32) std.mem.Allocator.Error!Value {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        return self.appendInst(block, ty, .{ .call = .{ .symbol = symbol, .args = list, .is_variadic = true, .num_fixed = num_fixed } });
    }

    /// Append a variadic indirect call through `target` with `args`, returning the
    /// result. Mirrors `appendCallV`, see there.
    pub fn appendCallIndirectV(self: *Function, block: Block, ty: Type, target: Value, args: []const Value, num_fixed: u32) std.mem.Allocator.Error!Value {
        const list = try self.internValues(args);
        return self.appendInst(block, ty, .{ .call_indirect = .{ .target = target, .args = list, .is_variadic = true, .num_fixed = num_fixed } });
    }

    /// A void-returning indirect call - the `call_indirect` counterpart of `appendVoidCall`. It
    /// is a statement (no result), so the backend emits the call without a result move, exactly
    /// as it does for a void direct `.call`. A `void (*fp)(void)` call reaches this.
    pub fn appendVoidCallIndirect(self: *Function, block: Block, target: Value, args: []const Value) std.mem.Allocator.Error!void {
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call_indirect = .{ .target = target, .args = list } });
    }

    /// The variadic form of `appendVoidCallIndirect`.
    pub fn appendVoidCallIndirectV(self: *Function, block: Block, target: Value, args: []const Value, num_fixed: u32) std.mem.Allocator.Error!void {
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call_indirect = .{ .target = target, .args = list, .is_variadic = true, .num_fixed = num_fixed } });
    }

    /// Append a `global_addr` for the named symbol, returning its address as `ty`
    /// (which should be `ptr`). The symbol is resolved at link time.
    pub fn appendGlobalAddr(self: *Function, block: Block, ty: Type, name: []const u8) std.mem.Allocator.Error!Value {
        const symbol = try self.internSymbol(name);
        return self.appendInst(block, ty, .{ .global_addr = .{ .symbol = symbol } });
    }

    /// Append a GOT-indirect `global_addr` for the named symbol, returning its address
    /// as `ty` (which should be `ptr`). The address is loaded from the GOT at run time
    /// (a data import from another shared object); only aarch64 lowers this today.
    pub fn appendGlobalAddrGot(self: *Function, block: Block, ty: Type, name: []const u8) std.mem.Allocator.Error!Value {
        const symbol = try self.internSymbol(name);
        return self.appendInst(block, ty, .{ .global_addr = .{ .symbol = symbol, .via_got = true } });
    }

    /// Append a call to `name` with `args` that discards its result (a statement).
    pub fn appendVoidCall(self: *Function, block: Block, name: []const u8, args: []const Value) std.mem.Allocator.Error!void {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call = .{ .symbol = symbol, .args = list } });
    }

    /// Append a variadic void call, `appendVoidCall`'s counterpart for a
    /// `void`-returning variadic callee, for example `void warn(const char *fmt, ...)`. Mirrors
    /// `appendCallV`, see there.
    pub fn appendVoidCallV(self: *Function, block: Block, name: []const u8, args: []const Value, num_fixed: u32) std.mem.Allocator.Error!void {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call = .{ .symbol = symbol, .args = list, .is_variadic = true, .num_fixed = num_fixed } });
    }

    /// Copy `pieces` (at most 4) into a fixed `[4]RetPiece`, defaulting the unused tail slots.
    fn retPieceArray(pieces: []const RetPiece) [4]RetPiece {
        std.debug.assert(pieces.len <= 4);
        var out: [4]RetPiece = @splat(.{});
        for (pieces, 0..) |p, i| out[i] = p;
        return out;
    }

    /// Append a call to `name` that returns a struct in registers. The
    /// call has no scalar result, since its "result" is written to memory. The backend stores the
    /// return registers into `ret_dest` (a frame-relative destination slot address) after the
    /// call, one per `pieces` entry, each into `[ret_dest + piece.offset]` from `piece`'s bank.
    /// `pieces` has one entry per return eightbyte (1 to 4). See `RetPiece` and `Call`'s doc.
    pub fn appendCallStructRet(self: *Function, block: Block, name: []const u8, args: []const Value, ret_dest: Value, pieces: []const RetPiece) std.mem.Allocator.Error!void {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call = .{ .symbol = symbol, .args = list, .ret_dest = ret_dest, .ret_regs = @intCast(pieces.len), .ret_pieces = retPieceArray(pieces) } });
    }

    /// `appendCallStructRet`'s indirect-call counterpart, a struct-returning call through a
    /// computed `target` pointer. Mirrors `appendCallStructRet`, see there and `CallIndirect`.
    pub fn appendCallIndirectStructRet(self: *Function, block: Block, target: Value, args: []const Value, ret_dest: Value, pieces: []const RetPiece) std.mem.Allocator.Error!void {
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call_indirect = .{ .target = target, .args = list, .ret_dest = ret_dest, .ret_regs = @intCast(pieces.len), .ret_pieces = retPieceArray(pieces) } });
    }

    /// Append a call to `name` that returns a struct through a hidden result pointer
    /// (the `.sret` plan for a struct over 16 bytes). The call has no scalar result: `args[0]`
    /// is the caller's destination slot address (the hidden pointer), and the callee writes its
    /// return value there. The frontend passes the same slot address on as this call's lvalue, so
    /// the call itself needs neither `ret_dest` nor `ret_regs`, since the callee did the store. See
    /// `Call`'s doc for the `sret` field.
    pub fn appendCallSret(self: *Function, block: Block, name: []const u8, args: []const Value) std.mem.Allocator.Error!void {
        const symbol = try self.internSymbol(name);
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call = .{ .symbol = symbol, .args = list, .sret = true } });
    }

    /// `appendCallSret`'s indirect-call counterpart, a hidden-pointer struct-returning call through
    /// a computed `target` pointer. Mirrors `appendCallSret`, see there and `CallIndirect`.
    pub fn appendCallIndirectSret(self: *Function, block: Block, target: Value, args: []const Value) std.mem.Allocator.Error!void {
        const list = try self.internValues(args);
        try self.appendStmt(block, .{ .call_indirect = .{ .target = target, .args = list, .sret = true } });
    }

    /// Append an aggregate construction, returning the struct value. `ty` must be
    /// the struct type of the field values.
    pub fn appendStructNew(self: *Function, block: Block, ty: Type, fields: []const Value) std.mem.Allocator.Error!Value {
        const list = try self.internValues(fields);
        return self.appendInst(block, ty, .{ .struct_new = .{ .fields = list } });
    }

    /// The instructions of a block, in order.
    pub fn blockInsts(self: *const Function, block: Block) []const Inst {
        return self.blocks.items[@intFromEnum(block)].insts.items;
    }

    /// The number of blocks in the function.
    pub fn blockCount(self: *const Function) usize {
        return self.blocks.items.len;
    }

    /// The parameters of a block, in order.
    pub fn blockParams(self: *const Function, block: Block) []const Value {
        return self.blocks.items[@intFromEnum(block)].params.items;
    }

    /// The value an instruction defines, if any.
    pub fn instResult(self: *const Function, inst: Inst) ?Value {
        return self.insts.items[@intFromEnum(inst)].result;
    }

    /// The number of instructions in the function.
    pub fn instCount(self: *const Function) usize {
        return self.insts.items.len;
    }

    // mutable accessors, for passes like legalization

    /// A mutable pointer to an instruction's opcode (to rewrite operands).
    pub fn opcodeMut(self: *Function, inst: Inst) *Opcode {
        return &self.insts.items[@intFromEnum(inst)].op;
    }

    /// A mutable pointer to a block's terminator slot.
    pub fn terminatorPtr(self: *Function, block: Block) *?Terminator {
        return &self.blocks.items[@intFromEnum(block)].term;
    }

    /// A mutable view of a value-list run (to rewrite variadic operands).
    pub fn valueListMut(self: *Function, list: ValueList) []Value {
        return self.value_lists.items[list.start..][0..list.len];
    }

    /// Replace every use of `from` with `to` across instruction operands, `if`
    /// edge arguments, and terminators (an SSA "replace all uses with").
    /// Definitions are untouched. The caller handles any now-dead `from`.
    pub fn replaceAllUses(self: *Function, from: Value, to: Value) void {
        const r = struct {
            fn repl(f: Value, t: Value, v: Value) Value {
                return if (v == f) t else v;
            }
        }.repl;
        for (0..self.instCount()) |i| {
            const op = self.opcodeMut(@enumFromInt(i));
            switch (op.*) {
                // A barrier carries a scope and no Value operand, so it has nothing to
                // replace. It joins the constants here for that reason only.
                .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
                .arith => |*a| {
                    a.lhs = r(from, to, a.lhs);
                    a.rhs = r(from, to, a.rhs);
                },
                .arith_imm => |*a| a.lhs = r(from, to, a.lhs),
                .icmp => |*c| {
                    c.lhs = r(from, to, c.lhs);
                    c.rhs = r(from, to, c.rhs);
                },
                .select => |*s| {
                    s.cond = r(from, to, s.cond);
                    s.then = r(from, to, s.then);
                    s.@"else" = r(from, to, s.@"else");
                },
                .extract => |*e| e.aggregate = r(from, to, e.aggregate),
                .convert => |*cv| cv.value = r(from, to, cv.value),
                .unary => |*u| u.value = r(from, to, u.value),
                .load => |*l| l.ptr = r(from, to, l.ptr),
                .store => |*st| {
                    st.value = r(from, to, st.value);
                    st.ptr = r(from, to, st.ptr);
                },
                .prefetch => |*pf| pf.ptr = r(from, to, pf.ptr),
                .va_start => |*vs| vs.list = r(from, to, vs.list),
                .va_arg => |*va| va.list = r(from, to, va.list),
                .va_end => |*ve| ve.list = r(from, to, ve.list),
                .dot => |*d| {
                    d.acc = r(from, to, d.acc);
                    d.a = r(from, to, d.a);
                    d.b = r(from, to, d.b);
                },
                .matmul => |*mm| {
                    mm.a = r(from, to, mm.a);
                    mm.b = r(from, to, mm.b);
                    mm.c = r(from, to, mm.c);
                },
                .atomic_rmw => |*a| {
                    a.ptr = r(from, to, a.ptr);
                    a.value = r(from, to, a.value);
                    // The compare operand exists only in the compare-exchange form, and it is
                    // a Value like the other two when it does exist.
                    if (a.compare) |*c| c.* = r(from, to, c.*);
                },
                .struct_new => |sn| for (self.valueListMut(sn.fields)) |*f| {
                    f.* = r(from, to, f.*);
                },
                .call => |*c| {
                    if (c.ret_dest) |*rd| rd.* = r(from, to, rd.*); // The register-return dest is a use.
                    for (self.valueListMut(c.args)) |*arg| arg.* = r(from, to, arg.*);
                },
                .call_indirect => |*c| {
                    c.target = r(from, to, c.target);
                    if (c.ret_dest) |*rd| rd.* = r(from, to, rd.*); // The register-return dest is a use.
                    for (self.valueListMut(c.args)) |*arg| arg.* = r(from, to, arg.*);
                },
                .@"if" => |*cf| {
                    cf.cond = r(from, to, cf.cond);
                    for (self.valueListMut(cf.then.args)) |*arg| arg.* = r(from, to, arg.*);
                    for (self.valueListMut(cf.@"else".args)) |*arg| arg.* = r(from, to, arg.*);
                },
            }
        }
        for (0..self.blockCount()) |bi| {
            const term = self.terminatorPtr(@enumFromInt(bi));
            if (term.*) |*t| switch (t.*) {
                .ret => |*ret| for (ret.values[0..ret.count]) |*vv| {
                    vv.* = r(from, to, vv.*);
                },
                .jump => |*j| for (self.valueListMut(j.args)) |*arg| {
                    arg.* = r(from, to, arg.*);
                },
            };
        }
    }

    /// A mutable pointer to a block's instruction list (to drop instructions).
    pub fn blockInstsMut(self: *Function, block: Block) *std.ArrayList(Inst) {
        return &self.blocks.items[@intFromEnum(block)].insts;
    }

    /// Create a fresh block-parameter value of type `ty`, without adding it to
    /// any block's parameter list. The caller installs it via `setBlockParams`.
    pub fn newParam(self: *Function, block: Block, ty: Type) std.mem.Allocator.Error!Value {
        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));
        try self.values.append(self.allocator, .{
            .ty = ty,
            .def = .{ .block_param = .{ .block = block, .index = 0 } },
        });
        return value;
    }

    /// Replace a block's parameter list with `params`.
    pub fn setBlockParams(self: *Function, block: Block, params: []const Value) std.mem.Allocator.Error!void {
        const data = &self.blocks.items[@intFromEnum(block)];
        data.params.clearRetainingCapacity();
        try data.params.appendSlice(self.allocator, params);
    }

    /// Create an instruction and its result value without adding it to any
    /// block. The caller places it via `setBlockInsts`.
    pub fn createInst(self: *Function, ty: Type, op: Opcode) std.mem.Allocator.Error!Value {
        const inst: Inst = @enumFromInt(@as(u32, @intCast(self.insts.items.len)));
        const value: Value = @enumFromInt(@as(u32, @intCast(self.values.items.len)));
        try self.values.append(self.allocator, .{ .ty = ty, .def = .{ .inst_result = inst } });
        try self.insts.append(self.allocator, .{ .op = op, .result = value });
        return value;
    }

    /// Replace a block's instruction list with `insts`.
    pub fn setBlockInsts(self: *Function, block: Block, insts: []const Inst) std.mem.Allocator.Error!void {
        const data = &self.blocks.items[@intFromEnum(block)];
        data.insts.clearRetainingCapacity();
        try data.insts.appendSlice(self.allocator, insts);
    }

    /// Permute the function's blocks into `order` (`order[i]` is the OLD block that becomes new
    /// index `i`) and remap every block reference to the new ids. `order` must be a permutation of
    /// `0..blockCount()` with `order[0]` the entry (Block 0), which stays first. Block params/
    /// insts/terminator travel with each block: the `BlockData` structs are moved (their
    /// `ArrayList` handles relocated), never copied field-by-field, so no inner list is
    /// reallocated or double-freed. Values are function-global, so reordering blocks needs no
    /// value remap. The CFG (edges) is unchanged, only the linear order differs. Foundation for
    /// the block-layout pass.
    ///
    /// CAVEAT: this remaps block references in terminators and `@"if"` edges only. It does NOT remap
    /// block ids encoded in ATTRIBUTES (`AttrTarget.block` keys, or the glsl/wasm/spirv structured
    /// control-flow "cf" custom attributes that store merge/continue block ids as int payloads). A
    /// caller must not reorder a function that carries block-keyed attributes, or it would leave those
    /// references stale. The block-layout pass (item 5) runs only on the machine-backend path and
    /// must skip any function with such attributes.
    pub fn reorderBlocks(self: *Function, allocator: std.mem.Allocator, order: []const Block) std.mem.Allocator.Error!void {
        const n = self.blockCount();
        std.debug.assert(order.len == n);
        std.debug.assert(order[0] == @as(Block, @enumFromInt(0))); // entry stays first

        // order must be a permutation of 0..n: every old id appears exactly once.
        const seen = try allocator.alloc(bool, n);
        defer allocator.free(seen);
        @memset(seen, false);
        for (order) |old| {
            const old_index = @intFromEnum(old);
            std.debug.assert(old_index < n);
            std.debug.assert(!seen[old_index]); // no id repeated
            seen[old_index] = true;
        }

        // new_id[old block index] = new block index.
        const new_id = try allocator.alloc(u32, n);
        defer allocator.free(new_id);
        for (order, 0..) |old, new_index| new_id[@intFromEnum(old)] = @intCast(new_index);

        // Permute the BlockData structs themselves: tmp[i] takes ownership of the storage that
        // used to live at the old index, then the memcpy-back writes those (moved, not copied)
        // structs into their new slots. No inner ArrayList is touched, so nothing is freed twice.
        const tmp = try allocator.alloc(BlockData, n);
        defer allocator.free(tmp);
        for (order, 0..) |old, new_index| tmp[new_index] = self.blocks.items[@intFromEnum(old)];
        @memcpy(self.blocks.items, tmp);

        // Remap every block reference (terminator jump targets, `if` then/else targets) through
        // new_id. Values are function-global, so nothing else needs remapping.
        var bi: usize = 0;
        while (bi < n) : (bi += 1) {
            const block: Block = @enumFromInt(bi);
            const tp = self.terminatorPtr(block);
            if (tp.*) |*t| switch (t.*) {
                .ret => {},
                .jump => |*j| {
                    const mapped = new_id[@intFromEnum(j.target)];
                    std.debug.assert(mapped < n);
                    j.target = @enumFromInt(mapped);
                },
            };
            for (self.blockInsts(block)) |inst| {
                const op = self.opcodeMut(inst);
                if (op.* == .@"if") {
                    const then_mapped = new_id[@intFromEnum(op.@"if".then.target)];
                    const else_mapped = new_id[@intFromEnum(op.@"if".@"else".target)];
                    std.debug.assert(then_mapped < n);
                    std.debug.assert(else_mapped < n);
                    op.@"if".then.target = @enumFromInt(then_mapped);
                    op.@"if".@"else".target = @enumFromInt(else_mapped);
                }
            }
        }
    }

    /// Clone `src`'s params and instructions into a NEW appended block with ALL-FRESH values,
    /// remapping every internal reference old->new. The terminator is copied (Block targets
    /// unchanged, Value args remapped). Fills `map` (old Value -> new Value) for every param and
    /// inst-result of `src`, so the caller can remap any reference from OUTSIDE `src` that should
    /// now point at the clone (e.g. rewiring one predecessor's edge onto the fresh copy). Does NOT
    /// modify `src`. Foundation for tail duplication in general jump threading: cloning a shared
    /// block gives one predecessor a private copy it can specialize.
    ///
    /// Every attribute on a copied parameter, instruction or instruction result travels with the
    /// copy (see `cloneAttrs`), so a cloned load keeps its `endian` and a cloned parameter keeps
    /// its `vulcan.gpu.builtin` tag. An attribute keyed by `src` ITSELF does not travel: see the
    /// note on `cloneAttrs`.
    pub fn cloneBlock(self: *Function, allocator: std.mem.Allocator, src: Block, map: *std.AutoHashMapUnmanaged(Value, Value)) std.mem.Allocator.Error!Block {
        const dst = try self.appendBlock();

        // The `original -> clone` record the attribute copy needs. It holds only what this call
        // clones, never anything the caller pre-seeded into `map`.
        var value_pairs: std.ArrayList(ValuePair) = .empty;
        defer value_pairs.deinit(allocator);
        var inst_pairs: std.ArrayList(InstPair) = .empty;
        defer inst_pairs.deinit(allocator);

        for (self.blockParams(src)) |p| {
            const np = try self.appendBlockParam(dst, self.valueType(p));
            try map.put(allocator, p, np);
            try value_pairs.append(allocator, .{ .old = p, .new = np });
        }

        for (self.blockInsts(src)) |inst| {
            const op = try self.remapOpcode(allocator, self.opcode(inst), map);
            if (self.instResult(inst)) |result| {
                const nr = try self.appendInst(dst, self.valueType(result), op);
                try map.put(allocator, result, nr);
                try value_pairs.append(allocator, .{ .old = result, .new = nr });
                try inst_pairs.append(allocator, .{ .old = inst, .new = self.definingInst(nr).? });
            } else {
                // @"if", store, prefetch, matmul: result-less statements.
                const ni = try self.appendStmtRaw(dst, op);
                try inst_pairs.append(allocator, .{ .old = inst, .new = ni });
            }
        }

        try self.cloneAttrs(value_pairs.items, inst_pairs.items);

        if (self.terminator(src)) |term| {
            const new_term: Terminator = switch (term) {
                .ret => |r| blk: {
                    var nr = r;
                    for (nr.values[0..nr.count]) |*vv| vv.* = remapValue(map, vv.*);
                    break :blk .{ .ret = nr };
                },
                .jump => |j| .{
                    .jump = .{
                        .target = j.target, // block targets are never remapped, only values are
                        .args = try self.remapValueList(allocator, j.args, map),
                    },
                },
            };
            self.setTerminator(dst, new_term);
        }

        return dst;
    }

    /// Copy `op`, replacing every `Value` operand that is a key of `map` with its mapped value (an
    /// operand not in `map`, i.e. defined outside the block being cloned, is left as-is). Non-Value
    /// fields (immediates, types, block targets, `matmul`'s shape/dtype metadata) are copied
    /// unchanged. Exhaustive over every `Opcode` variant (no `else` prong) so a newly added
    /// Value-carrying field cannot silently escape the remap. It mirrors the operand walks in
    /// `replaceAllUses` and `verify.zig`'s dominance check above.
    fn remapOpcode(self: *Function, allocator: std.mem.Allocator, op: Opcode, map: *const std.AutoHashMapUnmanaged(Value, Value)) std.mem.Allocator.Error!Opcode {
        return switch (op) {
            // A barrier's only field is its scope, which is not a Value, so the whole
            // opcode copies unchanged.
            .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => op,
            .arith => |a| .{ .arith = .{ .op = a.op, .lhs = remapValue(map, a.lhs), .rhs = remapValue(map, a.rhs) } },
            .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = remapValue(map, a.lhs), .imm = a.imm } },
            .icmp => |c| .{ .icmp = .{ .op = c.op, .lhs = remapValue(map, c.lhs), .rhs = remapValue(map, c.rhs) } },
            .select => |s| .{ .select = .{
                .cond = remapValue(map, s.cond),
                .then = remapValue(map, s.then),
                .@"else" = remapValue(map, s.@"else"),
            } },
            .struct_new => |sn| .{ .struct_new = .{ .fields = try self.remapValueList(allocator, sn.fields, map) } },
            .extract => |ex| .{ .extract = .{ .aggregate = remapValue(map, ex.aggregate), .index = ex.index } },
            .convert => |cv| .{ .convert = .{ .value = remapValue(map, cv.value) } },
            .unary => |u| .{ .unary = .{ .op = u.op, .value = remapValue(map, u.value) } },
            .call => |c| .{ .call = .{
                .symbol = c.symbol,
                .args = try self.remapValueList(allocator, c.args, map),
                .is_variadic = c.is_variadic,
                .num_fixed = c.num_fixed,
                .ret_dest = if (c.ret_dest) |rd| remapValue(map, rd) else null,
                .ret_regs = c.ret_regs,
                .ret_pieces = c.ret_pieces,
                .sret = c.sret,
            } },
            .call_indirect => |c| .{ .call_indirect = .{
                .target = remapValue(map, c.target),
                .args = try self.remapValueList(allocator, c.args, map),
                .is_variadic = c.is_variadic,
                .num_fixed = c.num_fixed,
                .ret_dest = if (c.ret_dest) |rd| remapValue(map, rd) else null,
                .ret_regs = c.ret_regs,
                .ret_pieces = c.ret_pieces,
                .sret = c.sret,
            } },
            .load => |ld| .{ .load = .{ .ptr = remapValue(map, ld.ptr), .@"volatile" = ld.@"volatile" } },
            .store => |st| .{ .store = .{ .value = remapValue(map, st.value), .ptr = remapValue(map, st.ptr), .@"volatile" = st.@"volatile" } },
            .prefetch => |pf| .{ .prefetch = .{ .ptr = remapValue(map, pf.ptr) } },
            .va_start => |vs| .{ .va_start = .{ .list = remapValue(map, vs.list) } },
            .va_arg => |va| .{ .va_arg = .{ .list = remapValue(map, va.list), .ty = va.ty } },
            .va_end => |ve| .{ .va_end = .{ .list = remapValue(map, ve.list) } },
            .dot => |d| .{ .dot = .{ .acc = remapValue(map, d.acc), .a = remapValue(map, d.a), .b = remapValue(map, d.b) } },
            .matmul => |mm| blk: {
                // Only a/b/c are Values. The m/n/k/dtype/accumulate/embedded/quant/input_signs are
                // compile-time metadata, copied unchanged.
                var remapped = mm;
                remapped.a = remapValue(map, mm.a);
                remapped.b = remapValue(map, mm.b);
                remapped.c = remapValue(map, mm.c);
                break :blk .{ .matmul = remapped };
            },
            .atomic_rmw => |a| blk: {
                // Only ptr/value/compare are Values. The operation, ordering and scope are
                // compile-time metadata, copied unchanged, like matmul above.
                var remapped = a;
                remapped.ptr = remapValue(map, a.ptr);
                remapped.value = remapValue(map, a.value);
                if (a.compare) |c| remapped.compare = remapValue(map, c);
                break :blk .{ .atomic_rmw = remapped };
            },
            .@"if" => |cond| .{ .@"if" = .{
                .cond = remapValue(map, cond.cond),
                .then = .{ .target = cond.then.target, .args = try self.remapValueList(allocator, cond.then.args, map) },
                .@"else" = .{ .target = cond.@"else".target, .args = try self.remapValueList(allocator, cond.@"else".args, map) },
            } },
        };
    }

    /// Remap a `ValueList`'s values through `map` (an unmapped value passes through unchanged) and
    /// intern the result as a fresh list. Used for every variadic operand `cloneBlock` touches
    /// (`struct_new`/`call`/`call_indirect` args, `if`/jump edge args).
    fn remapValueList(self: *Function, allocator: std.mem.Allocator, list: ValueList, map: *const std.AutoHashMapUnmanaged(Value, Value)) std.mem.Allocator.Error!ValueList {
        const src_vals = self.valueList(list);
        const scratch = try allocator.alloc(Value, src_vals.len);
        defer allocator.free(scratch);
        for (src_vals, 0..) |v, i| scratch[i] = remapValue(map, v);
        return self.internValues(scratch);
    }

    /// Intern a run of values into the value-list pool (for variadic operands).
    pub fn internValueList(self: *Function, vals: []const Value) std.mem.Allocator.Error!ValueList {
        return self.internValues(vals);
    }

    /// The total number of values in the function.
    pub fn valueCount(self: *const Function) usize {
        return self.values.items.len;
    }

    /// The argument values a jump (or branch edge) passes to its target's parameters.
    pub fn blockArgs(self: *const Function, jump: Jump) []const Value {
        return self.valueList(jump.args);
    }

    /// Copy values into the value-list pool, returning a handle to the run.
    pub fn internValues(self: *Function, vals: []const Value) std.mem.Allocator.Error!ValueList {
        const start: u32 = @intCast(self.value_lists.items.len);
        try self.value_lists.appendSlice(self.allocator, vals);
        return .{ .start = start, .len = @intCast(vals.len) };
    }

    /// Resolve a value-list handle to its slice.
    pub fn valueList(self: *const Function, list: ValueList) []const Value {
        return self.value_lists.items[list.start..][0..list.len];
    }

    /// Copy fp32-bit scales into the scale pool, returning a handle to the run. Used for a
    /// `matmul` quant epilogue's `per_column` scale: compile-time constant data, not Values.
    pub fn internScales(self: *Function, scales: []const u32) std.mem.Allocator.Error!ScaleList {
        const start: u32 = @intCast(self.scale_pool.items.len);
        try self.scale_pool.appendSlice(self.allocator, scales);
        return .{ .start = start, .len = @intCast(scales.len) };
    }

    /// Resolve a scale-list handle to its slice.
    pub fn scaleList(self: *const Function, list: ScaleList) []const u32 {
        return self.scale_pool.items[list.start..][0..list.len];
    }

    /// Copy int32 biases into the bias pool, returning a handle to the run. Used for a `matmul`
    /// quant epilogue's per-column bias: compile-time constant data, not Values.
    pub fn internBias(self: *Function, bias: []const i32) std.mem.Allocator.Error!BiasList {
        const start: u32 = @intCast(self.bias_pool.items.len);
        try self.bias_pool.appendSlice(self.allocator, bias);
        return .{ .start = start, .len = @intCast(bias.len) };
    }

    /// Resolve a bias-list handle to its slice.
    pub fn biasList(self: *const Function, list: BiasList) []const i32 {
        return self.bias_pool.items[list.start..][0..list.len];
    }

    /// The block's terminator, or null if it has not been set yet.
    pub fn terminator(self: *const Function, block: Block) ?Terminator {
        return self.blocks.items[@intFromEnum(block)].term;
    }

    /// The opcode of an instruction.
    pub fn opcode(self: *const Function, inst: Inst) Opcode {
        return self.insts.items[@intFromEnum(inst)].op;
    }

    /// The instruction that defines a value, or null if it is a block parameter.
    pub fn definingInst(self: *const Function, value: Value) ?Inst {
        return switch (self.values.items[@intFromEnum(value)].def) {
            .inst_result => |inst| inst,
            .block_param => null,
        };
    }

    /// Render the function in Vulcan's functional text format.
    pub fn format(self: *const Function, w: *std.Io.Writer) std.Io.Writer.Error!void {
        var func_attrs = self.attributesOf(.func);
        while (func_attrs.next()) |attr| {
            try w.writeAll("#[");
            try printAttr(w, attr);
            try w.writeAll("]\n");
        }
        // Whole-function metadata prints only when it is not at its default, so an
        // ordinary function still starts with a bare `fn {`. Each of these changes how
        // the function is called or how its symbol binds, so the text form must carry
        // them or a print/parse round trip silently makes a different function.
        try w.writeAll("fn");
        if (self.is_local) try w.writeAll(" local");
        if (self.is_variadic) try w.print(" variadic({d})", .{self.num_fixed_params});
        if (self.sret) try w.writeAll(" sret");
        try w.writeAll(" {\n");
        for (self.blocks.items, 0..) |block, bi| {
            // A block attribute prints above the label. The structured control-flow
            // lowerings key merge and continue blocks this way, so the text form must
            // carry it.
            var block_attrs = self.attributesOf(.{ .block = @enumFromInt(bi) });
            while (block_attrs.next()) |attr| {
                try w.writeAll("  #[");
                try printAttr(w, attr);
                try w.writeAll("]\n");
            }
            try w.print("  block{d}(", .{bi});
            for (block.params.items, 0..) |param, pi| {
                if (pi != 0) try w.writeAll(", ");
                try w.print("v{d}: {f}", .{ self.valueName(param), self.types.fmt(self.valueType(param)) });
                // A parameter attribute prints right after the parameter. This is where
                // the GPU path puts its `vulcan.gpu.builtin` tag, which decides whether
                // the parameter takes space in the parameter block at all.
                var param_attrs = self.attributesOf(.{ .value = param });
                while (param_attrs.next()) |attr| {
                    try w.writeAll(" #[");
                    try printAttr(w, attr);
                    try w.writeAll("]");
                }
            }
            try w.writeAll("):\n");

            for (block.insts.items) |inst| {
                // An attribute on the INSTRUCTION prints with a `#!` marker, an attribute
                // on the value it defines with a plain `#`. They are different targets, so
                // one spelling cannot stand for both.
                var inst_attrs = self.attributesOf(.{ .inst = inst });
                while (inst_attrs.next()) |attr| {
                    try w.writeAll("    #![");
                    try printAttr(w, attr);
                    try w.writeAll("]\n");
                }
                if (self.instResult(inst)) |result| {
                    var attrs = self.attributesOf(.{ .value = result });
                    while (attrs.next()) |attr| {
                        try w.writeAll("    #[");
                        try printAttr(w, attr);
                        try w.writeAll("]\n");
                    }
                }
                try w.writeAll("    ");
                try printInst(self, w, inst);
                try w.writeByte('\n');
            }

            // An unset terminator prints as an implicit `ret void`.
            try w.writeAll("    ");
            try printTerminator(self, w, block.term);
            try w.writeByte('\n');

            // Blocks that transfer control onward (`if` or `jump`) get a trailing
            // blank line. Purely returning blocks do not.
            if (self.branchesOut(block)) try w.writeByte('\n');
        }
        try w.writeAll("}");
    }

    /// Whether a block transfers control to other blocks (has a conditional or a
    /// jump terminator).
    fn branchesOut(self: *const Function, block: BlockData) bool {
        for (block.insts.items) |inst| {
            switch (self.insts.items[@intFromEnum(inst)].op) {
                .@"if" => return true,
                else => {},
            }
        }
        if (block.term) |term| switch (term) {
            .jump => return true,
            .ret => {},
        };
        return false;
    }
};

/// Look `v` up in a value remap built by `Function.cloneBlock`: a mapped value (defined inside the
/// block being cloned) resolves to its fresh clone, an unmapped value (defined outside the block)
/// passes through unchanged.
fn remapValue(map: *const std.AutoHashMapUnmanaged(Value, Value), v: Value) Value {
    return map.get(v) orelse v;
}

/// True when `ty` is f16 itself, or a vector/array/slice/struct that contains f16 anywhere in
/// its structure. Recursion always terminates: a composite can only reference an already-
/// interned element type, so the type table has no cycles.
fn typeContainsF16(table: *const TypeTable, ty: Type) bool {
    return switch (table.type_kind(ty)) {
        .float => |f| f == .f16,
        .vector => |v| typeContainsF16(table, v.elem),
        .array => |a| typeContainsF16(table, a.elem),
        .slice => |s| typeContainsF16(table, s.elem),
        .@"struct" => |fields| for (fields) |field| {
            if (typeContainsF16(table, field)) break true;
        } else false,
        .bool, .int, .ptr => false,
    };
}

/// True when `ty` is f128 itself, or a vector/array/slice/struct that contains f128 anywhere
/// in its structure. Same argument as typeContainsF16: the type table has no cycles.
fn typeContainsF128(table: *const TypeTable, ty: Type) bool {
    return switch (table.type_kind(ty)) {
        .float => |f| f == .f128,
        .vector => |v| typeContainsF128(table, v.elem),
        .array => |a| typeContainsF128(table, a.elem),
        .slice => |s| typeContainsF128(table, s.elem),
        .@"struct" => |fields| for (fields) |field| {
            if (typeContainsF128(table, field)) break true;
        } else false,
        .bool, .int, .ptr => false,
    };
}

/// True when any value in `func` (a block param or an instruction result) has a type that
/// contains f16, at any depth. No backend lowers f16 yet, and several would silently size or
/// treat it as f64 if it reached them (a miscompile), so every backend's compile/emit entry
/// calls this first and returns error.Unsupported rather than risk that. Whole-function scan
/// is conservative on purpose: rejecting an f16 value in dead code is acceptable, silently
/// mis-lowering a live one is not.
pub fn functionUsesF16(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const value: Value = @enumFromInt(@as(u32, @intCast(i)));
        if (typeContainsF16(&func.types, func.valueType(value))) return true;
    }
    return false;
}

/// True when any value of `func` is or contains f128. The backends without f128 codegen
/// reject such functions with a clear error instead of silently miscompiling them.
pub fn functionUsesF128(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const value: Value = @enumFromInt(@as(u32, @intCast(i)));
        if (typeContainsF128(&func.types, func.valueType(value))) return true;
    }
    return false;
}

/// True when any value has f16 nested inside a COMPOSITE (vector/array/slice/struct), as opposed to
/// a bare scalar f16. The scalar-f16 backends (aarch64/riscv64/wasm/x86_64) lower scalar f16 (held as
/// its f32 widening, converted at boundaries) but have NO path for f16 packed into a vector or
/// aggregate: such a value would fall through to the raw-vector/aggregate lowering and silently
/// miscompile the half lanes. Those backends call this after they stopped rejecting scalar f16, and
/// reject a composite-f16 function cleanly. No frontend produces composite f16 today (the vectorizer
/// works on f32/i32), so this never fires in practice; it is a defensive guard against a latent
/// silent miscompile, not a live limitation.
pub fn functionUsesCompositeF16(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const value: Value = @enumFromInt(@as(u32, @intCast(i)));
        const ty = func.valueType(value);
        switch (func.types.type_kind(ty)) {
            // A bare scalar float (f16/f32/f64) is handled directly; only f16 wrapped in a composite
            // is unsupported.
            .float => {},
            else => if (typeContainsF16(&func.types, ty)) return true,
        }
    }
    return false;
}

/// Render an attribute's body (the text inside the `#[...]`).
fn printAttr(w: *std.Io.Writer, attr: Attribute) std.Io.Writer.Error!void {
    switch (attr) {
        .@"inline" => try w.writeAll("inline"),
        .noreturn => try w.writeAll("noreturn"),
        .cold => try w.writeAll("cold"),
        .@"align" => |a| try w.print("align({d})", .{a}),
        .endian => |e| try w.print("endian({s})", .{@tagName(e)}),
        .custom => |c| {
            try w.print("{s}.{s}", .{ c.namespace, c.key });
            switch (c.value) {
                .flag => {},
                .int => |i| try w.print(" = {d}", .{i}),
                .string => |s| try w.print(" = \"{s}\"", .{s}),
            }
        },
    }
}

/// Render the extra fields a call carries after its argument list: the variadic marker,
/// the hidden-pointer struct return, and the register-return destination with its pieces.
/// Each part prints only when it is not at its default, so an ordinary call is unchanged.
/// A piece prints as `bank@offset:bytes`, with `i` for an integer register and `f` for a
/// floating-point one. `ret_regs` is recovered from the number of pieces, so it needs no
/// separate spelling. Only `ret_pieces[0..ret_regs]` carries meaning, so only that part
/// prints.
fn printCallExtras(
    self: *const Function,
    w: *std.Io.Writer,
    is_variadic: bool,
    num_fixed: u32,
    sret: bool,
    ret_dest: ?Value,
    ret_regs: u8,
    ret_pieces: [4]RetPiece,
) std.Io.Writer.Error!void {
    if (is_variadic) try w.print(" variadic({d})", .{num_fixed});
    if (sret) try w.writeAll(" sret");
    if (ret_dest != null or ret_regs != 0) {
        // A call carries at most 4 return-register pieces. Every construction site
        // asserts it and the bitcode decoder rejects a larger count, so the slice below
        // is in range.
        std.debug.assert(ret_regs <= ret_pieces.len);
        try w.writeAll(" retdest(");
        if (ret_dest) |d| try w.print("v{d}", .{self.valueName(d)}) else try w.writeAll("none");
        try w.writeAll(",[");
        for (ret_pieces[0..ret_regs], 0..) |p, i| {
            if (i != 0) try w.writeAll(",");
            try w.print("{s}@{d}:{d}", .{ if (p.fp) "f" else "i", p.offset, p.bytes });
        }
        try w.writeAll("])");
    }
}

/// Render an instruction statement. Constants bind with `const` and carry a type
/// annotation. Other results bind with `let`.
fn printInst(self: *const Function, w: *std.Io.Writer, inst: Inst) std.Io.Writer.Error!void {
    const data = self.insts.items[@intFromEnum(inst)];
    switch (data.op) {
        .iconst => |value| try w.print("const v{d}: {f} = {d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            value,
        }),
        .fconst => |value| try w.print("const v{d}: {f} = {d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            value,
        }),
        .fconst128 => |value| try w.print("const v{d}: {f} = {d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            @as(f128, @bitCast(value)),
        }),
        .arith => |a| try w.print("let v{d} = v{d} {s} v{d}", .{
            self.valueName(data.result.?),
            self.valueName(a.lhs),
            a.op.symbol(),
            self.valueName(a.rhs),
        }),
        .arith_imm => |a| try w.print("let v{d} = v{d} {s} {d}", .{
            self.valueName(data.result.?),
            self.valueName(a.lhs),
            a.op.symbol(),
            a.imm,
        }),
        .icmp => |cmp| try w.print("let v{d} = v{d} {s} v{d}", .{
            self.valueName(data.result.?),
            self.valueName(cmp.lhs),
            cmp.op.symbol(),
            self.valueName(cmp.rhs),
        }),
        .select => |sel| {
            try w.print("v{d} := if v{d} ", .{ self.valueName(data.result.?), self.valueName(sel.cond) });
            try w.writeAll("{ ");
            try w.print("v{d}", .{self.valueName(sel.then)});
            try w.writeAll(" } else { ");
            try w.print("v{d}", .{self.valueName(sel.@"else")});
            try w.writeAll(" }");
        },
        .struct_new => |sn| {
            try w.print("let v{d} = struct ", .{self.valueName(data.result.?)});
            try w.writeAll("{ ");
            for (self.valueList(sn.fields), 0..) |field, i| {
                if (i != 0) try w.writeAll(", ");
                try w.print("v{d}", .{self.valueName(field)});
            }
            try w.writeAll(" }");
        },
        .extract => |ex| try w.print("let v{d} = v{d}.#{d}", .{
            self.valueName(data.result.?),
            self.valueName(ex.aggregate),
            ex.index,
        }),
        .alloca => |al| try w.print("let v{d} = alloca {f}", .{
            self.valueName(data.result.?),
            self.types.fmt(al.elem),
        }),
        .global_addr => |ga| try w.print("let v{d} = global_addr{s} @{s}", .{
            self.valueName(data.result.?),
            if (ga.via_got) " got" else "",
            self.symbolName(ga.symbol),
        }),
        .call => |c| {
            if (data.result) |res| {
                try w.print("let v{d} = call {f} @{s}(", .{
                    self.valueName(res),
                    self.types.fmt(self.valueType(res)),
                    self.symbolName(c.symbol),
                });
            } else {
                try w.print("call @{s}(", .{self.symbolName(c.symbol)});
            }
            for (self.valueList(c.args), 0..) |arg, i| {
                if (i != 0) try w.writeAll(", ");
                try w.print("v{d}", .{self.valueName(arg)});
            }
            try w.writeAll(")");
            try printCallExtras(self, w, c.is_variadic, c.num_fixed, c.sret, c.ret_dest, c.ret_regs, c.ret_pieces);
        },
        .call_indirect => |c| {
            if (data.result) |res| {
                try w.print("let v{d} = call_indirect {f} v{d}(", .{ self.valueName(res), self.types.fmt(self.valueType(res)), self.valueName(c.target) });
            } else {
                try w.print("call_indirect v{d}(", .{self.valueName(c.target)});
            }
            for (self.valueList(c.args), 0..) |arg, i| {
                if (i != 0) try w.writeAll(", ");
                try w.print("v{d}", .{self.valueName(arg)});
            }
            try w.writeAll(")");
            try printCallExtras(self, w, c.is_variadic, c.num_fixed, c.sret, c.ret_dest, c.ret_regs, c.ret_pieces);
        },
        .convert => |cv| try w.print("let v{d} = convert {f}, v{d}", .{
            self.valueName(data.result.?),
            self.types.fmt(self.valueType(data.result.?)),
            self.valueName(cv.value),
        }),
        .unary => |u| try w.print("let v{d} = {s} {f}, v{d}", .{
            self.valueName(data.result.?),
            @tagName(u.op),
            self.types.fmt(self.valueType(data.result.?)),
            self.valueName(u.value),
        }),
        // `volatile` prints as a word after the mnemonic, the same shape as `global_addr
        // got`. It marks an access the optimizer must not remove, move or merge, so the
        // text form must carry it.
        .load => |ld| try w.print("let v{d} = load{s} {f}, v{d}", .{
            self.valueName(data.result.?),
            if (ld.@"volatile") " volatile" else "",
            self.types.fmt(self.valueType(data.result.?)),
            self.valueName(ld.ptr),
        }),
        .store => |st| try w.print("store{s} v{d}, v{d}", .{
            if (st.@"volatile") " volatile" else "",
            self.valueName(st.value),
            self.valueName(st.ptr),
        }),
        .prefetch => |pf| try w.print("prefetch v{d}", .{self.valueName(pf.ptr)}),
        .va_start => |vs| try w.print("va_start v{d}", .{self.valueName(vs.list)}),
        .va_arg => |va| try w.print("let v{d} = va_arg {f}, v{d}", .{
            self.valueName(data.result.?),
            self.types.fmt(va.ty),
            self.valueName(va.list),
        }),
        .va_end => |ve| try w.print("va_end v{d}", .{self.valueName(ve.list)}),
        // `atomic_rmw <op> <scope> <ordering> vPTR, vVALUE[, vCOMPARE]`, with a leading
        // `let vN = ` when the old value is read. The result type is NOT printed: it is
        // always the value operand's type, so printing it would be a second spelling of
        // one fact, and the two could disagree. The compare operand prints last and only
        // for the compare-exchange form, which is the only form that has one.
        .atomic_rmw => |a| {
            if (data.result) |res| try w.print("let v{d} = ", .{self.valueName(res)});
            try w.print("atomic_rmw {s} {s} {s} v{d}, v{d}", .{
                @tagName(a.op),
                @tagName(a.scope),
                @tagName(a.ordering),
                self.valueName(a.ptr),
                self.valueName(a.value),
            });
            if (a.compare) |c| try w.print(", v{d}", .{self.valueName(c)});
        },
        // A result-less statement, printed the same way as `prefetch` and `va_end`: the
        // mnemonic and its one operand. The operand here is the scope name, not a value.
        .barrier => |bar| try w.print("barrier {s}", .{@tagName(bar.scope)}),
        .dot => |d| try w.print("let v{d} = dot v{d}, v{d}, v{d}", .{
            self.valueName(data.result.?),
            self.valueName(d.acc),
            self.valueName(d.a),
            self.valueName(d.b),
        }),
        .matmul => |mm| {
            try w.print("matmul c=v{d}, a=v{d}, b=v{d} [{d} x {d} x {d}] {s}", .{
                self.valueName(mm.c),
                self.valueName(mm.a),
                self.valueName(mm.b),
                mm.m,
                mm.n,
                mm.k,
                @tagName(mm.dtype),
            });
            // `accumulate` picks between `c := a*b` and `c += a*b`. Two different
            // computations, so it prints.
            if (mm.accumulate) try w.writeAll(" acc");
            if (mm.embedded) try w.writeAll(" embedded");
            if (mm.input_signs) |s| {
                try w.print(" a_uns={},b_uns={}", .{ s.a_unsigned, s.b_unsigned });
            }
            if (mm.quant) |q| {
                // The per-column scales and the bias print as their VALUES, not as a
                // count. They are constant data the epilogue computes with, so a count
                // alone cannot rebuild the instruction.
                switch (q.scale) {
                    .scalar => |bits| try w.print(" quant(scalar=0x{X},relu={},{s}", .{ bits, q.relu, @tagName(q.out) }),
                    .per_column => |h| {
                        try w.writeAll(" quant(per_col[");
                        for (self.scaleList(h), 0..) |s, i| {
                            if (i != 0) try w.writeAll(",");
                            try w.print("0x{X}", .{s});
                        }
                        try w.print("],relu={},{s}", .{ q.relu, @tagName(q.out) });
                    },
                }
                if (q.bias) |bh| {
                    try w.writeAll(",bias[");
                    for (self.biasList(bh), 0..) |bv, i| {
                        if (i != 0) try w.writeAll(",");
                        try w.print("{d}", .{bv});
                    }
                    try w.writeAll("]");
                } else try w.writeAll(",bias=none");
                if (q.zero_point != 0) try w.print(",zp={d}", .{q.zero_point});
                try w.writeAll(")");
            }
        },
        .@"if" => |cond| {
            try w.print("if v{d} ", .{self.valueName(cond.cond)});
            try w.writeAll("{ ");
            try printEdge(self, w, cond.then);
            try w.writeAll(" } else { ");
            try printEdge(self, w, cond.@"else");
            try w.writeAll(" }");
        },
    }
}

/// Render a jump's target and the arguments it passes, e.g. `block1(v3, v4)`.
fn printEdge(self: *const Function, w: *std.Io.Writer, jump: Jump) std.Io.Writer.Error!void {
    try w.print("block{d}(", .{@intFromEnum(jump.target)});
    for (self.blockArgs(jump), 0..) |arg, i| {
        if (i != 0) try w.writeAll(", ");
        try w.print("v{d}", .{self.valueName(arg)});
    }
    try w.writeAll(")");
}

fn printTerminator(self: *const Function, w: *std.Io.Writer, term: ?Terminator) std.Io.Writer.Error!void {
    const t = term orelse {
        // An unset terminator is an implicit void return.
        try w.writeAll("ret void");
        return;
    };
    switch (t) {
        .ret => |r| {
            if (r.count == 0) {
                try w.writeAll("ret void");
            } else {
                try w.writeAll("ret ");
                for (r.slice(), 0..) |v, i| {
                    if (i != 0) try w.writeAll(", ");
                    try w.print("v{d}", .{self.valueName(v)});
                }
            }
        },
        .jump => |j| try printEdge(self, w, j),
    }
}

test "attributes attach to an entity and read back" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 1 });

    try func.addAttr(.{ .value = v }, .{ .@"align" = 16 });

    var it = func.attributesOf(.{ .value = v });
    try std.testing.expectEqual(Attribute{ .@"align" = 16 }, it.next().?);
    try std.testing.expectEqual(@as(?Attribute, null), it.next());
}

test "namespaced attributes carry a typed value and are owned" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    try func.addAttr(.func, .{ .custom = .{
        .namespace = "target",
        .key = "clone",
        .value = .{ .string = "rv64gcv" },
    } });

    var it = func.attributesOf(.func);
    const got = it.next().?;
    try std.testing.expectEqualStrings("target", got.custom.namespace);
    try std.testing.expectEqualStrings("clone", got.custom.key);
    try std.testing.expectEqualStrings("rv64gcv", got.custom.value.string);
}

test "a function creates distinct blocks" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const a = try func.appendBlock();
    const b = try func.appendBlock();

    try std.testing.expect(a != b);
}

test "block parameters are typed values" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const p = try func.appendBlockParam(block, i32_t);

    try std.testing.expectEqual(i32_t, func.valueType(p));
}

test "an instruction result is a typed value" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v = try func.appendInst(block, i32_t, .{ .iconst = 42 });

    try std.testing.expectEqual(i32_t, func.valueType(v));
}

test "store writes a value to a pointer" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.appendStore(entry, x, p);

    const insts = func.blockInsts(entry);
    const op = func.opcode(insts[insts.len - 1]);
    try std.testing.expectEqual(x, op.store.value);
    try std.testing.expectEqual(p, op.store.ptr);
}

test "prefetch hints an address and has no result" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    try func.appendPrefetch(entry, p);

    const insts = func.blockInsts(entry);
    const op = func.opcode(insts[insts.len - 1]);
    try std.testing.expectEqual(p, op.prefetch.ptr);
    try std.testing.expectEqual(null, func.instResult(insts[insts.len - 1]));
}

test "va_start/va_arg/va_end round-trip through print and clone" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const list = try func.appendBlockParam(entry, ptr_t);
    try func.appendVaStart(entry, list);
    const v = try func.appendVaArg(entry, list, i32_t);
    try func.appendVaEnd(entry, list);
    func.setTerminator(entry, .{ .ret = Ret.one(v) });

    const expected =
        \\fn {
        \\  block0(v0: ptr):
        \\    va_start v0
        \\    let v1 = va_arg i32, v0
        \\    va_end v0
        \\    ret v1
        \\}
    ;
    try std.testing.expectFmt(expected, "{f}", .{func});

    var cloned = try func.clone(std.testing.allocator);
    defer cloned.deinit();
    try std.testing.expectFmt(expected, "{f}", .{cloned});

    const insts = func.blockInsts(entry);
    try std.testing.expectEqual(list, func.opcode(insts[0]).va_start.list);
    try std.testing.expectEqual(list, func.opcode(insts[1]).va_arg.list);
    try std.testing.expectEqual(i32_t, func.opcode(insts[1]).va_arg.ty);
    try std.testing.expectEqual(i32_t, func.valueType(v));
    try std.testing.expectEqual(list, func.opcode(insts[2]).va_end.list);
}

test "Function.is_variadic/num_fixed_params default to non-variadic" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try std.testing.expect(!func.is_variadic);
    try std.testing.expectEqual(@as(u32, 0), func.num_fixed_params);
}

test "dot accumulates a 4-way INT8 dot-product" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v16i8 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });
    const v4i32 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });

    const entry = try func.appendBlock();
    const acc = try func.appendBlockParam(entry, v4i32);
    const a = try func.appendBlockParam(entry, v16i8);
    const b = try func.appendBlockParam(entry, v16i8);
    const result = try func.appendDot(entry, acc, a, b);

    try std.testing.expectEqual(v4i32, func.valueType(result));
    const insts = func.blockInsts(entry);
    const op = func.opcode(insts[insts.len - 1]);
    try std.testing.expectEqual(acc, op.dot.acc);
    try std.testing.expectEqual(a, op.dot.a);
    try std.testing.expectEqual(b, op.dot.b);
}

test "matmul writes c from a and b tile and has no result" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmul(entry, a, b, c, 4, 4, 4, .int8, false);

    const insts = func.blockInsts(entry);
    const op = func.opcode(insts[insts.len - 1]);
    try std.testing.expectEqual(a, op.matmul.a);
    try std.testing.expectEqual(b, op.matmul.b);
    try std.testing.expectEqual(c, op.matmul.c);
    try std.testing.expectEqual(@as(u16, 4), op.matmul.m);
    try std.testing.expectEqual(@as(u16, 4), op.matmul.n);
    try std.testing.expectEqual(@as(u16, 4), op.matmul.k);
    try std.testing.expectEqual(MatMulType.int8, op.matmul.dtype);
    try std.testing.expectEqual(false, op.matmul.accumulate);
    try std.testing.expectEqual(null, func.instResult(insts[insts.len - 1]));
}

test "struct construction prints its fields" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const s = try func.appendStructNew(entry, st, &.{ a, b });
    func.setTerminator(entry, .{ .ret = Ret.one(s) });

    try std.testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = struct { v0, v1 }
        \\    ret v2
        \\}
    , "{f}", .{func});
}

test "load reads a typed value from a pointer" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });

    try std.testing.expectEqual(i32_t, func.valueType(v));
    try std.testing.expectEqual(p, func.opcode(func.definingInst(v).?).load.ptr);
}

test "select picks between two values and is typed" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, i32_t, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } });

    try std.testing.expectEqual(i32_t, func.valueType(c));
    const op = func.opcode(func.definingInst(c).?);
    try std.testing.expectEqual(cond, op.select.cond);
    try std.testing.expectEqual(a, op.select.then);
    try std.testing.expectEqual(b, op.select.@"else");
}

test "icmp produces a bool and records its comparison" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });

    try std.testing.expectEqual(bool_t, func.valueType(c));
    const op = func.opcode(func.definingInst(c).?);
    try std.testing.expectEqual(CmpOp.gt, op.icmp.op);
    try std.testing.expectEqual(a, op.icmp.lhs);
    try std.testing.expectEqual(b, op.icmp.rhs);
}

test "arith records its operator and operands" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendInst(block, i32_t, .{ .iconst = 10 });
    const b = try func.appendInst(block, i32_t, .{ .iconst = 20 });
    const sum = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });

    const inst = func.definingInst(sum).?;
    const op = func.opcode(inst);
    try std.testing.expectEqual(BinOp.add, op.arith.op);
    try std.testing.expectEqual(a, op.arith.lhs);
    try std.testing.expectEqual(b, op.arith.rhs);
}

test "a block can be terminated with a return" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v = try func.appendInst(block, i32_t, .{ .iconst = 42 });
    func.setTerminator(block, .{ .ret = Ret.one(v) });

    const term_ret = func.terminator(block).?.ret;
    try std.testing.expectEqual(@as(u8, 1), term_ret.count);
    try std.testing.expectEqual(v, term_ret.values[0]);
}

test "a jump passes arguments to its target block params" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    const entry = try func.appendBlock();
    const target = try func.appendBlock();
    _ = try func.appendBlockParam(target, i32_t);

    const v = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    try func.setJump(entry, target, &.{v});

    const term = func.terminator(entry).?;
    try std.testing.expectEqual(target, term.jump.target);
    try std.testing.expectEqualSlices(Value, &.{v}, func.blockArgs(term.jump));
}

test "a conditional selects between two targets" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{x} }, .{ .target = else_b });

    // The conditional is an instruction in the block body, not a terminator.
    const if_inst = func.blockInsts(entry)[func.blockInsts(entry).len - 1];
    const cf = func.opcode(if_inst).@"if";
    try std.testing.expectEqual(cond, cf.cond);
    try std.testing.expectEqual(then_b, cf.then.target);
    try std.testing.expectEqual(else_b, cf.@"else".target);
    try std.testing.expectEqualSlices(Value, &.{x}, func.blockArgs(cf.then));
    try std.testing.expectEqual(@as(?Terminator, null), func.terminator(entry));
}

test "printing a minimal function" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 42 });
    func.setTerminator(entry, .{ .ret = Ret.one(v) });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    const v0: i32 = 42
        \\    ret v0
        \\}
    , "{f}", .{func});
}

test "printing a call names the callee, result type, and arguments" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const r = try func.appendCall(entry, i32_t, "add", &.{ a, b });
    func.setTerminator(entry, .{ .ret = Ret.one(r) });

    try std.testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = call i32 @add(v0, v1)
        \\    ret v2
        \\}
    , "{f}", .{func});
}

test "printing an alloca names the slot type" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    func.setTerminator(entry, .{ .ret = Ret.one(p) });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    let v0 = alloca i32
        \\    ret v0
        \\}
    , "{f}", .{func});
}

test "printing a convert names its target type and source" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const i = try func.appendInst(entry, i32_t, .{ .iconst = 3 });
    const f = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = i } });
    func.setTerminator(entry, .{ .ret = Ret.one(f) });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    const v0: i32 = 3
        \\    let v1 = convert f32, v0
        \\    ret v1
        \\}
    , "{f}", .{func});
}

test "printing a function with params, iadd, and a jump" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const exit = try func.appendBlock();
    const r = try func.appendBlockParam(exit, i32_t);

    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    try func.setJump(entry, exit, &.{sum});
    func.setTerminator(exit, .{ .ret = Ret.one(r) });

    try std.testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 + v1
        \\    block1(v2)
        \\
        \\  block1(v3: i32):
        \\    ret v3
        \\}
    , "{f}", .{func});
}

test "printing a conditional branch" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 5 });
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{x} }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = Ret.none() });
    func.setTerminator(else_b, .{ .ret = Ret.none() });

    try std.testing.expectFmt(
        \\fn {
        \\  block0():
        \\    const v0: bool = 1
        \\    const v1: i32 = 5
        \\    if v0 { block1(v1) } else { block2() }
        \\    ret void
        \\
        \\  block1():
        \\    ret void
        \\  block2():
        \\    ret void
        \\}
    , "{f}", .{func});
}

test "functionUsesF16 is false for a function with no f16 values" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    _ = try func.appendBlockParam(block, i32_t);
    _ = try func.appendInst(block, f32_t, .{ .iconst = 0 });

    try std.testing.expect(!functionUsesF16(&func));
}

test "functionUsesF16 is true for an f16 block param" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const block = try func.appendBlock();
    _ = try func.appendBlockParam(block, f16_t);

    try std.testing.expect(functionUsesF16(&func));
}

test "functionUsesF16 is true for an f16 instruction result" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const block = try func.appendBlock();
    _ = try func.appendInst(block, f16_t, .{ .iconst = 0 });

    try std.testing.expect(functionUsesF16(&func));
}

test "functionUsesF16 sees f16 nested inside a vector" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const vec_t = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f16_t } });
    const block = try func.appendBlock();
    _ = try func.appendBlockParam(block, vec_t);

    try std.testing.expect(functionUsesF16(&func));
}

test "functionUsesF16 sees f16 nested inside an array" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const arr_t = try func.types.intern(.{ .array = .{ .len = 2, .elem = f16_t } });
    const block = try func.appendBlock();
    _ = try func.appendBlockParam(block, arr_t);

    try std.testing.expect(functionUsesF16(&func));
}

test "functionUsesF16 sees f16 nested inside a struct field" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const struct_t = try func.types.intern(.{ .@"struct" = &.{ i32_t, f16_t } });
    const block = try func.appendBlock();
    _ = try func.appendBlockParam(block, struct_t);

    try std.testing.expect(functionUsesF16(&func));
}

const verify = @import("verify.zig");

test "reorderBlocks permutes a 3-block chain and remaps jump targets" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // entry(block0) -> block1 -> block2, a straight-line chain. block1 carries a
    // distinguishing const (99) so we can tell it apart after the move.
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const tail = try func.appendBlock();

    const marker = try func.appendInst(mid, i32_t, .{ .iconst = 99 });
    try func.setJump(entry, mid, &.{});
    try func.setJump(mid, tail, &.{});
    func.setTerminator(tail, .{ .ret = Ret.none() });
    _ = marker;

    // New order: entry stays first, old tail moves to index 1, old mid to index 2.
    try func.reorderBlocks(std.testing.allocator, &.{ entry, tail, mid });

    // New index 1 now holds the old tail block (empty, ret void terminator).
    const new_tail: Block = @enumFromInt(1);
    try std.testing.expectEqual(@as(usize, 0), func.blockInsts(new_tail).len);
    try std.testing.expectEqual(@as(u8, 0), func.terminator(new_tail).?.ret.count);

    // New index 2 now holds the old mid block, carrying the marker const and its jump.
    const new_mid: Block = @enumFromInt(2);
    const new_mid_insts = func.blockInsts(new_mid);
    try std.testing.expectEqual(@as(usize, 1), new_mid_insts.len);
    try std.testing.expectEqual(@as(i64, 99), func.opcode(new_mid_insts[0]).iconst);

    // The CFG edges are preserved under the new ids: entry -> new_mid (old mid is now index 2),
    // new_mid -> new_tail (old tail is now index 1).
    try std.testing.expectEqual(new_mid, func.terminator(entry).?.jump.target);
    try std.testing.expectEqual(new_tail, func.terminator(new_mid).?.jump.target);

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "reorderBlocks remaps if then/else edges" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);

    // An if-diamond: block0 -[if]-> then=block1, else=block2, both jump to merge=block3.
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
    try func.setJump(then_b, merge, &.{});
    try func.setJump(else_b, merge, &.{});
    func.setTerminator(merge, .{ .ret = Ret.none() });

    // Swap then_b and else_b's positions (and move merge before else_b).
    try func.reorderBlocks(std.testing.allocator, &.{ entry, else_b, then_b, merge });

    const new_then: Block = @enumFromInt(2); // old then_b
    const new_else: Block = @enumFromInt(1); // old else_b
    const new_merge: Block = @enumFromInt(3); // old merge, unchanged position

    const if_inst = func.blockInsts(entry)[func.blockInsts(entry).len - 1];
    const cf = func.opcode(if_inst).@"if";
    try std.testing.expectEqual(new_then, cf.then.target);
    try std.testing.expectEqual(new_else, cf.@"else".target);

    // Both original branches still land on merge, under its new id.
    try std.testing.expectEqual(new_merge, func.terminator(new_then).?.jump.target);
    try std.testing.expectEqual(new_merge, func.terminator(new_else).?.jump.target);

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "reorderBlocks identity permutation leaves the function unchanged" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = Ret.none() });
    func.setTerminator(else_b, .{ .ret = Ret.none() });

    const before = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{func});
    defer std.testing.allocator.free(before);

    try func.reorderBlocks(std.testing.allocator, &.{ entry, then_b, else_b });

    const after = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{func});
    defer std.testing.allocator.free(after);

    try std.testing.expectEqualStrings(before, after);

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "a variadic call's is_variadic/num_fixed round-trip through print and clone" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const r = try func.appendCallV(entry, i32_t, "printf", &.{ a, b }, 1);
    func.setTerminator(entry, .{ .ret = Ret.one(r) });

    // `is_variadic`/`num_fixed` carry no new print syntax, since this is plumbing only, with
    // no backend behavior yet. Printing must still exercise the arm cleanly and name the
    // callee and args exactly like a non-variadic call does.
    const printed = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{func});
    defer std.testing.allocator.free(printed);
    try std.testing.expect(std.mem.indexOf(u8, printed, "call i32 @printf(v0, v1)") != null);

    // cloneBlock (via remapOpcode) must carry the flag/count through, not silently reset
    // them to the non-variadic defaults.
    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(std.testing.allocator);
    const dst = try func.cloneBlock(std.testing.allocator, entry, &map);
    const dst_insts = func.blockInsts(dst);
    try std.testing.expectEqual(@as(usize, 1), dst_insts.len);
    const cloned_call = func.opcode(dst_insts[0]).call;
    try std.testing.expect(cloned_call.is_variadic);
    try std.testing.expectEqual(@as(u32, 1), cloned_call.num_fixed);
}

test "cloneBlock copies params and instructions with fresh values" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    // src(p): a = p + p; c = a > p; if c { then_b(a) } else { else_b }
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, i32_t);
    const a = try func.appendInst(src, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    const c = try func.appendInst(src, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = p } });
    try func.appendIf(src, c, .{ .target = then_b, .args = &.{a} }, .{ .target = else_b });

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(std.testing.allocator);
    const dst = try func.cloneBlock(std.testing.allocator, src, &map);

    // Every param and inst-result of src is mapped to a FRESH value.
    const fresh_p = map.get(p).?;
    const fresh_a = map.get(a).?;
    const fresh_c = map.get(c).?;
    try std.testing.expect(fresh_p != p);
    try std.testing.expect(fresh_a != a);
    try std.testing.expect(fresh_c != c);

    // The clone's arith references the fresh param, not the original.
    try std.testing.expectEqualSlices(Value, &.{fresh_p}, func.blockParams(dst));
    const dst_insts = func.blockInsts(dst);
    try std.testing.expectEqual(@as(usize, 3), dst_insts.len); // arith, icmp, if (a stmt too)
    const dst_arith = func.opcode(dst_insts[0]).arith;
    try std.testing.expectEqual(fresh_p, dst_arith.lhs);
    try std.testing.expectEqual(fresh_p, dst_arith.rhs);

    // The clone's if uses the fresh cond and the fresh then-edge argument.
    const dst_if_inst = func.blockInsts(dst)[func.blockInsts(dst).len - 1];
    const dst_if = func.opcode(dst_if_inst).@"if";
    try std.testing.expectEqual(fresh_c, dst_if.cond);
    try std.testing.expectEqual(then_b, dst_if.then.target); // block targets are unchanged
    try std.testing.expectEqualSlices(Value, &.{fresh_a}, func.blockArgs(dst_if.then));
    try std.testing.expectEqual(else_b, dst_if.@"else".target);

    // src is untouched: its instructions still reference the original values.
    const src_insts = func.blockInsts(src);
    try std.testing.expectEqualSlices(Value, &.{p}, func.blockParams(src));
    const src_arith = func.opcode(src_insts[0]).arith;
    try std.testing.expectEqual(p, src_arith.lhs);
    try std.testing.expectEqual(p, src_arith.rhs);
    const src_if = func.opcode(src_insts[src_insts.len - 1]).@"if";
    try std.testing.expectEqual(c, src_if.cond);
    try std.testing.expectEqualSlices(Value, &.{a}, func.blockArgs(src_if.then));
}

test "cloneBlock leaves external references unchanged" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // ext is defined in another block, not in src, so it must never appear in `map` and must
    // survive the clone identically.
    const other = try func.appendBlock();
    const ext = try func.appendInst(other, i32_t, .{ .iconst = 5 });

    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, i32_t);
    const mixed = try func.appendInst(src, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = ext } });
    func.setTerminator(src, .{ .ret = Ret.one(mixed) });

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(std.testing.allocator);
    const dst = try func.cloneBlock(std.testing.allocator, src, &map);

    try std.testing.expectEqual(@as(?Value, null), map.get(ext)); // never mapped: defined outside src

    const dst_arith = func.opcode(func.blockInsts(dst)[0]).arith;
    try std.testing.expectEqual(map.get(p).?, dst_arith.lhs); // in-src operand: remapped
    try std.testing.expectEqual(ext, dst_arith.rhs); // external operand: identical, unchanged
}

test "cloneBlock remaps a ret terminator's value" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, i32_t);
    const x = try func.appendInst(src, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    func.setTerminator(src, .{ .ret = Ret.one(x) });

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(std.testing.allocator);
    const dst = try func.cloneBlock(std.testing.allocator, src, &map);

    const dst_ret = func.terminator(dst).?.ret;
    try std.testing.expectEqual(@as(u8, 1), dst_ret.count);
    try std.testing.expectEqual(map.get(x).?, dst_ret.values[0]);
    const src_ret = func.terminator(src).?.ret; // src unchanged
    try std.testing.expectEqual(@as(u8, 1), src_ret.count);
    try std.testing.expectEqual(x, src_ret.values[0]);
}

test "cloneBlock remaps a jump terminator's args" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const target = try func.appendBlock();
    _ = try func.appendBlockParam(target, i32_t);
    func.setTerminator(target, .{ .ret = Ret.none() });

    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, i32_t);
    const x = try func.appendInst(src, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    try func.setJump(src, target, &.{x});

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(std.testing.allocator);
    const dst = try func.cloneBlock(std.testing.allocator, src, &map);

    const dst_term = func.terminator(dst).?;
    try std.testing.expectEqual(target, dst_term.jump.target); // block target is unchanged
    try std.testing.expectEqualSlices(Value, &.{map.get(x).?}, func.blockArgs(dst_term.jump));

    const src_term = func.terminator(src).?; // src unchanged
    try std.testing.expectEqualSlices(Value, &.{x}, func.blockArgs(src_term.jump));
}

test "cloneBlock produces a block verify accepts once wired into the CFG" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock(); // block 0: the entry, dominates everything below
    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, i32_t);
    const y = try func.appendInst(src, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    func.setTerminator(src, .{ .ret = Ret.one(y) });

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(std.testing.allocator);
    const dst = try func.cloneBlock(std.testing.allocator, src, &map);

    // entry conditionally jumps into either the original or the clone, both fed the same argument.
    const cond = try func.appendBlockParam(entry, bool_t);
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    try func.appendIf(entry, cond, .{ .target = src, .args = &.{v} }, .{ .target = dst, .args = &.{v} });

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

/// The int payload of the first `namespace.key` attribute on `target`, or null when absent. The
/// GPU layer has its own reader, in a module this one cannot import, so the test repeats it.
fn customIntAttr(func: *const Function, target: AttrTarget, namespace: []const u8, key: []const u8) ?i64 {
    var it = func.attributesOf(target);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, namespace)) continue;
            if (!std.mem.eql(u8, c.key, key)) continue;
            return switch (c.value) {
                .int => |n| n,
                .flag, .string => null,
            };
        },
        .@"inline", .noreturn, .cold, .@"align", .endian => {},
    };
    return null;
}

/// How many attributes sit on `target`.
fn attrCount(func: *const Function, target: AttrTarget) usize {
    var n: usize = 0;
    var it = func.attributesOf(target);
    while (it.next()) |_| n += 1;
    return n;
}

test "cloneBlock carries a parameter, result and instruction attribute onto the clone with the payload intact" {
    // A cloned block that drops its attributes changes meaning: a parameter that loses
    // `vulcan.gpu.builtin` is laid out in the parameter block instead of read from hardware,
    // and a load that loses `endian` stops being byte-swapped.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, ptr_t);
    const loaded = try func.appendInst(src, i32_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(src, loaded, p);
    const store_inst = func.blockInsts(src)[1];

    try func.addAttr(.{ .value = p }, .{ .custom = .{
        .namespace = "vulcan.gpu",
        .key = "builtin",
        .value = .{ .int = 3 },
    } });
    try func.addAttr(.{ .value = loaded }, .{ .endian = .big });
    try func.addAttr(.{ .inst = store_inst }, .{ .custom = .{
        .namespace = "debug",
        .key = "line",
        .value = .{ .int = 42 },
    } });

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(allocator);
    const dst = try func.cloneBlock(allocator, src, &map);

    // The clone's own parameter, result and instruction each carry the SAME payload.
    const new_p = func.blockParams(dst)[0];
    try std.testing.expect(new_p != p);
    try std.testing.expectEqual(@as(?i64, 3), customIntAttr(&func, .{ .value = new_p }, "vulcan.gpu", "builtin"));

    const new_loaded = map.get(loaded).?;
    var endian_it = func.attributesOf(.{ .value = new_loaded });
    try std.testing.expectEqual(Attribute{ .endian = .big }, endian_it.next().?);

    const new_store = func.blockInsts(dst)[1];
    try std.testing.expect(new_store != store_inst);
    try std.testing.expectEqual(@as(?i64, 42), customIntAttr(&func, .{ .inst = new_store }, "debug", "line"));

    // The originals keep exactly one attribute each: the copy added, it did not move.
    try std.testing.expectEqual(@as(usize, 1), attrCount(&func, .{ .value = p }));
    try std.testing.expectEqual(@as(usize, 1), attrCount(&func, .{ .inst = store_inst }));
}

test "cloneBlock does not carry a block attribute or a function attribute onto the clone" {
    // Suspicious case, the other direction. A `cf` attribute holds the merge block id of the
    // ORIGINAL region, so a second block must not claim to be that region's merge point, and a
    // function attribute already describes the function the clone sits in.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const merge = try func.appendBlock();
    const src = try func.appendBlock();
    const p = try func.appendBlockParam(src, i32_t);
    _ = try func.appendInst(src, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });

    try func.addAttr(.{ .block = src }, .{ .custom = .{
        .namespace = "cf",
        .key = "merge",
        .value = .{ .int = @intFromEnum(merge) },
    } });
    try func.addAttr(.{ .block = src }, .cold);
    try func.addAttr(.func, .@"inline");

    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(allocator);
    const dst = try func.cloneBlock(allocator, src, &map);

    // Nothing lands on the clone's block, and the source keeps both of its own.
    try std.testing.expectEqual(@as(usize, 0), attrCount(&func, .{ .block = dst }));
    try std.testing.expectEqual(@as(usize, 2), attrCount(&func, .{ .block = src }));
    try std.testing.expectEqual(@as(usize, 1), attrCount(&func, .func));
}

test "clone deep-copies a function and leaves the original untouched" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    _ = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = x } });
    const called = try func.appendCall(entry, i32_t, "callee", &.{x});
    func.setTerminator(entry, .{ .ret = Ret.one(called) });
    try func.addAttr(.{ .inst = @enumFromInt(0) }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = 12 } } });
    func.is_variadic = true;
    func.num_fixed_params = 1;
    func.sret = true;
    func.is_local = true;

    var copy = try func.clone(allocator);
    defer copy.deinit();

    // Whole-function metadata must survive. Each of these changes how the function is called
    // or how its symbol is bound, so a clone that defaults one of them back is a miscompile:
    // `is_local` in particular decides local versus global object binding.
    try std.testing.expect(copy.is_variadic);
    try std.testing.expectEqual(@as(u32, 1), copy.num_fixed_params);
    try std.testing.expect(copy.sret);
    try std.testing.expect(copy.is_local);

    // The attribute list carries over, with its string payloads re-owned by the copy.
    var it = copy.attributesOf(.{ .inst = @enumFromInt(0) });
    const attr = it.next() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("debug", attr.custom.namespace);
    try std.testing.expectEqualStrings("line", attr.custom.key);
    try std.testing.expectEqual(@as(i64, 12), attr.custom.value.int);
    try std.testing.expect(attr.custom.namespace.ptr != func.attributeEntries()[0].attr.custom.namespace.ptr);

    // Structure and interned tables match by construction (handles are index-preserved).
    try std.testing.expectEqual(func.blockCount(), copy.blockCount());
    try std.testing.expectEqual(func.valueCount(), copy.valueCount());
    try std.testing.expectEqual(func.types.count(), copy.types.count());
    try std.testing.expectEqual(func.symbolCount(), copy.symbolCount());
    try std.testing.expectEqualStrings("callee", copy.symbolName(0));

    // The copy's symbol string is INDEPENDENT storage (not aliased into the original).
    try std.testing.expect(copy.symbolName(0).ptr != func.symbolName(0).ptr);

    // Mutating the copy's CFG (edge splitting is the real use) must not touch the original.
    const before = func.blockCount();
    _ = try copy.appendBlock();
    try std.testing.expectEqual(before, func.blockCount());
    try std.testing.expectEqual(before + 1, copy.blockCount());
}

test "clone preserves handles for a function holding two pointer address spaces" {
    // The exact corruption the eql fix prevents: clone re-interns every kind in order and
    // depends on the n-th kind receiving handle n. If two address spaces collapsed to one
    // handle, every later handle would shift and values would be silently retyped.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const g = try func.types.ptrGlobal();
    const s = try func.types.intern(.{ .ptr = .shared });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const pg = try func.appendBlockParam(b, g);
    const ps = try func.appendBlockParam(b, s);
    const n = try func.appendBlockParam(b, i32_t);
    func.setTerminator(b, .{ .ret = Ret.one(n) });

    var copy = try func.clone(allocator);
    defer copy.deinit();

    try std.testing.expectEqual(
        types.AddressSpace.global,
        copy.types.type_kind(copy.valueType(pg)).ptr,
    );
    try std.testing.expectEqual(
        types.AddressSpace.shared,
        copy.types.type_kind(copy.valueType(ps)).ptr,
    );
    try std.testing.expectEqual(@as(u16, 32), copy.types.type_kind(copy.valueType(n)).int.bits);

    // The other direction of the whole-function metadata copy: this function sets none of the
    // flags, so the clone must set none either. A clone that forces one on would, for
    // `is_local`, hide an external symbol from cross-object resolution.
    try std.testing.expect(!copy.is_variadic);
    try std.testing.expectEqual(@as(u32, 0), copy.num_fixed_params);
    try std.testing.expect(!copy.sret);
    try std.testing.expect(!copy.is_local);
}
