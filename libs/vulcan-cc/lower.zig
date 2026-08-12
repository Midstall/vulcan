//! Lower a parsed C translation unit to Vulcan IR. Multiple functions with typed
//! parameters and locals (the A1 style: store each in an entry-block alloca, and read it
//! back with a `load`, at the variable's declared-type width). Each function lowers a
//! statement-list body: declarations, assignment and compound-assignment statements,
//! `{ }` compound blocks with lexical scoping, and `return`. Lowering threads a mutable
//! `L` context (a current-block cursor, a scoped env, and the target layout), so
//! control-flow lowering can move the cursor and open or close nested scopes, and typed
//! lowering can compute IR types and insert C's implicit conversions.
//!
//! `lowerExpr` returns a `TypedValue` (an IR `Value` plus its C type), not a bare
//! `Value`. Every arm computes its result `CType` per C's rules: integer promotion, and
//! the usual arithmetic conversions via `ctype.CType.commonType`. `L.convertTo` inserts
//! an IR `convert` wherever an operand's representation must change to match another
//! operand's, or the target's, for assignment, return, or initialization. For an
//! all-`int` program, every `commonType`/`convertTo` resolves `int` to `int` (same width
//! and signedness), so `convertTo` returns the operand unchanged and emits no `convert`.
//! This keeps the IR the same as an earlier, untyped lowering pass produced for an
//! all-`int` program. `.call` looks up the callee's parsed signature (this unit's own
//! `parser.Func`s, threaded into `L.funcs`), so each argument converts to the callee's
//! declared parameter type, and the call's result carries the callee's actual return
//! type, rather than assuming `int` for both.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const parser = @import("parser.zig");
const preproc = @import("preproc.zig");
const layout = @import("layout.zig");
const ctype = @import("ctype.zig");
const consteval = @import("consteval.zig");
const abi = @import("abi.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

/// A lowered expression's IR value together with its C type. `lowerExpr` returns this
/// instead of a bare `Value` so every caller knows the result's width and signedness
/// without re-deriving it. This is needed to pick the right `commonType`/`convertTo` at
/// each use site. `quals` is meaningful ONLY when this `TypedValue` came from `lowerAddr`
/// (an LVALUE): the lvalue's own qualifiers, computed per-arm (see `lowerAddr`'s doc), so a
/// write through it (`.assign`/`.incdec`) can reject a `const` target. `lowerExpr`'s own
/// (rvalue) results leave this at the default `.{}`. Qualification is not a property of a
/// value, only of the object an lvalue names.
pub const TypedValue = struct {
    value: Value,
    ty: ctype.CType,
    quals: ctype.Quals = .{},
    /// A BITFIELD lvalue. Non-null ONLY on an lvalue from `lowerAddr`'s `.member` arm for
    /// a bitfield member. A bitfield has no byte address of its own, so `value` is its
    /// STORAGE UNIT's address (`ty` is the declaring integer type), and this records where
    /// the field's bits sit within that unit. Every read (`lowerExpr`'s `.member`) and
    /// every write (`.assign`/`.incdec`) checks this and takes the shift/mask (read) or
    /// read-modify-write (write) path instead of a plain load/store.
    bitfield: ?Bitfield = null,

    pub const Bitfield = struct { bit_offset: u32, bit_width: u32 };
};

test "lower and JIT-run a constant return" {
    const allocator = std.testing.allocator;
    var mod = try compile(allocator, "int f(void){ return -7; }");
    defer mod.deinit(allocator);
    const f_ir = mod.find("f") orelse return error.NoFunc;
    var jitted = try target.native.jitModule(allocator, &.{.{ .name = "f", .func = f_ir }});
    defer jitted.deinit();
    const f = jitted.entry(*const fn () callconv(.c) i32, "f") orelse return error.NoEntry;
    try std.testing.expectEqual(@as(i32, -7), f());
}

/// A lowered, named function: its link name and its IR. Caller owns both.
pub const NamedFunction = struct { name: []u8, func: Function };

/// Which linker section a `DataObject` belongs in. Mirrors `backend.link.DataKind` (every
/// target backend's own copy of this same three-way enum), so `Module.data` can be handed to
/// `target.native.jitModuleData` (or a future non-JIT linker) with a one-line same-named-
/// variant mapping, without this frontend depending on any particular backend's `link.zig`.
pub const DataKind = enum { rodata, data, bss };

/// An internal relocation WITHIN a `DataObject`'s own bytes (an address-constant initializer,
/// `&g` or a string literal, folds to one of these via `consteval`): at byte offset `off`,
/// patch in the runtime address of another data object or function named `symbol` once the
/// module is linked or mapped. Mirrors `backend.link.DataReloc`.
pub const DataReloc = struct { off: usize, symbol: []const u8 };

/// A lowered module-level data object backing one global variable: its link name, which
/// section it belongs in, its initial bytes (empty for `.bss`. `size` gives the zero-filled
/// length instead), and any internal relocations. Caller owns `name` (see `Module.deinit`).
/// `bytes`/`relocs` are only owned when non-empty (a `.bss` object carries neither).
pub const DataObject = struct { name: []const u8, kind: DataKind, bytes: []const u8, size: u64, relocs: []const DataReloc = &.{} };

/// A lowered translation unit: every function it defines, plus every global variable's
/// backing `DataObject`. Caller owns the whole thing.
pub const Module = struct {
    funcs: []NamedFunction,
    data: []DataObject,
    /// Owned link-symbol names for an `extern` declaration this TU never defines: registered
    /// in the `GlobalTable` (so a reference to it still lowers, resolving to
    /// `error.UndefinedSymbol` only once the JIT tries to link it, see `compile`'s doc), but
    /// backing no `DataObject`, so nothing else owns or frees this name. It is tracked here
    /// purely so `deinit` can free it instead of leaking it. It stays empty whenever every
    /// `extern` in the TU has a definition (the common case. That definition's own
    /// `DataObject.name` IS the symbol).
    extern_syms: [][]const u8 = &.{},

    /// Free every function's owned name and IR, every data object's owned memory, every
    /// undefined-`extern` symbol name, then every list itself.
    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        for (self.funcs) |*nf| {
            nf.func.deinit();
            allocator.free(nf.name);
        }
        allocator.free(self.funcs);
        self.funcs = &.{};
        for (self.data) |*d| {
            allocator.free(d.name);
            if (d.bytes.len != 0) allocator.free(d.bytes);
            if (d.relocs.len != 0) allocator.free(d.relocs);
        }
        allocator.free(self.data);
        self.data = &.{};
        for (self.extern_syms) |s| allocator.free(s);
        allocator.free(self.extern_syms);
        self.extern_syms = &.{};
    }

    /// Find a function's IR by its link name, or `null` if the module has none by that
    /// name.
    pub fn find(self: *const Module, name: []const u8) ?*const Function {
        for (self.funcs) |*nf| if (std.mem.eql(u8, nf.name, name)) return &nf.func;
        return null;
    }
};

/// Where a `Binding`'s address comes from: an ordinary local or param's entry-block alloca
/// pointer (`.slot`, a `Value` already valid in every block since entry dominates all of
/// them), or a `static` local's link symbol (`.global`, resolved fresh at each use via
/// `appendGlobalAddr`, since that op, unlike an alloca, must be emitted into the CURRENT
/// block).
const BindingAddr = union(enum) { slot: Value, global: []const u8 };

/// A binding: an in-scope name bound to its address (an alloca slot, or for a `static`
/// local a data object's symbol, see `BindingAddr`) and its declared C type (which fixes
/// the element width and every load/store through it). Bindings are pushed as scopes open
/// and popped as they close. `lookup` scans back-to-front, so the newest (innermost)
/// binding for a name wins (correct C shadowing). `quals` is the bound OBJECT's own
/// qualifiers (from `parser.Param.quals`/`Stmt.decl.quals`).
const Binding = struct { name: []const u8, ty: ctype.CType, addr: BindingAddr, quals: ctype.Quals = .{} };

/// A resolved global's declared type and link symbol (which `.name` resolution needs to
/// emit a `global_addr`). The map value of `GlobalTable`. `quals` is the global OBJECT's
/// own qualifiers (from `parser.GlobalDecl.quals`). `no_tu_def` is true only for a name
/// every decl of which is a pure `extern` declaration (`isPureExternDecl`, no definition
/// anywhere in this TU. The same condition `compile`'s PASS 1 uses to route the name to
/// `extern_syms` instead of a `DataObject`): the object lives in ANOTHER translation unit
/// or a `.so`, so `resolveName` must resolve its address GOT-indirectly
/// (`appendGlobalAddrGot`) rather than directly (`appendGlobalAddr`). A direct
/// `global_addr` assumes the symbol is resolvable at ordinary link time within the same
/// image, which an imported `.so` data object is not (see the GOT-indirect data import
/// path). A name this TU itself defines (even one also declared `extern` elsewhere) leaves
/// this `false` and keeps the existing direct `global_addr`, unchanged from before.
const GlobalInfo = struct { ty: ctype.CType, symbol: []const u8, quals: ctype.Quals = .{}, no_tu_def: bool = false };

/// Every global variable this translation unit defines, name -> its type/symbol, built once
/// in `compile` from `unit.globals` and threaded (by pointer) into every `lowerFunction`
/// call's `L`. `.name` resolution (`resolveName`/`resolveNameType`) consults this only after
/// the local/param env misses, a local always shadows a global of the same name.
pub const GlobalTable = std.StringHashMapUnmanaged(GlobalInfo);

/// The enclosing loop (or switch) for break/continue. `brk` is the exit block (loop exit,
/// or switch exit). `cont` is where `continue` jumps (the header for while/do, the
/// increment block for for. Unused for a switch frame, `continue` skips those, see
/// `.continue_` below). `is_switch` marks a switch frame: `break` targets the nearest
/// frame regardless, but `continue` must skip switch frames to reach the enclosing loop
/// (a `continue` inside a `switch` inside a loop continues the LOOP, not the switch).
const LoopCtx = struct { brk: Block, cont: Block, is_switch: bool = false };

/// One `goto`-reachable label within the current function: its name and the IR block
/// `collectLabels` pre-allocated for it. A label has FUNCTION scope in C: a `goto`
/// anywhere in the function, even earlier in the source than the label itself (a forward
/// jump), can target it. So every label is collected up front, before any statement is
/// lowered.
const LabelBlock = struct { name: []const u8, block: Block };

/// The mutable lowering context for one function. `block` is the current insertion point;
/// control-flow lowering creates blocks and moves this cursor.
const L = struct {
    func: *Function,
    /// The function entry block. ALL allocas go here (not the current block): an alloca in a
    /// loop body would re-run every iteration (unbounded stack growth), and mem2reg only
    /// promotes entry-block allocas. Entry dominates every block, so an entry alloca dominates
    /// all its uses.
    entry: Block,
    block: Block,
    env: std.ArrayList(Binding),
    loops: std.ArrayList(LoopCtx),
    /// Every label this function declares, collected up front by `collectLabels` before
    /// any statement lowers. A function has few labels, so a linear scan (`labelBlock`) is
    /// plenty. This matches the codebase's ArrayList-not-hashmap style for small
    /// per-function tables (see `loops` above).
    labels: std.ArrayList(LabelBlock),
    allocator: std.mem.Allocator,
    /// The target layout: resolves `long`'s width and drives every `CType.sizeInBytes`/`irType`.
    layout: layout.TargetLayout,
    /// This function's declared return type. `.ret` converts to it, and the fall-off-the-end
    /// seal (and any dead-block seals) return a zero of it.
    ret_ty: ctype.CType,
    /// The IR type for plain `int` (`ctype.int_t.irType(func, layout)`, interned once). A
    /// convenience for the many places that need `int`'s width specifically (int literals,
    /// the logical/`!`/relational-result type, switch case-label constants' fallback).
    i32t: Type,
    ptrt: Type,
    boolt: Type,
    /// The IR type for a genuine 64-bit int, interned once: field offsets and the
    /// whole-struct-copy word loop both need this EXACT width regardless of target layout.
    /// Unlike `ctype.long_t` (which is only 64-bit on LP64 targets, 32-bit on ILP32), the
    /// struct storage blob `CType.irType` builds is ALWAYS a blob of 64-bit words, so indexing
    /// into it must match that fixed width, not whatever `long` happens to be.
    i64t: Type,
    /// Every function this translation unit defines (parsed, not yet lowered), so `.call`
    /// can look up the CALLEE's declared parameter/return types and convert args/result
    /// correctly instead of assuming `int`. There are only a handful of functions per TU,
    /// so a linear scan per call site is plenty.
    funcs: []const parser.Func,
    /// Every bodyless function DECLARATION this translation unit carries: a prototype
    /// (`int f(int);`) or an `extern` one (`extern int f(int);`), consulted by `.call` ONLY
    /// when `funcs` has no DEFINITION of the callee. The function is defined in another
    /// object or `.so`, and this call lowers to an undefined-symbol call the linker resolves
    /// (a `.so` PLT import). A locally-DEFINED function always wins (checked first), so this
    /// never shadows a same-TU definition.
    func_decls: []const parser.FuncDecl,
    /// Every global variable this translation unit defines, consulted by `.name` resolution
    /// whenever the local/param env has no binding for the name (see
    /// `resolveName`/`resolveNameType`).
    globals: *const GlobalTable,
    /// This function's own link name, so a `static` local can mint a symbol unique across
    /// the whole module (`<fn_name>.<var_name>.<counter>`).
    fn_name: []const u8,
    /// The module's data-object accumulator (owned by `compile`, shared by every function's
    /// `lowerFunction` call), so a `static` local's initializer can be appended alongside
    /// file-scope globals. `data.items.len` at mint time also doubles as the uniqueness
    /// counter: globals are appended before any function is lowered, and every later append
    /// strictly grows it, so no two symbols minted this way ever collide.
    data: *std.ArrayList(DataObject),
    /// The name of the current function's LAST fixed (named) parameter, when it is a
    /// variadic DEFINITION, else null for a non-variadic function. C11 7.15.1's
    /// `va_start(ap, last)` requires `last` to name that exact parameter (the one right
    /// before `...`). `.call`'s `__builtin_va_start` dispatch checks its second argument
    /// against this.
    last_fixed_param: ?[]const u8 = null,
    /// The current function's HIDDEN result pointer, when it returns a struct over 16 bytes
    /// (`abi.classify`'s `.sret` plan), else null for every other function. It is the
    /// function's FIRST entry-block parameter (prepended before the real parameters), and the
    /// `.ret` arm copies the return value through it and then returns the same pointer. See
    /// `Function.sret`.
    sret_ptr: ?Value = null,
    /// Backing storage for `CType`s synthesized DURING lowering (currently only `ptrTo`'s
    /// pointer wrapper for `&expr`'s result type). These have no parser AST node to own
    /// them, unlike every other `CType` here (which lives in the parser's arena). A
    /// dedicated arena, owned by `lowerFunction` and freed when this function is done
    /// lowering. Only O(pointer-depth) of these are synthesized per `&`, so there is no
    /// growth concern.
    type_arena: std.mem.Allocator,

    fn deinit(self: *L) void {
        self.env.deinit(self.allocator);
        self.loops.deinit(self.allocator);
        self.labels.deinit(self.allocator);
    }
    /// Arena-allocate a `CType` and wrap it in `.ptr`: the type of `&expr` (a NEW pointer
    /// type that doesn't already exist as a parser AST node, unlike a declared `int *p`'s
    /// type, which the declarator built). `quals` becomes the new pointer's POINTEE
    /// qualifiers: `&const_x` must yield `const int *` (an addrof site passes the lvalue's
    /// own `quals`), while a plain array or string decay to pointer passes the array's
    /// ELEMENT `quals` (so indexing back through the decayed pointer still sees the
    /// element's own qualification). Both call sites pass whatever is appropriate. This
    /// helper itself has no opinion, it just threads the value through.
    fn ptrTo(self: *L, pointee: ctype.CType, quals: ctype.Quals) Error!ctype.CType {
        const p = try self.type_arena.create(ctype.CType);
        p.* = pointee;
        return .{ .ptr = .{ .pointee = p, .quals = quals } };
    }
    /// Mark the current scope depth. Restore it with popScope to drop this scope's bindings.
    fn scopeMark(self: *L) usize {
        return self.env.items.len;
    }
    fn popScope(self: *L, mark: usize) void {
        self.env.shrinkRetainingCapacity(mark);
    }
    /// The binding for `name` (its slot and declared type), newest first, or null if undefined.
    fn lookup(self: *L, name: []const u8) ?Binding {
        var i = self.env.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.env.items[i].name, name)) return self.env.items[i];
        }
        return null;
    }
    /// Allocate a fresh stack slot of `ct`'s width in the ENTRY block (see `entry` doc).
    /// Appending an alloca to entry is valid even after entry already has a REAL terminator
    /// (`ret`/`jump`, e.g. From a `while`/`for` loop's own header jump), because a block's
    /// instruction list and its terminator are separate fields (`insts` vs `term`), the
    /// alloca is added to `insts`, the terminator is untouched.
    ///
    /// It is NOT valid to append after entry already holds an `if` INSTRUCTION, though (this
    /// was traced back to a switch-case bare-block-local crash): `if` is not a `Terminator`
    /// (`ir.function.If`'s own doc calls it "a non-terminating instruction" in the high
    /// profile). It is a plain entry in `insts` that every backend nonetheless treats as
    /// ending that block's LIVE code (aarch64/x86/riscv64 isel all set a `terminated` flag on
    /// it and stop walking predecessor-relevant state at that point). An instruction appended
    /// AFTER it is dead code that never executes. So a later local's alloca landing there
    /// (a `switch` case body's own `{ }` block, or equally a sibling declaration after a
    /// top-level `if`-statement, since both start their branch by calling `appendIf` directly
    /// on `self.block`, which is `self.entry` when they are the function's first statement)
    /// leaves that local's address never computed: any later load/store through it reads or
    /// writes a garbage register. The fix: when entry already carries an `if`, INSERT the new
    /// alloca just before it instead of appending after it. `if` stays the true last live
    /// instruction, and the alloca (no operands, so free to move) still dominates every use.
    /// Every existing program (entry with no `if` yet at alloc time) still appends exactly as
    /// before, unchanged.
    fn allocSlot(self: *L, ct: ctype.CType) Error!Value {
        const elem = try self.irTy(ct);
        const insts = self.func.blockInstsMut(self.entry);
        for (insts.items, 0..) |inst, i| {
            if (self.func.opcode(inst) == .@"if") {
                const v = try self.func.createInst(self.ptrt, .{ .alloca = .{ .elem = elem } });
                const new_inst = self.func.definingInst(v) orelse unreachable; // just created above
                try insts.insert(self.allocator, i, new_inst);
                return v;
            }
        }
        return self.func.appendInst(self.entry, self.ptrt, .{ .alloca = .{ .elem = elem } });
    }
    /// The IR type for `ct` under this function's target layout (int -> `Int`, ptr -> `ptr`,
    /// array -> IR `array`. Interning is idempotent, so repeated calls for the same `CType`
    /// return the same `Type` handle).
    fn irTy(self: *L, ct: ctype.CType) Error!Type {
        return ct.irType(self.func, self.layout);
    }
    /// Convert a typed value to `target`'s representation, emitting an IR `convert` when the
    /// width or signedness differs (int-int truncate/extend), the kind differs between int
    /// and float (int<->float and float<->float, e.g. `int` -> `double` on assignment, or
    /// `float` -> `double` widening), or the kind differs between int and pointer (a cast's
    /// `(long)p`/`(int*)n`, C11 6.3.2.3). Same width and signedness (int-int), or same
    /// `FloatKind` (float-float), is a no-op: it returns `tv.value` unchanged, emitting
    /// nothing. This is the behavior-preserving case for every all-`int` (or every
    /// all-same-float-width) program. Pointer-to-pointer (same `ptr_bits` width always) is
    /// also a no-op. Int<->pointer always emits a `.convert` (even same-width ptr<->long on
    /// this LP64 host): the source is a genuinely different IR type (`ptr` vs `Int`) even
    /// when the bit width matches, unlike the int-int same-width case, which is already the
    /// identical `Int` type and truly needs nothing emitted. A `.convert` between
    /// `gpr`-class operands (every backend puts `ptr` in the same register class as an
    /// integer) is exactly the int<->int truncate/sign-extend it already emits for int-int,
    /// so no new codegen shape is needed. See aarch64 isel's `.convert` case, whose
    /// `intBitsOf`/`isSignedInt` both default to 64/signed for a `ptr` operand or result.
    /// Anything that crosses into `array`/`struct` is still `error.Unsupported`. Those have
    /// no scalar representation to convert to or from at all.
    fn convertTo(self: *L, tv: TypedValue, target_ty: ctype.CType) Error!Value {
        // C99 `_Bool` (C11 6.3.1.2): converting ANY scalar TO `_Bool` is a compare-nonzero,
        // not a truncation. `_Bool b = 5;` gives `b == 1`, not `b == 5 & 0xff`. Checked
        // FIRST, ahead of every other arm below (int-int, float, pointer-int), since none of
        // those truncating/widening paths is the right one when the TARGET is `_Bool`. Only
        // the source's own kind matters for how the zero compares. A `_Bool` source is
        // already 0/1. Running it back through `truthy` is a harmless no-op (still 0/1 out),
        // not worth a separate fast path.
        if (target_ty.isBool()) {
            if (!(tv.ty.isInt() or tv.ty.isFloat() or tv.ty == .ptr)) return error.Unsupported;
            const cmp = try truthy(self, tv);
            const tt = try self.irTy(target_ty);
            return self.func.appendInst(self.block, tt, .{ .convert = .{ .value = cmp } });
        }
        if (tv.ty.isFloat() or target_ty.isFloat()) {
            if (!(tv.ty.isInt() or tv.ty.isFloat()) or !(target_ty.isInt() or target_ty.isFloat())) return error.Unsupported;
            if (tv.ty.eql(target_ty)) return tv.value; // same FloatKind: no-op
            const tt = try self.irTy(target_ty);
            return self.func.appendInst(self.block, tt, .{ .convert = .{ .value = tv.value } });
        }
        if (tv.ty.asInt()) |a| {
            if (target_ty == .ptr) { // int -> pointer, e.g. `(int*)n`
                const tt = try self.irTy(target_ty);
                return self.func.appendInst(self.block, tt, .{ .convert = .{ .value = tv.value } });
            }
            const b = target_ty.asInt() orelse return error.Unsupported;
            if (a.bits(self.layout) == b.bits(self.layout) and a.signed == b.signed) return tv.value;
            const tt = try self.irTy(target_ty);
            return self.func.appendInst(self.block, tt, .{ .convert = .{ .value = tv.value } });
        }
        if (tv.ty == .ptr) {
            if (target_ty == .ptr) return tv.value; // ptr-ptr: always same width, no-op
            if (target_ty.asInt() != null) { // pointer -> int, e.g. `(long)p`
                const tt = try self.irTy(target_ty);
                return self.func.appendInst(self.block, tt, .{ .convert = .{ .value = tv.value } });
            }
            return error.Unsupported;
        }
        return error.Unsupported;
    }
};

/// `ConstAssign`: a write (`.assign`, `.incdec`, compound-assign) whose destination lvalue
/// is `const`-qualified. See `lowerAddr`'s per-arm `quals` computation and the check at each
/// write site in `lowerExpr`.
pub const Error = parser.Error || error{ Unsupported, ConstAssign, UndeclaredCall, CallArityMismatch, BitfieldAddress } || std.mem.Allocator.Error;

/// Whether every byte of `bytes` is zero. `compile`'s section routing folds a `.data`
/// initializer that happens to evaluate to all-zero (and carries no reloc) down to `.bss`
/// instead, same as an absent initializer.
fn allZeroBytes(bytes: []const u8) bool {
    for (bytes) |b| if (b != 0) return false;
    return true;
}

/// Route a constant-folded initializer to a linker section: `const`-qualified goes to
/// `.rodata`. Otherwise an all-zero result with no relocation goes to `.bss` (same as no
/// initializer at all). Anything else goes to `.data`. Shared by `compile` (file-scope
/// globals) and `lowerStmt`'s `.decl` arm (`static` locals). Both fold an initializer to
/// bytes via `consteval.evalScalarInit` and route it through this exact same three-way rule.
fn sectionFor(is_const: bool, bytes: []const u8, relocs: []const DataReloc) DataKind {
    if (is_const) return .rodata;
    if (allZeroBytes(bytes) and relocs.len == 0) return .bss;
    return .data;
}

/// Whether `g` is a pure `extern` DECLARATION (`extern` storage with NO initializer), which
/// registers no `DataObject` of its own: the defining (non-`extern`, or `extern` with an
/// initializer, though that is nonstandard C) decl for the same name, wherever it appears in
/// the TU, supplies the one and only object. Every OTHER shape (`.none`/`.static`, or any
/// decl carrying an initializer) is a definition.
fn isPureExternDecl(g: parser.GlobalDecl) bool {
    return g.storage == .extern_ and g.init == null;
}

/// Compile C `source` to a multi-function IR `Module`. Parameters, return, and locals carry
/// their declared C type. For an all-`int` program every type is `int` (i32). Every
/// file-scope global becomes a `DataObject` in `Module.data`, routed by its initializer: no
/// initializer goes to zero-initialized `.bss`. `const`-qualified with a constant
/// initializer goes to `.rodata`. A constant initializer that folds to all-zero bytes (and
/// no reloc) goes to `.bss` (same as no initializer). Anything else goes to `.data`. A
/// non-constant initializer (`consteval.eval` hitting a `.name`/`.call`/...) surfaces as
/// `error.Unsupported`, never a runtime store. Each function is lowered with a
/// `GlobalTable`, so a `.name` reference to `g` resolves to it whenever no local/param
/// shadows it.
///
/// `extern`: `unit.globals` may list the SAME name more than once, a pure `extern`
/// declaration (`isPureExternDecl`) alongside its (possibly later, possibly earlier)
/// defining decl elsewhere in the TU. Every unique name is registered in `globals` and
/// emits AT MOST ONE `DataObject`, via TWO passes over `unit.globals`:
///   - PASS 1 registers every unique name into `globals` (name -> {ty, symbol}, dedup'ing an
///     `extern` against its definition) and, for a name the TU DEFINES, appends a placeholder
///     `.bss` `DataObject` (right size, no bytes yet) that PASS 2 fills in. A name with NO
///     defining decl anywhere in the TU is registered too (so a reference to it still lowers
///     to a `global_addr`), but backs no `DataObject`. `Module.extern_syms` owns its symbol
///     instead. Unresolved at link time gives `error.UndefinedSymbol` from the JIT, not a
///     compile-time failure.
///   - PASS 2, with the table now COMPLETE, folds each defining decl's initializer
///     (`consteval.evalInit`, which consults `globals` so an address-constant `&<any global>`
///     resolves REGARDLESS of source order: `int *p = &g; int g = 42;` works as well as the
///     reverse) and overwrites its placeholder with the routed bytes, kind, and relocs. A
///     non-constant initializer propagates `error.Unsupported` straight out.
/// Two passes (rather than register-and-emit in one) is what makes a forward `&g` reference
/// resolve. Functions, lowered only after BOTH passes, always see the complete table anyway.
pub fn compile(allocator: std.mem.Allocator, source: []const u8) Error!Module {
    return compileWithOpts(allocator, source, .{});
}

/// Like `compile`, but with an explicit preprocessor configuration, e.g. `-D` defines or an
/// `IncludeResolver` for `#include`. `compile` is the plain-C89 shorthand that delegates
/// here with `.{}`.
pub fn compileWithOpts(allocator: std.mem.Allocator, source: []const u8, pp: preproc.Options) Error!Module {
    return compileForTarget(allocator, source, layout.host(), pp);
}

/// Like `compileWithOpts`, but builds the frontend types from an EXPLICIT target layout instead
/// of the build host's. The one part of the frontend that varies by ARCH (not just width) is
/// `__builtin_va_list`, whose byte shape is a per-target ABI fact (see `ctype.builtinVaList`).
/// A `va_list` handed to real glibc `vsnprintf` must match the TARGET arch's ABI, so cross
/// tests reach the frontend through here with each arch's `layout.forArch(...)`.
/// `compile`/`compileWithOpts` delegate here with `layout.host()`, so every existing caller
/// produces the same output as before.
pub fn compileForTarget(allocator: std.mem.Allocator, source: []const u8, lay: layout.TargetLayout, pp: preproc.Options) Error!Module {
    var unit = try parser.parse(allocator, source, lay, pp);
    defer unit.deinit();

    var globals: GlobalTable = .empty;
    defer globals.deinit(allocator);

    var data: std.ArrayList(DataObject) = .empty;
    errdefer {
        for (data.items) |*d| {
            allocator.free(d.name);
            if (d.bytes.len != 0) allocator.free(d.bytes);
            if (d.relocs.len != 0) allocator.free(d.relocs);
        }
        data.deinit(allocator);
    }
    var extern_syms: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (extern_syms.items) |s| allocator.free(s);
        extern_syms.deinit(allocator);
    }
    // PASS 2's worklist: each DEFINED global's decl paired with the index of its placeholder
    // `DataObject` in `data`. Pre-reserved to `unit.globals.len` (an upper bound on unique
    // defined names), so its `appendAssumeCapacity` below can NEVER fail. That keeps PASS 1's
    // per-name error cleanup single-owner: once a name is handed to `data`/`extern_syms`, no
    // fallible op follows it in the same iteration, so exactly one owner ever frees it.
    var defs: std.ArrayList(struct { decl: parser.GlobalDecl, index: usize }) = .empty;
    defer defs.deinit(allocator);
    try defs.ensureTotalCapacity(allocator, unit.globals.len);
    try extern_syms.ensureTotalCapacity(allocator, unit.globals.len);

    // PASS 1: register names + placeholders.
    for (unit.globals) |g| {
        if (globals.contains(g.name)) continue; // an earlier decl of the same name already registered it
        // Find THE defining decl for this name anywhere in the TU (may be `g` itself, may be
        // earlier or later in the list), or `null` if every decl sharing this name is a pure
        // `extern` declaration. A REAL definition (non-null initializer) is preferred over a
        // TENTATIVE one (`int g;` with no init, non-extern): C allows `int g; int g = 5;` (in
        // EITHER order). The two name the same object, defined with value 5. So this must
        // select the initialized decl, or PASS 2 would emit `.bss` zero and drop the `= 5`
        // (a silent miscompile). A tentative decl is only kept when NO initialized one exists
        // (then the object correctly stays zero-initialized `.bss`). Two initialized decls for
        // one name is a C multiple-definition error. This takes the first (last-wins would be
        // equally arbitrary) rather than diagnosing it.
        var def: ?parser.GlobalDecl = null;
        for (unit.globals) |g2| {
            if (!std.mem.eql(u8, g2.name, g.name) or isPureExternDecl(g2)) continue;
            if (g2.init != null) {
                def = g2; // a real definition wins outright
                break;
            }
            if (def == null) def = g2; // remember a tentative, but keep scanning for a real def
        }
        // `name` backs BOTH whichever container owns it below (the placeholder `DataObject`,
        // or, when there is no definition anywhere, `extern_syms`) and the `GlobalTable` key
        // (never freed by `globals.deinit`. `StringHashMapUnmanaged` doesn't own key memory).
        // The `errdefer` covers the window until that owning append succeeds. Ordering the
        // owning append LAST (and the `defs`/`extern_syms` reservations above making the
        // trailing `appendAssumeCapacity` infallible) means a single owner frees `name` on any
        // error.
        const name = try allocator.dupe(u8, g.name);
        errdefer allocator.free(name);
        if (def) |dg| {
            try globals.put(allocator, name, .{ .ty = dg.ty, .symbol = name, .quals = dg.quals });
            const index = data.items.len;
            // A placeholder zero-initialized `.bss`, sized by its STORAGE stride (not the exact
            // C size, matters for aggregates, see `ctype.CType.storageSize`). PASS 2 overwrites
            // it in place if the initializer routes to `.data`/`.rodata`. `name` is now owned by
            // this `DataObject` (the `data` errdefer frees it hereafter).
            try data.append(allocator, .{ .name = name, .kind = .bss, .bytes = &.{}, .size = try dg.ty.storageSize(lay) });
            defs.appendAssumeCapacity(.{ .decl = dg, .index = index }); // infallible (reserved)
        } else {
            // No definition anywhere in the TU: `no_tu_def = true` so `resolveName` resolves
            // this name's address GOT-indirectly instead of directly.
            try globals.put(allocator, name, .{ .ty = g.ty, .symbol = name, .quals = g.quals, .no_tu_def = true });
            extern_syms.appendAssumeCapacity(name); // infallible (reserved); now owns `name`
        }
    }

    // PASS 2: fold each defining decl's initializer against the now-complete `globals` table
    // and overwrite its placeholder. A decl with no initializer keeps its PASS-1 `.bss`
    // placeholder untouched.
    for (defs.items) |entry| {
        const dg = entry.decl;
        const init = dg.init orelse continue;
        // `&globals` resolves an address-constant `&<any global>` (now the WHOLE table, so
        // source order no longer matters). `&data` mints an anonymous `.rodata` object for a
        // `char*`'s string-literal initializer. A non-constant initializer propagates
        // `consteval.eval`'s `error.Unsupported` straight out (the `try`), before the
        // placeholder is touched.
        const c = try consteval.evalInit(allocator, init, dg.ty, lay, &globals, &data);
        const kind = sectionFor(dg.is_const, c.bytes, c.relocs);
        if (kind == .bss) {
            // Folds to all-zero, no reloc: the `.bss` placeholder already models it (correct
            // size, no bytes), just release the folded bytes and leave the placeholder as-is.
            allocator.free(c.bytes);
            continue;
        }
        // `data.items[entry.index]` is stable across PASS 2's own minted-string appends (they
        // only grow `data`. Existing indices don't move), and read AFTER `evalInit` returns.
        const d = &data.items[entry.index];
        d.kind = kind;
        d.bytes = c.bytes;
        d.size = c.bytes.len;
        d.relocs = c.relocs;
    }

    var funcs: std.ArrayList(NamedFunction) = .empty;
    errdefer {
        for (funcs.items) |*nf| {
            nf.func.deinit();
            allocator.free(nf.name);
        }
        funcs.deinit(allocator);
    }
    for (unit.funcs) |fn_ast| {
        const name = try allocator.dupe(u8, fn_ast.name);
        errdefer allocator.free(name);
        var func = try lowerFunction(allocator, fn_ast, lay, unit.funcs, unit.func_decls, &globals, &data);
        errdefer func.deinit();
        try funcs.append(allocator, .{ .name = name, .func = func });
    }
    return .{ .funcs = try funcs.toOwnedSlice(allocator), .data = try data.toOwnedSlice(allocator), .extern_syms = try extern_syms.toOwnedSlice(allocator) };
}

/// Bind a struct/union-BY-VALUE parameter `p`: declare its ABI-classified scalar pieces as
/// entry block params (one per `abi.classify` eightbyte, in eightbyte order), store each
/// into a callee-owned slot, and bind `p.name` to that slot. This is the same "spill to an
/// alloca, bind the alloca" shape the ordinary scalar-param path below already uses, so
/// every later `.name`/`.member`/`&`/by-value-copy use of `p` inside the body sees an
/// ordinary local struct. This is what makes the copy a REAL C by-value copy: the callee's
/// slot is its own, distinct from whatever the caller passed the eightbytes from, so a
/// mutation through `p` inside the callee never touches the caller's object.
///
/// The `.registers` (`.integer` AND `.sse`) and `.memory_ref` (an oversized aggregate passed
/// by reference) plans are implemented, as is `.memory_stack` (i386's always-memory
/// convention). Every other shape fails closed with `error.Unsupported` rather than silently
/// mis-binding the parameter.
fn bindStructParam(l: *L, p: parser.Param) Error!void {
    const plan = abi.classify(p.ty, l.layout).arg;
    switch (plan) {
        .registers => |ebs| {
            const slot = try l.allocSlot(p.ty);
            for (ebs.slice()) |eb| {
                // An `.sse` eightbyte declares a FLOAT-typed entry param (an `f32` for a
                // lone/packed-pair `float` eightbyte, an `f64` for a `double` one) instead of
                // the `.integer` arm's `i64`. The backend already routes a float-typed param
                // to the next FP argument register positionally, so no backend change is
                // needed. Only the IR TYPE differs between the two arms.
                const pt = if (eb.class == .sse) try floatEightbyteType(l, eb) else l.i64t;
                const pv = try l.func.appendBlockParam(l.entry, pt);
                const dst = try byteOffset(l, slot, eb.offset);
                try l.func.appendStoreVol(l.block, pv, dst, false);
            }
            try l.env.append(l.allocator, .{ .name = p.name, .ty = p.ty, .addr = .{ .slot = slot }, .quals = p.quals });
        },
        // An oversized (more than 16 byte) aggregate no longer fits 1-2 integer registers,
        // so the ABI passes it BY REFERENCE. The caller already made a copy (see
        // `appendStructArg`'s `.memory_ref` arm) and passed a single pointer to it. The
        // callee still copies from that pointer into its OWN slot, exactly like the
        // `.registers` arm above does from its register-spilled eightbytes, so `p` is an
        // ordinary local struct either way, and a mutation through it can never reach the
        // caller's copy (which is already distinct from the caller's ORIGINAL object).
        .memory_ref => {
            const pv = try l.func.appendBlockParam(l.entry, l.ptrt);
            const slot = try l.allocSlot(p.ty);
            try copyStruct(l, p.ty.@"struct", pv, slot);
            try l.env.append(l.allocator, .{ .name = p.name, .ty = p.ty, .addr = .{ .slot = slot }, .quals = p.quals });
        },
        // A `.memory_stack` struct has NO classifier eightbyte list, so home it from
        // `ceil(size/word)` consecutive integer entry params into a fresh callee-owned slot.
        // This is the same "spill each incoming chunk, bind the slot" shape as the `.registers`
        // arm, just over whole-struct chunks instead of the classifier's eightbytes. The chunk is
        // one general register wide (8 bytes on a 64-bit target, 4 bytes on i386 whose backend is
        // integer-only and rejects a 64-bit chunk). This mirrors `appendStructArg`'s `.memory_stack`
        // arm word-for-word, so the caller's chunk order and the callee's param order agree. The
        // slot is the callee's own copy, so a mutation through `p` never reaches the caller.
        .memory_stack => {
            const slot = try l.allocSlot(p.ty);
            const size = p.ty.sizeInBytes(l.layout) catch return error.Unsupported;
            const word: u64 = l.layout.ptr_bits / 8;
            const wt = if (word == 8) l.i64t else l.i32t;
            const n = (size + word - 1) / word;
            var i: u64 = 0;
            while (i < n) : (i += 1) {
                const pv = try l.func.appendBlockParam(l.entry, wt);
                const dst = try byteOffset(l, slot, i * word);
                try l.func.appendStoreVol(l.block, pv, dst, false);
            }
            try l.env.append(l.allocator, .{ .name = p.name, .ty = p.ty, .addr = .{ .slot = slot }, .quals = p.quals });
        },
    }
}

/// Lower a single parsed function: bind each parameter to an entry-block alloca at the
/// parameter's declared width (the A1 style, mirroring the wasm frontend), then fold over its
/// statement-list body (locals get their own alloca at their declared width, expression
/// statements lower for effect, `return` converts to the function's return type and
/// terminates). A fall-off-the-end is sealed with `return 0` of the function's return type.
/// This is UB in C, but it keeps the IR valid so control-flow lowering, where not every path
/// need explicitly return, stays well-formed.
fn lowerFunction(allocator: std.mem.Allocator, fn_ast: parser.Func, lay: layout.TargetLayout, all_funcs: []const parser.Func, all_func_decls: []const parser.FuncDecl, globals: *const GlobalTable, data: *std.ArrayList(DataObject)) Error!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();

    // A variadic DEFINITION marks the IR function itself (distinct from `Call.is_variadic`,
    // which marks a CALL SITE, see `Function.is_variadic`'s doc). Every fixed (named)
    // parameter is still bound the ordinary way below. `num_fixed_params` is simply how many
    // of them there are.
    func.is_variadic = fn_ast.is_variadic;
    func.is_local = fn_ast.is_static; // A `static` function gets a LOCAL object symbol.
    func.num_fixed_params = if (fn_ast.is_variadic) @intCast(fn_ast.params.len) else 0;

    const i32t = try ctype.int_t.irType(&func, lay);
    const ptrt = try func.types.intern(.ptr);
    const boolt = try func.types.intern(.bool);
    const i64t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();

    var type_arena = std.heap.ArenaAllocator.init(allocator);
    defer type_arena.deinit();

    // C11 7.15.1: `va_start(ap, last)` requires `last` to name the LAST fixed parameter (the
    // one right before `...`). A variadic prototype with zero named parameters
    // (`int h(...);`) is already rejected as `error.Unsupported`, so a variadic definition
    // always has at least one.
    const last_fixed_param: ?[]const u8 = if (fn_ast.is_variadic and fn_ast.params.len > 0)
        fn_ast.params[fn_ast.params.len - 1].name
    else
        null;

    var l = L{ .func = &func, .entry = entry, .block = entry, .env = .empty, .loops = .empty, .labels = .empty, .allocator = allocator, .layout = lay, .ret_ty = fn_ast.ret, .i32t = i32t, .ptrt = ptrt, .boolt = boolt, .i64t = i64t, .funcs = all_funcs, .func_decls = all_func_decls, .globals = globals, .fn_name = fn_ast.name, .data = data, .last_fixed_param = last_fixed_param, .type_arena = type_arena.allocator() };
    defer l.deinit();

    // A function that returns a struct over 16 bytes (the `.sret` plan) receives a HIDDEN
    // result pointer as its first entry-block parameter. The caller allocates the
    // destination slot and passes its address there. The `.ret` arm copies the return value
    // through this pointer and returns the same address. It is prepended BEFORE the real
    // parameters (below), so every real parameter becomes block param 1+, and the backend homes
    // the hidden pointer positionally (x8 on aarch64, the first integer arg register on the
    // others). A variadic struct-returning function is extremely rare and its `num_fixed_params`
    // interaction is untested, so it fails closed.
    if (fn_ast.ret == .@"struct" and abi.classify(fn_ast.ret, lay).ret == .sret) {
        if (fn_ast.is_variadic) return error.Unsupported;
        func.sret = true;
        l.sret_ptr = try func.appendBlockParam(entry, ptrt);
    }

    for (fn_ast.params) |p| {
        // A struct/union-BY-VALUE parameter takes a SEPARATE path: it declares 1+ SCALAR
        // entry params (its ABI-classified eightbytes) rather than one param at its own
        // (aggregate) IR type, since the backend only knows how to place scalar args. Every
        // other parameter is otherwise ordinary.
        if (p.ty == .@"struct") {
            try bindStructParam(&l, p);
            continue;
        }
        const pt = try l.irTy(p.ty);
        const pv = try func.appendBlockParam(entry, pt);
        const slot = try l.allocSlot(p.ty);
        // Spilling a `volatile`-qualified parameter into its own slot is itself a write to
        // that volatile object, so it's volatile too (`p.quals` is the param's own
        // qualifiers, mirroring `resolveName`'s `bd.quals` for a plain local).
        try func.appendStoreVol(l.block, pv, slot, p.quals.is_volatile);
        try l.env.append(allocator, .{ .name = p.name, .ty = p.ty, .addr = .{ .slot = slot }, .quals = p.quals });
    }
    // A label has FUNCTION scope, so every label's block is pre-allocated BEFORE any
    // statement lowers. This is what lets a `goto` reach a label that appears later in the
    // source (a forward jump).
    try collectLabels(&l, fn_ast.body);
    const terminated = try lowerBlock(&l, fn_ast.body);
    if (!terminated) try sealRet(&l, l.block, l.ret_ty); // fall-off-the-end (UB in C; keeps IR valid)
    return func;
}

/// Walk `body` RECURSIVELY and pre-allocate an IR block for every `.label`, so `lowerBlock`
/// can wire a `goto` to it regardless of source order. Recurses into every statement shape
/// that holds sub-statements: `.block`, `.if_` (then and else), `.while_`, `.for_` (body,
/// and the init statement if present), `.do_`, `.switch_` (each case body), and a `.label`'s
/// own body (a label may itself prefix another label). A duplicate label name within one
/// function is ambiguous, so it fails closed.
fn collectLabels(l: *L, body: []const parser.Stmt) Error!void {
    for (body) |stmt| {
        switch (stmt) {
            .label => |lb| {
                if (labelBlock(l, lb.name) != null) return error.Unsupported; // duplicate label
                const blk = try l.func.appendBlock();
                try l.labels.append(l.allocator, .{ .name = lb.name, .block = blk });
                try collectLabels(l, @as(*const [1]parser.Stmt, lb.body)[0..1]);
            },
            .block => |b| try collectLabels(l, b),
            .if_ => |iff| {
                try collectLabels(l, iff.then);
                try collectLabels(l, iff.els);
            },
            .while_ => |w| try collectLabels(l, w.body),
            .for_ => |ff| {
                if (ff.init) |s| try collectLabels(l, @as(*const [1]parser.Stmt, s)[0..1]);
                try collectLabels(l, ff.body);
            },
            .do_ => |d| try collectLabels(l, d.body),
            .switch_ => |sw| for (sw.cases) |c| try collectLabels(l, c.body),
            // A multi-declarator group holds only `.decl`s, no labels can hide inside it.
            .decl_group, .decl, .expr, .ret, .break_, .continue_, .goto_ => {},
        }
    }
}

/// The block `collectLabels` pre-allocated for label `name`, or null if this function has
/// no such label.
fn labelBlock(l: *L, name: []const u8) ?Block {
    for (l.labels.items) |lb| if (std.mem.eql(u8, lb.name, name)) return lb.block;
    return null;
}

/// Whether any statement in `body` is a `.label`. Only the TOP LEVEL of `body` is checked
/// (not recursive), matching `lowerBlock`'s use: it decides whether dead code right here, in
/// this same statement list, can still be revived by a later label in it.
/// Lower a statement list in its own scope. Returns whether control was terminated (a
/// return/break/continue, or an if/switch whose arms all terminated), i.e. The current
/// block already has a terminator and no fall-through successor exists.
///
/// When this function has NO labels at all (`l.labels.items.len == 0`), the loop below
/// behaves exactly as it did before labels existed: a program with no labels lowers exactly
/// as before. The label-aware loop only runs for a function that declares at least one
/// label.
fn lowerBlock(l: *L, body: []const parser.Stmt) Error!bool {
    const mark = l.scopeMark();
    defer l.popScope(mark);
    if (l.labels.items.len == 0) {
        for (body) |stmt| {
            if (try lowerStmt(l, stmt)) return true; // rest is unreachable
        }
        return false;
    }
    var terminated = false;
    for (body) |stmt| {
        if (stmt == .label) {
            const lbl = labelBlock(l, stmt.label.name) orelse return error.Unsupported;
            if (!terminated) try l.func.setJump(l.block, lbl, &.{}); // fall-through into the label
            l.block = lbl;
            terminated = try lowerStmt(l, stmt.label.body.*);
            continue;
        }
        if (terminated) {
            // Dead code after a goto/return, in a function that has labels. A later label
            // can be nested arbitrarily deep inside this dead statement (`goto x; { x: ...; }`),
            // so this code cannot decide the dead run is truly unreachable by a shallow scan.
            // It always lowers the dead code into a fresh, otherwise unreferenced block. The
            // block stays unreachable at run time. This keeps every pre-allocated label block
            // reached and terminated, so the IR is well formed. A function with NO labels
            // already takes the early `l.labels.items.len == 0` path and skips this.
            l.block = try l.func.appendBlock();
            terminated = false;
        }
        terminated = try lowerStmt(l, stmt);
    }
    return terminated;
}

/// Whether `ty` is an array with a `len == 0` (unsized) dimension ANYWHERE in its nesting -
/// e.g. `int[2][0-sentinel]` as well as the plain `int[0-sentinel]`, used by `lowerStmt`'s
/// `.decl` arm to reject an unsized LOCAL array at any dimension, not just the outermost.
fn hasUnsizedDim(ty: ctype.CType) bool {
    return switch (ty) {
        .array => |a| a.len == 0 or hasUnsizedDim(a.elem.*),
        else => false,
    };
}

/// Lower one statement. Returns whether it terminated the current path.
fn lowerStmt(l: *L, stmt: parser.Stmt) Error!bool {
    switch (stmt) {
        .decl => |d| {
            // An unsized array dimension (`int a[];`, or `int a[][3]`, parsed with `len == 0`
            // as the "unsized" sentinel at whichever dimension it appears) is only meaningful
            // as a param declarator, which `parseParams` already decays to a pointer before it
            // ever reaches here, a LOCAL with no size at ANY dimension has no sensible storage
            // to reserve (invalid C), so fail closed rather than alloca 0 bytes. Checked
            // recursively (not just the outermost dimension) since a multi-dim local like
            // `int a[2][]` is exactly as invalid as `int a[]`.
            if (hasUnsizedDim(d.ty)) return error.Unsupported;
            // `static int n = 0;` persists across calls, so it is NOT a stack alloca at
            // all. It is a uniquely-named `DataObject` (its initializer, a compile-time
            // constant per `consteval`, baked into the object's bytes ONCE, never re-stored
            // at runtime), and the local NAME binds to that symbol (`BindingAddr.global`),
            // resolved through `appendGlobalAddr` exactly like a file-scope global. See
            // `resolveName`. The mint-a-unique-symbol, fold-initializer, section-routing
            // shape mirrors `compile`'s file-scope globals loop. `data.items.len` (already
            // past every file-scope global by the time any function is lowered, and
            // strictly growing after) doubles as the uniqueness counter, so two same-named
            // statics, even in different scopes of the same function, never collide.
            if (d.storage == .static) {
                const sym = try std.fmt.allocPrint(l.allocator, "{s}.{s}.{d}", .{ l.fn_name, d.name, l.data.items.len });
                errdefer l.allocator.free(sym);
                // A `static` local's initializer is a full `Initializer` (a brace-list or
                // string aggregate, same as a global's), so it folds through
                // `consteval.evalInit` exactly like `compile`'s file-scope globals loop.
                // `l.globals`/`l.data` likewise let an address-constant initializer
                // (`&<file-scope global>`, a string literal) resolve or mint here too.
                const obj: DataObject = if (d.init) |init| blk: {
                    const c = try consteval.evalInit(l.allocator, init, d.ty, l.layout, l.globals, l.data);
                    const kind = sectionFor(d.is_const, c.bytes, c.relocs);
                    if (kind == .bss) {
                        l.allocator.free(c.bytes);
                        break :blk .{ .name = sym, .kind = .bss, .bytes = &.{}, .size = try d.ty.storageSize(l.layout) };
                    }
                    break :blk .{ .name = sym, .kind = kind, .bytes = c.bytes, .size = c.bytes.len, .relocs = c.relocs };
                } else .{ .name = sym, .kind = .bss, .bytes = &.{}, .size = try d.ty.storageSize(l.layout) };
                try l.data.append(l.allocator, obj);
                try l.env.append(l.allocator, .{ .name = d.name, .ty = d.ty, .addr = .{ .global = sym }, .quals = d.quals });
                return false;
            }
            // Allocate the slot BEFORE lowering the initializer (like logand/logor/ternary's
            // own internal temp slots): `allocSlot` always targets `l.entry`, and if the
            // initializer is itself a branching construct evaluated while `l.block == l.entry`
            // (e.g. This decl is the first branch in the function), lowering it appends an
            // `if`/`jump` to entry BEFORE this call, allocating afterward would then append
            // the new alloca instruction AFTER that control transfer within entry's own
            // instruction list, corrupting the block (a branch must be the last thing in its
            // block). Allocating first, as every other allocSlot call site already does, avoids
            // the hazard entirely. It doesn't change name visibility since the binding itself
            // is still pushed to `env` only after both alloc and lowering, same as before.
            const slot = try l.allocSlot(d.ty);
            if (d.init) |init| {
                switch (init.value) {
                    // A bare scalar/pointer initializer, lowered as a single runtime store.
                    // One special case: `struct S s = (struct S){10,32};`/
                    // `int a[3] = (int[3]){1,2,3};` folds the compound literal's OWN
                    // brace-list directly into `s`'s slot, rather than materializing a
                    // separate unnamed temp and then copying it. This frontend has no
                    // general struct-by-value copy in this path, so this is how a compound
                    // literal reaches an AGGREGATE-typed decl at all. It is observably
                    // identical to C's "create unnamed object, then copy" semantics. A
                    // SCALAR decl (`int x = (int){5};`) still falls to the generic path
                    // below, unchanged: `lowerExpr` materializes the literal into its own
                    // temp and loads it, and `convertTo` then stores that value into `x`.
                    .expr => |e| {
                        if (e.* == .compound_literal and (d.ty == .array or d.ty == .@"struct")) {
                            try lowerAggregateInit(l, slot, d.ty, e.compound_literal.init);
                        } else if (d.ty == .@"struct") {
                            // `struct S q = <struct rvalue>;` (e.g. `= mk()` or `= p`). A
                            // struct initializer has no scalar rvalue, so resolve its
                            // ADDRESS (`lowerAddr`, the same model whole-struct assignment
                            // uses) and copy it word-for-word into the new local, C's
                            // copy-init.
                            const src = try lowerAddr(l, e);
                            if (src.ty != .@"struct" or src.ty.@"struct" != d.ty.@"struct") return error.Unsupported;
                            try copyStruct(l, d.ty.@"struct", src.value, slot);
                        } else {
                            const init_v = try lowerExpr(l, e);
                            const conv = try l.convertTo(init_v, d.ty);
                            // Initializing a `volatile`-qualified local is itself a write to
                            // that volatile object (`d.quals` is the local's own
                            // qualifiers).
                            try l.func.appendStoreVol(l.block, conv, slot, d.quals.is_volatile);
                        }
                    },
                    // A brace-list AGGREGATE initializer: a `static` local (above) or a
                    // file-scope global (`compile`) folds this at COMPILE time via
                    // `consteval`, but an ordinary automatic local has no home for the
                    // folded bytes to live in until the function actually runs, so it needs
                    // a runtime element-wise store sequence instead: `lowerAggregateInit`.
                    .list => try lowerAggregateInit(l, slot, d.ty, init),
                }
            }
            try l.env.append(l.allocator, .{ .name = d.name, .ty = d.ty, .addr = .{ .slot = slot }, .quals = d.quals });
            return false;
        },
        .expr => |e| {
            _ = try lowerExpr(l, e);
            return false;
        },
        .ret => |maybe_e| {
            // A valueless `return;` (`maybe_e == null`) seals with the same helper the
            // fall-off-the-end and dead-block paths already use. It emits a valueless `ret`
            // for `void` and an indeterminate zero for any other return type, matching gcc.
            const e = maybe_e orelse {
                try sealRet(l, l.block, l.ret_ty);
                return true;
            };
            // A struct returned BY VALUE has no scalar rvalue, so `lowerExpr` cannot lower
            // it. Resolve the return value's own ADDRESS (`lowerAddr`, the same model
            // whole-struct assignment uses), then return its ABI eightbytes through the
            // multi-value `Ret.many`. The backend places them into the return register pair.
            if (l.ret_ty == .@"struct") {
                // A `.sret` struct (over 16 bytes) has no return-register value. Copy the
                // return value THROUGH the hidden result pointer, then return that same
                // pointer (the ABI echoes it in the first return register, placed by the
                // count-1 `.ret` path). The `.registers` case below is the register-pair path.
                if (abi.classify(l.ret_ty, l.layout).ret == .sret) {
                    const aa = try lowerAddr(l, e);
                    try copyStruct(l, l.ret_ty.@"struct", aa.value, l.sret_ptr.?);
                    l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(l.sret_ptr.?) });
                    return true;
                }
                // Load each eightbyte at its own CLASS and width: an integer eightbyte as
                // `i64` (its offset is `index*8`), an `.sse` eightbyte as `f32`/`f64` at its
                // own byte offset (a RISC-V `{float; float;}` packs its two floats at
                // offsets 0 and 4, not 0 and 8). `Ret.many` returns the typed mix, and the
                // backend routes each value to the return register of its own bank by the
                // value's type.
                const ret_ebs = switch (abi.classify(l.ret_ty, l.layout).ret) {
                    .registers => |ebs| ebs,
                    .sret => unreachable, // dispatched above
                };
                if (l.layout.arch == .x86) return error.Unsupported; // i386 never returns in registers
                const aa = try lowerAddr(l, e);
                var vals: [4]Value = undefined;
                for (ret_ebs.slice(), 0..) |eb, i| {
                    const p = try byteOffset(l, aa.value, eb.offset);
                    const pt = if (eb.class == .sse) try floatEightbyteType(l, eb) else l.i64t;
                    vals[i] = try l.func.appendInst(l.block, pt, .{ .load = .{ .ptr = p, .@"volatile" = false } });
                }
                l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.many(vals[0..ret_ebs.count]) });
                return true;
            }
            const v = try lowerExpr(l, e);
            const rv = try l.convertTo(v, l.ret_ty);
            l.func.setTerminator(l.block, .{ .ret = ir.function.Ret.one(rv) });
            return true;
        },
        .block => |b| return lowerBlock(l, b),
        // A multi-declarator group lowers each `.decl` in the CURRENT scope, opening no
        // new one (unlike `.block`). A declaration never terminates the block.
        .decl_group => |stmts| {
            for (stmts) |s| _ = try lowerStmt(l, s);
            return false;
        },
        .if_ => |iff| {
            const cond_b = try truthy(l, try lowerExpr(l, iff.cond));
            const has_else = iff.els.len > 0;
            const then_b = try l.func.appendBlock();
            const else_b = try l.func.appendBlock();
            const merge_b = try l.func.appendBlock();
            try l.func.appendIf(l.block, cond_b, .{ .target = then_b }, .{ .target = if (has_else) else_b else merge_b });

            l.block = then_b;
            const then_term = try lowerBlock(l, iff.then);
            if (!then_term) try l.func.setJump(l.block, merge_b, &.{});

            var else_term = false;
            if (has_else) {
                l.block = else_b;
                else_term = try lowerBlock(l, iff.els);
                if (!else_term) try l.func.setJump(l.block, merge_b, &.{});
            } else {
                // else_b is unused. Seal it so it is a valid (dead) block, ret type matching
                // the function's return type (fconst 0.0 for a float return type, a valueless
                // ret for a void return type).
                try sealRet(l, else_b, l.ret_ty);
            }

            l.block = merge_b;
            const both_terminate = has_else and then_term and else_term;
            if (both_terminate) try sealRet(l, merge_b, l.ret_ty); // merge is unreachable; seal it
            return both_terminate;
        },
        .while_ => |w| {
            const header_b = try l.func.appendBlock();
            const body_b = try l.func.appendBlock();
            const exit_b = try l.func.appendBlock();
            try l.func.setJump(l.block, header_b, &.{});

            l.block = header_b;
            const cond_b = try truthy(l, try lowerExpr(l, w.cond));
            try l.func.appendIf(l.block, cond_b, .{ .target = body_b }, .{ .target = exit_b });

            try l.loops.append(l.allocator, .{ .brk = exit_b, .cont = header_b });
            l.block = body_b;
            const body_term = try lowerBlock(l, w.body);
            if (!body_term) try l.func.setJump(l.block, header_b, &.{}); // back-edge
            _ = l.loops.pop();

            l.block = exit_b;
            return false; // a while may exit (cond false / break), so control continues
        },
        .for_ => |ff| {
            const mark = l.scopeMark(); // the init var is scoped to the loop
            defer l.popScope(mark);
            if (ff.init) |s| _ = try lowerStmt(l, s.*);

            const header_b = try l.func.appendBlock();
            const body_b = try l.func.appendBlock();
            const incr_b = try l.func.appendBlock();
            const exit_b = try l.func.appendBlock();
            try l.func.setJump(l.block, header_b, &.{});

            l.block = header_b;
            if (ff.cond) |c| {
                const cond_b = try truthy(l, try lowerExpr(l, c));
                try l.func.appendIf(l.block, cond_b, .{ .target = body_b }, .{ .target = exit_b });
            } else {
                try l.func.setJump(header_b, body_b, &.{}); // no cond = infinite (needs break)
            }

            try l.loops.append(l.allocator, .{ .brk = exit_b, .cont = incr_b });
            l.block = body_b;
            const body_term = try lowerBlock(l, ff.body);
            if (!body_term) try l.func.setJump(l.block, incr_b, &.{});
            _ = l.loops.pop();

            l.block = incr_b;
            if (ff.incr) |e| _ = try lowerExpr(l, e);
            try l.func.setJump(incr_b, header_b, &.{});

            l.block = exit_b;
            return false;
        },
        .break_ => {
            const lc = if (l.loops.items.len == 0) return error.Unsupported else l.loops.items[l.loops.items.len - 1];
            try l.func.setJump(l.block, lc.brk, &.{});
            return true;
        },
        .continue_ => {
            var i = l.loops.items.len;
            const cont_b = while (i > 0) : (i -= 1) {
                if (!l.loops.items[i - 1].is_switch) break l.loops.items[i - 1].cont;
            } else return error.Unsupported;
            try l.func.setJump(l.block, cont_b, &.{});
            return true;
        },
        .do_ => |d| {
            const body_b = try l.func.appendBlock();
            const cond_b = try l.func.appendBlock();
            const exit_b = try l.func.appendBlock();
            try l.func.setJump(l.block, body_b, &.{});

            try l.loops.append(l.allocator, .{ .brk = exit_b, .cont = cond_b });
            l.block = body_b;
            const body_term = try lowerBlock(l, d.body);
            if (!body_term) try l.func.setJump(l.block, cond_b, &.{});
            _ = l.loops.pop();

            l.block = cond_b;
            const cb = try truthy(l, try lowerExpr(l, d.cond));
            try l.func.appendIf(l.block, cb, .{ .target = body_b }, .{ .target = exit_b });

            l.block = exit_b;
            return false;
        },
        .switch_ => |sw| {
            const v = try lowerExpr(l, sw.value);
            const vt = try l.irTy(v.ty);
            const exit_b = try l.func.appendBlock();

            // One block per case body, in source order.
            const body_blocks = try l.allocator.alloc(Block, sw.cases.len);
            defer l.allocator.free(body_blocks);
            for (body_blocks) |*bb| bb.* = try l.func.appendBlock();

            // Dispatch ladder: from the current block, chain equality tests. Each test block
            // branches to its case body or to the next test. Default (or exit) catches the rest.
            var default_target: Block = exit_b;
            for (sw.cases, 0..) |c, di| if (c.label == null) {
                default_target = body_blocks[di];
            };

            // A run of source-consecutive cases whose values ascend by exactly 1 and whose
            // intermediate bodies are all empty (so intra-run C fall-through is a no-op) is
            // equivalent, for dispatch, to a single value range: matching any value in
            // `[lo, hi]` and entering the run's FIRST body block reproduces matching that exact
            // value, because every body up to the matched one is empty and falls through. A run
            // of `range_cluster_min` or more cases dispatches with one `lo <= v <= hi` range test
            // (two compares) instead of that many equality tests, which shrinks a dense switch
            // like the C-locale `c_isalnum` (62 cases) from a 62-long compare ladder to three
            // range checks, the same shape gcc emits. The compares use the switch value's own
            // signedness (its promoted type), which is the correct membership test for a signed
            // or an unsigned controlling expression alike. A shorter run keeps the per-case
            // equality ladder unchanged, so every switch without such a run stays byte-identical.
            const range_cluster_min = 4;
            var test_block = l.block;
            var i: usize = 0;
            while (i < sw.cases.len) {
                if (sw.cases[i].label == null) {
                    i += 1; // The default arm is not a dispatch test, it is the fall-off target.
                    continue;
                }
                // Grow the run [i .. j]: extend past case `j` only when `j`'s body is empty and
                // the next case's value is exactly one greater (and is itself a real case label).
                var j = i;
                while (j + 1 < sw.cases.len and
                    sw.cases[j].body.len == 0 and
                    sw.cases[j].label.? != std.math.maxInt(i64) and // guards the `+ 1` below from overflow
                    sw.cases[j + 1].label != null and
                    sw.cases[j + 1].label.? == sw.cases[j].label.? + 1) : (j += 1)
                {}
                if (j - i + 1 >= range_cluster_min) {
                    const lo = sw.cases[i].label.?;
                    const hi = sw.cases[j].label.?;
                    const lo_c = try l.func.appendInst(test_block, vt, .{ .iconst = lo });
                    const ge = try l.func.appendInst(test_block, l.boolt, .{ .icmp = .{ .op = .ge, .lhs = v.value, .rhs = lo_c } });
                    const hi_block = try l.func.appendBlock();
                    const next_test = try l.func.appendBlock();
                    try l.func.appendIf(test_block, ge, .{ .target = hi_block }, .{ .target = next_test });
                    const hi_c = try l.func.appendInst(hi_block, vt, .{ .iconst = hi });
                    const le = try l.func.appendInst(hi_block, l.boolt, .{ .icmp = .{ .op = .le, .lhs = v.value, .rhs = hi_c } });
                    try l.func.appendIf(hi_block, le, .{ .target = body_blocks[i] }, .{ .target = next_test });
                    test_block = next_test;
                } else {
                    var k = i;
                    while (k <= j) : (k += 1) {
                        const labv = try l.func.appendInst(test_block, vt, .{ .iconst = sw.cases[k].label.? });
                        const eq = try l.func.appendInst(test_block, l.boolt, .{ .icmp = .{ .op = .eq, .lhs = v.value, .rhs = labv } });
                        const next_test = try l.func.appendBlock();
                        try l.func.appendIf(test_block, eq, .{ .target = body_blocks[k] }, .{ .target = next_test });
                        test_block = next_test;
                    }
                }
                i = j + 1;
            }
            // The last test block falls through to the default target.
            try l.func.setJump(test_block, default_target, &.{});

            // Lower each case body. Fall through to the next body block (C fallthrough).
            try l.loops.append(l.allocator, .{ .brk = exit_b, .cont = exit_b, .is_switch = true });
            for (sw.cases, 0..) |c, bi| {
                l.block = body_blocks[bi];
                const term = try lowerBlock(l, c.body);
                if (!term) {
                    const fallthrough = if (bi + 1 < sw.cases.len) body_blocks[bi + 1] else exit_b;
                    try l.func.setJump(l.block, fallthrough, &.{});
                }
            }
            _ = l.loops.pop();

            l.block = exit_b;
            return false;
        },
        .goto_ => |name| {
            // An unknown label is a clear error, not a silent no-op. A goto always
            // terminates the current path: whatever follows it, up to the next reachable
            // label, is dead code.
            const goto_target = labelBlock(l, name) orelse return error.Unsupported;
            try l.func.setJump(l.block, goto_target, &.{});
            return true;
        },
        .label => |lb| {
            // `lowerBlock`'s label-aware loop handles every `.label` that appears as a
            // direct element of a statement list (the normal case) itself, so this arm is a
            // safety net for the rare case of a label reached some other way. It assumes
            // the current block is NOT already terminated, which holds here because
            // `lowerStmt` is only ever called on a `.label` from a live path.
            const lbl = labelBlock(l, lb.name) orelse return error.Unsupported;
            try l.func.setJump(l.block, lbl, &.{});
            l.block = lbl;
            return lowerStmt(l, lb.body.*);
        },
    }
}

/// Build the `ctype.FuncType` for a function DEFINITION's signature (a function-designator-
/// as-a-value: `fp = add;`, `&add`, passing `add` as an arg): its declared return type and
/// the bare types of its params (dropping names, mirroring
/// `parser.parseFuncDeclaratorTail`'s own bare-type params). No parser AST node already carries
/// this shape for a plain top-level `Func` (only a grouped function-POINTER declarator builds a
/// `ctype.FuncType`. See that type's doc), so this synthesizes one, arena-allocated in
/// `l.type_arena` (freed with every other per-lowering synthesized `CType`). `fn_ast.ret` is
/// never actually absent: a source-level `void`-returning DEFINITION still carries `Func`'s own
/// placeholder (`ctype.int_t`, see `parse`'s `void`-function special case and
/// `finishFuncOrDecl`'s doc), so this always yields a non-null `ret`, the same pre-existing
/// conflation `findFunc`'s callers already live with elsewhere in this file. This is not
/// something this function changes or needs to resolve: no test here exercises a void
/// function-pointer value.
fn funcTypeOfDef(l: *L, fn_ast: *const parser.Func) Error!*const ctype.FuncType {
    const ret_box = try l.type_arena.create(ctype.CType);
    ret_box.* = fn_ast.ret;
    const params = try l.type_arena.alloc(ctype.CType, fn_ast.params.len);
    for (fn_ast.params, 0..) |p, i| params[i] = p.ty;
    const ft = try l.type_arena.create(ctype.FuncType);
    ft.* = .{ .ret = ret_box, .params = params };
    return ft;
}

/// Like `funcTypeOfDef`, but for a bodyless DECLARATION (a prototype or `extern`):
/// `parser.FuncDecl.ret` is already the right `?ctype.CType` shape (unlike `Func.ret`, a
/// declaration's `null` genuinely means `void`, no placeholder), so this only needs to box it.
fn funcTypeOfDecl(l: *L, decl: *const parser.FuncDecl) Error!*const ctype.FuncType {
    const ret_box: ?*const ctype.CType = if (decl.ret) |r| blk: {
        const b = try l.type_arena.create(ctype.CType);
        b.* = r;
        break :blk b;
    } else null;
    const params = try l.type_arena.alloc(ctype.CType, decl.params.len);
    for (decl.params, 0..) |p, i| params[i] = p.ty;
    const ft = try l.type_arena.create(ctype.FuncType);
    ft.* = .{ .ret = ret_box, .params = params };
    return ft;
}

/// Resolve a `.name` to its ADDRESS: the local/param binding's alloca slot if `n` is in
/// scope (`l.lookup` is checked FIRST, so a local/param always shadows a global of the same
/// name). Otherwise, if `n` names a module-level global, a `global_addr` IR op (the symbol's
/// runtime address, resolved at link time). Or, checked LAST since a function shares no
/// namespace with locals/globals here but a call site never reaches this path (see the
/// `.call` arm's own direct-name fast path), a function DESIGNATOR used as a VALUE
/// (`fp = add;`, `&add`, passing `add` as an arg): a `.func` `TypedValue`, no load (see
/// `lowerExpr`'s `.name` arm, which decays it exactly like an array name). A function DEFINED in
/// this TU resolves its address DIRECTLY (`appendGlobalAddr`), an ordinary intra-image code
/// symbol, exactly like an own-TU data global. A function only DECLARED here (`extern`,
/// defined in ANOTHER object or `.so`, never called by name anywhere in this TU, so no
/// `.call`-site relocation exists to redirect through a PLT stub) instead resolves
/// GOT-indirectly (`appendGlobalAddrGot`). This is EMPIRICALLY REQUIRED (verified by
/// `external_linkage.zig`'s function-pointer-to-an-extern-function tests): a direct
/// `adrp+add`/`lea`/`auipc+addi`-style address computation is fixed at LINK time, but an
/// imported symbol's real address is only known at LOAD time. So only a call-classified
/// relocation (redirected to a synthesized PLT stub, a locally-defined trampoline) or a GOT
/// read (the value the real `ld.so` writes via `GLOB_DAT`, exactly the data-import path) can
/// ever resolve it. Taking an extern function's address with NO call site in the same TU has
/// only the GOT option available. This reuses the SAME GOT-indirect machinery already proved
/// on all 4 arches. The resulting pointer's VALUE is the function's real address either way
/// (not a PLT stub address), so calling through it is a plain indirect call, identical to any
/// other function-pointer call. `error.Unsupported` if `n` is none of these. This IS
/// `lowerAddr`'s whole `.name` case. `lowerExpr`'s `.name` case reuses it too (a global loads
/// or decays through its resolved address exactly like a local does through its alloca slot).
fn resolveName(l: *L, n: []const u8) Error!TypedValue {
    if (l.lookup(n)) |bd| {
        const addr = switch (bd.addr) {
            .slot => |s| s, // the alloca slot IS &name
            // A `static` local: resolved fresh, in the CURRENT block, exactly like a
            // file-scope global just below. The symbol's runtime address is a `.data`/
            // `.bss`/`.rodata` object that persists across every call, not a per-call slot.
            .global => |sym| try l.func.appendGlobalAddr(l.block, l.ptrt, sym),
        };
        // `bd.quals` is this OBJECT's own qualifiers. `const int x` makes `x` itself a
        // const lvalue (a write to it is rejected). For an array, the parser always leaves
        // the object's own `quals` empty (fully absorbed into the array's ELEMENT `quals`
        // instead. See `parser.parseDeclarator`'s doc), so this is a no-op for
        // `const int a[3]`, which is instead caught by `.index`'s own `quals` (the array's
        // `.array.quals`, not this).
        return .{ .value = addr, .ty = bd.ty, .quals = bd.quals };
    }
    if (l.globals.get(n)) |g| {
        // A global with NO definition anywhere in this TU (`no_tu_def`, e.g.
        // `extern int counter;` with no accompanying definition) is defined in another
        // translation unit or a `.so`. Its address is resolved GOT-indirectly
        // (`appendGlobalAddrGot`, the data-import path) rather than directly, so the
        // linker can synthesize a GOT slot and GLOB_DAT import against the real definition. A
        // global THIS TU defines keeps the plain `appendGlobalAddr`, unchanged.
        const addr = if (g.no_tu_def)
            try l.func.appendGlobalAddrGot(l.block, l.ptrt, g.symbol)
        else
            try l.func.appendGlobalAddr(l.block, l.ptrt, g.symbol);
        return .{ .value = addr, .ty = g.ty, .quals = g.quals };
    }
    if (findFunc(l.funcs, n)) |fn_ast| {
        const ft = try funcTypeOfDef(l, fn_ast);
        const addr = try l.func.appendGlobalAddr(l.block, l.ptrt, n);
        return .{ .value = addr, .ty = .{ .func = ft } };
    }
    if (findFuncDecl(l.func_decls, n)) |decl| {
        const ft = try funcTypeOfDecl(l, decl);
        const addr = try l.func.appendGlobalAddrGot(l.block, l.ptrt, n);
        return .{ .value = addr, .ty = .{ .func = ft } };
    }
    return error.Unsupported;
}

/// Resolve a `.name` to its declared TYPE ONLY, emitting no IR (mirrors `resolveName` but for
/// `typeOf`/`lvalueTypeOf`'s unevaluated `sizeof` context. See `lvalueTypeOf`'s doc). Kept in
/// lock-step with `resolveName`'s function-designator arm: `sizeof`/`&` on a plain function
/// name computes the SAME `.func` type `resolveName` would, without emitting anything.
fn resolveNameType(l: *L, n: []const u8) Error!ctype.CType {
    if (l.lookup(n)) |bd| return bd.ty;
    if (l.globals.get(n)) |g| return g.ty;
    if (findFunc(l.funcs, n)) |fn_ast| return .{ .func = try funcTypeOfDef(l, fn_ast) };
    if (findFuncDecl(l.func_decls, n)) |decl| return .{ .func = try funcTypeOfDecl(l, decl) };
    return error.Unsupported;
}

/// The array `CType` for a string literal's DECODED `bytes`: `char[bytes.len + 1]` (the `+1`
/// is the NUL C always appends). Wired into `typeOf`/`lvalueTypeOf` directly. Unlike
/// `lowerStrLit` below, this mints no `DataObject` and emits no IR, so it's safe for
/// `sizeof`'s unevaluated context (`sizeof("abc") == 4` must not fabricate a `.rodata` object
/// for a string never actually read at runtime).
fn strLitArrayType(l: *L, bytes: []const u8) Error!ctype.CType {
    const elem = try l.type_arena.create(ctype.CType);
    elem.* = ctype.char_t;
    return .{ .array = .{ .elem = elem, .len = bytes.len + 1 } };
}

/// The array `CType` for a WIDE string literal's DECODED `bytes`: `int[bytes.len + 1]`.
/// `wchar_t` is a signed 32-bit int on every target this frontend supports, so the element
/// is `ctype.int_t`, not `char_t` (contrast `strLitArrayType`). Mirrors that function
/// exactly otherwise: no `DataObject` minted, no IR emitted, safe for `sizeof`'s
/// unevaluated context.
fn wstrLitArrayType(l: *L, bytes: []const u8) Error!ctype.CType {
    const elem = try l.type_arena.create(ctype.CType);
    elem.* = ctype.int_t;
    return .{ .array = .{ .elem = elem, .len = bytes.len + 1 } };
}

/// Lower a string literal to its ADDRESS: mint a fresh anonymous `.rodata` `DataObject`
/// (unique symbol `.str.<counter>`, bytes = the decoded literal plus a NUL terminator, one
/// per source OCCURRENCE, no dedup) and resolve its runtime address via `appendGlobalAddr`,
/// typed as the UNDECAYED `char[len+1]` array (mirrors `resolveName`'s result for an
/// array-typed `.name`. `lowerAddr`'s `.str_lit` case is exactly this, and `lowerExpr`
/// decays it to `char*`, same as any other array). `l.data.items.len` (already past every
/// earlier global/static/string) doubles as the uniqueness counter, the same trick a
/// `static` local's symbol uses. `ty`/`addr` are computed BEFORE the object is appended to
/// `l.data`, so a failure appending never leaves a dangling reference into freed memory. The
/// `errdefer`s below still own `nul_bytes`/`sym` right up until the append that hands them off.
fn lowerStrLit(l: *L, bytes: []const u8) Error!TypedValue {
    const nul_bytes = try l.allocator.alloc(u8, bytes.len + 1);
    errdefer l.allocator.free(nul_bytes);
    @memcpy(nul_bytes[0..bytes.len], bytes);
    nul_bytes[bytes.len] = 0;
    const sym = try std.fmt.allocPrint(l.allocator, ".str.{d}", .{l.data.items.len});
    errdefer l.allocator.free(sym);
    const ty = try strLitArrayType(l, bytes);
    const addr = try l.func.appendGlobalAddr(l.block, l.ptrt, sym);
    try l.data.append(l.allocator, .{ .name = sym, .kind = .rodata, .bytes = nul_bytes, .size = nul_bytes.len });
    return .{ .value = addr, .ty = ty };
}

/// Lower a WIDE string literal to its ADDRESS: mint a fresh anonymous `.rodata`
/// `DataObject` whose bytes are `bytes` WIDENED to 4 bytes per element (each `wchar_t`, a
/// signed 32-bit int, written little-endian, the byte order every target this frontend
/// supports uses) plus a 4-byte NUL element, resolve its runtime address via
/// `appendGlobalAddr`, typed as the UNDECAYED `int[len+1]` array. This mirrors `lowerStrLit`
/// exactly, one element width wider. See that function's doc comment for the shared shape
/// (unique `.str.<counter>` symbol keyed off `l.data.items.len`, `ty`/`addr` computed before
/// the object is appended so a failed append never leaves a dangling reference).
fn lowerWStrLit(l: *L, bytes: []const u8) Error!TypedValue {
    const wide_bytes = try l.allocator.alloc(u8, (bytes.len + 1) * 4);
    errdefer l.allocator.free(wide_bytes);
    for (bytes, 0..) |ch, idx| std.mem.writeInt(u32, wide_bytes[idx * 4 ..][0..4], ch, .little);
    std.mem.writeInt(u32, wide_bytes[bytes.len * 4 ..][0..4], 0, .little);
    const sym = try std.fmt.allocPrint(l.allocator, ".str.{d}", .{l.data.items.len});
    errdefer l.allocator.free(sym);
    const ty = try wstrLitArrayType(l, bytes);
    const addr = try l.func.appendGlobalAddr(l.block, l.ptrt, sym);
    try l.data.append(l.allocator, .{ .name = sym, .kind = .rodata, .bytes = wide_bytes, .size = wide_bytes.len });
    return .{ .value = addr, .ty = ty };
}

/// Lower an LVALUE expression to its ADDRESS: a `TypedValue` whose `.value` is a ptr (not
/// the lvalue's own value), whose `.ty` is the pointed-to/declared type of the lvalue itself
/// (not a pointer type), and whose `.quals` is the LVALUE's own qualifiers. Every write
/// site (`.assign`/`.incdec` in `lowerExpr`) checks `is_const` on this before storing. This
/// is the spine `&`/`*`/assignment share: `&e` is just `lowerAddr(e).value` (no load. See
/// `lowerExpr`'s `.addrof`, which also carries `.quals` into the resulting pointer's
/// pointee-quals so `&const_x` is `const int *`). `*p = v` and `*p` (as an rvalue) both
/// start by resolving `*p`'s address here, then either store or load through it. `.name` (a
/// local's entry-block alloca slot, or a global's `global_addr`, IS `&name`. See
/// `resolveName`. `.quals` is the bound OBJECT's own), `.deref` (the pointer's own VALUE,
/// lowered as an ordinary rvalue via `lowerExpr`, already IS the address. Its pointee type
/// AND pointee-quals both come off that pointer's `CType.ptr`. `const int *p` makes `*p` a
/// const lvalue), `.index` (arrays/pointers, same pointee-quals source as `.deref`, since
/// `arr[i]` is `*(arr + i)`), and `.member` (a field of a const struct is itself const, the
/// field's OWN `quals` OR-ed with the base struct lvalue's `quals`, so qualification
/// propagates through nested aggregates without each field needing to redundantly declare it).
/// `.str_lit` is always const (a string literal has no writable storage in this frontend's
/// model). Anything else isn't an lvalue: fails closed with `error.Unsupported` rather than
/// reaching a caller that assumes `.value` is an address.
fn lowerAddr(l: *L, expr: *const parser.Expr) Error!TypedValue {
    return switch (expr.*) {
        .name => |n| try resolveName(l, n),
        // A string literal isn't technically an lvalue in C, but its address is resolved
        // the exact same way an array name's is (see `lowerStrLit`). `sizeof` and
        // `.index`'s base decay both need this same undecayed-array-address shape. Marked
        // `const`: a string literal backs a `.rodata` object, so a direct write through it
        // (`"abc"[0] = 'x'`) is rejected, same as gcc's `-Wwrite-strings`.
        .str_lit => |s| blk: {
            var tv = try lowerStrLit(l, s);
            tv.quals = .{ .is_const = true };
            break :blk tv;
        },
        // A wide string literal is const for the same reason a narrow one is: its address
        // resolves through `lowerWStrLit`, otherwise identical to `.str_lit`.
        .wstr_lit => |s| blk: {
            var tv = try lowerWStrLit(l, s);
            tv.quals = .{ .is_const = true };
            break :blk tv;
        },
        .deref => |inner| blk: {
            const p = try lowerExpr(l, inner); // p is a pointer VALUE, not an lvalue itself
            const pointee = p.ty.pointee() orelse break :blk error.Unsupported;
            // `p.ty` is confirmed `.ptr` by the successful `pointee()` above, so `.ptr.quals`
            // (the POINTEE's own qualifiers, see `ctype.CType.ptr`'s doc) is safe to read.
            break :blk .{ .value = p.value, .ty = pointee.*, .quals = p.ty.ptr.quals };
        },
        // `arr[i]` = `*(arr + i)`: `base` decays an array to ptr-to-elem (via `lowerExpr`'s
        // `.name`/`.index` decay) or is already a pointer. Scale `idx` by the pointee's size
        // and add. Multi-dim (`m[i][j]`) falls out automatically: `ix.base` is itself
        // `.index{m, i}` here, and `lowerExpr`'s `.index` arm decays `m[i]` (when its element
        // type is itself an array, e.g. `int[3]`) to a pointer rather than loading it. So
        // `base` above is `&m[i]`, and `elem`/the scale factor come from THAT pointer's
        // pointee (`int`, size 4), giving `&m[i] + j*4 = &m[i][j]`, i.e.
        // `base_of_m + i*12 + j*4`. `base.ty.ptr.quals` (safe by the same reasoning as
        // `.deref` above) is the element's own qualifiers. For a decayed array this is the
        // array's ELEMENT quals (`.array.quals`, threaded through decay. See `lowerExpr`'s
        // `.name`/`.index`/`.member` decay sites), and for a real pointer it's the pointer's
        // own pointee-quals. Either way `const int a[3]; a[0] = 1;` and `const int *p; p[0] = 1;`
        // are both caught the same way.
        .index => |ix| blk: {
            const base = try lowerExpr(l, ix.base);
            const elem = base.ty.pointee() orelse break :blk error.Unsupported;
            const idx = try lowerExpr(l, ix.idx);
            const addr = try scaledAdd(l, base.value, idx, try elem.storageSize(l.layout));
            break :blk .{ .value = addr, .ty = elem.*, .quals = base.ty.ptr.quals };
        },
        // `base.field` / `base->field`: resolve the base struct's ADDRESS. For `.` that's
        // the base lvalue's own address (`lowerAddr`, no load: the struct itself lives at
        // that address, same idea as `.name`). For `->` the base is a POINTER rvalue
        // (`lowerExpr`) whose VALUE already IS the struct's address, no lvalue resolution of
        // the base needed at all. Then look up the field by name in the resolved `StructDef`
        // and offset: a plain `arith.add` of the field's byte offset (skipped entirely when
        // it's 0, matching `.index`'s style of avoiding a pointless add for the common
        // first-field/only-field case). `base_quals` is the base struct's OWN qualifiers
        // (through the pointer's pointee-quals for `->`, or the base lvalue's own `quals`
        // for `.`), OR-ed with the field's own `quals` below: a member of a const struct is
        // const even if the field itself wasn't declared so.
        .member => |m| blk: {
            var base_quals: ctype.Quals = .{};
            const base_addr: Value = if (m.arrow) ptrblk: {
                const p = try lowerExpr(l, m.base); // pointer value = struct address
                if (p.ty == .ptr) base_quals = p.ty.ptr.quals; // else: not a pointer, caught below by memberStructDef
                break :ptrblk p.value;
            } else lvblk: {
                const ba = try lowerAddr(l, m.base); // struct lvalue address
                base_quals = ba.quals;
                break :lvblk ba.value;
            };
            const sdef = try memberStructDef(l, m.base, m.arrow) orelse break :blk error.Unsupported;
            const fld = findField(sdef, m.field) orelse break :blk error.Unsupported;
            const addr = if (fld.offset == 0) base_addr else addr: {
                const off_v = try l.func.appendInst(l.block, l.i64t, .{ .iconst = @intCast(fld.offset) });
                break :addr try l.func.appendInst(l.block, l.ptrt, .{ .arith = .{ .op = .add, .lhs = base_addr, .rhs = off_v } });
            };
            // For a BITFIELD member, `addr` is its storage unit's address, not the field's
            // own (a bitfield has none). Carry the bit position so the read/write sites
            // shift and mask. `fld.bit_width == null` for every ordinary field, so this stays
            // `null` and the plain load/store path is unchanged.
            const bf: ?TypedValue.Bitfield = if (fld.bit_width) |w| .{ .bit_offset = fld.bit_offset, .bit_width = w } else null;
            break :blk .{ .value = addr, .ty = fld.ty, .quals = orQuals(fld.quals, base_quals), .bitfield = bf };
        },
        // `(type-name){ ... }` (C99 compound literal): an unnamed AUTOMATIC object. Allocate
        // its own stack slot (exactly like a local declaration's, see `allocSlot`'s own
        // hazard note. It always targets the entry block, so it's safe to call from anywhere
        // `lowerAddr` runs) and store the brace-list into it via the same engine a local
        // aggregate declaration uses (`materializeCompoundLiteral`). The slot's address IS
        // this expression's lvalue address, unqualified (`(struct S){1,2}.x = 5` is valid C:
        // a compound literal is a modifiable lvalue unless its own declared type is itself
        // qualified, which this frontend's `parseAbstractDeclarator` target never is).
        .compound_literal => |cl| blk: {
            const slot = try l.allocSlot(cl.ty);
            try materializeCompoundLiteral(l, slot, cl.ty, cl.init);
            break :blk .{ .value = slot, .ty = cl.ty, .quals = .{} };
        },
        // A struct-returning call is a usable lvalue: `lowerExpr`'s `.call` arm already
        // allocates a destination slot for a `.registers` return and yields its ADDRESS as
        // the call's value (ty = the struct), so this just runs that same lowering and hands
        // the address back. `mk().field`, `struct S q = mk();`, and passing `mk()` onward all
        // reach the slot through here. A NON-struct (scalar) call has no address (it is not
        // an lvalue in C), so it fails closed the same way the `else` below would.
        .call => blk: {
            const tv = try lowerExpr(l, expr);
            if (tv.ty != .@"struct") break :blk error.Unsupported;
            break :blk tv;
        },
        else => error.Unsupported, // not an lvalue
    };
}

/// Combine two `Quals` by OR-ing each flag: qualified if EITHER source says so. Used to
/// propagate qualification through aggregate access (`.member`'s field-quals-OR-base-quals)
/// and through array decay (an array reached via a qualified struct field/nested index is
/// itself qualified even when its own declared element type isn't), see the decay call
/// sites in `lowerExpr`.
fn orQuals(a: ctype.Quals, b: ctype.Quals) ctype.Quals {
    return .{ .is_const = a.is_const or b.is_const, .is_volatile = a.is_volatile or b.is_volatile };
}

/// The `StructDef` a `.member`'s base resolves to: for `->`, `base`'s (rvalue) type must be
/// a pointer to `.@"struct"`. For `.`, `base`'s LVALUE type must itself be `.@"struct"`.
/// Failures computing the base's type in the first place (e.g. An undefined name) propagate
/// as `error.Unsupported` via `try`, rather than being swallowed into a plain `null`.
fn memberStructDef(l: *L, base: *const parser.Expr, arrow: bool) Error!?*const ctype.StructDef {
    const base_ty = if (arrow) blk: {
        const t = try typeOf(l, base);
        break :blk (t.pointee() orelse return null).*;
    } else try lvalueTypeOf(l, base);
    return switch (base_ty) {
        .@"struct" => |s| s,
        else => null,
    };
}

/// Linear-scan a struct/union's fields by name (structs here are small, a handful of
/// fields, so no need for anything fancier).
fn findField(def: *const ctype.StructDef, name: []const u8) ?ctype.Field {
    for (def.fields) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

/// Same lookup as `findField`, but the field's INDEX rather than a copy of it:
/// `lowerAggregateInit`'s `.field` designator needs the index to resume POSITIONAL tracking
/// from (`pos + 1`), which `findField`'s `?ctype.Field` alone can't give back.
fn findFieldIndex(def: *const ctype.StructDef, name: []const u8) ?usize {
    for (def.fields, 0..) |f, i| {
        if (std.mem.eql(u8, f.name, name)) return i;
    }
    return null;
}

/// Read a BITFIELD value. `addr` is the storage unit's address and `ty` its declaring
/// integer type (`W` bits). The unit is loaded whole, then the field's bits are moved into
/// place by a left-then-right shift pair: shift LEFT so the field's most-significant bit
/// reaches the type's top bit, then shift RIGHT back down. The right shift is arithmetic
/// for a SIGNED type (sign-extending the field, so a signed `:4` of `0b1111` reads back `-1`)
/// and logical for an UNSIGNED type (zero-extending). The IR's `shr` derives that from its
/// operand's own signedness, exactly like an ordinary `>>`. Yields a value of type `ty`.
fn loadBitfield(l: *L, addr: Value, ty: ctype.CType, bf: TypedValue.Bitfield, is_volatile: bool) Error!Value {
    const t = try l.irTy(ty);
    const total: u32 = ty.asInt().?.bits(l.layout);
    var v = try l.func.appendInst(l.block, t, .{ .load = .{ .ptr = addr, .@"volatile" = is_volatile } });
    const lshift = total - bf.bit_offset - bf.bit_width;
    if (lshift != 0) {
        const s = try l.func.appendInst(l.block, t, .{ .iconst = lshift });
        v = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .shl, .lhs = v, .rhs = s } });
    }
    const rshift = total - bf.bit_width;
    if (rshift != 0) {
        const s = try l.func.appendInst(l.block, t, .{ .iconst = rshift });
        v = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .shr, .lhs = v, .rhs = s } });
    }
    return v;
}

/// Map a `__builtin_bswap16/32/64` callee name to its byte width, or null for any other
/// name. VCC lowers these GCC intrinsics itself (see `lowerBswap`). Glibc's `<bits/byteswap.h>`
/// inline helpers call them.
fn bswapBytes(name: []const u8) ?u32 {
    if (std.mem.eql(u8, name, "__builtin_bswap16")) return 2;
    if (std.mem.eql(u8, name, "__builtin_bswap32")) return 4;
    if (std.mem.eql(u8, name, "__builtin_bswap64")) return 8;
    return null;
}

/// Emit the byte reversal of `x` (a value of IR type `ity`) for a `bytes`-wide integer. GCC
/// lowers `__builtin_bswap*` to a `rev` instruction. VCC has no byteswap IR op, so it builds the
/// portable shift/mask/or equivalent that every backend already supports. Byte `i` of the input
/// moves to byte `bytes-1-i` of the result. The `& 0xFF` after each shift keeps the byte clean
/// even when `.shr` sign-extends, so the input signedness does not matter.
fn lowerBswap(l: *L, x: Value, bytes: u32, ity: Type) Error!Value {
    const f = l.func;
    var acc: ?Value = null;
    var i: u32 = 0;
    while (i < bytes) : (i += 1) {
        const src_shift: i64 = @intCast(i * 8);
        const dst_shift: i64 = @intCast((bytes - 1 - i) * 8);
        var byte = x;
        if (src_shift != 0) {
            const s = try f.appendInst(l.block, ity, .{ .iconst = src_shift });
            byte = try f.appendInst(l.block, ity, .{ .arith = .{ .op = .shr, .lhs = byte, .rhs = s } });
        }
        const mask = try f.appendInst(l.block, ity, .{ .iconst = 0xFF });
        byte = try f.appendInst(l.block, ity, .{ .arith = .{ .op = .bit_and, .lhs = byte, .rhs = mask } });
        if (dst_shift != 0) {
            const s = try f.appendInst(l.block, ity, .{ .iconst = dst_shift });
            byte = try f.appendInst(l.block, ity, .{ .arith = .{ .op = .shl, .lhs = byte, .rhs = s } });
        }
        acc = if (acc) |a| try f.appendInst(l.block, ity, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = byte } }) else byte;
    }
    return acc.?;
}

/// Write a BITFIELD value, a READ-MODIFY-WRITE, never a plain store. `value` must already
/// be of the declaring type `ty` (the caller converts it). The stored unit becomes
/// `(unit & ~(mask << off)) | ((value & mask) << off)`: the field's own bits cleared, then the
/// low `bit_width` bits of `value` OR-ed back in at `bit_offset`. Neighboring bitfields packed
/// into the same unit keep their values because only the field's own bit range is touched. The
/// constants are masked to `W` bits so a full-width or high-offset field stays representable.
fn storeBitfield(l: *L, addr: Value, ty: ctype.CType, bf: TypedValue.Bitfield, value: Value, is_volatile: bool) Error!void {
    const t = try l.irTy(ty);
    const total: u32 = ty.asInt().?.bits(l.layout);
    const type_mask: u64 = if (total >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(total)) - 1;
    const field_mask: u64 = if (bf.bit_width >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bf.bit_width)) - 1;
    const shifted_mask: u64 = (field_mask << @intCast(bf.bit_offset)) & type_mask;
    const clear_mask: u64 = (~shifted_mask) & type_mask;
    const unit = try l.func.appendInst(l.block, t, .{ .load = .{ .ptr = addr, .@"volatile" = is_volatile } });
    // `value & field_mask`, then shift it up to `bit_offset`.
    const fmask_v = try l.func.appendInst(l.block, t, .{ .iconst = @bitCast(field_mask) });
    var vbits = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .bit_and, .lhs = value, .rhs = fmask_v } });
    if (bf.bit_offset != 0) {
        const s = try l.func.appendInst(l.block, t, .{ .iconst = bf.bit_offset });
        vbits = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .shl, .lhs = vbits, .rhs = s } });
    }
    // `unit & clear_mask`, then OR the new bits in.
    const clear_v = try l.func.appendInst(l.block, t, .{ .iconst = @bitCast(clear_mask) });
    const cleared = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .bit_and, .lhs = unit, .rhs = clear_v } });
    const new_unit = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .bit_or, .lhs = cleared, .rhs = vbits } });
    try l.func.appendStoreVol(l.block, new_unit, addr, is_volatile);
}

/// Whole-struct/union assignment (`s1 = s2`): copy EXACTLY `def.size` bytes from `src` to
/// `dst`. The storage blob `CType.irType` reserves is rounded up to whole `i64` words, but a
/// struct FIELD (laid out at its exact C offset) can sit immediately after this struct's exact
/// `size` with no padding, a nested `struct{ struct S a; int b; }` puts `b` right after `a`'s
/// 4 bytes, so copying rounded-up words would clobber that neighbor. Copy the largest chunks
/// that fit inside `[0, size)`: `i64` words first, then an `i32`/`i16`/`i8` tail for the last
/// `size % 8` bytes. Structs are small, so this is a handful of fully-unrolled load/store pairs.
/// (Array elements are spaced by the rounded `storageSize` >= exact `size`, so an exact-size
/// copy into `arr[i]` also stays entirely within that element.)
fn copyStruct(l: *L, def: *const ctype.StructDef, src: Value, dst: Value) Error!void {
    const f = l.func;
    const chunks = [_]struct { bytes: u64, ty: Type }{
        .{ .bytes = 8, .ty = l.i64t },
        .{ .bytes = 4, .ty = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }) },
        .{ .bytes = 2, .ty = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }) },
        .{ .bytes = 1, .ty = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } }) },
    };
    var offset: u64 = 0;
    for (chunks) |c| {
        while (def.size - offset >= c.bytes) : (offset += c.bytes) {
            const src_addr = try byteOffset(l, src, offset);
            const dst_addr = try byteOffset(l, dst, offset);
            const v = try f.appendInst(l.block, c.ty, .{ .load = .{ .ptr = src_addr } });
            try f.appendStore(l.block, v, dst_addr);
        }
    }
}

/// The IR float type for an `.sse` `abi.Eightbyte`: `f32` for a 4-byte piece (a lone
/// `float`, or two `float`s packed into one x86_64 SysV eightbyte, loaded/stored as ONE
/// `f64`-sized raw copy either way, see the 8-byte arm's doc), `f64` for an 8-byte one.
/// Any other width can't occur (the classifier only ever emits 4 or 8 for `.sse`. See
/// `abi.Eightbyte`'s doc), so it fails closed rather than silently misreading the struct.
fn floatEightbyteType(l: *L, eb: abi.Eightbyte) Error!Type {
    return switch (eb.bytes) {
        4 => l.irTy(.{ .float = .f32 }),
        8 => l.irTy(.{ .float = .f64 }),
        else => error.Unsupported,
    };
}

/// `ptr + off` (bytes) as a `ptr`-typed value. `off == 0` returns `ptr` unchanged.
fn byteOffset(l: *L, ptr: Value, off: u64) Error!Value {
    if (off == 0) return ptr;
    const f = l.func;
    const off_v = try f.appendInst(l.block, l.i64t, .{ .iconst = @intCast(off) });
    return f.appendInst(l.block, l.ptrt, .{ .arith = .{ .op = .add, .lhs = ptr, .rhs = off_v } });
}

/// Store a zero of `n` bytes starting at `dest`, the largest chunk that fits at each
/// position (`i64` words, then an `i32`/`i16`/`i8` tail), same chunking `copyStruct` uses for
/// a struct-to-struct copy. Used by `lowerAggregateInit` to zero-fill a runtime aggregate
/// SLOT before its listed elements/fields are stored over the top, so a PARTIAL brace-list
/// (`int a[3] = {1};`) leaves every unlisted element at zero, matching C99 6.7.9p21.
fn zeroBytes(l: *L, dest: Value, n: u64) Error!void {
    const f = l.func;
    const chunks = [_]struct { bytes: u64, ty: Type }{
        .{ .bytes = 8, .ty = l.i64t },
        .{ .bytes = 4, .ty = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }) },
        .{ .bytes = 2, .ty = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }) },
        .{ .bytes = 1, .ty = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } }) },
    };
    var offset: u64 = 0;
    for (chunks) |c| {
        while (n - offset >= c.bytes) : (offset += c.bytes) {
            const addr = try byteOffset(l, dest, offset);
            const z = try f.appendInst(l.block, c.ty, .{ .iconst = 0 });
            try f.appendStore(l.block, z, addr);
        }
    }
}

/// Resolve one MORE step of a designator chain and, once it's exhausted, store `value` at the
/// resulting address. Mirrors `consteval.writeNested`'s role for the runtime path (see that
/// function's doc for the full rationale). `addr`/`ty` are the CURRENT sub-object's address
/// and type (the caller passes the position it already resolved from `chain[0]` at ITS own
/// level). `chain` is the REMAINING steps (never empty. Every call site resolves at least
/// one step first). Each remaining step just walks the address deeper (a `.field` step adds
/// `field.offset` via `byteOffset` and narrows `ty` to `field.ty`. A `.index` step adds
/// `idx * elem.storageSize` and narrows to `elem`). NOTHING is stored or zeroed until the
/// chain is fully consumed.
///
/// This is what fixes the bug the old single-step `subInitFor` peel had: that helper handed
/// the REST of a multi-step chain (e.g. `.b.x`'s `.x`) to a fresh recursive
/// `lowerAggregateInit` call scoped to the immediate sub-object (`.b`'s own struct), and
/// `lowerAggregateInit`'s `.list` arm unconditionally `zeroBytes`-zero-fills that sub-object
/// before storing into it. So a second top-level item sharing the same prefix (`.b.y = 6`
/// after `.b.x = 5`) re-zeroed and wiped out the store `.b.x = 5` had just made. Walking the
/// whole chain here before ever calling back into `lowerAggregateInit` (only at the TRUE leaf)
/// means a scalar leaf store never re-zeroes its siblings. A struct field that's a BITFIELD is
/// handled right here (not inside `lowerAggregateInit`, which has no bitfield read-modify-
/// write path) when it's the chain's LAST step. A leaf whose own value is itself a brace-list
/// still goes through `lowerAggregateInit`: correct, since the whole sub-object legitimately
/// being replaced as a unit is exactly when re-zeroing it is right.
fn storeDesignated(l: *L, addr: Value, ty: ctype.CType, chain: []const parser.Designator, value: *const parser.Initializer) Error!void {
    std.debug.assert(chain.len >= 1); // every call site resolves at least one step first
    switch (chain[0]) {
        .field => |name| {
            const def = switch (ty) {
                .@"struct" => |s| s,
                else => return error.Unsupported, // `.field` into a non-struct: invalid C
            };
            const idx = findFieldIndex(def, name) orelse return error.Unsupported;
            const fld = def.fields[idx];
            const field_addr = try byteOffset(l, addr, fld.offset);
            if (chain.len == 1) {
                if (fld.bit_width) |w| {
                    // A bitfield's brace-list entry must be a bare scalar. There is no
                    // address to recurse a nested brace-list into.
                    const e = switch (value.value) {
                        .expr => |e| e,
                        .list => return error.Unsupported,
                    };
                    const v = try lowerExpr(l, e);
                    const conv = try l.convertTo(v, fld.ty);
                    const bf: TypedValue.Bitfield = .{ .bit_offset = fld.bit_offset, .bit_width = w };
                    return storeBitfield(l, field_addr, fld.ty, bf, conv, false);
                }
                return lowerAggregateInit(l, field_addr, fld.ty, value);
            }
            return storeDesignated(l, field_addr, fld.ty, chain[1..], value);
        },
        .index => |idx| {
            const a = switch (ty) {
                .array => |arr| arr,
                else => return error.Unsupported, // `[i]` into a non-array: invalid C
            };
            if (idx >= a.len) return error.Unsupported;
            const stride = try a.elem.storageSize(l.layout);
            const elem_addr = try byteOffset(l, addr, idx * stride);
            if (chain.len == 1) return lowerAggregateInit(l, elem_addr, a.elem.*, value);
            return storeDesignated(l, elem_addr, a.elem.*, chain[1..], value);
        },
    }
}

/// Runtime element-wise lowering of a brace-list (or bare scalar) INITIALIZER into a SLOT at
/// `dest_addr`. This is the automatic-local counterpart of `consteval.evalInit`'s
/// compile-time folding, reused later by a compound literal. A `.list` on an array or
/// struct `ty` first ZERO-FILLS the whole `storageSize(ty)`-byte region (`zeroBytes`), then
/// stores each supplied element/field over the top, address-by-address, the same
/// `arith.add`-of-a-byte-offset pattern `lowerAddr`'s `.index`/`.member` arms use to reach
/// an element or field (`byteOffset`). `pos` tracks the CURRENT element/field position
/// exactly like `consteval.evalArrayInit`/`evalStructInit` does: a designator on an element
/// SETS it (an `[index]` for an array, a `.field` via `findFieldIndex` for a struct). A plain
/// element uses it then advances by one, so an all-positional list (no designators anywhere)
/// is `pos == i`, unchanged from before designators existed. LAST-WINS for a position more
/// than one element writes to falls straight out of this being a SEQUENCE OF STORES: a later
/// store to the same address just leaves the later value, no bookkeeping needed (unlike the
/// const-fold path's byte buffer, which does need explicit reloc dedup. See
/// `consteval.removeOverlappingRelocs`). A struct field that is a BITFIELD has no address of
/// its own, so it goes through the same read-modify-write `storeBitfield` a plain assignment
/// to it uses. The leaf case, a bare `.expr` against a scalar/pointer `ty`, is the ordinary
/// local scalar-init store (`convertTo` plus a plain store). An `.expr` against an AGGREGATE
/// `ty` (struct-by-value, or a string-literal char-array) has no runtime store sequence here
/// yet and fails closed rather than mis-storing.
fn lowerAggregateInit(l: *L, dest_addr: Value, ty: ctype.CType, init: *const parser.Initializer) Error!void {
    switch (init.value) {
        .list => |items| switch (ty) {
            .array => |a| {
                try zeroBytes(l, dest_addr, try ty.storageSize(l.layout));
                var pos: u64 = 0;
                for (items) |*item| {
                    if (item.designators.len > 0) {
                        pos = switch (item.designators[0]) {
                            .index => |idx| idx,
                            .field => return error.Unsupported, // `.field` on an array target: invalid C
                        };
                    }
                    if (pos >= a.len) return error.Unsupported; // designator/position past the array's end
                    var synth: [1]parser.Designator = undefined;
                    const chain: []const parser.Designator = if (item.designators.len > 0) item.designators else blk: {
                        synth[0] = .{ .index = pos };
                        break :blk synth[0..1];
                    };
                    const value_init: parser.Initializer = .{ .value = item.value };
                    try storeDesignated(l, dest_addr, ty, chain, &value_init);
                    pos += 1;
                }
            },
            .@"struct" => |s| {
                try zeroBytes(l, dest_addr, try ty.storageSize(l.layout));
                var pos: usize = 0;
                for (items) |*item| {
                    if (item.designators.len > 0) {
                        pos = switch (item.designators[0]) {
                            .field => |name| findFieldIndex(s, name) orelse return error.Unsupported,
                            .index => return error.Unsupported, // `[i]` on a struct target: invalid C
                        };
                    }
                    if (pos >= s.fields.len) return error.Unsupported; // designator/position past the last field
                    var synth: [1]parser.Designator = undefined;
                    const chain: []const parser.Designator = if (item.designators.len > 0) item.designators else blk: {
                        synth[0] = .{ .field = s.fields[pos].name };
                        break :blk synth[0..1];
                    };
                    const value_init: parser.Initializer = .{ .value = item.value };
                    try storeDesignated(l, dest_addr, ty, chain, &value_init);
                    pos += 1;
                }
            },
            else => return error.Unsupported, // a `{...}` brace-list against a scalar type is invalid C
        },
        .expr => |e| {
            if (ty == .array or ty == .@"struct") return error.Unsupported; // struct-by-value / char[]="..." - deferred
            const v = try lowerExpr(l, e);
            const conv = try l.convertTo(v, ty);
            try l.func.appendStore(l.block, conv, dest_addr);
        },
    }
}

/// Store a compound literal's `init` into the slot at `dest_addr`. This is the single place
/// `lowerAddr`'s `.compound_literal` arm and the `.decl` aggregate-init special case both
/// funnel through, so the materialize logic lives in exactly one spot (DRY).
/// `.array`/`.@"struct"` `ty`, or ANY scalar `ty` whose init is genuinely a multi-element
/// `.list`, goes straight to `lowerAggregateInit` (the engine above, unchanged). A SCALAR `ty`
/// is special: `parseInitializerValue` always parses a brace-list, so `(int){5}` arrives as a
/// ONE-ELEMENT `.list` wrapping `.expr = 5`. `lowerAggregateInit` itself rejects any `.list`
/// against a scalar target (a `{...}` brace-list on a scalar is invalid C in every OTHER
/// context), so that single element is unwrapped here first and stored as a plain scalar
/// (matching `evalScalarInit`'s identical "`.list` of exactly one plain `.expr`" unwrap for a
/// static scalar initializer). A multi-element `.list` against a scalar `ty` is invalid C and
/// fails closed via `lowerAggregateInit`'s own scalar-`.list` rejection.
fn materializeCompoundLiteral(l: *L, dest_addr: Value, ty: ctype.CType, init: *const parser.Initializer) Error!void {
    if (ty == .array or ty == .@"struct") return lowerAggregateInit(l, dest_addr, ty, init);
    const expr: *parser.Expr = switch (init.value) {
        .expr => |e| e,
        .list => |items| blk: {
            if (items.len == 1 and items[0].designators.len == 0 and items[0].value == .expr) break :blk items[0].value.expr;
            return lowerAggregateInit(l, dest_addr, ty, init); // multi-element/nested list against a scalar: invalid C, fails closed
        },
    };
    const v = try lowerExpr(l, expr);
    const conv = try l.convertTo(v, ty);
    try l.func.appendStore(l.block, conv, dest_addr);
}

/// Statically compute an LVALUE's type (mirrors `lowerAddr`, but WITHOUT lowering anything
/// to IR), `typeOf`'s `.assign`/`.addrof`/`.deref` arms need an lvalue's type for `sizeof`,
/// which per C is unevaluated (no side effects, no emitted instructions allowed even for the
/// pointer operand of a `*p` inside `sizeof`).
fn lvalueTypeOf(l: *L, expr: *const parser.Expr) Error!ctype.CType {
    return switch (expr.*) {
        .name => |n| try resolveNameType(l, n),
        .str_lit => |s| try strLitArrayType(l, s),
        .wstr_lit => |s| try wstrLitArrayType(l, s),
        .deref => |inner| blk: {
            const t = try typeOf(l, inner);
            const pointee = t.pointee() orelse break :blk error.Unsupported;
            break :blk pointee.*;
        },
        // `bt` is `ix.base`'s UNDECAYED static type (an array name's `typeOf` doesn't decay
        // it either, matching `sizeof arr` != `sizeof` a decayed pointer), so for a 2-D `m`,
        // `typeOf(m)` is the full `array{2, array{3,int}}` and this returns its `.elem`
        // (`array{3,int}`), no decay, since `sizeof` never decays its immediate operand.
        // Multi-dim `m[i][j]` recurses correctly with no extra code: `ix.base` here is
        // itself `.index{m,i}`, so `typeOf(ix.base)` re-enters this same arm (via the
        // `.index => lvalueTypeOf` line below) and returns `m[i]`'s undecayed type
        // (`array{3,int}`), from which this arm peels one more `.elem` to `int`.
        .index => |ix| blk: {
            const bt = try typeOf(l, ix.base);
            const elem = if (bt == .array) bt.array.elem else bt.pointee() orelse break :blk error.Unsupported;
            break :blk elem.*;
        },
        .member => |m| blk: {
            const sdef = try memberStructDef(l, m.base, m.arrow) orelse break :blk error.Unsupported;
            const fld = findField(sdef, m.field) orelse break :blk error.Unsupported;
            break :blk fld.ty;
        },
        // A compound literal's type is right on the node, no lowering needed. Needed here
        // so `(struct S){...}.field` resolves: `.member`'s `.` (non-arrow) path asks
        // `memberStructDef` for the base's type WITHOUT lowering it (`lowerAddr`'s own
        // `.member` arm lowers the base separately, exactly once, for the real address).
        .compound_literal => |cl| cl.ty,
        // A struct-returning call is an lvalue, so `mk().field` resolves the base type
        // through the same callee-return-type lookup `typeOf` does. A scalar call is an
        // rvalue, not an lvalue, so it fails closed the same way the `else` below would.
        .call => blk: {
            const t = try typeOf(l, expr);
            if (t != .@"struct") break :blk error.Unsupported;
            break :blk t;
        },
        else => error.Unsupported,
    };
}

/// Lower an expression (literals, names, unary -/~/!, `&`/`*`, binary arithmetic/bitwise/
/// shift, comparisons, assignment, and calls) to a `TypedValue` in `l.block`. Every arm
/// computes its own result `CType` per C's rules. Binary/compare/assign route both operands
/// through `CType.commonType` and `L.convertTo` (a no-op, emitting nothing, when the operand
/// already has the target's width/signedness, the case for every all-`int` program).
fn lowerExpr(l: *L, expr: *const parser.Expr) Error!TypedValue {
    const f = l.func;
    return switch (expr.*) {
        .int_lit => |v| blk: {
            const it = try l.irTy(v.ty);
            break :blk .{ .value = try f.appendInst(l.block, it, .{ .iconst = v.value }), .ty = v.ty };
        },
        .float_lit => |v| blk: {
            const ft = try l.irTy(v.ty);
            break :blk .{ .value = try f.appendInst(l.block, ft, .{ .fconst = v.value }), .ty = v.ty };
        },
        .negate => |inner| blk: {
            const iv = try lowerExpr(l, inner);
            // Float negate: `-d` is `0.0 - d`, typed at the operand's own float type (no
            // integer promotion applies). `.arith.sub` on float operands codegens fsub, same
            // as any float `-`. Kept separate from the int path below rather than folded,
            // since a float never promotes (its `promote` is the identity).
            if (iv.ty.isFloat()) {
                const ft = try l.irTy(iv.ty);
                const zero = try f.appendInst(l.block, ft, .{ .fconst = 0.0 });
                break :blk .{ .value = try f.appendInst(l.block, ft, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = iv.value } }), .ty = iv.ty };
            }
            if (!iv.ty.isInt()) break :blk error.Unsupported; // pointer/array negate: not yet supported
            const pt = iv.ty.promote();
            const pv = try l.convertTo(iv, pt);
            const it = try l.irTy(pt);
            const zero = try f.appendInst(l.block, it, .{ .iconst = 0 });
            break :blk .{ .value = try f.appendInst(l.block, it, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = pv } }), .ty = pt };
        },
        .complement => |inner| blk: {
            const iv = try lowerExpr(l, inner);
            if (!iv.ty.isInt()) break :blk error.Unsupported; // pointer/array complement: not yet supported
            const pt = iv.ty.promote();
            const pv = try l.convertTo(iv, pt);
            const it = try l.irTy(pt);
            const ones = try f.appendInst(l.block, it, .{ .iconst = -1 });
            break :blk .{ .value = try f.appendInst(l.block, it, .{ .arith = .{ .op = .bit_xor, .lhs = pv, .rhs = ones } }), .ty = pt };
        },
        .lognot => |inner| blk: {
            const iv = try lowerExpr(l, inner);
            if (iv.ty == .@"struct") break :blk error.Unsupported; // bare struct rvalue: not supported
            // Compare against a zero of the operand's OWN type, `fconst 0.0` for a float
            // (so `!d` is a float-to-float fcmp eq), `iconst 0` for an int.
            const zero = try zeroInto(l, l.block, iv.ty);
            const cmp = try f.appendInst(l.block, l.boolt, .{ .icmp = .{ .op = .eq, .lhs = iv.value, .rhs = zero } });
            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .convert = .{ .value = cmp } }), .ty = ctype.int_t };
        },
        .name => |n| blk: {
            // Resolves to a local/param's alloca slot OR a global's `global_addr`
            // (`resolveName` tries the local env first, so a local shadows a global of the
            // same name). Either way, `bd.value` is now this name's ADDRESS and the
            // load/decay logic below is identical for both.
            const bd = try resolveName(l, n);
            // Array-to-pointer DECAY: an array name used as an rvalue (not the operand of
            // `&`, which goes through `lowerAddr`, never here, or `sizeof`, which goes
            // through `typeOf`/`lvalueTypeOf`, also never here) decays to a pointer to its
            // first element: the array's own address (its alloca slot IS `&arr`, same as any
            // other `.name` lvalue), no load emitted.
            if (bd.ty == .array) break :blk .{ .value = bd.value, .ty = try l.ptrTo(bd.ty.array.elem.*, orQuals(bd.ty.array.quals, bd.quals)) };
            // A function-typed name (a bare function DESIGNATOR used as a value:
            // `fp = add;`, passing `add` as an arg) DECAYS to a pointer to itself, same shape
            // as an array name above, no load (a function has no scalar storage to read).
            // `&add` reaches the identical result through `lowerAddr`'s `.name` ->
            // `resolveName` -> `lowerExpr`'s `.addrof` arm below, which wraps
            // `bd.value`/`bd.ty` the same way.
            if (bd.ty == .func) break :blk .{ .value = bd.value, .ty = try l.ptrTo(bd.ty, .{}) };
            // A bare struct-typed name as an rvalue (`struct P p; p;`) has no scalar value to
            // load. Loading the whole storage blob isn't a thing this frontend supports. It
            // is only valid as the base of `.`/`->` or `&`, both of which go through
            // `lowerAddr`, never here.
            if (bd.ty == .@"struct") break :blk error.Unsupported;
            const t = try l.irTy(bd.ty);
            // A read through a `volatile`-qualified lvalue marks the `load` volatile
            // (`bd.quals` is this object's own qualifiers, from `resolveName`).
            break :blk .{ .value = try f.appendInst(l.block, t, .{ .load = .{ .ptr = bd.value, .@"volatile" = bd.quals.is_volatile } }), .ty = bd.ty };
        },
        .str_lit => |s| blk: {
            // A string literal as an rvalue DECAYS to `char*`, no load, exactly like an
            // array name above: `lowerStrLit`'s address (the array's own address) IS the
            // pointer to its first char.
            const a = try lowerStrLit(l, s);
            break :blk .{ .value = a.value, .ty = try l.ptrTo(a.ty.array.elem.*, a.ty.array.quals) };
        },
        // A wide string literal as an rvalue decays to `int*` (`wchar_t*`) the same way
        // `.str_lit` decays to `char*` just above.
        .wstr_lit => |s| blk: {
            const a = try lowerWStrLit(l, s);
            break :blk .{ .value = a.value, .ty = try l.ptrTo(a.ty.array.elem.*, a.ty.array.quals) };
        },
        // `(type-name){ ... }` as an RVALUE: materialize it (via `lowerAddr`, which owns the
        // one `materializeCompoundLiteral` call site) into its own temp slot, then apply the
        // exact same array-decay/struct-rejection/scalar-load rule `.name` above applies to
        // any other stack object. An array compound literal decays to ptr-to-elem (no load),
        // and a bare struct rvalue is `error.Unsupported` (the same gap as a plain
        // struct-typed name). A struct compound literal stays usable as an lvalue.
        // `&`/`.field`/the decl-init special case below all go through `lowerAddr`, never
        // here. A scalar loads.
        .compound_literal => blk: {
            const bd = try lowerAddr(l, expr);
            if (bd.ty == .array) break :blk .{ .value = bd.value, .ty = try l.ptrTo(bd.ty.array.elem.*, bd.ty.array.quals) };
            if (bd.ty == .@"struct") break :blk error.Unsupported;
            const t = try l.irTy(bd.ty);
            break :blk .{ .value = try f.appendInst(l.block, t, .{ .load = .{ .ptr = bd.value, .@"volatile" = false } }), .ty = bd.ty };
        },
        .addrof => |inner| blk: {
            // No load: `&e` IS `e`'s address, and `lowerAddr` never emits one for `.name`/
            // `.deref` (the two lvalue kinds this handles). `a.quals` is the lvalue's own
            // qualifiers, carried into the new pointer's POINTEE quals: `&const_x` is
            // `const int *`, so a later `*(&const_x) = ...` is caught by `lowerAddr`'s
            // `.deref` arm reading it straight back off `.ptr.quals`.
            const a = try lowerAddr(l, inner);
            // A BITFIELD has no address of its own. C forbids `&s.bf` (gcc: "cannot take
            // address of bit-field"). Without this guard, `a.value` here is the storage
            // UNIT's address, so `&s.bf` would silently yield a pointer to the wrong object
            // (the unit, not the field). Fail closed instead.
            if (a.bitfield != null) break :blk error.BitfieldAddress;
            break :blk .{ .value = a.value, .ty = try l.ptrTo(a.ty, a.quals) };
        },
        .deref => blk: {
            // `*p` as an RVALUE: resolve its address (the pointer's own value, from
            // `lowerAddr`'s `.deref` case) then load through it at the pointee's width.
            const a = try lowerAddr(l, expr);
            // A FUNCTION-typed pointee is not a loadable object: `*fp` is a function
            // DESIGNATOR that decays straight back to the same address (C11 6.3.2.1). Yield the
            // pointer value unchanged, no load, so `(*fp)()` calls through it (the same address
            // `fp()` would). Loading here would read the callee's first instruction as a pointer.
            if (a.ty == .func) break :blk .{ .value = a.value, .ty = a.ty, .quals = a.quals };
            const t = try l.irTy(a.ty);
            // `*p` where `p`'s pointee is `volatile`-qualified marks the load.
            break :blk .{ .value = try f.appendInst(l.block, t, .{ .load = .{ .ptr = a.value, .@"volatile" = a.quals.is_volatile } }), .ty = a.ty };
        },
        .index => blk: {
            // `arr[i]` as an RVALUE: resolve its address (`lowerAddr`'s `.index` case, base +
            // idx*sizeof(elem)) then load through it at the element's width. Same shape as
            // `.deref` above: `arr[i]` is exactly `*(arr + i)`.
            //
            // Array-to-pointer DECAY (mirrors `.name` above): if the indexed element is
            // ITSELF an array (`m[i]` where `m` is 2-D, so `m[i]`'s element type is
            // `int[3]`), `m[i]` is not a scalar load. It decays to a pointer to its own
            // first element, no load emitted. This is what feeds a chained `m[i][j]`'s outer
            // index: `lowerAddr`'s `.index` arm calls `lowerExpr(ix.base)` to get the base
            // pointer for its own scaled-add, so `m[i]` must come back as a pointer (to
            // `int`, scaled by 4) rather than a loaded `int[3]` value.
            const a = try lowerAddr(l, expr);
            if (a.ty == .array) break :blk .{ .value = a.value, .ty = try l.ptrTo(a.ty.array.elem.*, orQuals(a.ty.array.quals, a.quals)) };
            const t = try l.irTy(a.ty);
            // `arr[i]` through a `volatile`-qualified element marks the load.
            break :blk .{ .value = try f.appendInst(l.block, t, .{ .load = .{ .ptr = a.value, .@"volatile" = a.quals.is_volatile } }), .ty = a.ty };
        },
        .member => blk: {
            // `s.field` / `p->field` as an RVALUE: resolve the field's address (`lowerAddr`'s
            // `.member` case above) then load through it at the field's own width, same
            // shape as `.deref`/`.index`. A struct-typed field (a NESTED struct) has no
            // scalar value to load. It's only meaningful as the BASE of another `./->`
            // (which goes through `lowerAddr`, never here), so this fails closed rather than
            // emitting a bogus aggregate load. An array-typed field decays to a pointer to
            // its first element, same as `.name`/`.index` above, for the same reason (no
            // native aggregate load).
            const a = try lowerAddr(l, expr);
            // A BITFIELD member reads through the shift/mask path (`loadBitfield`), not a
            // plain load of its storage unit.
            if (a.bitfield) |bf| break :blk .{ .value = try loadBitfield(l, a.value, a.ty, bf, a.quals.is_volatile), .ty = a.ty };
            if (a.ty == .@"struct") break :blk error.Unsupported; // nested struct value: not supported as an rvalue
            if (a.ty == .array) break :blk .{ .value = a.value, .ty = try l.ptrTo(a.ty.array.elem.*, orQuals(a.ty.array.quals, a.quals)) };
            const t = try l.irTy(a.ty);
            // `s.field`/`p->field` through a `volatile`-qualified field marks the load.
            break :blk .{ .value = try f.appendInst(l.block, t, .{ .load = .{ .ptr = a.value, .@"volatile" = a.quals.is_volatile } }), .ty = a.ty };
        },
        .binary => |b| blk: {
            const lhs = try lowerExpr(l, b.lhs);
            const rhs = try lowerExpr(l, b.rhs);
            // A pointer operand takes the C11 6.5.6 pointer-arithmetic path (scaled by
            // `sizeof(pointee)`) instead of the integer usual-arithmetic-conversions path
            // below. `lowerPtrArith` also fails closed on anything nonsensical (`ptr * int`,
            // `ptr + ptr`, `int - ptr`).
            if (lhs.ty == .ptr or rhs.ty == .ptr) break :blk try lowerPtrArith(l, b.op, lhs, rhs);
            // A float operand (either side) skips straight to the shared int-or-float
            // `commonType`/`convertTo` path below. `<<`/`>>` never apply to a float operand
            // in C, so that's still an int-only special case just below this.
            if (!(lhs.ty.isInt() or lhs.ty.isFloat()) or !(rhs.ty.isInt() or rhs.ty.isFloat())) break :blk error.Unsupported; // array (or other non-int/non-ptr/non-float) rvalue: not yet supported
            // C11 6.5.7p3: `<<`/`>>` promote EACH operand independently, and the result
            // type is the promoted LEFT operand, the right operand's type must not
            // influence the left operand's conversion or the result (unlike every other
            // binary op, which uses the usual arithmetic conversions via `commonType`).
            // The IR's `shr` picks signed-vs-unsigned from its operand type, so routing
            // the left operand through a mixed `commonType` would silently turn an
            // arithmetic shift into a logical one.
            if (b.op == .shl or b.op == .shr) {
                const lt = lhs.ty.promote();
                const lv = try l.convertTo(lhs, lt);
                const rv = try l.convertTo(rhs, lt); // count's own width doesn't affect result signedness
                const ct = try l.irTy(lt);
                break :blk .{ .value = try f.appendInst(l.block, ct, .{ .arith = .{ .op = irBinOp(b.op), .lhs = lv, .rhs = rv } }), .ty = lt };
            }
            const c = ctype.CType.commonType(lhs.ty, rhs.ty, l.layout);
            const lv = try l.convertTo(lhs, c);
            const rv = try l.convertTo(rhs, c);
            const ct = try l.irTy(c);
            break :blk .{ .value = try f.appendInst(l.block, ct, .{ .arith = .{ .op = irBinOp(b.op), .lhs = lv, .rhs = rv } }), .ty = c };
        },
        .compare => |c| blk: {
            const lhs = try lowerExpr(l, c.lhs);
            const rhs = try lowerExpr(l, c.rhs);
            // Pointer compare (`p == q`, `p < q`, ...): both operands are already IR `ptr`
            // values (no scaling/conversion needed, unlike arithmetic), so `icmp` runs
            // directly on them, skip the integer `commonType` path entirely.
            const cmp = if (lhs.ty == .ptr or rhs.ty == .ptr) cmp: {
                if (lhs.ty != .ptr or rhs.ty != .ptr) {
                    // Exactly one side is a pointer (needed for `void *p = 0; p == 0;`):
                    // legal only when the OTHER side is the NULL POINTER CONSTANT (C11
                    // 6.3.2.3p3), a compile-time-zero integer expression. `consteval` folds
                    // it the same way an array dimension/initializer does. A non-constant or
                    // nonzero operand stays unsupported (`p == n` for a runtime int `n` is
                    // still not a thing this frontend does without a cast).
                    const ptr_is_lhs = lhs.ty == .ptr;
                    const other_expr = if (ptr_is_lhs) c.rhs else c.lhs;
                    const other_val = if (ptr_is_lhs) rhs else lhs;
                    const ptr_val = if (ptr_is_lhs) lhs else rhs;
                    const k = consteval.eval(l.allocator, other_expr, l.layout, l.globals) catch break :blk error.Unsupported;
                    if (k != 0) break :blk error.Unsupported;
                    const null_ptr = try l.convertTo(other_val, ptr_val.ty);
                    const lv2 = if (ptr_is_lhs) ptr_val.value else null_ptr;
                    const rv2 = if (ptr_is_lhs) null_ptr else ptr_val.value;
                    break :cmp try f.appendInst(l.block, l.boolt, .{ .icmp = .{ .op = irCmpOp(c.op), .lhs = lv2, .rhs = rv2 } });
                }
                break :cmp try f.appendInst(l.block, l.boolt, .{ .icmp = .{ .op = irCmpOp(c.op), .lhs = lhs.value, .rhs = rhs.value } });
            } else cmp: {
                if (!(lhs.ty.isInt() or lhs.ty.isFloat()) or !(rhs.ty.isInt() or rhs.ty.isFloat())) break :blk error.Unsupported; // array (or other non-int/non-ptr/non-float) rvalue: not yet supported
                // A float operand's `icmp` still compares directly (no separate "fcmp" IR
                // op): the IR derives fcmp-vs-icmp codegen from the OPERANDS' float-vs-int
                // register class, same mechanism as `.arith` deriving fadd-vs-add.
                const common = ctype.CType.commonType(lhs.ty, rhs.ty, l.layout);
                const lv = try l.convertTo(lhs, common);
                const rv = try l.convertTo(rhs, common);
                break :cmp try f.appendInst(l.block, l.boolt, .{ .icmp = .{ .op = irCmpOp(c.op), .lhs = lv, .rhs = rv } });
            };
            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .convert = .{ .value = cmp } }), .ty = ctype.int_t };
        },
        .assign => |a| blk: {
            // `target` is any lvalue (a name or `*p`): resolve its ADDRESS once via
            // `lowerAddr`, then store through it. `.name` is unchanged in effect (the
            // address is just the binding's slot, same as before).
            const dst = try lowerAddr(l, a.target);
            // A write through a `const`-qualified lvalue is a compile error, checked ONCE
            // here, before either the struct-copy path or the scalar store below, so it
            // covers a plain `=`, a compound `op=`, AND a whole-struct assignment identically.
            if (dst.quals.is_const) return error.ConstAssign;
            // A write to a BITFIELD lvalue is a read-modify-write of its storage unit
            // (`storeBitfield`), never a plain store: a plain store would clobber the
            // neighbor bitfields packed alongside it. A compound assign (`s.f += x`) reads the
            // field first (`loadBitfield`), combines, then writes back. The yielded value is the
            // truncated/sign-extended value the field now holds, so it re-reads after the store.
            if (dst.bitfield) |bf| {
                const rhs = try lowerExpr(l, a.value);
                const new_v: Value = if (a.op) |op| nv: {
                    const cur: TypedValue = .{ .value = try loadBitfield(l, dst.value, dst.ty, bf, dst.quals.is_volatile), .ty = dst.ty };
                    if (!(cur.ty.isInt()) or !(rhs.ty.isInt() or rhs.ty.isFloat())) break :blk error.Unsupported;
                    const c = ctype.CType.commonType(cur.ty, rhs.ty, l.layout);
                    const cv = try l.convertTo(cur, c);
                    const rv = try l.convertTo(rhs, c);
                    const ct = try l.irTy(c);
                    const res = try f.appendInst(l.block, ct, .{ .arith = .{ .op = irBinOp(op), .lhs = cv, .rhs = rv } });
                    break :nv try l.convertTo(.{ .value = res, .ty = c }, dst.ty);
                } else try l.convertTo(rhs, dst.ty);
                try storeBitfield(l, dst.value, dst.ty, bf, new_v, dst.quals.is_volatile);
                break :blk .{ .value = try loadBitfield(l, dst.value, dst.ty, bf, dst.quals.is_volatile), .ty = dst.ty };
            }
            // Whole-struct assignment (`s1 = s2`): the rhs is another struct LVALUE (its
            // address, via `lowerAddr`, NOT `lowerExpr`, which fails closed on a bare struct
            // rvalue), copied word-for-word into `dst`. Only meaningful for `=` (there's no
            // such thing as `s1 += s2`) and only between the SAME struct def (C has no
            // structural struct typing, and there's no field-by-field conversion here).
            if (dst.ty == .@"struct") {
                if (a.op != null) break :blk error.Unsupported; // compound assign on a struct: not a thing
                const src = try lowerAddr(l, a.value);
                if (src.ty != .@"struct" or src.ty.@"struct" != dst.ty.@"struct") break :blk error.Unsupported;
                try copyStruct(l, dst.ty.@"struct", src.value, dst.value);
                break :blk .{ .value = dst.value, .ty = dst.ty };
            }
            const rhs = try lowerExpr(l, a.value);
            // Compound assign (`target op= value`), covering int, float (`d += x`,
            // common-type/convert same as `.binary`), and pointer (`p += n`, `p -= n`, scaled
            // by `sizeof(*p)` via the same `lowerPtrArith` path ordinary `p + n` takes;
            // pointers never go through `commonType`, which asserts int-or-float operands,
            // so this must branch off before it). Array/struct targets still fail closed
            // below (neither `isInt` nor `isFloat`).
            const new_v = if (a.op) |op| nv: {
                const bt = try l.irTy(dst.ty);
                // A compound assign (`x += 1`) reads `dst` before combining, so that read is
                // volatile too when `dst` is a `volatile`-qualified lvalue.
                const cur: TypedValue = .{ .value = try f.appendInst(l.block, bt, .{ .load = .{ .ptr = dst.value, .@"volatile" = dst.quals.is_volatile } }), .ty = dst.ty };
                if (dst.ty == .ptr) break :nv (try lowerPtrArith(l, op, cur, rhs)).value;
                if (!(cur.ty.isInt() or cur.ty.isFloat()) or !(rhs.ty.isInt() or rhs.ty.isFloat())) break :blk error.Unsupported; // compound assign on array/struct: not yet supported
                const c = ctype.CType.commonType(cur.ty, rhs.ty, l.layout);
                const cv = try l.convertTo(cur, c);
                const rv = try l.convertTo(rhs, c);
                const ct = try l.irTy(c);
                const res = try f.appendInst(l.block, ct, .{ .arith = .{ .op = irBinOp(op), .lhs = cv, .rhs = rv } });
                break :nv try l.convertTo(.{ .value = res, .ty = c }, dst.ty);
            } else try l.convertTo(rhs, dst.ty);
            // A write through a `volatile`-qualified lvalue marks the store.
            try f.appendStoreVol(l.block, new_v, dst.value, dst.quals.is_volatile);
            break :blk .{ .value = new_v, .ty = dst.ty }; // assignment yields the stored value (C)
        },
        .incdec => |ie| blk: {
            // `++e`/`e++`/`--e`/`e--`: resolve `target`'s address once (same lvalue model
            // `.assign` uses), load the OLD value, compute NEW = OLD +/- 1 (int: `iconst 1`
            // plus `arith`. Float: `fconst 1.0` plus `arith`, which derives fadd/fsub from
            // the operand's float IR type same as everywhere else), or, for a pointer
            // target, OLD scaled by `sizeof(pointee)` via the same `lowerPtrArith` path
            // `p + 1` takes, store NEW back through the address, and yield NEW (prefix) or
            // OLD (postfix): exactly C's value semantics for `++`/`--`.
            const dst = try lowerAddr(l, ie.target);
            // `++`/`--` on a `const`-qualified lvalue is a write, exactly like `.assign`
            // above: reject before the load/arith/store sequence below.
            if (dst.quals.is_const) return error.ConstAssign;
            // `++`/`--` on a BITFIELD reads the field, adds/subtracts one, then writes it
            // back through the read-modify-write path (`storeBitfield`).
            if (dst.bitfield) |bf| {
                if (!dst.ty.isInt()) break :blk error.Unsupported; // a bitfield is always integer
                const bt = try l.irTy(dst.ty);
                const cur = try loadBitfield(l, dst.value, dst.ty, bf, dst.quals.is_volatile);
                const op: parser.BinOp = if (ie.inc) .add else .sub;
                const one = try f.appendInst(l.block, bt, .{ .iconst = 1 });
                const new_v = try f.appendInst(l.block, bt, .{ .arith = .{ .op = irBinOp(op), .lhs = cur, .rhs = one } });
                try storeBitfield(l, dst.value, dst.ty, bf, new_v, dst.quals.is_volatile);
                // Yield the field's post-store value (prefix) or the old value (postfix).
                break :blk if (ie.prefix) TypedValue{ .value = try loadBitfield(l, dst.value, dst.ty, bf, dst.quals.is_volatile), .ty = dst.ty } else TypedValue{ .value = cur, .ty = dst.ty };
            }
            const bt = try l.irTy(dst.ty);
            // `++`/`--` both reads and writes `dst`, so both the old-value load and the
            // new-value store below are volatile when `dst` is.
            const cur: TypedValue = .{ .value = try f.appendInst(l.block, bt, .{ .load = .{ .ptr = dst.value, .@"volatile" = dst.quals.is_volatile } }), .ty = dst.ty };
            const op: parser.BinOp = if (ie.inc) .add else .sub;
            const new_v: Value = if (dst.ty == .ptr) ptrv: {
                const one: TypedValue = .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 1 }), .ty = ctype.int_t };
                break :ptrv (try lowerPtrArith(l, op, cur, one)).value;
            } else if (cur.ty.isFloat()) fv: {
                const one = try f.appendInst(l.block, bt, .{ .fconst = 1.0 });
                break :fv try f.appendInst(l.block, bt, .{ .arith = .{ .op = irBinOp(op), .lhs = cur.value, .rhs = one } });
            } else iv: {
                if (!cur.ty.isInt()) break :blk error.Unsupported; // struct/array target: not supported
                const one = try f.appendInst(l.block, bt, .{ .iconst = 1 });
                break :iv try f.appendInst(l.block, bt, .{ .arith = .{ .op = irBinOp(op), .lhs = cur.value, .rhs = one } });
            };
            try f.appendStoreVol(l.block, new_v, dst.value, dst.quals.is_volatile);
            break :blk if (ie.prefix) TypedValue{ .value = new_v, .ty = dst.ty } else cur;
        },
        .call => |c| blk: {
            // `__builtin_va_start`/`__builtin_va_end`/`__builtin_va_copy` are the
            // pseudo-calls `<stdarg.h>`'s `va_start`/`va_end`/`va_copy` macros expand to (see
            // `stdarg.zig`). They are not real callable symbols, so they're intercepted
            // HERE, before `findFunc`/`findFuncDecl`/the indirect path below ever see them.
            // `__builtin_va_arg` is handled separately, as its own `Expr.va_arg` AST node
            // (parsed specially, see `parser.zig`, since its second argument is a TYPE
            // NAME, not an expression), not as a call at all.
            if (c.callee.* == .name) {
                // `__builtin_bswap16/32/64(x)` reverse a value's bytes. GCC lowers them to a
                // `rev` instruction. VCC has no byteswap IR op, so it emits the equivalent
                // shift/mask/or sequence, which every backend already supports. Glibc's
                // `<bits/byteswap.h>` inline helpers use these.
                if (bswapBytes(c.callee.name)) |bytes| {
                    if (c.args.len != 1) break :blk error.Unsupported;
                    const wide = bytes == 8;
                    const ity = if (wide) l.i64t else l.i32t;
                    const comp_ty = ctype.mkInt(if (wide) .longlong else .int, false);
                    const xv = try l.convertTo(try lowerExpr(l, c.args[0]), comp_ty);
                    const swapped = try lowerBswap(l, xv, bytes, ity);
                    const rty = switch (bytes) {
                        2 => ctype.mkInt(.short, false),
                        4 => ctype.mkInt(.int, false),
                        else => ctype.mkInt(.longlong, false),
                    };
                    break :blk .{ .value = swapped, .ty = rty };
                }
                if (std.mem.eql(u8, c.callee.name, "__builtin_va_start")) {
                    if (c.args.len != 2) break :blk error.Unsupported;
                    // C11 7.15.1: `last` must NAME the last fixed parameter.
                    if (c.args[1].* != .name or l.last_fixed_param == null or
                        !std.mem.eql(u8, c.args[1].name, l.last_fixed_param.?)) break :blk error.Unsupported;
                    const ap_addr = try lowerAddr(l, c.args[0]);
                    try f.appendVaStart(l.block, ap_addr.value);
                    break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                }
                if (std.mem.eql(u8, c.callee.name, "__builtin_va_end")) {
                    if (c.args.len != 1) break :blk error.Unsupported;
                    // Mirrors `__builtin_va_start`'s enclosing-function check above:
                    // `va_end` is meaningless outside a variadic definition.
                    if (l.last_fixed_param == null) break :blk error.Unsupported;
                    const ap_addr = try vaListAddr(l, c.args[0]);
                    try f.appendVaEnd(l.block, ap_addr);
                    break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                }
                if (std.mem.eql(u8, c.callee.name, "__builtin_va_copy")) {
                    if (c.args.len != 2) break :blk error.Unsupported;
                    const dst_addr = try vaListAddr(l, c.args[0]);
                    const src_addr = try vaListAddr(l, c.args[1]);
                    // A `va_list` object has no `StructDef` of its own on every target (e.g.
                    // riscv64/x86's is a bare pointer, not a struct), `copyStruct` only ever
                    // reads `.size` off the def it's given, so a throwaway one carrying just
                    // the per-target size (from `ctype.builtinVaList`, the same source `va_list
                    // ap;` locals size their own alloca from) is enough to byte-copy the whole
                    // object, regardless of its underlying CType shape.
                    const va_list_ty = ctype.builtinVaList(l.layout);
                    const va_list_size = try va_list_ty.sizeInBytes(l.layout);
                    const synthetic_def: ctype.StructDef = .{
                        .name = "__builtin_va_list",
                        .is_union = false,
                        .fields = &.{},
                        .size = va_list_size,
                        .alignment = 8,
                    };
                    try copyStruct(l, &synthetic_def, src_addr, dst_addr);
                    break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                }
            }
            // `c.callee` is a full expression. A `.name` naming a function DEFINITION or
            // DECLARATION in this TU is a direct call (checked first below). Anything else,
            // a `.name` binding an ordinary in-scope variable of function-pointer type,
            // `(*fp)(...)`, `tbl[i](...)`, ..., is an INDIRECT call through the callee's
            // runtime VALUE, handled after.
            if (c.callee.* == .name) {
                const name = c.callee.name;
                // C's innermost-binding-wins shadowing rule means an in-scope local/param
                // binding of `name` must be tried BEFORE the function tables below. This
                // mirrors `resolveName` (used for `name` in any other expression position),
                // which already checks `l.lookup` first. A local function-pointer variable
                // that shadows a file-scope function of the same name must call through the
                // LOCAL (the indirect path below), never direct-call the function it
                // shadows. Only when NO local/param binds this name do the function tables
                // get consulted for a direct call.
                if (l.lookup(name) == null) {
                    // A same-TU call's callee is usually one of this unit's own DEFINITIONS.
                    // Look that up first (a local definition always wins), so each argument
                    // converts to the CALLEE's declared parameter type (not the caller's), and
                    // the call's result carries the callee's actual return type.
                    if (findFunc(l.funcs, name)) |callee| {
                        // A variadic definition needs only its FIXED params matched exactly.
                        // Anything past that is a variadic argument, lowered by
                        // `lowerVariadicCallArgs` (default-argument-promoted, not converted to
                        // any declared parameter type, since there isn't one).
                        if (callee.is_variadic) {
                            if (c.args.len < callee.params.len) break :blk error.CallArityMismatch;
                            const num_fixed: u32 = @intCast(callee.params.len);
                            var args = try lowerVariadicCallArgs(l, c.args, callee.params, num_fixed);
                            defer args.deinit(l.allocator);
                            // Mirrors the non-variadic case below: a `void`-returning
                            // variadic DEFINITION yields no value.
                            if (callee.ret == .void_) {
                                try f.appendVoidCallV(l.block, name, args.items, num_fixed);
                                break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                            }
                            break :blk try directCallResult(l, name, callee.ret, args.items, num_fixed);
                        }
                        if (c.args.len != callee.params.len) break :blk error.CallArityMismatch;
                        var args = try lowerCallArgs(l, c.args, callee.params);
                        defer args.deinit(l.allocator);
                        // A `void`-returning DEFINITION (the real `void_` `CType`, not the
                        // bare-`void` special case's `int_t` placeholder) yields no value.
                        // This mirrors the bodyless void-prototype case below
                        // (`decl.ret == null`): `appendVoidCall` (a statement, no result
                        // operand) and a discarded placeholder `int 0` for this expression's
                        // `TypedValue`, same as `.cast_void`.
                        if (callee.ret == .void_) {
                            try f.appendVoidCall(l.block, name, args.items);
                            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                        }
                        break :blk try directCallResult(l, name, callee.ret, args.items, null);
                    }
                    // Not defined here: it may be a bodyless DECLARATION, a prototype or
                    // `extern` one, for a function defined in ANOTHER object or `.so`. It
                    // lowers to the SAME `appendCall`, but `name` stays undefined in this
                    // module, so the object writer emits it `SHN_UNDEF` and the linker binds
                    // it (a `.so` PLT import).
                    if (findFuncDecl(l.func_decls, name)) |decl| {
                        // Mirrors the definition case above: a variadic prototype needs
                        // only its fixed params matched exactly.
                        if (decl.is_variadic) {
                            if (c.args.len < decl.params.len) break :blk error.CallArityMismatch;
                            const num_fixed: u32 = @intCast(decl.params.len);
                            var args = try lowerVariadicCallArgs(l, c.args, decl.params, num_fixed);
                            defer args.deinit(l.allocator);
                            if (decl.ret) |ret_ty| break :blk try directCallResult(l, name, ret_ty, args.items, num_fixed);
                            try f.appendVoidCallV(l.block, name, args.items, num_fixed);
                            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                        }
                        // Non-variadic: require EXACT arity against the prototype's params (a
                        // mismatch is a clear error, not a silent truncation).
                        if (c.args.len != decl.params.len) break :blk error.CallArityMismatch;
                        var args = try lowerCallArgs(l, c.args, decl.params);
                        defer args.deinit(l.allocator);
                        // `ret == null` is a `void`-returning prototype: the call yields no value,
                        // so it's an `appendVoidCall` (a statement) and this expression's
                        // `TypedValue` is a discarded placeholder `int 0`, mirroring
                        // `.cast_void`, whose result a statement-expr caller (`lowerStmt`'s
                        // `.expr` arm) always throws away too.
                        if (decl.ret) |ret_ty| break :blk try directCallResult(l, name, ret_ty, args.items, null);
                        try f.appendVoidCall(l.block, name, args.items);
                        break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                    }
                    // `name` is neither a function definition nor a declaration, and no
                    // local/param binds it either: it may still be a FILE-SCOPE global
                    // variable of function-pointer type, handled by the indirect path below,
                    // which resolves it as an ordinary rvalue.
                    if (l.globals.get(name) == null) {
                        // An otherwise-unknown NAME that is CALLED is an IMPLICIT function
                        // declaration `extern int name()`, the behavior of the gcc version VCC
                        // advertises (`__GNUC__` 4.2, pre-C99-strict). Every argument gets the
                        // default argument promotions (no prototype to match), which is also the
                        // ABI an unprototyped call uses. The linker resolves `name`. Real code
                        // (gnulib) calls a handful of libc functions this way.
                        const no_params: []const ctype.CType = &.{};
                        var iargs = try lowerVariadicCallArgs(l, c.args, no_params, 0);
                        defer iargs.deinit(l.allocator);
                        break :blk try directCallResult(l, name, ctype.int_t, iargs.items, 0);
                    }
                }
                // else: `name` IS an in-scope local/param binding, it shadows any file-scope
                // function of the same name, so it MUST fall through to the indirect path
                // below rather than ever reaching `findFunc`/`findFuncDecl` above.
            }
            // Indirect call: the callee is an ordinary expression whose runtime VALUE must be
            // a function pointer (a local/global variable, `*fp`, `tbl[i]`, ...). Typecheck
            // args against the pointee `FuncType`, exact arity (variadic handled below), then
            // emit `call_indirect` through the callee's own value.
            const callee_tv = try lowerExpr(l, c.callee);
            // The callee value is the function's ADDRESS whether `c.callee` had pointer-to-
            // function type (`fp()`) or function type (`(*fp)()`, where `*fp` is a function
            // designator that decays straight back to the same address). Accept both.
            const ft = switch (callee_tv.ty) {
                .func => |fnt| fnt,
                else => blk2: {
                    const pointee = callee_tv.ty.pointee() orelse break :blk error.Unsupported; // not a pointer at all
                    if (pointee.* != .func) break :blk error.Unsupported; // a pointer, but not to a function
                    break :blk2 pointee.func;
                },
            };
            // Mirrors the direct-call cases above: a variadic function pointer needs only
            // its fixed params matched exactly.
            if (ft.is_variadic) {
                if (c.args.len < ft.params.len) break :blk error.CallArityMismatch;
                const num_fixed: u32 = @intCast(ft.params.len);
                var args = try lowerVariadicCallArgs(l, c.args, ft.params, num_fixed);
                defer args.deinit(l.allocator);
                // A VOID-returning variadic function pointer is a statement call (no result),
                // mirroring the void direct-call path.
                if (ft.ret == null) {
                    try f.appendVoidCallIndirectV(l.block, callee_tv.value, args.items, num_fixed);
                    break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                }
                break :blk try indirectCallResult(l, callee_tv.value, ft.ret.?.*, args.items, num_fixed);
            }
            if (c.args.len != ft.params.len) break :blk error.CallArityMismatch;
            var args = try lowerCallArgs(l, c.args, ft.params);
            defer args.deinit(l.allocator);
            // A void-returning function pointer (`void (*fp)(void)`) is a statement call
            // with a discarded placeholder result, the same as a void direct call. Gnulib's
            // `error.c` calls the `error_print_progname` hook this way.
            if (ft.ret == null) {
                try f.appendVoidCallIndirect(l.block, callee_tv.value, args.items);
                break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
            }
            break :blk try indirectCallResult(l, callee_tv.value, ft.ret.?.*, args.items, null);
        },
        .logand => |b| blk: {
            const slot = try l.allocSlot(ctype.int_t);
            const zero0 = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 });
            try f.appendStore(l.block, zero0, slot); // default: 0 (lhs false)
            const lc = try truthy(l, try lowerExpr(l, b.lhs));
            const rhs_b = try f.appendBlock();
            const merge_b = try f.appendBlock();
            try f.appendIf(l.block, lc, .{ .target = rhs_b }, .{ .target = merge_b });
            l.block = rhs_b;
            const rc = try truthy(l, try lowerExpr(l, b.rhs));
            const ri = try f.appendInst(l.block, l.i32t, .{ .convert = .{ .value = rc } });
            try f.appendStore(l.block, ri, slot);
            try f.setJump(l.block, merge_b, &.{});
            l.block = merge_b;
            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .load = .{ .ptr = slot } }), .ty = ctype.int_t };
        },
        .logor => |b| blk: {
            const slot = try l.allocSlot(ctype.int_t);
            const one1 = try f.appendInst(l.block, l.i32t, .{ .iconst = 1 });
            try f.appendStore(l.block, one1, slot); // default: 1 (lhs true, short-circuit)
            const lc = try truthy(l, try lowerExpr(l, b.lhs));
            const rhs_b = try f.appendBlock();
            const merge_b = try f.appendBlock();
            try f.appendIf(l.block, lc, .{ .target = merge_b }, .{ .target = rhs_b }); // lhs true -> merge(1); false -> eval rhs
            l.block = rhs_b;
            const rc = try truthy(l, try lowerExpr(l, b.rhs));
            const ri = try f.appendInst(l.block, l.i32t, .{ .convert = .{ .value = rc } });
            try f.appendStore(l.block, ri, slot);
            try f.setJump(l.block, merge_b, &.{});
            l.block = merge_b;
            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .load = .{ .ptr = slot } }), .ty = ctype.int_t };
        },
        .ternary => |t| blk: {
            // Allocate the result slot BEFORE lowering `cond` (like logand/logor): `cond`'s
            // `if` may land directly in the current block (e.g. A ternary as the function's
            // first expression puts it in `entry`), and an `if` must be the last thing
            // appended to whatever block holds it. An alloca appended to entry afterward
            // would land after it. `allocSlot` always targets entry regardless of `l.block`,
            // so it must run first. Its width isn't known until both arms are lowered, so it
            // starts as `int_t` and is widened in place below if `commonType` turns out wider
            // (every all-`int` program takes the no-op branch: `common == int_t` already).
            const slot = try l.allocSlot(ctype.int_t);
            const cc = try truthy(l, try lowerExpr(l, t.cond));
            const then_b = try f.appendBlock();
            const else_b = try f.appendBlock();
            const merge_b = try f.appendBlock();
            try f.appendIf(l.block, cc, .{ .target = then_b }, .{ .target = else_b });

            l.block = then_b;
            const tv = try lowerExpr(l, t.then);
            const then_end = l.block; // lowering `then` may itself open/move through blocks

            l.block = else_b;
            const ev = try lowerExpr(l, t.els);
            const else_end = l.block;

            // The common type of the two arms (C11 6.5.15p6, common-case subset): both integer
            // uses the usual arithmetic conversions. A pointer arm (with the other a pointer or
            // a null constant) yields the pointer type. Both float takes the wider. The result
            // slot is retyped to it in place (it started as `int_t`). `convertTo` handles the
            // int<->int/int<->ptr/ptr<->ptr coercions each arm needs.
            const common: ctype.CType = cmn: {
                if (tv.ty.isInt() and ev.ty.isInt()) break :cmn ctype.CType.commonType(tv.ty, ev.ty, l.layout);
                if (tv.ty == .ptr) break :cmn tv.ty;
                if (ev.ty == .ptr) break :cmn ev.ty;
                if (tv.ty.isFloat() and ev.ty.isFloat()) {
                    break :cmn if (tv.ty.float == .f64 or ev.ty.float == .f64) .{ .float = .f64 } else tv.ty;
                }
                break :blk error.Unsupported; // struct/array ternary arms: not yet supported
            };
            const elem = try l.irTy(common);
            if (f.definingInst(slot)) |alloc_inst| f.opcodeMut(alloc_inst).* = .{ .alloca = .{ .elem = elem } };

            l.block = then_end;
            const conv_then = try l.convertTo(tv, common);
            try f.appendStore(l.block, conv_then, slot);
            try f.setJump(l.block, merge_b, &.{});

            l.block = else_end;
            const conv_els = try l.convertTo(ev, common);
            try f.appendStore(l.block, conv_els, slot);
            try f.setJump(l.block, merge_b, &.{});

            l.block = merge_b;
            const ct = try l.irTy(common);
            break :blk .{ .value = try f.appendInst(l.block, ct, .{ .load = .{ .ptr = slot } }), .ty = common };
        },
        .sizeof_type => |ct| blk: {
            const bytes: i64 = @intCast(try ct.sizeInBytes(l.layout));
            const it = try l.irTy(ctype.ulong_t);
            break :blk .{ .value = try f.appendInst(l.block, it, .{ .iconst = bytes }), .ty = ctype.ulong_t };
        },
        .sizeof_expr => |inner| blk: {
            // `sizeof` is unevaluated: only `inner`'s TYPE is needed, computed statically by
            // `typeOf` WITHOUT lowering `inner` to IR, no side effects, no emitted instructions.
            const ct = try typeOf(l, inner);
            const bytes: i64 = @intCast(try ct.sizeInBytes(l.layout));
            const it = try l.irTy(ctype.ulong_t);
            break :blk .{ .value = try f.appendInst(l.block, it, .{ .iconst = bytes }), .ty = ctype.ulong_t };
        },
        .cast => |c| blk: {
            // `(type)e`: lower the operand, then convert it to the cast's target type.
            // `convertTo` covers int<->int, float<->float, and int<->pointer. Anything
            // else (array/struct) still fails closed.
            const v = try lowerExpr(l, c.operand);
            break :blk .{ .value = try l.convertTo(v, c.target), .ty = c.target };
        },
        .cast_void => |inner| blk: {
            // `(void)e`: evaluate for side effects only, result unused. There's no genuine
            // `void` `CType` to carry as this "expression"'s type, so it reports as a plain
            // `int` (never actually observed, a statement-expr's `TypedValue` is always
            // discarded by its caller, `lowerStmt`'s `.expr` arm).
            _ = try lowerExpr(l, inner);
            break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
        },
        .va_arg => |va| blk: {
            // `__builtin_va_arg(ap, ty)`: resolve `ap`'s ADDRESS (same as
            // `__builtin_va_start`/`_end`, `lowerAddr` not `lowerExpr`. The IR op's `list`
            // is the `va_list` object's address, not its value), then fetch the next
            // argument of `va.ty` from it.
            // `va_arg` needs only a valid `va_list` operand, NOT a variadic enclosing
            // function. A function that receives a `va_list` PARAMETER and reads it (gnulib's
            // `version_etc_arn(FILE *, ..., va_list authors)`) is not itself variadic, so a
            // `l.last_fixed_param == null` guard here would be too strict. `va_start` still
            // needs the last fixed parameter, but `va_arg` does not.
            const ap_addr = try vaListAddr(l, va.ap);
            const rt = try l.irTy(va.ty);
            break :blk .{ .value = try f.appendVaArg(l.block, ap_addr, rt), .ty = va.ty };
        },
        // `({ stmt* })` (GNU statement expression): lower every statement but the last in
        // its own scope, exactly like an ordinary `{ }` block (`lowerBlock`). The LAST one
        // is special: if it's an expression-statement, ITS value is the whole statement-
        // expression's value (no extra load/store, just `lowerExpr` on it directly);
        // anything else (a `return`/declaration/loop/... As the final statement) has no
        // expression value, so this reports the same discarded `int 0` placeholder
        // `.cast_void` uses (never actually observed. See there). A statement TERMINATING
        // control flow (`return`/`break`/`continue`) before reaching the last one leaves no
        // reachable tail to compute a value from AND no reachable block left to append a
        // placeholder into. This is out of scope (fails closed) rather than emitting into a
        // sealed block, which would build invalid IR.
        .stmt_expr => |stmts| blk: {
            const mark = l.scopeMark();
            defer l.popScope(mark);
            if (stmts.len == 0) break :blk .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
            for (stmts[0 .. stmts.len - 1]) |s| {
                if (try lowerStmt(l, s)) break :blk error.Unsupported; // terminated before the tail
            }
            const last = stmts[stmts.len - 1];
            break :blk switch (last) {
                .expr => |e| try lowerExpr(l, e),
                else => tail: {
                    if (try lowerStmt(l, last)) break :blk error.Unsupported;
                    break :tail .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
                },
            };
        },
        // `lhs, rhs` (the comma OPERATOR): evaluate `lhs` for its side effects and discard
        // the result, then evaluate `rhs`. Its `TypedValue` (value AND type) becomes the
        // whole comma expression's.
        .comma => |c| blk: {
            _ = try lowerExpr(l, c.lhs);
            break :blk try lowerExpr(l, c.rhs);
        },
    };
}

/// `ptr + idx*size` (an IR `ptr`-typed add): convert `idx` to `long`, multiply by `size`
/// (bytes), and add to `ptr`. This is `arr[i]`'s address (`lowerAddr`'s `.index` arm), the
/// same scaled-add shape as `lowerPtrArith`'s `ptr + int` case below, but taking the
/// element size directly rather than re-deriving it from a `CType.pointee()` (the caller
/// already resolved that to pick `size`).
fn scaledAdd(l: *L, ptr: Value, idx: TypedValue, size: u64) Error!Value {
    const f = l.func;
    const i64t = try l.irTy(ctype.long_t);
    const size_v = try f.appendInst(l.block, i64t, .{ .iconst = @intCast(size) });
    const idx_v = try l.convertTo(idx, ctype.long_t);
    const scaled = try f.appendInst(l.block, i64t, .{ .arith = .{ .op = .mul, .lhs = idx_v, .rhs = size_v } });
    return f.appendInst(l.block, l.ptrt, .{ .arith = .{ .op = .add, .lhs = ptr, .rhs = scaled } });
}

/// C11 6.5.6 pointer arithmetic, reached from `.binary` whenever either operand is a
/// pointer: `ptr + int`/`int + ptr` and `ptr - int` scale the integer operand by the
/// pointee's `sizeInBytes` and add/sub it to the pointer's raw (`ptr`-typed) value, result
/// typed as the pointer. `ptr - ptr` (both pointing to the same type) takes the raw byte
/// difference and divides by the pointee's size to get the element distance, result typed
/// `long` (`ptrdiff_t`). The IR's `arith` verifier explicitly allows `add`/`sub` between a
/// `ptr` operand and a mismatched-width `int` operand (`pointerArith` in the IR verifier).
/// That's exactly the shape emitted here, so no extra `convert` is needed for the pointer
/// side. Anything else with a pointer operand (`ptr + ptr`, `int - ptr`, `ptr * int`, ptr-ptr
/// of different pointee types, ...) has no C meaning here and fails closed with
/// `error.Unsupported` rather than emitting nonsense IR.
fn lowerPtrArith(l: *L, op: parser.BinOp, lhs: TypedValue, rhs: TypedValue) Error!TypedValue {
    const f = l.func;
    const i64t = try l.irTy(ctype.long_t);
    if (op == .sub and lhs.ty == .ptr and rhs.ty == .ptr) {
        const pa = lhs.ty.pointee() orelse return error.Unsupported;
        const pb = rhs.ty.pointee() orelse return error.Unsupported;
        if (!pa.eql(pb.*)) return error.Unsupported; // diffing pointers to different types
        const bytes: i64 = @intCast(try pa.storageSize(l.layout));
        const size_v = try f.appendInst(l.block, i64t, .{ .iconst = bytes });
        const diff = try f.appendInst(l.block, i64t, .{ .arith = .{ .op = .sub, .lhs = lhs.value, .rhs = rhs.value } });
        const elems = try f.appendInst(l.block, i64t, .{ .arith = .{ .op = .div, .lhs = diff, .rhs = size_v } });
        return .{ .value = elems, .ty = ctype.long_t };
    }
    if (op != .add and op != .sub) return error.Unsupported; // e.g. `ptr * int`
    // `int + ptr` commutes to `ptr + int`. `int - ptr` does not (no such C operator), so it
    // falls through to the `!int_side.ty.isInt()` check below via `rhs` staying the (ptr)
    // right-hand side.
    const sides = if (lhs.ty == .ptr) .{ lhs, rhs } else if (rhs.ty == .ptr and op == .add) .{ rhs, lhs } else return error.Unsupported;
    const ptr_side, const int_side = sides;
    if (!int_side.ty.isInt()) return error.Unsupported; // e.g. `ptr + ptr`
    const pointee = ptr_side.ty.pointee() orelse return error.Unsupported;
    const bytes: i64 = @intCast(try pointee.storageSize(l.layout));
    const size_v = try f.appendInst(l.block, i64t, .{ .iconst = bytes });
    const idx = try l.convertTo(int_side, ctype.long_t);
    const scaled = try f.appendInst(l.block, i64t, .{ .arith = .{ .op = .mul, .lhs = idx, .rhs = size_v } });
    const res = try f.appendInst(l.block, l.ptrt, .{ .arith = .{ .op = irBinOp(op), .lhs = ptr_side.value, .rhs = scaled } });
    return .{ .value = res, .ty = ptr_side.ty };
}

/// Statically compute an expression's `CType` WITHOUT lowering it to IR (no instructions
/// emitted, no side effects). Used only by `sizeof`, which per C is unevaluated. Mirrors
/// `lowerExpr`'s per-arm result-type computation. Keep the two in sync. A divergence here
/// would make `sizeof expr` disagree with the type an actual evaluation of `expr` would
/// produce (and thus with gcc).
/// The ADDRESS of the `va_list` OBJECT that `ap` names, for `va_arg`/`va_end`/`va_copy`.
/// The right value depends on the target's `va_list` shape AND whether `ap` is a local or a
/// passed-in parameter:
///   - array `va_list` (aarch64/x86_64, `__va_list_tag[1]`): the object is the STRUCT.
///       - a LOCAL `va_list ap;` has ARRAY type. The array decays to `&ap[0]` = the struct
///         address, so `lowerAddr` is right.
///       - a `va_list` PARAMETER is C-adjusted to a POINTER (`parseParams` decays it), and its
///         VALUE already IS the struct address (the caller decayed its own array to a pointer),
///         so `lowerExpr`, loading the pointer, is right. `lowerAddr` here would instead give
///         `&ap` (the pointer SLOT), which `va_arg` misreads as a `va_list` struct and crashes.
///         This is the case gnulib's `version_etc_va (..., va_list authors)` hits (`--version`).
///   - pointer `va_list` (riscv64/x86, a bare pointer): the object IS the pointer, and `va_arg`
///     steps it in place, so it always wants the pointer VARIABLE's address (`lowerAddr`),
///     local or parameter alike.
fn vaListAddr(l: *L, ap: *const parser.Expr) Error!Value {
    const target_is_array = switch (ctype.builtinVaList(l.layout)) {
        .array => true,
        else => false,
    };
    const ap_is_array = switch (try typeOf(l, ap)) {
        .array => true,
        else => false,
    };
    if (target_is_array and !ap_is_array) {
        // A decayed `va_list` parameter: its value is the struct address.
        return (try lowerExpr(l, ap)).value;
    }
    return (try lowerAddr(l, ap)).value;
}

fn typeOf(l: *L, expr: *const parser.Expr) Error!ctype.CType {
    return switch (expr.*) {
        .int_lit => |v| v.ty,
        .float_lit => |v| v.ty,
        .name => |n| try resolveNameType(l, n),
        .str_lit => |s| try strLitArrayType(l, s),
        .wstr_lit => |s| try wstrLitArrayType(l, s),
        .negate => |inner| negt: {
            const t = try typeOf(l, inner);
            if (t.isFloat()) break :negt t; // `-d` keeps the operand's float type (see lowerExpr)
            if (!t.isInt()) break :negt error.Unsupported;
            break :negt t.promote();
        },
        .complement => |inner| compt: {
            const t = try typeOf(l, inner);
            if (!t.isInt()) break :compt error.Unsupported;
            break :compt t.promote();
        },
        .lognot, .compare, .logand, .logor => ctype.int_t,
        // `sizeof(&e)` doesn't care about `e`'s qualifiers (a pointer's size is the same
        // regardless), so this unevaluated path passes empty `quals` rather than threading
        // `lvalueTypeOf` through the extra plumbing `lowerAddr`'s real (IR-emitting) path
        // needs for the const-check itself.
        .addrof => |inner| try l.ptrTo(try lvalueTypeOf(l, inner), .{}),
        .deref, .index, .member => try lvalueTypeOf(l, expr),
        .binary => |b| bin: {
            const lt = try typeOf(l, b.lhs);
            const rt = try typeOf(l, b.rhs);
            if (!(lt.isInt() or lt.isFloat()) or !(rt.isInt() or rt.isFloat())) break :bin error.Unsupported;
            break :bin if (b.op == .shl or b.op == .shr) lt.promote() else ctype.CType.commonType(lt, rt, l.layout);
        },
        .assign => |a| try lvalueTypeOf(l, a.target),
        .incdec => |ie| try lvalueTypeOf(l, ie.target),
        .call => |c| blk: {
            // Same lookup-first classification as `lowerExpr`'s `.call` arm, kept in
            // lock-step: an in-scope local/param binding of `name` (checked FIRST,
            // mirroring `resolveName`'s shadowing order) wins over a file-scope function of
            // the same name, so it must fall through to the indirect path below instead of
            // ever reaching `findFunc`/`findFuncDecl`.
            if (c.callee.* == .name) {
                const name = c.callee.name;
                if (l.lookup(name) == null) {
                    // A local DEFINITION first, then a bodyless DECLARATION (an external).
                    // A `void`-returning prototype (`ret == null`) reports as `int`, like
                    // `.cast_void` below, since there's no genuine `void` `CType` and a
                    // void call's `typeOf` is never actually consumed (its value is always
                    // discarded).
                    if (findFunc(l.funcs, name)) |callee| break :blk callee.ret;
                    if (findFuncDecl(l.func_decls, name)) |decl| break :blk decl.ret orelse ctype.int_t;
                    // An undeclared call is an implicit `int name()` (see `lowerExpr`'s
                    // `.call` arm), so its type is `int`.
                    if (l.globals.get(name) == null) break :blk ctype.int_t;
                }
            }
            // Indirect call: same pointee-`FuncType` rule as `lowerExpr`'s `.call` arm.
            // `sizeof(fp(...))`'s type is the callee's function-pointer pointee's declared
            // return type (`int` in place of a genuine `void`, same convention as the
            // direct-call arms just above).
            const ct = try typeOf(l, c.callee);
            const pointee = ct.pointee() orelse break :blk error.Unsupported;
            if (pointee.* != .func) break :blk error.Unsupported;
            break :blk if (pointee.func.ret) |r| r.* else ctype.int_t;
        },
        // A ternary's type is the common type of its two arms, the same rule `lowerExpr`'s
        // `.ternary` arm applies (widened to cover both pointer and float arms), so
        // `(c ? p : q)->field` and `sizeof (c ? a : b)` agree with an actual evaluation.
        .ternary => |t| tern: {
            const tt = try typeOf(l, t.then);
            const et = try typeOf(l, t.els);
            if (tt.isInt() and et.isInt()) break :tern ctype.CType.commonType(tt, et, l.layout);
            if (tt == .ptr) break :tern tt;
            if (et == .ptr) break :tern et;
            if (tt.isFloat() and et.isFloat()) break :tern if (tt.float == .f64 or et.float == .f64) .{ .float = .f64 } else tt;
            break :tern error.Unsupported;
        },
        .sizeof_type, .sizeof_expr => ctype.ulong_t,
        // A cast is always an rvalue (`sizeof (int)x` is unusual C, but its type is simply
        // the cast's own target, see `Expr.cast`'s doc comment).
        .cast => |c| c.target,
        .cast_void => ctype.int_t,
        // `sizeof(__builtin_va_arg(ap, ty))`: the resolved type is right on the node (parsed
        // at parse time, same as `sizeof_type`/`cast`'s targets), no need to recurse into
        // `ap` at all.
        .va_arg => |va| va.ty,
        // `sizeof(({ ... }))`: the same "last expression-statement's type, else `int`
        // (unobserved)" rule `lowerExpr`'s `.stmt_expr` arm uses, computed here WITHOUT
        // lowering any of the statements (an unevaluated context, same as `sizeof_expr`).
        .stmt_expr => |stmts| if (stmts.len != 0 and stmts[stmts.len - 1] == .expr)
            try typeOf(l, stmts[stmts.len - 1].expr)
        else
            ctype.int_t,
        // `sizeof((type-name){ ... })`: a compound literal is an lvalue of its own declared
        // type. The type is right on the node, no lowering needed (same as `.cast`'s target
        // just above).
        .compound_literal => |cl| cl.ty,
        // `sizeof(lhs, rhs)`: `lhs` is unevaluated for `typeOf`'s purposes (see
        // `lowerExpr`'s `.comma` arm. `sizeof` never lowers anything anyway). `rhs`'s type
        // is the whole comma expression's, matching `lowerExpr`.
        .comma => |c| try typeOf(l, c.rhs),
    };
}

/// Emit a zero constant of `ct`'s IR type into block `blk`: `fconst 0.0` for a FLOAT CType,
/// else `iconst 0`. A float-typed `iconst` is invalid IR, it only happens to yield the right
/// VALUE on the aarch64 host (bit pattern 0 == 0.0) and is REJECTED by riscv64 isel, so a
/// float zero must be a real `fconst`. Shared by the fall-off-end / dead-block return seals
/// (which target a specific block), `truthy`, and `.lognot` (both comparing a value against a
/// zero of its own, possibly-float, type so the icmp/fcmp is same-type-to-same-type).
fn zeroInto(l: *L, blk: Block, ct: ctype.CType) Error!Value {
    const t = try l.irTy(ct);
    return if (ct.isFloat())
        l.func.appendInst(blk, t, .{ .fconst = 0.0 })
    else
        l.func.appendInst(blk, t, .{ .iconst = 0 });
}

/// Seal block `blk` with a `ret` matching the function's return type `ret_ty`: a real zero
/// VALUE (`zeroInto`) for any non-void return type, or a VALUELESS `ret` when `ret_ty` is
/// `void_`. There is no zero value of type void, so `zeroInto` would report
/// `error.VoidValue`. The IR terminator's `ret` operand is already `?Value` (an unset one
/// means an implicit `ret void`, see `function.Terminator`'s doc comment), so a valueless
/// `ret` is ordinary IR, not a new shape. Every backend already handles it. Shared by the
/// fall-off-the-end seal, the two dead-block seals in the `.if_` arm (an unused `else`, and an
/// unreachable merge after both arms terminate), and a valueless `return;` statement itself
/// (the `.ret` arm's `null`-operand case).
fn sealRet(l: *L, blk: Block, ret_ty: ctype.CType) Error!void {
    if (ret_ty == .void_) {
        l.func.setTerminator(blk, .{ .ret = ir.function.Ret.none() });
        return;
    }
    // A struct-returning function that falls off the end (or `return;`s with no value)
    // still needs a well-typed terminator, so seal it. A `.registers` struct seals with N
    // indeterminate zero eightbytes. A `.sret` struct returns the hidden result pointer
    // with NO copy (an sret function that falls off the end wrote nothing, C UB either
    // way. This only keeps the IR valid). Both keep the terminator well-typed.
    if (ret_ty == .@"struct") {
        if (abi.classify(ret_ty, l.layout).ret == .sret) {
            l.func.setTerminator(blk, .{ .ret = ir.function.Ret.one(l.sret_ptr.?) });
            return;
        }
        // Seal with one indeterminate zero per eightbyte AT ITS OWN CLASS: an `iconst 0`
        // for an integer eightbyte, an `fconst 0.0` for an `.sse` one, so the ret value
        // types match the register banks the placement routes them to.
        const seal_ebs = switch (abi.classify(ret_ty, l.layout).ret) {
            .registers => |ebs| ebs,
            .sret => unreachable, // handled above
        };
        if (l.layout.arch == .x86) return error.Unsupported; // i386 never returns in registers
        var vals: [4]Value = undefined;
        for (seal_ebs.slice(), 0..) |eb, i| {
            vals[i] = if (eb.class == .sse)
                try l.func.appendInst(blk, try floatEightbyteType(l, eb), .{ .fconst = 0.0 })
            else
                try l.func.appendInst(blk, l.i64t, .{ .iconst = 0 });
        }
        l.func.setTerminator(blk, .{ .ret = ir.function.Ret.many(vals[0..seal_ebs.count]) });
        return;
    }
    const zero = try zeroInto(l, blk, ret_ty);
    l.func.setTerminator(blk, .{ .ret = ir.function.Ret.one(zero) });
}

/// Fill `out` with one `RetPiece` per return-register eightbyte of a struct-BY-VALUE
/// `.registers` return (integer and float) and return the count (1 to 4). Each piece
/// records its BANK (an `.sse` eightbyte is a floating register, else integer), its byte OFFSET
/// in the struct, and its WIDTH, so the backend's post-call store lands each return register in
/// the right dest slot at the right size. This wires the register-return placement on aarch64
/// (x0:x1), x86_64 (rax:rdx), and riscv64 (a0:a1), and extends each to the FP return bank
/// (v0..v3 / xmm0:xmm1 / fa0:fa1). I386 never reaches this arm (`classifyX86` returns `.sret` for
/// every struct), so the 32-bit x86 backend keeps failing closed as a defensive guard. Every
/// `.sret` return is dispatched BEFORE this helper, so its arm here is a defensive guard too.
fn structRetPieces(l: *L, ret_ty: ctype.CType, out: *[4]ir.function.RetPiece) Error!u8 {
    switch (abi.classify(ret_ty, l.layout).ret) {
        .registers => |ebs| {
            if (l.layout.arch == .x86) return error.Unsupported;
            for (ebs.slice(), 0..) |eb, i| out[i] = .{ .fp = eb.class == .sse, .offset = eb.offset, .bytes = eb.bytes };
            return @intCast(ebs.slice().len);
        },
        .sret => return error.Unsupported,
    }
}

/// Build a DIRECT call and yield its result `TypedValue`. A scalar/pointer return is the
/// ordinary `appendCall`/`appendCallV` rvalue, unchanged. A struct return classified
/// `.registers` allocates a fresh destination slot, builds the call with `ret_dest`/
/// `ret_regs` so the backend stores the return registers into it AFTER the call, and yields the
/// slot ADDRESS (ty = the struct), the frontend's struct-rvalue representation, usable as an
/// assignment source, a `.member` base, or a by-value argument (`lowerAddr`'s `.call` case wraps
/// this). `variadic_fixed` is the callee's fixed-param count for a variadic call, else null. A
/// variadic struct-returning call is untested and fails closed. Void returns never reach here
/// (their call sites emit `appendVoidCall` before calling this).
fn directCallResult(l: *L, name: []const u8, ret_ty: ctype.CType, args: []const Value, variadic_fixed: ?u32) Error!TypedValue {
    const f = l.func;
    // A prototype spelled `void f(...)` carries an explicit `void_` return (distinct from the
    // `ret == null` void the call sites already special-case). Emit a void call and yield a
    // discarded placeholder `int 0`, so the value is never materialized from a void type.
    if (ret_ty == .void_) {
        if (variadic_fixed) |nf| try f.appendVoidCallV(l.block, name, args, nf) else try f.appendVoidCall(l.block, name, args);
        return .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
    }
    if (ret_ty == .@"struct") {
        if (variadic_fixed != null) return error.Unsupported; // variadic struct return: not yet supported
        // A `.sret` struct (over 16 bytes) comes back through a hidden result pointer.
        // Allocate the destination slot and PREPEND its address as `args[0]`, then build the
        // call with `Call.sret` (no scalar result). The callee writes the struct through the
        // pointer, so the slot IS the call's lvalue.
        if (abi.classify(ret_ty, l.layout).ret == .sret) {
            const slot = try l.allocSlot(ret_ty);
            const new_args = try l.allocator.alloc(Value, args.len + 1);
            defer l.allocator.free(new_args);
            new_args[0] = slot;
            std.mem.copyForwards(Value, new_args[1..], args);
            try f.appendCallSret(l.block, name, new_args);
            return .{ .value = slot, .ty = ret_ty };
        }
        var pieces: [4]ir.function.RetPiece = undefined;
        const n = try structRetPieces(l, ret_ty, &pieces);
        const slot = try l.allocSlot(ret_ty);
        try f.appendCallStructRet(l.block, name, args, slot, pieces[0..n]);
        return .{ .value = slot, .ty = ret_ty };
    }
    const rt = try l.irTy(ret_ty);
    const v = if (variadic_fixed) |nf| try f.appendCallV(l.block, rt, name, args, nf) else try f.appendCall(l.block, rt, name, args);
    return .{ .value = v, .ty = ret_ty };
}

/// `directCallResult`'s INDIRECT-call counterpart (a call through a computed `target` pointer).
/// Mirrors it exactly, see there.
fn indirectCallResult(l: *L, callee: Value, ret_ty: ctype.CType, args: []const Value, variadic_fixed: ?u32) Error!TypedValue {
    const f = l.func;
    // A prototype spelled `void (*fp)(...)` carries an explicit `void_` return, emit a void
    // indirect call and yield a discarded placeholder, mirroring `directCallResult`.
    if (ret_ty == .void_) {
        if (variadic_fixed) |nf| try f.appendVoidCallIndirectV(l.block, callee, args, nf) else try f.appendVoidCallIndirect(l.block, callee, args);
        return .{ .value = try f.appendInst(l.block, l.i32t, .{ .iconst = 0 }), .ty = ctype.int_t };
    }
    if (ret_ty == .@"struct") {
        if (variadic_fixed != null) return error.Unsupported; // variadic struct return: not yet supported
        // `.sret` hidden-pointer return: mirrors `directCallResult`, see there.
        if (abi.classify(ret_ty, l.layout).ret == .sret) {
            const slot = try l.allocSlot(ret_ty);
            const new_args = try l.allocator.alloc(Value, args.len + 1);
            defer l.allocator.free(new_args);
            new_args[0] = slot;
            std.mem.copyForwards(Value, new_args[1..], args);
            try f.appendCallIndirectSret(l.block, callee, new_args);
            return .{ .value = slot, .ty = ret_ty };
        }
        var pieces: [4]ir.function.RetPiece = undefined;
        const n = try structRetPieces(l, ret_ty, &pieces);
        const slot = try l.allocSlot(ret_ty);
        try f.appendCallIndirectStructRet(l.block, callee, args, slot, pieces[0..n]);
        return .{ .value = slot, .ty = ret_ty };
    }
    const rt = try l.irTy(ret_ty);
    const v = if (variadic_fixed) |nf| try f.appendCallIndirectV(l.block, rt, callee, args, nf) else try f.appendCallIndirect(l.block, rt, callee, args);
    return .{ .value = v, .ty = ret_ty };
}

/// Coerce a typed value to a bool: `v != 0`, comparing against a zero of the operand's
/// own IR type (not necessarily `int`, a float operand compares against `fconst 0.0`).
fn truthy(l: *L, tv: TypedValue) Error!Value {
    const zero = try zeroInto(l, l.block, tv.ty);
    return l.func.appendInst(l.block, l.boolt, .{ .icmp = .{ .op = .ne, .lhs = tv.value, .rhs = zero } });
}

/// Map a parser binary op to the IR arithmetic op. Div/rem/shr pick their signed-vs-unsigned
/// variant from the (already `commonType`-converted) operand IR type's signedness.
fn irBinOp(op: parser.BinOp) ir.function.BinOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .rem => .rem,
        .bit_and => .bit_and,
        .bit_or => .bit_or,
        .bit_xor => .bit_xor,
        .shl => .shl,
        .shr => .shr,
    };
}

/// Decompose a struct/union-BY-VALUE argument into its ABI-classified scalar eightbytes,
/// loaded straight from `arg_addr` (the argument expression's OWN address, from `lowerAddr`;
/// `ty`'s a struct, so `lowerExpr` can't produce an rvalue for it), and appended to the flat
/// `out` arg list in eightbyte order, matching `bindStructParam`'s entry-param order at the
/// callee. Loading a full `i64` per eightbyte is correct even for a tail eightbyte smaller
/// than 8 bytes: the struct's alloca slot is storage-size-rounded up to a multiple of 8 (see
/// `CType.storageSize`), so the read stays in-bounds, and the upper bytes of a short tail
/// eightbyte are unspecified padding, which the real ABI also leaves unspecified, so this
/// still matches gcc byte-for-byte on every DEFINED byte.
///
/// The `.registers` (`.integer` AND `.sse`) and `.memory_ref` plans are implemented,
/// mirroring `bindStructParam` (see its doc for what's deferred to later work).
fn appendStructArg(l: *L, arg_addr: Value, ty: ctype.CType, out: *std.ArrayList(Value)) Error!void {
    const plan = abi.classify(ty, l.layout).arg;
    switch (plan) {
        .registers => |ebs| {
            for (ebs.slice()) |eb| {
                // An `.sse` eightbyte loads a FLOAT-typed value (see `floatEightbyteType`)
                // instead of the `.integer` arm's `i64`. The loaded value's IR type is what
                // routes it to the next FP argument register at the call site. The
                // classifier already put it at the right POSITION in `out`, in eightbyte
                // order, matching `bindStructParam`'s param order.
                const lt = if (eb.class == .sse) try floatEightbyteType(l, eb) else l.i64t;
                const p = try byteOffset(l, arg_addr, eb.offset);
                const v = try l.func.appendInst(l.block, lt, .{ .load = .{ .ptr = p, .@"volatile" = false } });
                try out.append(l.allocator, v);
            }
        },
        // An oversized (more than 16 byte) aggregate can't fit 1-2 integer registers, so
        // the ABI passes it BY REFERENCE. Copy the argument into a fresh caller-owned temp
        // and pass the temp's ADDRESS as a single pointer argument. The copy is required: C
        // still passes the struct BY VALUE even under a by-reference ABI, so the callee
        // (which may write through the pointer, see `bindStructParam`'s `.memory_ref` arm)
        // must never be able to mutate the caller's own object.
        .memory_ref => {
            const tmp = try l.allocSlot(ty);
            try copyStruct(l, ty.@"struct", arg_addr, tmp);
            try out.append(l.allocator, tmp);
        },
        // A `.memory_stack` struct has NO classifier eightbyte list (it is the i386
        // always-memory convention, and also x86_64's for a struct over 16 bytes). Pass it
        // by decomposing the WHOLE object into `ceil(size/word)` consecutive integer chunks,
        // loaded straight from `arg_addr`, appended in order. The chunk is one GENERAL register
        // wide (8 bytes on a 64-bit target, 4 bytes on i386). I386's backend is integer-only and
        // rejects a 64-bit load, so a 64-bit chunk would fail closed there. A word-wide chunk
        // pushes as one stack slot on i386 and lands in one integer register (or stack slot) on
        // x86_64. The alloca slot is storage-size-rounded up to a multiple of 8, so each read
        // stays in-bounds. The upper bytes of a short tail chunk are unspecified padding, which
        // the real ABI also leaves unspecified.
        .memory_stack => {
            const size = ty.sizeInBytes(l.layout) catch return error.Unsupported;
            const word: u64 = l.layout.ptr_bits / 8;
            const wt = if (word == 8) l.i64t else l.i32t;
            const n = (size + word - 1) / word;
            var i: u64 = 0;
            while (i < n) : (i += 1) {
                const p = try byteOffset(l, arg_addr, i * word);
                const v = try l.func.appendInst(l.block, wt, .{ .load = .{ .ptr = p, .@"volatile" = false } });
                try out.append(l.allocator, v);
            }
        },
    }
}

/// Lower `args` positionally, converting each to its slot's declared parameter type. This is
/// factored out of `lowerExpr`'s `.call` arm, which otherwise would need three near-identical
/// copies of this loop (a defined-func's params, a bodyless decl's params, an indirect
/// callee's `FuncType.params`). `params` is `anytype` because those three shapes differ: a
/// defined-func/decl's `parser.Param` carries its type in a `.ty` field, while a
/// `FuncType`'s params are already bare `ctype.CType`s. The `@TypeOf` branch below picks the
/// right one per instantiation (comptime-resolved, so each call site compiles to exactly the
/// same code an inlined loop would).
/// Arity is NOT checked here. Every call site already validates `args.len == params.len`
/// before calling this.
///
/// A struct/union-typed parameter takes the BY-VALUE decomposition path (`appendStructArg`,
/// via the argument's ADDRESS from `lowerAddr`) instead of the ordinary `lowerExpr`+
/// `convertTo` scalar path, since a struct has no scalar rvalue `lowerExpr` can produce.
/// Every other (scalar) parameter is ordinary.
fn lowerCallArgs(l: *L, args: []const *parser.Expr, params: anytype) Error!std.ArrayList(Value) {
    var out: std.ArrayList(Value) = .empty;
    errdefer out.deinit(l.allocator);
    for (args, params) |a, p| {
        const ty = if (@TypeOf(p) == ctype.CType) p else p.ty;
        if (ty == .@"struct") {
            const aa = try lowerAddr(l, a);
            try appendStructArg(l, aa.value, ty, &out);
        } else {
            const av = try lowerExpr(l, a);
            try out.append(l.allocator, try l.convertTo(av, ty));
        }
    }
    return out;
}

/// C's default argument promotions (C89 3.3.2.2 / C99 6.5.2.2p6), applied to a variadic
/// call's arguments PAST the callee's fixed parameters. There is no declared parameter type
/// to convert against for those, so the argument's OWN type widens instead:
/// `float` widens to `double` (matches `printf("%f", x)` expecting a `double` on the stack/
/// register even when `x` is `float`). An integer narrower than `int` (`char`/`short`,
/// either signedness) promotes to `int_t`, the same rule `IntType.promote` already applies
/// for ordinary integer promotion, just also applied to a `float` operand here. Any other
/// type (already `int`-rank-or-wider, `double`, pointer, struct) passes through unchanged.
fn defaultArgPromote(ty: ctype.CType) ctype.CType {
    if (ty.isFloat()) return .{ .float = .f64 };
    if (ty.asInt()) |it| {
        if (@intFromEnum(it.rank) < @intFromEnum(ctype.Rank.int)) return ctype.int_t;
    }
    return ty;
}

/// Lower a VARIADIC call's `args`: the first `num_fixed` convert to the callee's own
/// declared FIXED parameter type, exactly like `lowerCallArgs`. Every arg from `num_fixed`
/// on has no declared parameter type to convert against. `defaultArgPromote` applies
/// instead, using the ARGUMENT's own type (`typeOf`). Mirrors `lowerCallArgs`'s `anytype`
/// params trick for the same three caller shapes (`parser.Param`/bare `ctype.CType`). Unlike
/// `lowerCallArgs` this can't `for(args, params)` zip (there are more `args` than `params`),
/// so it indexes both explicitly instead.
///
/// A struct/union FIXED parameter takes the same `appendStructArg` by-value path
/// `lowerCallArgs` uses (see its doc). A struct passed as an ANONYMOUS (past-`num_fixed`)
/// variadic argument has no ABI convention implemented here yet, so it fails closed with
/// `error.Unsupported` rather than guessing one.
fn lowerVariadicCallArgs(l: *L, args: []const *parser.Expr, fixed_params: anytype, num_fixed: u32) Error!std.ArrayList(Value) {
    var out: std.ArrayList(Value) = .empty;
    errdefer out.deinit(l.allocator);
    for (args, 0..) |a, i| {
        if (i < num_fixed) {
            const p = fixed_params[i];
            const ty = if (@TypeOf(p) == ctype.CType) p else p.ty;
            if (ty == .@"struct") {
                const aa = try lowerAddr(l, a);
                try appendStructArg(l, aa.value, ty, &out);
            } else {
                const av = try lowerExpr(l, a);
                try out.append(l.allocator, try l.convertTo(av, ty));
            }
        } else {
            const arg_ty = try typeOf(l, a);
            if (arg_ty == .@"struct") return error.Unsupported; // struct as an anonymous variadic arg: not yet
            const av = try lowerExpr(l, a);
            // An array or function argument has already undergone the usual array-to-pointer /
            // function-to-pointer conversion, so `av` is already the decayed pointer value (a
            // string literal such as `"hi"` lowers to a `ptr`). Its promoted type IS that pointer,
            // so append the decayed value directly rather than converting against the undecayed
            // array/function type (which `convertTo` cannot target). Every other type goes through
            // the ordinary default argument promotion.
            if (arg_ty == .array or arg_ty == .func) {
                try out.append(l.allocator, av.value);
            } else {
                try out.append(l.allocator, try l.convertTo(av, defaultArgPromote(arg_ty)));
            }
        }
    }
    return out;
}

/// Find `name`'s parsed function (for its declared parameter/return types), or `null` if
/// this translation unit defines no such function.
fn findFunc(funcs: []const parser.Func, name: []const u8) ?*const parser.Func {
    for (funcs) |*fn_ast| if (std.mem.eql(u8, fn_ast.name, name)) return fn_ast;
    return null;
}

/// Find `name`'s bodyless declaration (a prototype / `extern`, for its declared
/// parameter/return types), or `null` if this translation unit carries no such declaration.
/// Consulted by `.call` only after `findFunc` misses: an external callee.
fn findFuncDecl(func_decls: []const parser.FuncDecl, name: []const u8) ?*const parser.FuncDecl {
    for (func_decls) |*decl| if (std.mem.eql(u8, decl.name, name)) return decl;
    return null;
}

/// Map a parser comparison op to the IR's comparison op (same names, different types).
fn irCmpOp(op: parser.CmpOp) ir.function.CmpOp {
    return switch (op) {
        .eq => .eq,
        .ne => .ne,
        .lt => .lt,
        .le => .le,
        .gt => .gt,
        .ge => .ge,
    };
}
