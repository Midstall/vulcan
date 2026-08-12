//! Compile-time constant evaluator for scalar global initializers. A static initializer
//! must be a compile-time constant. `eval` folds a constant integer expression to an
//! `i64`: int literals, unary `-`, `~`, `!`, binary arithmetic, bitwise, and shift,
//! comparison and equality, and `sizeof(type-name)` or `sizeof expr`, also reused for a
//! constant-expression array dimension. Anything that is not foldable here, such as a
//! `.name`, `.call`, `.deref`, `.addrof`, a string literal, or any other non-constant
//! node, fails closed with `error.Unsupported`. It is never silently treated as zero or
//! lowered as a runtime store. `evalScalarInit` wraps `eval` for a scalar target: the
//! folded `i64` is written little-endian, truncated to the target `CType`'s width, ready
//! to hand straight to a `lower.DataObject`. `evalInit` generalizes this to aggregates:
//! an `.array` or `.@"struct"` target recursively folds each brace-list element or
//! field, or, for a `char` array, a bare string-literal expression, into its own byte
//! range of the object's buffer, zero-filling anything a short list leaves out.
//!
//! Address constants: for a pointer-typed target, `&<global name>` and a bare string
//! literal are both compile-time or link-time constants. `evalScalarInit` and `evalInit`
//! fold them to pointer-sized zero bytes plus a `Const.relocs` entry
//! (`lower.DataReloc{ .off, .symbol }`) asking the linker or JIT to patch in the
//! target's runtime address later. This needs two things `eval`'s pure-`i64` fold never
//! did: the file's `GlobalTable`, to resolve `&g` to `g`'s link symbol, and the module's
//! data-object accumulator, to mint a fresh anonymous `.rodata` object for a string
//! literal, the same mechanism `lower.lowerStrLit` uses for a string in expression
//! context. Both are threaded in as optional pointers (`globals` and `data` below),
//! null wherever a caller has neither at hand, such as this file's own unit tests,
//! which never construct address constants. `&<local>`, meaning anything other than
//! `&<name-that-resolves-in-globals>`, or a name `globals` does not know about, is
//! `error.Unsupported`. Consteval never sees a function's local scope at all, so any
//! name absent from `globals` is indistinguishable from, and exactly as unsupported as,
//! a genuine local. This is the fail-closed behavior this evaluator requires.

const std = @import("std");
const parser = @import("parser.zig");
const ctype = @import("ctype.zig");
const layout = @import("layout.zig");
const lower = @import("lower.zig");

pub const Error = parser.Error;

/// A folded initializer: its little-endian bytes, sized to the target `CType`, or
/// pointer-sized zeros for an address constant, and any internal relocations
/// (`lower.DataReloc`, non-empty only for an address constant such as `&g` or a string
/// literal, see `tryAddressConst`). The caller owns `bytes` and `relocs`, allocated
/// from the `arena` passed to `evalScalarInit`.
pub const Const = struct { bytes: []u8, relocs: []lower.DataReloc };

/// Compute the `CType` of a `sizeof expr` operand, for `eval`'s `.sizeof_expr` arm below.
/// This is deliberately narrow: consteval never sees a function's local or global scope
/// (see this file's top doc comment), so only the operand shapes `eval` itself can
/// already fold without one, such as literals, `sizeof`, and arithmetic combinations of
/// those, resolve here. A `.name`, `.call`, or any other scope-dependent operand fails
/// closed with `error.Unsupported`, the same fail-closed rule `eval` already applies to
/// those shapes.
fn typeOfConst(expr: *const parser.Expr, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable) Error!ctype.CType {
    return switch (expr.*) {
        .int_lit => |v| v.ty,
        .float_lit => |v| v.ty,
        .sizeof_type, .sizeof_expr => ctype.ulong_t,
        // A global variable's `sizeof`: its declared type comes from `globals`, the same
        // table `lower.resolveName` consults. `globals` is null in the parser's
        // array-dim, enum, and case contexts, where a name is never a compile-time
        // constant, so `sizeof <name>` there fails closed exactly as before.
        // `slotvec0 = { sizeof slot0, ... }` in real code (gnulib's quotearg.c) folds
        // through here.
        .name => |n| blk: {
            const g = globals orelse return error.Unsupported;
            const gi = g.get(n) orelse return error.Unsupported;
            break :blk gi.ty;
        },
        .negate => |inner| blk: {
            const t = try typeOfConst(inner, lay, globals);
            break :blk if (t.isFloat()) t else t.promote();
        },
        .complement => |inner| blk: {
            const t = try typeOfConst(inner, lay, globals);
            break :blk t.promote();
        },
        .lognot, .compare => ctype.int_t,
        // A string literal's type is `char[len + 1]`, or `int[len + 1]` for a wide one,
        // including the terminating NUL, so `sizeof "abc"` folds to 4. Real code writes
        // `sizeof "..." - 1` for a literal's length. `&ctype.char_t` and `&ctype.int_t`
        // are addresses of module constants, so no allocation is needed for the element type.
        .str_lit => |s| .{ .array = .{ .elem = &ctype.char_t, .len = s.len + 1 } },
        .wstr_lit => |s| .{ .array = .{ .elem = &ctype.int_t, .len = s.len + 1 } },
        .binary => |b| blk: {
            const lt = try typeOfConst(b.lhs, lay, globals);
            const rt = try typeOfConst(b.rhs, lay, globals);
            break :blk if (b.op == .shl or b.op == .shr) lt.promote() else ctype.CType.commonType(lt, rt, lay);
        },
        else => error.Unsupported,
    };
}

/// Fold a compile-time constant integer expression to an `i64`, using plain, wrapping
/// `i64` arithmetic. These initializers are small integers, so width and sign niceties
/// beyond that are out of scope. `lay` is threaded through for future, non-integer
/// constant forms. It is unused by these node shapes.
pub fn eval(arena: std.mem.Allocator, expr: *const parser.Expr, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable) Error!i64 {
    return switch (expr.*) {
        .int_lit => |v| v.value,
        .negate => |x| 0 -% try eval(arena, x, lay, globals),
        .complement => |x| ~try eval(arena, x, lay, globals),
        .lognot => |x| if (try eval(arena, x, lay, globals) == 0) 1 else 0,
        .binary => |b| blk: {
            const l = try eval(arena, b.lhs, lay, globals);
            const r = try eval(arena, b.rhs, lay, globals);
            // Guard the cases that would panic on the wrapping or checked host
            // arithmetic below, as an early error return, not a switch-prong value,
            // which would have to peer-resolve against the plain `i64` prongs below,
            // before folding the actual op:
            //  - div or rem by zero: `@divTrunc` or `@rem` by 0 traps.
            //  - `@divTrunc(minInt(i64), -1)` two's-complement overflow. `INT_MIN / -1`
            //    is C undefined behavior anyway, so failing closed is correct. `@rem`
            //    on the same operands does not overflow, so `.rem` only needs the
            //    `r == 0` guard.
            //  - a shift count that is negative or at least the 64-bit width: `<<` or
            //    `>>` by such a count traps.
            switch (b.op) {
                .div => if (r == 0 or (l == std.math.minInt(i64) and r == -1)) return error.Unsupported,
                .rem => if (r == 0) return error.Unsupported,
                .shl, .shr => if (r < 0 or r >= 64) return error.Unsupported,
                else => {},
            }
            break :blk switch (b.op) {
                .add => l +% r,
                .sub => l -% r,
                .mul => l *% r,
                .div => @divTrunc(l, r),
                .rem => @rem(l, r),
                .bit_and => l & r,
                .bit_or => l | r,
                .bit_xor => l ^ r,
                .shl => l << @as(u6, @intCast(r)),
                .shr => l >> @as(u6, @intCast(r)),
            };
        },
        .compare => |c| blk: {
            const l = try eval(arena, c.lhs, lay, globals);
            const r = try eval(arena, c.rhs, lay, globals);
            const result = switch (c.op) {
                .eq => l == r,
                .ne => l != r,
                .lt => l < r,
                .le => l <= r,
                .gt => l > r,
                .ge => l >= r,
            };
            break :blk if (result) 1 else 0;
        },
        // `sizeof(type-name)`: a pure type-size lookup, no scope needed at all.
        .sizeof_type => |ct| @intCast(try ct.sizeInBytes(lay)),
        // `sizeof expr`: the operand is unevaluated, so only its type matters, resolved
        // by the narrow, scope-free `typeOfConst` above. A scope-dependent operand
        // fails closed the same way the rest of this function does.
        .sizeof_expr => |inner| @intCast(try (try typeOfConst(inner, lay, globals)).sizeInBytes(lay)),
        // A cast in a constant expression (C11 6.6p6). Real headers fold `(int) sizeof (x)`
        // into an array dimension (glibc's `fd_set` uses `(8 * (int) sizeof (__fd_mask))`).
        // Evaluate the operand, then narrow to an INTEGER target's width and signedness. A
        // pointer target keeps the integer value unchanged (a constant like `(void *) 0`). A
        // float or other target is not a constant this folder handles.
        .cast => |c| blk: {
            const v = try eval(arena, c.operand, lay, globals);
            switch (c.target) {
                .int => |it| {
                    const bits = it.bits(lay);
                    if (bits >= 64) break :blk v;
                    const mask = (@as(u64, 1) << @intCast(bits)) - 1;
                    const low = @as(u64, @bitCast(v)) & mask;
                    // Sign-extend from `bits` for a signed target, else keep the
                    // zero-extended low bits.
                    if (it.signed and (low & (@as(u64, 1) << @intCast(bits - 1))) != 0) {
                        break :blk @bitCast(low | ~mask);
                    }
                    break :blk @bitCast(low);
                },
                .ptr => break :blk v,
                else => return error.Unsupported,
            }
        },
        // `&&`/`||` short-circuit to a 0/1 result. Real headers fold them inside static-assert
        // conditions.
        .logand => |b| blk: {
            if (try eval(arena, b.lhs, lay, globals) == 0) break :blk 0;
            break :blk if (try eval(arena, b.rhs, lay, globals) != 0) 1 else 0;
        },
        .logor => |b| blk: {
            if (try eval(arena, b.lhs, lay, globals) != 0) break :blk 1;
            break :blk if (try eval(arena, b.rhs, lay, globals) != 0) 1 else 0;
        },
        // `cond ? a : b`. Only the taken branch is evaluated. gnulib's `verify`/static-assert
        // macros fold `(COND ? 1 : -1)` into an array dimension or bit-field width.
        .ternary => |t| if (try eval(arena, t.cond, lay, globals) != 0)
            try eval(arena, t.then, lay, globals)
        else
            try eval(arena, t.els, lay, globals),
        // `.name`/`.call`/`.deref`/`.addrof`/`.index`/`.member`/`.assign`, or a string literal:
        // none of these are a compile-time constant this folder handles.
        else => error.Unsupported,
    };
}

/// A single-element reloc `Const`: pointer-sized (`lay.ptr_bits / 8`) zero bytes, with
/// one `DataReloc` at offset 0 naming `symbol`. This is the shape every address-constant
/// leaf, such as `&g` or a string literal, folds to, before any enclosing aggregate
/// offsets it. See `evalArrayInit` and `evalStructInit`.
fn relocConst(arena: std.mem.Allocator, lay: layout.TargetLayout, symbol: []const u8) Error!Const {
    const size: usize = @intCast(lay.ptr_bits / 8);
    const bytes = try arena.alloc(u8, size);
    @memset(bytes, 0);
    const relocs = try arena.dupe(lower.DataReloc, &.{.{ .off = 0, .symbol = symbol }});
    return .{ .bytes = bytes, .relocs = relocs };
}

/// Mint a fresh anonymous `.rodata` string `DataObject` for `s`, bytes plus a NUL
/// terminator, with a unique `.str.<counter>` symbol keyed off `data.items.len` at mint
/// time, and return its symbol. This is the same mechanism `lower.lowerStrLit` uses for
/// a string literal in expression context, reused here for one appearing as a pointer's
/// static initializer. The returned symbol is the exact string handed to the new
/// `DataObject.name`: one allocation, one owner, the appended `DataObject`, freed once
/// by `Module.deinit`. Nothing else frees it, including the `DataReloc` this symbol
/// ends up in, which never owns its `.symbol`.
fn mintStringObject(arena: std.mem.Allocator, data: *std.ArrayList(lower.DataObject), s: []const u8) Error![]const u8 {
    const nul_bytes = try arena.alloc(u8, s.len + 1);
    errdefer arena.free(nul_bytes);
    @memcpy(nul_bytes[0..s.len], s);
    nul_bytes[s.len] = 0;
    const sym = try std.fmt.allocPrint(arena, ".str.{d}", .{data.items.len});
    errdefer arena.free(sym);
    try data.append(arena, .{ .name = sym, .kind = .rodata, .bytes = nul_bytes, .size = nul_bytes.len });
    return sym;
}

/// If `expr` is an address constant, `&<global name>` or a bare string literal, fold it
/// to a reloc `Const` (see `relocConst`). Returns `null` if `expr` is not this shape at
/// all, and the caller falls back to the plain `eval` int fold, for example `int *p = 0;`'s
/// null-pointer constant. `&<anything but a name>`, such as `&arr[0]` or `&s.field`, or
/// a name `globals` cannot resolve, such as a local or a truly undefined name, is a
/// definite address-constant shape that just is not a valid one here. It reports
/// `error.Unsupported`, not `null`. Falling through to `eval` would only re-derive the
/// same `error.Unsupported`, since `eval` never handles `.addrof` or `.str_lit` either.
fn tryAddressConst(arena: std.mem.Allocator, expr: *const parser.Expr, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!?Const {
    return switch (expr.*) {
        .addrof => |inner| blk: {
            const name = switch (inner.*) {
                .name => |n| n,
                else => return error.Unsupported, // &local, &arr[i], &s.field, and so on: not a link-time constant
            };
            const g = globals orelse return error.Unsupported;
            const info = g.get(name) orelse return error.Unsupported; // Undefined, or a local. Consteval sees no local scope.
            break :blk try relocConst(arena, lay, info.symbol);
        },
        .str_lit => |s| blk: {
            const d = data orelse return error.Unsupported;
            const sym = try mintStringObject(arena, d, s);
            break :blk try relocConst(arena, lay, sym);
        },
        // A bare global array or function name in a pointer-constant context decays to
        // the address of the object, a link-time reloc, with no `&` needed:
        // `char *val = slot0;` where `slot0` is a `char[N]` (gnulib's quotearg.c). A name
        // of any other type would be reading the object's runtime value, which is not a
        // constant, so it falls through to `eval`, which reports it.
        .name => |n| blk: {
            const g = globals orelse break :blk null;
            const info = g.get(n) orelse break :blk null;
            break :blk switch (info.ty) {
                .array, .func => try relocConst(arena, lay, info.symbol),
                else => null,
            };
        },
        else => null,
    };
}

/// Fold a scalar global's initializer to its backing bytes, for `ty`'s width. Only a
/// bare `Initializer.expr`, or a single-element `.list` wrapping one such as
/// `int g = {5};`, is a scalar initializer this function accepts. A multi-element
/// brace-list on a scalar target is invalid C and reports `error.Unsupported`. `arena`
/// need not be an actual arena, since the parser's AST nodes are walked, never mutated.
/// It is whatever allocator should own the returned `Const.bytes`, which the caller
/// (`lower.compile`) hands to a `DataObject` that outlives this call. `globals` and
/// `data` resolve or mint address constants for a pointer-typed `ty` (see
/// `tryAddressConst`). Pass `null` for both when `ty` can never be a pointer target
/// here, such as in this file's own unit tests, or the caller has neither at hand.
pub fn evalScalarInit(arena: std.mem.Allocator, init: *const parser.Initializer, ty: ctype.CType, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!Const {
    const expr = switch (init.value) {
        .expr => |e| e,
        .list => |list| if (list.len == 1 and list[0].designators.len == 0 and list[0].value == .expr) list[0].value.expr else return error.Unsupported,
    };
    if (ty == .ptr) {
        if (try tryAddressConst(arena, expr, lay, globals, data)) |c| return c;
    }
    const v = try eval(arena, expr, lay, globals);
    const size = try ty.sizeInBytes(lay);
    var full: [8]u8 = undefined;
    std.mem.writeInt(u64, &full, @bitCast(v), .little);
    const bytes = try arena.dupe(u8, full[0..size]);
    return .{ .bytes = bytes, .relocs = &.{} };
}

/// Whether `ty` is a `char`-rank integer, of any signedness. This is the only element
/// type a bare string-literal initializer (`char s[] = "abc"`, not a brace list) can
/// fill, per `evalArrayInit` below.
fn isCharType(ty: ctype.CType) bool {
    return switch (ty) {
        .int => |i| i.rank == .char,
        else => false,
    };
}

/// Fold an aggregate, array or struct or union, global's initializer to its backing
/// bytes. `evalInit` below is the general entry point every caller, such as
/// `lower.compile` and `lower.lowerStmt`'s `static` local arm, should use. It dispatches
/// a scalar `ty` straight to `evalScalarInit`, and an array or struct `ty` to
/// `evalArrayInit` or `evalStructInit` here. Every one of these returns bytes sized to
/// `ty.storageSize(lay)`, not `sizeInBytes`. They agree for a scalar or a struct or
/// array with no padding, but differ for a struct whose C size is not a multiple of 8,
/// since the backend's blob storage always rounds up (see `ctype.CType.storageSize`'s
/// own doc). Matching that storage width is what keeps a `DataObject`'s buffer big
/// enough for how the backend actually reads and writes it, in whole i64 words, and
/// matches the exact same convention already used for a global aggregate with no
/// initializer. `compile` and `lowerStmt` size their `.bss` objects with `storageSize`,
/// not `sizeInBytes`.
pub fn evalInit(arena: std.mem.Allocator, init: *const parser.Initializer, ty: ctype.CType, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!Const {
    return switch (ty) {
        .array => |a| evalArrayInit(arena, init, a.elem.*, a.len, lay, globals, data),
        .@"struct" => |s| evalStructInit(arena, init, s, ty, lay, globals, data),
        else => evalScalarInit(arena, init, ty, lay, globals, data),
    };
}

/// Write a list element's value at its designator chain's leaf slot. `ty` and `off` are
/// the current sub-object's type and cumulative byte offset into the top-level `bytes`
/// buffer. The caller passes the position it already resolved from `chain[0]`. `chain`
/// is the remaining designator steps to walk, never empty, since the caller always
/// resolves at least one step before calling this. Each remaining step just moves `off`
/// deeper: a `.field` step adds `field.offset` and narrows `ty` to `field.ty`, and a
/// `.index` step adds `idx * elem.storageSize(lay)` and narrows `ty` to `elem`. No byte
/// is written and nothing is zeroed until the chain is fully consumed. Only then, in
/// `writeLeaf`, does this fold `value` and write it, at the computed offset, into the
/// same `bytes` buffer every other item in this list also writes into.
///
/// Walking the whole chain here, and writing the scalar leaf directly with no
/// re-entrant `evalInit` over the sub-object it lives in, avoids zero-filling a
/// sub-object more than once. A naive single-step peel that hands the rest of a
/// multi-step chain, for example `.b.x`'s `.x`, to a fresh recursive `evalInit` call
/// scoped to the immediate sub-object, `.b`'s own `struct P`, would break: `evalInit`
/// unconditionally zero-fills whatever object it is folding before writing into it, so
/// a second top-level item sharing the same prefix, `.b.y = 6` after `.b.x = 5`, would
/// re-zero and overwrite the sub-object `.b.x = 5` had just written into, silently
/// discarding it. Walking the whole chain here makes two top-level items that share a
/// nested prefix both survive. Each only ever touches its own leaf's bytes, never its
/// siblings'. A leaf whose own value is itself a brace-list, `.b = {5,6}`, a whole
/// sub-object assigned as a unit, still goes through `evalInit` and `writeLeaf` at that
/// point. This is correct, since the whole sub-object legitimately being replaced is
/// exactly when re-zeroing it is intended.
fn writeNested(arena: std.mem.Allocator, bytes: []u8, relocs: *std.ArrayList(lower.DataReloc), ty: ctype.CType, off: usize, chain: []const parser.Designator, value: *const parser.Initializer, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!void {
    std.debug.assert(chain.len >= 1); // every call site resolves at least one step first
    switch (chain[0]) {
        .field => |name| {
            const def = switch (ty) {
                .@"struct" => |s| s,
                else => return error.Unsupported, // `.field` into a non-struct: invalid C
            };
            const idx = findFieldIndex(def, name) orelse return error.Unsupported;
            const field = def.fields[idx];
            const foff = off + @as(usize, @intCast(field.offset));
            if (chain.len == 1) {
                // A bitfield field must pack its value into the shared storage unit
                // starting from the least significant bit, exactly like the runtime
                // path's `lower.storeBitfield`. A raw `writeLeaf` `@memcpy` would
                // instead overwrite the whole unit un-shifted, so two bitfields in one
                // unit would clobber each other, and the folded global would disagree
                // with the identical initializer lowered as a local.
                if (field.bit_width) |w| return writeBitfieldLeaf(arena, bytes, field.ty, foff, field.bit_offset, w, value, lay);
                return writeLeaf(arena, bytes, relocs, field.ty, foff, try field.ty.sizeInBytes(lay), value, lay, globals, data);
            }
            return writeNested(arena, bytes, relocs, field.ty, foff, chain[1..], value, lay, globals, data);
        },
        .index => |idx| {
            const a = switch (ty) {
                .array => |arr| arr,
                else => return error.Unsupported, // `[i]` into a non-array: invalid C
            };
            if (idx >= a.len) return error.Unsupported;
            const stride: usize = @intCast(try a.elem.storageSize(lay));
            const eoff = off + @as(usize, @intCast(idx)) * stride;
            if (chain.len == 1) {
                return writeLeaf(arena, bytes, relocs, a.elem.*, eoff, stride, value, lay, globals, data);
            }
            return writeNested(arena, bytes, relocs, a.elem.*, eoff, chain[1..], value, lay, globals, data);
        },
    }
}

/// Fold `value` against `ty`, either a plain scalar through `evalScalarInit` or a
/// `.list`-driven whole sub-object through `evalArrayInit` or `evalStructInit`, since
/// `evalInit` dispatches either way, and copy its bytes into `bytes[off..]`, capped to
/// `max_copy` bytes. That cap is a struct field's exact C size, `sizeInBytes`, never the
/// rounded `storageSize`, so an aggregate-typed field's own trailing padding never
/// spills onto its next sibling's bytes, or an array element's full stride, since
/// elements are spaced at `storageSize`. `writeNested`'s two leaf arms are this
/// function's only callers. See its doc for why this is the only place a leaf ever gets
/// zero-filled.
fn writeLeaf(arena: std.mem.Allocator, bytes: []u8, relocs: *std.ArrayList(lower.DataReloc), ty: ctype.CType, off: usize, max_copy: usize, value: *const parser.Initializer, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!void {
    const c = try evalInit(arena, value, ty, lay, globals, data);
    defer arena.free(c.bytes);
    defer arena.free(c.relocs);
    const copy_len = @min(c.bytes.len, max_copy);
    removeOverlappingRelocs(relocs, off, off + copy_len);
    @memcpy(bytes[off .. off + copy_len], c.bytes[0..copy_len]);
    for (c.relocs) |r| try relocs.append(arena, .{ .off = r.off + off, .symbol = r.symbol });
}

/// Pack a bitfield's scalar `value` into the storage unit at `bytes[off..]`, starting
/// from the least significant bit. This is the compile-time twin of
/// `lower.storeBitfield`, keeping the fold and the runtime path in agreement. It
/// read-modify-writes the unit rather than overwriting it, so a second bitfield sharing
/// the unit keeps the first bitfield's bits. `unit_ty` is the field's declared integer
/// type, the whole storage unit, `bit_off` is the field's first bit within that unit,
/// and `w` is its width. The unit is a little-endian integer, matching every target and
/// `evalScalarInit`'s own `writeInt(... .little)`. A bitfield's value must be a bare
/// scalar. A brace-list against a bitfield is invalid C and fails closed.
fn writeBitfieldLeaf(arena: std.mem.Allocator, bytes: []u8, unit_ty: ctype.CType, off: usize, bit_off: u32, w: u32, value: *const parser.Initializer, lay: layout.TargetLayout) Error!void {
    const expr = switch (value.value) {
        .expr => |e| e,
        .list => return error.Unsupported, // a brace-list against a bitfield: invalid C
    };
    const v: u64 = @bitCast(try eval(arena, expr, lay, null));
    const unit_size: usize = @intCast(try unit_ty.sizeInBytes(lay));
    const total_bits: u32 = @intCast(unit_size * 8);
    const type_mask: u64 = if (total_bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(total_bits)) - 1;
    const field_mask: u64 = if (w >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(w)) - 1;
    const shifted_mask: u64 = (field_mask << @intCast(bit_off)) & type_mask;
    const clear_mask: u64 = (~shifted_mask) & type_mask;
    // Read the current unit little-endian (zero-padded to 8 bytes), clear the field's bits, OR
    // the new value's low `w` bits shifted up to `bit_off`, and write the unit back.
    var buf: [8]u8 = [_]u8{0} ** 8;
    @memcpy(buf[0..unit_size], bytes[off .. off + unit_size]);
    const cur = std.mem.readInt(u64, &buf, .little);
    const new_unit = (cur & clear_mask) | (((v & field_mask) << @intCast(bit_off)) & type_mask);
    std.mem.writeInt(u64, &buf, new_unit, .little);
    @memcpy(bytes[off .. off + unit_size], buf[0..unit_size]);
}

/// Drop every reloc in `[lo, hi)` from `relocs`. A later designated write to a byte
/// range a prior element already wrote a relocated pointer into must not leave that
/// stale reloc behind, so the later write wins. This is a no-op when nothing overlaps,
/// as it always is for a positional-only list, since positions there never repeat.
/// `orderedRemove` preserves the relative order of what is kept, so this never
/// reorders `relocs` in the common case.
fn removeOverlappingRelocs(relocs: *std.ArrayList(lower.DataReloc), lo: usize, hi: usize) void {
    var i: usize = 0;
    while (i < relocs.items.len) {
        if (relocs.items[i].off >= lo and relocs.items[i].off < hi) {
            _ = relocs.orderedRemove(i);
        } else {
            i += 1;
        }
    }
}

/// Find `name`'s index in `def.fields`, for a `.field` designator. Returns `null` if
/// `def` has no field by that name. The caller fails closed with `error.Unsupported`,
/// the same as any other designator this frontend cannot resolve.
fn findFieldIndex(def: *const ctype.StructDef, name: []const u8) ?usize {
    for (def.fields, 0..) |f, i| {
        if (std.mem.eql(u8, f.name, name)) return i;
    }
    return null;
}

/// Fold an array's initializer: a brace-`.list`, recursively `evalInit` for each
/// element at byte offset `pos * elem.storageSize(lay)`, not `sizeInBytes`, since a
/// struct element is stored at its rounded storage stride, the same rule pointer-walk
/// and array-of-struct addressing already rely on elsewhere in this frontend, or, only
/// when `elem` is `char`, a bare `.expr` that is a string literal (`char s[] = "abc"` or
/// `char s[4] = "abc"`): its decoded bytes plus a NUL, zero-filled to `len`. `pos`
/// starts at 0 and tracks the current position. A `[index]` designator on an element
/// sets it. A plain element uses it, then advances by one. For an all-positional list
/// with no designators anywhere, this is exactly `pos == i`. A position, whether
/// designated or reached positionally, at or past `len` is `error.Unsupported`, and
/// never silently truncates or wraps. Any position a later element does not reach
/// stays zero, since the buffer starts zeroed, and a position written more than once
/// keeps only the last write, through `removeOverlappingRelocs` plus a plain `@memcpy`
/// over the same bytes, which is naturally last-wins. Every recursive `evalInit` call's
/// own returned bytes and relocs are freed right after being copied into this array's
/// buffer. `arena` is not necessarily an actual arena (see `evalScalarInit`'s doc), so
/// an intermediate per-element allocation must not outlive this call, or it leaks
/// under a real, non-arena allocator.
fn evalArrayInit(arena: std.mem.Allocator, init: *const parser.Initializer, elem: ctype.CType, len: u64, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!Const {
    const stride: usize = @intCast(try elem.storageSize(lay));
    const total: usize = @intCast(len * stride);
    const bytes = try arena.alloc(u8, total);
    errdefer arena.free(bytes);
    @memset(bytes, 0);

    if (init.value == .expr and init.value.expr.* == .str_lit and isCharType(elem)) {
        const s = init.value.expr.str_lit;
        if (s.len + 1 > len) return error.Unsupported; // too long even counting the NUL
        @memcpy(bytes[0..s.len], s);
        return .{ .bytes = bytes, .relocs = &.{} };
    }
    if (init.value != .list) return error.Unsupported; // A non-list, non-string shape cannot
    // initialize an array here.
    const list = init.value.list;

    // `writeNested` needs the whole array `CType`, since its `.index` arm reads `.elem`
    // and `.len` off it, which this function only receives split apart as `elem` and
    // `len`, so re-wrap it. `&elem` stays valid for this whole call: `elem` is this
    // function's own stack parameter, and `writeNested`'s use of it is entirely
    // synchronous, never stored past this call tree.
    const arr_ty: ctype.CType = .{ .array = .{ .elem = &elem, .len = len } };

    var relocs: std.ArrayList(lower.DataReloc) = .empty;
    errdefer relocs.deinit(arena);
    var pos: u64 = 0;
    for (list) |*item| {
        if (item.designators.len > 0) {
            pos = switch (item.designators[0]) {
                .index => |idx| idx,
                .field => return error.Unsupported, // `.field` on an array target: invalid C
            };
        }
        if (pos >= len) return error.Unsupported;
        var synth: [1]parser.Designator = undefined;
        const chain: []const parser.Designator = if (item.designators.len > 0) item.designators else blk: {
            synth[0] = .{ .index = pos };
            break :blk synth[0..1];
        };
        const value_init: parser.Initializer = .{ .value = item.value };
        try writeNested(arena, bytes, &relocs, arr_ty, 0, chain, &value_init, lay, globals, data);
        pos += 1;
    }
    return .{ .bytes = bytes, .relocs = try relocs.toOwnedSlice(arena) };
}

/// Fold a struct or union's brace-`.list` initializer: element `pos` maps to
/// `def.fields[pos]`. `pos` tracks the current field position exactly like
/// `evalArrayInit`'s: a `.field` designator sets it via `findFieldIndex`, and a plain
/// element uses it, then advances it. For an all-positional list this is `pos == i`.
/// It recursively calls `evalInit`, and copies at `field.offset` for exactly
/// `field.ty.sizeInBytes(lay)` bytes, the C layout stride `parser.parseStructOrUnion`
/// already computed fields at, not `storageSize`, which is only the right stride for an
/// array's elements (see `evalArrayInit`); a struct field sits immediately next to its
/// neighbor's exact size, with no rounding. A position past the last field is
/// `error.Unsupported`. A field no element reaches stays zero. A field written more
/// than once keeps only the last write (see `evalArrayInit`'s doc).
fn evalStructInit(arena: std.mem.Allocator, init: *const parser.Initializer, def: *const ctype.StructDef, ty: ctype.CType, lay: layout.TargetLayout, globals: ?*const lower.GlobalTable, data: ?*std.ArrayList(lower.DataObject)) Error!Const {
    if (init.value != .list) return error.Unsupported;
    const list = init.value.list;

    const total: usize = @intCast(try ty.storageSize(lay));
    const bytes = try arena.alloc(u8, total);
    errdefer arena.free(bytes);
    @memset(bytes, 0);

    var relocs: std.ArrayList(lower.DataReloc) = .empty;
    errdefer relocs.deinit(arena);
    var pos: usize = 0;
    for (list) |*item| {
        if (item.designators.len > 0) {
            pos = switch (item.designators[0]) {
                .field => |name| findFieldIndex(def, name) orelse return error.Unsupported,
                .index => return error.Unsupported, // `[i]` on a struct target: invalid C
            };
        }
        if (pos >= def.fields.len) return error.Unsupported;
        var synth: [1]parser.Designator = undefined;
        const chain: []const parser.Designator = if (item.designators.len > 0) item.designators else blk: {
            synth[0] = .{ .field = def.fields[pos].name };
            break :blk synth[0..1];
        };
        const value_init: parser.Initializer = .{ .value = item.value };
        try writeNested(arena, bytes, &relocs, ty, 0, chain, &value_init, lay, globals, data);
        pos += 1;
    }
    return .{ .bytes = bytes, .relocs = try relocs.toOwnedSlice(arena) };
}

test "eval folds int literals and arithmetic" {
    const l = layout.host();
    var five: parser.Expr = .{ .int_lit = .{ .value = 5, .ty = ctype.int_t } };
    try std.testing.expectEqual(@as(i64, 5), try eval(std.testing.allocator, &five, l, null));

    var two: parser.Expr = .{ .int_lit = .{ .value = 2, .ty = ctype.int_t } };
    var three: parser.Expr = .{ .int_lit = .{ .value = 3, .ty = ctype.int_t } };
    var four: parser.Expr = .{ .int_lit = .{ .value = 4, .ty = ctype.int_t } };
    var mul: parser.Expr = .{ .binary = .{ .op = .mul, .lhs = &three, .rhs = &four } };
    var add: parser.Expr = .{ .binary = .{ .op = .add, .lhs = &two, .rhs = &mul } };
    try std.testing.expectEqual(@as(i64, 14), try eval(std.testing.allocator, &add, l, null));

    var neg: parser.Expr = .{ .negate = &five };
    try std.testing.expectEqual(@as(i64, -5), try eval(std.testing.allocator, &neg, l, null));
}

test "eval rejects a non-constant name" {
    const l = layout.host();
    var name: parser.Expr = .{ .name = "x" };
    try std.testing.expectError(error.Unsupported, eval(std.testing.allocator, &name, l, null));
}

test "evalScalarInit writes width-correct little-endian bytes" {
    const allocator = std.testing.allocator;
    const l = layout.host();
    var five: parser.Expr = .{ .int_lit = .{ .value = 5, .ty = ctype.int_t } };
    const init: parser.Initializer = .{ .value = .{ .expr = &five } };

    const c = try evalScalarInit(allocator, &init, ctype.int_t, l, null, null);
    defer allocator.free(c.bytes);
    try std.testing.expectEqualSlices(u8, &.{ 5, 0, 0, 0 }, c.bytes);

    const c2 = try evalScalarInit(allocator, &init, ctype.char_t, l, null, null);
    defer allocator.free(c2.bytes);
    try std.testing.expectEqualSlices(u8, &.{5}, c2.bytes);
}

test "evalInit fills a partial array init with zeros" {
    const allocator = std.testing.allocator;
    const l = layout.host();
    var seven: parser.Expr = .{ .int_lit = .{ .value = 7, .ty = ctype.int_t } };
    var elems = [_]parser.Initializer{.{ .value = .{ .expr = &seven } }};
    const init: parser.Initializer = .{ .value = .{ .list = &elems } };
    const arr_ty: ctype.CType = .{ .array = .{ .elem = &ctype.int_t, .len = 4 } };

    const c = try evalInit(allocator, &init, arr_ty, l, null, null);
    defer allocator.free(c.bytes);
    defer allocator.free(c.relocs);
    try std.testing.expectEqualSlices(u8, &.{ 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, c.bytes);
    try std.testing.expectEqual(@as(usize, 0), c.relocs.len);
}

test "evalInit writes struct fields at their offsets" {
    const allocator = std.testing.allocator;
    const l = layout.host();
    const fields = [_]ctype.Field{
        .{ .name = "x", .ty = ctype.int_t, .offset = 0 },
        .{ .name = "y", .ty = ctype.int_t, .offset = 4 },
    };
    const def: ctype.StructDef = .{ .name = "P", .is_union = false, .fields = &fields, .size = 8, .alignment = 4 };
    const struct_ty: ctype.CType = .{ .@"struct" = &def };

    var three: parser.Expr = .{ .int_lit = .{ .value = 3, .ty = ctype.int_t } };
    var four: parser.Expr = .{ .int_lit = .{ .value = 4, .ty = ctype.int_t } };
    var elems = [_]parser.Initializer{ .{ .value = .{ .expr = &three } }, .{ .value = .{ .expr = &four } } };
    const init: parser.Initializer = .{ .value = .{ .list = &elems } };

    const c = try evalInit(allocator, &init, struct_ty, l, null, null);
    defer allocator.free(c.bytes);
    defer allocator.free(c.relocs);
    try std.testing.expectEqualSlices(u8, &.{ 3, 0, 0, 0, 4, 0, 0, 0 }, c.bytes);
}

test "evalInit copies a string literal into a char array, zero-filling the rest" {
    const allocator = std.testing.allocator;
    const l = layout.host();
    var str: parser.Expr = .{ .str_lit = "hi" };
    const init: parser.Initializer = .{ .value = .{ .expr = &str } };
    const arr_ty: ctype.CType = .{ .array = .{ .elem = &ctype.char_t, .len = 4 } };

    const c = try evalInit(allocator, &init, arr_ty, l, null, null);
    defer allocator.free(c.bytes);
    defer allocator.free(c.relocs);
    try std.testing.expectEqualSlices(u8, &.{ 'h', 'i', 0, 0 }, c.bytes);
}

test "evalInit rejects too many array elements" {
    const allocator = std.testing.allocator;
    const l = layout.host();
    var one: parser.Expr = .{ .int_lit = .{ .value = 1, .ty = ctype.int_t } };
    var two: parser.Expr = .{ .int_lit = .{ .value = 2, .ty = ctype.int_t } };
    var elems = [_]parser.Initializer{ .{ .value = .{ .expr = &one } }, .{ .value = .{ .expr = &two } } };
    const init: parser.Initializer = .{ .value = .{ .list = &elems } };
    const arr_ty: ctype.CType = .{ .array = .{ .elem = &ctype.int_t, .len = 1 } };
    try std.testing.expectError(error.Unsupported, evalInit(allocator, &init, arr_ty, l, null, null));
}
