//! Inter-procedural function inlining. Replaces a `call` to a small callee with a
//! clone of its body. Callee parameters become the call's arguments, its
//! instructions are spliced into the caller at the call site, and its returned
//! value replaces the call's result.
//!
//! Both paths handle leaf callees (no calls of their own) with scalar-typed values.
//! `inlineCall` takes a one-block `ret` callee and splices it in with no new blocks,
//! which is the cleanest result. `inlineCallMulti` takes a callee that has its own
//! control flow (`if`, loops, `store`, multiple `ret`s) and clones it block by block.
//! It splits the caller block at the call into a continuation holding the post-call
//! code, clones the callee's reachable blocks with values, blocks, and types remapped
//! in reverse-postorder, and rewrites each `ret` into a jump to the continuation whose
//! parameter carries the inlined return value. Aggregate types (struct_new, extract,
//! vectors) are still excluded.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;
const Type = ir.types.Type;
const Opcode = ir.function.Opcode;

pub const Error = std.mem.Allocator.Error;

/// Resolves a callee name to its function body (or null if unavailable).
pub const Lookup = struct {
    context: *anyopaque,
    func: *const fn (context: *anyopaque, name: []const u8) ?*const Function,

    pub fn get(self: Lookup, name: []const u8) ?*const Function {
        return self.func(self.context, name);
    }
};

/// Decides, per call site (by the caller block it sits in), whether to inline it.
/// Used by profile-guided inlining to inline only hot calls.
pub const Filter = struct {
    context: *anyopaque,
    func: *const fn (context: *anyopaque, block_index: usize) bool,

    fn allow(self: Filter, block_index: usize) bool {
        return self.func(self.context, block_index);
    }
};

/// Inline every inlinable call in `caller`, repeatedly, up to a cap. Returns
/// whether anything was inlined.
pub fn run(allocator: std.mem.Allocator, caller: *Function, lookup: Lookup) Error!bool {
    return runFiltered(allocator, caller, lookup, null);
}

/// Like `run`, but only inline call sites the `filter` permits (e.g. hot ones).
pub fn runFiltered(allocator: std.mem.Allocator, caller: *Function, lookup: Lookup, filter: ?Filter) Error!bool {
    var changed = false;
    var budget: usize = 256; // guard against pathological expansion
    while (budget > 0) : (budget -= 1) {
        if (!try inlineOne(allocator, caller, lookup, filter)) break;
        changed = true;
    }
    return changed;
}

fn scalar(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .bool, .int, .float, .ptr => true,
        else => false,
    };
}

/// Whether `callee` is simple enough for this pass to inline.
fn inlinable(callee: *const Function) bool {
    // SM12 T3: a variadic-defining callee's `va_start`/`va_arg` machinery reads the CALL
    // SITE's own spilled unnamed arguments (a later task's backend concern) - inlining its
    // body would need to rewire that plumbing too, which this pass does not do. `va_start`/
    // `va_end` (no result) are already excluded by the result-less check below; `va_arg`
    // (has a result) needs this explicit guard.
    if (callee.is_variadic) return false;
    if (callee.blockCount() != 1) return false;
    const entry: Block = @enumFromInt(0);
    const term = callee.terminator(entry) orelse return false;
    if (term != .ret) return false;
    // A multi-value return (SM14 M4d-a: a register-pair or HFA struct return) is not rewired
    // by this pass yet. Refuse it, the same way every backend fails closed on it. This never
    // fires today, since no frontend emits a multi-value return before M4d-c.
    if (term.ret.count > 1) return false;
    for (callee.blockParams(entry)) |p| if (!scalar(callee, callee.valueType(p))) return false;
    for (callee.blockInsts(entry)) |inst| {
        const result = callee.instResult(inst) orelse return false; // store/if/void-call
        if (callee.opcode(inst) == .call or callee.opcode(inst) == .call_indirect) return false; // keep it leaf
        if (!scalar(callee, callee.valueType(result))) return false;
        if (callee.opcode(inst) == .alloca and !scalar(callee, callee.opcode(inst).alloca.elem)) return false;
    }
    return true;
}

/// Find and inline a single call, returning whether one was inlined.
fn inlineOne(allocator: std.mem.Allocator, caller: *Function, lookup: Lookup, filter: ?Filter) Error!bool {
    for (0..caller.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (caller.blockInsts(block), 0..) |inst, idx| {
            if (caller.opcode(inst) != .call) continue;
            if (filter) |f| if (!f.allow(bi)) continue;
            const name = caller.symbolName(caller.opcode(inst).call.symbol);
            const callee = lookup.get(name) orelse continue;
            if (inlinable(callee)) {
                try inlineCall(allocator, caller, @intCast(bi), idx, inst, callee); // single-block fast path
                return true;
            }
            if (inlinableMulti(callee)) {
                try inlineCallMulti(allocator, caller, @intCast(bi), idx, inst, callee);
                return true;
            }
            continue;
        }
    }
    return false;
}

fn inlineCall(allocator: std.mem.Allocator, caller: *Function, bi: u32, call_idx: usize, call_inst: Inst, callee: *const Function) Error!void {
    const entry: Block = @enumFromInt(0);
    const block: Block = @enumFromInt(bi);

    // Copy the call's arguments and result before mutating the caller (the value
    // pool may reallocate during cloning).
    const call = caller.opcode(call_inst).call;
    const arg_src = caller.valueList(call.args);
    const args = try allocator.dupe(Value, arg_src);
    defer allocator.free(args);
    const call_result = caller.instResult(call_inst);

    var vmap: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer vmap.deinit(allocator);
    var tmap: std.AutoHashMapUnmanaged(Type, Type) = .empty;
    defer tmap.deinit(allocator);

    // Callee parameters map to the call arguments.
    //
    // A parameter's attributes deliberately STOP here. The argument is a value the caller
    // already owns and already uses elsewhere, so stamping the callee's parameter attributes
    // onto it would change how the caller's own value reads everywhere it appears, not just
    // inside the inlined body. `inlineCallMulti` gives the callee's parameters fresh caller
    // parameters, so there the attributes do travel.
    for (callee.blockParams(entry), 0..) |p, k| try vmap.put(allocator, p, args[k]);

    // The `callee entity -> caller clone` record for the attribute copy below.
    var value_pairs: std.ArrayList(Function.ValuePair) = .empty;
    defer value_pairs.deinit(allocator);
    var inst_pairs: std.ArrayList(Function.InstPair) = .empty;
    defer inst_pairs.deinit(allocator);

    // Clone each callee instruction onto the end of the caller's block.
    const old_len = caller.blockInsts(block).len;
    for (callee.blockInsts(entry)) |cinst| {
        const cres = callee.instResult(cinst).?;
        const rty = try mapType(caller, callee, &tmap, callee.valueType(cres));
        const op = try mapOpcode(caller, callee, vmap, &tmap, callee.opcode(cinst));
        const nres = try caller.appendInst(block, rty, op);
        try vmap.put(allocator, cres, nres);
        try value_pairs.append(allocator, .{ .old = cres, .new = nres });
        try inst_pairs.append(allocator, .{ .old = cinst, .new = caller.definingInst(nres).? });
    }

    // Carry the callee's instruction and result attributes onto the copies. An inlined
    // `endian` load that came back plain is a byte order the backend stops applying, the same
    // class of loss as an inlined `volatile` load that came back ordinary.
    try caller.cloneAttrsFrom(callee, value_pairs.items, inst_pairs.items);

    // The callee's returned value replaces the call's result everywhere.
    if (call_result) |r| {
        const callee_ret = callee.terminator(entry).?.ret;
        if (callee_ret.count == 1) {
            substituteValue(caller, r, vmap.get(callee_ret.values[0]).?);
        }
    }

    // Splice the cloned instructions (now at the block tail) into the call's
    // position, dropping the call itself.
    try reorder(allocator, caller, block, call_idx, old_len);
}

/// Rebuild the block's instruction list so the freshly-cloned instructions (at
/// indices `>= old_len`) sit where the call was, and the call is removed.
fn reorder(allocator: std.mem.Allocator, caller: *Function, block: Block, call_idx: usize, old_len: usize) Error!void {
    const insts = caller.blockInstsMut(block);
    var rebuilt: std.ArrayList(Inst) = .empty;
    defer rebuilt.deinit(allocator);
    try rebuilt.appendSlice(allocator, insts.items[0..call_idx]); // before the call
    try rebuilt.appendSlice(allocator, insts.items[old_len..]); // the clones
    try rebuilt.appendSlice(allocator, insts.items[call_idx + 1 .. old_len]); // after the call
    insts.clearRetainingCapacity();
    try insts.appendSlice(allocator, rebuilt.items);
}

/// Re-intern a callee (scalar) type in the caller's type table.
fn mapType(caller: *Function, callee: *const Function, tmap: *std.AutoHashMapUnmanaged(Type, Type), ty: Type) Error!Type {
    if (tmap.get(ty)) |m| return m;
    const mapped = try caller.types.intern(callee.types.type_kind(ty)); // scalar kinds carry no nested types
    try tmap.put(caller.allocator, ty, mapped);
    return mapped;
}

fn mapOpcode(caller: *Function, callee: *const Function, vmap: std.AutoHashMapUnmanaged(Value, Value), tmap: *std.AutoHashMapUnmanaged(Type, Type), op: Opcode) Error!Opcode {
    const m = struct {
        fn v(vm: std.AutoHashMapUnmanaged(Value, Value), x: Value) Value {
            return vm.get(x).?;
        }
    }.v;
    return switch (op) {
        .iconst, .fconst, .fconst128 => op,
        .arith => |a| .{ .arith = .{ .op = a.op, .lhs = m(vmap, a.lhs), .rhs = m(vmap, a.rhs) } },
        .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = m(vmap, a.lhs), .imm = a.imm } },
        .icmp => |c| .{ .icmp = .{ .op = c.op, .lhs = m(vmap, c.lhs), .rhs = m(vmap, c.rhs) } },
        .select => |s| .{ .select = .{ .cond = m(vmap, s.cond), .then = m(vmap, s.then), .@"else" = m(vmap, s.@"else") } },
        .convert => |cv| .{ .convert = .{ .value = m(vmap, cv.value) } },
        .unary => |u| .{ .unary = .{ .op = u.op, .value = m(vmap, u.value) } },
        // `volatile` is an observable side effect, not a hint. A callee that reads an MMIO
        // register keeps that read observable after it is inlined, so carry the flag.
        .load => |l| .{ .load = .{ .ptr = m(vmap, l.ptr), .@"volatile" = l.@"volatile" } },
        .alloca => |al| .{ .alloca = .{ .elem = try mapType(caller, callee, tmap, al.elem) } },
        .global_addr => |ga| .{ .global_addr = .{ .symbol = try caller.internSymbol(callee.symbolName(ga.symbol)), .via_got = ga.via_got } },
        // dot is pure, like arith: remap its 3 operands. (Its vector operand types
        // fail the `scalar` gate today, so this is unreachable in practice, but the
        // remap is here so a future vector-aware inline path needs no new wiring.)
        .dot => |d| .{ .dot = .{ .acc = m(vmap, d.acc), .a = m(vmap, d.a), .b = m(vmap, d.b) } },
        // Inlining an atomic is a straight code move: the caller runs it exactly as often
        // as the callee did, so the count of read-modify-writes is unchanged. Only the
        // three operands need remapping; the operation, ordering and scope copy unchanged.
        // The reduction form never reaches here (`inlinable` refuses a result-less
        // instruction); the multi-block path builds that one itself.
        .atomic_rmw => |a| blk: {
            var mapped = a;
            mapped.ptr = m(vmap, a.ptr);
            mapped.value = m(vmap, a.value);
            if (a.compare) |c| mapped.compare = m(vmap, c);
            break :blk .{ .atomic_rmw = mapped };
        },
        // Excluded by `inlinable`: these never reach here. `va_start`/`va_arg`/`va_end` are
        // excluded by `inlinable`'s `callee.is_variadic` guard (SM12 T3) - a variadic callee
        // is never considered inlinable at all, so these three never reach here either.
        // `barrier` joins them: it has no result, so `inlinable`'s result-less check refuses
        // a callee that contains one and this arm is never reached.
        .extract, .struct_new, .store, .prefetch, .matmul, .call, .call_indirect, .@"if", .va_start, .va_arg, .va_end, .barrier => unreachable,
    };
}

/// Replace every use of `from` with `to` across all instructions, `if` edges,
/// and terminators.
fn substituteValue(func: *Function, from: Value, to: Value) void {
    const r = struct {
        fn repl(f: Value, t: Value, v: Value) Value {
            return if (v == f) t else v;
        }
    }.repl;
    for (0..func.instCount()) |i| {
        const op = func.opcodeMut(@enumFromInt(i));
        switch (op.*) {
            .atomic_rmw => |*a| {
                a.ptr = r(from, to, a.ptr);
                a.value = r(from, to, a.value);
                if (a.compare) |*c| c.* = r(from, to, c.*);
            },
            // A barrier carries no Value operand to substitute.
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
            .struct_new => |sn| for (func.valueListMut(sn.fields)) |*f| {
                f.* = r(from, to, f.*);
            },
            .call => |c| for (func.valueListMut(c.args)) |*arg| {
                arg.* = r(from, to, arg.*);
            },
            .call_indirect => |*c| {
                c.target = r(from, to, c.target);
                for (func.valueListMut(c.args)) |*arg| arg.* = r(from, to, arg.*);
            },
            .@"if" => |*cf| {
                cf.cond = r(from, to, cf.cond);
                for (func.valueListMut(cf.then.args)) |*arg| arg.* = r(from, to, arg.*);
                for (func.valueListMut(cf.@"else".args)) |*arg| arg.* = r(from, to, arg.*);
            },
        }
    }
    for (0..func.blockCount()) |bi| {
        const term = func.terminatorPtr(@enumFromInt(bi));
        if (term.*) |*t| switch (t.*) {
            .ret => |*ret| for (ret.values[0..ret.count]) |*vv| {
                vv.* = r(from, to, vv.*);
            },
            .jump => |*j| for (func.valueListMut(j.args)) |*arg| {
                arg.* = r(from, to, arg.*);
            },
        };
    }
}

/// Whether `callee` (any number of blocks) can be inlined by the multi-block path. It must be a leaf
/// with scalar parameter and result types and no aggregate ops. Control flow, stores, loops, and
/// multiple returns are fine, since each `ret` turns into a jump to a continuation block.
fn inlinableMulti(callee: *const Function) bool {
    // SM12 T3: see `inlinable`'s matching guard - never inline a variadic-defining callee.
    if (callee.is_variadic) return false;
    for (0..callee.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (callee.blockParams(block)) |p| if (!scalar(callee, callee.valueType(p))) return false;
        // A multi-value return in any block is not rewired by this pass (SM14 M4d-a). Refuse
        // it, matching the backends' fail-closed stance. Never fires before M4d-c.
        if (callee.terminator(block)) |term| {
            if (term == .ret and term.ret.count > 1) return false;
        }
        for (callee.blockInsts(block)) |inst| switch (callee.opcode(inst)) {
            .call, .call_indirect => return false, // keep it leaf, no nested inlining here
            .struct_new, .extract => return false, // aggregate type remap is not handled
            .@"if" => {}, // control flow, cloned in a later pass
            .store => |st| _ = st, // yields no value, cloned in a later pass
            .prefetch => |pf| _ = pf, // yields no value, cloned in a later pass
            // Either form is fine: the reading one has a result, the reduction one has
            // none, and the clone loop below builds whichever it finds.
            .atomic_rmw => |a| _ = a,
            .matmul => |mm| _ = mm, // yields no value, cloned in a later pass
            .alloca => |al| if (!scalar(callee, al.elem)) return false,
            else => {
                const r = callee.instResult(inst) orelse return false;
                if (!scalar(callee, callee.valueType(r))) return false;
            },
        };
    }
    return true;
}

/// The successor block indices of `b` (from its `if` exit instruction or its jump terminator).
fn successorsOf(callee: *const Function, b: u32, buf: *[2]u32) []const u32 {
    const block: Block = @enumFromInt(b);
    for (callee.blockInsts(block)) |inst| {
        if (callee.opcode(inst) == .@"if") {
            const cf = callee.opcode(inst).@"if";
            buf[0] = @intFromEnum(cf.then.target);
            buf[1] = @intFromEnum(cf.@"else".target);
            return buf[0..2];
        }
    }
    if (callee.terminator(block)) |t| switch (t) {
        .jump => |j| {
            buf[0] = @intFromEnum(j.target);
            return buf[0..1];
        },
        .ret => {},
    };
    return buf[0..0];
}

/// The reachable blocks of `callee` in reverse-postorder (so a value's defining block precedes every
/// use, letting a single forward clone pass resolve all instruction operands). Caller owns the slice.
fn reachableRpo(allocator: std.mem.Allocator, callee: *const Function) Error![]u32 {
    const n = callee.blockCount();
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);
    var order: std.ArrayList(u32) = .empty;
    errdefer order.deinit(allocator);
    var stack: std.ArrayList(struct { b: u32, ci: usize }) = .empty;
    defer stack.deinit(allocator);
    if (n > 0) {
        visited[0] = true;
        try stack.append(allocator, .{ .b = 0, .ci = 0 });
    }
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        var buf: [2]u32 = undefined;
        const succs = successorsOf(callee, top.b, &buf);
        if (top.ci < succs.len) {
            const s = succs[top.ci];
            top.ci += 1;
            if (!visited[s]) {
                visited[s] = true;
                try stack.append(allocator, .{ .b = s, .ci = 0 });
            }
        } else {
            try order.append(allocator, top.b); // postorder: emit after all successors
            _ = stack.pop();
        }
    }
    std.mem.reverse(u32, order.items); // postorder reversed = RPO
    return order.toOwnedSlice(allocator);
}

fn mapV(vmap: std.AutoHashMapUnmanaged(Value, Value), v: Value) Value {
    return vmap.get(v).?;
}

/// Remap a callee value list (an edge's arguments) into fresh caller values. Caller owns the slice.
fn remapArgs(allocator: std.mem.Allocator, callee: *const Function, vmap: std.AutoHashMapUnmanaged(Value, Value), list: ir.function.ValueList) Error![]Value {
    const src = callee.valueList(list);
    const out = try allocator.alloc(Value, src.len);
    for (src, 0..) |v, i| out[i] = mapV(vmap, v);
    return out;
}

fn inlineCallMulti(allocator: std.mem.Allocator, caller: *Function, bi: u32, call_idx: usize, call_inst: Inst, callee: *const Function) Error!void {
    const b_block: Block = @enumFromInt(bi);
    const call = caller.opcode(call_inst).call;
    const args = try allocator.dupe(Value, caller.valueList(call.args));
    defer allocator.free(args);
    const call_result = caller.instResult(call_inst);

    // Split the caller block at the call: a continuation block takes the code after the call, and the
    // call's result becomes the continuation's parameter (fed by each inlined `ret`).
    const cont = try caller.appendBlock();
    var cont_param: ?Value = null;
    if (call_result) |r| cont_param = try caller.appendBlockParam(cont, caller.valueType(r));
    {
        const b_insts = caller.blockInstsMut(b_block);
        for (b_insts.items[call_idx + 1 ..]) |inst| try caller.blockInstsMut(cont).append(allocator, inst);
        b_insts.shrinkRetainingCapacity(call_idx); // drop the call and the moved tail
    }
    caller.terminatorPtr(cont).* = caller.terminatorPtr(b_block).*; // CONT inherits B's original exit
    caller.terminatorPtr(b_block).* = null;
    if (call_result) |r| substituteValue(caller, r, cont_param.?);

    // Clone the callee's reachable blocks.
    var vmap: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer vmap.deinit(allocator);
    var tmap: std.AutoHashMapUnmanaged(Type, Type) = .empty;
    defer tmap.deinit(allocator);
    var bmap: std.AutoHashMapUnmanaged(u32, Block) = .empty;
    defer bmap.deinit(allocator);
    const rpo = try reachableRpo(allocator, callee);
    defer allocator.free(rpo);

    // The `callee entity -> caller clone` record for the attribute copy at the end.
    var value_pairs: std.ArrayList(Function.ValuePair) = .empty;
    defer value_pairs.deinit(allocator);
    var inst_pairs: std.ArrayList(Function.InstPair) = .empty;
    defer inst_pairs.deinit(allocator);

    for (rpo) |cb| try bmap.put(allocator, cb, try caller.appendBlock());
    for (rpo) |cb| { // params of every cloned block (the entry's take the call args by jump)
        const nb = bmap.get(cb).?;
        for (callee.blockParams(@enumFromInt(cb))) |p| {
            const np = try caller.appendBlockParam(nb, try mapType(caller, callee, &tmap, callee.valueType(p)));
            try vmap.put(allocator, p, np);
            // This path gives every callee parameter, the entry's included, a FRESH caller
            // parameter, so a parameter attribute describes the copy just as well as the
            // original and travels with it.
            try value_pairs.append(allocator, .{ .old = p, .new = np });
        }
    }
    for (rpo) |cb| { // instructions, in RPO so operands are already mapped. `if` is handled below
        const nb = bmap.get(cb).?;
        for (callee.blockInsts(@enumFromInt(cb))) |cinst| switch (callee.opcode(cinst)) {
            // The `if` is rebuilt by the control-flow pass below, which records its own pair.
            .@"if" => {},
            // `appendStoreVol`, not `appendStore`: `appendStore` hardcodes `volatile = false`, which
            // turns an MMIO write in the callee into an ordinary store the next pass can move or
            // delete. Carry the callee op's own flag instead.
            .store => |st| {
                try caller.appendStoreVol(nb, mapV(vmap, st.value), mapV(vmap, st.ptr), st.@"volatile");
                try inst_pairs.append(allocator, .{ .old = cinst, .new = lastInst(caller, nb) });
            },
            .prefetch => |pf| {
                try caller.appendPrefetch(nb, mapV(vmap, pf.ptr));
                try inst_pairs.append(allocator, .{ .old = cinst, .new = lastInst(caller, nb) });
            },
            // The result is OPTIONAL, so this arm cannot go through the `else` below,
            // which unwraps a result that a reduction-form atomic does not have. Build
            // whichever form the callee held.
            .atomic_rmw => |a| {
                var mapped = a;
                mapped.ptr = mapV(vmap, a.ptr);
                mapped.value = mapV(vmap, a.value);
                if (a.compare) |c| mapped.compare = mapV(vmap, c);
                if (callee.instResult(cinst)) |cres| {
                    const nres = try caller.appendAtomicRmw(nb, mapped);
                    try vmap.put(allocator, cres, nres);
                    try value_pairs.append(allocator, .{ .old = cres, .new = nres });
                    try inst_pairs.append(allocator, .{ .old = cinst, .new = caller.definingInst(nres).? });
                } else {
                    try caller.appendAtomicRmwStmt(nb, mapped);
                    try inst_pairs.append(allocator, .{ .old = cinst, .new = lastInst(caller, nb) });
                }
            },
            // A `per_column` scale's ScaleList handle (and a bias's BiasList handle) is relative to
            // the CALLEE's pools, which is meaningless in the caller (a different function, different
            // pools). Re-intern both through the spec builder so each handle is re-resolved via the
            // CALLEE's scaleList/biasList and re-interned into the CALLER's pools; scalar/null quants
            // carry no pool handle, so this still round-trips them unchanged.
            .matmul => |mm| {
                if (mm.quant) |q| {
                    const spec: Function.MatMulQuantSpec = .{
                        .scale_scalar = switch (q.scale) {
                            .scalar => |sb| sb,
                            .per_column => null,
                        },
                        .scale_per_column = switch (q.scale) {
                            .scalar => null,
                            .per_column => |h| callee.scaleList(h),
                        },
                        .bias = if (q.bias) |bh| callee.biasList(bh) else null,
                        .zero_point = q.zero_point,
                        .relu = q.relu,
                        .out = q.out,
                        .input_signs = mm.input_signs, // plain value, no pool handle, copies as-is
                    };
                    try caller.appendMatmulQuantSpec(nb, mapV(vmap, mm.a), mapV(vmap, mm.b), mapV(vmap, mm.c), mm.m, mm.n, mm.k, mm.dtype, mm.accumulate, spec);
                } else if (mm.input_signs) |s|
                    try caller.appendMatmulSigned(nb, mapV(vmap, mm.a), mapV(vmap, mm.b), mapV(vmap, mm.c), mm.m, mm.n, mm.k, mm.dtype, mm.accumulate, s)
                else
                    try caller.appendMatmul(nb, mapV(vmap, mm.a), mapV(vmap, mm.b), mapV(vmap, mm.c), mm.m, mm.n, mm.k, mm.dtype, mm.accumulate);
                // The builders above default `embedded` to false; carry the callee op's flag onto the
                // just-appended matmul (the last inst in nb) so a self-contained matmul stays
                // self-contained after inlining into a caller that has live values around it.
                const last = lastInst(caller, nb);
                // The matmul builders each append exactly one statement, so the last inst in nb is
                // that matmul; assert it rather than silently flag-flip an unrelated opcode.
                std.debug.assert(caller.opcode(last) == .matmul);
                if (mm.embedded) caller.opcodeMut(last).matmul.embedded = true;
                try inst_pairs.append(allocator, .{ .old = cinst, .new = last });
            },
            else => |op| {
                const cres = callee.instResult(cinst).?;
                const rty = try mapType(caller, callee, &tmap, callee.valueType(cres));
                const nres = try caller.appendInst(nb, rty, try mapOpcode(caller, callee, vmap, &tmap, op));
                try vmap.put(allocator, cres, nres);
                try value_pairs.append(allocator, .{ .old = cres, .new = nres });
                try inst_pairs.append(allocator, .{ .old = cinst, .new = caller.definingInst(nres).? });
            },
        };
    }
    for (rpo) |cb| { // control flow: `if` exits and jump/ret terminators
        const nb = bmap.get(cb).?;
        const cblock: Block = @enumFromInt(cb);
        var if_cf: ?ir.function.If = null;
        var if_src: ?Inst = null;
        for (callee.blockInsts(cblock)) |cinst| {
            if (callee.opcode(cinst) == .@"if") {
                if_cf = callee.opcode(cinst).@"if";
                if_src = cinst;
                break;
            }
        }
        if (if_cf) |cf| {
            const ta = try remapArgs(allocator, callee, vmap, cf.then.args);
            defer allocator.free(ta);
            const ea = try remapArgs(allocator, callee, vmap, cf.@"else".args);
            defer allocator.free(ea);
            try caller.appendIf(nb, mapV(vmap, cf.cond), .{ .target = bmap.get(@intFromEnum(cf.then.target)).?, .args = ta }, .{ .target = bmap.get(@intFromEnum(cf.@"else".target)).?, .args = ea });
            try inst_pairs.append(allocator, .{ .old = if_src.?, .new = lastInst(caller, nb) });
            continue;
        }
        if (callee.terminator(cblock)) |t| switch (t) {
            .ret => |r| {
                var mapped: [4]Value = undefined;
                for (r.slice(), 0..) |rv, i| mapped[i] = mapV(vmap, rv);
                try caller.setJump(nb, cont, mapped[0..r.count]);
            },
            .jump => |j| {
                const ja = try remapArgs(allocator, callee, vmap, j.args);
                defer allocator.free(ja);
                try caller.setJump(nb, bmap.get(@intFromEnum(j.target)).?, ja);
            },
        };
    }

    // Carry the callee's parameter, instruction and result attributes onto the copies, now that
    // every copy exists. An inlined `endian` load that came back plain is a byte order the
    // backend stops applying, the same class of loss as an inlined `volatile` load that came
    // back ordinary.
    try caller.cloneAttrsFrom(callee, value_pairs.items, inst_pairs.items);

    // Enter the inlined body from the (now truncated) caller block, passing the call arguments.
    try caller.setJump(b_block, bmap.get(0).?, args);
}

/// The instruction a statement builder just appended to `block`: the block's last one. Each
/// builder above appends exactly one instruction and returns nothing, so this recovers its
/// handle.
fn lastInst(caller: *const Function, block: Block) Inst {
    const appended = caller.blockInsts(block);
    return appended[appended.len - 1];
}

const TestLookup = struct {
    callee: *const Function,
    name: []const u8,
    fn get(ctx: *anyopaque, name: []const u8) ?*const Function {
        const self: *TestLookup = @ptrCast(@alignCast(ctx));
        return if (std.mem.eql(u8, name, self.name)) self.callee else null;
    }
};

test "inlines a leaf helper and replaces the call result" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee: madd(a, b) = a*b + a
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        const bb = try callee.appendBlockParam(b, t);
        const prod = try callee.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
        const sum = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });
    }

    // caller: f(x) = madd(x, x) + 1
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const b = try caller.appendBlock();
    const x = try caller.appendBlockParam(b, t);
    const call = try caller.appendCall(b, t, "madd", &.{ x, x });
    const r = try caller.appendArithImm(b, t, .add, call, 1);
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var lk = TestLookup{ .callee = &callee, .name = "madd" };
    const lookup = Lookup{ .context = &lk, .func = TestLookup.get };
    try std.testing.expect(try run(allocator, &caller, lookup));

    // The call is gone, replaced by the cloned mul/add, and `r` adds 1 to the
    // inlined sum.
    for (caller.blockInsts(b)) |inst| try std.testing.expect(caller.opcode(inst) != .call);
}

test "inlines a multi-block, two-return callee (the call is replaced by cloned control flow)" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee max(a, b): if a > b return a else return b  (3 blocks, two `ret`s)
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const bool_t = try callee.types.intern(.bool);
        const entry = try callee.appendBlock();
        const tb = try callee.appendBlock();
        const eb = try callee.appendBlock();
        const a = try callee.appendBlockParam(entry, t);
        const b = try callee.appendBlockParam(entry, t);
        const tv = try callee.appendBlockParam(tb, t);
        const ev = try callee.appendBlockParam(eb, t);
        const cmp = try callee.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        try callee.appendIf(entry, cmp, .{ .target = tb, .args = &.{a} }, .{ .target = eb, .args = &.{b} });
        callee.setTerminator(tb, .{ .ret = ir.function.Ret.one(tv) });
        callee.setTerminator(eb, .{ .ret = ir.function.Ret.one(ev) });
    }

    // caller f(x): return max(x, 5) + 1
    var caller = Function.init(allocator);
    defer caller.deinit();
    const ct = try caller.types.intern(i32k);
    const cb = try caller.appendBlock();
    const x = try caller.appendBlockParam(cb, ct);
    const c5 = try caller.appendInst(cb, ct, .{ .iconst = 5 });
    const m = try caller.appendCall(cb, ct, "max", &.{ x, c5 });
    _ = try caller.appendArithImm(cb, ct, .add, m, 1);
    caller.setTerminator(cb, .{ .ret = ir.function.Ret.one(m) }); // the add stays separate, which is fine for a structural check

    var lk = TestLookup{ .callee = &callee, .name = "max" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));
    // The call instruction is gone: it was replaced by the cloned callee body.
    for (0..caller.blockCount()) |bi| {
        for (caller.blockInsts(@enumFromInt(bi))) |inst| {
            try std.testing.expect(caller.opcode(inst) != .call);
        }
    }
    try std.testing.expect(caller.blockCount() > 1); // control flow was cloned in
}

test "inlined via_got global_addr keeps via_got=true (not reconstructed as false)" {
    const allocator = std.testing.allocator;

    // callee getg(): return &G (via GOT)
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const g = try callee.appendGlobalAddrGot(b, ptr_t, "G");
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(g) });
    }

    // caller f(): return getg()
    var caller = Function.init(allocator);
    defer caller.deinit();
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const call = try caller.appendCall(b, ptr_t, "getg", &.{});
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "getg" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    // The call is gone, replaced by the cloned global_addr. It must still carry
    // via_got=true: `mapOpcode`'s `.global_addr` arm must forward the flag, not
    // rebuild the op from just `.symbol` (which would silently default it false
    // and turn GOT-indirect addressing into direct addressing).
    var found = false;
    for (caller.blockInsts(b)) |inst| {
        if (caller.opcode(inst) == .global_addr) {
            found = true;
            try std.testing.expect(caller.opcode(inst).global_addr.via_got);
        }
    }
    try std.testing.expect(found);
}

test "inlined volatile load keeps volatile=true (single-block path)" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee read_reg(p): return *(volatile int *)p  -- the canonical MMIO read helper.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const p = try callee.appendBlockParam(b, ptr_t);
        const v = try callee.appendInst(b, t, .{ .load = .{ .ptr = p, .@"volatile" = true } });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }

    // caller f(p): return read_reg(p)
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const p = try caller.appendBlockParam(b, ptr_t);
    const call = try caller.appendCall(b, t, "read_reg", &.{p});
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "read_reg" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    // The cloned load must still be volatile. `mapOpcode`'s `.load` arm must forward the
    // flag; a rebuild from just `.ptr` defaults it to false and makes a hardware register
    // read an ordinary load that a later pass can move, duplicate or delete.
    var loads: usize = 0;
    for (caller.blockInsts(b)) |inst| {
        if (caller.opcode(inst) == .load) {
            loads += 1;
            try std.testing.expect(caller.opcode(inst).load.@"volatile");
        }
    }
    try std.testing.expectEqual(@as(usize, 1), loads);
}

test "inlined volatile store and volatile load both keep volatile=true (multi-block path)" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee rmw(dst, src, v): *(volatile int *)dst = v; return *(volatile int *)src
    // The store gives it no result, so `inlinable` refuses it and `inlineCallMulti` takes it.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const dst = try callee.appendBlockParam(b, ptr_t);
        const src = try callee.appendBlockParam(b, ptr_t);
        const v = try callee.appendBlockParam(b, t);
        try callee.appendStoreVol(b, v, dst, true);
        const l = try callee.appendInst(b, t, .{ .load = .{ .ptr = src, .@"volatile" = true } });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });
    }

    // caller f(dst, src, v): return rmw(dst, src, v)
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const dst = try caller.appendBlockParam(b, ptr_t);
    const src = try caller.appendBlockParam(b, ptr_t);
    const v = try caller.appendBlockParam(b, t);
    const call = try caller.appendCall(b, t, "rmw", &.{ dst, src, v });
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "rmw" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    // Both accesses must survive with the flag SET. The store goes through `appendStoreVol`,
    // since `appendStore` hardcodes `volatile = false`; the load goes through `mapOpcode`.
    var stores: usize = 0;
    var loads: usize = 0;
    for (0..caller.blockCount()) |bi| {
        for (caller.blockInsts(@enumFromInt(bi))) |inst| switch (caller.opcode(inst)) {
            .store => |st| {
                stores += 1;
                try std.testing.expect(st.@"volatile");
            },
            .load => |ld| {
                loads += 1;
                try std.testing.expect(ld.@"volatile");
            },
            else => {},
        };
    }
    try std.testing.expectEqual(@as(usize, 1), stores);
    try std.testing.expectEqual(@as(usize, 1), loads);
}

test "inlining a plain load and store leaves volatile=false (the flag is carried, not forced)" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee copy(dst, src): *dst = *src; return *src  -- ordinary, non-volatile memory.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const dst = try callee.appendBlockParam(b, ptr_t);
        const src = try callee.appendBlockParam(b, ptr_t);
        const l = try callee.appendInst(b, t, .{ .load = .{ .ptr = src } });
        try callee.appendStore(b, l, dst);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const dst = try caller.appendBlockParam(b, ptr_t);
    const src = try caller.appendBlockParam(b, ptr_t);
    const call = try caller.appendCall(b, t, "copy", &.{ dst, src });
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "copy" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    // The other direction: a non-volatile access must NOT become volatile, which would block
    // every legal optimization on ordinary memory.
    for (0..caller.blockCount()) |bi| {
        for (caller.blockInsts(@enumFromInt(bi))) |inst| switch (caller.opcode(inst)) {
            .store => |st| try std.testing.expect(!st.@"volatile"),
            .load => |ld| try std.testing.expect(!ld.@"volatile"),
            else => {},
        };
    }
}

/// The int payload of the first `namespace.key` attribute on `target`, or null when absent.
fn testCustomInt(func: *const Function, target: ir.function.AttrTarget, namespace: []const u8, key: []const u8) ?i64 {
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
fn testAttrCount(func: *const Function, target: ir.function.AttrTarget) usize {
    var n: usize = 0;
    var it = func.attributesOf(target);
    while (it.next()) |_| n += 1;
    return n;
}

test "inlining carries the callee load's endian attribute onto the clone (single-block path)" {
    // `endian` picks a byte-swapping load in the riscv64 backend. A callee helper that reads a
    // big-endian field and comes back plain after inlining reads the bytes in the wrong order,
    // the same class of loss as an inlined `volatile` load that comes back ordinary.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee read_be(p): return *p, with the result tagged big-endian.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const p = try callee.appendBlockParam(b, ptr_t);
        const v = try callee.appendInst(b, t, .{ .load = .{ .ptr = p } });
        try callee.addAttr(.{ .value = v }, .{ .endian = .big });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const p = try caller.appendBlockParam(b, ptr_t);
    const call = try caller.appendCall(b, t, "read_be", &.{p});
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "read_be" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    var tagged: usize = 0;
    for (caller.blockInsts(b)) |inst| {
        if (caller.opcode(inst) != .load) continue;
        const result = caller.instResult(inst).?;
        var it = caller.attributesOf(.{ .value = result });
        try std.testing.expectEqual(ir.function.Attribute{ .endian = .big }, it.next().?);
        tagged += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), tagged);
}

test "inlining does not stamp a callee parameter's attribute onto the caller's argument (single-block path)" {
    // Suspicious case, the other direction. On this path a callee parameter maps to a value the
    // CALLER already owns and uses elsewhere. Copying the parameter's attribute onto it would
    // change how the caller's own value reads everywhere it appears, not only inside the body.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        try callee.addAttr(.{ .value = a }, .{ .@"align" = 16 });
        const sum = try callee.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const b = try caller.appendBlock();
    const x = try caller.appendBlockParam(b, t);
    const call = try caller.appendCall(b, t, "twice", &.{x});
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "twice" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    try std.testing.expectEqual(@as(usize, 0), testAttrCount(&caller, .{ .value = x }));
    // The whole caller gains nothing: the callee held only that one parameter attribute.
    try std.testing.expectEqual(@as(usize, 0), caller.attributeEntries().len);
}

test "inlining carries a parameter and a store attribute onto the clone (multi-block path)" {
    // This path gives every callee parameter a FRESH caller parameter, so a parameter attribute
    // describes the copy just as well as the original and must travel with it.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee rmw(dst, src, v): *dst = v; return *src. The store leaves it result-less, so
    // `inlinable` refuses it and `inlineCallMulti` takes it.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const dst = try callee.appendBlockParam(b, ptr_t);
        const src = try callee.appendBlockParam(b, ptr_t);
        const v = try callee.appendBlockParam(b, t);
        try callee.appendStoreVol(b, v, dst, false);
        const store_inst = callee.blockInsts(b)[0];
        const l = try callee.appendInst(b, t, .{ .load = .{ .ptr = src } });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });

        try callee.addAttr(.{ .value = dst }, .{ .@"align" = 16 });
        try callee.addAttr(.{ .inst = store_inst }, .{ .custom = .{
            .namespace = "debug",
            .key = "line",
            .value = .{ .int = 77 },
        } });
        try callee.addAttr(.{ .value = l }, .{ .endian = .big });
        try callee.addAttr(.func, .@"inline"); // must NOT travel: it describes the callee
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const dst = try caller.appendBlockParam(b, ptr_t);
    const src = try caller.appendBlockParam(b, ptr_t);
    const v = try caller.appendBlockParam(b, t);
    const call = try caller.appendCall(b, t, "rmw", &.{ dst, src, v });
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "rmw" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    // The cloned body block's first parameter is the clone of the callee's `dst`.
    var aligned: usize = 0;
    var lines: usize = 0;
    var big: usize = 0;
    for (caller.attributeEntries()) |entry| switch (entry.attr) {
        .@"align" => |n| {
            try std.testing.expectEqual(@as(u32, 16), n);
            aligned += 1;
        },
        .endian => |e| {
            try std.testing.expectEqual(ir.function.Attribute{ .endian = .big }, ir.function.Attribute{ .endian = e });
            big += 1;
        },
        .custom => {
            try std.testing.expectEqual(@as(?i64, 77), testCustomInt(&caller, entry.target, "debug", "line"));
            lines += 1;
        },
        .@"inline", .noreturn, .cold => try std.testing.expect(false), // the callee's, must not travel
    };
    try std.testing.expectEqual(@as(usize, 1), aligned);
    try std.testing.expectEqual(@as(usize, 1), lines);
    try std.testing.expectEqual(@as(usize, 1), big);

    // Each landed on the clone, not on a value the caller already owned.
    try std.testing.expectEqual(@as(usize, 0), testAttrCount(&caller, .{ .value = dst }));
    try std.testing.expectEqual(@as(usize, 0), testAttrCount(&caller, .{ .value = src }));
    try std.testing.expectEqual(@as(usize, 0), testAttrCount(&caller, .func));
}

test "inlining an atomic keeps every field, on the single-block path" {
    // Inlining an atomic is a straight code move: the caller runs it exactly as often as the
    // callee did. `mapOpcode` must remap all three operands and copy the operation, ordering
    // and scope, and it must not reach the `unreachable` arm.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee cas(p, desired, expected): return atomic compare-exchange
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const ptr_t = try callee.types.ptrGlobal();
        const b = try callee.appendBlock();
        const p = try callee.appendBlockParam(b, ptr_t);
        const desired = try callee.appendBlockParam(b, t);
        const expected = try callee.appendBlockParam(b, t);
        const old = try callee.appendAtomicRmw(b, .{
            .op = .compare_exchange,
            .ptr = p,
            .value = desired,
            .compare = expected,
            .ordering = .acq_rel,
            .scope = .system,
        });
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(old) });
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    // The caller's parameters are declared in the REVERSE order of the callee's, so no
    // callee value number coincides with the caller value it maps to. Without that, a
    // missing remap would leave the callee's own handle behind and the check below would
    // still pass by numeric accident.
    const e = try caller.appendBlockParam(b, t);
    const d = try caller.appendBlockParam(b, t);
    const p = try caller.appendBlockParam(b, ptr_t);
    const call = try caller.appendCall(b, t, "cas", &.{ p, d, e });
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "cas" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    var found: usize = 0;
    for (caller.blockInsts(b)) |inst| {
        if (caller.opcode(inst) != .atomic_rmw) continue;
        found += 1;
        const a = caller.opcode(inst).atomic_rmw;
        try std.testing.expectEqual(ir.function.AtomicOp.compare_exchange, a.op);
        try std.testing.expectEqual(ir.function.AtomicOrdering.acq_rel, a.ordering);
        try std.testing.expectEqual(ir.function.AtomicScope.system, a.scope);
        // The operands are the CALLER's arguments now.
        try std.testing.expectEqual(p, a.ptr);
        try std.testing.expectEqual(d, a.value);
        try std.testing.expectEqual(e, a.compare.?);
    }
    try std.testing.expectEqual(@as(usize, 1), found);
}

test "inlining a RESULT-LESS atomic keeps it result-less, on the multi-block path" {
    // The single-block path refuses a result-less callee instruction, so only the multi-block
    // path clones this form. It needs its own arm there: the `else` prong unwraps a result
    // the reduction form does not have, and would panic.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee bump(p, v, c): if (c) atomic add; return 0  -- two blocks, so the multi path runs.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const bool_t = try callee.types.intern(.bool);
        const ptr_t = try callee.types.ptrGlobal();
        const entry = try callee.appendBlock();
        const then_b = try callee.appendBlock();
        const done = try callee.appendBlock();
        const p = try callee.appendBlockParam(entry, ptr_t);
        const v = try callee.appendBlockParam(entry, t);
        const c = try callee.appendBlockParam(entry, bool_t);
        try callee.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = done, .args = &.{} });
        try callee.appendAtomicRmwStmt(then_b, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
        try callee.setJump(then_b, done, &.{});
        const zero = try callee.appendInst(done, t, .{ .iconst = 0 });
        callee.setTerminator(done, .{ .ret = ir.function.Ret.one(zero) });
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const bool_t = try caller.types.intern(.bool);
    const ptr_t = try caller.types.ptrGlobal();
    const b = try caller.appendBlock();
    const p = try caller.appendBlockParam(b, ptr_t);
    const v = try caller.appendBlockParam(b, t);
    const c = try caller.appendBlockParam(b, bool_t);
    const call = try caller.appendCall(b, t, "bump", &.{ p, v, c });
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(call) });

    var lk = TestLookup{ .callee = &callee, .name = "bump" };
    try std.testing.expect(try run(allocator, &caller, .{ .context = &lk, .func = TestLookup.get }));

    var found: usize = 0;
    for (0..caller.blockCount()) |bi| {
        for (caller.blockInsts(@enumFromInt(bi))) |inst| {
            if (caller.opcode(inst) != .atomic_rmw) continue;
            found += 1;
            // Still result-less: it did not gain a destination on the way in.
            try std.testing.expectEqual(@as(?ir.function.Value, null), caller.instResult(inst));
            const a = caller.opcode(inst).atomic_rmw;
            try std.testing.expectEqual(ir.function.AtomicOp.add, a.op);
            try std.testing.expectEqual(ir.function.AtomicOrdering.relaxed, a.ordering);
            try std.testing.expectEqual(ir.function.AtomicScope.device, a.scope);
            try std.testing.expectEqual(@as(?ir.function.Value, null), a.compare);
            // This path gives every callee parameter a FRESH caller parameter and passes
            // the call arguments in by jump, so the operands name those copies, not `p`
            // and `v`. What matters is that they were remapped to something the caller
            // defines, which `verify` below proves by dominance.
            try std.testing.expect(caller.types.type_kind(caller.valueType(a.ptr)) == .ptr);
            try std.testing.expect(caller.types.type_kind(caller.valueType(a.value)) == .int);
        }
    }
    try std.testing.expectEqual(@as(usize, 1), found);

    var diags = try ir.verify.verify(allocator, &caller, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}
