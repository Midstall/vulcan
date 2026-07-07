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
    if (callee.blockCount() != 1) return false;
    const entry: Block = @enumFromInt(0);
    const term = callee.terminator(entry) orelse return false;
    if (term != .ret) return false;
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
    for (callee.blockParams(entry), 0..) |p, k| try vmap.put(allocator, p, args[k]);

    // Clone each callee instruction onto the end of the caller's block.
    const old_len = caller.blockInsts(block).len;
    for (callee.blockInsts(entry)) |cinst| {
        const cres = callee.instResult(cinst).?;
        const rty = try mapType(caller, callee, &tmap, callee.valueType(cres));
        const op = try mapOpcode(caller, callee, vmap, &tmap, callee.opcode(cinst));
        const nres = try caller.appendInst(block, rty, op);
        try vmap.put(allocator, cres, nres);
    }

    // The callee's returned value replaces the call's result everywhere.
    if (call_result) |r| {
        if (callee.terminator(entry).?.ret) |ret_val| {
            substituteValue(caller, r, vmap.get(ret_val).?);
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
        .iconst, .fconst => op,
        .arith => |a| .{ .arith = .{ .op = a.op, .lhs = m(vmap, a.lhs), .rhs = m(vmap, a.rhs) } },
        .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = m(vmap, a.lhs), .imm = a.imm } },
        .icmp => |c| .{ .icmp = .{ .op = c.op, .lhs = m(vmap, c.lhs), .rhs = m(vmap, c.rhs) } },
        .select => |s| .{ .select = .{ .cond = m(vmap, s.cond), .then = m(vmap, s.then), .@"else" = m(vmap, s.@"else") } },
        .convert => |cv| .{ .convert = .{ .value = m(vmap, cv.value) } },
        .unary => |u| .{ .unary = .{ .op = u.op, .value = m(vmap, u.value) } },
        .load => |l| .{ .load = .{ .ptr = m(vmap, l.ptr) } },
        .alloca => |al| .{ .alloca = .{ .elem = try mapType(caller, callee, tmap, al.elem) } },
        .global_addr => |ga| .{ .global_addr = .{ .symbol = try caller.internSymbol(callee.symbolName(ga.symbol)) } },
        // Excluded by `inlinable`: these never reach here.
        .extract, .struct_new, .store, .call, .call_indirect, .@"if" => unreachable,
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
            .iconst, .fconst, .alloca, .global_addr => {},
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
            .ret => |*v| if (v.*) |vv| {
                v.* = r(from, to, vv);
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
    for (0..callee.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (callee.blockParams(block)) |p| if (!scalar(callee, callee.valueType(p))) return false;
        for (callee.blockInsts(block)) |inst| switch (callee.opcode(inst)) {
            .call, .call_indirect => return false, // keep it leaf, no nested inlining here
            .struct_new, .extract => return false, // aggregate type remap is not handled
            .@"if" => {}, // control flow, cloned in a later pass
            .store => |st| _ = st, // yields no value, cloned in a later pass
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

    for (rpo) |cb| try bmap.put(allocator, cb, try caller.appendBlock());
    for (rpo) |cb| { // params of every cloned block (entry params get the call args below)
        const nb = bmap.get(cb).?;
        for (callee.blockParams(@enumFromInt(cb))) |p| {
            const np = try caller.appendBlockParam(nb, try mapType(caller, callee, &tmap, callee.valueType(p)));
            try vmap.put(allocator, p, np);
        }
    }
    for (rpo) |cb| { // instructions, in RPO so operands are already mapped. `if` is handled below
        const nb = bmap.get(cb).?;
        for (callee.blockInsts(@enumFromInt(cb))) |cinst| switch (callee.opcode(cinst)) {
            .@"if" => {},
            .store => |st| try caller.appendStore(nb, mapV(vmap, st.value), mapV(vmap, st.ptr)),
            else => |op| {
                const cres = callee.instResult(cinst).?;
                const rty = try mapType(caller, callee, &tmap, callee.valueType(cres));
                const nres = try caller.appendInst(nb, rty, try mapOpcode(caller, callee, vmap, &tmap, op));
                try vmap.put(allocator, cres, nres);
            },
        };
    }
    for (rpo) |cb| { // control flow: `if` exits and jump/ret terminators
        const nb = bmap.get(cb).?;
        const cblock: Block = @enumFromInt(cb);
        var if_cf: ?ir.function.If = null;
        for (callee.blockInsts(cblock)) |cinst| {
            if (callee.opcode(cinst) == .@"if") {
                if_cf = callee.opcode(cinst).@"if";
                break;
            }
        }
        if (if_cf) |cf| {
            const ta = try remapArgs(allocator, callee, vmap, cf.then.args);
            defer allocator.free(ta);
            const ea = try remapArgs(allocator, callee, vmap, cf.@"else".args);
            defer allocator.free(ea);
            try caller.appendIf(nb, mapV(vmap, cf.cond), .{ .target = bmap.get(@intFromEnum(cf.then.target)).?, .args = ta }, .{ .target = bmap.get(@intFromEnum(cf.@"else".target)).?, .args = ea });
            continue;
        }
        if (callee.terminator(cblock)) |t| switch (t) {
            .ret => |v| {
                if (v) |rv| try caller.setJump(nb, cont, &.{mapV(vmap, rv)}) else try caller.setJump(nb, cont, &.{});
            },
            .jump => |j| {
                const ja = try remapArgs(allocator, callee, vmap, j.args);
                defer allocator.free(ja);
                try caller.setJump(nb, bmap.get(@intFromEnum(j.target)).?, ja);
            },
        };
    }

    // Enter the inlined body from the (now truncated) caller block, passing the call arguments.
    try caller.setJump(b_block, bmap.get(0).?, args);
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
        callee.setTerminator(b, .{ .ret = sum });
    }

    // caller: f(x) = madd(x, x) + 1
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(i32k);
    const b = try caller.appendBlock();
    const x = try caller.appendBlockParam(b, t);
    const call = try caller.appendCall(b, t, "madd", &.{ x, x });
    const r = try caller.appendArithImm(b, t, .add, call, 1);
    caller.setTerminator(b, .{ .ret = r });

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
        callee.setTerminator(tb, .{ .ret = tv });
        callee.setTerminator(eb, .{ .ret = ev });
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
    caller.setTerminator(cb, .{ .ret = m }); // the add stays separate, which is fine for a structural check

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
