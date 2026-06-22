//! Inter-procedural function inlining. Replaces a `call` to a small callee with a
//! clone of its body: callee parameters become the call's arguments, its
//! instructions are spliced into the caller at the call site, and its returned
//! value replaces the call's result.
//!
//! Inlines only single-block, leaf, scalar callees: one block ending in `ret`, no
//! calls of its own (no recursion), every instruction produces a result (no
//! stores/`if`), and scalar types only (trivial type remapping). Multi-block
//! callees need block splitting and are not handled.

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
            if (!inlinable(callee)) continue;
            try inlineCall(allocator, caller, @intCast(bi), idx, inst, callee);
            return true;
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
