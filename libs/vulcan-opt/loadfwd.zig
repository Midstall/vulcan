//! Redundant-load elimination: within a block, forward the value of a store to a later load of the
//! same address, and reuse an earlier load of an address for a later one, when no aliasing store (or
//! call) sits between them. This is the in-memory analogue of mem2reg: mem2reg promotes the slots
//! whose address never escapes, this handles the memory that stays in memory (escaped locals, heap,
//! incoming pointers). A light alias oracle keeps it cheap and correct: two distinct allocas never
//! alias, two distinct globals never alias, an alloca and a global never alias, and anything else is
//! conservatively assumed to alias. A call or a matmul may touch any memory, so it clears everything.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "loadfwd", .run = run };

/// One address with an available value: `ptr` currently holds `value` (from a store or a prior
/// load), valid until an aliasing store or a barrier invalidates it.
const Avail = struct { ptr: Value, value: Value };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    var changed = false;
    var avail: std.ArrayList(Avail) = .empty;
    defer avail.deinit(allocator);
    var keep: std.ArrayList(Inst) = .empty;
    defer keep.deinit(allocator);

    for (0..func.blockCount()) |bi| {
        avail.clearRetainingCapacity(); // availability does not cross block boundaries here
        keep.clearRetainingCapacity();
        var block_changed = false;
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .load => |ld| {
                    const result = func.instResult(inst).?;
                    if (find(avail.items, func, ld.ptr)) |v| {
                        func.replaceAllUses(result, v);
                        block_changed = true;
                        continue; // drop the now-redundant load
                    }
                    try avail.append(allocator, .{ .ptr = ld.ptr, .value = result });
                },
                .store => |st| {
                    invalidateAliasing(&avail, func, st.ptr);
                    try avail.append(allocator, .{ .ptr = st.ptr, .value = st.value });
                },
                // A call or matmul may write any memory; a prefetch is a pure hint. Everything else is
                // pure (no memory effect) and leaves availability intact.
                .call, .call_indirect, .matmul => avail.clearRetainingCapacity(),
                else => {},
            }
            try keep.append(allocator, inst);
        }
        if (block_changed) {
            try func.setBlockInsts(@enumFromInt(bi), keep.items);
            changed = true;
        }
    }
    return changed;
}

/// The available value for `ptr`, if some entry holds exactly that address.
fn find(items: []const Avail, func: *const Function, ptr: Value) ?Value {
    for (items) |a| {
        if (sameAddress(func, a.ptr, ptr)) return a.value;
    }
    return null;
}

/// Drop every availability entry whose address may alias `ptr` (a store to `ptr` may have
/// overwritten it). Entries at provably-distinct addresses survive.
fn invalidateAliasing(avail: *std.ArrayList(Avail), func: *const Function, ptr: Value) void {
    var i: usize = 0;
    while (i < avail.items.len) {
        if (mayAlias(func, avail.items[i].ptr, ptr)) {
            _ = avail.swapRemove(i);
        } else i += 1;
    }
}

fn sameAddress(func: *const Function, a: Value, b: Value) bool {
    if (a == b) return true;
    const ba = baseOf(func, a);
    const bb = baseOf(func, b);
    return ba.offset == bb.offset and std.meta.eql(ba.root, bb.root);
}

const Root = union(enum) { alloca: Inst, global: u32, unknown: Value };
const Addr = struct { root: Root, offset: i64 };

/// Resolve `v` to a base address plus a constant offset, following `arith_imm` add/sub chains to an
/// `alloca` or `global_addr` root when possible. An unknown root carries the value itself, so two
/// unknown roots are equal only when they are the same SSA value.
fn baseOf(func: *const Function, v: Value) Addr {
    var cur = v;
    var offset: i64 = 0;
    // Bounded by the def chain length; each step moves to a strictly earlier-defined value.
    var steps: usize = 0;
    while (steps < 64) : (steps += 1) {
        const inst = func.definingInst(cur) orelse return .{ .root = .{ .unknown = cur }, .offset = offset };
        switch (func.opcode(inst)) {
            .alloca => return .{ .root = .{ .alloca = inst }, .offset = offset },
            .global_addr => |g| return .{ .root = .{ .global = g.symbol }, .offset = offset },
            .arith_imm => |a| switch (a.op) {
                .add => {
                    offset += a.imm;
                    cur = a.lhs;
                },
                .sub => {
                    offset -= a.imm;
                    cur = a.lhs;
                },
                else => return .{ .root = .{ .unknown = cur }, .offset = offset },
            },
            else => return .{ .root = .{ .unknown = cur }, .offset = offset },
        }
    }
    return .{ .root = .{ .unknown = cur }, .offset = offset };
}

/// Whether a store to `b` may alias the address `a`. Distinct identified roots (two allocas, two
/// globals, an alloca vs a global) never alias; everything else is assumed to alias.
fn mayAlias(func: *const Function, a: Value, b: Value) bool {
    if (a == b) return true;
    const ra = baseOf(func, a).root;
    const rb = baseOf(func, b).root;
    return switch (ra) {
        .alloca => |ia| switch (rb) {
            .alloca => |ib| ia == ib, // distinct allocas never alias
            .global => false,
            .unknown => true,
        },
        .global => |ga| switch (rb) {
            .global => |gb| ga == gb, // distinct globals never alias
            .alloca => false,
            .unknown => true,
        },
        .unknown => true, // an unknown base could be anything
    };
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn i32Ty(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "a store forwards its value to a later load of the same address" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t); // an incoming pointer (mem2reg won't touch it)
    const v = try func.appendBlockParam(b, t);
    try func.appendStore(b, v, p);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(v, func.terminator(b).?.ret.?); // load forwarded the stored value
}

test "a second load of an address reuses the first" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const y1 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const y2 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = y1, .rhs = y2 } });
    func.setTerminator(b, .{ .ret = sum });

    try testing.expect(try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(y1, add.lhs);
    try testing.expectEqual(y1, add.rhs); // the second load reused the first
}

test "a store to a distinct alloca does not kill an available load" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    const a = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    const c = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    const first = try func.appendInst(b, t, .{ .load = .{ .ptr = c } });
    try func.appendStore(b, v, a); // store to a distinct alloca
    const second = try func.appendInst(b, t, .{ .load = .{ .ptr = c } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(b, .{ .ret = sum });

    try testing.expect(try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(first, add.rhs); // the store to `a` did not kill the load of `c`
}

test "a store to a possibly-aliasing pointer forces a reload" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const q = try func.appendBlockParam(b, ptr_t); // unknown base: may alias p
    const v = try func.appendBlockParam(b, t);
    const first = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b, v, q);
    const second = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(b, .{ .ret = sum });

    try testing.expect(!try runOnce(allocator, &func)); // second load cannot be forwarded
}
