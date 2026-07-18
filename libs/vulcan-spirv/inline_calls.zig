//! Function inlining for SPIR-V modules.
//!
//! Vulkan shaders may split work across helper `OpFunction`s called via `OpFunctionCall`
//! (e.g. vkcube's fragment shader calls `linearToSrgb`). The rest of the lowering models a
//! single entry function, so this pass rewrites a multi-function module into an equivalent
//! single-function module with no `OpFunctionCall`: every call is replaced by an inlined,
//! id-renamed copy of the callee's body spliced into the caller's control-flow graph.
//!
//! Vulkan shaders are non-recursive by specification, so inlining always terminates. A
//! cycle (which a malformed module could contain) is detected and rejected.
//!
//! The pass works on the raw SPIR-V word stream (before the IR lowering) so the existing
//! lowering then sees one ordinary function. glslang passes arguments by pointer: it stores
//! into a `Function`-storage `OpVariable` and passes that pointer. The callee's
//! `OpFunctionParameter` is itself a pointer. Binding therefore aliases the parameter id to
//! the argument id (both pointers). The callee's local `OpVariable`s are hoisted into the
//! caller's entry block. The resulting function still contains `Function`-storage variables,
//! which the SSA-construction (mem2reg) pre-pass promotes, exactly as for the entry
//! function's own locals.
//!
//! All intermediate state is arena-allocated, only the returned word stream uses the
//! caller's allocator, so operand-slice ownership needs no manual tracking.

const std = @import("std");
const binary = @import("binary.zig");
const op = @import("opcodes.zig");

pub const Error = binary.Error || std.mem.Allocator.Error || error{ Unsupported, MalformedModule, RecursionDetected };

/// Operand `i` of a decoded instruction, or `error.MalformedModule` if the instruction is too
/// short. `binary.Reader` only guarantees `word_count >= 1`, so a truncated instruction from the
/// untrusted word stream carries fewer operands than its opcode needs; indexing past them would
/// read out of bounds. `inlineCalls` is a public entry point over raw `[]const u32`.
fn operandAt(operands: []const u32, i: usize) Error!u32 {
    if (i >= operands.len) return error.MalformedModule;
    return operands[i];
}

/// One instruction inside a function body: opcode + operand words (arena-owned once cloned).
const Inst = struct {
    opcode: u16,
    operands: []const u32,
};

/// A basic block: its label id and its instructions, the last of which is the terminator.
const Blk = struct {
    label: u32,
    insts: std.ArrayListUnmanaged(Inst) = .empty,
};

/// One decoded function: result id, parameter ids, local `OpVariable` instructions (hoisted
/// on inline), and the body blocks.
const Func = struct {
    id: u32,
    decl: []const u32, // the OpFunction operand words ([resultType, result, control, type])
    params: std.ArrayListUnmanaged(Inst) = .empty, // OpFunctionParameter instructions
    locals: std.ArrayListUnmanaged(Inst) = .empty, // OpVariable (Function-storage) instructions
    blocks: std.ArrayListUnmanaged(Blk) = .empty,
};

/// Inline every `OpFunctionCall` in the module's entry (first) function to a fixpoint and
/// drop the other functions, returning a fresh, self-contained SPIR-V word stream. When the
/// module has a single function and no calls, returns a verbatim copy. Caller owns the
/// result and must free it.
pub fn inlineCalls(gpa: std.mem.Allocator, words: []const u32) Error![]u32 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var r = try binary.Reader.init(words);
    const header = r.header;

    var header_insts: std.ArrayListUnmanaged(Inst) = .empty;
    var funcs: std.ArrayListUnmanaged(Func) = .empty;

    var cur_func: ?*Func = null;
    var cur_block: ?*Blk = null;
    var n_calls: usize = 0;

    while (try r.next()) |inst| {
        const operands = inst.operands; // borrowed from `words`, safe until return
        switch (inst.opcode) {
            op.Function => {
                // [resultType, result, control, type]; the result id is used to index callees.
                try funcs.append(a, .{ .id = try operandAt(operands, 1), .decl = operands });
                cur_func = &funcs.items[funcs.items.len - 1];
                cur_block = null;
            },
            op.FunctionParameter => {
                if (operands.len < 2) return error.MalformedModule; // [type, result]; result read later
                try (cur_func orelse return error.MalformedModule).params.append(a, .{ .opcode = inst.opcode, .operands = operands });
            },
            op.FunctionEnd => {
                cur_func = null;
                cur_block = null;
            },
            op.Label => {
                const f = cur_func orelse return error.MalformedModule;
                try f.blocks.append(a, .{ .label = try operandAt(operands, 0) });
                cur_block = &f.blocks.items[f.blocks.items.len - 1];
            },
            op.Variable => {
                if (cur_func != null and operands.len >= 3 and operands[2] == op.StorageClass.function) {
                    try cur_func.?.locals.append(a, .{ .opcode = inst.opcode, .operands = operands });
                } else if (cur_block) |b| {
                    try b.insts.append(a, .{ .opcode = inst.opcode, .operands = operands });
                } else if (cur_func == null) {
                    try header_insts.append(a, .{ .opcode = inst.opcode, .operands = operands });
                } else return error.MalformedModule;
            },
            op.FunctionCall => {
                n_calls += 1;
                const b = cur_block orelse return error.MalformedModule;
                try b.insts.append(a, .{ .opcode = inst.opcode, .operands = operands });
            },
            else => {
                if (cur_block) |b| {
                    try b.insts.append(a, .{ .opcode = inst.opcode, .operands = operands });
                } else if (cur_func == null) {
                    try header_insts.append(a, .{ .opcode = inst.opcode, .operands = operands });
                } else return error.MalformedModule; // inside a function before its first label
            },
        }
    }

    if (funcs.items.len == 0) return error.MalformedModule;

    if (funcs.items.len == 1 and n_calls == 0) {
        return gpa.dupe(u32, words);
    }

    // callee function id -> its Func.
    var by_id = std.AutoHashMapUnmanaged(u32, *Func).empty;
    for (funcs.items[1..]) |*f| try by_id.put(a, f.id, f);

    const entry = &funcs.items[0];

    // Fresh ids start past the module's id bound.
    var next_id: u32 = header.id_bound;

    // Inlining state. A callee's returns are merged into the call result via an `OpPhi` at the
    // continuation block, carrying each `OpReturnValue` from its source block. To keep those
    // phi predecessor labels valid, every callee is FULLY inlined (made call-free) BEFORE it is
    // cloned into a caller, so a clone's blocks are never re-split by a later nested inline. The
    // recursion (callee-before-caller) terminates because Vulkan shaders are non-recursive. A
    // cycle is detected via the `inlining` in-progress set.
    var state = std.AutoHashMapUnmanaged(u32, FuncState).empty;
    var ctx: InlineCtx = .{ .a = a, .by_id = &by_id, .next_id = &next_id, .state = &state };

    // Fully inline the entry function (recursively inlining its callees first).
    try ensureInlined(&ctx, entry);

    // Ids defined by the now-removed (non-entry) functions: their function/param/local/result
    // ids. Header `OpName`/`OpMemberName`/`OpDecorate` referencing these would dangle, so they
    // are dropped to keep the output a self-contained, valid module.
    var removed = std.AutoHashMapUnmanaged(u32, void).empty;
    for (funcs.items[1..]) |*f| {
        try removed.put(a, f.id, {});
        for (f.params.items) |p| try removed.put(a, p.operands[1], {});
        for (f.locals.items) |lv| try removed.put(a, lv.operands[1], {});
        for (f.blocks.items) |blk| {
            try removed.put(a, blk.label, {});
            for (blk.insts.items) |inst| {
                if (resultIdOf(inst)) |rid| try removed.put(a, rid, {});
            }
        }
    }

    // Re-serialize with the caller's allocator: header, then the single inlined function.
    var out: std.ArrayListUnmanaged(u32) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, &.{ binary.magic, header.version, header.generator, next_id, 0 });
    for (header_insts.items) |inst| {
        if (referencesRemoved(inst, &removed)) continue;
        try emit(gpa, &out, inst.opcode, inst.operands);
    }

    try emit(gpa, &out, op.Function, entry.decl);
    for (entry.params.items) |p| try emit(gpa, &out, op.FunctionParameter, p.operands);
    if (entry.blocks.items.len == 0) return error.MalformedModule;
    for (entry.blocks.items, 0..) |blk, bi| {
        try emit(gpa, &out, op.Label, &.{blk.label});
        if (bi == 0) {
            // Hoist every local variable into the entry block, right after its label.
            for (entry.locals.items) |lv| try emit(gpa, &out, lv.opcode, lv.operands);
        }
        for (blk.insts.items) |inst| try emit(gpa, &out, inst.opcode, inst.operands);
    }
    try emit(gpa, &out, op.FunctionEnd, &.{});

    return out.toOwnedSlice(gpa);
}

/// Whether a header debug/decoration instruction targets a removed callee id (so it should
/// be dropped). `OpName`/`OpMemberName`/`OpDecorate`/`OpMemberDecorate` carry their target id
/// in operand 0. Other header instructions are kept.
fn referencesRemoved(inst: Inst, removed: *const std.AutoHashMapUnmanaged(u32, void)) bool {
    return switch (inst.opcode) {
        op.Name, op.MemberName, op.Decorate, op.MemberDecorate => inst.operands.len >= 1 and removed.contains(inst.operands[0]),
        else => false,
    };
}

/// Per-function inlining progress (cycle detection + memoize "already fully inlined").
const FuncState = enum { pending, in_progress, done };

/// Shared inlining context (the allocator, callee table, id allocator, per-function state).
const InlineCtx = struct {
    a: std.mem.Allocator,
    by_id: *const std.AutoHashMapUnmanaged(u32, *Func),
    next_id: *u32,
    state: *std.AutoHashMapUnmanaged(u32, FuncState),

    fn allocId(self: InlineCtx) u32 {
        const id = self.next_id.*;
        self.next_id.* += 1;
        return id;
    }
};

/// Fully inline `func` so its body contains no `OpFunctionCall`. Each callee is recursively
/// fully inlined FIRST (so its blocks are call-free and will not be re-split when cloned),
/// then spliced in. A function reached while already `in_progress` is a recursion cycle.
fn ensureInlined(ctx: *InlineCtx, func: *Func) Error!void {
    switch (ctx.state.get(func.id) orelse .pending) {
        .done => return,
        .in_progress => return error.RecursionDetected,
        .pending => {},
    }
    try ctx.state.put(ctx.a, func.id, .in_progress);

    // Inline calls until none remain. Because every callee is fully inlined before cloning,
    // the cloned (call-free) blocks are never re-split, so one expansion pass per remaining
    // call suffices. The loop just re-scans after each splice. The bound guards against a
    // logic error (real cycles are caught by the in_progress state above).
    var guard: usize = 0;
    while (true) {
        guard += 1;
        if (guard > 100000) return error.RecursionDetected;
        const expanded = try inlineFirstCall(ctx, func);
        if (!expanded) break;
    }

    try ctx.state.put(ctx.a, func.id, .done);
}

/// Inline the first `OpFunctionCall` found in `func` (recursively ensuring the callee is fully
/// inlined first), rewriting `func`'s blocks. Returns true if a call was inlined, false if the
/// function is already call-free.
fn inlineFirstCall(ctx: *InlineCtx, func: *Func) Error!bool {
    const a = ctx.a;

    // Locate the first call: (block index, instruction index).
    var found: ?struct { bi: usize, ci: usize } = null;
    outer: for (func.blocks.items, 0..) |blk, bi| {
        for (blk.insts.items, 0..) |inst, ci| {
            if (inst.opcode == op.FunctionCall) {
                found = .{ .bi = bi, .ci = ci };
                break :outer;
            }
        }
    }
    const loc = found orelse return false;

    const blk = func.blocks.items[loc.bi];
    const call = blk.insts.items[loc.ci];
    // [resultType, result, callee, arg0, ...]
    if (call.operands.len < 3) return error.MalformedModule; // need at least [type, result, callee]
    const ret_id = call.operands[1];
    const callee_id = call.operands[2];
    const args = call.operands[3..];
    const callee = ctx.by_id.get(callee_id) orelse return error.MalformedModule;
    if (callee.params.items.len != args.len) return error.MalformedModule;

    // Fully inline the callee first (callee-before-caller), so its blocks are call-free.
    try ensureInlined(ctx, callee);

    const after_label = ctx.allocId();

    // Clone the callee. Param ids alias the call args. Each return becomes a phi edge into the
    // continuation. Since the callee is call-free, these cloned blocks are never re-split.
    const clone = try cloneCallee(ctx, callee, args, after_label);

    var new_blocks: std.ArrayListUnmanaged(Blk) = .empty;
    // Blocks before the call's block: keep verbatim.
    for (func.blocks.items[0..loc.bi]) |b| try new_blocks.append(a, b);

    // "before" block: instructions preceding the call, then a branch to the clone entry.
    var before: Blk = .{ .label = blk.label };
    try before.insts.appendSlice(a, blk.insts.items[0..loc.ci]);
    try before.insts.append(a, .{ .opcode = op.Branch, .operands = try a.dupe(u32, &.{clone.entry_label}) });
    try new_blocks.append(a, before);

    // Hoist the cloned locals onto the function (its entry block).
    for (clone.locals.items) |lv| try func.locals.append(a, lv);

    // The cloned callee blocks.
    for (clone.blocks.items) |cb| try new_blocks.append(a, cb);

    // "after" block: the return-value phi (if any), then the continuation instructions.
    var after: Blk = .{ .label = after_label };
    if (clone.return_edges.items.len > 0 and !clone.returns_void) {
        var phi_ops: std.ArrayListUnmanaged(u32) = .empty;
        try phi_ops.append(a, call.operands[0]); // result type
        try phi_ops.append(a, ret_id);
        for (clone.return_edges.items) |e| {
            try phi_ops.append(a, e.value);
            try phi_ops.append(a, e.pred_label);
        }
        try after.insts.append(a, .{ .opcode = op.Phi, .operands = phi_ops.items });
    }
    try after.insts.appendSlice(a, blk.insts.items[loc.ci + 1 ..]);
    try new_blocks.append(a, after);

    // Blocks after the call's block: keep verbatim.
    for (func.blocks.items[loc.bi + 1 ..]) |b| try new_blocks.append(a, b);

    func.blocks = new_blocks;
    return true;
}

/// One return edge from a cloned callee: the (remapped) returned value and the (remapped)
/// label of the block it returned from. Because the callee is fully inlined before cloning,
/// this label is the actual predecessor of the continuation in the final CFG.
const RetEdge = struct { value: u32, pred_label: u32 };

const Clone = struct {
    entry_label: u32,
    blocks: std.ArrayListUnmanaged(Blk) = .empty,
    locals: std.ArrayListUnmanaged(Inst) = .empty,
    return_edges: std.ArrayListUnmanaged(RetEdge) = .empty,
    returns_void: bool = false,
};

/// Clone a (call-free) callee's body with fresh result/label ids. Parameter ids are remapped
/// to the caller's argument ids. `OpReturnValue %v` becomes `OpBranch %cont_label`, recording
/// `(v, source block label)` as a phi edge merged at the continuation. `OpReturn` records a
/// void edge. `OpUnreachable`/`OpKill` are kept (no continuation edge). Locals are remapped and
/// collected for hoisting.
fn cloneCallee(ctx: *InlineCtx, callee: *const Func, args: []const u32, cont_label: u32) Error!Clone {
    const a = ctx.a;
    var clone: Clone = .{ .entry_label = 0 };

    if (callee.blocks.items.len == 0) return error.MalformedModule; // a callee with no OpLabel
    var remap = std.AutoHashMapUnmanaged(u32, u32).empty;
    for (callee.params.items, 0..) |p, i| try remap.put(a, p.operands[1], args[i]);
    for (callee.locals.items) |lv| try remap.put(a, lv.operands[1], ctx.allocId());
    for (callee.blocks.items) |blk| {
        try remap.put(a, blk.label, ctx.allocId());
        for (blk.insts.items) |inst| {
            if (resultIdOf(inst)) |rid| {
                if (!remap.contains(rid)) try remap.put(a, rid, ctx.allocId());
            }
        }
    }

    clone.entry_label = remap.get(callee.blocks.items[0].label).?;

    for (callee.locals.items) |lv| {
        const new_ops = try remapOperands(a, lv.opcode, lv.operands, &remap);
        try clone.locals.append(a, .{ .opcode = lv.opcode, .operands = new_ops });
    }

    for (callee.blocks.items) |blk| {
        const nb_label = remap.get(blk.label).?;
        var nb: Blk = .{ .label = nb_label };
        for (blk.insts.items) |inst| {
            switch (inst.opcode) {
                op.Return => {
                    clone.returns_void = true;
                    try clone.return_edges.append(a, .{ .value = 0, .pred_label = nb_label });
                    try nb.insts.append(a, .{ .opcode = op.Branch, .operands = try a.dupe(u32, &.{cont_label}) });
                },
                op.ReturnValue => {
                    const rv = try operandAt(inst.operands, 0); // [value]
                    const v = remap.get(rv) orelse rv;
                    try clone.return_edges.append(a, .{ .value = v, .pred_label = nb_label });
                    try nb.insts.append(a, .{ .opcode = op.Branch, .operands = try a.dupe(u32, &.{cont_label}) });
                },
                op.Unreachable => {
                    // The callee's structured MERGE block after both `if` arms already
                    // returned (glslang emits `%merge = OpLabel / OpUnreachable`). Once the
                    // returns above became branches to the continuation, this block has no
                    // predecessors - it is dead. But a downstream `OpUnreachable` lowers to a
                    // function RETURN, which on Volta+ becomes an EXIT (warp terminate). An
                    // EXIT physically laid out INSIDE a divergent convergence-barrier region
                    // (between this inlined if's BSSY and BSYNC) corrupts the warp's 2x2-quad
                    // state, breaking the TEX / derivative SHFL that depend on quad uniformity
                    // (the cause of vkcube's noisy / collapsed face). Route it to the
                    // continuation like a return instead, so the whole shader has exactly ONE
                    // EXIT (the real epilogue) and the warp reconverges cleanly. Add a phi
                    // edge so the continuation phi stays well-formed. The value is never
                    // selected (the block is unreachable) so any in-scope value (the first
                    // real return value, or 0 for void) is a safe placeholder.
                    const placeholder: u32 = if (clone.return_edges.items.len > 0) clone.return_edges.items[0].value else 0;
                    try clone.return_edges.append(a, .{ .value = placeholder, .pred_label = nb_label });
                    try nb.insts.append(a, .{ .opcode = op.Branch, .operands = try a.dupe(u32, &.{cont_label}) });
                },
                op.Kill => {
                    // A real fragment discard: keep it. It must NOT branch to the
                    // continuation (it terminates the invocation, no continuation edge).
                    try nb.insts.append(a, .{ .opcode = inst.opcode, .operands = try a.dupe(u32, inst.operands) });
                },
                else => {
                    const new_ops = try remapOperands(a, inst.opcode, inst.operands, &remap);
                    try nb.insts.append(a, .{ .opcode = inst.opcode, .operands = new_ops });
                },
            }
        }
        try clone.blocks.append(a, nb);
    }

    return clone;
}

/// A fresh operand slice with id-reference operands remapped through `remap` (literals left
/// as-is). Caller owns the slice.
fn remapOperands(a: std.mem.Allocator, opcode: u16, operands: []const u32, remap: *const std.AutoHashMapUnmanaged(u32, u32)) Error![]u32 {
    const out = try a.alloc(u32, operands.len);
    for (operands, 0..) |w, i| {
        out[i] = if (isIdOperand(opcode, i)) (remap.get(w) orelse w) else w;
    }
    return out;
}

/// Whether operand index `i` of `opcode` is an id reference (vs a literal). Only the opcodes
/// that appear in shader function bodies are classified precisely. An unknown opcode treats
/// ALL operands as ids (the conservative default for the data/control instructions in the
/// shapes we accept, none of which mix unclassified literals).
fn isIdOperand(opcode: u16, i: usize) bool {
    return switch (opcode) {
        op.Constant => i < 2, // [type, result, literal bits...]
        op.ConstantComposite, op.ConstantTrue, op.ConstantFalse => true,
        op.ExtInst => i != 3, // [type, result, set, instruction(literal), args...]
        op.CompositeExtract => i < 3, // [type, result, composite, literal indices...]
        op.VectorShuffle => i < 4, // [type, result, v1, v2, literal components...]
        op.Store => i < 2, // [ptr, value, (MemoryOperand literal)?]
        op.Load => i < 3, // [type, result, ptr, (MemoryOperand literal)?]
        op.SelectionMerge => i < 1, // [mergeBlock(id), control(literal)]
        op.LoopMerge => i < 2, // [merge, continue, control(literal)...]
        else => true,
    };
}

/// The result id an instruction defines, if any.
fn resultIdOf(inst: Inst) ?u32 {
    return switch (inst.opcode) {
        op.Store,
        op.Branch,
        op.BranchConditional,
        op.Return,
        op.ReturnValue,
        op.Unreachable,
        op.Kill,
        op.SelectionMerge,
        op.LoopMerge,
        op.Name,
        op.MemberName,
        op.Decorate,
        op.MemberDecorate,
        op.Nop,
        => null,
        else => if (inst.operands.len >= 2) inst.operands[1] else null,
    };
}

fn emit(gpa: std.mem.Allocator, out: *std.ArrayListUnmanaged(u32), opcode: u16, operands: []const u32) Error!void {
    const word_count: u32 = @intCast(1 + operands.len);
    try out.append(gpa, (word_count << 16) | opcode);
    try out.appendSlice(gpa, operands);
}

// Tests
const testing = std.testing;

fn disassembleCallCount(words: []const u32) !usize {
    var r = try binary.Reader.init(words);
    var n: usize = 0;
    while (try r.next()) |inst| {
        if (inst.opcode == op.FunctionCall) n += 1;
    }
    return n;
}

fn functionCount(words: []const u32) !usize {
    var r = try binary.Reader.init(words);
    var n: usize = 0;
    while (try r.next()) |inst| {
        if (inst.opcode == op.Function) n += 1;
    }
    return n;
}

test "single function, no calls: verbatim copy" {
    const a = testing.allocator;
    var b = try binary.Builder.init(a, 9);
    defer b.deinit(a);
    try b.emit(a, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(a, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(a, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(a, op.Label, &.{6});
    try b.emit(a, op.ReturnValue, &.{1});
    try b.emit(a, op.FunctionEnd, &.{});

    const out = try inlineCalls(a, b.words.items);
    defer a.free(out);
    try testing.expectEqualSlices(u32, b.words.items, out);
}

test "inlines a single-block helper that doubles its argument" {
    const a = testing.allocator;
    // float dbl(float x) { return x * 2.0 }  (by-value param, not glslang's pointer form)
    // void main-ish entry: %r = dbl(%c3), return %r.
    // ids: f32=1 fnFF=2 fnF=3 c2=4 c3=5 dbl=6 dx=7 dblEntry=8 mul=9 main=10 mEntry=11 r=12
    var b = try binary.Builder.init(a, 13);
    defer b.deinit(a);
    try b.emit(a, op.TypeFloat, &.{ 1, 32 });
    try b.emit(a, op.TypeFunction, &.{ 2, 1, 1 }); // float(float)
    try b.emit(a, op.TypeFunction, &.{ 3, 1 }); // float()
    try b.emit(a, op.Constant, &.{ 1, 4, 0x40000000 }); // 2.0
    try b.emit(a, op.Constant, &.{ 1, 5, 0x40400000 }); // 3.0
    // entry function (no params) first.
    try b.emit(a, op.Function, &.{ 1, 10, 0, 3 });
    try b.emit(a, op.Label, &.{11});
    try b.emit(a, op.FunctionCall, &.{ 1, 12, 6, 5 }); // r = dbl(3.0)
    try b.emit(a, op.ReturnValue, &.{12});
    try b.emit(a, op.FunctionEnd, &.{});
    // dbl helper.
    try b.emit(a, op.Function, &.{ 1, 6, 0, 2 });
    try b.emit(a, op.FunctionParameter, &.{ 1, 7 });
    try b.emit(a, op.Label, &.{8});
    try b.emit(a, op.FMul, &.{ 1, 9, 7, 4 }); // x * 2.0
    try b.emit(a, op.ReturnValue, &.{9});
    try b.emit(a, op.FunctionEnd, &.{});

    const out = try inlineCalls(a, b.words.items);
    defer a.free(out);

    try testing.expectEqual(@as(usize, 0), try disassembleCallCount(out));
    try testing.expectEqual(@as(usize, 1), try functionCount(out));

    // The inlined FMul must use the call argument (id 5 = 3.0) as the multiplicand: the
    // parameter id 7 was aliased to the argument 5.
    var r = try binary.Reader.init(out);
    var found_mul = false;
    while (try r.next()) |inst| {
        if (inst.opcode == op.FMul) {
            found_mul = true;
            try testing.expectEqual(@as(u32, 5), inst.operands[2]); // x -> arg 5
            try testing.expectEqual(@as(u32, 4), inst.operands[3]); // 2.0 constant kept
        }
    }
    try testing.expect(found_mul);
}

test "inlines a multi-block helper with a branch (clamp-ish select)" {
    const a = testing.allocator;
    // float pick(bool c) { if (c) return 1.0 else return 0.0 }  entry: r = pick(true).
    // ids: bool=1 f32=2 fnFb=3 fnF=4 c1=5 c0=6 t=7 pick=8 cp=9 e=10 then=11 els=12
    //      merge=13 main=14 me=15 r=16
    var b = try binary.Builder.init(a, 17);
    defer b.deinit(a);
    try b.emit(a, op.TypeBool, &.{1});
    try b.emit(a, op.TypeFloat, &.{ 2, 32 });
    try b.emit(a, op.TypeFunction, &.{ 3, 2, 1 }); // float(bool)
    try b.emit(a, op.TypeFunction, &.{ 4, 2 }); // float()
    try b.emit(a, op.Constant, &.{ 2, 5, 0x3f800000 }); // 1.0
    try b.emit(a, op.Constant, &.{ 2, 6, 0 }); // 0.0
    try b.emit(a, op.ConstantTrue, &.{ 1, 7 });
    // entry
    try b.emit(a, op.Function, &.{ 2, 14, 0, 4 });
    try b.emit(a, op.Label, &.{15});
    try b.emit(a, op.FunctionCall, &.{ 2, 16, 8, 7 }); // r = pick(true)
    try b.emit(a, op.ReturnValue, &.{16});
    try b.emit(a, op.FunctionEnd, &.{});
    // pick
    try b.emit(a, op.Function, &.{ 2, 8, 0, 3 });
    try b.emit(a, op.FunctionParameter, &.{ 1, 9 }); // c
    try b.emit(a, op.Label, &.{10});
    try b.emit(a, op.SelectionMerge, &.{ 13, 0 });
    try b.emit(a, op.BranchConditional, &.{ 9, 11, 12 });
    try b.emit(a, op.Label, &.{11});
    try b.emit(a, op.ReturnValue, &.{5}); // return 1.0
    try b.emit(a, op.Label, &.{12});
    try b.emit(a, op.ReturnValue, &.{6}); // return 0.0
    try b.emit(a, op.Label, &.{13});
    try b.emit(a, op.Unreachable, &.{});
    try b.emit(a, op.FunctionEnd, &.{});

    const out = try inlineCalls(a, b.words.items);
    defer a.free(out);

    try testing.expectEqual(@as(usize, 0), try disassembleCallCount(out));
    try testing.expectEqual(@as(usize, 1), try functionCount(out));

    // The continuation block carries an OpPhi merging the returns. The two real return
    // edges are 1.0 (id 5, then) and 0.0 (id 6, else). The structured merge block's
    // OpUnreachable is now ALSO routed to the continuation (so the whole shader has one
    // EXIT and the warp reconverges - the vkcube quad-corruption fix), contributing a
    // THIRD phi edge whose value is a never-selected placeholder (the first return value).
    // So the phi has 3 edges = 8 operands. The BranchConditional is preserved.
    var r = try binary.Reader.init(out);
    var found_phi = false;
    var found_bc = false;
    var n_unreachable: usize = 0;
    while (try r.next()) |inst| {
        if (inst.opcode == op.Phi) {
            found_phi = true;
            try testing.expectEqual(@as(usize, 8), inst.operands.len); // [type, result, v0,p0, v1,p1, v2,p2]
            // The two real returns (1.0=id5, 0.0=id6) must both appear among the values.
            // The third (placeholder) reuses one of them. So {5,6} is a subset of the values.
            const vals = [_]u32{ inst.operands[2], inst.operands[4], inst.operands[6] };
            var has5 = false;
            var has6 = false;
            for (vals) |v| {
                if (v == 5) has5 = true;
                if (v == 6) has6 = true;
            }
            try testing.expect(has5 and has6);
        }
        if (inst.opcode == op.BranchConditional) found_bc = true;
        if (inst.opcode == op.Unreachable) n_unreachable += 1;
    }
    try testing.expect(found_phi);
    try testing.expect(found_bc);
    // The OpUnreachable was rewritten to a branch (routed to the continuation), so NONE
    // survive in the inlined output - no warp-killing terminator inside the if region.
    try testing.expectEqual(@as(usize, 0), n_unreachable);
}

test "inlines nested calls to a fixpoint" {
    const a = testing.allocator;
    // float inner(float x){return x+1} float outer(float x){return inner(x)+1}
    // entry: r = outer(c). Two levels of call.
    // ids: f32=1 fnFF=2 fnF=3 c1=4 cIn=5
    //   inner=6 ip=7 ie=8 iadd=9
    //   outer=10 op_=11 oe=12 ocall=13 oadd=14
    //   main=15 me=16 r=17
    var b = try binary.Builder.init(a, 18);
    defer b.deinit(a);
    try b.emit(a, op.TypeFloat, &.{ 1, 32 });
    try b.emit(a, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(a, op.TypeFunction, &.{ 3, 1 });
    try b.emit(a, op.Constant, &.{ 1, 4, 0x3f800000 }); // 1.0
    try b.emit(a, op.Constant, &.{ 1, 5, 0x40a00000 }); // 5.0
    // entry
    try b.emit(a, op.Function, &.{ 1, 15, 0, 3 });
    try b.emit(a, op.Label, &.{16});
    try b.emit(a, op.FunctionCall, &.{ 1, 17, 10, 5 }); // r = outer(5.0)
    try b.emit(a, op.ReturnValue, &.{17});
    try b.emit(a, op.FunctionEnd, &.{});
    // inner
    try b.emit(a, op.Function, &.{ 1, 6, 0, 2 });
    try b.emit(a, op.FunctionParameter, &.{ 1, 7 });
    try b.emit(a, op.Label, &.{8});
    try b.emit(a, op.FAdd, &.{ 1, 9, 7, 4 });
    try b.emit(a, op.ReturnValue, &.{9});
    try b.emit(a, op.FunctionEnd, &.{});
    // outer
    try b.emit(a, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(a, op.FunctionParameter, &.{ 1, 11 });
    try b.emit(a, op.Label, &.{12});
    try b.emit(a, op.FunctionCall, &.{ 1, 13, 6, 11 }); // inner(x)
    try b.emit(a, op.FAdd, &.{ 1, 14, 13, 4 }); // + 1
    try b.emit(a, op.ReturnValue, &.{14});
    try b.emit(a, op.FunctionEnd, &.{});

    const out = try inlineCalls(a, b.words.items);
    defer a.free(out);

    try testing.expectEqual(@as(usize, 0), try disassembleCallCount(out));
    try testing.expectEqual(@as(usize, 1), try functionCount(out));
    // Two FAdds survive (inner's +1 and outer's +1).
    var r = try binary.Reader.init(out);
    var n_add: usize = 0;
    while (try r.next()) |inst| {
        if (inst.opcode == op.FAdd) n_add += 1;
    }
    try testing.expectEqual(@as(usize, 2), n_add);
}

test "rejects a recursive call cycle" {
    const a = testing.allocator;
    // entry calls f, f calls f (illegal in Vulkan, but assert we don't loop forever).
    // ids: void=1 fnV=2 f=3 fe=4 main=5 me=6 t=7
    var b = try binary.Builder.init(a, 8);
    defer b.deinit(a);
    try b.emit(a, op.TypeVoid, &.{1});
    try b.emit(a, op.TypeFunction, &.{ 2, 1 });
    // entry
    try b.emit(a, op.Function, &.{ 1, 5, 0, 2 });
    try b.emit(a, op.Label, &.{6});
    try b.emit(a, op.FunctionCall, &.{ 1, 7, 3 }); // f()
    try b.emit(a, op.Return, &.{});
    try b.emit(a, op.FunctionEnd, &.{});
    // f calls itself
    try b.emit(a, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(a, op.Label, &.{4});
    try b.emit(a, op.FunctionCall, &.{ 1, 9, 3 }); // f() -> recursion
    try b.emit(a, op.Return, &.{});
    try b.emit(a, op.FunctionEnd, &.{});

    try testing.expectError(error.RecursionDetected, inlineCalls(a, b.words.items));
}

test "rejects a truncated OpFunction (missing result id)" {
    const a = testing.allocator;
    var b = try binary.Builder.init(a, 4);
    defer b.deinit(a);
    // OpFunction is [resultType, result, control, type]; this one omits the result id (and
    // more), so reading operand 1 (used as the callee id) would slice past the instruction.
    try b.emit(a, op.Function, &.{1});
    try testing.expectError(error.MalformedModule, inlineCalls(a, b.words.items));
}
