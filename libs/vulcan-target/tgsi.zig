//! Lowers a graphics Vulcan IR function to TGSI text.
//!
//! TGSI (Tungsten Graphics Shader Infrastructure) is Gallium/virgl's textual
//! shader form: `tgsi_text_translate` parses exactly this text, so the emitted
//! bytes feed straight to a virgl shader-create command. GPU-paravirtual
//! counterpart of the nvidia SASS backend: rather than selecting a native ISA, the
//! graphics IR is rendered as TGSI declarations and opcodes.
//!
//! The graphics IR carries the same `vulcan.gpu` attribute tags the SPIR-V
//! graphics lowering produces and the nvidia isel reads:
//!   * entry-block params tagged `attr` = ATTR_GENERIC0 + loc*0x10 + comp*4
//!     (a vertex/fragment input attribute slot), one per vector component.
//!   * output stores whose pointer is a tag-carrier iconst tagged either
//!     `out_attr` (a vertex output: ATTR_POSITION for the clip-space position,
//!     ATTR_GENERIC0 + loc*0x10 + comp*4 for a varying) or `color_out` (a
//!     fragment render-target color component index 0..3).
//!
//! TGSI mapping:
//!   VS inputs   -> `DCL IN[loc]`
//!   VS position -> `DCL OUT[n], POSITION`
//!   VS varying  -> `DCL OUT[n], GENERIC[loc]`
//!   FS input    -> `DCL IN[n], GENERIC[loc], PERSPECTIVE`
//!   FS color    -> `DCL OUT[0], COLOR`
//! plus one `MOV` per output register copying its source input register, then
//! `END`.
//!
//! Only the passthrough class (output <- input, per register) is lowered. A shader
//! with arithmetic in the body returns `error.Unsupported`. General TGSI expression
//! lowering is future work.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// Attribute slot bases (shared with the nvidia encoder's interface convention).
const ATTR_POSITION: u32 = 0x70;
const ATTR_GENERIC0: u32 = 0x80;

/// The shader stage. Selected from the function's `vulcan.gpu` "stage" tag the
/// SPIR-V lowering sets (vertex / fragment). Compute has no TGSI form here.
pub const Stage = enum { vertex, fragment };

/// A decoded graphics input attribute: which IN register (the location) and which
/// of its components (x=0..w=3) a scalar param feeds.
const InputSlot = struct { reg: u32, comp: u8 };

/// A decoded graphics output: a TGSI OUT register plus its semantic.
const OutKind = enum { position, generic, color, psize, fragdepth };
const OutputSlot = struct { kind: OutKind, location: u32, comp: u8 };

/// The attribute slot the SPIR-V lowering routes gl_PointSize to (a scalar, below
/// the clip-space position slot).
const ATTR_POINT_SIZE: u32 = 0x6c;

/// The four TGSI swizzle/writemask channel letters, by component index.
const channel = [4]u8{ 'x', 'y', 'z', 'w' };

/// Where a scalar SSA value lives in the emitted TGSI. Every IR value the backend
/// materializes resolves to one of these, so an operand renders to a TGSI source.
const Src = union(enum) {
    input: InputSlot, // an IN[reg] component (a vertex attribute / interpolated varying)
    imm: u32, // the .x of a declared IMM[k]
    temp: u32, // the .x of a declared TEMP[t] (a computed scalar)
    /// One component of a uniform-block member: CONST[unit][idx].comp (std140).
    constbuf: struct { unit: u32, idx: u32, comp: u8 },
    /// One component of a texture-sample result held in TEMP[temp].comp.
    texcomp: struct { temp: u32, comp: u8 },
    /// A declared system value SV[idx].x (a vertex/instance id / face builtin input).
    sysval: u32,
    /// A component of gl_FragCoord, read from the POSITION-semantic input register.
    fragcoord: u8,
};

/// The reserved TGSI IN register for gl_FragCoord (POSITION semantic). Placed high so
/// it never collides with a Location-based generic varying.
const FRAG_COORD_REG: u32 = 16;

/// A TGSI system-value semantic a builtin input declares as.
const Sysval = enum {
    vertexid,
    instanceid,
    face,
    fn name(self: Sysval) []const u8 {
        return switch (self) {
            .vertexid => "VERTEXID",
            .instanceid => "INSTANCEID",
            .face => "FACE",
        };
    }
};

/// A declared immediate: the 32 raw bits of the scalar plus whether it prints as a
/// float (FLT32) or an integer (INT32).
const ImmVal = struct { bits: u32, is_float: bool };

/// Accumulated state for one output register: which scalar source feeds each of its
/// written components, plus its semantic (position / varying / color).
const OutReg = struct {
    present: bool = false,
    kind: OutKind = .generic,
    location: u32 = 0,
    comp_src: [4]?Src = .{ null, null, null, null },
    mask: u4 = 0,
};

/// Render a scalar `Src` as a TGSI source operand (a single-channel read).
fn writeSrc(w: *Writer, src: Src) Error!void {
    switch (src) {
        .input => |s| try w.print("IN[{d}].{c}", .{ s.reg, channel[s.comp] }),
        .imm => |k| try w.print("IMM[{d}].x", .{k}),
        .temp => |t| try w.print("TEMP[{d}].x", .{t}),
        .constbuf => |c| try w.print("CONST[{d}][{d}].{c}", .{ c.unit, c.idx, channel[c.comp] }),
        .texcomp => |c| try w.print("TEMP[{d}].{c}", .{ c.temp, channel[c.comp] }),
        .sysval => |idx| try w.print("SV[{d}].x", .{idx}),
        .fragcoord => |comp| try w.print("IN[{d}].{c}", .{ FRAG_COORD_REG, channel[comp] }),
    }
}

/// The TGSI opcode mnemonic for a binary op, chosen by operand domain. Integer
/// arithmetic uses the U*/I* integer ops (index math must not run through the
/// float ALU). Floats use the plain ops. Float subtraction reuses ADD with a
/// negated second source (handled by the caller). Integer subtraction is handled
/// separately (TGSI has no integer source-negate). `signed` refines div/shift.
fn binOpMnemonic(op: ir.function.BinOp, is_int: bool, signed: bool) Error![]const u8 {
    return switch (op) {
        .add, .sub => if (is_int) "UADD" else "ADD",
        .mul => if (is_int) "UMUL" else "MUL",
        .div => if (is_int) (if (signed) "IDIV" else "UDIV") else "DIV",
        .bit_and => "AND",
        .bit_or => "OR",
        .bit_xor => "XOR",
        .shl => "SHL",
        .shr => if (signed) "ISHR" else "USHR",
        .rem => error.Unsupported, // handled separately (float sequence / MOD / UMOD)
    };
}

/// The TGSI opcode mnemonic for a unary op on the result (float) type.
fn unaryMnemonic(op: ir.function.UnaryOp) []const u8 {
    return switch (op) {
        .reinterpret => "MOV", // bit-preserving move (TGSI temps are typeless bits)
        .sqrt => "SQRT",
        .ceil => "CEIL",
        .floor => "FLR",
        .trunc => "TRUNC",
        .nearest => "ROUND",
    };
}

/// Whether block `blk` contains a discard (a call_indirect to a discard_fn param).
/// Marks the dead-end arm of an `if (cond) discard;` (whose kill block ends the
/// invocation, so the if's post-dominator is the function exit, not a merge block).
fn blockHasDiscard(func: *const Function, blk: Block) bool {
    for (func.blockInsts(blk)) |inst| {
        if (func.opcode(inst) == .call_indirect) {
            const c = func.opcode(inst).call_indirect;
            if (hasFlag(func, c.target, "discard_fn")) return true;
        }
    }
    return false;
}

/// The `@"if"` instruction ending block `blk`, if any (the conditional branch the
/// SPIR-V lowering emits for OpBranchConditional). A block has at most one.
fn ifOf(func: *const Function, blk: Block) ?ir.function.If {
    for (func.blockInsts(blk)) |inst| {
        if (func.opcode(inst) == .@"if") return func.opcode(inst).@"if";
    }
    return null;
}

/// The successor block indices of `blk`: the two `@"if"` arms, or the terminator's
/// jump target, or none (a `ret`, which flows to the virtual exit node `n`).
fn blockSuccIdx(func: *const Function, blk: Block, n: usize, buf: *[2]usize) []const usize {
    if (ifOf(func, blk)) |c| {
        buf[0] = @intFromEnum(c.then.target);
        buf[1] = @intFromEnum(c.@"else".target);
        return buf[0..2];
    }
    switch (func.terminator(blk) orelse ir.function.Terminator{ .ret = null }) {
        .ret => {
            buf[0] = n; // the virtual exit
            return buf[0..1];
        },
        .jump => |j| {
            buf[0] = @intFromEnum(j.target);
            return buf[0..1];
        },
    }
}

/// Compute each block's immediate post-dominator (an index in [0, n], where n is a
/// virtual exit node all `ret`s flow to). Uses iterative set post-dominance over
/// u64 bitsets, so it supports up to 63 blocks (ample for a shader). ipdom[i] is
/// the reconvergence block of a divergent `if` at block i. Returns Unsupported for
/// a function too large to bitset.
fn computeIpdom(allocator: std.mem.Allocator, func: *const Function) Error![]usize {
    const n = func.blockCount();
    if (n >= 63) return error.Unsupported;
    const total = n + 1; // + the virtual exit node at index n
    const exit = n;

    var pdom = try allocator.alloc(u64, total);
    defer allocator.free(pdom);
    const all: u64 = if (total >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(total)) - 1;
    for (0..total) |i| pdom[i] = all;
    pdom[exit] = @as(u64, 1) << @intCast(exit); // the exit post-dominates only itself

    // Iterate to a fixpoint: pdom[i] = {i} ∪ (∩ over successors s of pdom[s]).
    var changed = true;
    while (changed) {
        changed = false;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var buf: [2]usize = undefined;
            const succs = blockSuccIdx(func, @enumFromInt(i), n, &buf);
            var inter: u64 = all;
            for (succs) |s| inter &= pdom[s];
            const next = (@as(u64, 1) << @intCast(i)) | inter;
            if (next != pdom[i]) {
                pdom[i] = next;
                changed = true;
            }
        }
    }

    // ipdom[i] = the strict post-dominator with the largest pdom set (the closest one
    // is post-dominated by all the farther ones, so its set is the biggest).
    const ipdom = try allocator.alloc(usize, n);
    errdefer allocator.free(ipdom);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var best: usize = exit;
        var best_pop: usize = 0;
        var d: usize = 0;
        while (d < total) : (d += 1) {
            if (d == i) continue;
            if (pdom[i] & (@as(u64, 1) << @intCast(d)) == 0) continue; // d must post-dominate i
            const pop = @popCount(pdom[d]);
            if (pop > best_pop) {
                best_pop = pop;
                best = d;
            }
        }
        ipdom[i] = best;
    }
    return ipdom;
}

/// One emitted step: an IR instruction to lower, or a structured-control marker the
/// planner interleaves between blocks. The emitter runs the list linearly, so the
/// control-flow recursion lives only in the planner, not the per-instruction switch.
const Step = union(enum) {
    inst: ir.function.Inst,
    begin_if: Value, // UIF on the condition
    else_,
    end_if,
    begin_loop, // BGNLOOP
    end_loop, // ENDLOOP
    brk, // BRK (break out of the enclosing loop)
    /// Write a branch's phi arguments into the target block's parameter temps.
    phi: struct { target: Block, args: []const Value },
};

/// Compute which blocks are loop headers: the target of a back edge (an edge u->v
/// where v dominates u). Dominators are found by the same iterative bitset dataflow
/// (forward CFG, entry = block 0 as root).
fn computeLoopHeaders(allocator: std.mem.Allocator, func: *const Function) Error![]bool {
    const n = func.blockCount();
    if (n >= 63) return error.Unsupported;
    var dom = try allocator.alloc(u64, n);
    defer allocator.free(dom);
    const all: u64 = if (n >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1;
    dom[0] = 1; // the entry dominates only itself among its own dominators
    for (1..n) |i| dom[i] = all;

    var changed = true;
    while (changed) {
        changed = false;
        var i: usize = 1;
        while (i < n) : (i += 1) {
            var inter: u64 = all;
            var u: usize = 0;
            while (u < n) : (u += 1) {
                var buf: [2]usize = undefined;
                const succs = blockSuccIdx(func, @enumFromInt(u), n, &buf);
                for (succs) |s| {
                    if (s == i) inter &= dom[u];
                }
            }
            const next = (@as(u64, 1) << @intCast(i)) | inter;
            if (next != dom[i]) {
                dom[i] = next;
                changed = true;
            }
        }
    }

    const headers = try allocator.alloc(bool, n);
    errdefer allocator.free(headers);
    @memset(headers, false);
    var u: usize = 0;
    while (u < n) : (u += 1) {
        var buf: [2]usize = undefined;
        const succs = blockSuccIdx(func, @enumFromInt(u), n, &buf);
        for (succs) |s| {
            // v = s dominates u  ==>  u->s is a back edge and s is a loop header.
            if (s < n and (dom[u] & (@as(u64, 1) << @intCast(s))) != 0) headers[s] = true;
        }
    }
    return headers;
}

/// Structurizes a CFG into IF/ELSE/ENDIF + BGNLOOP/BRK/ENDLOOP steps, resolving phis
/// to the target blocks' parameter temps on each edge.
const Planner = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    ipdom: []const usize,
    headers: []const bool,
    steps: *std.ArrayListUnmanaged(Step),

    /// A loop being emitted: reaching `header` is a back edge (iteration end). Reaching
    /// `merge` (the loop exit) is a break.
    const Loop = struct { header: usize, merge: usize };

    /// Emit the phi write for an edge to `target`, then classify where control goes:
    /// `.done` if the edge terminated this region (a back edge, a break, or the stop
    /// block), else `.cont` to keep planning at `target`.
    fn edge(self: Planner, target: Block, args: []const Value, stop: usize, loop: ?Loop) Error!enum { done, cont } {
        try self.steps.append(self.allocator, .{ .phi = .{ .target = target, .args = args } });
        const ti = @intFromEnum(target);
        if (loop) |lp| {
            if (ti == lp.header) return .done; // back edge: ENDLOOP jumps back to the top
            if (ti == lp.merge) {
                try self.steps.append(self.allocator, .brk); // loop exit: break out
                return .done;
            }
        }
        if (ti == stop) return .done;
        return .cont;
    }

    fn region(self: Planner, cur: usize, stop: usize, loop: ?Loop) Error!void {
        const n = self.func.blockCount();
        var c = cur;
        var guard: usize = 0;
        while (c != stop and c != n) {
            guard += 1;
            if (guard > 2 * n + 2) return error.Unsupported; // irreducible / unexpected shape

            // A loop header (that isn't the one we're currently emitting) starts a loop.
            if (self.headers[c] and !(loop != null and loop.?.header == c)) {
                const merge = self.ipdom[c];
                if (merge == c or merge == n) return error.Unsupported;
                try self.steps.append(self.allocator, .begin_loop);
                try self.region(c, merge, .{ .header = c, .merge = merge });
                try self.steps.append(self.allocator, .end_loop);
                c = merge;
                continue;
            }

            const blk: Block = @enumFromInt(c);
            for (self.func.blockInsts(blk)) |inst| try self.steps.append(self.allocator, .{ .inst = inst });

            if (ifOf(self.func, blk)) |ifc| {
                const t_i = @intFromEnum(ifc.then.target);
                const e_i = @intFromEnum(ifc.@"else".target);
                if (t_i == e_i) {
                    if (try self.edge(ifc.then.target, self.func.valueList(ifc.then.args), stop, loop) == .done) return;
                    c = t_i;
                    continue;
                }
                const merge = self.ipdom[c];
                // An early-exit `if (cond) discard;`: the kill arm ends the invocation, so
                // the if has no shared merge (its post-dominator is the function exit). Emit
                // the discard arm inside the UIF and continue at the other (the real merge).
                if (merge == n and stop != n) {
                    const then_dead = blockHasDiscard(self.func, ifc.then.target);
                    const else_dead = blockHasDiscard(self.func, ifc.@"else".target);
                    if (then_dead == else_dead) return error.Unsupported; // not a single-arm discard
                    try self.steps.append(self.allocator, .{ .begin_if = ifc.cond });
                    const cont = if (then_dead) blk: {
                        try self.arm(ifc.then.target, self.func.valueList(ifc.then.args), n, loop);
                        break :blk ifc.@"else";
                    } else blk: {
                        try self.steps.append(self.allocator, .else_);
                        try self.arm(ifc.@"else".target, self.func.valueList(ifc.@"else".args), n, loop);
                        break :blk ifc.then;
                    };
                    try self.steps.append(self.allocator, .end_if);
                    if (try self.edge(cont.target, self.func.valueList(cont.args), stop, loop) == .done) return;
                    c = @intFromEnum(cont.target);
                    continue;
                }
                if (merge == c or (merge == n and stop != n)) return error.Unsupported;
                try self.steps.append(self.allocator, .{ .begin_if = ifc.cond });
                try self.arm(ifc.then.target, self.func.valueList(ifc.then.args), merge, loop);
                try self.steps.append(self.allocator, .else_);
                try self.arm(ifc.@"else".target, self.func.valueList(ifc.@"else".args), merge, loop);
                try self.steps.append(self.allocator, .end_if);
                c = merge;
            } else switch (self.func.terminator(blk) orelse ir.function.Terminator{ .ret = null }) {
                .ret => return,
                .jump => |j| {
                    if (try self.edge(j.target, self.func.valueList(j.args), stop, loop) == .done) return;
                    c = @intFromEnum(j.target);
                },
            }
        }
    }

    /// Plan one arm of an if: write its edge phis, then plan its region up to the if's
    /// `merge` (unless the edge is a back edge / break / straight-to-merge).
    fn arm(self: Planner, target: Block, args: []const Value, merge: usize, loop: ?Loop) Error!void {
        if (try self.edge(target, args, merge, loop) == .done) return;
        try self.region(@intFromEnum(target), merge, loop);
    }
};

/// Lower a graphics IR `func` to TGSI text. The text is NUL-terminated and
/// dword-padded (virglrenderer reads it as a token-aligned shader blob). The
/// caller owns the returned slice (free with `allocator`).
pub fn lower(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    const stage = stageOf(func) orelse return error.Unsupported;

    const entry: Block = @enumFromInt(0);
    const params = func.blockParams(entry);

    // Classify params. `attr` -> an IN[reg].comp input, `binding` (without a
    // `sampler_desc`) -> a UBO base pointer read as CONST[unit] (unit = the binding).
    // `sampler_fn`/`sampler_desc`/`builtin` params are not lowered yet (Unsupported).
    var src_of = std.AutoHashMapUnmanaged(Value, Src){};
    defer src_of.deinit(allocator);
    // A UBO base pointer's accumulated address: which CONST unit + a byte offset. The
    // load path traces `arith add base, off` chains through this map.
    // A UBO address: the CONST unit, the accumulated CONSTANT byte offset, and an
    // optional DYNAMIC byte-offset term (e.g. gl_InstanceIndex*stride) that lowers
    // to a relative CONST[unit][ADDR[0].x + idx] read.
    const Addr = struct { unit: u32, off: u32, dyn: ?Value = null };
    var addr_of = std.AutoHashMapUnmanaged(Value, Addr){};
    defer addr_of.deinit(allocator);
    var addr_used = false; // whether any relative CONST read declared DCL ADDR[0]
    var const_dynamic = [_]bool{false} ** 16; // CONST unit -> read with a dynamic index
    // Derivative (dFdx/dFdy/fwidth) gradient buffer: a load from grad_base + i*4 is a
    // DDX/DDY of the varying named by the i-th `grad_slot` func attr. grad_addr tracks
    // the byte offset from the base param.
    var grad_addr = std.AutoHashMapUnmanaged(Value, u32){};
    defer grad_addr.deinit(allocator);
    var grad_slots: std.ArrayListUnmanaged(u32) = .empty; // index -> packed (slot<<1 | axis_y)
    defer grad_slots.deinit(allocator);
    {
        var it = func.attributesOf(.func);
        while (it.next()) |attr| switch (attr) {
            .custom => |cu| if (std.mem.eql(u8, cu.namespace, "vulcan.gpu") and std.mem.eql(u8, cu.key, "grad_slot")) {
                switch (cu.value) {
                    .int => |v| try grad_slots.append(allocator, @intCast(v)),
                    else => {},
                }
            },
            else => {},
        };
    }
    // Sampler descriptor param -> its SAMP unit (the Vulkan binding).
    var samp_of = std.AutoHashMapUnmanaged(Value, u32){};
    defer samp_of.deinit(allocator);
    var in_present = [_]bool{false} ** 32;
    var const_used = [_]bool{false} ** 16; // CONST unit -> referenced
    var const_max_idx = [_]u32{0} ** 16; // CONST unit -> highest vec4 index read
    var samp_used = [_]bool{false} ** 16; // SAMP unit -> a TEX referenced it
    var sv_sem = [_]Sysval{.vertexid} ** 4; // SV index -> its semantic
    var sv_count: u32 = 0; // number of declared system values
    var fc_used = false; // gl_FragCoord (a POSITION input) is read
    for (params) |p| {
        if (attrTag(func, p, "attr")) |slot| {
            if (slot < ATTR_GENERIC0) return error.Unsupported; // only generic inputs
            const off = slot - ATTR_GENERIC0;
            const reg = off / 0x10;
            const comp: u8 = @intCast((off % 0x10) / 4);
            if (reg >= in_present.len) return error.Unsupported;
            in_present[reg] = true;
            try src_of.put(allocator, p, .{ .input = .{ .reg = reg, .comp = comp } });
        } else if (attrTag(func, p, "builtin")) |bi| {
            if (bi == 15) {
                // gl_FragCoord: a component of the POSITION-semantic input register.
                const comp = attrTag(func, p, "bicomp") orelse 0;
                fc_used = true;
                try src_of.put(allocator, p, .{ .fragcoord = @intCast(comp) });
            } else {
                // gl_VertexIndex (42) / gl_InstanceIndex (43) / gl_FrontFacing (17): a
                // system value input read as SV[idx].x.
                const sem: Sysval = switch (bi) {
                    42 => .vertexid,
                    43 => .instanceid,
                    17 => .face,
                    else => return error.Unsupported,
                };
                if (sv_count >= sv_sem.len) return error.Unsupported;
                const idx = sv_count;
                sv_sem[idx] = sem;
                sv_count += 1;
                try src_of.put(allocator, p, .{ .sysval = idx });
            }
        } else if (hasFlag(func, p, "sampler_fn") or hasFlag(func, p, "discard_fn")) {
            // The host-sampler / discard function pointers: a TGSI target ignores them
            // (TEX / KILL need no host function). Nothing to declare or bind.
        } else if (attrTag(func, p, "sampler_desc") != null) {
            const bind = attrTag(func, p, "binding") orelse 0;
            if (bind >= samp_used.len) return error.Unsupported;
            try samp_of.put(allocator, p, bind);
        } else if (hasFlag(func, p, "grad_buf")) {
            // The derivative gradient buffer base: a load from it is a DDX/DDY of the
            // varying named by the corresponding grad_slot func attr.
            try grad_addr.put(allocator, p, 0);
        } else if (attrTag(func, p, "binding")) |bind| {
            if (bind >= const_used.len) return error.Unsupported;
            try addr_of.put(allocator, p, .{ .unit = bind, .off = 0 });
        } else return error.Unsupported; // an untagged / builtin param
    }

    // The computed body is emitted as we walk (each arith/unary/convert result is a
    // fresh TEMP), and output stores are collected into out_regs to emit as MOVs
    // after the compute instructions. Constants are materialized lazily so pointer-
    // offset iconsts (consumed only by address math) never occupy an IMM slot.
    var body = Writer{ .allocator = allocator };
    defer body.list.deinit(allocator);
    var imms: std.ArrayListUnmanaged(ImmVal) = .empty;
    defer imms.deinit(allocator);
    var pending = std.AutoHashMapUnmanaged(Value, ImmVal){}; // constant value -> its bits, not yet in IMM[]
    defer pending.deinit(allocator);
    var int_of = std.AutoHashMapUnmanaged(Value, i64){}; // integer-constant value -> its value (for address offsets)
    defer int_of.deinit(allocator);
    // Texture-sample result tracking: the out-pointer alloca (and its `+c*4` element
    // pointers) resolve to a component of a TEX result held in one TEMP.
    const TexElem = struct { temp: u32, comp: u8 };
    var tex_addr = std.AutoHashMapUnmanaged(Value, TexElem){};
    defer tex_addr.deinit(allocator);
    var out_regs = [_]OutReg{.{}} ** 32;
    var temp_count: u32 = 0;
    var line: u32 = 1;

    // Resolve an operand to its scalar Src, materializing a pending constant into an
    // IMM[] slot on first real use.
    const Resolver = struct {
        allocator: std.mem.Allocator,
        src_of: *std.AutoHashMapUnmanaged(Value, Src),
        pending: *std.AutoHashMapUnmanaged(Value, ImmVal),
        imms: *std.ArrayListUnmanaged(ImmVal),
        fn get(self: @This(), v: Value) Error!Src {
            if (self.src_of.get(v)) |s| return s;
            if (self.pending.get(v)) |im| {
                const idx: u32 = @intCast(self.imms.items.len);
                try self.imms.append(self.allocator, im);
                try self.src_of.put(self.allocator, v, .{ .imm = idx });
                return .{ .imm = idx };
            }
            return error.Unsupported;
        }
    };
    const rz = Resolver{ .allocator = allocator, .src_of = &src_of, .pending = &pending, .imms = &imms };

    // Build the emission plan. A single-block function is a straight run of its
    // instructions. A multi-block function is structurized (IF/ELSE/ENDIF) by the
    // post-dominator planner, and each non-entry block param (a phi) gets a TEMP the
    // branch edges write. `steps` interleaves instructions with control markers so
    // the emitter below stays a single linear pass.
    var steps: std.ArrayListUnmanaged(Step) = .empty;
    defer steps.deinit(allocator);
    const nblocks = func.blockCount();
    if (nblocks == 1) {
        for (func.blockInsts(entry)) |inst| try steps.append(allocator, .{ .inst = inst });
    } else {
        var bi: usize = 1;
        while (bi < nblocks) : (bi += 1) {
            for (func.blockParams(@enumFromInt(bi))) |p| {
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, p, .{ .temp = t });
            }
        }
        const ipdom = try computeIpdom(allocator, func);
        defer allocator.free(ipdom);
        const headers = try computeLoopHeaders(allocator, func);
        defer allocator.free(headers);
        const planner = Planner{ .allocator = allocator, .func = func, .ipdom = ipdom, .headers = headers, .steps = &steps };
        try planner.region(0, nblocks, null);
    }

    for (steps.items) |step| {
        const inst = switch (step) {
            .inst => |i| i,
            .begin_if => |cond| {
                try body.print("  {d}: UIF ", .{line});
                try writeSrc(&body, try rz.get(cond));
                try body.put("\n");
                line += 1;
                continue;
            },
            .else_ => {
                try body.print("  {d}: ELSE\n", .{line});
                line += 1;
                continue;
            },
            .end_if => {
                try body.print("  {d}: ENDIF\n", .{line});
                line += 1;
                continue;
            },
            .begin_loop => {
                try body.print("  {d}: BGNLOOP\n", .{line});
                line += 1;
                continue;
            },
            .end_loop => {
                try body.print("  {d}: ENDLOOP\n", .{line});
                line += 1;
                continue;
            },
            .brk => {
                try body.print("  {d}: BRK\n", .{line});
                line += 1;
                continue;
            },
            .phi => |pw| {
                // Write each of the target block's parameter temps from the edge's
                // matching phi argument (skip a self-copy where arg == the param temp).
                const tparams = func.blockParams(pw.target);
                if (tparams.len != pw.args.len) return error.Unsupported;
                for (tparams, pw.args) |param, arg| {
                    const dst = src_of.get(param) orelse return error.Unsupported;
                    const dt = switch (dst) {
                        .temp => |t| t,
                        else => return error.Unsupported,
                    };
                    const asrc = try rz.get(arg);
                    if (asrc == .temp and asrc.temp == dt) continue;
                    try body.print("  {d}: MOV TEMP[{d}].x, ", .{ line, dt });
                    try writeSrc(&body, asrc);
                    try body.put("\n");
                    line += 1;
                }
                continue;
            },
        };
        const result = func.instResult(inst);
        switch (func.opcode(inst)) {
            .iconst => |n| {
                // An out-slot tag carrier is never read. A real integer constant is
                // recorded (its value, for address math) and pended for lazy IMM use.
                if (result) |r| {
                    const is_out_tag = attrTag(func, r, "out_attr") != null or attrTag(func, r, "color_out") != null;
                    if (!is_out_tag) {
                        try int_of.put(allocator, r, n);
                        try pending.put(allocator, r, .{ .bits = @bitCast(@as(i32, @truncate(n))), .is_float = false });
                    }
                }
            },
            .fconst => |x| {
                const r = result orelse continue;
                try pending.put(allocator, r, .{ .bits = @bitCast(@as(f32, @floatCast(x))), .is_float = true });
            },
            .arith => |a| {
                const r = result orelse return error.Unsupported;
                // A pointer-typed result is address arithmetic: fold the constant byte
                // offset into the running address (a UBO base -> CONST, or a texture
                // out-pointer alloca -> a TEX result component) instead of emitting an op.
                if (func.types.type_kind(func.valueType(r)) == .ptr) {
                    if (int_of.get(a.rhs)) |add| {
                        // A constant byte offset: fold into the running address.
                        if (addr_of.get(a.lhs)) |base| {
                            try addr_of.put(allocator, r, .{ .unit = base.unit, .off = base.off +% @as(u32, @intCast(add)), .dyn = base.dyn });
                        } else if (tex_addr.get(a.lhs)) |base| {
                            try tex_addr.put(allocator, r, .{ .temp = base.temp, .comp = @intCast(@as(u32, base.comp) + @as(u32, @intCast(add)) / 4) });
                        } else if (grad_addr.get(a.lhs)) |base_off| {
                            try grad_addr.put(allocator, r, base_off +% @as(u32, @intCast(add)));
                        } else return error.Unsupported;
                    } else if (addr_of.get(a.lhs)) |base| {
                        // A dynamic byte offset (e.g. gl_InstanceIndex*stride): record it
                        // for a relative CONST read. Only one dynamic term is supported.
                        if (base.dyn != null) return error.Unsupported;
                        try addr_of.put(allocator, r, .{ .unit = base.unit, .off = base.off, .dyn = a.rhs });
                    } else return error.Unsupported;
                    continue;
                }
                const lhs = try rz.get(a.lhs);
                const rhs = try rz.get(a.rhs);
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, r, .{ .temp = t });
                if (a.op == .rem) {
                    // TGSI has no float remainder op. For a float result emit the
                    // trunc-based sequence (FRem / C fmod): t = x - y*trunc(x/y). An
                    // integer result uses MOD (signed) / UMOD (unsigned) directly.
                    const rk = func.types.type_kind(func.valueType(r));
                    if (rk == .float) {
                        const s = temp_count;
                        temp_count += 1;
                        try body.print("  {d}: DIV TEMP[{d}].x, ", .{ line, s });
                        try writeSrc(&body, lhs);
                        try body.put(", ");
                        try writeSrc(&body, rhs);
                        try body.put("\n");
                        line += 1;
                        try body.print("  {d}: TRUNC TEMP[{d}].x, TEMP[{d}].x\n", .{ line, s, s });
                        line += 1;
                        try body.print("  {d}: MUL TEMP[{d}].x, TEMP[{d}].x, ", .{ line, s, s });
                        try writeSrc(&body, rhs);
                        try body.put("\n");
                        line += 1;
                        try body.print("  {d}: ADD TEMP[{d}].x, ", .{ line, t });
                        try writeSrc(&body, lhs);
                        try body.print(", -TEMP[{d}].x\n", .{s});
                        line += 1;
                    } else {
                        const signed = rk == .int and rk.int.signedness == .signed;
                        try body.print("  {d}: {s} TEMP[{d}].x, ", .{ line, if (signed) "MOD" else "UMOD", t });
                        try writeSrc(&body, lhs);
                        try body.put(", ");
                        try writeSrc(&body, rhs);
                        try body.put("\n");
                        line += 1;
                    }
                    continue;
                }
                const rk2 = func.types.type_kind(func.valueType(r));
                const is_int = rk2 == .int;
                const signed = is_int and rk2.int.signedness == .signed;
                if (a.op == .sub and is_int) {
                    // Integer subtract: TGSI has no integer source-negate, so negate
                    // the rhs (INEG) into a scratch temp, then UADD.
                    const s = temp_count;
                    temp_count += 1;
                    try body.print("  {d}: INEG TEMP[{d}].x, ", .{ line, s });
                    try writeSrc(&body, rhs);
                    try body.put("\n");
                    line += 1;
                    try body.print("  {d}: UADD TEMP[{d}].x, ", .{ line, t });
                    try writeSrc(&body, lhs);
                    try body.print(", TEMP[{d}].x\n", .{s});
                    line += 1;
                } else {
                    const mnem = try binOpMnemonic(a.op, is_int, signed);
                    try body.print("  {d}: {s} TEMP[{d}].x, ", .{ line, mnem, t });
                    try writeSrc(&body, lhs);
                    try body.put(", ");
                    if (a.op == .sub) try body.put("-"); // float sub via negated source
                    try writeSrc(&body, rhs);
                    try body.put("\n");
                    line += 1;
                }
            },
            .arith_imm => |a| {
                // Arithmetic against a constant (integer immediate op). Subtraction
                // folds into an add against the negated immediate (no source negate).
                const r = result orelse return error.Unsupported;
                const rk = func.types.type_kind(func.valueType(r));
                const is_int = rk == .int;
                const signed = is_int and rk.int.signedness == .signed;
                const use_op: ir.function.BinOp = if (a.op == .sub) .add else a.op;
                const eff_imm: i64 = if (a.op == .sub) -a.imm else a.imm;
                const mnem = try binOpMnemonic(use_op, is_int, signed);
                const lhs = try rz.get(a.lhs);
                const imm_idx: u32 = @intCast(imms.items.len);
                const imm_val: ImmVal = if (is_int)
                    .{ .bits = @bitCast(@as(i32, @truncate(eff_imm))), .is_float = false }
                else
                    .{ .bits = @bitCast(@as(f32, @floatFromInt(eff_imm))), .is_float = true };
                try imms.append(allocator, imm_val);
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, r, .{ .temp = t });
                try body.print("  {d}: {s} TEMP[{d}].x, ", .{ line, mnem, t });
                try writeSrc(&body, lhs);
                try body.put(", ");
                try writeSrc(&body, .{ .imm = imm_idx });
                try body.put("\n");
                line += 1;
            },
            .unary => |u| {
                const r = result orelse return error.Unsupported;
                const val = try rz.get(u.value);
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, r, .{ .temp = t });
                try body.print("  {d}: {s} TEMP[{d}].x, ", .{ line, unaryMnemonic(u.op), t });
                try writeSrc(&body, val);
                try body.put("\n");
                line += 1;
            },
            .convert => |c| {
                const r = result orelse return error.Unsupported;
                const val = try rz.get(c.value);
                // Direction from the operand type vs the result type (int<->float).
                const src_ty = func.types.type_kind(func.valueType(c.value));
                const dst_ty = func.types.type_kind(func.valueType(r));
                const mnem = convertMnemonic(src_ty, dst_ty) orelse return error.Unsupported;
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, r, .{ .temp = t });
                try body.print("  {d}: {s} TEMP[{d}].x, ", .{ line, mnem, t });
                try writeSrc(&body, val);
                try body.put("\n");
                line += 1;
            },
            .icmp => |cmp| {
                // A comparison -> a TGSI set-op producing an integer boolean (~0/0).
                const r = result orelse return error.Unsupported;
                const lhs = try rz.get(cmp.lhs);
                const rhs = try rz.get(cmp.rhs);
                const ty = func.types.type_kind(func.valueType(cmp.lhs));
                const ce = cmpMnemonic(cmp.op, ty) orelse return error.Unsupported;
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, r, .{ .temp = t });
                try body.print("  {d}: {s} TEMP[{d}].x, ", .{ line, ce.mnem, t });
                try writeSrc(&body, if (ce.swap) rhs else lhs);
                try body.put(", ");
                try writeSrc(&body, if (ce.swap) lhs else rhs);
                try body.put("\n");
                line += 1;
            },
            .select => |s| {
                // A value-producing conditional -> UCMP (cond != 0 ? then : else).
                const r = result orelse return error.Unsupported;
                const cond = try rz.get(s.cond);
                const then_v = try rz.get(s.then);
                const else_v = try rz.get(s.@"else");
                const t = temp_count;
                temp_count += 1;
                try src_of.put(allocator, r, .{ .temp = t });
                try body.print("  {d}: UCMP TEMP[{d}].x, ", .{ line, t });
                try writeSrc(&body, cond);
                try body.put(", ");
                try writeSrc(&body, then_v);
                try body.put(", ");
                try writeSrc(&body, else_v);
                try body.put("\n");
                line += 1;
            },
            .store => |st| {
                const out = outputSlot(func, st.ptr) orelse return error.Unsupported;
                const reg = outReg(out);
                if (reg >= out_regs.len) return error.Unsupported;
                const src = try rz.get(st.value);
                const e = &out_regs[reg];
                e.present = true;
                e.kind = out.kind;
                e.location = out.location;
                e.comp_src[out.comp] = src;
                e.mask |= @as(u4, 1) << @intCast(out.comp);
            },
            .load => |ld| {
                const r = result orelse continue;
                // A reload of a texture-sample result -> that component of the TEX temp.
                if (tex_addr.get(ld.ptr)) |te| {
                    try src_of.put(allocator, r, .{ .texcomp = .{ .temp = te.temp, .comp = te.comp } });
                } else if (grad_addr.get(ld.ptr)) |goff| {
                    // A derivative: the grad_slot at index off/4 names (varying slot, axis).
                    // Emit DDX/DDY of that varying into a temp.
                    const gi = goff / 4;
                    if (gi >= grad_slots.items.len) return error.Unsupported;
                    const packed_gs = grad_slots.items[gi];
                    const gslot = packed_gs >> 1;
                    const axis_y = (packed_gs & 1) != 0;
                    if (gslot < ATTR_GENERIC0) return error.Unsupported;
                    const goff2 = gslot - ATTR_GENERIC0;
                    const greg = goff2 / 0x10;
                    const gcomp: u8 = @intCast((goff2 % 0x10) / 4);
                    const t = temp_count;
                    temp_count += 1;
                    try src_of.put(allocator, r, .{ .temp = t });
                    try body.print("  {d}: {s} TEMP[{d}].x, IN[{d}].{c}\n", .{ line, if (axis_y) "DDY" else "DDX", t, greg, channel[gcomp] });
                    line += 1;
                } else {
                    // Otherwise a UBO read: CONST[unit][idx].comp (std140: idx = off/16,
                    // component = (off%16)/4).
                    const a = addr_of.get(ld.ptr) orelse return error.Unsupported;
                    const idx = a.off / 16;
                    const comp: u8 = @intCast((a.off % 16) / 4);
                    const_used[a.unit] = true;
                    if (a.dyn) |dyn_val| {
                        // A DYNAMIC index (e.g. gl_InstanceIndex*stride): load the vec4
                        // index (byte offset >> 4) into ADDR[0], then materialize the
                        // relative CONST[unit][ADDR[0].x + idx].comp read into a temp
                        // (snapshotting the value so a later ADDR reload cannot alias it).
                        addr_used = true;
                        const_dynamic[a.unit] = true;
                        const dyn = try rz.get(dyn_val);
                        const shift_imm: u32 = @intCast(imms.items.len);
                        try imms.append(allocator, .{ .bits = 4, .is_float = false }); // >> 4 = / 16
                        const it = temp_count;
                        temp_count += 1;
                        try body.print("  {d}: USHR TEMP[{d}].x, ", .{ line, it });
                        try writeSrc(&body, dyn);
                        try body.print(", IMM[{d}].x\n", .{shift_imm});
                        line += 1;
                        try body.print("  {d}: UARL ADDR[0].x, TEMP[{d}].x\n", .{ line, it });
                        line += 1;
                        const t = temp_count;
                        temp_count += 1;
                        try body.print("  {d}: MOV TEMP[{d}].x, CONST[{d}][ADDR[0].x + {d}].{c}\n", .{ line, t, a.unit, idx, channel[comp] });
                        line += 1;
                        try src_of.put(allocator, r, .{ .temp = t });
                    } else {
                        if (idx > const_max_idx[a.unit]) const_max_idx[a.unit] = idx;
                        try src_of.put(allocator, r, .{ .constbuf = .{ .unit = a.unit, .idx = idx, .comp = comp } });
                    }
                }
            },
            .alloca => {
                // The 16-byte stack slot the SPIR-V image-sample lowering uses as the
                // sampler out-pointer: reserve a TEMP to hold the RGBA TEX result. The
                // alloca pointer (and its `+c*4` element pointers) resolve to its comps.
                const r = result orelse return error.Unsupported;
                const t = temp_count;
                temp_count += 1;
                try tex_addr.put(allocator, r, .{ .temp = t, .comp = 0 });
            },
            .call_indirect => |c| {
                // A discard call `discard_fn()` -> KILL (unconditional at this point; the
                // structurizer's UIF/ELSE already gates it for a conditional discard).
                if (hasFlag(func, c.target, "discard_fn")) {
                    try body.print("  {d}: KILL\n", .{line});
                    line += 1;
                    continue;
                }
                // A sampler call `sampler_fn(desc, u, v, lod, out_ptr)` -> a TEX. Only the
                // synthesized host-sampler pointer target is recognized here. The lod arg
                // (args[3]) is not yet honored by this backend (always TEX = base level /
                // implicit LOD); an explicit-LOD TXL is a virgl follow-up.
                if (!hasFlag(func, c.target, "sampler_fn")) return error.Unsupported;
                const args = func.valueList(c.args);
                if (args.len != 5) return error.Unsupported; // {desc, u, v, lod, out_ptr}
                const unit = samp_of.get(args[0]) orelse return error.Unsupported;
                const dst = tex_addr.get(args[4]) orelse return error.Unsupported;
                const u = try rz.get(args[1]);
                const v = try rz.get(args[2]);
                samp_used[unit] = true;
                // Gather u,v into one coord register's .xy, then sample into the result
                // TEMP. TGSI TEX reads the coordinate from a single register.
                const coord = temp_count;
                temp_count += 1;
                try body.print("  {d}: MOV TEMP[{d}].x, ", .{ line, coord });
                try writeSrc(&body, u);
                try body.put("\n");
                line += 1;
                try body.print("  {d}: MOV TEMP[{d}].y, ", .{ line, coord });
                try writeSrc(&body, v);
                try body.put("\n");
                line += 1;
                try body.print("  {d}: TEX TEMP[{d}], TEMP[{d}], SAMP[{d}], 2D\n", .{ line, dst.temp, coord, unit });
                line += 1;
            },
            .@"if" => {}, // the conditional branch: emitted as UIF/ELSE/ENDIF control steps
            else => return error.Unsupported, // (unreachable in graphics functions)
        }
    }

    // Emit the output MOVs after the computed instructions.
    {
        var reg: u32 = 0;
        while (reg < out_regs.len) : (reg += 1) {
            const e = out_regs[reg];
            if (!e.present) continue;
            if (passthroughInput(e)) |in_reg| {
                // Every written component copies IN[in_reg]'s matching component: emit
                // the compact register-wide (or masked) MOV, preserving the exact form
                // virglrenderer's known-good passthrough shaders use.
                if (e.mask == 0b1111) {
                    try body.print("  {d}: MOV OUT[{d}], IN[{d}]\n", .{ line, reg, in_reg });
                } else {
                    try body.print("  {d}: MOV OUT[{d}].", .{ line, reg });
                    try body.putMask(e.mask);
                    try body.print(", IN[{d}].", .{in_reg});
                    try body.putMask(e.mask);
                    try body.put("\n");
                }
                line += 1;
            } else {
                // A computed / mixed-source output: one MOV per written component.
                var c: u3 = 0;
                while (c < 4) : (c += 1) {
                    const s = e.comp_src[c] orelse continue;
                    try body.print("  {d}: MOV OUT[{d}].{c}, ", .{ line, reg, channel[c] });
                    try writeSrc(&body, s);
                    try body.put("\n");
                    line += 1;
                }
            }
        }
        try body.print("  {d}: END\n", .{line});
    }

    // Assemble: header declarations (IN, OUT, TEMP, IMM) then the body.
    var buf = Writer{ .allocator = allocator };
    defer buf.list.deinit(allocator);

    try buf.put(if (stage == .vertex) "VERT\n" else "FRAG\n");

    // Input declarations. Vertex inputs are bare vertex attributes. Fragment inputs
    // are perspective-interpolated generic varyings.
    {
        var reg: u32 = 0;
        while (reg < in_present.len) : (reg += 1) {
            if (!in_present[reg]) continue;
            if (stage == .vertex) {
                try buf.print("DCL IN[{d}]\n", .{reg});
            } else {
                try buf.print("DCL IN[{d}], GENERIC[{d}], PERSPECTIVE\n", .{ reg, reg });
            }
        }
        // gl_FragCoord: the fragment window-space position (POSITION semantic input).
        if (fc_used) try buf.print("DCL IN[{d}], POSITION\n", .{FRAG_COORD_REG});
    }

    // System-value (builtin input) declarations: gl_VertexIndex / gl_InstanceIndex.
    {
        var i: u32 = 0;
        while (i < sv_count) : (i += 1) {
            try buf.print("DCL SV[{d}], {s}\n", .{ i, sv_sem[i].name() });
        }
    }

    // Output declarations.
    {
        var reg: u32 = 0;
        while (reg < out_regs.len) : (reg += 1) {
            const e = out_regs[reg];
            if (!e.present) continue;
            switch (e.kind) {
                .position => try buf.print("DCL OUT[{d}], POSITION\n", .{reg}),
                .generic => try buf.print("DCL OUT[{d}], GENERIC[{d}]\n", .{ reg, e.location }),
                .color => try buf.print("DCL OUT[{d}], COLOR\n", .{reg}),
                .psize => try buf.print("DCL OUT[{d}], PSIZE\n", .{reg}),
                .fragdepth => try buf.print("DCL OUT[{d}], POSITION\n", .{reg}),
            }
        }
    }

    // Uniform-block (CONST) declarations: one per referenced binding unit, sized to
    // the highest vec4 index the shader reads. A dynamically-indexed unit declares a
    // large range (the runtime index is bounded by the SET_CONSTANT_BUFFER data).
    {
        var unit: u32 = 0;
        while (unit < const_used.len) : (unit += 1) {
            if (!const_used[unit]) continue;
            const hi: u32 = if (const_dynamic[unit]) 4095 else const_max_idx[unit];
            try buf.print("DCL CONST[{d}][0..{d}]\n", .{ unit, hi });
        }
    }

    // Address register for relative CONST indexing (dynamic UBO reads).
    if (addr_used) try buf.put("DCL ADDR[0]\n");

    // Sampler + sampler-view declarations, one per referenced SAMP unit (a 2D float
    // combined-image-sampler: the classic TGSI form virglrenderer's TEX consumes).
    {
        var unit: u32 = 0;
        while (unit < samp_used.len) : (unit += 1) {
            if (!samp_used[unit]) continue;
            try buf.print("DCL SAMP[{d}]\n", .{unit});
            try buf.print("DCL SVIEW[{d}], 2D, FLOAT\n", .{unit});
        }
    }

    // Scratch temp + immediate declarations, only when the body uses them (so a pure
    // passthrough shader declares neither, matching the known-good output exactly).
    if (temp_count > 0) try buf.print("DCL TEMP[0..{d}]\n", .{temp_count - 1});
    for (imms.items, 0..) |im, i| {
        if (im.is_float) {
            try buf.print("IMM[{d}] FLT32 {{ {d}, 0, 0, 0}}\n", .{ i, @as(f32, @bitCast(im.bits)) });
        } else {
            try buf.print("IMM[{d}] INT32 {{ {d}, 0, 0, 0}}\n", .{ i, @as(i32, @bitCast(im.bits)) });
        }
    }

    try buf.put(body.list.items);

    // NUL-terminate and dword-pad: virglrenderer reads the shader text as a
    // token-aligned blob, so the byte length must round up to a multiple of 4.
    try buf.putByte(0);
    while (buf.list.items.len % 4 != 0) try buf.putByte(0);

    return buf.list.toOwnedSlice(allocator);
}

/// If every written component of `e` copies the matching component of a single
/// input register (an identity passthrough), return that input register index.
fn passthroughInput(e: OutReg) ?u32 {
    var in_reg: ?u32 = null;
    var c: u3 = 0;
    while (c < 4) : (c += 1) {
        if (e.mask & (@as(u4, 1) << @intCast(c)) == 0) continue;
        const s = e.comp_src[c] orelse return null;
        switch (s) {
            .input => |slot| {
                if (slot.comp != c) return null; // component permutation, not identity
                if (in_reg) |r| {
                    if (r != slot.reg) return null; // mixed source registers
                } else in_reg = slot.reg;
            },
            else => return null,
        }
    }
    return in_reg;
}

/// The TGSI set-on-comparison op for a CmpOp + operand type. Comparisons produce
/// an integer boolean (~0 true / 0 false) that UCMP (select) tests as non-zero.
/// `swap` means emit the operands reversed (a<=b lowers to b>=a, a>b to b<a, since
/// TGSI has no direct FSLE/FSGT). Returns null for an unsupported operand kind.
const CmpEmit = struct { mnem: []const u8, swap: bool };
fn cmpMnemonic(op: ir.function.CmpOp, ty: ir.types.TypeKind) ?CmpEmit {
    const is_float = ty == .float;
    const unsigned = ty == .int and ty.int.signedness == .unsigned;
    return switch (op) {
        .eq => .{ .mnem = if (is_float) "FSEQ" else "USEQ", .swap = false },
        .ne => .{ .mnem = if (is_float) "FSNE" else "USNE", .swap = false },
        .lt => .{ .mnem = if (is_float) "FSLT" else if (unsigned) "USLT" else "ISLT", .swap = false },
        .ge => .{ .mnem = if (is_float) "FSGE" else if (unsigned) "USGE" else "ISGE", .swap = false },
        .le => .{ .mnem = if (is_float) "FSGE" else if (unsigned) "USGE" else "ISGE", .swap = true },
        .gt => .{ .mnem = if (is_float) "FSLT" else if (unsigned) "USLT" else "ISLT", .swap = true },
    };
}

/// The TGSI numeric-convert mnemonic for a source->result scalar kind pair, or null
/// if the pair is not a supported int<->float conversion.
fn convertMnemonic(src: ir.types.TypeKind, dst: ir.types.TypeKind) ?[]const u8 {
    const src_float = src == .float;
    const dst_float = dst == .float;
    if (src_float == dst_float) return "MOV"; // same domain: a bit-preserving move
    if (dst_float) {
        // int -> float
        return switch (src) {
            .int => |i| if (i.signedness == .signed) "I2F" else "U2F",
            else => null,
        };
    }
    // float -> int
    return switch (dst) {
        .int => |i| if (i.signedness == .signed) "F2I" else "F2U",
        else => null,
    };
}

/// Accumulating text builder over an ArrayList. Keeps TGSI emission self-contained.
const Writer = struct {
    allocator: std.mem.Allocator,
    list: std.ArrayList(u8) = .empty,

    fn put(self: *Writer, s: []const u8) Error!void {
        try self.list.appendSlice(self.allocator, s);
    }
    fn putByte(self: *Writer, b: u8) Error!void {
        try self.list.append(self.allocator, b);
    }
    fn print(self: *Writer, comptime fmt: []const u8, args: anytype) Error!void {
        var tmp: [64]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, fmt, args) catch return error.Unsupported;
        try self.put(s);
    }
    /// Emit the swizzle channel letters for a writemask (mask 0b0011 -> "xy").
    fn putMask(self: *Writer, mask: u4) Error!void {
        var c: u3 = 0;
        while (c < 4) : (c += 1) {
            if (mask & (@as(u4, 1) << @intCast(c)) != 0) try self.putByte(channel[c]);
        }
    }
};

/// The output register index a slot maps to. POSITION/varyings keep their slot's
/// location. The fragment color is OUT[0].
fn outReg(out: OutputSlot) u32 {
    return switch (out.kind) {
        .position => 0,
        .generic => out.location + 1, // OUT[0] is POSITION, varyings follow
        .color => out.location, // the render-target index (MRT); OUT[0] for single-RT
        .psize => 31, // a reserved high register (varyings never reach it)
        .fragdepth => 30, // a reserved register for the FS depth (POSITION semantic)
    };
}

/// Decode an output store pointer's tag into an OutputSlot, or null if untagged.
fn outputSlot(func: *const Function, ptr: Value) ?OutputSlot {
    if (attrTag(func, ptr, "color_out")) |packed_co| {
        // color_out = render-target index * 4 + component (MRT).
        return .{ .kind = .color, .location = packed_co / 4, .comp = @intCast(packed_co % 4) };
    }
    if (attrTag(func, ptr, "frag_depth") != null) {
        // gl_FragDepth: the fragment depth, written to the .z of a POSITION output.
        return .{ .kind = .fragdepth, .location = 0, .comp = 2 };
    }
    if (attrTag(func, ptr, "out_attr")) |slot| {
        if (slot == ATTR_POINT_SIZE) {
            return .{ .kind = .psize, .location = 0, .comp = 0 }; // gl_PointSize (a scalar)
        }
        if (slot >= ATTR_POSITION and slot < ATTR_GENERIC0) {
            return .{ .kind = .position, .location = 0, .comp = @intCast((slot - ATTR_POSITION) / 4) };
        }
        if (slot >= ATTR_GENERIC0) {
            const off = slot - ATTR_GENERIC0;
            return .{ .kind = .generic, .location = off / 0x10, .comp = @intCast((off % 0x10) / 4) };
        }
    }
    return null;
}

/// The shader stage tagged on the function by the SPIR-V graphics lowering.
fn stageOf(func: *const Function) ?Stage {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "stage")) {
            return switch (c.value) {
                .string => |s| if (std.mem.eql(u8, s, "vertex"))
                    .vertex
                else if (std.mem.eql(u8, s, "fragment"))
                    .fragment
                else
                    null,
                else => null,
            };
        },
        else => {},
    };
    return null;
}

/// Whether value `v` carries a `vulcan.gpu` custom attribute named `key` (any
/// value form, e.g. the `.flag` marker the sampler-fn param uses).
fn hasFlag(func: *const Function, v: Value, key: []const u8) bool {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) return true,
        else => {},
    };
    return false;
}

/// A `vulcan.gpu` integer attribute named `key` on value `v`, or null.
fn attrTag(func: *const Function, v: Value, key: []const u8) ?u32 {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) {
            return switch (c.value) {
                .int => |n| @intCast(n),
                else => null,
            };
        },
        else => {},
    };
    return null;
}

const testing = std.testing;

test "lower a passthrough vertex shader to TGSI (position + color varying)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // Two vec4 inputs: position (loc 0) and color (loc 1), scalarized to 4 params
    // each, tagged with their attribute byte slots.
    var pos_in: [4]Value = undefined;
    var col_in: [4]Value = undefined;
    inline for (0..4) |c| {
        pos_in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos_in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    inline for (0..4) |c| {
        col_in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = col_in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + 0x10 + c * 4 } } });
    }
    // Store position -> OUT[0] POSITION, color -> OUT[1] GENERIC[0].
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, pos_in[c], ptr);
    }
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_GENERIC0 + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
        try func.appendStore(b, col_in[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    const want =
        "VERT\n" ++
        "DCL IN[0]\n" ++
        "DCL IN[1]\n" ++
        "DCL OUT[0], POSITION\n" ++
        "DCL OUT[1], GENERIC[0]\n" ++
        "  1: MOV OUT[0], IN[0]\n" ++
        "  2: MOV OUT[1], IN[1]\n" ++
        "  3: END\n";
    try testing.expectStringStartsWith(tgsi, want);
    try testing.expectEqual(@as(u8, 0), tgsi[want.len]); // NUL-terminated
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4); // dword padded
}

test "lower a passthrough fragment shader to TGSI (interpolated varying -> color)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // One vec4 input varying (loc 0), scalarized. The color output is the four
    // color_out components (R0..R3) copied straight from it.
    var col_in: [4]Value = undefined;
    inline for (0..4) |c| {
        col_in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = col_in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, col_in[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    const want =
        "FRAG\n" ++
        "DCL IN[0], GENERIC[0], PERSPECTIVE\n" ++
        "DCL OUT[0], COLOR\n" ++
        "  1: MOV OUT[0], IN[0]\n" ++
        "  2: END\n";
    try testing.expectStringStartsWith(tgsi, want);
    try testing.expectEqual(@as(u8, 0), tgsi[want.len]);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower an arithmetic fragment shader (mul of two input components) to TGSI" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // One vec4 input varying (loc 0), scalarized.
    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // color.x = in.x * in.y (a computed component). color.yzw passthrough in.yzw.
    const prod = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = in[0], .rhs = in[1] } });
    const store_srcs = [4]Value{ prod, in[1], in[2], in[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, store_srcs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    // The computed component uses a scratch temp. The passthrough components read the
    // input directly. Assert the salient lines are present.
    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL TEMP[0..0]\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MUL TEMP[0].x, IN[0].x, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].y, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].w, IN[0].w\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "END\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower an add-with-constant vertex shader (arith_imm + fconst pool) to TGSI" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var pos: [4]Value = undefined;
    inline for (0..4) |c| {
        pos[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // A float constant scaled into the position x via mul (fconst + arith), and a
    // bias added to y via arith_imm. Exercises both the fconst and arith_imm pools.
    const half = try func.appendInst(b, f32_t, .{ .fconst = 0.5 });
    const sx = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = pos[0], .rhs = half } });
    const sy = try func.appendInst(b, f32_t, .{ .arith_imm = .{ .op = .add, .lhs = pos[1], .imm = 3 } });
    const out_srcs = [4]Value{ sx, sy, pos[2], pos[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, out_srcs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "IMM[0] FLT32 { 0.5, 0, 0, 0}\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MUL TEMP[0].x, IN[0].x, IMM[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "ADD TEMP[1].x, IN[0].y, IMM[1].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].y, TEMP[1].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a UBO-reading fragment shader (uniform vec4 color) to TGSI CONST[]" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();

    // A UBO base pointer at binding 0 (the SPIR-V lowering's buffer entry param).
    const ubo = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = ubo }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "binding", .value = .{ .int = 0 } } });

    // color.c = load(ubo + c*4) for c in 0..3 -> CONST[0][0].{x,y,z,w}. Component 0
    // reads the base directly. The rest add a byte offset (the std140 layout).
    inline for (0..4) |c| {
        const eptr = if (c == 0) ubo else blk: {
            const off = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c * 4) });
            break :blk try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = ubo, .rhs = off } });
        };
        const val = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = eptr } });
        const sptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = sptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, val, sptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL CONST[0][0..0]\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, CONST[0][0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].y, CONST[0][0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].w, CONST[0][0].w\n") != null);
    // No IMM pollution: the pointer-offset iconsts must not have been materialized.
    try testing.expect(std.mem.indexOf(u8, tgsi, "IMM[") == null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a UBO-transform vertex shader (uniform scale of position) to TGSI" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();

    // Position input (loc 0) + a UBO at binding 0 holding a scale vec4.
    var pos: [4]Value = undefined;
    inline for (0..4) |c| {
        pos[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const ubo = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = ubo }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "binding", .value = .{ .int = 0 } } });

    // out.pos.c = pos.c * uniform.scale.c  (a per-component multiply by CONST[0][0].c).
    inline for (0..4) |c| {
        const eptr = if (c == 0) ubo else blk: {
            const off = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c * 4) });
            break :blk try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = ubo, .rhs = off } });
        };
        const scale = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = eptr } });
        const scaled = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = pos[c], .rhs = scale } });
        const sptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = sptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, scaled, sptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL CONST[0][0..0]\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MUL TEMP[0].x, IN[0].x, CONST[0][0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MUL TEMP[3].x, IN[0].w, CONST[0][0].w\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a texturing fragment shader (sampler2D) to TGSI SAMP/TEX" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();

    // Input uv (loc 0, two components), a sampler descriptor (binding 0), and the
    // synthesized host-sampler function pointer (ignored by a GPU/TGSI target).
    var uv: [2]Value = undefined;
    inline for (0..2) |c| {
        uv[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = uv[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const samp = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = samp }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_desc", .value = .{ .int = 1 } } });
    try func.addAttr(.{ .value = samp }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "binding", .value = .{ .int = 0 } } });
    const sfn = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = sfn }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_fn", .value = .flag } });

    // allocate out_ptr, call sampler_fn(samp, u, v, lod, out_ptr), then reload the 4 texels.
    const out_ptr = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const lod0 = try func.appendInst(b, f32_t, .{ .fconst = 0 });
    const args = [5]Value{ samp, uv[0], uv[1], lod0, out_ptr };
    _ = try func.appendCallIndirect(b, f32_t, sfn, &args);
    inline for (0..4) |c| {
        const eptr = if (c == 0) out_ptr else blk: {
            const off = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c * 4) });
            break :blk try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = out_ptr, .rhs = off } });
        };
        const texel = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = eptr } });
        const sptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = sptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, texel, sptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL SAMP[0]\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL SVIEW[0], 2D, FLOAT\n") != null);
    // The coord (temp 1) is gathered from the uv input, then TEX into the result (temp 0).
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV TEMP[1].x, IN[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV TEMP[1].y, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "TEX TEMP[0], TEMP[1], SAMP[0], 2D\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].w, TEMP[0].w\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a vertex shader reading gl_InstanceIndex to TGSI SV[] INSTANCEID" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var pos: [4]Value = undefined;
    inline for (0..4) |c| {
        pos[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // gl_InstanceIndex (builtin 43), converted to float and added to position.x.
    const inst = try func.appendBlockParam(b, i32_t);
    try func.addAttr(.{ .value = inst }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = 43 } } });
    const fi = try func.appendInst(b, f32_t, .{ .convert = .{ .value = inst } });
    const sx = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = pos[0], .rhs = fi } });
    const outs = [4]Value{ sx, pos[1], pos[2], pos[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, outs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL SV[0], INSTANCEID\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "I2F TEMP[0].x, SV[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "ADD TEMP[1].x, IN[0].x, TEMP[0].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a float remainder (mod) to the TGSI trunc-based sequence" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // color.x = mod(in.x, in.y), passthrough the rest.
    const m = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .rem, .lhs = in[0], .rhs = in[1] } });
    const outs = [4]Value{ m, in[1], in[2], in[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, outs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    // t = TEMP[0] (result), s = TEMP[1] (scratch): x - y*trunc(x/y).
    try testing.expect(std.mem.indexOf(u8, tgsi, "DIV TEMP[1].x, IN[0].x, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "TRUNC TEMP[1].x, TEMP[1].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MUL TEMP[1].x, TEMP[1].x, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "ADD TEMP[0].x, IN[0].x, -TEMP[1].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a select/icmp fragment shader (ternary) to TGSI FSLT + UCMP" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // color.x = (in.x < in.y) ? in.z : in.w, passthrough the rest.
    const cond = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = in[0], .rhs = in[1] } });
    const sel = try func.appendInst(b, f32_t, .{ .select = .{ .cond = cond, .then = in[2], .@"else" = in[3] } });
    const outs = [4]Value{ sel, in[1], in[2], in[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, outs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "FSLT TEMP[0].x, IN[0].x, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "UCMP TEMP[1].x, TEMP[0].x, IN[0].z, IN[0].w\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[1].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a per-instance UBO fetch (dynamic CONST index by gl_InstanceIndex) to TGSI ADDR" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();

    var pos: [4]Value = undefined;
    inline for (0..4) |c| {
        pos[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const ubo = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = ubo }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "binding", .value = .{ .int = 0 } } });
    const inst = try func.appendBlockParam(b, i32_t);
    try func.addAttr(.{ .value = inst }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = 43 } } });

    // offset = gl_InstanceIndex * 16 (the vec4 array stride). base_dyn = ubo + offset.
    const stride = try func.appendInst(b, i32_t, .{ .iconst = 16 });
    const scaled = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = inst, .rhs = stride } });
    const base_dyn = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = ubo, .rhs = scaled } });
    // out.pos.c = pos.c + ubo.arr[gl_InstanceIndex].c
    inline for (0..4) |c| {
        const eptr = if (c == 0) base_dyn else blk: {
            const off = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c * 4) });
            break :blk try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base_dyn, .rhs = off } });
        };
        const v = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = eptr } });
        const sum = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = pos[c], .rhs = v } });
        const sptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = sptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, sum, sptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL CONST[0][0..4095]\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL ADDR[0]\n") != null);
    // Integer index math (NOT the float ALU), then ADDR load + a relative CONST read.
    try testing.expect(std.mem.indexOf(u8, tgsi, "UMUL TEMP[0].x, SV[0].x,") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "UARL ADDR[0].x,") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "CONST[0][ADDR[0].x + 0].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a multi-block if/else diamond (branch + phi) to TGSI UIF/ELSE/ENDIF" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // 4 blocks: entry -> (then | else) -> merge. color.x = (in.x<in.y) ? in.z : in.w,
    // computed by a real branch with a phi at the merge (not a select).
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(entry, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const phi = try func.appendBlockParam(merge, f32_t); // the selected value

    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = in[0], .rhs = in[1] } });
    try func.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
    try func.setJump(then_b, merge, &.{in[2]}); // then edge: phi = in.z
    try func.setJump(else_b, merge, &.{in[3]}); // else edge: phi = in.w

    const outs = [4]Value{ phi, in[1], in[2], in[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(merge, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(merge, outs[c], ptr);
    }
    func.setTerminator(merge, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    // phi -> TEMP[0] (non-entry param temps assigned first). cond -> TEMP[1].
    try testing.expect(std.mem.indexOf(u8, tgsi, "FSLT TEMP[1].x, IN[0].x, IN[0].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "UIF TEMP[1].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV TEMP[0].x, IN[0].z\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "ELSE\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV TEMP[0].x, IN[0].w\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "ENDIF\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    // The UIF must come before the ELSE which comes before the ENDIF.
    const uif = std.mem.indexOf(u8, tgsi, "UIF ").?;
    const els = std.mem.indexOf(u8, tgsi, "ELSE\n").?;
    const endi = std.mem.indexOf(u8, tgsi, "ENDIF\n").?;
    try testing.expect(uif < els and els < endi);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a while-loop (back edge + loop-carried phi) to TGSI BGNLOOP/BRK/ENDLOOP" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    // 4 blocks: entry -> header(acc,i) -> (body -> back to header | exit to merge).
    //   acc starts at in.x, i at 0. While i < 3, add in.y to acc then i++. color.x = acc.
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const merge = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(entry, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const acc = try func.appendBlockParam(header, f32_t); // TEMP[0]
    const i = try func.appendBlockParam(header, i32_t); // TEMP[1]
    const res = try func.appendBlockParam(merge, f32_t); // TEMP[2]

    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ in[0], zero }); // init acc=in.x, i=0

    const three = try func.appendInst(header, i32_t, .{ .iconst = 3 });
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = three } });
    try func.appendIf(header, cond, .{ .target = body }, .{ .target = merge, .args = &.{acc} }); // exit: res = acc

    const acc2 = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = in[1] } });
    const inc_i = try func.appendInst(body, i32_t, .{ .arith_imm = .{ .op = .add, .lhs = i, .imm = 1 } });
    try func.setJump(body, header, &.{ acc2, inc_i }); // back edge: acc=acc2, i=i+1

    const outs = [4]Value{ res, in[1], in[2], in[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(merge, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(merge, outs[c], ptr);
    }
    func.setTerminator(merge, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    const bgn = std.mem.indexOf(u8, tgsi, "BGNLOOP\n");
    const uif = std.mem.indexOf(u8, tgsi, "UIF ");
    const brk = std.mem.indexOf(u8, tgsi, "BRK\n");
    const endl = std.mem.indexOf(u8, tgsi, "ENDLOOP\n");
    try testing.expect(bgn != null and uif != null and brk != null and endl != null);
    try testing.expect(bgn.? < uif.? and uif.? < brk.? and brk.? < endl.?);
    // The back edge updates the loop-carried temps (acc=TEMP[0], i=TEMP[1]).
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV TEMP[0].x, TEMP[4].x\n") != null); // acc = acc2
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[2].x\n") != null); // color.x = res
    // i < 3 is a signed integer compare -> ISLT (not the float FSLT).
    try testing.expect(std.mem.indexOf(u8, tgsi, "ISLT ") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a vertex shader writing gl_PointSize to a TGSI PSIZE output" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var pos: [4]Value = undefined;
    inline for (0..4) |c| {
        pos[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, pos[c], ptr);
    }
    // gl_PointSize = 8.0 -> the ATTR_POINT_SIZE (0x6c) output slot.
    const size = try func.appendInst(b, f32_t, .{ .fconst = 8.0 });
    const sptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POINT_SIZE) });
    try func.addAttr(.{ .value = sptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POINT_SIZE } } });
    try func.appendStore(b, size, sptr);
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL OUT[31], PSIZE\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "IMM[0] FLT32 { 8, 0, 0, 0}\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[31].x, IMM[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0], IN[0]\n") != null); // position still passes through
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a derivative (dFdx of a varying) to TGSI DDX" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    // The grad_slot for index 0: varying slot ATTR_GENERIC0 (loc 0, comp x), axis x.
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_slot", .value = .{ .int = @as(i64, ATTR_GENERIC0) << 1 } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const gbuf = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = gbuf }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_buf", .value = .{ .int = 0 } } });

    // color.x = dFdx(varying.x) = load(grad_buf) (index 0), passthrough the rest.
    const deriv = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = gbuf } });
    const outs = [4]Value{ deriv, in[1], in[2], in[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, outs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DDX TEMP[0].x, IN[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, TEMP[0].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a fragment shader reading gl_FragCoord + gl_FrontFacing to TGSI POSITION/FACE" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // gl_FragCoord (vec4, builtin 15) + gl_FrontFacing (bool, builtin 17).
    var fc: [4]Value = undefined;
    inline for (0..4) |c| {
        fc[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = fc[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = 15 } } });
        try func.addAttr(.{ .value = fc[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "bicomp", .value = .{ .int = c } } });
    }
    const ff = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = ff }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = 17 } } });
    try func.addAttr(.{ .value = ff }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "bicomp", .value = .{ .int = 0 } } });

    // color = (fragcoord.x, fragcoord.y, frontfacing, fragcoord.w).
    const srcs = [4]Value{ fc[0], fc[1], ff, fc[3] };
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, srcs[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL IN[16], POSITION\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL SV[0], FACE\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].x, IN[16].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].y, IN[16].y\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].z, SV[0].x\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0].w, IN[16].w\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a fragment shader writing gl_FragDepth to a TGSI POSITION.z output" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // color passthrough, plus gl_FragDepth = in.x.
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, in[c], ptr);
    }
    const dptr = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    try func.addAttr(.{ .value = dptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "frag_depth", .value = .{ .int = 0 } } });
    try func.appendStore(b, in[0], dptr);
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL OUT[30], POSITION\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[30].z, IN[0].x\n") != null);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower a fragment shader with a conditional discard to TGSI UIF/KILL/ENDIF" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const void_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 0 } });

    const entry = try func.appendBlock();
    const kill = try func.appendBlock();
    const merge = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(entry, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    const dfn = try func.appendBlockParam(entry, ptr_t);
    try func.addAttr(.{ .value = dfn }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "discard_fn", .value = .flag } });

    // if (in.x < in.y) discard; color = in;
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = in[0], .rhs = in[1] } });
    try func.appendIf(entry, cond, .{ .target = kill }, .{ .target = merge });
    _ = try func.appendCallIndirect(kill, void_t, dfn, &.{});
    func.setTerminator(kill, .{ .ret = null });
    inline for (0..4) |c| {
        const ptr = try func.appendInst(merge, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(merge, in[c], ptr);
    }
    func.setTerminator(merge, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    const uif = std.mem.indexOf(u8, tgsi, "UIF ");
    const kil = std.mem.indexOf(u8, tgsi, "KILL\n");
    const endi = std.mem.indexOf(u8, tgsi, "ENDIF\n");
    const col = std.mem.indexOf(u8, tgsi, "MOV OUT[0], IN[0]\n"); // the passthrough color, compacted
    try testing.expect(uif != null and kil != null and endi != null and col != null);
    try testing.expect(uif.? < kil.? and kil.? < endi.? and endi.? < col.?); // discard gated, color after
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test "lower an MRT fragment shader (two color targets) to TGSI OUT[0]/OUT[1] COLOR" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    var in: [4]Value = undefined;
    inline for (0..4) |c| {
        in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    // Target 0: passthrough color (color_out 0..3). Target 1: color_out 4..7, reversed.
    const t1 = [4]Value{ in[3], in[2], in[1], in[0] };
    inline for (0..4) |c| {
        const p0 = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = p0 }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, in[c], p0);
        const p1 = try func.appendInst(b, i32_t, .{ .iconst = @intCast(4 + c) });
        try func.addAttr(.{ .value = p1 }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = 4 + c } } });
        try func.appendStore(b, t1[c], p1);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL OUT[0], COLOR\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "DCL OUT[1], COLOR\n") != null);
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[0], IN[0]\n") != null); // target 0 passthrough
    try testing.expect(std.mem.indexOf(u8, tgsi, "MOV OUT[1].x, IN[0].w\n") != null); // target1.x = in.w
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test {
    testing.refAllDecls(@This());
}
