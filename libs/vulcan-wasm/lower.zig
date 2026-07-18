//! Lower a parsed Wasm module to Vulcan IR. Walks the binary sections, then turns each
//! function's stack-machine body into SSA via an operand stack of IR `Value`s: each
//! opcode pops its inputs and pushes its result.
//!
//! Wasm integers carry no signedness (the opcode picks, e.g. `div_s` vs `div_u`). The
//! Vulcan backend derives signedness from the operand type. Values are kept canonically
//! signed. An unsigned opcode coerces its operands to the unsigned type with `x | 0` (a
//! no-op the optimizer folds away).

const std = @import("std");
const ir = @import("vulcan-ir");
const reader = @import("reader.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Cursor = reader.Cursor;

pub const Error = reader.Error || error{Unsupported} || std.mem.Allocator.Error;

/// Upper bound on function-table entries a module may declare. An element segment's
/// init offset is untrusted; without a cap a tiny module could name a near-maxInt(u32)
/// slot and force the table-growth loop into a multi-GB allocation. 1M funcref entries
/// is already far beyond any real module we JIT.
const max_table_entries: u32 = 1 << 20;

pub const LoweredFunction = struct {
    name: []u8,
    func: Function,
};

/// A module global's initial value (the raw 64-bit slot contents). The runtime
/// initializes its globals buffer with these before running the module.
pub const GlobalInit = struct { value: i64 };

/// An active data segment: `bytes` to copy into linear memory at `offset`.
pub const DataSegment = struct { offset: u32, bytes: []u8 };

pub const Module = struct {
    functions: []LoweredFunction,
    /// Whether the module declares a linear memory, and its initial size in pages.
    has_memory: bool = false,
    min_pages: u32 = 0,
    /// Whether functions take the hidden context pointer (any memory/globals/table/
    /// imports). The runtime must pass the context to exported calls iff this is set.
    needs_context: bool = false,
    /// The module's globals, in index order. A function that reads/writes a global
    /// takes a hidden "globals base" pointer to a buffer the runtime fills with these.
    globals: []GlobalInit,
    /// Active data segments to copy into the memory buffer before running.
    data: []DataSegment,
    /// The function table as function indices (from element segments). A function
    /// using `call_indirect` takes a hidden "table base" pointer to a buffer the
    /// runtime fills with the JITed address of each of these functions.
    table: []u32,
    /// Imported function names ("module.field"), in import index order. A function
    /// that calls an import takes a hidden "imports base" pointer to a buffer the
    /// runtime fills with the host address of each import.
    imports: [][]u8,

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        for (self.functions) |*lf| {
            lf.func.deinit();
            allocator.free(lf.name);
        }
        for (self.data) |seg| allocator.free(seg.bytes);
        for (self.imports) |name| allocator.free(name);
        allocator.free(self.functions);
        allocator.free(self.globals);
        allocator.free(self.data);
        allocator.free(self.table);
        allocator.free(self.imports);
        self.functions = &.{};
        self.globals = &.{};
        self.data = &.{};
        self.table = &.{};
        self.imports = &.{};
    }

    pub fn find(self: *const Module, name: []const u8) ?*const Function {
        for (self.functions) |*lf| if (std.mem.eql(u8, lf.name, name)) return &lf.func;
        return null;
    }
};

const FuncType = struct { params: []const u8, results: []const u8 };
const Export = struct { name: []const u8, idx: u32 };

/// A Wasm local: its stack slot (alloca pointer) and value type. Reads and writes
/// become load/store, keeping the value correct across control flow.
const Local = struct { ptr: Value, ty: ir.types.Type };

/// Module-level context a function body needs while lowering: resolving `call` targets
/// and accessing linear memory.
const Ctx = struct {
    types: []const FuncType,
    func_type_idx: []const u32,
    exports: []const Export,
    /// Whether the module declares a linear memory. When set, every function takes a
    /// hidden leading pointer parameter (the memory base), threaded through calls.
    has_memory: bool,
    min_pages: u32,
    /// The value type of each global (for `global.get` load typing). When non-empty,
    /// every function takes a hidden "globals base" pointer (after the memory base).
    global_types: []const u8,
    /// Whether the module declares a function table. When set, every function takes a
    /// hidden "table base" pointer (after the globals base) for `call_indirect`.
    has_table: bool,
    /// The number of imported functions (which occupy the low function indices).
    n_imports: u32,
    /// Whether any function is imported. When set, every function takes a hidden
    /// "imports base" pointer (after the table base).
    has_imports: bool,
};

/// Parse and lower the whole module.
pub fn module(allocator: std.mem.Allocator, bytes: []const u8) Error!Module {
    var types_list: std.ArrayList(FuncType) = .empty;
    defer types_list.deinit(allocator);
    var func_type_idx: std.ArrayList(u32) = .empty;
    defer func_type_idx.deinit(allocator);
    var code_bodies: std.ArrayList([]const u8) = .empty;
    defer code_bodies.deinit(allocator);
    var exports: std.ArrayList(Export) = .empty;
    defer exports.deinit(allocator);
    var has_memory = false;
    var min_pages: u32 = 0;
    var global_types: std.ArrayList(u8) = .empty;
    defer global_types.deinit(allocator);
    var global_inits: std.ArrayList(GlobalInit) = .empty;
    defer global_inits.deinit(allocator);
    const RawData = struct { offset: u32, bytes: []const u8 };
    var data_segs: std.ArrayList(RawData) = .empty;
    defer data_segs.deinit(allocator);
    var has_table = false;
    var table_idx: std.ArrayList(u32) = .empty;
    defer table_idx.deinit(allocator);
    var import_names: std.ArrayList([]u8) = .empty;
    // deinit registered before the errdefer so on error the items are freed first, then the
    // buffer (deferred statements run in reverse registration order).
    defer import_names.deinit(allocator);
    errdefer for (import_names.items) |s| allocator.free(s);
    var import_type_idx: std.ArrayList(u32) = .empty;
    defer import_type_idx.deinit(allocator);

    var c = Cursor{ .bytes = bytes };
    if (!std.mem.eql(u8, try c.take(4), &reader.magic)) return error.InvalidWasm;
    if (!std.mem.eql(u8, try c.take(4), &reader.version)) return error.InvalidWasm;

    while (!c.atEnd()) {
        const id = try c.byte();
        const size = try c.u32leb();
        const content = try c.take(size);
        var s = Cursor{ .bytes = content };
        switch (id) {
            1 => { // Type section
                const n = try s.u32leb();
                for (0..n) |_| {
                    if ((try s.byte()) != 0x60) return error.InvalidWasm;
                    const np = try s.u32leb();
                    const params = try s.take(np);
                    const nr = try s.u32leb();
                    const results = try s.take(nr);
                    try types_list.append(allocator, .{ .params = params, .results = results });
                }
            },
            2 => { // Import section
                const n = try s.u32leb();
                for (0..n) |_| {
                    const mod = try s.name();
                    const field = try s.name();
                    const kind = try s.byte();
                    if (kind == 0x00) { // imported function
                        const joined = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ mod, field });
                        errdefer allocator.free(joined);
                        try import_names.append(allocator, joined);
                        try import_type_idx.append(allocator, try s.u32leb());
                    } else { // other import kinds: skip the descriptor
                        switch (kind) {
                            0x01 => { // table
                                _ = try s.byte();
                                const fl = try s.byte();
                                _ = try s.u32leb();
                                if (fl & 1 != 0) _ = try s.u32leb();
                            },
                            0x02 => { // memory
                                const fl = try s.byte();
                                _ = try s.u32leb();
                                if (fl & 1 != 0) _ = try s.u32leb();
                            },
                            0x03 => { // global
                                _ = try s.byte();
                                _ = try s.byte();
                            },
                            else => return error.Unsupported,
                        }
                    }
                }
            },
            3 => { // Function section
                const n = try s.u32leb();
                for (0..n) |_| try func_type_idx.append(allocator, try s.u32leb());
            },
            7 => { // Export section
                const n = try s.u32leb();
                for (0..n) |_| {
                    const nm = try s.name();
                    const kind = try s.byte();
                    const idx = try s.u32leb();
                    if (kind == 0x00) try exports.append(allocator, .{ .name = nm, .idx = idx });
                }
            },
            4 => { // Table section
                const n = try s.u32leb();
                if (n > 0) {
                    has_table = true;
                    _ = try s.byte(); // element type (funcref)
                    const flags = try s.byte();
                    _ = try s.u32leb(); // min
                    if (flags & 0x01 != 0) _ = try s.u32leb(); // max (ignored)
                }
            },
            5 => { // Memory section
                const n = try s.u32leb();
                if (n > 0) {
                    has_memory = true;
                    const flags = try s.byte();
                    min_pages = try s.u32leb();
                    if (flags & 0x01 != 0) _ = try s.u32leb(); // max pages (ignored)
                }
            },
            9 => { // Element section (active, table 0, func indices)
                const n = try s.u32leb();
                for (0..n) |_| {
                    const flag = try s.byte();
                    if (flag != 0x00) return error.Unsupported;
                    // The init offset is an untrusted sleb: reject negative / >u32 rather
                    // than panicking in @intCast.
                    const offset = std.math.cast(u32, try constExpr(&s)) orelse return error.InvalidWasm;
                    const cnt = try s.u32leb();
                    for (0..cnt) |j| {
                        const fi = try s.u32leb();
                        // `offset` is untrusted: compute the slot with checked add, then cap
                        // the resulting table size so a near-maxInt(u32) offset cannot drive
                        // the growth loop below into a multi-GB allocation.
                        const jj = std.math.cast(u32, j) orelse return error.InvalidWasm;
                        const slot = std.math.add(u32, offset, jj) catch return error.InvalidWasm;
                        if (slot >= max_table_entries) return error.InvalidWasm;
                        while (table_idx.items.len <= slot) try table_idx.append(allocator, 0);
                        table_idx.items[slot] = fi;
                    }
                }
            },
            6 => { // Global section
                const n = try s.u32leb();
                for (0..n) |_| {
                    const vt = try s.byte();
                    _ = try s.byte(); // mutability (0 = const, 1 = var)
                    try global_types.append(allocator, vt);
                    try global_inits.append(allocator, .{ .value = try constExpr(&s) });
                }
            },
            10 => { // Code section
                const n = try s.u32leb();
                for (0..n) |_| {
                    const bs = try s.u32leb();
                    try code_bodies.append(allocator, try s.take(bs));
                }
            },
            11 => { // Data section (active segments only)
                const n = try s.u32leb();
                for (0..n) |_| {
                    const flag = try s.byte();
                    if (flag != 0x00) return error.Unsupported; // only active, memory 0
                    // The init offset is an untrusted sleb: reject negative / >u32.
                    const offset = std.math.cast(u32, try constExpr(&s)) orelse return error.InvalidWasm;
                    const len = try s.u32leb();
                    try data_segs.append(allocator, .{ .offset = offset, .bytes = try s.take(len) });
                }
            },
            else => {}, // skip unhandled sections
        }
    }

    const nfuncs = code_bodies.items.len;
    if (func_type_idx.items.len != nfuncs) return error.InvalidWasm;
    const n_imports: u32 = @intCast(import_names.items.len);

    // Combined function-index -> type-index map: imports first, then defined.
    var all_type_idx: std.ArrayList(u32) = .empty;
    defer all_type_idx.deinit(allocator);
    try all_type_idx.appendSlice(allocator, import_type_idx.items);
    try all_type_idx.appendSlice(allocator, func_type_idx.items);

    const ctx = Ctx{
        .types = types_list.items,
        .func_type_idx = all_type_idx.items,
        .exports = exports.items,
        .has_memory = has_memory,
        .min_pages = min_pages,
        .global_types = global_types.items,
        .has_table = has_table,
        .n_imports = n_imports,
        .has_imports = n_imports > 0,
    };
    const table = try allocator.dupe(u32, table_idx.items);
    errdefer allocator.free(table);
    const imports = try import_names.toOwnedSlice(allocator);
    errdefer {
        for (imports) |s| allocator.free(s);
        allocator.free(imports);
    }

    const globals = try allocator.dupe(GlobalInit, global_inits.items);
    errdefer allocator.free(globals);

    var data_list: std.ArrayList(DataSegment) = .empty;
    errdefer {
        for (data_list.items) |seg| allocator.free(seg.bytes);
        data_list.deinit(allocator);
    }
    for (data_segs.items) |raw| {
        const dup = try allocator.dupe(u8, raw.bytes);
        errdefer allocator.free(dup);
        try data_list.append(allocator, .{ .offset = raw.offset, .bytes = dup });
    }
    const data = try data_list.toOwnedSlice(allocator);
    errdefer {
        for (data) |seg| allocator.free(seg.bytes);
        allocator.free(data);
    }

    var list: std.ArrayList(LoweredFunction) = .empty;
    errdefer {
        for (list.items) |*lf| {
            lf.func.deinit();
            allocator.free(lf.name);
        }
        list.deinit(allocator);
    }
    for (0..nfuncs) |i| {
        // Defined function i has function index n_imports + i (imports occupy the low
        // indices). Exports and call symbols use that combined index.
        const name = try makeName(allocator, ctx.exports, n_imports + @as(u32, @intCast(i)));
        errdefer allocator.free(name);
        const tidx = func_type_idx.items[i];
        if (tidx >= types_list.items.len) return error.InvalidWasm;
        var func = try lowerFunction(allocator, ctx, types_list.items[tidx], code_bodies.items[i]);
        errdefer func.deinit();
        try list.append(allocator, .{ .name = name, .func = func });
    }
    const needs_context = has_memory or global_types.items.len > 0 or has_table or n_imports > 0;
    return .{ .functions = try list.toOwnedSlice(allocator), .has_memory = has_memory, .min_pages = min_pages, .needs_context = needs_context, .globals = globals, .data = data, .table = table, .imports = imports };
}

/// Evaluate a constant init expression (one numeric const, then `end`) to its raw
/// 64-bit slot value.
fn constExpr(s: *Cursor) Error!i64 {
    const op = try s.byte();
    const v: i64 = switch (op) {
        0x41, 0x42 => try s.sleb(), // i32.const / i64.const
        0x43 => @intCast(std.mem.readInt(u32, (try s.take(4))[0..4], .little)), // f32 bits
        0x44 => @bitCast(std.mem.readInt(u64, (try s.take(8))[0..8], .little)), // f64 bits
        else => return error.Unsupported,
    };
    if ((try s.byte()) != 0x0B) return error.InvalidWasm;
    return v;
}

/// Export name for function `i` (duped), or a synthetic `func{i}`. Serves as both the
/// lookup name and the link symbol, so `call` targets resolve to it.
fn makeName(allocator: std.mem.Allocator, exports: []const Export, i: usize) Error![]u8 {
    for (exports) |e| if (e.idx == i) return allocator.dupe(u8, e.name);
    return std.fmt.allocPrint(allocator, "func{d}", .{i});
}

fn irType(func: *Function, valtype: u8) Error!ir.types.Type {
    return switch (valtype) {
        0x7F => func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }), // i32
        0x7E => func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } }), // i64
        0x7D => func.types.intern(.{ .float = .f32 }), // f32
        0x7C => func.types.intern(.{ .float = .f64 }), // f64
        else => error.Unsupported, // reference types come later
    };
}

fn lowerFunction(allocator: std.mem.Allocator, ctx: Ctx, ft: FuncType, body: []const u8) Error!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32s = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64s = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32u = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i64u = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const i8s = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i8u = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const i16s = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 16 } });
    const i16u = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const entry = try func.appendBlock();

    // A module with any hidden state (memory/globals/table/imports) gives every
    // function a single hidden leading "context" pointer to a struct of base pointers
    // at fixed offsets (mem 0, globals 8, table 16, imports 24, import-context 32). The
    // bases are loaded from it at entry. Calls thread the context through; a host-import
    // call also forwards the import-context pointer as the callee's hidden first argument.
    const ptr_t = try func.types.intern(.ptr);
    const needs_ctx = ctx.has_memory or ctx.global_types.len > 0 or ctx.has_table or ctx.has_imports;
    const context: ?Value = if (needs_ctx) try func.appendBlockParam(entry, ptr_t) else null;
    const loadBase = struct {
        fn f(fc: *Function, blk: Block, pt: ir.types.Type, ctxv: Value, off: i64) Error!Value {
            const p = if (off == 0) ctxv else try fc.appendArithImm(blk, pt, .add, ctxv, off);
            return fc.appendInst(blk, pt, .{ .load = .{ .ptr = p } });
        }
    }.f;
    const mem_base: ?Value = if (ctx.has_memory) try loadBase(&func, entry, ptr_t, context.?, 0) else null;
    const globals_base: ?Value = if (ctx.global_types.len > 0) try loadBase(&func, entry, ptr_t, context.?, 8) else null;
    const table_base: ?Value = if (ctx.has_table) try loadBase(&func, entry, ptr_t, context.?, 16) else null;
    const imports_base: ?Value = if (ctx.has_imports) try loadBase(&func, entry, ptr_t, context.?, 24) else null;
    // The opaque import-context pointer host imports receive as their hidden first arg.
    const import_ctx: ?Value = if (ctx.has_imports) try loadBase(&func, entry, ptr_t, context.?, 32) else null;
    var locals: std.ArrayList(Local) = .empty;
    defer locals.deinit(allocator);
    var pvals: std.ArrayList(Value) = .empty;
    defer pvals.deinit(allocator);
    for (ft.params) |vt| try pvals.append(allocator, try func.appendBlockParam(entry, try irType(&func, vt)));
    for (ft.params, 0..) |vt, i| {
        const ty = try irType(&func, vt);
        const slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = ty } });
        try func.appendStore(entry, pvals.items[i], slot);
        try locals.append(allocator, .{ .ptr = slot, .ty = ty });
    }

    var c = Cursor{ .bytes = body };
    const ngroups = try c.u32leb();
    for (0..ngroups) |_| {
        const n = try c.u32leb();
        const ty = try irType(&func, try c.byte());
        const is_float = switch (func.types.type_kind(ty)) {
            .float => true,
            else => false,
        };
        for (0..n) |_| {
            const slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = ty } });
            // Zero-init with a constant of the local's own kind: a float local needs a
            // float zero, not an integer one typed as float (which a backend would
            // materialize in the wrong register file).
            const z = if (is_float)
                try func.appendInst(entry, ty, .{ .fconst = 0 })
            else
                try func.appendInst(entry, ty, .{ .iconst = 0 });
            try func.appendStore(entry, z, slot);
            try locals.append(allocator, .{ .ptr = slot, .ty = ty });
        }
    }

    var stack: std.ArrayList(Value) = .empty;
    defer stack.deinit(allocator);

    var l = L{ .func = &func, .block = entry, .stack = &stack, .allocator = allocator, .bool_as = i32s, .ptr_t = ptr_t, .i32_t = i32s, .mem_base = mem_base, .min_pages = ctx.min_pages, .globals_base = globals_base, .global_types = ctx.global_types, .table_base = table_base, .imports_base = imports_base, .import_ctx = import_ctx, .n_imports = ctx.n_imports, .context = context };

    // Structured control flow. Each block/loop/if pushes a `Frame`. `reachable` tracks
    // whether the current point is live. Dead code after an unconditional branch is not
    // modelled: only `end`/`else` may follow one.
    var control: std.ArrayList(Frame) = .empty;
    defer control.deinit(allocator);
    var reachable = true;
    var dead_depth: usize = 0; // nested block/loop/if depth while skipping unreachable code

    var done = false;
    while (!done) {
        const op = try c.byte();
        // Unreachable code (after br/return/unreachable until the matching end/else): skip it,
        // tracking block nesting so the frame's own end/else still reaches the normal handling.
        if (!reachable) {
            switch (op) {
                0x02, 0x03, 0x04 => {
                    _ = try blockType(&c, &func);
                    dead_depth += 1;
                    continue;
                },
                0x0B => if (dead_depth > 0) {
                    dead_depth -= 1;
                    continue;
                }, // else: this frame's end, handled below
                0x05 => if (dead_depth > 0) continue, // else: this frame's else, handled below
                else => {
                    try skipImmediates(&c, op);
                    continue;
                },
            }
        }
        switch (op) {
            0x00 => reachable = false, // unreachable
            0x01 => {}, // nop
            0x02 => try pushFrame(&l, &control, allocator, .block, try blockType(&c, &func)), // block
            0x03 => try pushFrame(&l, &control, allocator, .loop, try blockType(&c, &func)), // loop
            0x04 => try emitIf(&l, &control, allocator, try blockType(&c, &func)), // if
            0x05 => try emitElse(&l, &control, &reachable), // else
            0x0C => try emitBr(&l, &control, try c.u32leb(), &reachable), // br
            0x0D => try emitBrIf(&l, &control, try c.u32leb()), // br_if
            0x0E => try emitBrTable(&l, &control, &c, &reachable), // br_table
            0x0F => try emitReturn(&l, ft, &reachable), // return
            0x0B => if (control.items.len == 0) {
                done = true;
            } else {
                try endFrame(&l, &control, &reachable);
            },
            0x10 => try call(&l, ctx, allocator, try c.u32leb()), // call
            0x11 => { // call_indirect (type index, then table index)
                const type_idx = try c.u32leb();
                _ = try c.u32leb(); // table index (0)
                try callIndirect(&l, ctx, allocator, type_idx);
            },
            0x1A => _ = stack.pop() orelse return error.InvalidWasm, // drop
            0x1B => try selectOp(&l, null), // select
            0x1C => { // select t (typed): a result-type vector (one entry in MVP)
                const n = try c.u32leb();
                var vt: ?u8 = null;
                for (0..n) |_| vt = try c.byte();
                try selectOp(&l, vt);
            },
            0x23 => try globalGet(&l, try c.u32leb()), // global.get
            0x24 => try globalSet(&l, try c.u32leb()), // global.set
            0x20 => { // local.get
                const lc = try localAt(locals.items, try c.u32leb());
                try stack.append(allocator, try func.appendInst(l.block, lc.ty, .{ .load = .{ .ptr = lc.ptr } }));
            },
            0x21 => { // local.set
                const lc = try localAt(locals.items, try c.u32leb());
                try func.appendStore(l.block, stack.pop() orelse return error.InvalidWasm, lc.ptr);
            },
            0x22 => { // local.tee
                const lc = try localAt(locals.items, try c.u32leb());
                try func.appendStore(l.block, stack.getLastOrNull() orelse return error.InvalidWasm, lc.ptr);
            },
            0x41 => try stack.append(allocator, try func.appendInst(l.block, i32s, .{ .iconst = try c.sleb() })), // i32.const
            0x42 => try stack.append(allocator, try func.appendInst(l.block, i64s, .{ .iconst = try c.sleb() })), // i64.const

            0x45 => try eqz(&l, i32s), // i32.eqz
            0x46 => try cmp(&l, .eq, null), // i32.eq
            0x47 => try cmp(&l, .ne, null), // i32.ne
            0x48 => try cmp(&l, .lt, null), // i32.lt_s
            0x49 => try cmp(&l, .lt, i32u), // i32.lt_u
            0x4A => try cmp(&l, .gt, null), // i32.gt_s
            0x4B => try cmp(&l, .gt, i32u), // i32.gt_u
            0x4C => try cmp(&l, .le, null), // i32.le_s
            0x4D => try cmp(&l, .le, i32u), // i32.le_u
            0x4E => try cmp(&l, .ge, null), // i32.ge_s
            0x4F => try cmp(&l, .ge, i32u), // i32.ge_u

            0x6A => try bin(&l, .add, i32s, null), // i32.add
            0x6B => try bin(&l, .sub, i32s, null), // i32.sub
            0x6C => try bin(&l, .mul, i32s, null), // i32.mul
            0x6D => try bin(&l, .div, i32s, null), // i32.div_s
            0x6E => try bin(&l, .div, i32s, i32u), // i32.div_u
            0x6F => try bin(&l, .rem, i32s, null), // i32.rem_s
            0x70 => try bin(&l, .rem, i32s, i32u), // i32.rem_u
            0x71 => try bin(&l, .bit_and, i32s, null), // i32.and
            0x72 => try bin(&l, .bit_or, i32s, null), // i32.or
            0x73 => try bin(&l, .bit_xor, i32s, null), // i32.xor
            0x74 => try bin(&l, .shl, i32s, null), // i32.shl
            0x75 => try bin(&l, .shr, i32s, null), // i32.shr_s
            0x76 => try bin(&l, .shr, i32s, i32u), // i32.shr_u

            0x50 => try eqz(&l, i64s), // i64.eqz
            0x51 => try cmp(&l, .eq, null), // i64.eq
            0x52 => try cmp(&l, .ne, null), // i64.ne
            0x53 => try cmp(&l, .lt, null), // i64.lt_s
            0x54 => try cmp(&l, .lt, i64u), // i64.lt_u
            0x55 => try cmp(&l, .gt, null), // i64.gt_s
            0x56 => try cmp(&l, .gt, i64u), // i64.gt_u
            0x57 => try cmp(&l, .le, null), // i64.le_s
            0x58 => try cmp(&l, .le, i64u), // i64.le_u
            0x59 => try cmp(&l, .ge, null), // i64.ge_s
            0x5A => try cmp(&l, .ge, i64u), // i64.ge_u

            0x7C => try bin(&l, .add, i64s, null), // i64.add
            0x7D => try bin(&l, .sub, i64s, null), // i64.sub
            0x7E => try bin(&l, .mul, i64s, null), // i64.mul
            0x7F => try bin(&l, .div, i64s, null), // i64.div_s
            0x80 => try bin(&l, .div, i64s, i64u), // i64.div_u
            0x81 => try bin(&l, .rem, i64s, null), // i64.rem_s
            0x82 => try bin(&l, .rem, i64s, i64u), // i64.rem_u
            0x83 => try bin(&l, .bit_and, i64s, null), // i64.and
            0x84 => try bin(&l, .bit_or, i64s, null), // i64.or
            0x85 => try bin(&l, .bit_xor, i64s, null), // i64.xor
            0x86 => try bin(&l, .shl, i64s, null), // i64.shl
            0x87 => try bin(&l, .shr, i64s, null), // i64.shr_s
            0x88 => try bin(&l, .shr, i64s, i64u), // i64.shr_u

            0x77 => try rotate(&l, true, i32s, i32u, 32), // i32.rotl
            0x78 => try rotate(&l, false, i32s, i32u, 32), // i32.rotr
            0x89 => try rotate(&l, true, i64s, i64u, 64), // i64.rotl
            0x8A => try rotate(&l, false, i64s, i64u, 64), // i64.rotr
            0x67 => try emitClz(&l, i32s, i32u, 32), // i32.clz
            0x68 => try emitCtz(&l, i32s, i32u, 32), // i32.ctz
            0x69 => try l.push(try emitPopcount(&l, try l.pop(), i32s, i32u, 32)), // i32.popcnt
            0x79 => try emitClz(&l, i64s, i64u, 64), // i64.clz
            0x7A => try emitCtz(&l, i64s, i64u, 64), // i64.ctz
            0x7B => try l.push(try emitPopcount(&l, try l.pop(), i64s, i64u, 64)), // i64.popcnt
            0xC0 => try signExt(&l, i32s, 24), // i32.extend8_s
            0xC1 => try signExt(&l, i32s, 16), // i32.extend16_s
            0xC2 => try signExt(&l, i64s, 56), // i64.extend8_s
            0xC3 => try signExt(&l, i64s, 48), // i64.extend16_s
            0xC4 => try signExt(&l, i64s, 32), // i64.extend32_s

            0x43 => { // f32.const (4 raw IEEE bytes)
                const v: f32 = @bitCast(std.mem.readInt(u32, (try c.take(4))[0..4], .little));
                try stack.append(allocator, try func.appendInst(l.block, f32_t, .{ .fconst = @floatCast(v) }));
            },
            0x44 => { // f64.const (8 raw IEEE bytes)
                const v: f64 = @bitCast(std.mem.readInt(u64, (try c.take(8))[0..8], .little));
                try stack.append(allocator, try func.appendInst(l.block, f64_t, .{ .fconst = v }));
            },

            0x5B => try cmp(&l, .eq, null), // f32.eq
            0x5C => try cmp(&l, .ne, null), // f32.ne
            0x5D => try cmp(&l, .lt, null), // f32.lt
            0x5E => try cmp(&l, .gt, null), // f32.gt
            0x5F => try cmp(&l, .le, null), // f32.le
            0x60 => try cmp(&l, .ge, null), // f32.ge
            0x61 => try cmp(&l, .eq, null), // f64.eq
            0x62 => try cmp(&l, .ne, null), // f64.ne
            0x63 => try cmp(&l, .lt, null), // f64.lt
            0x64 => try cmp(&l, .gt, null), // f64.gt
            0x65 => try cmp(&l, .le, null), // f64.le
            0x66 => try cmp(&l, .ge, null), // f64.ge

            0x8B => try fabs(&l, f32_t), // f32.abs
            0x99 => try fabs(&l, f64_t), // f64.abs
            0x8C => try fneg(&l, f32_t), // f32.neg
            0x92 => try bin(&l, .add, f32_t, null), // f32.add
            0x93 => try bin(&l, .sub, f32_t, null), // f32.sub
            0x94 => try bin(&l, .mul, f32_t, null), // f32.mul
            0x95 => try bin(&l, .div, f32_t, null), // f32.div
            0x8D => try unaryOp(&l, .ceil, f32_t), // f32.ceil
            0x8E => try unaryOp(&l, .floor, f32_t), // f32.floor
            0x8F => try unaryOp(&l, .trunc, f32_t), // f32.trunc
            0x90 => try unaryOp(&l, .nearest, f32_t), // f32.nearest
            0x91 => try unaryOp(&l, .sqrt, f32_t), // f32.sqrt
            0x96 => try fminmax(&l, false), // f32.min
            0x97 => try fminmax(&l, true), // f32.max
            0x98 => try copysign(&l, f32_t, i32s, 0x80000000), // f32.copysign
            0x9A => try fneg(&l, f64_t), // f64.neg
            0xA0 => try bin(&l, .add, f64_t, null), // f64.add
            0xA1 => try bin(&l, .sub, f64_t, null), // f64.sub
            0xA2 => try bin(&l, .mul, f64_t, null), // f64.mul
            0xA3 => try bin(&l, .div, f64_t, null), // f64.div
            0x9B => try unaryOp(&l, .ceil, f64_t), // f64.ceil
            0x9C => try unaryOp(&l, .floor, f64_t), // f64.floor
            0x9D => try unaryOp(&l, .trunc, f64_t), // f64.trunc
            0x9E => try unaryOp(&l, .nearest, f64_t), // f64.nearest
            0x9F => try unaryOp(&l, .sqrt, f64_t), // f64.sqrt
            0xA4 => try fminmax(&l, false), // f64.min
            0xA5 => try fminmax(&l, true), // f64.max
            0xA6 => try copysign(&l, f64_t, i64s, @bitCast(@as(u64, 0x8000000000000000))), // f64.copysign

            0xA7 => try cvt(&l, i32s), // i32.wrap_i64 (keep low 32)
            0xA8 => try cvt(&l, i32s), // i32.trunc_f32_s
            0xA9 => try cvtFloatToUint(&l, i32u, i32s), // i32.trunc_f32_u
            0xAA => try cvt(&l, i32s), // i32.trunc_f64_s
            0xAB => try cvtFloatToUint(&l, i32u, i32s), // i32.trunc_f64_u
            0xAC => try extendI32S(&l, i64s), // i64.extend_i32_s (sign extend)
            0xAD => try cvt(&l, i64s), // i64.extend_i32_u (zero extend)
            0xAE => try cvt(&l, i64s), // i64.trunc_f32_s
            0xAF => try cvtFloatToUint(&l, i64u, i64s), // i64.trunc_f32_u
            0xB0 => try cvt(&l, i64s), // i64.trunc_f64_s
            0xB1 => try cvtFloatToUint(&l, i64u, i64s), // i64.trunc_f64_u
            0xB2 => try cvt(&l, f32_t), // f32.convert_i32_s
            0xB3 => try cvtUintToFloat(&l, i32u, f32_t), // f32.convert_i32_u
            0xB4 => try cvt(&l, f32_t), // f32.convert_i64_s
            0xB5 => try cvtUintToFloat(&l, i64u, f32_t), // f32.convert_i64_u
            0xB6 => try cvt(&l, f32_t), // f32.demote_f64
            0xB7 => try cvt(&l, f64_t), // f64.convert_i32_s
            0xB8 => try cvtUintToFloat(&l, i32u, f64_t), // f64.convert_i32_u
            0xB9 => try cvt(&l, f64_t), // f64.convert_i64_s
            0xBA => try cvtUintToFloat(&l, i64u, f64_t), // f64.convert_i64_u
            0xBB => try cvt(&l, f64_t), // f64.promote_f32
            0xBC => try unaryOp(&l, .reinterpret, i32s), // i32.reinterpret_f32
            0xBD => try unaryOp(&l, .reinterpret, i64s), // i64.reinterpret_f64
            0xBE => try unaryOp(&l, .reinterpret, f32_t), // f32.reinterpret_i32
            0xBF => try unaryOp(&l, .reinterpret, f64_t), // f64.reinterpret_i64

            0x28 => try memLoad(&l, &c, i32s), // i32.load
            0x29 => try memLoad(&l, &c, i64s), // i64.load
            0x2A => try memLoad(&l, &c, f32_t), // f32.load
            0x2B => try memLoad(&l, &c, f64_t), // f64.load
            0x2C => try memLoadN(&l, &c, i8s, i32s, false), // i32.load8_s
            0x2D => try memLoadN(&l, &c, i8u, i32s, false), // i32.load8_u
            0x2E => try memLoadN(&l, &c, i16s, i32s, false), // i32.load16_s
            0x2F => try memLoadN(&l, &c, i16u, i32s, false), // i32.load16_u
            0x30 => try memLoadN(&l, &c, i8s, i64s, true), // i64.load8_s
            0x31 => try memLoadN(&l, &c, i8u, i64s, false), // i64.load8_u
            0x32 => try memLoadN(&l, &c, i16s, i64s, true), // i64.load16_s
            0x33 => try memLoadN(&l, &c, i16u, i64s, false), // i64.load16_u
            0x34 => try memLoadN(&l, &c, i32s, i64s, true), // i64.load32_s
            0x35 => try memLoadN(&l, &c, i32u, i64s, false), // i64.load32_u
            0x36 => try memStore(&l, &c), // i32.store
            0x37 => try memStore(&l, &c), // i64.store
            0x38 => try memStore(&l, &c), // f32.store
            0x39 => try memStore(&l, &c), // f64.store
            0x3A => try memStoreN(&l, &c, i8s), // i32.store8
            0x3B => try memStoreN(&l, &c, i16s), // i32.store16
            0x3C => try memStoreN(&l, &c, i8s), // i64.store8
            0x3D => try memStoreN(&l, &c, i16s), // i64.store16
            0x3E => try memStoreN(&l, &c, i32s), // i64.store32
            0xFC => { // saturating conversions (aarch64 fcvtzs/zu already saturate)
                switch (try c.u32leb()) {
                    0, 2 => try cvt(&l, i32s), // i32.trunc_sat_f32/64_s
                    1, 3 => try cvtFloatToUint(&l, i32u, i32s), // i32.trunc_sat_f32/64_u
                    4, 6 => try cvt(&l, i64s), // i64.trunc_sat_f32/64_s
                    5, 7 => try cvtFloatToUint(&l, i64u, i64s), // i64.trunc_sat_f32/64_u
                    10 => { // memory.copy (dest memidx, src memidx)
                        _ = try c.byte();
                        _ = try c.byte();
                        try memCopy(&l);
                    },
                    11 => { // memory.fill (memidx)
                        _ = try c.byte();
                        try memFill(&l);
                    },
                    else => return error.Unsupported,
                }
            },
            0x3F => { // memory.size (fixed: the declared minimum)
                _ = try c.byte();
                try stack.append(allocator, try func.appendInst(l.block, i32s, .{ .iconst = l.min_pages }));
            },
            0x40 => { // memory.grow (unsupported growth: returns -1)
                _ = try c.byte();
                _ = stack.pop() orelse return error.InvalidWasm;
                try stack.append(allocator, try func.appendInst(l.block, i32s, .{ .iconst = -1 }));
            },

            else => return error.Unsupported,
        }
    }

    // Terminate the final block (unless an explicit `return` already did). A
    // reachable fall-off returns the top of stack for a result function.
    if (func.terminator(l.block) == null) {
        if (reachable and ft.results.len != 0) {
            func.setTerminator(l.block, .{ .ret = stack.getLastOrNull() orelse return error.InvalidWasm });
        } else {
            func.setTerminator(l.block, .{ .ret = null });
        }
    }
    return func;
}

const Frame = struct {
    kind: enum { block, loop, @"if" },
    /// Where `br` to this frame jumps (a block/if's continuation, a loop's header).
    branch_target: Block,
    /// How many values a `br` to this frame carries (a loop continues with its 0 input
    /// params in MVP, a block/if carries its result arity).
    branch_arity: usize,
    /// Where control continues after the matching `end`.
    cont: Block,
    /// How many values flow into `cont` at the matching `end` (the result arity).
    cont_arity: usize,
    /// `cont`'s block parameter (the result), pushed onto the stack after `end`.
    cont_param: ?Value,
    /// The `if`'s else block, until consumed by `else` (or wired to `cont` at `end`).
    else_block: ?Block,
    /// The operand-stack height when the frame was entered.
    stack_base: usize,
    /// Whether `cont` has at least one predecessor (so it is reachable).
    cont_reached: bool,
};

/// Read a block type: empty (0x40 -> null) or a single value type. Multi-value
/// (type-index) blocks are not supported.
fn blockType(c: *Cursor, func: *Function) Error!?ir.types.Type {
    const bt = try c.byte();
    return switch (bt) {
        0x40 => null,
        0x7F, 0x7E, 0x7D, 0x7C => try irType(func, bt),
        else => error.Unsupported,
    };
}

/// Jump from `from` to `target` carrying the top `arity` operand-stack values.
fn jumpArgs(l: *L, from: Block, target: Block, arity: usize) Error!void {
    if (l.stack.items.len < arity) return error.InvalidWasm;
    try l.func.setJump(from, target, l.stack.items[l.stack.items.len - arity ..]);
}

fn pushFrame(l: *L, control: *std.ArrayList(Frame), allocator: std.mem.Allocator, kind: enum { block, loop }, result_ty: ?ir.types.Type) Error!void {
    const arity: usize = if (result_ty != null) 1 else 0;
    const cont = try l.func.appendBlock();
    const cont_param = if (result_ty) |t| try l.func.appendBlockParam(cont, t) else null;
    if (kind == .loop) {
        const header = try l.func.appendBlock();
        try l.func.setJump(l.block, header, &.{});
        l.block = header;
        try control.append(allocator, .{ .kind = .loop, .branch_target = header, .branch_arity = 0, .cont = cont, .cont_arity = arity, .cont_param = cont_param, .else_block = null, .stack_base = l.stack.items.len, .cont_reached = false });
    } else {
        try control.append(allocator, .{ .kind = .block, .branch_target = cont, .branch_arity = arity, .cont = cont, .cont_arity = arity, .cont_param = cont_param, .else_block = null, .stack_base = l.stack.items.len, .cont_reached = false });
    }
}

fn emitIf(l: *L, control: *std.ArrayList(Frame), allocator: std.mem.Allocator, result_ty: ?ir.types.Type) Error!void {
    const cond = try l.pop(); // i32 condition (nonzero is true, the IR `if` tests it)
    const arity: usize = if (result_ty != null) 1 else 0;
    const then_b = try l.func.appendBlock();
    const else_b = try l.func.appendBlock();
    const cont = try l.func.appendBlock();
    const cont_param = if (result_ty) |t| try l.func.appendBlockParam(cont, t) else null;
    try l.func.appendIf(l.block, cond, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    l.block = then_b;
    try control.append(allocator, .{ .kind = .@"if", .branch_target = cont, .branch_arity = arity, .cont = cont, .cont_arity = arity, .cont_param = cont_param, .else_block = else_b, .stack_base = l.stack.items.len, .cont_reached = false });
}

fn emitElse(l: *L, control: *std.ArrayList(Frame), reachable: *bool) Error!void {
    if (control.items.len == 0) return error.InvalidWasm;
    const f = &control.items[control.items.len - 1];
    if (f.kind != .@"if" or f.else_block == null) return error.InvalidWasm;
    if (reachable.*) {
        try jumpArgs(l, l.block, f.cont, f.cont_arity);
        f.cont_reached = true;
    }
    l.stack.shrinkRetainingCapacity(f.stack_base);
    l.block = f.else_block.?;
    f.else_block = null; // the else side is now current
    reachable.* = true; // the else block is entered from the `if`
}

fn endFrame(l: *L, control: *std.ArrayList(Frame), reachable: *bool) Error!void {
    var f = control.pop().?;
    // An `if` with no `else`: its else edge continues straight to `cont` (only valid
    // for an arity-0 result).
    if (f.kind == .@"if" and f.else_block != null) {
        try l.func.setJump(f.else_block.?, f.cont, &.{});
        f.cont_reached = true;
    }
    if (reachable.*) {
        try jumpArgs(l, l.block, f.cont, f.cont_arity);
        f.cont_reached = true;
    }
    l.stack.shrinkRetainingCapacity(f.stack_base);
    l.block = f.cont;
    if (f.cont_param) |p| try l.push(p); // the block's result is cont's parameter
    reachable.* = f.cont_reached;
}

fn emitBr(l: *L, control: *std.ArrayList(Frame), depth: u32, reachable: *bool) Error!void {
    if (depth >= control.items.len) return error.InvalidWasm;
    const f = &control.items[control.items.len - 1 - depth];
    try jumpArgs(l, l.block, f.branch_target, f.branch_arity);
    if (f.kind != .loop) f.cont_reached = true; // br to a block/if reaches its cont
    reachable.* = false;
}

fn emitBrIf(l: *L, control: *std.ArrayList(Frame), depth: u32) Error!void {
    if (depth >= control.items.len) return error.InvalidWasm;
    const cond = try l.pop(); // i32
    const f = &control.items[control.items.len - 1 - depth];
    const fallthrough = try l.func.appendBlock();
    // The taken edge carries the branch arity. The fall-through keeps the values on
    // the operand stack (valid in the new block by dominance).
    const taken: []const Value = if (f.branch_arity == 0 or l.stack.items.len < f.branch_arity) &.{} else l.stack.items[l.stack.items.len - f.branch_arity ..];
    try l.func.appendIf(l.block, cond, .{ .target = f.branch_target, .args = taken }, .{ .target = fallthrough, .args = &.{} });
    if (f.kind != .loop) f.cont_reached = true;
    l.block = fallthrough;
}

/// `br_table`: pop the index, then branch to `targets[index]` (or `default`),
/// lowered as a chain of equality checks. Only arity-0 labels are supported.
fn emitBrTable(l: *L, control: *std.ArrayList(Frame), c: *Cursor, reachable: *bool) Error!void {
    const count = try c.u32leb();
    var targets: std.ArrayList(u32) = .empty;
    defer targets.deinit(l.allocator);
    for (0..count) |_| try targets.append(l.allocator, try c.u32leb());
    const default = try c.u32leb();

    const idx = try l.pop();
    for (targets.items, 0..) |depth, n| {
        if (depth >= control.items.len) return error.InvalidWasm;
        const f = &control.items[control.items.len - 1 - depth];
        if (f.branch_arity != 0) return error.Unsupported; // value-carrying br_table later
        const k = try l.func.appendInst(l.block, l.i32_t, .{ .iconst = @intCast(n) });
        const cond = try l.func.appendInst(l.block, l.i32_t, .{ .icmp = .{ .op = .eq, .lhs = idx, .rhs = k } });
        const next = try l.func.appendBlock();
        try l.func.appendIf(l.block, cond, .{ .target = f.branch_target, .args = &.{} }, .{ .target = next, .args = &.{} });
        if (f.kind != .loop) f.cont_reached = true;
        l.block = next;
    }
    if (default >= control.items.len) return error.InvalidWasm;
    const df = &control.items[control.items.len - 1 - default];
    if (df.branch_arity != 0) return error.Unsupported;
    try l.func.setJump(l.block, df.branch_target, &.{});
    if (df.kind != .loop) df.cont_reached = true;
    reachable.* = false;
}

fn emitReturn(l: *L, ft: FuncType, reachable: *bool) Error!void {
    if (ft.results.len == 0) {
        l.func.setTerminator(l.block, .{ .ret = null });
    } else {
        l.func.setTerminator(l.block, .{ .ret = l.stack.getLastOrNull() orelse return error.InvalidWasm });
    }
    reachable.* = false;
}

fn localAt(locals: []const Local, idx: u32) Error!Local {
    if (idx >= locals.len) return error.InvalidWasm;
    return locals[idx];
}

/// The lowering context for one function body: the operand stack plus the emit
/// target. `bool_as` is the IR type comparison results take (Wasm compares yield i32
/// 0/1).
const L = struct {
    func: *Function,
    block: Block,
    stack: *std.ArrayList(Value),
    allocator: std.mem.Allocator,
    bool_as: ir.types.Type,
    ptr_t: ir.types.Type,
    i32_t: ir.types.Type = undefined,
    /// The linear-memory base (a hidden leading parameter), if the module has memory.
    mem_base: ?Value = null,
    min_pages: u32 = 0,
    /// The globals base (a hidden parameter after the memory base), if any globals.
    globals_base: ?Value = null,
    global_types: []const u8 = &.{},
    /// The table base (a hidden parameter after the globals base), for call_indirect.
    table_base: ?Value = null,
    /// The imports base (a hidden parameter after the table base): host addresses.
    imports_base: ?Value = null,
    /// The opaque import-context pointer, forwarded to host imports as their hidden
    /// first argument so they can reach per-instance host state.
    import_ctx: ?Value = null,
    n_imports: u32 = 0,
    /// The hidden context pointer (threaded to defined-function calls), if any.
    context: ?Value = null,

    fn pop(self: *L) Error!Value {
        return self.stack.pop() orelse error.InvalidWasm;
    }
    fn push(self: *L, v: Value) Error!void {
        try self.stack.append(self.allocator, v);
    }
    /// Reinterpret `v` as type `ty` (same bits). Used to switch signedness for the
    /// unsigned opcode variants. `x | 0` is a fold the optimizer removes.
    fn coerce(self: *L, v: Value, ty: ir.types.Type) Error!Value {
        return self.func.appendArithImm(self.block, ty, .bit_or, v, 0);
    }
    /// The number of hidden leading parameters (the context pointer: 0 or 1).
    fn hiddenCount(self: *const L) usize {
        return @intFromBool(self.context != null);
    }
    /// Fill the leading slot of `args` with the context pointer, if any.
    fn fillHidden(self: *const L, args: []Value) void {
        if (self.context) |cx| args[0] = cx;
    }
};

/// Pop two operands, emit `a op b` typed `res`, push the result. When `op_ty` is
/// set, both operands are first coerced to it (the unsigned opcode variants).
fn bin(l: *L, op: ir.function.BinOp, res: ir.types.Type, op_ty: ?ir.types.Type) Error!void {
    var b = try l.pop();
    var a = try l.pop();
    if (op_ty) |t| {
        a = try l.coerce(a, t);
        b = try l.coerce(b, t);
    }
    try l.push(try l.func.appendInst(l.block, res, .{ .arith = .{ .op = op, .lhs = a, .rhs = b } }));
}

/// Pop two operands, emit the comparison (result i32 0/1), push it.
fn cmp(l: *L, op: ir.function.CmpOp, op_ty: ?ir.types.Type) Error!void {
    var b = try l.pop();
    var a = try l.pop();
    if (op_ty) |t| {
        a = try l.coerce(a, t);
        b = try l.coerce(b, t);
    }
    try l.push(try l.func.appendInst(l.block, l.bool_as, .{ .icmp = .{ .op = op, .lhs = a, .rhs = b } }));
}

/// `x == 0` as i32 (the `eqz` opcodes). `zero_ty` is the operand's width.
fn eqz(l: *L, zero_ty: ir.types.Type) Error!void {
    const a = try l.pop();
    const z = try l.func.appendInst(l.block, zero_ty, .{ .iconst = 0 });
    try l.push(try l.func.appendInst(l.block, l.bool_as, .{ .icmp = .{ .op = .eq, .lhs = a, .rhs = z } }));
}

/// `select`: pop cond, b, a, then push `cond ? a : b`. The result type is the explicit
/// type (typed select) or the type of the operands.
fn selectOp(l: *L, vt: ?u8) Error!void {
    const cond = try l.pop();
    const b = try l.pop();
    const a = try l.pop();
    const ty = if (vt) |v| try irType(l.func, v) else l.func.valueType(a);
    try l.push(try l.func.appendInst(l.block, ty, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } }));
}

/// A pointer to global `idx`'s 8-byte slot in the globals buffer.
fn globalPtr(l: *L, idx: u32) Error!Value {
    const base = l.globals_base orelse return error.InvalidWasm;
    if (idx >= l.global_types.len) return error.InvalidWasm;
    return l.func.appendArithImm(l.block, l.ptr_t, .add, base, @as(i64, idx) * 8);
}

fn globalGet(l: *L, idx: u32) Error!void {
    if (idx >= l.global_types.len) return error.InvalidWasm;
    const ty = try irType(l.func, l.global_types[idx]);
    const ptr = try globalPtr(l, idx);
    try l.push(try l.func.appendInst(l.block, ty, .{ .load = .{ .ptr = ptr } }));
}

fn globalSet(l: *L, idx: u32) Error!void {
    const ptr = try globalPtr(l, idx);
    try l.func.appendStore(l.block, try l.pop(), ptr);
}

/// The effective address pointer for a memory access: `mem_base + (addr + offset)`.
fn memPtr(l: *L, addr: Value, offset: u32) Error!Value {
    const base = l.mem_base orelse return error.InvalidWasm;
    const a = if (offset == 0) addr else try l.func.appendArithImm(l.block, l.i32_t, .add, addr, @intCast(offset));
    return l.func.appendInst(l.block, l.ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = a } });
}

/// `mem_base + base + index` as a byte pointer (for the bulk-memory loops).
fn byteAddr(l: *L, blk: Block, mem: Value, base: Value, index: Value) Error!Value {
    const off = try l.func.appendInst(blk, l.i32_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = index } });
    return l.func.appendInst(blk, l.ptr_t, .{ .arith = .{ .op = .add, .lhs = mem, .rhs = off } });
}

/// memory.fill: a forward byte loop storing `val`'s low byte into `size` bytes at `dest`.
fn memFill(l: *L) Error!void {
    const i32t = l.i32_t;
    const i32u = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i8t = try l.func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const size = try l.pop();
    const val = try l.pop();
    const dest = try l.pop();
    const mem = l.mem_base orelse return error.InvalidWasm;

    const header = try l.func.appendBlock();
    const iv = try l.func.appendBlockParam(header, i32t);
    const body = try l.func.appendBlock();
    const jv = try l.func.appendBlockParam(body, i32t);
    const cont = try l.func.appendBlock();

    const zero = try l.func.appendInst(l.block, i32t, .{ .iconst = 0 });
    try l.func.setJump(l.block, header, &.{zero});

    l.block = header;
    const cond = try l.func.appendInst(header, l.bool_as, .{ .icmp = .{ .op = .lt, .lhs = try l.coerce(iv, i32u), .rhs = try l.coerce(size, i32u) } });
    try l.func.appendIf(header, cond, .{ .target = body, .args = &.{iv} }, .{ .target = cont, .args = &.{} });

    l.block = body;
    const addr = try byteAddr(l, body, mem, dest, jv);
    try l.func.appendStore(body, try l.coerce(val, i8t), addr);
    const next = try l.func.appendArithImm(body, i32t, .add, jv, 1);
    try l.func.setJump(body, header, &.{next});

    l.block = cont;
}

/// memory.copy: a forward byte loop copying `size` bytes from `src` to `dest`. Forward-only,
/// correct for disjoint regions and dest <= src. Overlapping dest > src is not handled yet
/// (it would need a backward pass).
fn memCopy(l: *L) Error!void {
    const i32t = l.i32_t;
    const i32u = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i8u = try l.func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const size = try l.pop();
    const src = try l.pop();
    const dest = try l.pop();
    const mem = l.mem_base orelse return error.InvalidWasm;

    const header = try l.func.appendBlock();
    const iv = try l.func.appendBlockParam(header, i32t);
    const body = try l.func.appendBlock();
    const jv = try l.func.appendBlockParam(body, i32t);
    const cont = try l.func.appendBlock();

    const zero = try l.func.appendInst(l.block, i32t, .{ .iconst = 0 });
    try l.func.setJump(l.block, header, &.{zero});

    l.block = header;
    const cond = try l.func.appendInst(header, l.bool_as, .{ .icmp = .{ .op = .lt, .lhs = try l.coerce(iv, i32u), .rhs = try l.coerce(size, i32u) } });
    try l.func.appendIf(header, cond, .{ .target = body, .args = &.{iv} }, .{ .target = cont, .args = &.{} });

    l.block = body;
    const saddr = try byteAddr(l, body, mem, src, jv);
    const byte = try l.func.appendInst(body, i8u, .{ .load = .{ .ptr = saddr } });
    const daddr = try byteAddr(l, body, mem, dest, jv);
    try l.func.appendStore(body, byte, daddr);
    const next = try l.func.appendArithImm(body, i32t, .add, jv, 1);
    try l.func.setJump(body, header, &.{next});

    l.block = cont;
}

/// Advance past an opcode's immediates without decoding it, to skip unreachable code. The
/// caller handles the control structure (block/loop/if/end/else). Everything else lands here.
fn skipImmediates(c: *Cursor, op: u8) Error!void {
    switch (op) {
        0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24 => _ = try c.u32leb(), // br/br_if/call/local/global
        0x11 => { // call_indirect: type index, table index
            _ = try c.u32leb();
            _ = try c.u32leb();
        },
        0x41, 0x42 => _ = try c.sleb(), // i32/i64.const
        0x43 => _ = try c.take(4), // f32.const
        0x44 => _ = try c.take(8), // f64.const
        0x28...0x3E => { // loads/stores: align, offset
            _ = try c.u32leb();
            _ = try c.u32leb();
        },
        0x3F, 0x40 => _ = try c.byte(), // memory.size/grow: memidx
        0x0E => { // br_table: label vector then default
            const n = try c.u32leb();
            for (0..n + 1) |_| _ = try c.u32leb();
        },
        0x1C => { // select t: result-type vector
            const n = try c.u32leb();
            for (0..n) |_| _ = try c.byte();
        },
        0xFC => {
            switch (try c.u32leb()) {
                8, 10, 12, 14 => { // memory.init / memory.copy / table.init / table.copy: two indices
                    _ = try c.u32leb();
                    _ = try c.u32leb();
                },
                9, 11, 13, 15, 16, 17 => _ = try c.u32leb(), // *.drop / memory.fill / table.grow/size/fill
                else => {}, // trunc_sat (0..7): no further immediate
            }
        },
        else => {}, // arithmetic/compare/convert/drop/select/nop/unreachable/return: no immediate
    }
}

/// A load of `ty` from linear memory (memarg = alignment hint then offset).
fn memLoad(l: *L, c: *Cursor, ty: ir.types.Type) Error!void {
    _ = try c.u32leb(); // alignment hint (ignored)
    const offset = try c.u32leb();
    const addr = try l.pop();
    const ptr = try memPtr(l, addr, offset);
    try l.push(try l.func.appendInst(l.block, ty, .{ .load = .{ .ptr = ptr } }));
}

/// A store to linear memory (the value's type drives the width).
fn memStore(l: *L, c: *Cursor) Error!void {
    _ = try c.u32leb(); // alignment hint (ignored)
    const offset = try c.u32leb();
    const value = try l.pop();
    const addr = try l.pop();
    const ptr = try memPtr(l, addr, offset);
    try l.func.appendStore(l.block, value, ptr);
}

/// A narrow load (load8/16/32): load the `narrow` width (the backend sign/zero-
/// extends per its signedness), then widen to `canonical`. `sext64` does the extra
/// 32->64 sign extension for the signed i64 variants.
fn memLoadN(l: *L, c: *Cursor, narrow: ir.types.Type, canonical: ir.types.Type, sext64: bool) Error!void {
    _ = try c.u32leb();
    const offset = try c.u32leb();
    const addr = try l.pop();
    const ptr = try memPtr(l, addr, offset);
    const v = try l.func.appendInst(l.block, narrow, .{ .load = .{ .ptr = ptr } });
    if (sext64) {
        const w = try l.func.appendInst(l.block, canonical, .{ .convert = .{ .value = v } });
        const hi = try l.func.appendArithImm(l.block, canonical, .shl, w, 32);
        try l.push(try l.func.appendArithImm(l.block, canonical, .shr, hi, 32));
    } else {
        try l.push(try l.coerce(v, canonical));
    }
}

/// A narrow store (store8/16/32): reinterpret the value at the `narrow` width so the
/// backend stores only those low bytes.
fn memStoreN(l: *L, c: *Cursor, narrow: ir.types.Type) Error!void {
    _ = try c.u32leb();
    const offset = try c.u32leb();
    const value = try l.pop();
    const addr = try l.pop();
    const ptr = try memPtr(l, addr, offset);
    try l.func.appendStore(l.block, try l.coerce(value, narrow), ptr);
}

/// A single-operand op (float rounding/sqrt, or a bit reinterpret) producing `ty`.
fn unaryOp(l: *L, op: ir.function.UnaryOp, ty: ir.types.Type) Error!void {
    const a = try l.pop();
    try l.push(try l.func.appendInst(l.block, ty, .{ .unary = .{ .op = op, .value = a } }));
}

/// copysign(x, y) = (x & ~signbit) | (y & signbit), done on the integer bits.
fn copysign(l: *L, fty: ir.types.Type, ity: ir.types.Type, signbit: i64) Error!void {
    const y = try l.pop();
    const x = try l.pop();
    const xi = try l.func.appendInst(l.block, ity, .{ .unary = .{ .op = .reinterpret, .value = x } });
    const yi = try l.func.appendInst(l.block, ity, .{ .unary = .{ .op = .reinterpret, .value = y } });
    const xpart = try andConst(l, xi, ~signbit, ity);
    const ypart = try andConst(l, yi, signbit, ity);
    const bits = try l.func.appendInst(l.block, ity, .{ .arith = .{ .op = .bit_or, .lhs = xpart, .rhs = ypart } });
    try l.push(try l.func.appendInst(l.block, fty, .{ .unary = .{ .op = .reinterpret, .value = bits } }));
}

/// Float absolute value: `(x < 0) ? -x : x` (no dedicated fabs in the backend).
fn fabs(l: *L, ty: ir.types.Type) Error!void {
    const a = try l.pop();
    const z = try l.func.appendInst(l.block, ty, .{ .fconst = 0 });
    const neg = try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .sub, .lhs = z, .rhs = a } });
    const cond = try l.func.appendInst(l.block, l.i32_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = z } });
    try l.push(try l.func.appendInst(l.block, ty, .{ .select = .{ .cond = cond, .then = neg, .@"else" = a } }));
}

/// Float negate: `0 - x` (the backend lacks a dedicated fneg, subtract from zero).
fn fneg(l: *L, ty: ir.types.Type) Error!void {
    const a = try l.pop();
    const z = try l.func.appendInst(l.block, ty, .{ .fconst = 0 });
    try l.push(try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .sub, .lhs = z, .rhs = a } }));
}

/// Float min/max as a compare + select. (NaN follows select semantics rather than
/// Wasm's NaN propagation, exact NaN handling needs a dedicated op.)
fn fminmax(l: *L, max: bool) Error!void {
    const b = try l.pop();
    const a = try l.pop();
    const ty = l.func.valueType(a);
    const cond = try l.func.appendInst(l.block, l.i32_t, .{ .icmp = .{ .op = if (max) .gt else .lt, .lhs = a, .rhs = b } });
    try l.push(try l.func.appendInst(l.block, ty, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } }));
}

fn andConst(l: *L, v: Value, c: i64, uty: ir.types.Type) Error!Value {
    const k = try l.func.appendInst(l.block, uty, .{ .iconst = c });
    return l.func.appendInst(l.block, uty, .{ .arith = .{ .op = .bit_and, .lhs = v, .rhs = k } });
}

/// Population count via the SWAR bit-twiddling algorithm (no dedicated opcode). All
/// in the unsigned type so the shifts are logical.
fn emitPopcount(l: *L, x: Value, ty: ir.types.Type, uty: ir.types.Type, comptime W: comptime_int) Error!Value {
    const m1: i64 = if (W == 32) 0x55555555 else @bitCast(@as(u64, 0x5555555555555555));
    const m2: i64 = if (W == 32) 0x33333333 else @bitCast(@as(u64, 0x3333333333333333));
    const m4: i64 = if (W == 32) 0x0F0F0F0F else @bitCast(@as(u64, 0x0F0F0F0F0F0F0F0F));
    const h: i64 = if (W == 32) 0x01010101 else @bitCast(@as(u64, 0x0101010101010101));
    const sh: i64 = if (W == 32) 24 else 56;
    var v = try l.coerce(x, uty);
    {
        const s = try l.func.appendArithImm(l.block, uty, .shr, v, 1);
        const a = try andConst(l, s, m1, uty);
        v = try l.func.appendInst(l.block, uty, .{ .arith = .{ .op = .sub, .lhs = v, .rhs = a } });
    }
    {
        const lo = try andConst(l, v, m2, uty);
        const s = try l.func.appendArithImm(l.block, uty, .shr, v, 2);
        const hi = try andConst(l, s, m2, uty);
        v = try l.func.appendInst(l.block, uty, .{ .arith = .{ .op = .add, .lhs = lo, .rhs = hi } });
    }
    {
        const s = try l.func.appendArithImm(l.block, uty, .shr, v, 4);
        const sum = try l.func.appendInst(l.block, uty, .{ .arith = .{ .op = .add, .lhs = v, .rhs = s } });
        v = try andConst(l, sum, m4, uty);
    }
    const hc = try l.func.appendInst(l.block, uty, .{ .iconst = h });
    const mul = try l.func.appendInst(l.block, uty, .{ .arith = .{ .op = .mul, .lhs = v, .rhs = hc } });
    v = try l.func.appendArithImm(l.block, uty, .shr, mul, sh);
    return l.coerce(v, ty);
}

/// Count trailing zeros: popcount((x & -x) - 1). x == 0 yields W (correct).
fn emitCtz(l: *L, ty: ir.types.Type, uty: ir.types.Type, comptime W: comptime_int) Error!void {
    const x = try l.pop();
    const z = try l.func.appendInst(l.block, ty, .{ .iconst = 0 });
    const neg = try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .sub, .lhs = z, .rhs = x } });
    const low = try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .bit_and, .lhs = x, .rhs = neg } });
    const m = try l.func.appendArithImm(l.block, ty, .sub, low, 1);
    try l.push(try emitPopcount(l, m, ty, uty, W));
}

/// Count leading zeros: W - popcount(smear-MSB-down(x)). x == 0 yields W.
fn emitClz(l: *L, ty: ir.types.Type, uty: ir.types.Type, comptime W: comptime_int) Error!void {
    const x = try l.pop();
    var v = try l.coerce(x, uty);
    var sh: i64 = 1;
    while (sh < W) : (sh *= 2) {
        const s = try l.func.appendArithImm(l.block, uty, .shr, v, sh);
        v = try l.func.appendInst(l.block, uty, .{ .arith = .{ .op = .bit_or, .lhs = v, .rhs = s } });
    }
    const pc = try emitPopcount(l, v, ty, uty, W);
    const wc = try l.func.appendInst(l.block, ty, .{ .iconst = W });
    try l.push(try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .sub, .lhs = wc, .rhs = pc } }));
}

/// Bit rotate via shifts: rotl(x,n) = (x << n) | (x >>u (w-n)), rotr swaps them. The
/// backend masks shift counts mod w, so n == 0 yields x unchanged.
fn rotate(l: *L, left: bool, ty: ir.types.Type, uty: ir.types.Type, width: i64) Error!void {
    const n = try l.pop();
    const x = try l.pop();
    const w = try l.func.appendInst(l.block, ty, .{ .iconst = width });
    const wn = try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .sub, .lhs = w, .rhs = n } });
    const xu = try l.coerce(x, uty);
    const shifted_by_n = try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = if (left) .shl else .shr, .lhs = if (left) x else xu, .rhs = n } });
    const shifted_by_wn = try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = if (left) .shr else .shl, .lhs = if (left) xu else x, .rhs = wn } });
    try l.push(try l.func.appendInst(l.block, ty, .{ .arith = .{ .op = .bit_or, .lhs = shifted_by_n, .rhs = shifted_by_wn } }));
}

/// Sign-extend the low bits of an integer: `(x << s) >> s` arithmetically.
fn signExt(l: *L, ty: ir.types.Type, shift: i64) Error!void {
    const x = try l.pop();
    const hi = try l.func.appendArithImm(l.block, ty, .shl, x, shift);
    try l.push(try l.func.appendArithImm(l.block, ty, .shr, hi, shift));
}

/// A numeric conversion (int<->float, float<->float, or int truncation) to `dst`.
fn cvt(l: *L, dst: ir.types.Type) Error!void {
    const a = try l.pop();
    try l.push(try l.func.appendInst(l.block, dst, .{ .convert = .{ .value = a } }));
}

/// Float to unsigned integer (`trunc_*_u`): convert with an unsigned result so the
/// backend uses fcvtzu, then keep the value canonically signed.
fn cvtFloatToUint(l: *L, uint_ty: ir.types.Type, canonical: ir.types.Type) Error!void {
    const a = try l.pop();
    const r = try l.func.appendInst(l.block, uint_ty, .{ .convert = .{ .value = a } });
    try l.push(try l.coerce(r, canonical));
}

/// Unsigned integer to float (`convert_*_u`): coerce the source to unsigned so the
/// backend uses ucvtf, then convert.
fn cvtUintToFloat(l: *L, src_u: ir.types.Type, dst: ir.types.Type) Error!void {
    const a = try l.pop();
    const ua = try l.coerce(a, src_u);
    try l.push(try l.func.appendInst(l.block, dst, .{ .convert = .{ .value = ua } }));
}

/// Sign-extend an i32 to i64 (`i64.extend_i32_s`): widen then `<< 32 >> 32` (the
/// arithmetic right shift carries the sign).
fn extendI32S(l: *L, i64s: ir.types.Type) Error!void {
    const a = try l.pop();
    const w = try l.func.appendInst(l.block, i64s, .{ .convert = .{ .value = a } });
    const hi = try l.func.appendArithImm(l.block, i64s, .shl, w, 32);
    try l.push(try l.func.appendArithImm(l.block, i64s, .shr, hi, 32));
}

/// Lower `call_indirect`: pop the table index, load the function address from
/// `table_base[index*8]`, then call it through that address (threading the hidden
/// params, which the in-table callees also take).
fn callIndirect(l: *L, ctx: Ctx, allocator: std.mem.Allocator, type_idx: u32) Error!void {
    if (type_idx >= ctx.types.len) return error.InvalidWasm;
    const sig = ctx.types[type_idx];
    const base = l.table_base orelse return error.InvalidWasm;

    const elem = try l.pop();
    const scaled = try l.func.appendArithImm(l.block, l.i32_t, .mul, elem, 8);
    const slot = try l.func.appendInst(l.block, l.ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = scaled } });
    const target = try l.func.appendInst(l.block, l.ptr_t, .{ .load = .{ .ptr = slot } });

    const nargs = sig.params.len;
    const extra = l.hiddenCount();
    const args = try allocator.alloc(Value, nargs + extra);
    defer allocator.free(args);
    l.fillHidden(args);
    var k = nargs;
    while (k > 0) {
        k -= 1;
        args[extra + k] = try l.pop();
    }

    const rty = if (sig.results.len == 0) l.i32_t else try irType(l.func, sig.results[0]);
    const r = try l.func.appendCallIndirect(l.block, rty, target, args);
    if (sig.results.len != 0) try l.push(r);
}

/// Lower a `call` to function `idx`: pop its arguments (top of stack is the last
/// argument), emit a call to its symbol, and push the result (if it returns one).
fn call(l: *L, ctx: Ctx, allocator: std.mem.Allocator, idx: u32) Error!void {
    if (idx >= ctx.func_type_idx.len) return error.InvalidWasm;
    const sig = ctx.types[ctx.func_type_idx[idx]];
    const nargs = sig.params.len;
    if (l.stack.items.len < nargs) return error.InvalidWasm;

    // A call to an imported function dispatches through the imports-base buffer of host
    // addresses. The host function receives the instance's import-context pointer as a
    // hidden first argument (so it can reach per-instance host state without a global),
    // followed by the Wasm arguments.
    if (idx < ctx.n_imports) {
        const base = l.imports_base orelse return error.InvalidWasm;
        const slot = try l.func.appendArithImm(l.block, l.ptr_t, .add, base, @as(i64, idx) * 8);
        const target = try l.func.appendInst(l.block, l.ptr_t, .{ .load = .{ .ptr = slot } });
        const host_ctx = l.import_ctx orelse return error.InvalidWasm;
        const args = try allocator.alloc(Value, nargs + 1);
        defer allocator.free(args);
        args[0] = host_ctx;
        var ki = nargs;
        while (ki > 0) {
            ki -= 1;
            args[1 + ki] = try l.pop();
        }
        const rty = if (sig.results.len == 0) l.i32_t else try irType(l.func, sig.results[0]);
        const r = try l.func.appendCallIndirect(l.block, rty, target, args);
        if (sig.results.len != 0) try l.push(r);
        return;
    }

    // A direct call to a defined function threads the hidden leading arguments
    // (memory, globals, table, imports base) through.
    const extra = l.hiddenCount();
    const args = try allocator.alloc(Value, nargs + extra);
    defer allocator.free(args);
    l.fillHidden(args);
    var k = nargs;
    while (k > 0) {
        k -= 1;
        args[extra + k] = try l.pop();
    }

    const name = try makeName(allocator, ctx.exports, idx);
    defer allocator.free(name);
    if (sig.results.len == 0) {
        try l.func.appendVoidCall(l.block, name, args);
    } else {
        const rty = try irType(l.func, sig.results[0]);
        try l.push(try l.func.appendCall(l.block, rty, name, args));
    }
}

/// Wrap `content` (a single section body) in a minimal valid module header + section
/// header so `module()` reaches the section-parsing code under test.
fn oneSectionModule(comptime id: u8, comptime content: []const u8) [8 + 2 + content.len]u8 {
    return reader.magic ++ reader.version ++ [_]u8{ id, content.len } ++ content[0..content.len].*;
}

test "element segment with negative init offset is rejected, not a panic" {
    // Element section (id 9): n=1, flag=0x00, init expr `i32.const -1` (0x41 0x7F 0x0B),
    // cnt=1, funcidx=0. The -1 offset must be rejected by the u32 cast, not @intCast-panic.
    const content = [_]u8{ 0x01, 0x00, 0x41, 0x7F, 0x0B, 0x01, 0x00 };
    const bytes = oneSectionModule(9, &content);
    try std.testing.expectError(error.InvalidWasm, module(std.testing.allocator, &bytes));
}

test "element segment with over-cap init offset is rejected, not an OOM" {
    // Same shape but init expr `i32.const 2097152` (> max_table_entries): the slot cap must
    // reject it before the table-growth loop attempts a multi-MB/GB allocation.
    const content = [_]u8{ 0x01, 0x00, 0x41, 0x80, 0x80, 0x80, 0x01, 0x0B, 0x01, 0x00 };
    const bytes = oneSectionModule(9, &content);
    try std.testing.expectError(error.InvalidWasm, module(std.testing.allocator, &bytes));
}
