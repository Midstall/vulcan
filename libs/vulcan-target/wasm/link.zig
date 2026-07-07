//! Wasm module linking: compile a set of named functions, resolve intra-module
//! calls, and produce a linked Wasm module with a symbol table.
//!
//! Unlike native targets, Wasm linking is simpler: it only resolves function
//! call indices within the module. There are no relocations in the traditional
//! sense, Wasm uses direct function indices.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const encode = @import("encode.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Terminator = ir.function.Terminator;

pub const Error = isel.Error;

/// A named function gathered for compilation. The module borrows the function.
pub const Entry = struct { name: []const u8, func: *const Function };

/// A grouping of named functions compiled into a Wasm module.
pub const Module = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,

    pub fn init(allocator: std.mem.Allocator) Module {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Module) void {
        self.entries.deinit(self.allocator);
    }

    pub fn addFunction(self: *Module, name: []const u8, func: *const Function) std.mem.Allocator.Error!void {
        try self.entries.append(self.allocator, .{ .name = name, .func = func });
    }
};

/// A symbol's resolved location in the Wasm module.
pub const Symbol = struct { name: []const u8, index: u32 };

/// A linked Wasm module: the raw bytecode blob and a symbol table.
pub const Linked = struct {
    /// The complete Wasm module as bytes (all sections concatenated).
    module: []u8,
    /// Function symbols: index into the module's function table.
    symbols: []Symbol,
    /// Number of imports (functions defined outside this module).
    import_count: u32,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.module);
        allocator.free(self.symbols);
    }

    pub fn symbolIndex(self: *const Linked, name: []const u8) ?u32 {
        for (self.symbols) |s| {
            if (std.mem.eql(u8, s.name, name)) return s.index;
        }
        return null;
    }
};

/// A Wasm function type signature (shared with isel, which resolves call_indirect
/// type indices against the deduplicated type list this module builds).
const Signature = isel.Signature;

/// Intern `sig` into `unique` (deduplicating by params+results) and return its
/// index. The interned copy is owned by `unique`, the caller keeps `sig`.
fn internSignature(allocator: std.mem.Allocator, unique: *std.ArrayList(Signature), sig: Signature) std.mem.Allocator.Error!u32 {
    for (unique.items, 0..) |u, i| {
        if (std.mem.eql(encode.ValType, sig.params, u.params) and
            std.mem.eql(encode.ValType, sig.results, u.results)) return @intCast(i);
    }
    const idx: u32 = @intCast(unique.items.len);
    try unique.append(allocator, .{
        .params = try allocator.dupe(encode.ValType, sig.params),
        .results = try allocator.dupe(encode.ValType, sig.results),
    });
    return idx;
}

/// Compute the callee signature of a `call_indirect` instruction (its argument
/// types map to params, its result type to a single result).
fn indirectSignature(allocator: std.mem.Allocator, func: *const Function, args: ir.function.ValueList, result: ?Value) Error!Signature {
    const types = &func.types;
    var params = std.ArrayList(encode.ValType).empty;
    errdefer params.deinit(allocator);
    for (func.valueList(args)) |arg| {
        try params.append(allocator, encode.irTypeToWasm(types, func.valueType(arg)) orelse return error.Unsupported);
    }
    var results = std.ArrayList(encode.ValType).empty;
    errdefer results.deinit(allocator);
    if (result) |rv| {
        try results.append(allocator, encode.irTypeToWasm(types, func.valueType(rv)) orelse return error.Unsupported);
    }
    return .{
        .params = try params.toOwnedSlice(allocator),
        .results = try results.toOwnedSlice(allocator),
    };
}

/// Compute the Wasm signature for an IR function.
fn computeSignature(func: *const Function) Error!Signature {
    const types = &func.types;

    // Parameters are the entry block's block parameters.
    var params = std.ArrayList(encode.ValType).empty;
    errdefer params.deinit(func.allocator);

    const entry_block: Block = @enumFromInt(0);
    for (func.blockParams(entry_block)) |param| {
        const ty = func.valueType(param);
        if (encode.irTypeToWasm(types, ty)) |vt| {
            try params.append(func.allocator, vt);
        } else {
            return error.Unsupported;
        }
    }

    // Results are determined by the return value type.
    var results = std.ArrayList(encode.ValType).empty;
    errdefer results.deinit(func.allocator);

    // Find a return terminator in any block.
    var ret_value: ?Value = null;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        if (func.terminator(block)) |t| {
            switch (t) {
                .ret => |rv| {
                    ret_value = rv;
                    break;
                },
                else => {},
            }
            if (ret_value) |_| break;
        }
    }

    if (ret_value) |rv| {
        const ty = func.valueType(rv);
        if (encode.irTypeToWasm(types, ty)) |vt| {
            try results.append(func.allocator, vt);
        } else {
            return error.Unsupported;
        }
    }

    return .{
        .params = try params.toOwnedSlice(func.allocator),
        .results = try results.toOwnedSlice(func.allocator),
    };
}

/// Compile and link every function in `module` into a Wasm binary.
pub fn compileModule(allocator: std.mem.Allocator, module: *const Module) Error!Linked {
    var symbols = std.ArrayList(Symbol).empty;
    errdefer symbols.deinit(allocator);

    // Build the deduplicated type section first: entry-function signatures, then
    // any call_indirect callee signatures. Doing this before isel gives every
    // indirect call a stable type index to name.
    var unique_types = std.ArrayList(Signature).empty;
    defer {
        for (unique_types.items) |s| {
            allocator.free(s.params);
            allocator.free(s.results);
        }
        unique_types.deinit(allocator);
    }

    var type_indices = try allocator.alloc(u32, module.entries.items.len);
    defer allocator.free(type_indices);

    for (module.entries.items, 0..) |entry, i| {
        const sig = try computeSignature(entry.func);
        defer {
            allocator.free(sig.params);
            allocator.free(sig.results);
        }
        type_indices[i] = try internSignature(allocator, &unique_types, sig);
        try symbols.append(allocator, .{ .name = entry.name, .index = @intCast(i) });
    }

    for (module.entries.items) |entry| {
        for (0..entry.func.blockCount()) |bi| {
            const block: Block = @enumFromInt(bi);
            for (entry.func.blockInsts(block)) |inst| {
                switch (entry.func.opcode(inst)) {
                    .call_indirect => |ci| {
                        const sig = try indirectSignature(allocator, entry.func, ci.args, entry.func.instResult(inst));
                        defer {
                            allocator.free(sig.params);
                            allocator.free(sig.results);
                        }
                        _ = try internSignature(allocator, &unique_types, sig);
                    },
                    else => {},
                }
            }
        }
    }

    // If any function allocates, the module gets a mutable i32 stack-pointer global
    // (index 0) that framed functions carve their allocas from.
    var needs_stack = false;
    for (module.entries.items) |entry| {
        for (0..entry.func.blockCount()) |bi| {
            for (entry.func.blockInsts(@enumFromInt(bi))) |inst| {
                if (entry.func.opcode(inst) == .alloca) needs_stack = true;
            }
        }
    }

    // Compile each function body against the finalized type list and function names.
    const func_names = try allocator.alloc([]const u8, module.entries.items.len);
    defer allocator.free(func_names);
    for (module.entries.items, 0..) |entry, i| func_names[i] = entry.name;
    const resolver = isel.ModuleResolver{
        .sigs = unique_types.items,
        .func_names = func_names,
        .sp_global = if (needs_stack) 0 else null,
    };

    var functions = std.ArrayList([]u8).empty;
    errdefer {
        for (functions.items) |code| allocator.free(code);
        functions.deinit(allocator);
    }
    for (module.entries.items) |entry| {
        const compiled = try isel.selectFunction(allocator, entry.func, &resolver);
        try functions.append(allocator, compiled.code);
    }

    const result = try buildWasmModule(allocator, functions.items, unique_types.items, type_indices, symbols.items, needs_stack);

    for (functions.items) |code| allocator.free(code);
    functions.deinit(allocator);
    symbols.deinit(allocator);

    return result;
}

/// Reserve a fixed-width 5-byte LEB128 slot for a section length, returning its
/// offset. Fixed width means the length can be backpatched without shifting the
/// bytes that follow. Any decoder that does not require minimal LEB (the Wasm spec
/// permits non-minimal) accepts it.
fn reserveSectionLen(buf: *std.ArrayList(u8), allocator: std.mem.Allocator) std.mem.Allocator.Error!usize {
    const at = buf.items.len;
    try buf.appendSlice(allocator, &[_]u8{ 0x80, 0x80, 0x80, 0x80, 0x00 });
    return at;
}

/// Backpatch the 5-byte length slot at `at` with `len`.
fn patchSectionLen(buf: *std.ArrayList(u8), at: usize, len: u32) void {
    var v = len;
    for (0..5) |i| {
        var b: u8 = @intCast(v & 0x7F);
        if (i < 4) b |= 0x80;
        buf.items[at + i] = b;
        v >>= 7;
    }
}

/// Build a complete Wasm binary module from compiled function bodies. `unique_types`
/// is the deduplicated type section and `type_indices[i]` names the type of
/// function `i` (parallel to `functions`).
fn buildWasmModule(
    allocator: std.mem.Allocator,
    functions: []const []u8,
    unique_types: []const Signature,
    type_indices: []const u32,
    symbols: []const Symbol,
    needs_stack: bool,
) Error!Linked {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    var leb_buf: [10]u8 = undefined;

    // Wasm magic header and version.
    try buf.appendSlice(allocator, &[_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 });

    const n_funcs: u32 = @intCast(functions.len);

    // Type section (id=1)
    {
        try buf.append(allocator, 0x01); // type section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;

        const n_types: u32 = @intCast(unique_types.len);
        {
            const n = encode.encodeU32leb(&leb_buf, n_types);
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }

        for (unique_types) |sig| {
            try buf.append(allocator, 0x60); // function type

            {
                const n = encode.encodeU32leb(&leb_buf, @intCast(sig.params.len));
                try buf.appendSlice(allocator, leb_buf[0..n]);
            }
            for (sig.params) |pt| {
                try buf.append(allocator, pt.toByte());
            }

            {
                const n = encode.encodeU32leb(&leb_buf, @intCast(sig.results.len));
                try buf.appendSlice(allocator, leb_buf[0..n]);
            }
            for (sig.results) |rt| {
                try buf.append(allocator, rt.toByte());
            }
        }

        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Function section (id=3)
    {
        try buf.append(allocator, 0x03); // function section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;
        {
            const n = encode.encodeU32leb(&leb_buf, n_funcs);
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }

        for (type_indices) |type_idx| {
            {
                const n = encode.encodeU32leb(&leb_buf, type_idx);
                try buf.appendSlice(allocator, leb_buf[0..n]);
            }
        }

        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Sections must appear in ascending id order per the Wasm spec. Table (4),
    // Memory (5), Export (7), Element (9), Code (10) follow the Function section.

    // Table section (id=4): every function is placed in one funcref table at its own
    // index, so a `call_indirect` target value is just the callee's function index.
    if (n_funcs > 0) {
        try buf.append(allocator, 0x04); // table section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;
        try buf.append(allocator, 0x01); // one table
        try buf.append(allocator, 0x70); // funcref
        try buf.append(allocator, 0x00); // limits: min only
        {
            const n = encode.encodeU32leb(&leb_buf, n_funcs);
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }
        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Memory section (id=5): one memory of one page, used for alloca storage.
    {
        try buf.append(allocator, 0x05); // memory section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;

        try buf.append(allocator, 0x01); // one memory
        try buf.append(allocator, 0x00); // limits: min only
        try buf.append(allocator, 0x01); // min pages: 1

        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Global section (id=6): a mutable i32 stack pointer initialized to the top of the
    // one-page memory, growing down. Only present when a function allocates.
    if (needs_stack) {
        try buf.append(allocator, 0x06); // global section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;

        try buf.append(allocator, 0x01); // one global
        try buf.append(allocator, 0x7F); // i32
        try buf.append(allocator, 0x01); // mutable
        try buf.append(allocator, 0x41); // i32.const
        {
            const n = encode.encodeS32leb(&leb_buf, 65536); // one page = stack top
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }
        try buf.append(allocator, 0x0B); // end

        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Export section (id=7)
    {
        try buf.append(allocator, 0x07); // export section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;
        {
            const n = encode.encodeU32leb(&leb_buf, @intCast(symbols.len));
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }

        for (symbols) |sym| {
            {
                const n = encode.encodeU32leb(&leb_buf, @intCast(sym.name.len));
                try buf.appendSlice(allocator, leb_buf[0..n]);
            }
            try buf.appendSlice(allocator, sym.name);
            try buf.append(allocator, 0x00); // export kind: function
            {
                const n = encode.encodeU32leb(&leb_buf, sym.index);
                try buf.appendSlice(allocator, leb_buf[0..n]);
            }
        }

        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Element section (id=9): one active segment on table 0 at offset 0, listing
    // every function so indirect calls resolve.
    if (n_funcs > 0) {
        try buf.append(allocator, 0x09); // element section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;
        try buf.append(allocator, 0x01); // one segment
        try buf.append(allocator, 0x00); // flags: active, table 0
        try buf.appendSlice(allocator, &[_]u8{ 0x41, 0x00, 0x0B }); // offset: i32.const 0, end
        {
            const n = encode.encodeU32leb(&leb_buf, n_funcs);
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }
        for (0..n_funcs) |fi| {
            const n = encode.encodeU32leb(&leb_buf, @intCast(fi));
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }
        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    // Code section (id=10)
    {
        try buf.append(allocator, 0x0A); // code section id
        const len_at = try reserveSectionLen(&buf, allocator);
        const section_start = buf.items.len;
        {
            const n = encode.encodeU32leb(&leb_buf, n_funcs);
            try buf.appendSlice(allocator, leb_buf[0..n]);
        }

        for (functions) |body| {
            const body_len: u32 = @intCast(body.len);
            {
                const n = encode.encodeU32leb(&leb_buf, body_len);
                try buf.appendSlice(allocator, leb_buf[0..n]);
            }
            try buf.appendSlice(allocator, body);
        }

        patchSectionLen(&buf, len_at, @intCast(buf.items.len - section_start));
    }

    return .{
        .module = try buf.toOwnedSlice(allocator),
        .symbols = try allocator.dupe(Symbol, symbols),
        .import_count = 0,
    };
}
