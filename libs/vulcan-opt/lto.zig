//! Link-time optimization. A whole-program `Module` gathers the functions of
//! several separately-compiled units (loaded from bitcode), then optimizes across
//! unit boundaries: cross-module inlining (the inliner re-interns the callee's
//! types/symbols into the caller, so a module-wide lookup spans modules) and
//! dead-function elimination via call-graph reachability from the exported roots.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const inlining = @import("inline.zig");
const constfold = @import("constfold.zig");
const gvn = @import("gvn.zig");
const licm = @import("licm.zig");
const dce = @import("dce.zig");

const Function = ir.function.Function;

/// The per-function cleanup pipeline run after cross-module inlining.
const cleanup_pipeline = [_]pass.Pass{ constfold.pass_def, gvn.pass_def, licm.pass_def, dce.pass_def };

pub const Error = std.mem.Allocator.Error || ir.bitcode.Error;

/// A whole-program module: named functions it owns.
pub const Module = struct {
    allocator: std.mem.Allocator,
    names: std.ArrayList([]u8) = .empty,
    funcs: std.ArrayList(*Function) = .empty,

    pub fn init(allocator: std.mem.Allocator) Module {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Module) void {
        for (self.names.items) |n| self.allocator.free(n);
        for (self.funcs.items) |f| {
            f.deinit();
            self.allocator.destroy(f);
        }
        self.names.deinit(self.allocator);
        self.funcs.deinit(self.allocator);
    }

    /// Take ownership of `func` under `name`. The caller must not deinit `func`.
    pub fn add(self: *Module, name: []const u8, func: Function) Error!void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const slot = try self.allocator.create(Function);
        errdefer self.allocator.destroy(slot);
        slot.* = func;
        try self.names.append(self.allocator, owned_name);
        errdefer _ = self.names.pop();
        try self.funcs.append(self.allocator, slot);
    }

    pub fn count(self: *const Module) usize {
        return self.funcs.items.len;
    }

    pub fn get(self: *const Module, name: []const u8) ?*Function {
        for (self.names.items, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return self.funcs.items[i];
        }
        return null;
    }

    /// A name lookup over this module, for cross-module inlining.
    pub fn lookup(self: *const Module) inlining.Lookup {
        return .{ .context = @constCast(self), .func = lookupFn };
    }

    fn lookupFn(context: *anyopaque, name: []const u8) ?*const Function {
        const self: *const Module = @ptrCast(@alignCast(context));
        return self.get(name);
    }
};

/// Serialize a module: a count, then each function's name and bitcode.
pub fn encode(allocator: std.mem.Allocator, module: *const Module) Error![]u8 {
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);
    try appendU32(allocator, &bytes, @intCast(module.count()));
    for (module.names.items, 0..) |name, i| {
        try appendU32(allocator, &bytes, @intCast(name.len));
        try bytes.appendSlice(allocator, name);
        const fb = try ir.bitcode.encode(allocator, module.funcs.items[i]);
        defer allocator.free(fb);
        try appendU32(allocator, &bytes, @intCast(fb.len));
        try bytes.appendSlice(allocator, fb);
    }
    return bytes.toOwnedSlice(allocator);
}

/// Deserialize a module written by `encode`. The caller owns it.
pub fn decode(allocator: std.mem.Allocator, data: []const u8) Error!Module {
    var module = Module.init(allocator);
    errdefer module.deinit();
    var r = Cursor{ .data = data };
    const n = try r.takeU32();
    for (0..n) |_| {
        const name = try r.takeBytes(try r.takeU32());
        const fb = try r.takeBytes(try r.takeU32());
        const func = try ir.bitcode.decode(allocator, fb);
        try module.add(name, func);
    }
    return module;
}

const Cursor = struct {
    data: []const u8,
    pos: usize = 0,
    fn takeU32(self: *Cursor) Error!u32 {
        if (self.pos + 4 > self.data.len) return error.MalformedBitcode;
        const v = std.mem.readInt(u32, self.data[self.pos..][0..4], .little);
        self.pos += 4;
        return v;
    }
    fn takeBytes(self: *Cursor, len: u32) Error![]const u8 {
        if (self.pos + len > self.data.len) return error.MalformedBitcode;
        const s = self.data[self.pos .. self.pos + len];
        self.pos += len;
        return s;
    }
};

fn appendU32(allocator: std.mem.Allocator, list: *std.ArrayList(u8), v: u32) Error!void {
    try list.appendSlice(allocator, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
}

/// Optimize a whole-program module: inline across units, clean up each function,
/// then drop functions unreachable from `roots`. Returns whether it changed it.
pub fn link(allocator: std.mem.Allocator, module: *Module, roots: []const []const u8) Error!bool {
    var changed = false;
    const lk = module.lookup();
    for (module.funcs.items) |func| {
        if (try inlining.run(allocator, func, lk)) changed = true;
        if (try pass.runToFixpoint(allocator, func, &cleanup_pipeline, 16)) changed = true;
    }
    if (try eliminateDead(allocator, module, roots)) changed = true;
    return changed;
}

/// Remove functions not reachable through calls from any root. Returns whether
/// any were removed.
fn eliminateDead(allocator: std.mem.Allocator, module: *Module, roots: []const []const u8) Error!bool {
    var reachable: std.StringHashMapUnmanaged(void) = .empty;
    defer reachable.deinit(allocator);
    var work: std.ArrayList([]const u8) = .empty;
    defer work.deinit(allocator);

    for (roots) |root| {
        if (module.get(root) != null and !reachable.contains(root)) {
            try reachable.put(allocator, root, {});
            try work.append(allocator, root);
        }
    }
    while (work.pop()) |name| {
        const func = module.get(name).?;
        for (0..func.blockCount()) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) == .call) {
                    const callee = func.symbolName(func.opcode(inst).call.symbol);
                    if (module.get(callee) != null and !reachable.contains(callee)) {
                        try reachable.put(allocator, callee, {});
                        try work.append(allocator, callee);
                    }
                }
            }
        }
    }

    // Keep only reachable functions, deinit the rest.
    var removed = false;
    var w: usize = 0;
    for (module.names.items, 0..) |name, i| {
        const func = module.funcs.items[i];
        if (reachable.contains(name)) {
            module.names.items[w] = name;
            module.funcs.items[w] = func;
            w += 1;
        } else {
            module.allocator.free(name);
            func.deinit();
            module.allocator.destroy(func);
            removed = true;
        }
    }
    module.names.shrinkRetainingCapacity(w);
    module.funcs.shrinkRetainingCapacity(w);
    return removed;
}

fn i32k(func: *Function) std.mem.Allocator.Error!ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "cross-module inlining then dead-function elimination" {
    const allocator = std.testing.allocator;
    var module = Module.init(allocator);
    defer module.deinit();

    // unit A: helper(a, b) = a*b + a  (leaf, inlinable)
    {
        var f = Function.init(allocator);
        const t = try i32k(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const bb = try f.appendBlockParam(b, t);
        const prod = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
        const sum = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
        f.setTerminator(b, .{ .ret = sum });
        try module.add("helper", f);
    }
    // unit B: entry(x) = helper(x, x) + 1
    {
        var f = Function.init(allocator);
        const t = try i32k(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const call = try f.appendCall(b, t, "helper", &.{ x, x });
        const r = try f.appendArithImm(b, t, .add, call, 1);
        f.setTerminator(b, .{ .ret = r });
        try module.add("entry", f);
    }

    try std.testing.expectEqual(@as(usize, 2), module.count());
    _ = try link(allocator, &module, &.{"entry"});

    // helper was inlined into entry, leaving it uncalled, so DFE dropped it.
    try std.testing.expectEqual(@as(usize, 1), module.count());
    try std.testing.expect(module.get("entry") != null);
    try std.testing.expect(module.get("helper") == null);
    for (module.get("entry").?.blockInsts(@enumFromInt(0))) |inst| {
        try std.testing.expect(module.get("entry").?.opcode(inst) != .call);
    }
}

test "module round-trips through bitcode" {
    const allocator = std.testing.allocator;
    var module = Module.init(allocator);
    defer module.deinit();
    {
        var f = Function.init(allocator);
        const t = try i32k(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const r = try f.appendArithImm(b, t, .mul, x, 3);
        f.setTerminator(b, .{ .ret = r });
        try module.add("triple", f);
    }

    const bytes = try encode(allocator, &module);
    defer allocator.free(bytes);
    var back = try decode(allocator, bytes);
    defer back.deinit();

    try std.testing.expectEqual(@as(usize, 1), back.count());
    const a = try std.fmt.allocPrint(allocator, "{f}", .{module.get("triple").?});
    defer allocator.free(a);
    const b2 = try std.fmt.allocPrint(allocator, "{f}", .{back.get("triple").?});
    defer allocator.free(b2);
    try std.testing.expectEqualStrings(a, b2);
}
