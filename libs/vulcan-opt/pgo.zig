//! Profile-guided optimization. Three pieces: a `Profile` (per-function block
//! execution counts) with a binary format, `instrument` (insert a block-entry
//! counter into a module-level array so a run produces counts), and `guidedInline`
//! (inline only the call sites in hot blocks).
//!
//! Instrumentation is transparent: counters live in their own global array and
//! never feed the function's result. Collecting counts back from a run needs a
//! runtime that emits the array, so the guided passes are tested with given
//! profiles.

const std = @import("std");
const ir = @import("vulcan-ir");
const inlining = @import("inline.zig");
const lto = @import("lto.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;
const Type = ir.types.Type;

pub const Error = std.mem.Allocator.Error || error{MalformedProfile};

/// Per-function block execution counts.
pub const Profile = struct {
    allocator: std.mem.Allocator,
    names: std.ArrayList([]u8) = .empty,
    counts: std.ArrayList([]u64) = .empty,

    pub fn init(allocator: std.mem.Allocator) Profile {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Profile) void {
        for (self.names.items) |n| self.allocator.free(n);
        for (self.counts.items) |c| self.allocator.free(c);
        self.names.deinit(self.allocator);
        self.counts.deinit(self.allocator);
    }

    /// Record `counts` (copied) for the function `name` (copied).
    pub fn add(self: *Profile, name: []const u8, counts: []const u64) Error!void {
        const n = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(n);
        const c = try self.allocator.dupe(u64, counts);
        errdefer self.allocator.free(c);
        try self.names.append(self.allocator, n);
        errdefer _ = self.names.pop();
        try self.counts.append(self.allocator, c);
    }

    pub fn get(self: *const Profile, name: []const u8) ?[]const u64 {
        for (self.names.items, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return self.counts.items[i];
        }
        return null;
    }
};

/// Serialize a profile: count, then per function name and its block counts.
pub fn encode(allocator: std.mem.Allocator, profile: *const Profile) Error![]u8 {
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);
    try putU32(allocator, &bytes, @intCast(profile.names.items.len));
    for (profile.names.items, 0..) |name, i| {
        try putU32(allocator, &bytes, @intCast(name.len));
        try bytes.appendSlice(allocator, name);
        const counts = profile.counts.items[i];
        try putU32(allocator, &bytes, @intCast(counts.len));
        for (counts) |c| try putU64(allocator, &bytes, c);
    }
    return bytes.toOwnedSlice(allocator);
}

pub fn decode(allocator: std.mem.Allocator, data: []const u8) Error!Profile {
    var profile = Profile.init(allocator);
    errdefer profile.deinit();
    var pos: usize = 0;
    const n = try getU32(data, &pos);
    for (0..n) |_| {
        const name_len = try getU32(data, &pos);
        const name = try takeBytes(data, &pos, name_len);
        const c_len = try getU32(data, &pos);
        const counts = try allocator.alloc(u64, c_len);
        defer allocator.free(counts);
        for (counts) |*c| c.* = try getU64(data, &pos);
        try profile.add(name, counts);
    }
    return profile;
}

fn putU32(a: std.mem.Allocator, l: *std.ArrayList(u8), v: u32) Error!void {
    try l.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
}
fn putU64(a: std.mem.Allocator, l: *std.ArrayList(u8), v: u64) Error!void {
    try l.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u64, v)));
}
fn getU32(data: []const u8, pos: *usize) Error!u32 {
    if (pos.* + 4 > data.len) return error.MalformedProfile;
    const v = std.mem.readInt(u32, data[pos.*..][0..4], .little);
    pos.* += 4;
    return v;
}
fn getU64(data: []const u8, pos: *usize) Error!u64 {
    if (pos.* + 8 > data.len) return error.MalformedProfile;
    const v = std.mem.readInt(u64, data[pos.*..][0..8], .little);
    pos.* += 8;
    return v;
}
fn takeBytes(data: []const u8, pos: *usize, len: u32) Error![]const u8 {
    if (pos.* + len > data.len) return error.MalformedProfile;
    const s = data[pos.* .. pos.* + len];
    pos.* += len;
    return s;
}

/// Instrument `func`: at the entry of every block, increment a per-block counter
/// in the global array named `counters_symbol`. Returns the number of counters
/// (= block count). The linker must provide a `.bss` array of `count * 8` bytes.
/// The increments never feed the function's result, so it stays transparent.
pub fn instrument(allocator: std.mem.Allocator, func: *Function, counters_symbol: []const u8) Error!usize {
    const ptr_t = try func.types.intern(.ptr);
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const n = func.blockCount();

    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        const before = func.blockInsts(block).len;

        // counters[bi] += 1
        const base = try func.appendGlobalAddr(block, ptr_t, counters_symbol);
        const slot = try func.appendArithImm(block, ptr_t, .add, base, @intCast(bi * 8));
        const cur = try func.appendInst(block, i64_t, .{ .load = .{ .ptr = slot } });
        const inc = try func.appendArithImm(block, i64_t, .add, cur, 1);
        try func.appendStore(block, inc, slot);

        // The five new instructions were appended to the tail. Move them to the
        // front so they run on block entry, before any `if`.
        try moveTailToFront(allocator, func, block, before);
    }
    return n;
}

/// Reorder a block's instruction list so the instructions added after index
/// `split` come first.
fn moveTailToFront(allocator: std.mem.Allocator, func: *Function, block: Block, split: usize) Error!void {
    const insts = func.blockInstsMut(block);
    var rebuilt: std.ArrayList(Inst) = .empty;
    defer rebuilt.deinit(allocator);
    try rebuilt.appendSlice(allocator, insts.items[split..]); // the new counter ops
    try rebuilt.appendSlice(allocator, insts.items[0..split]); // the original body
    insts.clearRetainingCapacity();
    try insts.appendSlice(allocator, rebuilt.items);
}

const HotFilter = struct {
    counts: []const u64,
    threshold: u64,
    fn allow(context: *anyopaque, block_index: usize) bool {
        const self: *const HotFilter = @ptrCast(@alignCast(context));
        return block_index < self.counts.len and self.counts[block_index] >= self.threshold;
    }
};

/// Profile-guided inlining: inline only the call sites in blocks whose execution
/// count (from `profile`) is at least `threshold`. Returns whether it changed it.
pub fn guidedInline(allocator: std.mem.Allocator, module: *lto.Module, profile: *const Profile, threshold: u64) Error!bool {
    var changed = false;
    const lk = module.lookup();
    for (module.names.items, 0..) |name, i| {
        const counts = profile.get(name) orelse continue;
        var filt = HotFilter{ .counts = counts, .threshold = threshold };
        const filter = inlining.Filter{ .context = &filt, .func = HotFilter.allow };
        if (try inlining.runFiltered(allocator, module.funcs.items[i], lk, filter)) changed = true;
    }
    return changed;
}

fn i32k(func: *Function) std.mem.Allocator.Error!Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "profile round-trips through its binary format" {
    const allocator = std.testing.allocator;
    var p = Profile.init(allocator);
    defer p.deinit();
    try p.add("hot", &.{ 1000, 0, 999 });
    try p.add("cold", &.{5});

    const bytes = try encode(allocator, &p);
    defer allocator.free(bytes);
    var back = try decode(allocator, bytes);
    defer back.deinit();

    try std.testing.expectEqualSlices(u64, &.{ 1000, 0, 999 }, back.get("hot").?);
    try std.testing.expectEqualSlices(u64, &.{5}, back.get("cold").?);
    try std.testing.expect(back.get("missing") == null);
    try std.testing.expectError(error.MalformedProfile, decode(allocator, "\x01\x00\x00\x00"));
}

test "instrumentation adds a transparent counter to every block" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32k(&func);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
    func.setTerminator(b, .{ .ret = r });

    const before = func.blockInsts(b).len;
    const counters = try instrument(allocator, &func, "prof_counters");
    try std.testing.expectEqual(@as(usize, 1), counters); // one block, one counter

    // Five counter instructions were prepended. The original body still follows,
    // and the result (the mul) is unchanged.
    try std.testing.expectEqual(before + 5, func.blockInsts(b).len);
    try std.testing.expectEqual(r, func.terminator(b).?.ret.?);
    var has_store = false;
    for (func.blockInsts(b)) |inst| {
        if (func.opcode(inst) == .store) has_store = true;
    }
    try std.testing.expect(has_store);
}

test "guided inlining inlines hot calls and skips cold ones" {
    const allocator = std.testing.allocator;
    var module = lto.Module.init(allocator);
    defer module.deinit();

    // helper(a) = a + a  (leaf)
    {
        var f = Function.init(allocator);
        const t = try i32k(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        f.setTerminator(b, .{ .ret = s });
        try module.add("helper", f);
    }
    // caller(x): block0 if x>0 -> hot else cold, hot: r=helper(x) jump done(r)
    // cold: r2=helper(x) jump done(r2), done(z): ret z. Two calls in two blocks.
    {
        var f = Function.init(allocator);
        const t = try i32k(&f);
        const bool_t = try f.types.intern(.bool);
        const b0 = try f.appendBlock();
        const hot = try f.appendBlock();
        const cold = try f.appendBlock();
        const done = try f.appendBlock();
        const x = try f.appendBlockParam(b0, t);
        const z = try f.appendBlockParam(done, t);
        const zero = try f.appendInst(b0, t, .{ .iconst = 0 });
        const c = try f.appendInst(b0, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = zero } });
        try f.appendIf(b0, c, .{ .target = hot, .args = &.{} }, .{ .target = cold, .args = &.{} });
        const r = try f.appendCall(hot, t, "helper", &.{x});
        try f.setJump(hot, done, &.{r});
        const r2 = try f.appendCall(cold, t, "helper", &.{x});
        try f.setJump(cold, done, &.{r2});
        f.setTerminator(done, .{ .ret = z });
        try module.add("caller", f);
    }

    // Profile: block 1 (hot) ran a lot, block 2 (cold) never.
    var profile = Profile.init(allocator);
    defer profile.deinit();
    try profile.add("caller", &.{ 1, 1000, 0, 1 });

    try std.testing.expect(try guidedInline(allocator, &module, &profile, 100));

    // The hot block's call was inlined. The cold block still has its call.
    const caller = module.get("caller").?;
    var hot_calls: usize = 0;
    var cold_calls: usize = 0;
    for (caller.blockInsts(@enumFromInt(1))) |inst| {
        if (caller.opcode(inst) == .call) hot_calls += 1;
    }
    for (caller.blockInsts(@enumFromInt(2))) |inst| {
        if (caller.opcode(inst) == .call) cold_calls += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), hot_calls); // hot inlined
    try std.testing.expectEqual(@as(usize, 1), cold_calls); // cold kept
}
