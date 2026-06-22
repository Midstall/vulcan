//! Natural-loop analysis. A back-edge is a CFG edge `latch -> header` where the
//! header dominates the latch. Its natural loop is the header plus every block
//! that can reach the latch without passing through the header. Back-edges to the
//! same header merge into one loop. Each loop records its preheader (a single
//! non-loop predecessor whose only successor is the header) when one exists.

const std = @import("std");
const ir = @import("vulcan-ir");
const cfg_mod = @import("cfg.zig");
const dominators = @import("dominators.zig");

const Function = ir.function.Function;

pub const Loop = struct {
    header: u32,
    /// `body[block]` is true when the block belongs to this loop.
    body: []bool,
    /// The unique loop preheader, if the loop has one.
    preheader: ?u32,

    pub fn contains(self: *const Loop, block: usize) bool {
        return self.body[block];
    }
};

pub const LoopInfo = struct {
    loops: []Loop,

    pub fn deinit(self: *LoopInfo, allocator: std.mem.Allocator) void {
        for (self.loops) |*l| allocator.free(l.body);
        allocator.free(self.loops);
    }
};

/// Compute the natural loops of `func`. The caller owns the result.
pub fn analyze(allocator: std.mem.Allocator, func: *const Function) std.mem.Allocator.Error!LoopInfo {
    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    const n = cfg.blockCount();

    // Map each loop header to its (unioned) body bitset.
    var headers: std.AutoHashMapUnmanaged(u32, []bool) = .empty;
    defer headers.deinit(allocator);
    errdefer {
        var it = headers.valueIterator();
        while (it.next()) |b| allocator.free(b.*);
    }

    for (0..n) |b| {
        for (cfg.successors(b)) |s| {
            if (!doms.dominates(s, b)) continue; // not a back-edge
            const gop = try headers.getOrPut(allocator, s);
            if (!gop.found_existing) {
                gop.value_ptr.* = try allocator.alloc(bool, n);
                @memset(gop.value_ptr.*, false);
            }
            try addNaturalLoop(allocator, &cfg, s, @intCast(b), gop.value_ptr.*);
        }
    }

    var loops = try allocator.alloc(Loop, headers.count());
    var i: usize = 0;
    var it = headers.iterator();
    while (it.next()) |entry| {
        const h = entry.key_ptr.*;
        const body = entry.value_ptr.*;
        loops[i] = .{ .header = h, .body = body, .preheader = findPreheader(&cfg, h, body) };
        i += 1;
    }
    return .{ .loops = loops };
}

/// Add the natural loop of back-edge `latch -> header` into `body`.
fn addNaturalLoop(allocator: std.mem.Allocator, cfg: *const cfg_mod.Cfg, header: u32, latch: u32, body: []bool) std.mem.Allocator.Error!void {
    body[header] = true;
    if (body[latch]) return; // already gathered
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(allocator);
    body[latch] = true;
    try stack.append(allocator, latch);
    while (stack.pop()) |m| {
        for (cfg.predecessors(m)) |p| {
            if (!body[p]) {
                body[p] = true;
                try stack.append(allocator, p);
            }
        }
    }
}

/// The preheader of a loop: a single predecessor of the header that is outside
/// the loop and whose only successor is the header.
fn findPreheader(cfg: *const cfg_mod.Cfg, header: u32, body: []const bool) ?u32 {
    var pre: ?u32 = null;
    for (cfg.predecessors(header)) |p| {
        if (body[p]) continue; // a latch (loop-internal back-edge)
        if (pre != null) return null; // more than one entry into the loop
        pre = p;
    }
    const p = pre orelse return null;
    const succs = cfg.successors(p);
    return if (succs.len == 1 and succs[0] == header) p else null;
}

test "detects a natural loop and its preheader" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    // entry -> loop(0), loop: if i<n -> body else done, body -> loop(i+1), done: ret
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});
    func.setTerminator(done, .{ .ret = i });

    var info = try analyze(allocator, &func);
    defer info.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), info.loops.len);
    const l = info.loops[0];
    try std.testing.expectEqual(@as(u32, 1), l.header); // loop block
    try std.testing.expect(l.contains(1) and l.contains(2)); // loop + body
    try std.testing.expect(!l.contains(0) and !l.contains(3)); // entry, done outside
    try std.testing.expectEqual(@as(?u32, 0), l.preheader); // entry is the preheader
}
