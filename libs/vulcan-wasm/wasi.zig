//! A small wasi_snapshot_preview1 host runtime: proc_exit, fd_write, args/environ,
//! random_get, and clock_time_get, with other calls returning ENOSYS.
//!
//! Per-instance and portable. Host functions reach their instance's linear memory, argv,
//! injected I/O, and allocator through a `Ctx` whose pointer the engine forwards as each
//! host import's hidden first argument (see `engine.setImportContext`). There is no module
//! global, so any number of WASI instances run side by side without clobbering each other,
//! and there are no hardwired syscalls: all output, entropy, and time go through the
//! injected `std.Io`, so the runtime is correct on every OS the `Io` implementation covers.

const std = @import("std");

/// Per-instance WASI host state, injected via the engine's import-context pointer. The
/// caller builds one, points the instance at it with `setImportContext`, and keeps it
/// alive across the guest's `_start`.
pub const Ctx = struct {
    /// The instance's linear memory; guest addresses index into this.
    mem: []u8 = &.{},
    /// The program arguments exposed via args_get / args_sizes_get.
    args: []const []const u8 = &.{},
    /// Injected I/O: stdout/stderr writes, entropy, and time all route through it.
    io: std.Io,
    /// Injected allocator for the host-side scratch that fuller WASI calls (fd_read,
    /// path_open, poll_oneoff, ...) need. The minimal call set below does not allocate,
    /// but this is the layer's allocation injection point rather than a hidden global.
    allocator: std.mem.Allocator,
};

// WASI errno values used here.
const ESUCCESS: i32 = 0;
const EBADF: i32 = 8;
const EFAULT: i32 = 21;
const EIO: i32 = 29;
const ENOSYS: i32 = 52;

// WASI clockid_t values (subset).
const CLOCKID_REALTIME: i32 = 0;
const CLOCKID_MONOTONIC: i32 = 1;
const CLOCKID_PROCESS_CPUTIME: i32 = 2;
const CLOCKID_THREAD_CPUTIME: i32 = 3;

/// Recover the per-instance `Ctx` from the engine's import-context pointer. Reads only,
/// but writes flow through `Ctx.mem`, which is a `[]u8`.
fn ctxOf(cptr: *anyopaque) *const Ctx {
    return @ptrCast(@alignCast(cptr));
}

fn proc_exit(_: *anyopaque, code: i32) callconv(.c) i32 {
    std.process.exit(@intCast(code & 0xff));
}

/// fd_write(fd, iovs, iovs_len, nwritten): gather-write the iovecs to stdout (fd 1) or
/// stderr (fd 2) via the injected I/O. Other descriptors are rejected.
fn fd_write(cptr: *anyopaque, fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) callconv(.c) i32 {
    if (fd != 1 and fd != 2) return EBADF;
    const ctx = ctxOf(cptr);
    const file = if (fd == 2) std.Io.File.stderr() else std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = file.writer(ctx.io, &buf);
    const base: u32 = @bitCast(iovs);
    var total: u32 = 0;
    var i: u32 = 0;
    while (i < @as(u32, @bitCast(iovs_len))) : (i += 1) {
        const e = base + i * 8;
        const p = std.mem.readInt(u32, ctx.mem[e..][0..4], .little);
        const len = std.mem.readInt(u32, ctx.mem[e + 4 ..][0..4], .little);
        if (len == 0) continue;
        // writeAll drains the whole slice (handling short writes internally); a failure
        // is a real I/O fault surfaced as EIO rather than a silently dropped write.
        w.interface.writeAll(ctx.mem[p..][0..len]) catch return EIO;
        total += len;
    }
    w.interface.flush() catch return EIO;
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(nwritten))..][0..4], total, .little);
    return ESUCCESS;
}

/// args_sizes_get(argc_out, buf_size_out): argc and the total bytes of the NUL-terminated args.
fn args_sizes_get(cptr: *anyopaque, argc_out: i32, buf_size_out: i32) callconv(.c) i32 {
    const ctx = ctxOf(cptr);
    var buf_size: u32 = 0;
    for (ctx.args) |a| buf_size += @intCast(a.len + 1);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(argc_out))..][0..4], @intCast(ctx.args.len), .little);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(buf_size_out))..][0..4], buf_size, .little);
    return ESUCCESS;
}

/// args_get(argv_out, buf_out): the pointer array then the NUL-terminated arg bytes.
fn args_get(cptr: *anyopaque, argv_out: i32, buf_out: i32) callconv(.c) i32 {
    const ctx = ctxOf(cptr);
    var argv: u32 = @bitCast(argv_out);
    var buf: u32 = @bitCast(buf_out);
    for (ctx.args) |a| {
        std.mem.writeInt(u32, ctx.mem[argv..][0..4], buf, .little);
        argv += 4;
        @memcpy(ctx.mem[buf .. buf + a.len], a);
        ctx.mem[buf + a.len] = 0;
        buf += @intCast(a.len + 1);
    }
    return ESUCCESS;
}

fn environ_sizes_get(cptr: *anyopaque, count_out: i32, buf_size_out: i32) callconv(.c) i32 {
    const ctx = ctxOf(cptr);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(count_out))..][0..4], 0, .little);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(buf_size_out))..][0..4], 0, .little);
    return ESUCCESS;
}
fn environ_get(_: *anyopaque, _: i32, _: i32) callconv(.c) i32 {
    return ESUCCESS;
}

fn random_get(cptr: *anyopaque, buf: i32, len: i32) callconv(.c) i32 {
    const ctx = ctxOf(cptr);
    const off: u32 = @bitCast(buf);
    const n: u32 = @bitCast(len);
    // Cryptographically secure entropy through the injected I/O. A failure surfaces as an
    // I/O error rather than leaving the guest buffer uninitialized.
    ctx.io.randomSecure(ctx.mem[off..][0..n]) catch return EIO;
    return ESUCCESS;
}

/// clock_time_get(id, precision, time_out): the clock's current time, in nanoseconds.
fn clock_time_get(cptr: *anyopaque, id: i32, _: i64, time_out: i32) callconv(.c) i32 {
    const ctx = ctxOf(cptr);
    const clock: std.Io.Clock = switch (id) {
        CLOCKID_REALTIME => .real,
        CLOCKID_PROCESS_CPUTIME => .cpu_process,
        CLOCKID_THREAD_CPUTIME => .cpu_thread,
        else => .awake, // CLOCKID_MONOTONIC and any unknown id map to the monotonic clock
    };
    const ts = std.Io.Timestamp.now(ctx.io, clock);
    const ns = std.math.cast(u64, ts.nanoseconds) orelse std.math.maxInt(u64);
    std.mem.writeInt(u64, ctx.mem[@as(u32, @bitCast(time_out))..][0..8], ns, .little);
    return ESUCCESS;
}

/// No preopened directories, so the fd-table scan at startup ends immediately.
fn fd_prestat_get(_: *anyopaque, _: i32, _: i32) callconv(.c) i32 {
    return EBADF;
}

/// A catch-all for unimplemented WASI calls (any arity, since the C calling convention
/// ignores extra arguments, including the leading import-context pointer).
fn unsupported() callconv(.c) i32 {
    return ENOSYS;
}

/// The host address bound to import "wasi_snapshot_preview1.<field>", or the unsupported stub
/// for a recognized-but-unimplemented WASI call. Returns null for a non-WASI import.
pub fn resolve(name: []const u8) ?usize {
    const prefix = "wasi_snapshot_preview1.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const field = name[prefix.len..];
    const table = .{
        .{ "proc_exit", @intFromPtr(&proc_exit) },
        .{ "fd_write", @intFromPtr(&fd_write) },
        .{ "args_sizes_get", @intFromPtr(&args_sizes_get) },
        .{ "args_get", @intFromPtr(&args_get) },
        .{ "environ_sizes_get", @intFromPtr(&environ_sizes_get) },
        .{ "environ_get", @intFromPtr(&environ_get) },
        .{ "random_get", @intFromPtr(&random_get) },
        .{ "clock_time_get", @intFromPtr(&clock_time_get) },
        .{ "fd_prestat_get", @intFromPtr(&fd_prestat_get) },
    };
    inline for (table) |entry| {
        if (std.mem.eql(u8, field, entry[0])) return entry[1];
    }
    return @intFromPtr(&unsupported);
}

test "args_get/args_sizes_get reach argv through the per-instance context, not a global" {
    // Exercises the import-context threading at the host-function level: the ctx pointer
    // the engine forwards is recovered and its injected memory/args are used directly.
    var mem = [_]u8{0} ** 128;
    var ctx = Ctx{
        .mem = &mem,
        .args = &.{ "prog", "arg1" },
        .io = std.testing.io,
        .allocator = std.testing.allocator,
    };

    try std.testing.expectEqual(ESUCCESS, args_sizes_get(&ctx, 0, 4));
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, mem[0..4], .little)); // argc
    try std.testing.expectEqual(@as(u32, 10), std.mem.readInt(u32, mem[4..8], .little)); // "prog\0" + "arg1\0"

    try std.testing.expectEqual(ESUCCESS, args_get(&ctx, 16, 64));
    const p0 = std.mem.readInt(u32, mem[16..20], .little);
    const p1 = std.mem.readInt(u32, mem[20..24], .little);
    try std.testing.expectEqualStrings("prog", std.mem.sliceTo(mem[p0..], 0));
    try std.testing.expectEqualStrings("arg1", std.mem.sliceTo(mem[p1..], 0));
}
