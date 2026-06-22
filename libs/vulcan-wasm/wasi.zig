//! A small wasi_snapshot_preview1 host runtime: proc_exit, fd_write, args/environ, random_get,
//! and clock_time_get, with other calls returning ENOSYS. Host functions reach the instance's
//! linear memory through a global context, since import calls pass only the wasm arguments.

const std = @import("std");
const linux = std.os.linux;

const Ctx = struct {
    mem: []u8 = &.{},
    args: []const []const u8 = &.{},
};

var ctx: Ctx = .{};

/// Bind the runtime to an instance's memory and the program's argv before calling `_start`.
pub fn setup(memory: []u8, args: []const []const u8) void {
    ctx = .{ .mem = memory, .args = args };
}

// WASI errno values used here.
const ESUCCESS: i32 = 0;
const EBADF: i32 = 8;
const EFAULT: i32 = 21;
const ENOSYS: i32 = 52;

fn proc_exit(code: i32) callconv(.c) i32 {
    std.process.exit(@intCast(code & 0xff));
}

/// fd_write(fd, iovs, iovs_len, nwritten): gather-write the iovecs to stdout (fd 1) or
/// stderr (fd 2). Other descriptors are rejected.
fn fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) callconv(.c) i32 {
    if (fd != 1 and fd != 2) return EBADF;
    const host_fd: i32 = if (fd == 2) 2 else 1;
    const base: u32 = @bitCast(iovs);
    var total: u32 = 0;
    var i: u32 = 0;
    while (i < @as(u32, @bitCast(iovs_len))) : (i += 1) {
        const e = base + i * 8;
        const buf = std.mem.readInt(u32, ctx.mem[e..][0..4], .little);
        const len = std.mem.readInt(u32, ctx.mem[e + 4 ..][0..4], .little);
        if (len == 0) continue;
        _ = linux.write(host_fd, ctx.mem[buf..].ptr, len);
        total += len;
    }
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(nwritten))..][0..4], total, .little);
    return ESUCCESS;
}

/// args_sizes_get(argc_out, buf_size_out): argc and the total bytes of the NUL-terminated args.
fn args_sizes_get(argc_out: i32, buf_size_out: i32) callconv(.c) i32 {
    var buf_size: u32 = 0;
    for (ctx.args) |a| buf_size += @intCast(a.len + 1);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(argc_out))..][0..4], @intCast(ctx.args.len), .little);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(buf_size_out))..][0..4], buf_size, .little);
    return ESUCCESS;
}

/// args_get(argv_out, buf_out): the pointer array then the NUL-terminated arg bytes.
fn args_get(argv_out: i32, buf_out: i32) callconv(.c) i32 {
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

fn environ_sizes_get(count_out: i32, buf_size_out: i32) callconv(.c) i32 {
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(count_out))..][0..4], 0, .little);
    std.mem.writeInt(u32, ctx.mem[@as(u32, @bitCast(buf_size_out))..][0..4], 0, .little);
    return ESUCCESS;
}
fn environ_get(_: i32, _: i32) callconv(.c) i32 {
    return ESUCCESS;
}

fn random_get(buf: i32, len: i32) callconv(.c) i32 {
    const off: u32 = @bitCast(buf);
    const n: u32 = @bitCast(len);
    _ = linux.getrandom(ctx.mem[off..].ptr, n, 0);
    return ESUCCESS;
}

/// clock_time_get(id, precision, time_out): a monotonic nanosecond timestamp.
fn clock_time_get(_: i32, _: i64, time_out: i32) callconv(.c) i32 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    const ns = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    std.mem.writeInt(u64, ctx.mem[@as(u32, @bitCast(time_out))..][0..8], ns, .little);
    return ESUCCESS;
}

/// No preopened directories, so the fd-table scan at startup ends immediately.
fn fd_prestat_get(_: i32, _: i32) callconv(.c) i32 {
    return EBADF;
}

/// A catch-all for unimplemented WASI calls (any arity, since the C calling convention
/// ignores extra arguments).
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
