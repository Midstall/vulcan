//! Platform abstraction for JIT executable memory. A `Provider` supplies and protects the
//! pages: posix mmap/mprotect by default, or a freestanding embedder's own (UEFI boot
//! services). The arch-specific instruction-cache sync is a comptime hook on the generic
//! `Buffer`, keeping the OS and arch layers independent.

const std = @import("std");
const builtin = @import("builtin");

pub const page_align = std.heap.page_size_min;

/// A region of executable-capable memory.
pub const ExecMemory = struct { ptr: [*]align(page_align) u8, len: usize };

pub const Error = error{ EmptyCode, AllocFailed, ProtectFailed };

/// How a platform obtains, protects, and releases executable memory. `ctx` carries any
/// runtime state the implementation needs (e.g. UEFI boot services).
pub const Provider = struct {
    ctx: ?*anyopaque = null,
    /// Allocate `len` writable, page-aligned bytes (rounded up). Returns null on failure.
    allocFn: *const fn (ctx: ?*anyopaque, len: usize) ?ExecMemory,
    /// Make `mem` read+execute (W^X). Returns false on failure. May be a no-op where
    /// memory is already executable (e.g. UEFI boot time).
    protectFn: *const fn (ctx: ?*anyopaque, mem: ExecMemory) bool,
    freeFn: *const fn (ctx: ?*anyopaque, mem: ExecMemory) void,
};

/// The posix provider: `mmap` writable pages, `mprotect` to R+X, `munmap` to free.
pub const posix: Provider = .{ .allocFn = posixAlloc, .protectFn = posixProtect, .freeFn = posixFree };

fn posixAlloc(_: ?*anyopaque, len: usize) ?ExecMemory {
    const m = std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return null;
    return .{ .ptr = m.ptr, .len = m.len };
}
fn posixProtect(_: ?*anyopaque, mem: ExecMemory) bool {
    const rc = std.posix.system.mprotect(mem.ptr, mem.len, .{ .READ = true, .EXEC = true });
    return std.posix.errno(rc) == .SUCCESS;
}
fn posixFree(_: ?*anyopaque, mem: ExecMemory) void {
    std.posix.munmap(mem.ptr[0..mem.len]);
}

const uefi = std.os.uefi;

/// The UEFI provider: boot-services pages (executable at boot, so protect is a no-op).
pub const uefi_provider: Provider = .{ .allocFn = uefiAlloc, .protectFn = uefiProtect, .freeFn = uefiFree };

fn uefiAlloc(_: ?*anyopaque, len: usize) ?ExecMemory {
    const bs = uefi.system_table.boot_services orelse return null;
    const pages = (len + 4095) / 4096;
    const mem = bs.allocatePages(.any, .loader_code, pages) catch return null;
    return .{ .ptr = @ptrCast(mem.ptr), .len = pages * 4096 };
}
fn uefiProtect(_: ?*anyopaque, _: ExecMemory) bool {
    return true;
}
fn uefiFree(_: ?*anyopaque, mem: ExecMemory) void {
    const bs = uefi.system_table.boot_services orelse return;
    const page_ptr: [*]align(4096) uefi.Page = @ptrCast(@alignCast(mem.ptr));
    // freeFn cannot propagate (its signature returns void) and a leaked page at
    // JIT-buffer teardown is non-fatal, but the failure must be observable rather
    // than silently swallowed.
    bs.freePages(page_ptr[0 .. mem.len / 4096]) catch |err|
        std.log.warn("uefiFree: freePages failed: {s}", .{@errorName(err)});
}

/// The provider for the current target: UEFI boot-services pages on UEFI, posix mmap otherwise.
pub const default_provider: Provider = if (builtin.os.tag == .uefi) uefi_provider else posix;

/// A W^X executable buffer parameterized by the arch instruction-cache sync hook
/// (a no-op on cache-coherent ISAs like x86, `dc`/`ic` on AArch64, `fence.i` on RISC-V).
pub fn Buffer(comptime syncICache: fn (mem: []const u8) void) type {
    return struct {
        const Self = @This();
        memory: []align(page_align) u8,
        provider: Provider,

        /// Map `code` into executable memory from `provider`: allocate writable, copy,
        /// flip to read+execute, then synchronize the instruction cache.
        pub fn mapWith(provider: Provider, code: []const u8) Error!Self {
            if (code.len == 0) return error.EmptyCode;
            const em = provider.allocFn(provider.ctx, code.len) orelse return error.AllocFailed;
            const memory = em.ptr[0..em.len];
            @memcpy(memory[0..code.len], code);
            if (!provider.protectFn(provider.ctx, em)) {
                provider.freeFn(provider.ctx, em);
                return error.ProtectFailed;
            }
            syncICache(memory[0..code.len]);
            return .{ .memory = memory, .provider = provider };
        }

        /// Map `code` using the current target's default provider.
        pub fn map(code: []const u8) Error!Self {
            return mapWith(default_provider, code);
        }

        pub fn deinit(self: *Self) void {
            self.provider.freeFn(self.provider.ctx, .{ .ptr = self.memory.ptr, .len = self.memory.len });
            self.memory = &.{};
        }

        /// A function pointer of type `Fn` to byte `offset` within the buffer.
        pub fn entry(self: *const Self, comptime Fn: type, offset: usize) Fn {
            return @ptrCast(@alignCast(self.memory.ptr + offset));
        }
    };
}

fn noSync(_: []const u8) void {}

test "Buffer maps code W^X via the posix provider" {
    const code = [_]u8{ 0xC3, 0x90 }; // ret, nop
    var buf = try Buffer(noSync).map(&code);
    defer buf.deinit();
    try std.testing.expectEqualSlices(u8, &code, buf.memory[0..code.len]);
}
