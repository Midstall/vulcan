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

/// The permission a sub-range of a `MappedImage` reservation is set to: code
/// (read+execute), rodata (read-only), or data/bss (read+write).
pub const Prot = enum { rx, r, rw };

/// How a platform obtains, protects, and releases executable memory. `ctx` carries any
/// runtime state the implementation needs (e.g. UEFI boot services).
pub const Provider = struct {
    ctx: ?*anyopaque = null,
    /// Allocate `len` writable, page-aligned bytes (rounded up). Returns null on failure.
    allocFn: *const fn (ctx: ?*anyopaque, len: usize) ?ExecMemory,
    /// Make `mem` read+execute (W^X). Returns false on failure. May be a no-op where
    /// memory is already executable (e.g. UEFI boot time).
    protectFn: *const fn (ctx: ?*anyopaque, mem: ExecMemory) bool,
    /// Set `[ptr, ptr + len)` (a sub-range of a prior `allocFn` reservation, page-aligned
    /// at both ends) to `prot`. Returns false on failure or where unsupported.
    protectRangeFn: *const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, prot: Prot) bool,
    freeFn: *const fn (ctx: ?*anyopaque, mem: ExecMemory) void,
};

/// The posix provider: `mmap` writable pages, `mprotect` to R+X, `munmap` to free.
pub const posix: Provider = .{ .allocFn = posixAlloc, .protectFn = posixProtect, .protectRangeFn = posixProtectRange, .freeFn = posixFree };

fn posixAlloc(_: ?*anyopaque, len: usize) ?ExecMemory {
    const m = std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return null;
    return .{ .ptr = m.ptr, .len = m.len };
}
fn posixProtect(_: ?*anyopaque, mem: ExecMemory) bool {
    const rc = std.posix.system.mprotect(mem.ptr, mem.len, .{ .READ = true, .EXEC = true });
    return std.posix.errno(rc) == .SUCCESS;
}
fn posixProtectRange(_: ?*anyopaque, ptr: [*]u8, len: usize, prot: Prot) bool {
    const flags: std.posix.system.PROT = switch (prot) {
        .rx => .{ .READ = true, .EXEC = true },
        .r => .{ .READ = true },
        .rw => .{ .READ = true, .WRITE = true },
    };
    // `ptr` is a sub-range of a prior `allocFn` reservation, page-aligned at both ends
    // (the `Provider.protectRangeFn` contract above), but its static type carries no
    // alignment. A libc-linked build resolves `mprotect` to the extern C signature,
    // which requires the pointer aligned at the type level, not just at runtime.
    const rc = std.posix.system.mprotect(@ptrCast(@alignCast(ptr)), len, flags);
    return std.posix.errno(rc) == .SUCCESS;
}
fn posixFree(_: ?*anyopaque, mem: ExecMemory) void {
    std.posix.munmap(mem.ptr[0..mem.len]);
}

const uefi = std.os.uefi;

/// The UEFI provider: boot-services pages (executable at boot, so protect is a no-op).
pub const uefi_provider: Provider = .{ .allocFn = uefiAlloc, .protectFn = uefiProtect, .protectRangeFn = uefiProtectRange, .freeFn = uefiFree };

fn uefiAlloc(_: ?*anyopaque, len: usize) ?ExecMemory {
    const bs = uefi.system_table.boot_services orelse return null;
    const pages = (len + 4095) / 4096;
    const mem = bs.allocatePages(.any, .loader_code, pages) catch return null;
    return .{ .ptr = @ptrCast(mem.ptr), .len = pages * 4096 };
}
fn uefiProtect(_: ?*anyopaque, _: ExecMemory) bool {
    return true;
}
fn uefiProtectRange(_: ?*anyopaque, _: [*]u8, _: usize, _: Prot) bool {
    // UEFI boot-services memory has no per-range permission API exposed here. A caller that
    // requires real W^X on this platform must not rely on this.
    return false;
}
fn uefiFree(_: ?*anyopaque, mem: ExecMemory) void {
    const bs = uefi.system_table.boot_services orelse return;
    const page_ptr: [*]align(4096) uefi.Page = @ptrCast(@alignCast(mem.ptr));
    // freeFn cannot propagate an error, since its signature returns void, and a leaked page at
    // JIT-buffer teardown is non-fatal. But the failure must be observable, not silently
    // swallowed.
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

/// A single contiguous W^X reservation carrying code, rodata, data, and bss for one JIT
/// image, laid out at page-aligned sub-offsets in that fixed order: code, rodata, data, bss. So
/// PC-relative references between code and data stay within one address range, never independent
/// mmaps. `map` allocates the whole reservation writable and copies the initial contents in.
/// `finalize` then applies final permissions per section: code R+X, rodata R-only, data and bss
/// stay R+W. It also syncs the instruction cache over the code span, so no code byte is ever both
/// writable and executable at once. This is parameterized by the arch instruction-cache sync
/// hook, same as `Buffer`.
pub fn MappedImage(comptime syncICache: fn (mem: []const u8) void) type {
    return struct {
        const Self = @This();

        memory: []align(page_align) u8,
        provider: Provider,
        code_off: usize,
        code_len: usize,
        rodata_off: usize,
        rodata_len: usize,
        data_off: usize,
        data_len: usize,
        bss_off: usize,
        bss_len: usize,
        /// Set by `finalize` once `rodata` has been dropped to read-only. This documents the
        /// W^X intent structurally. A real fault-on-write test lives at the native.zig level
        /// once a full JITed program exists to exercise it end to end.
        is_rodata_protected: bool = false,

        /// Reserve one page-aligned region sized for code, rodata, data, and bss, in that
        /// order. A zero-length section still gets a valid page-aligned offset, so address
        /// computation stays uniform. Copy code, rodata, and data into place, and zero the bss
        /// span. The whole reservation is still read and write. Call `finalize` to apply the
        /// final per-section W^X permissions.
        ///
        /// Section starts are aligned to `std.heap.pageSize()`, the *runtime* page
        /// granularity, not the comptime `page_align` minimum. `mprotect` requires its
        /// address argument aligned to the real OS page size. On a host where that is larger
        /// than `page_align`, for example aarch64 with 64 KiB pages vs. a 4 KiB
        /// `page_size_min`, aligning only to `page_align` leaves `finalize` failing with
        /// EINVAL. The runtime page size is always a power-of-two multiple of `page_align`,
        /// so every offset is still a multiple of `page_align` too.
        pub fn map(provider: Provider, code: []const u8, rodata: []const u8, data: []const u8, bss_len: usize) Error!Self {
            const psize = std.heap.pageSize();
            const code_off: usize = 0;
            const rodata_off = std.mem.alignForward(usize, code_off + code.len, psize);
            const data_off = std.mem.alignForward(usize, rodata_off + rodata.len, psize);
            const bss_off = std.mem.alignForward(usize, data_off + data.len, psize);
            const total = std.mem.alignForward(usize, bss_off + bss_len, psize);

            const em = provider.allocFn(provider.ctx, total) orelse return error.AllocFailed;
            const memory = em.ptr[0..em.len];

            @memcpy(memory[code_off..][0..code.len], code);
            @memcpy(memory[rodata_off..][0..rodata.len], rodata);
            @memcpy(memory[data_off..][0..data.len], data);
            @memset(memory[bss_off..][0..bss_len], 0);

            return .{
                .memory = memory,
                .provider = provider,
                .code_off = code_off,
                .code_len = code.len,
                .rodata_off = rodata_off,
                .rodata_len = rodata.len,
                .data_off = data_off,
                .data_len = data.len,
                .bss_off = bss_off,
                .bss_len = bss_len,
            };
        }

        /// Apply final W^X permissions: code becomes read+execute, rodata becomes
        /// read-only, data and bss stay read+write. Sections are page-aligned so each
        /// `protectRangeFn` call covers exactly one section with no bleed into its
        /// neighbor. Then synchronizes the instruction cache over the code span.
        pub fn finalize(self: *Self) Error!void {
            if (self.code_len != 0) {
                if (!self.provider.protectRangeFn(self.provider.ctx, self.memory.ptr + self.code_off, self.code_len, .rx))
                    return error.ProtectFailed;
            }
            if (self.rodata_len != 0) {
                if (!self.provider.protectRangeFn(self.provider.ctx, self.memory.ptr + self.rodata_off, self.rodata_len, .r))
                    return error.ProtectFailed;
            }
            self.is_rodata_protected = true;
            syncICache(self.memory[self.code_off..][0..self.code_len]);
        }

        /// A raw pointer to byte `off` within the reservation.
        pub fn ptr(self: *const Self, off: usize) [*]u8 {
            return self.memory.ptr + off;
        }

        pub fn deinit(self: *Self) void {
            self.provider.freeFn(self.provider.ctx, .{ .ptr = self.memory.ptr, .len = self.memory.len });
            self.memory = &.{};
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

test "MappedImage lays out and zeroes sections on page boundaries" {
    const code = [_]u8{0} ** 8;
    const ro = [_]u8{ 1, 2, 3, 4 };
    const da = [_]u8{ 5, 6, 7, 8, 9, 10, 11, 12 };
    var img = try MappedImage(noSync).map(default_provider, &code, &ro, &da, 16);
    defer img.deinit();
    try std.testing.expect(img.rodata_off % page_align == 0);
    try std.testing.expect(img.data_off % page_align == 0);
    try std.testing.expect(img.bss_off % page_align == 0);
    try std.testing.expectEqualSlices(u8, &ro, img.ptr(img.rodata_off)[0..4]);
    try std.testing.expectEqualSlices(u8, &da, img.ptr(img.data_off)[0..8]);
    for (img.ptr(img.bss_off)[0..16]) |b| try std.testing.expectEqual(@as(u8, 0), b);
}

// Regression note: a real per-arch `ret0` executable-proof test is deferred until a
// full JIT path already produces host machine code to run. Here `finalize` is exercised
// end to end against the real posix provider, not a mock. It actually calls `mprotect`
// to flip code to R+X and rodata to R-only, and this test asserts that call succeeds
// and leaves section contents intact. The W^X *fault-on-write* property for rodata is
// recorded structurally via `is_rodata_protected`, and proven with a real fault test
// once a full JITed program exists.
test "finalize applies W^X permissions without corrupting section contents" {
    const code = [_]u8{0xC3} ** 8; // filler bytes, not executed here
    const ro = [_]u8{ 1, 2, 3, 4 };
    var img = try MappedImage(noSync).map(default_provider, &code, &ro, &.{}, 0);
    defer img.deinit();
    try std.testing.expect(!img.is_rodata_protected);

    try img.finalize();

    try std.testing.expect(img.is_rodata_protected);
    try std.testing.expectEqualSlices(u8, &code, img.ptr(img.code_off)[0..code.len]);
    try std.testing.expectEqualSlices(u8, &ro, img.ptr(img.rodata_off)[0..4]);
}
