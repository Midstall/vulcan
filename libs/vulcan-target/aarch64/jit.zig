//! In-process JIT for AArch64: link a module into one image, map it into W^X
//! executable memory (from a pluggable `jit_platform.Provider`), and hand back typed
//! function pointers. The host is aarch64, so the mapped code runs natively. The
//! image is position-independent (every `bl` is PC-relative), so its address is free.

const std = @import("std");
const builtin = @import("builtin");
const link = @import("link.zig");
const platform = @import("../jit_platform.zig");

pub const Error = link.Error || platform.Error;
pub const Provider = platform.Provider;

/// A W^X executable buffer that synchronizes the AArch64 instruction cache.
pub const CodeBuffer = platform.Buffer(syncICache);

/// Synchronize the instruction stream with freshly written code: clean the D-cache
/// and invalidate the I-cache to the point of unification (line sizes from `ctr_el0`).
/// A no-op off aarch64 (the code could not run there anyway).
fn syncICache(memory: []const u8) void {
    if (builtin.cpu.arch != .aarch64) return;
    const ctr = asm volatile ("mrs %[r], ctr_el0"
        : [r] "=r" (-> usize),
    );
    const dline = @as(usize, 4) << @intCast((ctr >> 16) & 0xf);
    const iline = @as(usize, 4) << @intCast(ctr & 0xf);
    const start = @intFromPtr(memory.ptr);
    const end = start + memory.len;
    var a = start & ~(dline - 1);
    while (a < end) : (a += dline) asm volatile ("dc cvau, %[a]"
        :
        : [a] "r" (a),
        : .{ .memory = true });
    asm volatile ("dsb ish" ::: .{ .memory = true });
    a = start & ~(iline - 1);
    while (a < end) : (a += iline) asm volatile ("ic ivau, %[a]"
        :
        : [a] "r" (a),
        : .{ .memory = true });
    asm volatile ("dsb ish" ::: .{ .memory = true });
    asm volatile ("isb" ::: .{ .memory = true });
}

/// A JIT-compiled module: its live executable buffer plus the linker symbols
/// (each function's byte offset into the buffer).
pub const Compiled = struct {
    buffer: CodeBuffer,
    linked: link.Linked,

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        self.linked.deinit(allocator);
        self.buffer.deinit();
    }

    /// A typed, callable pointer to the function named `name`, or null if not defined.
    pub fn funcPointer(self: *const Compiled, comptime Fn: type, name: []const u8) ?Fn {
        const offset = self.linked.addressOf(name) orelse return null;
        return self.buffer.entry(Fn, offset);
    }
};

/// JIT-compile a module: link it into one position-independent image and map that
/// image into executable memory. The caller owns the result.
pub fn compileModule(allocator: std.mem.Allocator, module: *const link.Module) Error!Compiled {
    var linked = try link.compileModule(allocator, module);
    errdefer linked.deinit(allocator);
    const buffer = try CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    return .{ .buffer = buffer, .linked = linked };
}
