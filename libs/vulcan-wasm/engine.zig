//! WebAssembly engine: load a module, JIT it for the host, set up its linear
//! memory/globals/table/imports, and call its exports with a typed interface. Runnable
//! layer over `lower.zig` and `vulcan-target.native` (host JIT).
//!
//! Lowered functions take one hidden "context" pointer to four base pointers
//! (memory, globals, table, imports) at fixed offsets. `Instance` builds and owns that
//! context. Callers pass only the Wasm arguments.

const std = @import("std");
const lower = @import("lower.zig");
const vt = @import("vulcan-target");
const native = vt.native;

/// Pluggable executable-memory provider (e.g. UEFI boot services).
pub const Provider = native.Provider;
/// A block of executable memory handed out by a `Provider`.
pub const ExecMemory = vt.jit_platform.ExecMemory;
/// Provider for the current target (UEFI boot services on UEFI, posix mmap otherwise).
pub const default_provider = vt.jit_platform.default_provider;

pub const page_size: usize = 65536;

pub const Error = lower.Error || native.Error || error{ MissingExport, ImportCountMismatch };

/// An instantiated module: JITed code plus its mutable state.
pub const Instance = struct {
    allocator: std.mem.Allocator,
    module: lower.Module,
    jitted: native.JittedModule,
    memory: []u8,
    globals: []i64,
    table: []usize,
    imports: []usize,
    /// Base pointers passed to functions: [memory, globals, table, imports].
    context: [4]usize align(8),

    /// Load, JIT (host's default provider), and set up a module. `host_imports` gives
    /// the host address for each import in `module.imports` order. Caller owns it.
    pub fn instantiate(allocator: std.mem.Allocator, bytes: []const u8, host_imports: []const usize) Error!Instance {
        return instantiateWith(allocator, vt.jit_platform.default_provider, bytes, host_imports);
    }

    /// Like `instantiate`, but maps the JITed code from an explicit executable-memory
    /// `provider` (e.g. UEFI boot-services pages on a freestanding host).
    pub fn instantiateWith(allocator: std.mem.Allocator, provider: Provider, bytes: []const u8, host_imports: []const usize) Error!Instance {
        var module = try lower.module(allocator, bytes);
        errdefer module.deinit(allocator);

        var mfs: std.ArrayList(native.ModuleFunction) = .empty;
        defer mfs.deinit(allocator);
        for (module.functions) |*lf| try mfs.append(allocator, .{ .name = lf.name, .func = &lf.func });
        var jitted = try native.jitModuleWith(allocator, provider, mfs.items);
        errdefer jitted.deinit();

        // Linear memory: declared size, grown to cover the data segments, zeroed then
        // initialized.
        var mem_bytes: usize = @as(usize, module.min_pages) * page_size;
        for (module.data) |seg| mem_bytes = @max(mem_bytes, seg.offset + seg.bytes.len);
        const memory = try allocator.alloc(u8, mem_bytes);
        errdefer allocator.free(memory);
        @memset(memory, 0);
        for (module.data) |seg| @memcpy(memory[seg.offset..][0..seg.bytes.len], seg.bytes);

        const globals = try allocator.alloc(i64, module.globals.len);
        errdefer allocator.free(globals);
        for (module.globals, 0..) |g, i| globals[i] = g.value;

        // The function table holds the JITed address of each referenced function.
        const table = try allocator.alloc(usize, module.table.len);
        errdefer allocator.free(table);
        for (module.table, 0..) |func_idx, t| {
            const name = module.functions[func_idx].name;
            table[t] = @intFromPtr(jitted.entry(*const fn () callconv(.c) void, name) orelse return error.MissingExport);
        }

        if (host_imports.len != module.imports.len) return error.ImportCountMismatch;
        const imports = try allocator.dupe(usize, host_imports);
        errdefer allocator.free(imports);

        return .{
            .allocator = allocator,
            .module = module,
            .jitted = jitted,
            .memory = memory,
            .globals = globals,
            .table = table,
            .imports = imports,
            .context = .{ @intFromPtr(memory.ptr), @intFromPtr(globals.ptr), @intFromPtr(table.ptr), @intFromPtr(imports.ptr) },
        };
    }

    pub fn deinit(self: *Instance) void {
        self.jitted.deinit();
        self.module.deinit(self.allocator);
        self.allocator.free(self.memory);
        self.allocator.free(self.globals);
        self.allocator.free(self.table);
        self.allocator.free(self.imports);
    }

    /// Raw address of exported function `name`.
    fn addrOf(self: *Instance, name: []const u8) Error!usize {
        return @intFromPtr(self.jitted.entry(*const fn () callconv(.c) void, name) orelse return error.MissingExport);
    }
    fn ctxPtr(self: *Instance) *anyopaque {
        return @ptrCast(&self.context);
    }

    /// Call an exported function with 0/1/2/3 Wasm arguments of the given types,
    /// returning `Ret`. The hidden context pointer (when the module needs one) is
    /// supplied automatically.
    pub fn call0(self: *Instance, comptime Ret: type, name: []const u8) Error!Ret {
        const a = try self.addrOf(name);
        return if (self.module.needs_context)
            @as(*const fn (*anyopaque) callconv(.c) Ret, @ptrFromInt(a))(self.ctxPtr())
        else
            @as(*const fn () callconv(.c) Ret, @ptrFromInt(a))();
    }
    pub fn call1(self: *Instance, comptime Ret: type, comptime A0: type, name: []const u8, a0: A0) Error!Ret {
        const a = try self.addrOf(name);
        return if (self.module.needs_context)
            @as(*const fn (*anyopaque, A0) callconv(.c) Ret, @ptrFromInt(a))(self.ctxPtr(), a0)
        else
            @as(*const fn (A0) callconv(.c) Ret, @ptrFromInt(a))(a0);
    }
    pub fn call2(self: *Instance, comptime Ret: type, comptime A0: type, comptime A1: type, name: []const u8, a0: A0, a1: A1) Error!Ret {
        const a = try self.addrOf(name);
        return if (self.module.needs_context)
            @as(*const fn (*anyopaque, A0, A1) callconv(.c) Ret, @ptrFromInt(a))(self.ctxPtr(), a0, a1)
        else
            @as(*const fn (A0, A1) callconv(.c) Ret, @ptrFromInt(a))(a0, a1);
    }
    pub fn call3(self: *Instance, comptime Ret: type, comptime A0: type, comptime A1: type, comptime A2: type, name: []const u8, a0: A0, a1: A1, a2: A2) Error!Ret {
        const a = try self.addrOf(name);
        return if (self.module.needs_context)
            @as(*const fn (*anyopaque, A0, A1, A2) callconv(.c) Ret, @ptrFromInt(a))(self.ctxPtr(), a0, a1, a2)
        else
            @as(*const fn (A0, A1, A2) callconv(.c) Ret, @ptrFromInt(a))(a0, a1, a2);
    }
};

test {
    std.testing.refAllDecls(@This());
}
