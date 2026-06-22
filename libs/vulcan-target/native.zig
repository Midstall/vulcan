//! Native target: picks the Vulcan backend matching the host CPU at comptime, so a
//! JIT can compile for the running arch without branching on it. Used by in-process
//! JITs (e.g. a Wasm runtime) and the UEFI runtime.
//!
//! `compile` returns host machine code as bytes (normalizing the `[]u32` backends).
//! `jitFunction` maps that into a W^X executable buffer with the correct per-arch
//! instruction-cache handling and returns a callable buffer. Executable memory comes
//! from the hosted (posix) JIT buffer. A freestanding provider (UEFI boot-services
//! pages) slots in for the firmware runtime.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;

/// The host CPU architecture this build targets.
pub const arch = builtin.cpu.arch;

/// Runtime host-CPU feature detection. `arch` above is fixed at comptime (a JIT runs
/// on its own architecture), but extensions (AVX/NEON/SVE/RVV) vary per CPU and are
/// detected at runtime via `host.detect()`, so codegen targets the silicon executing
/// the binary rather than the machine it was built on.
pub const host = @import("host.zig");

/// The Vulcan backend for the host architecture.
pub const backend = switch (arch) {
    .aarch64 => @import("aarch64.zig"),
    .x86_64 => @import("x86_64.zig"),
    .x86 => @import("x86.zig"),
    .riscv64 => @import("riscv64.zig"),
    else => @compileError("vulcan-target.native: host arch '" ++ @tagName(arch) ++ "' is not a Vulcan target"),
};

/// A W^X executable buffer of host machine code.
pub const CodeBuffer = backend.jit.CodeBuffer;

pub const Error = backend.isel.Error || backend.jit.Error || backend.link.Error || std.mem.Allocator.Error;

/// Compile `func` to host machine code (bytes). The caller owns the slice. The
/// `[]u32`-emitting backends (aarch64, riscv64) are reinterpreted as bytes.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    const raw = try backend.isel.selectFunction(allocator, func);
    if (comptime @TypeOf(raw) == []u8) return raw;
    defer allocator.free(raw);
    return allocator.dupe(u8, std.mem.sliceAsBytes(raw));
}

/// Compile `func` and map it into an executable buffer ready to call.
pub fn jitFunction(allocator: std.mem.Allocator, func: *const Function) Error!CodeBuffer {
    const code = try compile(allocator, func);
    defer allocator.free(code);
    return CodeBuffer.map(code);
}

/// A named function to link into a module.
pub const ModuleFunction = struct { name: []const u8, func: *const Function };

/// A linked, mapped module: executable code plus the byte offset of each function.
pub const JittedModule = struct {
    buffer: CodeBuffer,
    symbols: []Symbol,
    allocator: std.mem.Allocator,

    pub const Symbol = struct { name: []const u8, offset: usize };

    pub fn deinit(self: *JittedModule) void {
        self.buffer.deinit();
        self.allocator.free(self.symbols);
    }

    /// A typed pointer to the function exported as `name`, if present.
    pub fn entry(self: *const JittedModule, comptime Fn: type, name: []const u8) ?Fn {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name)) return self.buffer.entry(Fn, s.offset);
        return null;
    }
};

/// The pluggable executable-memory provider type (e.g. for UEFI boot services).
pub const Provider = @import("jit_platform.zig").Provider;

/// Link `funcs` and map the result executable from the host's default (posix)
/// provider. The symbol names are borrowed from `funcs` and must outlive the result.
pub fn jitModule(allocator: std.mem.Allocator, funcs: []const ModuleFunction) Error!JittedModule {
    return jitModuleWith(allocator, @import("jit_platform.zig").default_provider, funcs);
}

/// Like `jitModule`, but maps the code from an explicit executable-memory `provider`
/// (e.g. UEFI boot-services pages on a freestanding host).
pub fn jitModuleWith(allocator: std.mem.Allocator, provider: Provider, funcs: []const ModuleFunction) Error!JittedModule {
    var m = backend.link.Module{};
    defer m.deinit(allocator);
    for (funcs) |f| try m.addFunction(allocator, f.name, f.func);

    var linked = try backend.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    const bytes = if (comptime @TypeOf(linked.code) == []u8) linked.code else std.mem.sliceAsBytes(linked.code);
    var buf = try CodeBuffer.mapWith(provider, bytes);
    errdefer buf.deinit();

    const symbols = try allocator.alloc(JittedModule.Symbol, linked.symbols.len);
    for (linked.symbols, 0..) |s, i| symbols[i] = .{ .name = s.name, .offset = s.offset };
    return .{ .buffer = buf, .symbols = symbols, .allocator = allocator };
}

test "native: compiles and runs a function in-process on the host" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const d = try func.appendArithImm(b, t, .mul, x, 2);
    const r = try func.appendArithImm(b, t, .add, d, 1);
    func.setTerminator(b, .{ .ret = r });

    var buf = try jitFunction(allocator, &func);
    defer buf.deinit();
    const f = buf.entry(*const fn (i64) callconv(.c) i64, 0);
    try std.testing.expectEqual(@as(i64, 41), f(20)); // 20*2 + 1
}
