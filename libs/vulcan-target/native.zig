//! Native target: picks the Vulcan backend matching the host CPU at comptime, so a
//! JIT can compile for the running arch without branching on it. Used by in-process
//! JITs (e.g. a Wasm runtime) and the UEFI runtime.
//!
//! `compile` returns host machine code as bytes (normalizing the `[]u32` backends).
//! `jitFunction` maps that into a W^X executable buffer with the correct per-arch
//! instruction-cache handling and returns a callable buffer. Executable memory comes
//! from the hosted (posix) JIT buffer. A freestanding provider (UEFI boot-services
//! pages) slots in for the firmware runtime.
//!
//! `jitModuleData` links a module's functions AND its named data globals (`.rodata`/
//! `.data`/`.bss` objects, referenced from code via `global_addr`) into one
//! `MappedImage`: real W^X per section (code R+X, rodata R-only, data/bss R+W), not
//! just documented intent. `jitModule`/`jitModuleWith` are the no-data case, kept
//! byte-identical by delegating here with an empty data set.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const link = @import("vulcan-link");
const mm = @import("vulcan-opt").microarch;

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

/// The host architecture (`arch` above) as `vulcan-link`'s `Arch` enum, for callers
/// (the regression test below, and `vcc`'s default `-target`) that need it in that
/// shape rather than `builtin.cpu.Arch`. `null` for a host arch with no Vulcan
/// backend - unreached in practice, since `backend` above already `@compileError`s
/// for those.
pub fn hostLinkArch() ?link.Arch {
    return switch (arch) {
        .aarch64 => .aarch64,
        .x86_64 => .x86_64,
        .x86 => .x86,
        .riscv64 => .riscv64,
        else => null,
    };
}

/// A W^X executable buffer of host machine code.
pub const CodeBuffer = backend.jit.CodeBuffer;

pub const Error = backend.isel.Error || backend.jit.Error || backend.link.Error || std.mem.Allocator.Error;

/// Compile `func` to host machine code (bytes). The caller owns the slice. The
/// `[]u32`-emitting backends (aarch64, riscv64) are reinterpreted as bytes.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    // The in-process JIT runs the code on THIS host, so it must be compiled for the host calling
    // convention. On an aarch64 Darwin host that is Apple's arm64 ABI (packed stack arguments), which
    // differs from AAPCS64 once arguments spill past the registers; a function JIT-executed there must
    // read and write its stack arguments the way the host caller passes them. Everywhere else the host
    // ABI is AAPCS64, so this is byte-identical to the plain `selectFunction` path.
    if (comptime arch == .aarch64) {
        const abi: backend.isel.Abi = if (builtin.os.tag.isDarwin()) .apple else .aapcs64;
        var compiled = try backend.isel.compileFunction(allocator, func, .{ .abi = abi });
        defer compiled.deinit(allocator);
        return allocator.dupe(u8, std.mem.sliceAsBytes(compiled.code));
    }
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

/// A named data global to link into a module: `.rodata` (kind = `.rodata`), `.data`
/// (kind = `.data`), or `.bss` (kind = `.bss`, `bytes` empty and `size` giving the
/// zero-initialized length; otherwise `size == bytes.len`). The name is what
/// `global_addr` in JITed code refers to it by, and what `JittedModule.dataAddr`
/// looks it up by. `name`/`bytes` are borrowed and must outlive the `JittedModule`.
/// `relocs` (default empty) are internal relocations WITHIN this object - a
/// pointer-valued global (e.g. `char *s = "hi";`) whose initial bytes must be
/// patched to another object's runtime address once the module is mapped (see
/// `backend.link.DataReloc`); `bss` objects carry none.
pub const ModuleData = struct { name: []const u8, bytes: []const u8, kind: backend.link.DataKind, size: u64, relocs: []const backend.link.DataReloc = &.{} };

/// A data object's section, arch-independent (mirrors every backend's own
/// `link.DataKind`, which are all the same 3-way enum - see `writeObjectDataFor`).
pub const DataKind = enum { rodata, data, bss };

/// An internal relocation within a data object's bytes, arch-independent (mirrors
/// every backend's own `link.DataReloc`).
pub const DataReloc = struct { off: usize, symbol: []const u8 };

/// A named data global to link into a module for `writeObjectDataFor`: the
/// arch-independent counterpart of `ModuleData` (which is pinned to the HOST
/// backend's `link.DataKind`/`link.DataReloc` types). A caller that doesn't know
/// yet which of the 4 backends it is targeting builds `ObjData` once and lets
/// `writeObjectDataFor` translate it into whichever backend's `link.Module` the
/// requested `target` needs.
pub const ObjData = struct { name: []const u8, bytes: []const u8, kind: DataKind, size: u64, relocs: []const DataReloc = &.{} };

/// A linked, mapped module: one `MappedImage` (code | rodata | data | bss, real
/// W^X per section) plus each function's byte offset and each data object's
/// resolved runtime address.
pub const JittedModule = struct {
    image: backend.jit.MappedImage,
    symbols: []Symbol,
    data: []DataAddr,
    allocator: std.mem.Allocator,

    pub const Symbol = struct { name: []const u8, offset: usize };
    /// A data object's resolved runtime address (its section's base offset plus
    /// its own within-section offset).
    pub const DataAddr = struct { name: []const u8, addr: [*]u8 };

    pub fn deinit(self: *JittedModule) void {
        self.image.deinit();
        self.allocator.free(self.symbols);
        self.allocator.free(self.data);
    }

    /// A typed pointer to the function exported as `name`, if present.
    pub fn entry(self: *const JittedModule, comptime Fn: type, name: []const u8) ?Fn {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name))
            return @ptrCast(@alignCast(self.image.ptr(self.image.code_off + s.offset)));
        return null;
    }

    /// The runtime address of the data object exported as `name`, if present (for
    /// tests and embedders that need to read or write a global directly).
    pub fn dataAddr(self: *const JittedModule, name: []const u8) ?[*]u8 {
        for (self.data) |d| if (std.mem.eql(u8, d.name, name)) return d.addr;
        return null;
    }
};

/// The pluggable executable-memory provider type (e.g. for UEFI boot services).
pub const Provider = @import("jit_platform.zig").Provider;

/// Link `funcs` (no data globals) and map the result executable from the host's
/// default (posix) provider. The symbol names are borrowed from `funcs` and must
/// outlive the result.
pub fn jitModule(allocator: std.mem.Allocator, funcs: []const ModuleFunction) Error!JittedModule {
    return jitModuleData(allocator, funcs, &.{});
}

/// Like `jitModule`, but maps the code from an explicit executable-memory `provider`
/// (e.g. UEFI boot-services pages on a freestanding host).
pub fn jitModuleWith(allocator: std.mem.Allocator, provider: Provider, funcs: []const ModuleFunction) Error!JittedModule {
    return jitModuleDataWith(allocator, provider, funcs, &.{});
}

/// Link `funcs` and `data` into one module and map the result (code R+X, rodata
/// R-only, data/bss R+W) from the host's default (posix) provider.
pub fn jitModuleData(allocator: std.mem.Allocator, funcs: []const ModuleFunction, data: []const ModuleData) Error!JittedModule {
    return jitModuleDataWith(allocator, @import("jit_platform.zig").default_provider, funcs, data);
}

/// Like `jitModuleData`, but maps from an explicit executable-memory `provider`
/// (e.g. UEFI boot-services pages on a freestanding host).
///
/// Builds a `backend.link.Module` from `funcs`/`data`, compiles and links it, lays
/// out each section's bytes at its data objects' (already section-aligned)
/// offsets, maps the whole image, resolves every data object's runtime address,
/// and patches each carried-forward `global_addr` relocation to its symbol's
/// resolved address (a data object or, degenerately, a function). A `global_addr`
/// naming neither fails closed with `error.UndefinedSymbol` rather than leaving a
/// placeholder address live.
pub fn jitModuleDataWith(allocator: std.mem.Allocator, provider: Provider, funcs: []const ModuleFunction, data: []const ModuleData) Error!JittedModule {
    var m = backend.link.Module{};
    defer m.deinit(allocator);
    for (funcs) |f| try m.addFunction(allocator, f.name, f.func);
    for (data) |d| switch (d.kind) {
        .rodata => try m.addDataRelocs(allocator, d.name, d.bytes, d.relocs),
        .data => try m.addWritableRelocs(allocator, d.name, d.bytes, d.relocs),
        .bss => try m.addBss(allocator, d.name, d.size),
    };

    var linked = try backend.link.compileModule(allocator, &m);
    defer linked.deinit(allocator);

    // Each section's length is the end of its furthest-placed object (`DataSym.off`
    // is already aligned within its section; a length of 0 is fine - `MappedImage`
    // still gives an empty section a valid page-aligned offset).
    var rodata_len: usize = 0;
    var data_len: usize = 0;
    var bss_len: usize = 0;
    for (linked.data) |d| switch (d.kind) {
        .rodata => rodata_len = @max(rodata_len, d.off + d.size),
        .data => data_len = @max(data_len, d.off + d.size),
        .bss => bss_len = @max(bss_len, d.off + d.size),
    };

    // Concatenate each section's object bytes at their aligned offsets, leaving any
    // padding gaps zeroed (bss is zeroed wholesale by `MappedImage.map` itself).
    const rodata_bytes = try allocator.alloc(u8, rodata_len);
    defer allocator.free(rodata_bytes);
    @memset(rodata_bytes, 0);
    const data_bytes = try allocator.alloc(u8, data_len);
    defer allocator.free(data_bytes);
    @memset(data_bytes, 0);
    for (linked.data) |d| switch (d.kind) {
        .rodata => @memcpy(rodata_bytes[d.off..][0..d.size], d.bytes),
        .data => @memcpy(data_bytes[d.off..][0..d.size], d.bytes),
        .bss => {},
    };

    const code_bytes = if (comptime @TypeOf(linked.code) == []u8) linked.code else std.mem.sliceAsBytes(linked.code);
    var image = try backend.jit.MappedImage.map(provider, code_bytes, rodata_bytes, data_bytes, bss_len);
    errdefer image.deinit();

    // Resolve each data object's runtime address: its section's base offset plus
    // its own within-section offset.
    const data_addrs = try allocator.alloc(JittedModule.DataAddr, linked.data.len);
    errdefer allocator.free(data_addrs);
    for (linked.data, 0..) |d, i| {
        const section_off = switch (d.kind) {
            .rodata => image.rodata_off,
            .data => image.data_off,
            .bss => image.bss_off,
        };
        data_addrs[i] = .{ .name = d.name, .addr = image.ptr(section_off + d.off) };
    }

    // Patch every data object's own internal relocations (a pointer-valued global
    // whose initial bytes must hold another object's runtime address, e.g. `char
    // *s = "hi";`): write the pointer-sized absolute address of `reloc.symbol` at
    // `section_base + datasym.off + reloc.off`. This runs BEFORE `image.finalize()`
    // while every section (including rodata) is still writable - a rodata object
    // (e.g. a `const char *`) can itself carry a reloc and must be patched before
    // rodata is frozen read-only.
    for (linked.data) |d| {
        if (d.relocs.len == 0) continue;
        const section_off = switch (d.kind) {
            .rodata => image.rodata_off,
            .data => image.data_off,
            .bss => image.bss_off,
        };
        for (d.relocs) |reloc| {
            const target: [*]u8 = blk: {
                for (data_addrs) |da| if (std.mem.eql(u8, da.name, reloc.symbol)) break :blk da.addr;
                for (linked.symbols) |s| if (std.mem.eql(u8, s.name, reloc.symbol)) break :blk image.ptr(image.code_off + s.offset);
                return error.UndefinedSymbol;
            };
            const site = image.ptr(section_off + d.off + reloc.off)[0..@sizeOf(usize)];
            std.mem.writeInt(usize, site, @intFromPtr(target), .little);
        }
    }

    // Patch every carried-forward `global_addr` relocation now that its target has
    // a runtime address: look the symbol up in data first, then functions (a
    // `global_addr` could in principle name either).
    for (linked.relocs) |r| {
        const target: [*]u8 = blk: {
            for (data_addrs) |d| if (std.mem.eql(u8, d.name, r.symbol)) break :blk d.addr;
            for (linked.symbols) |s| if (std.mem.eql(u8, s.name, r.symbol)) break :blk image.ptr(image.code_off + s.offset);
            return error.UndefinedSymbol;
        };
        // `r.offset` is word-indexed on the `[]u32`-emitting backends (aarch64,
        // riscv64) but already byte-indexed where code is `[]u8` (x86-64, x86-32) -
        // the same distinction `code_bytes` above normalizes for the mapped image.
        const reloc_scale: usize = if (comptime @TypeOf(linked.code) == []u8) 1 else 4;
        const site_addr = @intFromPtr(image.ptr(image.code_off + r.offset * reloc_scale));
        backend.link.applyGlobalReloc(&image, r, site_addr, @intFromPtr(target));
    }

    try image.finalize();

    const symbols = try allocator.alloc(JittedModule.Symbol, linked.symbols.len);
    errdefer allocator.free(symbols);
    for (linked.symbols, 0..) |s, i| symbols[i] = .{ .name = s.name, .offset = s.offset };

    return .{ .image = image, .symbols = symbols, .data = data_addrs, .allocator = allocator };
}

/// Link `funcs` and `data` into one `backend.link.Module` (mirroring
/// `jitModuleDataWith`'s module-build) and serialize it to a real ELF relocatable
/// object (`.o` bytes) via the host backend's `object.writeModule`, instead of
/// compiling and mapping it in-process. The caller owns the returned bytes.
pub fn writeObjectData(allocator: std.mem.Allocator, funcs: []const ModuleFunction, data: []const ModuleData) Error![]u8 {
    var m = backend.link.Module{};
    defer m.deinit(allocator);
    for (funcs) |f| try m.addFunction(allocator, f.name, f.func);
    for (data) |d| switch (d.kind) {
        .rodata => try m.addDataRelocs(allocator, d.name, d.bytes, d.relocs),
        .data => try m.addWritableRelocs(allocator, d.name, d.bytes, d.relocs),
        .bss => try m.addBss(allocator, d.name, d.size),
    };
    return backend.object.writeModule(allocator, &m);
}

/// Like `writeObjectData`, but emits a real ELF relocatable object for `target`
/// REGARDLESS of the host architecture - this is what lets `vcc -target <arch>`
/// cross-emit objects for an arch other than the one `vcc` itself runs on.
///
/// Unlike `backend` above (pinned to the host at comptime), all 4 backend modules
/// (aarch64/x86_64/x86/riscv64) are importable on any host; only the JIT/in-process
/// paths (which actually execute the generated code) need to be host-pinned. Each
/// switch arm builds THAT backend's own `link.Module` via its `add*` methods (the
/// methods are uniform across all 4 - the only difference is which module's `Module`/
/// `object.writeModule` gets called) from the neutral `funcs`/`data`, so the caller
/// builds its inputs once and can target any of the 4 without branching itself.
pub fn writeObjectDataFor(allocator: std.mem.Allocator, target: link.Arch, funcs: []const ModuleFunction, data: []const ObjData) Error![]u8 {
    return writeObjectDataForModel(allocator, target, funcs, data, null);
}

/// Like `writeObjectDataFor`, but selects code for a specific microarch `model` on
/// the 3 model-capable backends (aarch64, x86_64, riscv64). The caller guarantees
/// `model.?.arch` matches `target`, so this does not re-check the arch itself. `x86`
/// (32-bit) has no model support and always takes the generic path. A null `model`
/// keeps every backend on its generic path, so `writeObjectDataFor` above is exactly
/// this function called with `null`, so it stays byte-identical to before it existed.
pub fn writeObjectDataForModel(allocator: std.mem.Allocator, target: link.Arch, funcs: []const ModuleFunction, data: []const ObjData, model: ?*const mm.Model) Error![]u8 {
    return switch (target) {
        .aarch64 => writeObjectDataWithModel(@import("aarch64.zig"), allocator, funcs, data, model),
        .x86_64 => writeObjectDataWithModel(@import("x86_64.zig"), allocator, funcs, data, model),
        .riscv64 => writeObjectDataWithModel(@import("riscv64.zig"), allocator, funcs, data, model),
        .x86 => writeObjectDataWith(@import("x86.zig"), allocator, funcs, data), // 32-bit x86 has no model
    };
}

/// `writeObjectDataWith`'s module-assembly step, factored out so a test can inspect
/// the built module's `data.items[i].relocs` directly (see the regression test
/// below) instead of only being able to observe it indirectly through whatever
/// `B.object.writeModule` happens to do with it. Builds backend module `B`'s own
/// `link.Module` from the neutral `funcs`/`data`, translating `DataKind`/`DataReloc`
/// into `B.link`'s own, structurally-identical types.
///
/// `m.addDataRelocs`/`m.addWritableRelocs` BORROW the `relocs` slice they're given
/// (they store it, not copy it) - a caller holding onto the returned module (here,
/// `writeObjectDataWith`'s later `B.object.writeModule` call) reads it back out
/// later. So every translated `relocs` slice allocated in the loop below must stay
/// alive for as long as the returned module is in use, not just through its own
/// loop iteration - that premature per-iteration free was the bug this shape
/// guards against. `allocated` collects each translated slice so the caller can
/// free them once it is done with the module, via the returned value's `deinit`.
fn buildBackendModule(comptime B: type, allocator: std.mem.Allocator, funcs: []const ModuleFunction, data: []const ObjData) Error!struct {
    module: B.link.Module,
    allocated: std.ArrayListUnmanaged([]B.link.DataReloc),

    fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
        self.module.deinit(alloc);
        for (self.allocated.items) |relocs| alloc.free(relocs);
        self.allocated.deinit(alloc);
    }
} {
    var m: B.link.Module = .{};
    errdefer m.deinit(allocator);
    var allocated: std.ArrayListUnmanaged([]B.link.DataReloc) = .empty;
    errdefer {
        for (allocated.items) |relocs| allocator.free(relocs);
        allocated.deinit(allocator);
    }

    for (funcs) |f| try m.addFunction(allocator, f.name, f.func);
    for (data) |d| {
        if (d.kind == .bss) {
            try m.addBss(allocator, d.name, d.size);
            continue;
        }
        var relocs: []B.link.DataReloc = &.{};
        if (d.relocs.len != 0) {
            relocs = try allocator.alloc(B.link.DataReloc, d.relocs.len);
            for (d.relocs, relocs) |r, *o| o.* = .{ .off = r.off, .symbol = r.symbol };
            try allocated.append(allocator, relocs);
        }
        switch (d.kind) {
            .rodata => try m.addDataRelocs(allocator, d.name, d.bytes, relocs),
            .data => try m.addWritableRelocs(allocator, d.name, d.bytes, relocs),
            .bss => unreachable,
        }
    }
    return .{ .module = m, .allocated = allocated };
}

/// Shared body of `writeObjectDataFor`'s 4 branches: `buildBackendModule` then
/// serialize the result with `B`'s own `object.writeModule`. `B` is
/// `aarch64.zig`/`x86_64.zig`/`x86.zig`/`riscv64.zig`. The built module (and every
/// translated reloc slice it borrows) stays alive - via `defer built.deinit(...)`,
/// which runs after the `return` expression below is evaluated, not before - for
/// the whole `writeModule` call.
fn writeObjectDataWith(comptime B: type, allocator: std.mem.Allocator, funcs: []const ModuleFunction, data: []const ObjData) Error![]u8 {
    var built = try buildBackendModule(B, allocator, funcs, data);
    defer built.deinit(allocator);
    return B.object.writeModule(allocator, &built.module);
}

/// Like `writeObjectDataWith`, but for one of the 3 model-capable backends (`B` is
/// `aarch64.zig`/`x86_64.zig`/`riscv64.zig`): sets the built module's `model` before
/// serializing, so `B.object.writeModule` selects code tuned for it. A null `model`
/// leaves the built module's `model` at its default `null`, so this is
/// byte-identical to `writeObjectDataWith` in that case.
fn writeObjectDataWithModel(comptime B: type, allocator: std.mem.Allocator, funcs: []const ModuleFunction, data: []const ObjData, model: ?*const mm.Model) Error![]u8 {
    var built = try buildBackendModule(B, allocator, funcs, data);
    defer built.deinit(allocator);
    built.module.model = model;
    return B.object.writeModule(allocator, &built.module);
}

/// A `std.mem.Allocator` decorator that overwrites freed memory with a fixed
/// poison byte before actually freeing it through `child`, used only by the
/// regression test below. Rather than relying on `std.testing.allocator`'s own
/// (less predictable - it doesn't scrub freed content, only bucket metadata; see
/// `std.heap.DebugAllocator`) use-after-free behavior, this makes a premature free
/// deterministically corrupt whatever is later read through the dangling slice.
const PoisonOnFree = struct {
    child: std.mem.Allocator,

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        return self.child.vtable.alloc(self.child.ptr, len, alignment, ret_addr);
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        return self.child.vtable.resize(self.child.ptr, memory, alignment, new_len, ret_addr);
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        return self.child.vtable.remap(self.child.ptr, memory, alignment, new_len, ret_addr);
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        @memset(memory, 0xee);
        self.child.vtable.free(self.child.ptr, memory, alignment, ret_addr);
    }

    fn allocator(self: *PoisonOnFree) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
};

test "buildBackendModule: a translated data reloc survives past its own loop iteration" {
    // Regression for the Critical finding on this code: the reloc-translation loop
    // in `buildBackendModule` used to `defer allocator.free(relocs)` INSIDE the
    // `for (data)` loop, so each translated `relocs` slice was freed at the end of
    // ITS OWN iteration - before the caller (`writeObjectDataWith`'s later
    // `B.object.writeModule` call) could read it back out of the module it's
    // borrowed into. `object.writeModule` on every backend happens to never read a
    // data object's `.relocs` today (only the in-process JIT linker,
    // `link.compileModule`, does), so that specific downstream call can't observe
    // the bug either way - this test instead directly proves the invariant
    // `writeObjectDataWith` depends on: the built module's `data.items[i].relocs`
    // is still exactly what was requested by the time the caller is done reading
    // it. Uses `PoisonOnFree` so the check is deterministic rather than relying on
    // `std.testing.allocator` happening to leave old content in place.
    //
    // Confirmed this binds: temporarily restoring the old per-iteration
    // `defer allocator.free(relocs)` (removing `allocated`/its outer `defer`) makes
    // this test fail - `d.relocs[0].symbol` reads back as 16 poisoned (`0xee`)
    // bytes reinterpreted as a `[]const u8`, so `expectEqualStrings` fails
    // (and would just as easily crash on an unluckier poisoned pointer/len).
    const gpa = std.testing.allocator;
    var poison = PoisonOnFree{ .child = gpa };
    const allocator = poison.allocator();

    // riscv64, matching the real `writeObjectDataFor(.riscv64, ...)` path - not
    // load-bearing for THIS test (which never reaches `object.writeModule`, so
    // which backend is picked doesn't change what's under test here). Separately
    // discovered while fixing this: NONE of the 4 backends' `object.writeModule`
    // actually reads a data object's `.relocs` yet (only the in-process JIT
    // linker, `link.compileModule`, does) - not just aarch64, as the P2a note
    // assumed. That's a real, pre-existing gap (a data-object internal reloc is
    // silently dropped from every emitted `.o`), but a separate one from the
    // use-after-free this test guards against, and out of scope for this fix.
    const B = @import("riscv64.zig");
    const g_bytes = [_]u8{0} ** 4;
    const p_bytes = [_]u8{0} ** 8;

    var built = try buildBackendModule(B, allocator, &.{}, &.{
        .{ .name = "g", .bytes = &g_bytes, .kind = .data, .size = g_bytes.len },
        .{ .name = "p", .bytes = &p_bytes, .kind = .data, .size = p_bytes.len, .relocs = &.{.{ .off = 0, .symbol = "g" }} },
    });
    defer built.deinit(allocator);

    var found = false;
    for (built.module.data.items) |d| {
        if (!std.mem.eql(u8, d.name, "p")) continue;
        found = true;
        try std.testing.expectEqual(@as(usize, 1), d.relocs.len);
        try std.testing.expectEqual(@as(usize, 0), d.relocs[0].off);
        try std.testing.expectEqualStrings("g", d.relocs[0].symbol);
    }
    try std.testing.expect(found);
}

test "writeObjectDataFor(hostLinkArch, ...) is byte-identical to writeObjectData on the host" {
    // Regression: `writeObjectData` (host-pinned, `backend.link.Module` directly)
    // must stay byte-for-byte unchanged now that `writeObjectDataFor` exists as a
    // second, more general path to the same object bytes for the SAME arch.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const g = try func.appendGlobalAddr(b, ptr_t, "g");
    const v = try func.appendInst(b, t, .{ .load = .{ .ptr = g } });
    const r = try func.appendArithImm(b, t, .add, v, 1);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    const g_bytes = [_]u8{ 5, 0, 0, 0 };
    const old_bytes = try writeObjectData(
        allocator,
        &.{.{ .name = "f", .func = &func }},
        &.{.{ .name = "g", .bytes = &g_bytes, .kind = .data, .size = g_bytes.len }},
    );
    defer allocator.free(old_bytes);

    const host_arch = hostLinkArch().?;
    const new_bytes = try writeObjectDataFor(
        allocator,
        host_arch,
        &.{.{ .name = "f", .func = &func }},
        &.{.{ .name = "g", .bytes = &g_bytes, .kind = .data, .size = g_bytes.len }},
    );
    defer allocator.free(new_bytes);

    try std.testing.expectEqualSlices(u8, old_bytes, new_bytes);
}

test "writeObjectDataForModel(model=null) is byte-identical to writeObjectDataFor, on every model-capable arch" {
    // Regression: `writeObjectDataForModel` is the new, more general entry point that
    // `writeObjectDataFor` now delegates to with a null model. A null model must keep
    // every model-capable backend (aarch64, x86_64, riscv64) on its untuned, generic
    // path. This invariant protects existing `.o` output from this change.
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const g = try func.appendGlobalAddr(b, ptr_t, "g");
    const v = try func.appendInst(b, t, .{ .load = .{ .ptr = g } });
    const r = try func.appendArithImm(b, t, .add, v, 1);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    const g_bytes = [_]u8{ 5, 0, 0, 0 };
    const funcs = &[_]ModuleFunction{.{ .name = "f", .func = &func }};
    const data = &[_]ObjData{.{ .name = "g", .bytes = &g_bytes, .kind = .rodata, .size = g_bytes.len }};

    const model_capable_arches = [_]link.Arch{ .aarch64, .x86_64, .riscv64 };
    for (model_capable_arches) |a| {
        const via_for = try writeObjectDataFor(allocator, a, funcs, data);
        defer allocator.free(via_for);
        const via_for_model = try writeObjectDataForModel(allocator, a, funcs, data, null);
        defer allocator.free(via_for_model);
        try std.testing.expectEqualSlices(u8, via_for, via_for_model);
    }
}

test "writeObjectDataForModel tunes aarch64 output for ampere-altra: loop-header alignment changes the bytes" {
    // Preferred "model changes output" proof (see the task brief): a fusible
    // compare-and-branch (`icmp` feeding `if`) alone does NOT differ, because base-ISA
    // compare-into-branch fusion is already the default (`ModelCaps.fuse_cmp_branch`
    // defaults to true, see aarch64/isel.zig's `ModelCaps` doc comment). The
    // ampere-altra model's real point of difference from the default caps is its
    // `fetch_align = 32` (aarch64/isel.zig, `capsForModel`), which only has an effect
    // on a genuine LOOP HEADER block. So this kernel places the icmp-feeding-if at a
    // loop header (a back edge from `body` to `loop`), mirroring
    // tests/microarch_e2e.zig's `buildSumLoop`: `for (i = 0; i < n; i++) s += i;`.
    if (mm.modelFor(.@"ampere-altra").arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const i64_t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 64 } };

    var func = Function.init(allocator);
    defer func.deinit();
    const i64_t = try func.types.intern(i64_t_kind);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    const i = try func.appendBlockParam(loop, i64_t);
    const s = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i64_t);
    const bs = try func.appendBlockParam(body, i64_t);
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done });
    const ns = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = bi } });
    const ni = try func.appendArithImm(body, i64_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(s) });

    const funcs = &[_]ModuleFunction{.{ .name = "f", .func = &func }};
    const untuned = try writeObjectDataForModel(allocator, .aarch64, funcs, &.{}, null);
    defer allocator.free(untuned);
    const tuned = try writeObjectDataForModel(allocator, .aarch64, funcs, &.{}, mm.modelFor(.@"ampere-altra"));
    defer allocator.free(tuned);

    try std.testing.expect(!std.mem.eql(u8, untuned, tuned));
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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var buf = try jitFunction(allocator, &func);
    defer buf.deinit();
    const f = buf.entry(*const fn (i64) callconv(.c) i64, 0);
    try std.testing.expectEqual(@as(i64, 41), f(20)); // 20*2 + 1
}

test "native: jitModule links functions with no data globals" {
    // The no-data path: `jitModule` delegates to `jitModuleData` with an empty
    // data set. This is a direct regression test that delegation still runs.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const five = try func.appendInst(b, t, .{ .iconst = 5 });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(five) });

    var jm = try jitModule(allocator, &.{.{ .name = "f", .func = &func }});
    defer jm.deinit();
    const f = jm.entry(*const fn () callconv(.c) i32, "f").?;
    try std.testing.expectEqual(@as(i32, 5), f());
}

// The tests below prove the full JIT data path end to end on the host: a
// hand-built IR module with `.rodata`/`.data` objects and `global_addr` references
// really runs, and rodata is really mapped read-only (not just documented as such).
// Host-gated: the aarch64 isel (`global_addr` -> adrp+add) and link.zig (`Data`/
// `DataSym`/`applyGlobalReloc`) support this only on aarch64 today.

test "jitModuleData: reads a rodata global at its resolved runtime address" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var f = Function.init(allocator);
    defer f.deinit();
    const ptr_t = try f.types.intern(.ptr);
    const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try f.appendBlock();
    const g = try f.appendGlobalAddr(b, ptr_t, "g");
    const v = try f.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    const bytes = [_]u8{42};
    var jm = try jitModuleData(
        allocator,
        &.{.{ .name = "f", .func = &f }},
        &.{.{ .name = "g", .bytes = &bytes, .kind = .rodata, .size = bytes.len }},
    );
    defer jm.deinit();

    const entry_fn = jm.entry(*const fn () callconv(.c) i8, "f").?;
    try std.testing.expectEqual(@as(i8, 42), entry_fn());
}

test "jitModuleData: a .data global is written by one function and read by another" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // setw() { *(i32*)&w = 7; } - a void-returning store into the writable global.
    var setw = Function.init(allocator);
    defer setw.deinit();
    {
        const ptr_t = try setw.types.intern(.ptr);
        const i32_t = try setw.types.intern(i32k);
        const b = try setw.appendBlock();
        const w = try setw.appendGlobalAddr(b, ptr_t, "w");
        const seven = try setw.appendInst(b, i32_t, .{ .iconst = 7 });
        try setw.appendStore(b, seven, w);
        setw.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    }

    // f() { setw(); return *(i32*)&w; } - calls setw, then reads the global back.
    var f = Function.init(allocator);
    defer f.deinit();
    {
        const ptr_t = try f.types.intern(.ptr);
        const i32_t = try f.types.intern(i32k);
        const b = try f.appendBlock();
        try f.appendVoidCall(b, "setw", &.{});
        const w = try f.appendGlobalAddr(b, ptr_t, "w");
        const r = try f.appendInst(b, i32_t, .{ .load = .{ .ptr = w } });
        f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }

    const zeros = [_]u8{0} ** 4;
    var jm = try jitModuleData(
        allocator,
        &.{ .{ .name = "setw", .func = &setw }, .{ .name = "f", .func = &f } },
        &.{.{ .name = "w", .bytes = &zeros, .kind = .data, .size = zeros.len }},
    );
    defer jm.deinit();

    const entry_fn = jm.entry(*const fn () callconv(.c) i32, "f").?;
    try std.testing.expectEqual(@as(i32, 7), entry_fn());
}

test "jitModuleData: a .data global's internal reloc is patched to another object's runtime address" {
    // p (.data, 8 zero bytes) carries a `DataReloc` at offset 0 naming rodata
    // object s ("hi\0"). f() derefs p as a pointer-to-pointer then a byte: this
    // only returns 'h' if `p`'s bytes were really patched to `s`'s runtime
    // address before the image was finalized (proving the reloc, not `global_addr`
    // resolution alone - `p` itself holds no `global_addr` reference to `s`).
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var f = Function.init(allocator);
    defer f.deinit();
    const ptr_t = try f.types.intern(.ptr);
    const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try f.appendBlock();
    const p_addr = try f.appendGlobalAddr(b, ptr_t, "p"); // &p
    const s_addr = try f.appendInst(b, ptr_t, .{ .load = .{ .ptr = p_addr } }); // *(char **)&p, i.e. the patched &s
    const v = try f.appendInst(b, i8_t, .{ .load = .{ .ptr = s_addr } }); // *s
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    const s_bytes = [_]u8{ 'h', 'i', 0 };
    const p_bytes = [_]u8{0} ** 8;
    var jm = try jitModuleData(
        allocator,
        &.{.{ .name = "f", .func = &f }},
        &.{
            .{ .name = "s", .bytes = &s_bytes, .kind = .rodata, .size = s_bytes.len },
            .{ .name = "p", .bytes = &p_bytes, .kind = .data, .size = p_bytes.len, .relocs = &.{.{ .off = 0, .symbol = "s" }} },
        },
    );
    defer jm.deinit();

    const entry_fn = jm.entry(*const fn () callconv(.c) i8, "f").?;
    try std.testing.expectEqual(@as(i8, 104), entry_fn()); // 'h'
}

/// The `rwxp` permission field of the `/proc/self/maps` line whose address range
/// contains `addr`, or null if no mapping covers it. Each line begins `start-end
/// perms ...` with hex `start`/`end` and a 4-char `perms` like `r-xp`. Linux-only.
///
/// Read via an explicit `read` loop, NOT `readFileAlloc`: `/proc/self/maps` reports
/// `st_size == 0`, so a size-trusting reader yields an empty buffer.
fn procMapsPerms(allocator: std.mem.Allocator, addr: usize) !?[4]u8 {
    const linux = std.os.linux;
    const fd_rc = linux.openat(linux.AT.FDCWD, "/proc/self/maps", .{ .ACCMODE = .RDONLY }, 0);
    if (linux.errno(fd_rc) != .SUCCESS) return error.OpenFailed;
    const fd: i32 = @intCast(fd_rc);
    defer _ = linux.close(fd);

    var maps: std.ArrayListUnmanaged(u8) = .empty;
    defer maps.deinit(allocator);
    var chunk: [4096]u8 = undefined;
    while (true) {
        const rc = linux.read(fd, &chunk, chunk.len);
        if (linux.errno(rc) != .SUCCESS) return error.ReadFailed;
        if (rc == 0) break;
        try maps.appendSlice(allocator, chunk[0..rc]);
    }
    var lines = std.mem.splitScalar(u8, maps.items, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const dash = std.mem.indexOfScalar(u8, line, '-') orelse continue;
        const sp = std.mem.indexOfScalarPos(u8, line, dash, ' ') orelse continue;
        const start = std.fmt.parseInt(usize, line[0..dash], 16) catch continue;
        const end = std.fmt.parseInt(usize, line[dash + 1 .. sp], 16) catch continue;
        if (addr < start or addr >= end) continue;
        const perms = line[sp + 1 ..];
        if (perms.len < 4) continue;
        return perms[0..4].*;
    }
    return null;
}

test "jitModuleData: sections are mapped W^X (rodata r--, code r-x, data rw-)" {
    // Deterministic W^X proof via `/proc/self/maps` rather than a fork/SIGSEGV
    // probe (whose signal disposition survives `fork` under the debug segfault
    // handler and is not reproducible when the test binary is run directly). This
    // asserts the actual page permissions the kernel recorded after `finalize`.
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // f() { return *(i8*)&g; } - keeps a rodata global live; plus a `.data` object
    // `w` so all three section kinds (code/rodata/data) exist to inspect.
    var f = Function.init(allocator);
    defer f.deinit();
    const ptr_t = try f.types.intern(.ptr);
    const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try f.appendBlock();
    const g = try f.appendGlobalAddr(b, ptr_t, "g");
    const v = try f.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    const ro = [_]u8{42};
    const wr = [_]u8{0} ** 4;
    var jm = try jitModuleData(
        allocator,
        &.{.{ .name = "f", .func = &f }},
        &.{
            .{ .name = "g", .bytes = &ro, .kind = .rodata, .size = ro.len },
            .{ .name = "w", .bytes = &wr, .kind = .data, .size = wr.len },
        },
    );
    defer jm.deinit();

    // Sanity: the mapping really is live and correct before inspecting permissions.
    const entry_fn = jm.entry(*const fn () callconv(.c) i8, "f").?;
    try std.testing.expectEqual(@as(i8, 42), entry_fn());

    // rodata is read-only (the W^X property under test): r, NOT w, NOT x.
    const ro_perms = (try procMapsPerms(allocator, @intFromPtr(jm.dataAddr("g").?))).?;
    try std.testing.expectEqualSlices(u8, "r--", ro_perms[0..3]);

    // code is read + execute, NOT writable.
    const code_perms = (try procMapsPerms(allocator, @intFromPtr(entry_fn))).?;
    try std.testing.expectEqualSlices(u8, "r-x", code_perms[0..3]);

    // a `.data` object is read + write, NOT executable.
    const data_perms = (try procMapsPerms(allocator, @intFromPtr(jm.dataAddr("w").?))).?;
    try std.testing.expectEqualSlices(u8, "rw-", data_perms[0..3]);
}

// Host-gated: riscv64's `global_addr` isel (auipc+addi, pcrel_hi20/pcrel_lo12) and
// link.zig's `applyGlobalReloc` support in-process JIT parity with the other three
// backends (task 8). Mirrors the aarch64 rodata-read test above; the paired-reloc
// carry/patch logic itself is also covered by a non-executing structural test in
// riscv64/link.zig that runs on any host (this one only runs, and proves execution,
// on an actual riscv64 host).
test "jitModuleData: riscv64 host reads a rodata global at its resolved runtime address" {
    if (builtin.cpu.arch != .riscv64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var f = Function.init(allocator);
    defer f.deinit();
    const ptr_t = try f.types.intern(.ptr);
    const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try f.appendBlock();
    const g = try f.appendGlobalAddr(b, ptr_t, "g");
    const v = try f.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    const bytes = [_]u8{42};
    var jm = try jitModuleData(
        allocator,
        &.{.{ .name = "f", .func = &f }},
        &.{.{ .name = "g", .bytes = &bytes, .kind = .rodata, .size = bytes.len }},
    );
    defer jm.deinit();

    const entry_fn = jm.entry(*const fn () callconv(.c) i8, "f").?;
    try std.testing.expectEqual(@as(i8, 42), entry_fn());
}
