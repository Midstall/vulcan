//! In-process JIT: link a module into executable memory and hand back a callable
//! function pointer. Hosted layer (touches OS memory via mmap), kept out of the
//! freestanding core.
//!
//! The memory machinery is architecture-agnostic: map bytes, flip to executable
//! (W^X), cast the entry to a function pointer. RISC-V output is
//! position-independent (every relocation is PC-relative), so the load address
//! never changes the bytes.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const object = @import("object.zig");
const link = @import("link.zig");
const ld = @import("ld.zig");
const platform = @import("../jit_platform.zig");

pub const Error = std.mem.Allocator.Error || object.Error || ld.Error || platform.Error;
pub const Provider = platform.Provider;

/// A W^X executable buffer that synchronizes the instruction cache (the memory comes
/// from a pluggable `jit_platform.Provider`, posix by default, UEFI via `mapWith`).
pub const CodeBuffer = platform.Buffer(syncICache);

/// Synchronize the instruction stream with freshly written code. RISC-V uses
/// `fence.i` (mandatory before executing JIT output on real hardware). aarch64
/// cleans the D-cache and invalidates the I-cache to the point of unification.
/// x86 has a coherent I-cache, so it is a no-op.
fn syncICache(memory: []const u8) void {
    switch (builtin.cpu.arch) {
        .riscv32, .riscv64 => asm volatile ("fence.i" ::: .{ .memory = true }),
        .aarch64 => {
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
        },
        else => {},
    }
}

/// A JIT-compiled module: its live executable code plus the symbol table giving
/// each function's byte offset (the image is linked at base 0, so an address is
/// just an offset into `buffer`).
pub const Compiled = struct {
    buffer: CodeBuffer,
    image: ld.Image,

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        self.image.deinit(allocator);
        self.buffer.deinit();
    }

    /// A typed, callable pointer to the function named `name`, or null if it is
    /// not defined in the module. `Fn` must be a function-pointer type with the
    /// C calling convention, e.g. `*const fn (i64, i64) callconv(.c) i64`.
    pub fn funcPointer(self: *const Compiled, comptime Fn: type, name: []const u8) ?Fn {
        const offset = self.image.addressOf(name) orelse return null; // base 0 => address is offset
        return self.buffer.entry(Fn, @intCast(offset));
    }
};

/// JIT-compile a module: emit an object, link it into one image at base 0
/// (relocations are PC-relative, so the bytes are position-independent), and map
/// that image into executable memory. The caller owns the result.
pub fn compileModule(allocator: std.mem.Allocator, module: *const link.Module) Error!Compiled {
    return compileModuleResolved(allocator, module, null);
}

/// Like `compileModule`, but undefined symbols called by the module are bound to
/// absolute host addresses via `resolver` (e.g. runtime helpers, or functions in
/// a previously-jitted module). Each external call goes through a GOT stub the
/// linker appends, so it can reach any 64-bit address. The caller owns the result.
pub fn compileModuleResolved(allocator: std.mem.Allocator, module: *const link.Module, resolver: ?ld.Resolver) Error!Compiled {
    const obj = try object.writeModule(allocator, module);
    defer allocator.free(obj);
    var image = try ld.linkObjectsResolved(allocator, &.{obj}, 0, resolver);
    errdefer image.deinit(allocator);
    const buffer = try CodeBuffer.map(image.code);
    return .{ .buffer = buffer, .image = image };
}

test "JIT maps machine code and calls it in process" {
    // The host runs jitted code natively, so the JIT machinery is
    // execution-tested with hand-assembled host instructions. (RISC-V codegen is
    // execution-validated on River, see ld.zig.)
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    // aarch64: `add w0, w0, #42` then `ret`, returns its argument plus 42.
    const code = [_]u8{ 0x00, 0xa8, 0x00, 0x11, 0xc0, 0x03, 0x5f, 0xd6 };
    var buf = try CodeBuffer.map(&code);
    defer buf.deinit();

    const add42 = buf.entry(*const fn (u32) callconv(.c) u32, 0);
    try std.testing.expectEqual(@as(u32, 142), add42(100));
}

test "JIT compiles a RISC-V module into executable memory" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // A self-contained module: caller() -> callee(x).
    var callee = ir.function.Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = x });
    }
    var caller = ir.function.Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = r });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "callee", &callee);

    var compiled = try compileModule(allocator, &module);
    defer compiled.deinit(allocator);

    // The executable memory holds exactly the relocated linked image (the same
    // bytes River validates by execution).
    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var reference = try ld.linkObjects(allocator, &.{obj}, 0);
    defer reference.deinit(allocator);
    try std.testing.expectEqualSlices(u8, reference.code, compiled.buffer.memory[0..reference.code.len]);

    // Both functions resolve to in-buffer pointers. An unknown name does not.
    const FnT = *const fn (i32) callconv(.c) i32;
    try std.testing.expect(compiled.funcPointer(FnT, "caller") != null);
    try std.testing.expect(compiled.funcPointer(FnT, "callee") != null);
    try std.testing.expect(compiled.funcPointer(FnT, "missing") == null);
}

const encode = @import("encode.zig");
const harness = @import("tests/harness.zig");

test "the GOT stub's absolute load-and-jump executes on River" {
    // Execution-validate the exact sequence the JIT emits for an external call
    // (`auipc t0,_`, then `ld t0,_(t0)`, then `jr t0`, loading an absolute address from a GOT
    // slot) on River. The structural test above proves the JIT builds this
    // sequence. This proves it works. The image loads right after the harness's
    // argument stub, so the GOT holds the real runtime address.
    const allocator = std.testing.allocator;

    // Measure the harness stub to find where the image loads (and store the
    // correct absolute target address into the GOT).
    const stub = try harness.buildStub(allocator, &.{37});
    defer allocator.free(stub);
    const load_at = harness.load_address + @as(u64, stub.len) * 4;

    // entry: load target from the GOT at offset 16 and tail-jump to it. target
    // (at offset 24) adds 5 to a0 and returns straight to the harness.
    const target_addr = load_at + 24;
    const words = [_]u32{
        encode.auipc(.x5, 0), // 0:  t0 = pc (+hi(16)=0)
        encode.ld(.x5, .x5, 16), // 4:  t0 = *(t0 + 16) = GOT slot
        encode.jalr(.x0, .x5, 0), // 8:  jr t0 (tail jump, ra preserved)
        encode.addi(.x0, .x0, 0), // 12: nop pad (8-align the GOT)
        @truncate(target_addr), // 16: GOT slot, low word
        @truncate(target_addr >> 32), // 20: GOT slot, high word
        encode.addi(.x10, .x10, 5), // 24: target: a0 += 5
        encode.jalr(.x0, .x1, 0), // 28: ret -> harness
    };

    try std.testing.expectEqual(@as(i64, 42), try harness.runCode(std.testing.io, allocator, &words, &.{37}, harness.river));
}

/// Decode the (sign-extended) byte offset of a `jal` instruction word.
fn jalOffset(w: u32) i64 {
    const imm: u32 = (((w >> 31) & 1) << 20) | (((w >> 12) & 0xff) << 12) |
        (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3ff) << 1);
    return @as(i64, @as(i21, @bitCast(@as(u21, @truncate(imm)))));
}

test "JIT binds an external call to a host address through a GOT stub" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry(x) -> helper(x), where helper is external (resolved at JIT time).
    var entry = ir.function.Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const b = try entry.appendBlock();
        const x = try entry.appendBlockParam(b, t);
        const r = try entry.appendCall(b, t, "helper", &.{x});
        entry.setTerminator(b, .{ .ret = r });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);

    // A resolver binding "helper" to an arbitrary 64-bit host address.
    const Host = struct {
        const helper_addr: u64 = 0x0000_3fab_cd00_1230;
        fn resolve(_: *anyopaque, name: []const u8) ?u64 {
            return if (std.mem.eql(u8, name, "helper")) helper_addr else null;
        }
    };
    var ctx: u8 = 0;
    const resolver = ld.Resolver{ .context = &ctx, .func = Host.resolve };

    var compiled = try compileModuleResolved(allocator, &module, resolver);
    defer compiled.deinit(allocator);
    const mem = compiled.buffer.memory;

    // "helper" resolves to a stub inside the buffer: `auipc t0,_; ld t0,_(t0); jr t0`.
    const stub_off: usize = @intCast(compiled.image.addressOf("helper").?);
    const w0 = std.mem.readInt(u32, mem[stub_off..][0..4], .little);
    const w1 = std.mem.readInt(u32, mem[stub_off + 4 ..][0..4], .little);
    const w2 = std.mem.readInt(u32, mem[stub_off + 8 ..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 0x17), w0 & 0x7f); // auipc
    try std.testing.expectEqual(@as(u32, 0x03), w1 & 0x7f); // load
    try std.testing.expectEqual(@as(u32, 0x3), (w1 >> 12) & 0x7); // funct3 = ld
    try std.testing.expectEqual(encode.jalr(.x0, .x5, 0), w2); // jr t0

    // The stub's auipc+ld read a GOT slot holding the resolved host address.
    var found_got = false;
    var off: usize = 0;
    while (off + 8 <= mem.len) : (off += 8) {
        if (std.mem.readInt(u64, mem[off..][0..8], .little) == Host.helper_addr) {
            found_got = true;
            break;
        }
    }
    try std.testing.expect(found_got);

    // entry's `jal ra` to "helper" was routed to the stub.
    const entry_off: usize = @intCast(compiled.image.addressOf("entry").?);
    var routed = false;
    var i: usize = entry_off;
    while (i + 4 <= stub_off) : (i += 4) {
        const w = std.mem.readInt(u32, mem[i..][0..4], .little);
        if ((w & 0x7f) == 0x6f and ((w >> 7) & 0x1f) == 1) { // jal with rd = ra
            const target = @as(i64, @intCast(i)) + jalOffset(w);
            if (target == @as(i64, @intCast(stub_off))) routed = true;
        }
    }
    try std.testing.expect(routed);
}
