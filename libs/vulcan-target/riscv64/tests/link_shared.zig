//! Coverage for the shared linker (`vulcan-link`) as used by the riscv64 backend.
//! These tests were re-homed from the old `riscv64/ld.zig` in-file tests when its
//! logic moved into `libs/vulcan-link`: they build real `.o` inputs via
//! `object.writeModule` (available only here in the consumer, not in the std-only
//! link library) and drive `link.linkObjects` / `link.writeExecutable`, so the
//! backend's object-linking and River execution coverage is preserved.

const std = @import("std");
const ir = @import("vulcan-ir");
const object = @import("../object.zig");
const link = @import("../link.zig");
const harness = @import("harness.zig");
const ld = @import("vulcan-link");

const Function = ir.function.Function;

test "our linker resolves a cross-object call to the reference bytes" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee, compiled to its own object.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    var callee_mod: link.Module = .{};
    defer callee_mod.deinit(allocator);
    try callee_mod.addFunction(allocator, "callee", &callee);
    const callee_obj = try object.writeModule(allocator, &callee_mod);
    defer allocator.free(callee_obj);

    // caller, compiled to its own object: "callee" is an undefined external.
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var caller_mod: link.Module = .{};
    defer caller_mod.deinit(allocator);
    try caller_mod.addFunction(allocator, "caller", &caller);
    const caller_obj = try object.writeModule(allocator, &caller_mod);
    defer allocator.free(caller_obj);

    // The in-memory IR linker's bytes are the reference (lld matches these too).
    var combined: link.Module = .{};
    defer combined.deinit(allocator);
    try combined.addFunction(allocator, "callee", &callee);
    try combined.addFunction(allocator, "caller", &caller);
    var reference = try link.compileModule(allocator, &combined);
    defer reference.deinit(allocator);

    // The object linker, given the two objects (callee first), must produce the
    // same relocated code.
    var image = try ld.linkObjects(allocator, &.{ callee_obj, caller_obj }, 0x80000000);
    defer image.deinit(allocator);

    try std.testing.expectEqual(reference.code.len * 4, image.code.len);
    for (reference.code, 0..) |word, i| {
        const got = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(word, got);
    }
    try std.testing.expectEqual(@as(?u64, 0x80000000), image.addressOf("callee"));

    // The linker can wrap the image into a runnable ET_EXEC entering at "caller".
    const exe = try ld.writeExecutable(.riscv64, allocator, &image, "caller");
    defer allocator.free(exe);
    try std.testing.expectEqualSlices(u8, "\x7fELF", exe[0..4]);
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, exe[16..18], .little)); // ET_EXEC
    try std.testing.expectEqual(@as(u16, 243), std.mem.readInt(u16, exe[18..20], .little)); // EM_RISCV
    // e_entry is caller's absolute address (callee is one word, so caller@base+4).
    try std.testing.expectEqual(@as(u64, 0x80000004), std.mem.readInt(u64, exe[24..32], .little));
    // The single PT_LOAD maps the code at the image base.
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, exe[56..58], .little)); // e_phnum
    try std.testing.expectEqual(@as(u64, 0x80000000), std.mem.readInt(u64, exe[64 + 16 ..][0..8], .little)); // p_vaddr
    // The loadable bytes are exactly the linked image (at the page-aligned code offset).
    try std.testing.expectEqualSlices(u8, image.code, exe[0x1000..]);

    // Unresolved entry name is rejected.
    try std.testing.expectError(error.UndefinedSymbol, ld.writeExecutable(.riscv64, allocator, &image, "nope"));
}

test "code linked by our own linker runs correctly on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry(a, b) -> add(a, b). The stub calls whatever sits at code[0], so the
    // entry must be linked first. add() is a separate object.
    var add = Function.init(allocator);
    defer add.deinit();
    {
        const t = try add.types.intern(i32k);
        const b = try add.appendBlock();
        const x = try add.appendBlockParam(b, t);
        const y = try add.appendBlockParam(b, t);
        const s = try add.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
        add.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });
    }
    var add_mod: link.Module = .{};
    defer add_mod.deinit(allocator);
    try add_mod.addFunction(allocator, "add", &add);
    const add_obj = try object.writeModule(allocator, &add_mod);
    defer allocator.free(add_obj);

    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const b = try entry.appendBlock();
        const x = try entry.appendBlockParam(b, t);
        const y = try entry.appendBlockParam(b, t);
        const r = try entry.appendCall(b, t, "add", &.{ x, y });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var entry_mod: link.Module = .{};
    defer entry_mod.deinit(allocator);
    try entry_mod.addFunction(allocator, "entry", &entry);
    const entry_obj = try object.writeModule(allocator, &entry_mod);
    defer allocator.free(entry_obj);

    // Link entry first so it lands at the image start. add() follows.
    var image = try ld.linkObjects(allocator, &.{ entry_obj, add_obj }, harness.load_address);
    defer image.deinit(allocator);

    // Reinterpret the byte image as machine words and execute it on River.
    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);

    try std.testing.expectEqual(@as(i64, 42), try harness.runCode(io, allocator, words, &.{ 20, 22 }, harness.river));
}

test "global data loaded via PC-relative addressing runs on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), where K is a module-level i32 constant holding 42.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.ptrGlobal();
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "K");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }

    const k_bytes = [_]u8{ 42, 0, 0, 0 }; // i32 42, little-endian
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "K", &k_bytes);

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);

    // The linker resolves the PCREL_HI20/LO12 pair. The result runs on River.
    var image = try ld.linkObjects(allocator, &.{obj}, harness.load_address);
    defer image.deinit(allocator);

    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 42), try harness.runCode(io, allocator, words, &.{}, harness.river));
}

test "initialized writable .data global is read back on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&D), D a writable i32 initialized to 7 (lives in `.data`).
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.ptrGlobal();
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "D");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    const d_bytes = [_]u8{ 7, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addWritable(allocator, "D", &d_bytes);

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var image = try ld.linkObjects(allocator, &.{obj}, harness.load_address);
    defer image.deinit(allocator);

    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 7), try harness.runCode(io, allocator, words, &.{}, harness.river));
}

test "zero-initialized .bss global is writable and reads back on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() stores 99 to B then returns it. B is a zero-init i32 in `.bss`.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.ptrGlobal();
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "B");
        const c = try entry.appendInst(b, t, .{ .iconst = 99 });
        try entry.appendStore(b, c, p);
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addBss(allocator, "B", 4);

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var image = try ld.linkObjects(allocator, &.{obj}, harness.load_address);
    defer image.deinit(allocator);

    // The image carries no .bss bytes, but memsz covers it.
    try std.testing.expect(image.memsz > image.code.len);

    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 99), try harness.runCode(io, allocator, words, &.{}, harness.river));
}
