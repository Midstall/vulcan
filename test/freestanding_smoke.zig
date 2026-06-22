//! A freestanding build smoke test: it exercises Vulcan's compiler core (IR
//! construction, the optimizer, and the SPIR-V frontend) using only a caller-
//! supplied allocator over a static buffer - no libc, no OS, no syscalls. It is
//! compiled for a freestanding target by the build's `freestanding` step. If it
//! builds, those libraries are proven usable inside a baremetal/UEFI program.

const std = @import("std");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const spirv = @import("vulcan-spirv");
const pe = @import("vulcan-pe");
const image = @import("vulcan-image");

/// Build a small IR function, optimize it, and lower a tiny SPIR-V module - the
/// codegen-front half of the pipeline, all allocator-only.
fn run(allocator: std.mem.Allocator) !void {
    // IR construction + optimization.
    var func = ir.function.Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const c3 = try func.appendInst(b, t, .{ .iconst = 3 });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c2, .rhs = c3 } });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = sum, .rhs = x } });
    func.setTerminator(b, .{ .ret = r });
    _ = try opt.optimize(allocator, &func);
    _ = try opt.lowerdiv.run(allocator, &func);

    // SPIR-V frontend: lower a hand-assembled module to IR.
    var bld = try spirv.binary.Builder.init(allocator, 9);
    defer bld.deinit(allocator);
    const o = spirv.opcodes;
    try bld.emit(allocator, o.TypeInt, &.{ 1, 32, 1 });
    try bld.emit(allocator, o.TypeFunction, &.{ 2, 1, 1, 1 });
    try bld.emit(allocator, o.Function, &.{ 1, 3, 0, 2 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 4 });
    try bld.emit(allocator, o.FunctionParameter, &.{ 1, 5 });
    try bld.emit(allocator, o.Label, &.{6});
    try bld.emit(allocator, o.IMul, &.{ 1, 7, 4, 5 });
    try bld.emit(allocator, o.ReturnValue, &.{7});
    try bld.emit(allocator, o.FunctionEnd, &.{});
    var lowered = try spirv.lowerModule(allocator, bld.words.items);
    lowered.deinit();

    // Executable-container emission (UEFI PE32+ and a baremetal flat binary) is
    // pure byte generation and must also work freestanding.
    const code = [_]u8{ 0x00, 0x00, 0x80, 0xd2, 0xc0, 0x03, 0x5f, 0xd6 };
    const uefi = try pe.writeUefiImage(allocator, &code, code.len, 0, .aarch64);
    allocator.free(uefi);

    const flat = try image.flatBinary(allocator, &code, code.len, true);
    allocator.free(flat);
}

/// The freestanding entry point: drive the codegen over a static buffer. Marked
/// `export` so the object's root is analyzed (this forces the imports above to
/// compile for the freestanding target).
export fn vulcan_freestanding_smoke() void {
    var buf: [256 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    run(fba.allocator()) catch {};
}
