//! Frontend-to-GPU path: SPIR-V binary lowered to Vulcan IR, selected to a SASS
//! compute kernel. Validation is structural (the emitted instruction stream).
//! live GPU execution runs from prism's compute dispatch, not this repo.

const std = @import("std");
const spirv = @import("vulcan-spirv");
const opt = @import("vulcan-opt");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");

const op = spirv.opcodes;

/// Opcodes present in `kernel.code`, one per 4-dword instruction.
fn hasOpcode(code: []const u32, opcode: u32) bool {
    var i: usize = 0;
    while (i < code.len) : (i += 4) {
        if (code[i] & 0xfff == opcode) return true;
    }
    return false;
}

fn countOpcode(code: []const u32, opcode: u32) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < code.len) : (i += 4) {
        if (code[i] & 0xfff == opcode) n += 1;
    }
    return n;
}

test "SPIR-V compute function -> IR -> SASS kernel (x*y - x)" {
    const allocator = std.testing.allocator;

    // The function f(int x, int y) returning x*y - x exercises IMAD and ISUB.
    // ids: int=1, fnty=2, f=3, x=4, y=5, entry=6, prod=7, diff=8.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(allocator, op.ISub, &.{ 1, 8, 7, 4 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();

    // SPIR-V -> IR -> SASS compute kernel.
    var kernel = try isel.compileKernel(allocator, &func);
    defer kernel.deinit(allocator);

    // The kernel loads the output pointer + two inputs (4x LDC), multiplies,
    // subtracts, stores, and exits.
    try std.testing.expectEqual(@as(usize, 4), countOpcode(kernel.code, 0xb82)); // LDC x4
    try std.testing.expect(hasOpcode(kernel.code, 0x224)); // IMAD
    try std.testing.expect(hasOpcode(kernel.code, 0x210)); // IADD3 (the subtract)
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT

    // The subtract's srcB negate bit (bit 63 -> word offset +1, bit 31) is set on
    // the IADD3 produced from OpISub.
    var i: usize = 0;
    var saw_negated_iadd3 = false;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0x210 and (kernel.code[i + 1] >> 31) & 1 == 1) saw_negated_iadd3 = true;
    }
    try std.testing.expect(saw_negated_iadd3);
}

test "SPIR-V function composes with the optimizer before SASS codegen" {
    const allocator = std.testing.allocator;

    // The function f(int x) returning (x + 7) * x folds a constant into the stream.
    var b = try spirv.binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 1, 3, 7 });
    try b.emit(allocator, op.Function, &.{ 1, 4, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IAdd, &.{ 1, 7, 5, 3 });
    try b.emit(allocator, op.IMul, &.{ 1, 8, 7, 5 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    _ = try opt.optimize(allocator, &func);

    var kernel = try isel.compileKernel(allocator, &func);
    defer kernel.deinit(allocator);
    // Still a well-formed kernel: it loads the input, computes, stores, and exits.
    try std.testing.expect(hasOpcode(kernel.code, 0xb82)); // LDC
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT
}

test "SPIR-V conversions -> SASS I2F/F2I" {
    const allocator = std.testing.allocator;

    // The function f(int x) returning int(float(x) * float(x)) squares via the float path.
    // ids: int=1, float=2, fnty=3, f=4, x=5, entry=6, fx=7, sq=8, r=9.
    var b = try spirv.binary.Builder.init(allocator, 10);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeFunction, &.{ 3, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 4, 0, 3 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.ConvertSToF, &.{ 2, 7, 5 });
    try b.emit(allocator, op.FMul, &.{ 2, 8, 7, 7 });
    try b.emit(allocator, op.ConvertFToS, &.{ 1, 9, 8 });
    try b.emit(allocator, op.ReturnValue, &.{9});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    var kernel = try isel.compileKernel(allocator, &func);
    defer kernel.deinit(allocator);

    try std.testing.expect(hasOpcode(kernel.code, 0x306)); // I2F (base 0x106 | reg form)
    try std.testing.expect(hasOpcode(kernel.code, 0x220)); // FMUL
    try std.testing.expect(hasOpcode(kernel.code, 0x305)); // F2I
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG
}

test "SPIR-V compute shader -> SASS kernel (buffer load/store + thread id)" {
    const allocator = std.testing.allocator;

    // The shader main() assigns data[gl_GlobalInvocationID.x] = data[gid] * 2.
    var b = try spirv.binary.Builder.init(allocator, 23);
    defer b.deinit(allocator);
    try b.emit(allocator, op.ExecutionMode, &.{ 16, op.ExecutionModeKind.local_size, 64, 1, 1 }); // local_size_x = 64
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.global_invocation_id });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 1 });
    try b.emit(allocator, op.TypeInt, &.{ 3, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, op.Constant, &.{ 3, 12, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 2 });
    try b.emit(allocator, op.Variable, &.{ 5, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 15, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, op.Label, &.{17});
    try b.emit(allocator, op.AccessChain, &.{ 6, 18, 14, 12 });
    try b.emit(allocator, op.Load, &.{ 3, 19, 18 });
    try b.emit(allocator, op.AccessChain, &.{ 10, 20, 15, 12, 19 });
    try b.emit(allocator, op.Load, &.{ 2, 21, 20 });
    try b.emit(allocator, op.IMul, &.{ 2, 22, 21, 13 });
    try b.emit(allocator, op.Store, &.{ 20, 22 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try spirv.lowerModule(allocator, b.words.items);
    defer func.deinit();
    var kernel = try isel.compileKernel(allocator, &func);
    defer kernel.deinit(allocator);

    // The invocation id is blockIdx.x * local_size_x + threadIdx.x: two S2R reads
    // (tid + ctaid) and a MOV of the local size (64). The buffer base comes from the
    // constant bank (a 64-bit pair = two LDC), the element address from a 64-bit
    // IADD3 carry add, then LDG, the multiply, STG, and EXIT (no output pointer).
    try std.testing.expectEqual(@as(usize, 2), countOpcode(kernel.code, 0x919)); // S2R x2 (tid + ctaid)
    var saw_localsize_mov = false;
    var k: usize = 0;
    while (k < kernel.code.len) : (k += 4) {
        if (kernel.code[k] & 0xfff == 0x802 and kernel.code[k + 1] == 64) saw_localsize_mov = true; // MOV imm 64
    }
    try std.testing.expect(saw_localsize_mov); // local_size_x folded in
    try std.testing.expectEqual(@as(usize, 2), countOpcode(kernel.code, 0xb82)); // LDC x2 (buffer ptr)
    try std.testing.expect(hasOpcode(kernel.code, 0x981)); // LDG (load data[i])
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG (store data[i])
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT

    // The carry-add pair: an IADD3 writing a carry-out predicate (P6) at bits 81-83,
    // and an IADD3 reading a carry-in predicate at bits 87-89.
    var saw_cout = false;
    var saw_cin = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff != 0x210) continue;
        if ((kernel.code[i + 2] >> 17) & 0x7 == 6) saw_cout = true; // carry-out -> P6 at bit 81
        if ((kernel.code[i + 2] >> 23) & 0x7 == 6) saw_cin = true; // carry-in <- P6 at bit 87
    }
    try std.testing.expect(saw_cout);
    try std.testing.expect(saw_cin);

    // The scoreboard scheduler ran: the LDG (variable latency) carries a write
    // barrier (wr_bar at bits 110-112 is a real scoreboard, not 7 = none), and some
    // later instruction waits on a scoreboard (a non-zero wait mask at 116-121).
    var ldg_bar: u32 = 7;
    var saw_wait = false;
    i = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0x981) ldg_bar = (kernel.code[i + 3] >> 14) & 0x7; // LDG wr_bar
        if ((kernel.code[i + 3] >> 20) & 0x3f != 0) saw_wait = true; // some wait mask set
    }
    try std.testing.expect(ldg_bar < 6); // a real scoreboard was assigned to the load
    try std.testing.expect(saw_wait);
}

test "SASS: a lowered division compiles to a kernel (register reuse)" {
    const allocator = std.testing.allocator;
    const Function = ir.function.Function;

    // f(x, y) = x / y (unsigned). The GPU has no integer divide, so opt.lowerdiv
    // expands it to ~256 instructions with 32 short-lived compares. The linear-scan
    // allocator reuses registers so it fits the 6 predicates / 250 GPRs.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const q = try func.appendInst(b, t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = q });

    try std.testing.expect(try opt.lowerdiv.run(allocator, &func));
    var kernel = try isel.compileKernel(allocator, &func);
    defer kernel.deinit(allocator);

    // The expansion lowers to shifts, compares, and selects, ending in STG + EXIT.
    try std.testing.expect(hasOpcode(kernel.code, 0x219)); // SHF (shift)
    try std.testing.expect(hasOpcode(kernel.code, 0x20c)); // ISETP (compare)
    try std.testing.expect(hasOpcode(kernel.code, 0x207)); // SEL
    try std.testing.expect(hasOpcode(kernel.code, 0x986)); // STG (output)
    try std.testing.expect(hasOpcode(kernel.code, 0x94d)); // EXIT
    // Reuse kept the register count modest despite ~256 instructions.
    try std.testing.expect(kernel.reg_count <= 32);
}
