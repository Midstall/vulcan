//! SPIR-V disassembler: turn a SPIR-V binary word stream into canonical spirv-dis-style
//! text (`%result = OpName operands`). It walks the module with `binary.Reader` and renders
//! each instruction from a per-opcode grammar table describing its result form and operand
//! kinds (id / literal / string). Ids print as `%N`, literals as decimal, strings quoted.
//!
//! The table covers every opcode the frontend and emitter use; an unknown opcode renders as
//! `Op<number>` with its operands shown as ids. This is a reading aid (objdump for SPIR-V),
//! not a re-assembler, so it does not need the full SPIR-V grammar.

const std = @import("std");
const binary = @import("binary.zig");
const op = @import("opcodes.zig");

/// The kind of a single operand, for rendering. Beyond raw ids / literals / strings, a few
/// operand positions are well-known enumerations that spirv-dis prints by name.
const Kind = enum { id, lit, string, capability, storage_class, exec_model, exec_mode, decoration };

/// Whether an instruction produces a result id, and whether a result-type id precedes it.
const Result = enum { none, result, typed };

/// A per-opcode rendering descriptor: its mnemonic, its result form, the kinds of its leading
/// fixed operands, and the kind that every remaining (variadic) operand takes.
const Form = struct {
    name: []const u8,
    result: Result = .none,
    fixed: []const Kind = &.{},
    rest: Kind = .id,
};

fn form(opcode: u16) Form {
    return switch (opcode) {
        // Module setup / debug / annotation.
        op.Capability => .{ .name = "OpCapability", .fixed = &.{.capability} },
        op.MemoryModel => .{ .name = "OpMemoryModel", .fixed = &.{ .lit, .lit } },
        op.EntryPoint => .{ .name = "OpEntryPoint", .fixed = &.{ .exec_model, .id, .string }, .rest = .id },
        op.ExecutionMode => .{ .name = "OpExecutionMode", .fixed = &.{ .id, .exec_mode }, .rest = .lit },
        op.Source => .{ .name = "OpSource", .fixed = &.{ .lit, .lit }, .rest = .lit },
        op.Name => .{ .name = "OpName", .fixed = &.{ .id, .string } },
        op.MemberName => .{ .name = "OpMemberName", .fixed = &.{ .id, .lit, .string } },
        op.ExtInstImport => .{ .name = "OpExtInstImport", .result = .result, .fixed = &.{.string} },
        op.ExtInst => .{ .name = "OpExtInst", .result = .typed, .fixed = &.{ .id, .lit }, .rest = .id },
        op.Decorate => .{ .name = "OpDecorate", .fixed = &.{ .id, .decoration }, .rest = .lit },
        op.MemberDecorate => .{ .name = "OpMemberDecorate", .fixed = &.{ .id, .lit, .decoration }, .rest = .lit },

        // Types.
        op.TypeVoid => .{ .name = "OpTypeVoid", .result = .result },
        op.TypeBool => .{ .name = "OpTypeBool", .result = .result },
        op.TypeInt => .{ .name = "OpTypeInt", .result = .result, .fixed = &.{ .lit, .lit } },
        op.TypeFloat => .{ .name = "OpTypeFloat", .result = .result, .fixed = &.{.lit} },
        op.TypeVector => .{ .name = "OpTypeVector", .result = .result, .fixed = &.{ .id, .lit } },
        op.TypeMatrix => .{ .name = "OpTypeMatrix", .result = .result, .fixed = &.{ .id, .lit } },
        op.TypeImage => .{ .name = "OpTypeImage", .result = .result, .fixed = &.{ .id, .lit, .lit, .lit, .lit, .lit, .lit }, .rest = .lit },
        op.TypeSampledImage => .{ .name = "OpTypeSampledImage", .result = .result, .fixed = &.{.id} },
        op.TypeArray => .{ .name = "OpTypeArray", .result = .result, .fixed = &.{ .id, .id } },
        op.TypeRuntimeArray => .{ .name = "OpTypeRuntimeArray", .result = .result, .fixed = &.{.id} },
        op.TypeStruct => .{ .name = "OpTypeStruct", .result = .result, .rest = .id },
        op.TypePointer => .{ .name = "OpTypePointer", .result = .result, .fixed = &.{ .storage_class, .id } },
        op.TypeFunction => .{ .name = "OpTypeFunction", .result = .result, .fixed = &.{.id}, .rest = .id },

        // Constants.
        op.ConstantTrue => .{ .name = "OpConstantTrue", .result = .typed },
        op.ConstantFalse => .{ .name = "OpConstantFalse", .result = .typed },
        op.Constant => .{ .name = "OpConstant", .result = .typed, .rest = .lit },
        op.ConstantComposite => .{ .name = "OpConstantComposite", .result = .typed, .rest = .id },

        // Functions and control flow.
        op.Function => .{ .name = "OpFunction", .result = .typed, .fixed = &.{ .lit, .id } },
        op.FunctionParameter => .{ .name = "OpFunctionParameter", .result = .typed },
        op.FunctionEnd => .{ .name = "OpFunctionEnd" },
        op.FunctionCall => .{ .name = "OpFunctionCall", .result = .typed, .fixed = &.{.id}, .rest = .id },
        op.Label => .{ .name = "OpLabel", .result = .result },
        op.Branch => .{ .name = "OpBranch", .fixed = &.{.id} },
        op.BranchConditional => .{ .name = "OpBranchConditional", .fixed = &.{ .id, .id, .id }, .rest = .lit },
        op.Switch => .{ .name = "OpSwitch", .fixed = &.{ .id, .id }, .rest = .id },
        op.Phi => .{ .name = "OpPhi", .result = .typed, .rest = .id },
        op.Return => .{ .name = "OpReturn" },
        op.ReturnValue => .{ .name = "OpReturnValue", .fixed = &.{.id} },
        op.Unreachable => .{ .name = "OpUnreachable" },
        op.Kill => .{ .name = "OpKill" },
        op.SelectionMerge => .{ .name = "OpSelectionMerge", .fixed = &.{ .id, .lit } },
        op.LoopMerge => .{ .name = "OpLoopMerge", .fixed = &.{ .id, .id, .lit } },

        // Memory.
        op.Variable => .{ .name = "OpVariable", .result = .typed, .fixed = &.{.storage_class}, .rest = .id },
        op.Load => .{ .name = "OpLoad", .result = .typed, .fixed = &.{.id}, .rest = .lit },
        op.Store => .{ .name = "OpStore", .fixed = &.{ .id, .id }, .rest = .lit },
        op.AccessChain => .{ .name = "OpAccessChain", .result = .typed, .fixed = &.{.id}, .rest = .id },

        // Composites (literal component/index tails).
        op.VectorShuffle => .{ .name = "OpVectorShuffle", .result = .typed, .fixed = &.{ .id, .id }, .rest = .lit },
        op.CompositeConstruct => .{ .name = "OpCompositeConstruct", .result = .typed, .rest = .id },
        op.CompositeExtract => .{ .name = "OpCompositeExtract", .result = .typed, .fixed = &.{.id}, .rest = .lit },

        else => namedArithmetic(opcode),
    };
}

/// Opcodes whose whole operand list is ids after the (type, result) prefix: arithmetic,
/// comparisons, conversions, and the vector/matrix products. Unknown opcodes fall through to
/// a generic `Op<number>` so the stream still disassembles.
fn namedArithmetic(opcode: u16) Form {
    const name: []const u8 = switch (opcode) {
        op.Undef => "OpUndef",
        op.IAdd => "OpIAdd",
        op.FAdd => "OpFAdd",
        op.ISub => "OpISub",
        op.FSub => "OpFSub",
        op.IMul => "OpIMul",
        op.FMul => "OpFMul",
        op.UDiv => "OpUDiv",
        op.SDiv => "OpSDiv",
        op.FDiv => "OpFDiv",
        op.UMod => "OpUMod",
        op.SRem => "OpSRem",
        op.SMod => "OpSMod",
        op.FRem => "OpFRem",
        op.SNegate => "OpSNegate",
        op.FNegate => "OpFNegate",
        op.ShiftRightLogical => "OpShiftRightLogical",
        op.ShiftRightArithmetic => "OpShiftRightArithmetic",
        op.ShiftLeftLogical => "OpShiftLeftLogical",
        op.BitwiseOr => "OpBitwiseOr",
        op.BitwiseXor => "OpBitwiseXor",
        op.BitwiseAnd => "OpBitwiseAnd",
        op.Not => "OpNot",
        op.LogicalEqual => "OpLogicalEqual",
        op.LogicalNotEqual => "OpLogicalNotEqual",
        op.LogicalOr => "OpLogicalOr",
        op.LogicalAnd => "OpLogicalAnd",
        op.LogicalNot => "OpLogicalNot",
        op.IEqual => "OpIEqual",
        op.INotEqual => "OpINotEqual",
        op.UGreaterThan => "OpUGreaterThan",
        op.SGreaterThan => "OpSGreaterThan",
        op.UGreaterThanEqual => "OpUGreaterThanEqual",
        op.SGreaterThanEqual => "OpSGreaterThanEqual",
        op.ULessThan => "OpULessThan",
        op.SLessThan => "OpSLessThan",
        op.ULessThanEqual => "OpULessThanEqual",
        op.SLessThanEqual => "OpSLessThanEqual",
        op.FOrdEqual => "OpFOrdEqual",
        op.FOrdNotEqual => "OpFOrdNotEqual",
        op.FOrdLessThan => "OpFOrdLessThan",
        op.FOrdGreaterThan => "OpFOrdGreaterThan",
        op.FOrdLessThanEqual => "OpFOrdLessThanEqual",
        op.FOrdGreaterThanEqual => "OpFOrdGreaterThanEqual",
        op.ConvertFToU => "OpConvertFToU",
        op.ConvertFToS => "OpConvertFToS",
        op.ConvertSToF => "OpConvertSToF",
        op.ConvertUToF => "OpConvertUToF",
        op.UConvert => "OpUConvert",
        op.SConvert => "OpSConvert",
        op.FConvert => "OpFConvert",
        op.Bitcast => "OpBitcast",
        op.VectorTimesScalar => "OpVectorTimesScalar",
        op.MatrixTimesScalar => "OpMatrixTimesScalar",
        op.VectorTimesMatrix => "OpVectorTimesMatrix",
        op.MatrixTimesVector => "OpMatrixTimesVector",
        op.MatrixTimesMatrix => "OpMatrixTimesMatrix",
        op.Dot => "OpDot",
        op.Select => "OpSelect",
        op.SampledImage => "OpSampledImage",
        op.ImageSampleImplicitLod => "OpImageSampleImplicitLod",
        op.ImageSampleExplicitLod => "OpImageSampleExplicitLod",
        op.ImageGather => "OpImageGather",
        op.Image => "OpImage",
        op.ImageFetch => "OpImageFetch",
        op.DPdx => "OpDPdx",
        op.DPdy => "OpDPdy",
        op.Fwidth => "OpFwidth",
        op.Nop => "OpNop",
        else => return .{ .name = "Op?", .result = .typed, .rest = .id },
    };
    // Every opcode above yields a typed result with all-id operands.
    return .{ .name = name, .result = .typed, .rest = .id };
}

/// Names for the well-known context-free enumerations spirv-dis prints by name. An unknown
/// value falls back to its decimal so the listing never loses information.
fn enumName(kind: Kind, value: u32) ?[]const u8 {
    return switch (kind) {
        .capability => switch (value) {
            0 => "Matrix",
            1 => "Shader",
            else => null,
        },
        .storage_class => switch (value) {
            op.StorageClass.uniform_constant => "UniformConstant",
            op.StorageClass.input => "Input",
            op.StorageClass.uniform => "Uniform",
            op.StorageClass.output => "Output",
            op.StorageClass.function => "Function",
            op.StorageClass.push_constant => "PushConstant",
            op.StorageClass.storage_buffer => "StorageBuffer",
            else => null,
        },
        .exec_model => switch (value) {
            op.ExecutionModel.vertex => "Vertex",
            op.ExecutionModel.fragment => "Fragment",
            op.ExecutionModel.gl_compute => "GLCompute",
            else => null,
        },
        .exec_mode => switch (value) {
            op.ExecutionModeKind.origin_upper_left => "OriginUpperLeft",
            op.ExecutionModeKind.local_size => "LocalSize",
            op.ExecutionModeKind.local_size_id => "LocalSizeId",
            else => null,
        },
        .decoration => switch (value) {
            op.Decoration.block => "Block",
            op.Decoration.row_major => "RowMajor",
            op.Decoration.col_major => "ColMajor",
            op.Decoration.array_stride => "ArrayStride",
            op.Decoration.matrix_stride => "MatrixStride",
            op.Decoration.builtin => "BuiltIn",
            op.Decoration.location => "Location",
            op.Decoration.binding => "Binding",
            op.Decoration.descriptor_set => "DescriptorSet",
            op.Decoration.offset => "Offset",
            else => null,
        },
        else => null,
    };
}

/// Decode a NUL-terminated SPIR-V literal string from `words` starting at index `i`, returning
/// the byte slice (borrowing into a scratch buffer is avoided by writing straight to `out`) and
/// the number of words consumed.
fn appendString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), words: []const u32, i: usize) !usize {
    try out.append(allocator, '"');
    var w = i;
    outer: while (w < words.len) : (w += 1) {
        var word = words[w];
        var b: usize = 0;
        while (b < 4) : (b += 1) {
            const byte: u8 = @truncate(word);
            word >>= 8;
            if (byte == 0) {
                w += 1;
                break :outer;
            }
            try out.append(allocator, byte);
        }
    }
    try out.append(allocator, '"');
    return w - i;
}

/// Disassemble a SPIR-V word stream into text. Caller owns the result.
pub fn format(allocator: std.mem.Allocator, words: []const u32) (binary.Error || std.mem.Allocator.Error)![]u8 {
    var reader = try binary.Reader.init(words);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    const v = reader.header.version;
    try out.print(allocator, "; SPIR-V\n; Version: {d}.{d}\n; Bound: {d}\n", .{ (v >> 16) & 0xff, (v >> 8) & 0xff, reader.header.id_bound });

    // Width of the result column so the `=` signs line up (like spirv-dis). `%` + up to the
    // id-bound's decimal width.
    var idw: usize = 1;
    var bound = reader.header.id_bound;
    while (bound >= 10) : (bound /= 10) idw += 1;
    const col = idw + 1; // leading '%'

    while (try reader.next()) |inst| {
        const f = form(inst.opcode);
        const ops = inst.operands;
        // Operand cursor: skip the result-type and result-id words when present.
        var cur: usize = 0;
        const type_id: ?u32 = if (f.result == .typed and ops.len > 0) blk: {
            cur += 1;
            break :blk ops[0];
        } else null;
        const result_id: ?u32 = switch (f.result) {
            .none => null,
            .result => if (ops.len > cur) blk: {
                const r = ops[cur];
                cur += 1;
                break :blk r;
            } else null,
            .typed => if (ops.len > cur) blk: {
                const r = ops[cur];
                cur += 1;
                break :blk r;
            } else null,
        };

        // Result column: right-justified `%id = `, or blanks when there is no result.
        if (result_id) |r| {
            var buf: [16]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "%{d}", .{r}) catch unreachable;
            const pad = if (s.len < col) col - s.len else 0;
            try out.appendNTimes(allocator, ' ', pad);
            try out.appendSlice(allocator, s);
            try out.appendSlice(allocator, " = ");
        } else {
            try out.appendNTimes(allocator, ' ', col + 3);
        }

        try out.appendSlice(allocator, f.name);
        if (f.name.len == 3) try out.print(allocator, "{d}", .{inst.opcode}); // "Op?" fallback

        // A typed instruction names its result type first.
        if (type_id) |t| try out.print(allocator, " %{d}", .{t});

        // Remaining operands: the fixed kinds, then the variadic `rest` kind.
        var oi: usize = 0;
        while (cur < ops.len) {
            const kind = if (oi < f.fixed.len) f.fixed[oi] else f.rest;
            oi += 1;
            switch (kind) {
                .id => {
                    try out.print(allocator, " %{d}", .{ops[cur]});
                    cur += 1;
                },
                .lit => {
                    try out.print(allocator, " {d}", .{ops[cur]});
                    cur += 1;
                },
                .string => {
                    try out.append(allocator, ' ');
                    cur += try appendString(allocator, &out, ops, cur);
                },
                else => { // an enumeration: print its name, or its decimal if unknown
                    if (enumName(kind, ops[cur])) |nm| {
                        try out.print(allocator, " {s}", .{nm});
                    } else {
                        try out.print(allocator, " {d}", .{ops[cur]});
                    }
                    cur += 1;
                },
            }
        }
        try out.append(allocator, '\n');
    }
    return out.toOwnedSlice(allocator);
}

/// Disassemble from raw bytes (a `.spv` file). Length must be a multiple of 4.
pub fn formatBytes(allocator: std.mem.Allocator, bytes: []const u8) (binary.Error || std.mem.Allocator.Error)![]u8 {
    if (bytes.len % 4 != 0) return error.Truncated;
    const words = try allocator.alloc(u32, bytes.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*wp, i| wp.* = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
    return format(allocator, words);
}

test "disassembles a small module with types, a constant, and a store" {
    const a = std.testing.allocator;
    var b = try binary.Builder.init(a, 10);
    defer b.deinit(a);
    // %1 = OpTypeFloat 32 ; %2 = OpConstant %1 <bits> ; %3 = OpTypeVector %1 4
    try b.emit(a, op.TypeFloat, &.{ 1, 32 });
    try b.emit(a, op.Constant, &.{ 1, 2, 0x40490fdb }); // 3.14159...
    try b.emit(a, op.TypeVector, &.{ 3, 1, 4 });
    try b.emit(a, op.Store, &.{ 8, 2 });

    const text = try format(a, b.words.items);
    defer a.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "%1 = OpTypeFloat 32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpConstant %1 1078530011") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "%3 = OpTypeVector %1 4") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpStore %8 %2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "; Bound: 10") != null);
}

test "renders string operands quoted (OpName, OpEntryPoint)" {
    const a = std.testing.allocator;
    var b = try binary.Builder.init(a, 20);
    defer b.deinit(a);
    // OpName %5 "main" : the string "main" packs into two words {'m','a','i','n'} then NUL word.
    try b.emit(a, op.Name, &.{ 5, packWord("main"), 0 });

    const text = try format(a, b.words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpName %5 \"main\"") != null);
}

/// Pack up to 4 ASCII bytes little-endian into a word (test helper).
fn packWord(s: []const u8) u32 {
    var w: u32 = 0;
    for (s, 0..) |c, i| w |= @as(u32, c) << @intCast(i * 8);
    return w;
}

test "renders well-known enums by name (Capability, StorageClass, ExecutionModel)" {
    const a = std.testing.allocator;
    var b = try binary.Builder.init(a, 20);
    defer b.deinit(a);
    try b.emit(a, op.Capability, &.{1}); // Shader
    try b.emit(a, op.EntryPoint, &.{ 0, 3, packWord("main"), 0, 7 }); // Vertex
    try b.emit(a, op.TypePointer, &.{ 6, op.StorageClass.output, 5 }); // Output
    try b.emit(a, op.Decorate, &.{ 7, op.Decoration.builtin, 0 }); // BuiltIn

    const text = try format(a, b.words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpCapability Shader") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpEntryPoint Vertex %3 \"main\" %7") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpTypePointer Output %5") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "OpDecorate %7 BuiltIn 0") != null);
}

test "unknown opcode still disassembles as Op<number>" {
    const a = std.testing.allocator;
    var b = try binary.Builder.init(a, 5);
    defer b.deinit(a);
    try b.emit(a, 9999, &.{ 1, 2 }); // not in the table
    const text = try format(a, b.words.items);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "Op?9999") != null);
}

test "formatBytes reads a little-endian byte stream" {
    const a = std.testing.allocator;
    var b = try binary.Builder.init(a, 4);
    defer b.deinit(a);
    try b.emit(a, op.TypeVoid, &.{1});
    const bytes = std.mem.sliceAsBytes(b.words.items);
    const text = try formatBytes(a, bytes);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "%1 = OpTypeVoid") != null);
}
