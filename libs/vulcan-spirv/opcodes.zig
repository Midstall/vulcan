//! Subset of SPIR-V opcodes the frontend understands. Numeric values are from the
//! SPIR-V specification. Grouped by role.

pub const Nop = 0;
pub const Undef = 1;

// Debug / annotation (skipped by the frontend).
pub const Name = 5;
pub const MemberName = 6;
pub const Decorate = 71;
pub const MemberDecorate = 72;

// Module setup.
pub const EntryPoint = 15;
pub const ExecutionMode = 16;
pub const Capability = 17;
pub const MemoryModel = 14;
pub const Source = 3;
pub const ExtInstImport = 11;
pub const ExtInst = 12;

/// GLSL.std.450 extended-instruction numbers the frontend lowers (to select +
/// compare). The set itself (OpExtInstImport "GLSL.std.450") is assumed present.
pub const Glsl = struct {
    pub const round_even = 2;
    pub const trunc = 3;
    pub const f_abs = 4;
    pub const s_abs = 5;
    pub const floor = 8;
    pub const ceil = 9;
    pub const sin = 13;
    pub const cos = 14;
    pub const tan = 15;
    pub const asin = 16;
    pub const acos = 17;
    pub const atan = 18;
    pub const pow = 26;
    pub const exp = 27;
    pub const log = 28;
    pub const exp2 = 29;
    pub const log2 = 30;
    pub const sqrt = 31;
    pub const inverse_sqrt = 32;
    pub const f_min = 37;
    pub const u_min = 38;
    pub const s_min = 39;
    pub const f_max = 40;
    pub const u_max = 41;
    pub const s_max = 42;
    pub const f_clamp = 43;
    pub const u_clamp = 44;
    pub const s_clamp = 45;
};

// Types.
pub const TypeVoid = 19;
pub const TypeBool = 20;
pub const TypeInt = 21;
pub const TypeFloat = 22;
pub const TypeVector = 23;
pub const TypeImage = 25;
pub const TypeSampledImage = 27;
pub const TypeArray = 28;
pub const TypeRuntimeArray = 29;
pub const TypeStruct = 30;
pub const TypePointer = 32;
pub const TypeFunction = 33;

/// Sample a sampled-image with implicit level-of-detail (fragment shaders only).
pub const ImageSampleImplicitLod = 87;

/// `OpTypeImage` Dim operand (subset the frontend emits).
pub const Dim = struct {
    pub const dim_2d = 1;
};

/// SPIR-V storage classes (subset the frontend recognizes).
pub const StorageClass = struct {
    pub const uniform_constant = 0; // opaque resources: samplers / images
    pub const input = 1;
    pub const uniform = 2;
    pub const output = 3;
    pub const function = 7;
    pub const push_constant = 9;
    pub const storage_buffer = 12;
};

/// SPIR-V decorations the frontend reads.
pub const Decoration = struct {
    pub const block = 2; // a struct used as a UBO/push-constant interface block
    pub const array_stride = 6;
    pub const builtin = 11;
    pub const location = 30;
    pub const binding = 33;
    pub const descriptor_set = 34;
    pub const offset = 35;
};

/// SPIR-V builtins the frontend recognizes (value of a BuiltIn decoration).
pub const BuiltIn = struct {
    pub const position = 0; // gl_Position (a vertex-shader clip-space output)
    pub const frag_coord = 15; // gl_FragCoord (a fragment-shader window-space input)
    pub const workgroup_id = 26;
    pub const local_invocation_id = 27;
    pub const global_invocation_id = 28;
    pub const vertex_index = 42; // gl_VertexIndex (a vertex-shader input)
};

/// SPIR-V execution models (first operand of OpEntryPoint): which pipeline stage the
/// entry point drives.
pub const ExecutionModel = struct {
    pub const vertex = 0;
    pub const fragment = 4;
    pub const gl_compute = 5;
};

/// SPIR-V execution modes (value in an OpExecutionMode).
pub const ExecutionModeKind = struct {
    pub const origin_upper_left = 7; // mandatory for a Fragment entry point
    pub const local_size = 17; // followed by literal x, y, z workgroup dimensions
    pub const local_size_id = 38; // the same, given as constant ids
};

// Constants.
pub const ConstantTrue = 41;
pub const ConstantFalse = 42;
pub const Constant = 43;
pub const ConstantComposite = 44;

// Functions and control flow.
pub const Function = 54;
pub const FunctionParameter = 55;
pub const FunctionEnd = 56;
pub const FunctionCall = 57;
pub const Label = 248;
pub const Branch = 249;
pub const BranchConditional = 250;
pub const Phi = 245;
pub const Return = 253;
pub const ReturnValue = 254;
pub const Unreachable = 255;
pub const SelectionMerge = 247;
pub const LoopMerge = 246;

// Integer arithmetic.
pub const IAdd = 128;
pub const FAdd = 129;
pub const ISub = 130;
pub const FSub = 131;
pub const IMul = 132;
pub const FMul = 133;
pub const UDiv = 134;
pub const SDiv = 135;
pub const FDiv = 136;
pub const UMod = 137;
pub const SRem = 138;
pub const SMod = 139;
pub const FRem = 140;
pub const SNegate = 126;
pub const FNegate = 127;

// Fragment derivatives (valid only in a Fragment entry point).
pub const DPdx = 207;
pub const DPdy = 208;
pub const Fwidth = 209;
pub const Kill = 252; // discard: terminate the fragment invocation

// Bitwise / shift.
pub const ShiftRightLogical = 194;
pub const ShiftRightArithmetic = 195;
pub const ShiftLeftLogical = 196;
pub const BitwiseOr = 197;
pub const BitwiseXor = 198;
pub const BitwiseAnd = 199;
pub const Not = 200;

// Logical (operate on bool, unlike the bitwise ops).
pub const LogicalEqual = 164;
pub const LogicalNotEqual = 165;
pub const LogicalOr = 166;
pub const LogicalAnd = 167;
pub const LogicalNot = 168;

// Comparison.
pub const IEqual = 170;
pub const INotEqual = 171;
pub const UGreaterThan = 172;
pub const SGreaterThan = 173;
pub const UGreaterThanEqual = 174;
pub const SGreaterThanEqual = 175;
pub const ULessThan = 176;
pub const SLessThan = 177;
pub const ULessThanEqual = 178;
pub const SLessThanEqual = 179;
pub const FOrdEqual = 180;
pub const FOrdNotEqual = 182;
pub const FOrdLessThan = 184;
pub const FOrdGreaterThan = 186;
pub const FOrdLessThanEqual = 188;
pub const FOrdGreaterThanEqual = 190;

// Numeric conversions.
pub const ConvertFToU = 109;
pub const ConvertFToS = 110;
pub const ConvertSToF = 111;
pub const ConvertUToF = 112;
pub const UConvert = 113;
pub const SConvert = 114;
pub const FConvert = 115;
pub const Bitcast = 124;

// Composites / vectors.
pub const VectorShuffle = 79;
pub const CompositeConstruct = 80;
pub const CompositeExtract = 81;
pub const VectorTimesScalar = 142;
pub const Dot = 148;

// Misc.
pub const Select = 169;
pub const Variable = 59;
pub const Load = 61;
pub const Store = 62;
pub const AccessChain = 65;
