//! vulcan-link: a shared, `std`-only, architecture-generic static ELF linker. It parses
//! relocatable objects, resolves cross-object and intra-object relocations, binds
//! external calls through per-architecture GOT stubs when a resolver is given, and wraps
//! the result in a static executable. The generic engine lives in `elf.zig` (parsing)
//! and `resolve.zig` (the link driver + ELF-exec writer); each architecture's reloc math,
//! stub mechanism, optional compression, and executable parameters live in `arch/<a>.zig`.
//! Adding an architecture means adding an `arch/<a>.zig` and wiring it in `resolve.zig`.
//!
//! The linker imports only `std`: it defines its own ELF constants, reloc types, and
//! layout structs. This keeps `vulcan-target` free to depend on it without a cycle.

const std = @import("std");

pub const elf = @import("vulcan-link/elf.zig");
pub const dynamic = @import("vulcan-link/dynamic.zig");
pub const archive = @import("vulcan-link/archive.zig");
pub const resolve = @import("vulcan-link/resolve.zig");
pub const script = @import("vulcan-link/script.zig");
pub const layout = @import("vulcan-link/layout.zig");
pub const riscv64 = @import("vulcan-link/arch/riscv64.zig");
pub const aarch64 = @import("vulcan-link/arch/aarch64.zig");
pub const x86_64 = @import("vulcan-link/arch/x86_64.zig");
pub const x86 = @import("vulcan-link/arch/x86.zig");

// Generic types.
pub const Arch = elf.Arch;
pub const Error = elf.Error;
pub const ResolvedSymbol = elf.ResolvedSymbol;
pub const Image = elf.Image;
pub const Resolver = elf.Resolver;
pub const RelocType = elf.RelocType;
pub const fromEMachine = elf.fromEMachine;

// The layout model: segments + per-(object, section) placement, the interface between
// address assignment and relocation.
pub const Segment = elf.Segment;
pub const SecPlace = elf.SecPlace;
pub const Placement = elf.Placement;

// Archive (`.a`) parsing.
pub const Member = archive.Member;
pub const isArchive = archive.isArchive;
pub const parseArchive = archive.parseArchive;

// The link driver + executable writers.
pub const Input = resolve.Input;
pub const linkObjects = resolve.linkObjects;
pub const linkObjectsResolved = resolve.linkObjectsResolved;
pub const linkObjectsCompressed = resolve.linkObjectsCompressed;
pub const linkInputs = resolve.linkInputs;
pub const linkInputsResolved = resolve.linkInputsResolved;
pub const writeElfExec = resolve.writeElfExec;
pub const writeElfSegments = resolve.writeElfSegments;
pub const writeExecutable = resolve.writeExecutable;

// Dynamic linking (SM10 P4a): emit an ET_DYN `.so` / dynamic ET_EXEC, and read an
// ET_DYN's exports back. See dynamic.zig / resolve.linkDynamic.
pub const linkDynamic = resolve.linkDynamic;
pub const DynMode = dynamic.DynMode;
pub const DynOptions = dynamic.DynOptions;
pub const DynInput = dynamic.DynInput;
pub const readSharedExports = dynamic.readSharedExports;
pub const SharedExports = dynamic.SharedExports;

// Script-driven layout: consume a parsed `Script` + objects, produce a `Placement`, then
// relocate + link (see layout.zig / resolve.linkInputsScript).
pub const computeScriptPlacement = layout.computeScriptPlacement;
pub const secKindForPattern = layout.secKindForPattern;
pub const linkInputsScript = resolve.linkInputsScript;
pub const ScriptLinked = resolve.ScriptLinked;
pub const ScriptError = resolve.ScriptError;

// Linker-script parsing (text -> AST; no layout/linking - see script.zig).
pub const ScriptDiagnostic = script.Diagnostic;
pub const ScriptParseError = script.ParseError;
pub const parseScript = script.parse;
pub const Script = script.Script;
pub const ScriptBinOp = script.BinOp;
pub const ScriptExpr = script.Expr;
pub const ScriptRegionFlags = script.RegionFlags;
pub const ScriptRegion = script.Region;
pub const ScriptAssign = script.Assign;
pub const ScriptInputSpec = script.InputSpec;
pub const ScriptLma = script.Lma;
pub const ScriptSectionCmd = script.SectionCmd;
pub const ScriptOutputSection = script.OutputSection;
pub const ScriptCommand = script.Command;

test {
    std.testing.refAllDecls(@This());
}
