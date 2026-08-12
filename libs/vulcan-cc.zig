//! vulcan-cc: a C89 frontend for Vulcan. Parses C source, lowers it to Vulcan IR, and
//! (through the target backends) executes or emits it. This root module re-exports the
//! pipeline stages and the top-level `compile` entry point.

const std = @import("std");

pub const lexer = @import("vulcan-cc/lexer.zig");
pub const preproc = @import("vulcan-cc/preproc.zig");
pub const parser = @import("vulcan-cc/parser.zig");
pub const lower = @import("vulcan-cc/lower.zig");
pub const layout = @import("vulcan-cc/layout.zig");
pub const ctype = @import("vulcan-cc/ctype.zig");
/// The per-arch ABI struct/union classifier (SM14 M4d-a T2). See `abi.zig` - pure, not yet
/// wired into any call/return lowering.
pub const abi = @import("vulcan-cc/abi.zig");
/// `<stdarg.h>`'s built-in bytes/resolver (SM12 T1). See `stdarg.zig`.
pub const stdarg = @import("vulcan-cc/stdarg.zig");
/// `<stddef.h>`'s built-in bytes/resolver (SM13 M3a T4). See `stddef.zig`.
pub const stddef = @import("vulcan-cc/stddef.zig");
/// `<limits.h>`'s built-in bytes/resolver (M6). See `limits.zig`.
pub const limits = @import("vulcan-cc/limits.zig");
/// The host-side filesystem `#include` resolver (SM13 M3a T4). See `fs_resolver.zig`.
pub const fs_resolver = @import("vulcan-cc/fs_resolver.zig");

/// Compile C source to a single-function IR module. See `lower.compile`.
pub const compile = lower.compile;
/// Like `compile`, but with an explicit preprocessor configuration (SM8) - `-D` defines,
/// an `IncludeResolver` for `#include`, etc. See `lower.compileWithOpts`.
pub const compileWithOpts = lower.compileWithOpts;
/// Like `compileWithOpts`, but with an EXPLICIT target layout so `__builtin_va_list` takes the
/// target arch's ABI shape (SM12 T5) rather than the build host's. See `lower.compileForTarget`.
pub const compileForTarget = lower.compileForTarget;
pub const Module = lower.Module;
/// A module-level data object (SM7 Task 2's globals walking skeleton) - see `lower.DataObject`.
pub const DataObject = lower.DataObject;
pub const DataKind = lower.DataKind;
pub const DataReloc = lower.DataReloc;
/// The error set `compile` can return (lexer, parser, and lowering failures).
pub const Error = lower.Error;

test {
    std.testing.refAllDecls(@This());
}
