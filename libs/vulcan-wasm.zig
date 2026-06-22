//! WebAssembly frontend: read a Wasm binary, lower each function to Vulcan IR, JIT it
//! for the host, run it.
//!
//! `lower.zig` parses and lowers to IR (depends only on `vulcan-ir`). `engine.zig` is
//! the runnable layer (JIT plus memory/globals/table/imports setup, via
//! `vulcan-target.native`).

const std = @import("std");

const lower = @import("vulcan-wasm/lower.zig");
const engine = @import("vulcan-wasm/engine.zig");

// Lowering: parse a Wasm binary to IR.
pub const Module = lower.Module;
pub const LoweredFunction = lower.LoweredFunction;

/// Parse and lower a Wasm binary to IR. Caller owns the result.
pub fn load(allocator: std.mem.Allocator, bytes: []const u8) lower.Error!Module {
    return lower.module(allocator, bytes);
}

// Engine: JIT the lowered module and call its exports.
pub const Instance = engine.Instance;
pub const Provider = engine.Provider;
pub const ExecMemory = engine.ExecMemory;
pub const default_provider = engine.default_provider;
pub const page_size = engine.page_size;

/// A small wasi_snapshot_preview1 host runtime for running WASI modules.
pub const wasi = @import("vulcan-wasm/wasi.zig");

/// Errors the frontend can raise (lowering plus JIT/instantiation).
pub const Error = engine.Error;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(@import("vulcan-wasm/reader.zig"));
    std.testing.refAllDecls(lower);
    std.testing.refAllDecls(engine);
}
