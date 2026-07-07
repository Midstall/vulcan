//! Wasm (MVP) target: encoding, instruction selection, and codegen for
//! WebAssembly. Lowers Vulcan IR to Wasm bytecode and produces Wasm binaries.
//!
//! Unlike native targets that emit machine code, the Wasm target emits Wasm
//! binary modules. The instruction selection assigns IR values to Wasm locals
//! and emits Wasm opcodes directly.

const std = @import("std");

pub const encode = @import("wasm/encode.zig");
pub const disasm = @import("wasm/disasm.zig");
pub const isel = @import("wasm/isel.zig");
pub const link = @import("wasm/link.zig");
pub const object = @import("wasm/object.zig");

test {
    std.testing.refAllDecls(@This());
}
