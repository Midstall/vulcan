//! Multi-backend target seam: register sets, calling conventions, ABI, layout,
//! encoding, and codegen per target. riscv64 is first-class. aarch64 in bring-up.
//! Freestanding-clean.

const std = @import("std");

pub const riscv64 = @import("vulcan-target/riscv64.zig");
pub const aarch64 = @import("vulcan-target/aarch64.zig");
pub const x86 = @import("vulcan-target/x86.zig");
pub const x86_64 = @import("vulcan-target/x86_64.zig");
pub const nvidia = @import("vulcan-target/nvidia.zig");
pub const wasm = @import("vulcan-target/wasm.zig");

/// Portable C99 source backend: lowers an IR function to equivalent C.
pub const c = @import("vulcan-target/c.zig");

/// JavaScript source backend: lowers an IR function to equivalent JS.
pub const js = @import("vulcan-target/js.zig");

/// virgl/Gallium TGSI text: lowers a graphics IR function to the textual shader
/// form `tgsi_text_translate` accepts (the GPU-paravirtual path).
pub const tgsi = @import("vulcan-target/tgsi.zig");

/// Executable container formats. PE32+/COFF for UEFI applications (arch-agnostic
/// container). Flat binary for baremetal. ELF executables live in each backend's
/// linker (`ld.writeElfExec`).
pub const pe = @import("vulcan-target/pe.zig");
pub const image = @import("vulcan-target/image.zig");

/// DWARF debug-info emitter (arch-independent container), for debugger support.
pub const dwarf = @import("vulcan-target/dwarf.zig");

/// Minimal ELF reader: locate `.text` + machine so a disassembler can be picked.
pub const elf_read = @import("vulcan-target/elf_read.zig");

/// Shared register-allocation support (target-independent live intervals).
pub const regalloc = @import("vulcan-target/regalloc.zig");

/// Shared, target-independent Wimmer-Franz register allocator (target abstraction + algorithm).
pub const wimmer = @import("vulcan-target/wimmer.zig");

/// Native target: the backend matching the host CPU, for in-process JIT.
pub const native = @import("vulcan-target/native.zig");

/// Runtime host-CPU feature detection (cpuid / HWCAP), so a JIT targets the silicon
/// it runs on rather than the machine it was built on.
pub const host = @import("vulcan-target/host.zig");

/// Pluggable W^X executable-memory layer for the JITs (posix host by default,
/// UEFI/Windows/macOS providers slot in via `mapWith`).
pub const jit_platform = @import("vulcan-target/jit_platform.zig");

test {
    std.testing.refAllDecls(@This());
}
