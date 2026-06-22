const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The core IR library. Freestanding-clean: no libc, no OS, no global state.
    const vulcan_ir = b.addModule("vulcan-ir", .{
        .root_source_file = b.path("libs/vulcan-ir.zig"),
        .target = target,
        .optimize = optimize,
    });

    // The optimization framework: target-independent IR analyses and transforms.
    // Freestanding-clean. Depends only on the IR.
    const vulcan_opt = b.addModule("vulcan-opt", .{
        .root_source_file = b.path("libs/vulcan-opt.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "vulcan-ir", .module = vulcan_ir }},
    });

    // The SPIR-V frontend: read a SPIR-V binary and lower it to Vulcan IR.
    // Freestanding-clean. Depends only on the IR.
    const vulcan_spirv = b.addModule("vulcan-spirv", .{
        .root_source_file = b.path("libs/vulcan-spirv.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "vulcan-ir", .module = vulcan_ir }},
    });

    // The target seam: register sets, ABI, encoding, and codegen per target. Also
    // sees the SPIR-V frontend so it can execution-validate SPIR-V -> IR -> native.
    const vulcan_target = b.addModule("vulcan-target", .{
        .root_source_file = b.path("libs/vulcan-target.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-spirv", .module = vulcan_spirv },
        },
    });

    // The WebAssembly frontend: read a Wasm binary, lower it to IR, then JIT and run it.
    // Lowering depends only on the IR. The engine layer adds host JIT + memory/globals/
    // table/imports setup via the target seam.
    const vulcan_wasm = b.addModule("vulcan-wasm", .{
        .root_source_file = b.path("libs/vulcan-wasm.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    });

    // The GLSL frontend: parse GLSL source and lower it to Vulcan IR, and (via the
    // SPIR-V writer) emit SPIR-V.
    const vulcan_glsl = b.addModule("vulcan-glsl", .{
        .root_source_file = b.path("libs/vulcan-glsl.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-spirv", .module = vulcan_spirv },
        },
    });

    // The container emitters (PE/flat image) are standalone pure-byte modules, used by
    // the freestanding proof below.
    const vulcan_pe = b.createModule(.{ .root_source_file = b.path("libs/vulcan-target/pe.zig"), .target = target, .optimize = optimize });
    const vulcan_image = b.createModule(.{ .root_source_file = b.path("libs/vulcan-target/image.zig"), .target = target, .optimize = optimize });

    // The compiler core compiled as a link-free object: if it builds for a no-OS target,
    // those libraries are usable inside a baremetal/UEFI program (allocator-only, no
    // libc/syscalls). Built for whatever `-Dtarget` selects.
    const freestanding_proof = b.addObject(.{ .name = "vulcan-freestanding", .root_module = b.createModule(.{
        .root_source_file = b.path("test/freestanding_smoke.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-spirv", .module = vulcan_spirv },
            .{ .name = "vulcan-pe", .module = vulcan_pe },
            .{ .name = "vulcan-image", .module = vulcan_image },
        },
    }) });

    // The deliverable follows `-Dtarget`'s OS: a no-OS target installs the freestanding
    // proof object. Any hosted/UEFI target installs the Wasm runner, whose `main` is the
    // host CLI or the boot-time app (chosen at comptime). One target in, one output out.
    if (target.result.os.tag == .freestanding) {
        // An object has no standard install procedure. Install the emitted `.o` directly.
        const install_obj = b.addInstallBinFile(freestanding_proof.getEmittedBin(), "vulcan-freestanding.o");
        b.getInstallStep().dependOn(&install_obj.step);
    } else {
        // The Wasm runner JITs to native code, so it only builds for an arch the native
        // backend supports (UEFI targets among them), not for e.g. wasm32-wasi.
        const native_arch = switch (target.result.cpu.arch) {
            .aarch64, .x86_64, .x86, .riscv64 => true,
            else => false,
        };
        if (native_arch) {
            const uefi_mod = b.dependency("uefi", .{ .target = target, .optimize = optimize }).module("uefi");
            const wasm_cli = b.addExecutable(.{ .name = "vulcan-wasm", .root_module = b.createModule(.{
                .root_source_file = b.path("frontends/vulcan-wasm.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{ .{ .name = "vulcan-wasm", .module = vulcan_wasm }, .{ .name = "uefi", .module = uefi_mod } },
            }) });
            b.installArtifact(wasm_cli);
            const run_cli = b.addRunArtifact(wasm_cli);
            if (b.args) |a| run_cli.addArgs(a);
            const run_step = b.step("run-wasm", "Run the host Wasm CLI: -- <file.wasm> [export] [i32 args]");
            run_step.dependOn(&run_cli.step);
        }

        // The GLSL -> SPIR-V shader compiler, a host-only tool (no UEFI entry point).
        if (target.result.os.tag != .uefi) {
            const glsl_cli = b.addExecutable(.{ .name = "vulcan-glsl", .root_module = b.createModule(.{
                .root_source_file = b.path("frontends/vulcan-glsl.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "vulcan-glsl", .module = vulcan_glsl }},
            }) });
            b.installArtifact(glsl_cli);
            const run_glsl = b.addRunArtifact(glsl_cli);
            if (b.args) |a| run_glsl.addArgs(a);
            const run_glsl_step = b.step("run-glsl", "Run the GLSL->SPIR-V compiler: -- <input.glsl> [stage] [-o out.spv]");
            run_glsl_step.dependOn(&run_glsl.step);
        }
    }

    const test_step = b.step("test", "Run all tests");

    const ir_tests = b.addTest(.{ .root_module = vulcan_ir });
    test_step.dependOn(&b.addRunArtifact(ir_tests).step);

    const opt_tests = b.addTest(.{ .root_module = vulcan_opt });
    test_step.dependOn(&b.addRunArtifact(opt_tests).step);

    const spirv_tests = b.addTest(.{ .root_module = vulcan_spirv });
    test_step.dependOn(&b.addRunArtifact(spirv_tests).step);

    const target_tests = b.addTest(.{ .root_module = vulcan_target });
    test_step.dependOn(&b.addRunArtifact(target_tests).step);

    // The Wasm frontend's tests: structural (parsing + lowering) plus the engine.
    const wasm_tests = b.addTest(.{ .root_module = vulcan_wasm });
    test_step.dependOn(&b.addRunArtifact(wasm_tests).step);

    // Wasm execution tests: lower Wasm to IR, then JIT for the host and run.
    const wasm_exec = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-wasm/tests/native.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-wasm", .module = vulcan_wasm },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(wasm_exec).step);

    // GLSL frontend tests: parsing/lowering (IR only), plus execution (GLSL -> IR ->
    // host JIT -> run) for scalar functions.
    const glsl_tests = b.addTest(.{ .root_module = vulcan_glsl });
    test_step.dependOn(&b.addRunArtifact(glsl_tests).step);
    const glsl_exec = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-glsl/tests/native.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-glsl", .module = vulcan_glsl },
            .{ .name = "vulcan-spirv", .module = vulcan_spirv },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(glsl_exec).step);

    // The freestanding object is a compile check too (no run), so `test` keeps the core
    // building for whatever target is selected.
    test_step.dependOn(&freestanding_proof.step);
}
