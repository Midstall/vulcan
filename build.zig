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

    // The GLSL -> SPIR-V shader compiler executable (declared before the if/else so both
    // the install/run-glsl step and the spirv tests can reference it).
    const glsl_cli = b.addExecutable(.{ .name = "vulcan-glsl", .root_module = b.createModule(.{
        .root_source_file = b.path("frontends/vulcan-glsl.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "vulcan-glsl", .module = vulcan_glsl }},
    }) });
    b.installArtifact(glsl_cli);

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

        // The GLSL -> SPIR-V shader compiler CLI runner (glsl_cli is declared above).
        if (target.result.os.tag != .uefi) {
            const run_glsl = b.addRunArtifact(glsl_cli);
            if (b.args) |a| run_glsl.addArgs(a);
            const run_glsl_step = b.step("run-glsl", "Run the GLSL->SPIR-V compiler: -- <input.glsl> [stage] [-o out.spv]");
            run_glsl_step.dependOn(&run_glsl.step);

            // The GLSL -> target-disassembly debugging tool: compile a shader and print the
            // native (or Wasm) code for one of its functions.
            const disasm_cli = b.addExecutable(.{ .name = "vulcan-disasm", .root_module = b.createModule(.{
                .root_source_file = b.path("frontends/vulcan-disasm.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "vulcan-target", .module = vulcan_target },
                    .{ .name = "vulcan-spirv", .module = vulcan_spirv },
                },
            }) });
            b.installArtifact(disasm_cli);
            const run_disasm = b.addRunArtifact(disasm_cli);
            if (b.args) |a| run_disasm.addArgs(a);
            const run_disasm_step = b.step("run-disasm", "Disassemble an ELF or SPIR-V binary to text assembly: -- <file>");
            run_disasm_step.dependOn(&run_disasm.step);

            // The microarch benchmark harness: JIT-compiles a fixed kernel set with and without
            // the microarch optimizer and measures the cycle gains for a chosen (or detected)
            // model. Builds for any hosted target; the non-host-arch and no-perf paths handle
            // non-aarch64/non-linux gracefully at runtime (see tools/uarch-bench.zig).
            const uarch_bench_cli = b.addExecutable(.{ .name = "vulcan-uarch-bench", .root_module = b.createModule(.{
                .root_source_file = b.path("tools/uarch-bench.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "vulcan-ir", .module = vulcan_ir },
                    .{ .name = "vulcan-opt", .module = vulcan_opt },
                    .{ .name = "vulcan-target", .module = vulcan_target },
                },
            }) });
            b.installArtifact(uarch_bench_cli);
            const run_uarch_bench = b.addRunArtifact(uarch_bench_cli);
            if (b.args) |a| run_uarch_bench.addArgs(a);
            const run_uarch_bench_step = b.step("run-uarch-bench", "Benchmark the microarch optimizer: -- [--model <tag> | --list | --custom]");
            run_uarch_bench_step.dependOn(&run_uarch_bench.step);
        }
    }

    // Declared above at top level so the spirv tests can depend on it.

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
            .{ .name = "vulcan-glsl", .module = vulcan_glsl },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(wasm_exec).step);

    // C backend execution tests: emit C from IR, compile with the host `cc`, run, and
    // cross-check against the native JIT. Skips when `cc` is absent.
    const c_exec = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/c/tests/native.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-glsl", .module = vulcan_glsl },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(c_exec).step);

    // JS backend execution tests: emit JS from IR, run it with Node.js, cross-check against
    // the native answer. Skips when no Node is found.
    const js_exec = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/js/tests/native.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-glsl", .module = vulcan_glsl },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(js_exec).step);

    // Three-way differential: native JIT vs the C backend (cc) vs the JS backend (node) over
    // GLSL, requiring all three to agree. Skips gracefully when a tool/host is unavailable.
    const differential = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-glsl", .module = vulcan_glsl },
            .{ .name = "vulcan-wasm", .module = vulcan_wasm },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(differential).step);

    // Loop-unroll differential oracle: build a loop, unroll one copy under a wide
    // model, JIT both the original and the unrolled function on the host, and
    // require identical results for every input. Runs where the native JIT has a
    // backend (aarch64/x86_64/riscv64/x86).
    const unroll_diff = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/unroll_differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(unroll_diff).step);

    // Prefetch differential oracle: build a function twice, hand-insert a `prefetch`
    // hint in one copy, JIT both on the host, and require identical results for every
    // input. Proves the PRFM (aarch64) / dropped-hint (elsewhere) lowering has no
    // observable effect. Runs where the native JIT has a backend.
    const prefetch_diff = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/prefetch_differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(prefetch_diff).step);

    // Vector memory coalescing differential oracle: build a scalar elementwise memory kernel twice,
    // run one copy through microarch.optimize (which coalesces contiguous scalar loads/stores into
    // wide vector loads/stores), JIT both on the host, and require identical per-element results,
    // including the safety case where a store between the loads makes coalescing decline. Runs where
    // the native JIT has a backend.
    const vector_mem_diff = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/vector_mem_differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(vector_mem_diff).step);

    // INT8 dot-product differential oracle: build a scalar `sum a[i]*b[i]` int8 reduction twice,
    // vectorize one copy to SDOT/UDOT under the ampere-altra model, JIT both on the host, and require
    // identical results across lengths including non-multiples of 16 (exercising the remainder loop),
    // for both signed (SDOT) and unsigned (UDOT). Aarch64-only; skips elsewhere.
    const dotprod_diff = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/dotprod_differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(dotprod_diff).step);

    // Microarch optimizer end-to-end harness (spec chunk 10 capstone): three kernels compiled both
    // plainly and through the full pipeline (microarch.optimize + the model-aware backend compile),
    // JIT'd on the host, results required identical, cycles logged. Aarch64-only; skips elsewhere.
    const microarch_e2e = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/microarch_e2e.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(microarch_e2e).step);

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
