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

    // The target-neutral accelerator kernel ABI: builtins, parameter layout, and the launch
    // metadata a runtime reads. Freestanding-clean. Depends only on the IR, so the frontends
    // can tag kernels with it without depending on the target seam. It is declared before the
    // frontends because they import it.
    const vulcan_gpu = b.addModule("vulcan-gpu", .{
        .root_source_file = b.path("libs/vulcan-gpu.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "vulcan-ir", .module = vulcan_ir }},
    });

    // The SPIR-V frontend: read a SPIR-V binary and lower it to Vulcan IR. Freestanding-clean.
    // Depends on the IR and on the kernel ABI, whose vocabulary it tags kernel parameters with.
    const vulcan_spirv = b.addModule("vulcan-spirv", .{
        .root_source_file = b.path("libs/vulcan-spirv.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-gpu", .module = vulcan_gpu },
        },
    });

    // The shared static ELF linker: parse relocatable objects, resolve relocations,
    // bind external calls through per-arch GOT stubs, and wrap the result in a static
    // executable. Imports only `std` (defines its own ELF/reloc structs), so the target
    // seam can depend on it without a dependency cycle.
    const vulcan_link = b.addModule("vulcan-link", .{
        .root_source_file = b.path("libs/vulcan-link.zig"),
        .target = target,
        .optimize = optimize,
    });

    // The target seam: register sets, ABI, encoding, and codegen per target. Also
    // sees the SPIR-V frontend so it can execution-validate SPIR-V -> IR -> native,
    // the kernel ABI so the accelerator backends read the launch contract, and the
    // shared linker for object linking/JIT/executable emission.
    const vulcan_target = b.addModule("vulcan-target", .{
        .root_source_file = b.path("libs/vulcan-target.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-spirv", .module = vulcan_spirv },
            .{ .name = "vulcan-gpu", .module = vulcan_gpu },
            .{ .name = "vulcan-link", .module = vulcan_link },
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

    // The C frontend: parse C source and lower it to Vulcan IR.
    const vulcan_cc = b.addModule("vulcan-cc", .{
        .root_source_file = b.path("libs/vulcan-cc.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
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

    // The Vulcan C Compiler driver executable (clang-compatible CLI surface, SM1).
    // Hoisted into a `const` module (SM15 M5a) so both the exe and its in-file `parseArgs`
    // tests below share one module, same pattern as `vulcan_ld_module`.
    const vcc_module = b.createModule(.{
        .root_source_file = b.path("frontends/vcc.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-cc", .module = vulcan_cc },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-link", .module = vulcan_link },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
        },
    });
    const vcc_cli = b.addExecutable(.{ .name = "vcc", .root_module = vcc_module });
    b.installArtifact(vcc_cli);

    // The `ld.vulcan` linker frontend CLI: a standalone driver over the shared static
    // linker (`vulcan-link`), so the linker is usable with any compiler's `.o`/`.a`
    // output, not just Vulcan's own frontends. Only `run`/`main` (what actually ships)
    // depends on `vulcan-link`; `vulcan-target`/`vulcan-ir` are pulled into the module
    // solely for the in-file test's synthetic `.o` inputs. Hoisted into a `const` so
    // both the exe and its test below share one module.
    const vulcan_ld_module = b.createModule(.{
        .root_source_file = b.path("frontends/vulcan-ld.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-link", .module = vulcan_link },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-ir", .module = vulcan_ir },
        },
    });
    const vulcan_ld_cli = b.addExecutable(.{ .name = "ld.vulcan", .root_module = vulcan_ld_module });
    b.installArtifact(vulcan_ld_cli);

    // The deliverable follows `-Dtarget`'s OS: a no-OS target installs the freestanding
    // proof object. Any hosted/UEFI target installs the Wasm runner, whose `main` is the
    // host CLI or the boot-time app (chosen at comptime). One target in, one output out.
    //
    // The Wasm CLI is held here, outside the branch, so the toolchain test at the test step
    // can drive the freshly built binary. It stays null on a target that does not build it.
    var wasm_cli_artifact: ?*std.Build.Step.Compile = null;
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
            wasm_cli_artifact = wasm_cli;
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

    const gpu_tests = b.addTest(.{ .root_module = vulcan_gpu });
    test_step.dependOn(&b.addRunArtifact(gpu_tests).step);

    const spirv_tests = b.addTest(.{ .root_module = vulcan_spirv });
    test_step.dependOn(&b.addRunArtifact(spirv_tests).step);

    const target_tests = b.addTest(.{ .root_module = vulcan_target });
    test_step.dependOn(&b.addRunArtifact(target_tests).step);

    // The shared linker's own unit tests (std-only: ELF parsing + RVC-compression
    // primitives). Object-input link tests live in the riscv64 consumer tests.
    const link_tests = b.addTest(.{ .root_module = vulcan_link });
    test_step.dependOn(&b.addRunArtifact(link_tests).step);

    // `ld.vulcan`'s own CLI test: builds synthetic `.o`/`.a` inputs, drives `run`
    // directly (no process spawn), and natively executes the produced ELF.
    const vulcan_ld_tests = b.addTest(.{ .root_module = vulcan_ld_module });
    test_step.dependOn(&b.addRunArtifact(vulcan_ld_tests).step);

    // `vcc`'s own `parseArgs` tests (SM15 M5a): drives the flag-classification/version-probe
    // surface IN-PROCESS (no process spawn), same pattern as `vulcan_ld_tests` above.
    const vcc_tests_cli = b.addTest(.{ .root_module = vcc_module });
    test_step.dependOn(&b.addRunArtifact(vcc_tests_cli).step);

    // Wimmer-Franz allocator target-abstraction unit tests: assert the aarch64 RegDescription
    // (pools, entry-param pinning, call clobbers, scratch) the shared allocator consumes.
    const wimmer_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/wimmer_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(wimmer_tests).step);

    // GPU offload execution tests: lower a kernel to a host loop nest with `vulcan-gpu`, JIT
    // it for the host, run the whole grid, and assert the buffer it wrote. Every other kernel
    // test is structural, so this is the one that proves a kernel computes the right numbers.
    const gpu_offload_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/gpu_offload.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-gpu", .module = vulcan_gpu },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(gpu_offload_tests).step);

    // Matmul expansion execution tests: rewrite the et-soc tensor `matmul` into a scalar loop
    // nest with `vulcan-ir.expand`, JIT it for the host, and assert the C matrix it wrote. A
    // wrong index or a dropped accumulate has the same opcodes as a correct nest, so only running
    // it tells them apart. This is the reference a tensor lowering gets checked against.
    const matmul_expand_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/matmul_expand.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(matmul_expand_tests).step);

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

    // Wasm tests over modules a REAL toolchain produced: `zig cc` builds them at test time,
    // and the WASI one runs through the freshly built CLI. Every other Wasm test assembles
    // its module by hand, which is how a loader fault that refused every linker-produced
    // module stayed invisible. `zig_exe` is the same `zig` that builds this, so the test does
    // not depend on the PATH. Wired only where the CLI exists (a native arch, hosted or UEFI).
    if (wasm_cli_artifact) |cli| {
        const wasm_toolchain_opts = b.addOptions();
        wasm_toolchain_opts.addOption([]const u8, "zig_exe", b.graph.zig_exe);
        wasm_toolchain_opts.addOptionPath("vulcan_wasm_bin", cli.getEmittedBin());
        const wasm_toolchain = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("libs/vulcan-wasm/tests/toolchain.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vulcan-wasm", .module = vulcan_wasm },
                .{ .name = "build_options", .module = wasm_toolchain_opts.createModule() },
            },
        }) });
        test_step.dependOn(&b.addRunArtifact(wasm_toolchain).step);
    }

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

    // C frontend execution tests: lower C to IR, JIT on the host, and diff against gcc.
    const cc_exec = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-cc/tests/native.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-cc", .module = vulcan_cc },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-link", .module = vulcan_link },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(cc_exec).step);

    // SM11 Task 3: external direct calls, proven end to end - the C frontend lowers a call to
    // a declared-external function into an undefined-symbol call that the SM10 dynamic linker
    // resolves to a real `.so` PLT import (native aarch64 + qemu x86_64/i386/riscv64).
    const cc_external = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-cc/tests/external_linkage.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-cc", .module = vulcan_cc },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-link", .module = vulcan_link },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(cc_external).step);

    // SM15 M5c CAPSTONE: the default-executable AUTOLINK, proven by RUNNING the linked binary.
    // The freshly built `vcc` binary's path is handed to the test through a `build_options`
    // module (`vcc_cli.getEmittedBin()` also makes the test depend on that binary, so it is
    // built first). The test drives `vcc hello.c -o hello` with no hand-supplied crt/`-lc`/
    // `--dynamic-linker`, runs `./hello`, and asserts its output - skipping when the host
    // toolchain (gcc/crt/glibc/loader) is absent.
    const vcc_autolink_opts = b.addOptions();
    vcc_autolink_opts.addOptionPath("vcc_bin", vcc_cli.getEmittedBin());
    const cc_autolink = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-cc/tests/autolink.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = vcc_autolink_opts.createModule() },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(cc_autolink).step);

    // SM12 Task 1: the variadic frontend foundation (`...`, `__builtin_va_list`,
    // `<stdarg.h>`, default argument promotions, the IR call `is_variadic`/`num_fixed`
    // flags). SM12 Task 2 adds the backend call-side end-to-end tests (a vcc program that
    // CALLS real glibc `printf`, linked against `libc.so.6` and run under the real `ld.so`),
    // so the module now also imports `vulcan-target` and `vulcan-link`.
    const cc_variadic = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-cc/tests/variadic.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-cc", .module = vulcan_cc },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-link", .module = vulcan_link },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(cc_variadic).step);

    // SM13 M3a Task 5 (CAPSTONE): VCC's own preprocessor - `SystemPredef` (Task 3) +
    // `FsResolver` (Task 4) + the char-literal/`__has_*` (Task 1) and variadic-macro (Task 2)
    // work underneath - reduces the REAL host glibc `#include <stdio.h>` chain to no-error.
    // Only needs `vulcan-cc` itself (pure preprocessing, no lowering/codegen/linking).
    const cc_preproc_glibc = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-cc/tests/preproc_glibc.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-cc", .module = vulcan_cc },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(cc_preproc_glibc).step);

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

    // Accumulator-splitting unroll differential oracle: build a reduction twice, split-unroll one copy
    // (main loop with K independent partials + remainder), JIT both on the host, and require identical
    // results for every input (including trip counts not divisible by K, and below K).
    const splitunroll_diff = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/splitunroll_differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(splitunroll_diff).step);

    // Loop-vectorizer differential oracle: build a map loop twice, run loopvec (and the full pipeline
    // so SLP widens it) on one, JIT both, run over real arrays, require bit-identical output for every
    // trip count (0 / below V / exactly V / multiples / non-multiples).
    const loopvec_diff = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/loopvec_differential.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-opt", .module = vulcan_opt },
            .{ .name = "vulcan-target", .module = vulcan_target },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(loopvec_diff).step);

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

    // SM10 P2c Task 4: runtime `-target` dispatch. For each of the 4 backends, emits
    // an `int main(void){return 42;}` object via `native.writeObjectDataFor` (host-
    // independent - the switch dispatches to that backend's own object writer
    // regardless of which arch this test binary itself runs on), links + wraps it in
    // a runnable ELF, and executes it: natively for aarch64, under qemu-<arch> for
    // the other three (skips cleanly if that qemu isn't installed).
    const cross_target = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("libs/vulcan-target/tests/cross_target.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulcan-ir", .module = vulcan_ir },
            .{ .name = "vulcan-target", .module = vulcan_target },
            .{ .name = "vulcan-link", .module = vulcan_link },
        },
    }) });
    test_step.dependOn(&b.addRunArtifact(cross_target).step);

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

    const cc_tests = b.addTest(.{ .root_module = vulcan_cc });
    test_step.dependOn(&b.addRunArtifact(cc_tests).step);

    // The freestanding object is a compile check too (no run), so `test` keeps the core
    // building for whatever target is selected.
    test_step.dependOn(&freestanding_proof.step);
}
