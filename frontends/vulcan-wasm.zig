//! vulcan-wasm: JIT and run WebAssembly with the Vulcan engine. One file, two comptime-picked
//! targets sharing the engine. Only the entry point, executable-memory provider, and I/O differ.
//!   - Hosted: `vulcan-wasm <file.wasm> [export] [i32 args...]`, JITs for the host, prints the result.
//!   - UEFI: a boot app that JITs an embedded module into boot-services memory and prints main()
//!     to the console (boots under QEMU+OVMF).

const std = @import("std");
const builtin = @import("builtin");
const engine = @import("vulcan-wasm");
const uefi = @import("uefi");

/// The entry point for the build target: the hosted CLI or the UEFI app.
pub const main = if (builtin.os.tag == .uefi) uefiMain else hostedMain;

// Route std.log and panics to the firmware console on UEFI. The host keeps the defaults:
// uefi.zig's panic spins forever, which would hang the CLI.
pub const std_options: std.Options = if (builtin.os.tag == .uefi) uefi.std_options else .{};
pub const panic = if (builtin.os.tag == .uefi) uefi.panic else std.debug.FullPanic(std.debug.defaultPanic);

fn hostedMain(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var it = try init.minimal.args.iterateAllocator(allocator); // cross-platform (works on WASI)
    defer it.deinit();
    _ = it.skip(); // argv0
    const path = it.next() orelse {
        std.debug.print("usage: vulcan-wasm <file.wasm> [export] [i32 args...]\n", .{});
        return error.Usage;
    };

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));

    // Peek the imports: a module importing wasi_snapshot_preview1 runs under the WASI runtime.
    var module = try engine.load(allocator, bytes);
    var is_wasi = false;
    for (module.imports) |n| {
        if (std.mem.startsWith(u8, n, "wasi_snapshot_preview1.")) {
            is_wasi = true;
            break;
        }
    }

    if (is_wasi) {
        const host_imports = try allocator.alloc(usize, module.imports.len);
        for (module.imports, 0..) |n, i| host_imports[i] = engine.wasi.resolve(n) orelse {
            std.debug.print("vulcan-wasm: unsupported import '{s}'\n", .{n});
            module.deinit(allocator);
            return error.Unsupported;
        };
        module.deinit(allocator);

        var inst = try engine.Instance.instantiate(allocator, bytes, host_imports);
        defer inst.deinit();
        engine.wasi.setup(inst.memory, &.{path}); // argv[0] = the module path
        inst.call0(void, "_start") catch |e| {
            std.debug.print("vulcan-wasm: {s}\n", .{@errorName(e)});
            return e;
        };
        return; // _start returned, or proc_exit already exited the process
    }
    module.deinit(allocator);

    // Otherwise, call an export with i32 arguments.
    const export_name = it.next() orelse "main";
    var iargs: [3]i32 = undefined;
    var nargs: usize = 0;
    while (it.next()) |arg| {
        if (nargs >= iargs.len) return error.TooManyArgs;
        iargs[nargs] = try std.fmt.parseInt(i32, arg, 10);
        nargs += 1;
    }

    var inst = try engine.Instance.instantiate(allocator, bytes, &.{});
    defer inst.deinit();
    const result = switch (nargs) {
        0 => try inst.call0(i32, export_name),
        1 => try inst.call1(i32, i32, export_name, iargs[0]),
        2 => try inst.call2(i32, i32, i32, export_name, iargs[0], iargs[1]),
        else => try inst.call3(i32, i32, i32, i32, export_name, iargs[0], iargs[1], iargs[2]),
    };
    // The computed result is the program's output: stdout, not stderr.
    var buf: [64]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print("{d}\n", .{result});
    try w.interface.flush();
}

// An embedded module returning 1337: (func (export "main") (result i32) (i32.const 1000)
// (i32.const 337) i32.add). The engine JITs it into boot-services memory (engine.default_provider).
const wasm_blob = [_]u8{
    0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7F, // type: () -> i32
    0x03, 0x02, 0x01, 0x00, // func 0 has type 0
    0x07, 0x08, 0x01, 0x04, 'm', 'a', 'i', 'n', 0x00, 0x00, // export "main" = func 0
    0x0A, 0x0B, 0x01, 0x09, 0x00, // code section: 1 body of 9 bytes, 0 locals
    0x41, 0xE8, 0x07, // i32.const 1000
    0x41, 0xD1, 0x02, // i32.const 337
    0x6A, // i32.add
    0x0B, // end
};

// Working memory for lowering, IR, and instance buffers. The JITed code comes from
// boot-services pages instead.
var heap_buf: [4 * 1024 * 1024]u8 align(16) = undefined;

fn uefiMain() void {
    const out = uefi.init();
    var fba = std.heap.FixedBufferAllocator.init(&heap_buf);
    const allocator = fba.allocator();

    out.writeAll("Vulcan: JITing WebAssembly under UEFI... main() = ") catch {};
    // FIXME: accept a path to a WASM file.
    var inst = engine.Instance.instantiate(allocator, &wasm_blob, &.{}) catch {
        out.writeAll("instantiate failed\n") catch {};
        uefi.halt();
    };
    defer inst.deinit();
    const result = inst.call0(i32, "main") catch {
        out.writeAll("call failed\n") catch {};
        uefi.halt();
    };
    out.print("{d}\n", .{result}) catch {}; // 1337
    uefi.halt(); // keep the output on screen
}
