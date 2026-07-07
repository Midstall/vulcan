//! vulcan-glsl: a GLSL -> SPIR-V shader compiler CLI over the `vulcan-glsl` frontend.
//!
//! Usage:
//!   vulcan-glsl <input.glsl> [fragment|vertex|compute] [-o <output.spv>]
//!
//! Parses the GLSL source, lowers it to Vulcan IR, and emits a SPIR-V entry-point shader
//! for the chosen stage (default: fragment). The binary is written to `<output.spv>` (or
//! `<input>.spv` if `-o` is omitted), and a one-line summary is printed to stdout.

const std = @import("std");
const glsl = @import("vulcan-glsl");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var it = try init.minimal.args.iterateAllocator(allocator); // cross-platform (works on WASI)
    defer it.deinit();
    _ = it.skip(); // argv0
    const input = it.next() orelse {
        std.debug.print("usage: vulcan-glsl <input.glsl> [fragment|vertex|compute] [-o <output.spv>]\n", .{});
        return error.Usage;
    };

    var stage: glsl.Stage = .fragment;
    var output: ?[]const u8 = null;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-o")) {
            output = it.next() orelse {
                std.debug.print("error: -o needs an output path\n", .{});
                return error.Usage;
            };
        } else if (std.mem.eql(u8, arg, "fragment")) {
            stage = .fragment;
        } else if (std.mem.eql(u8, arg, "vertex")) {
            stage = .vertex;
        } else if (std.mem.eql(u8, arg, "compute")) {
            stage = .compute;
        } else {
            std.debug.print("error: unknown argument '{s}'\n", .{arg});
            return error.Usage;
        }
    }

    const source = try std.Io.Dir.cwd().readFileAlloc(io, input, allocator, .limited(16 * 1024 * 1024));
    const words = try glsl.compileShaderToSpirv(allocator, source, stage);

    const out_path = output orelse try std.fmt.allocPrint(allocator, "{s}.spv", .{input});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = std.mem.sliceAsBytes(words) });

    var buf: [256]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print("{s}: {s} shader, {d} words -> {s}\n", .{ input, @tagName(stage), words.len, out_path });
    try w.interface.flush();
}
