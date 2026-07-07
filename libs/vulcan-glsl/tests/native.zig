//! Execution tests for the GLSL frontend: compile GLSL source to Vulcan IR, JIT it for
//! the host via `vulcan-target.native`, and run it. The host is the JIT target, so scalar
//! GLSL functions run natively in-process.

const std = @import("std");
const glsl = @import("vulcan-glsl");
const spirv = @import("vulcan-spirv");
const target = @import("vulcan-target");
const native = target.native;

test "dwarf: riscv64 also builds a source-line table from the same debug.line attrs" {
    // The line-info pipeline is target-independent: riscv64 isel reads the same debug.line IR
    // attributes to produce (offset, line) rows, proving DWARF is not aarch64-only.
    const a = std.testing.allocator;
    const dwarf = target.dwarf;
    const src = "int f(int a) {\n  int b = a + 5;\n  return b * 2;\n}";
    var module = try glsl.compile(a, src);
    defer module.deinit(a);
    const func = module.find("f") orelse return error.MissingFunction;

    const cl = try target.riscv64.isel.selectFunctionWithLines(a, func);
    defer a.free(cl.code);
    defer a.free(cl.lines);
    // Rows for line 2 (int b = a + 5) and line 3 (return b * 2).
    var saw2 = false;
    var saw3 = false;
    for (cl.lines) |e| {
        if (e.line == 2) saw2 = true;
        if (e.line == 3) saw3 = true;
    }
    try std.testing.expect(saw2 and saw3);

    // And it emits a readelf-decodable .debug_line at real riscv64 offsets.
    const base: u64 = 0x2000;
    const rows = try a.alloc(dwarf.LineRow, cl.lines.len);
    defer a.free(rows);
    for (cl.lines, 0..) |e, i| rows[i] = .{ .address = base + e.offset, .line = e.line };
    const elf = try dwarf.emitLineElf(a, "rv.glsl", rows, base + cl.code.len * 4);
    defer a.free(elf);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "rv.o", .data = elf });
    const res = std.process.run(a, std.testing.io, .{ .argv = &.{ "readelf", "--debug-dump=decodedline", "rv.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "rv.glsl") != null);
}

test "dwarf object: a compiled GLSL module ships inline DWARF (functions + lines, readelf)" {
    // The capstone: a real relocatable .o carrying .text + .symtab + .debug_info + .debug_line,
    // so objdump/gdb see function names, PC ranges, AND source lines on actual compiled code.
    const a = std.testing.allocator;
    const aa = target.aarch64;
    const src = "int f(int a) {\n  int b = a + 5;\n  return b * 2;\n}\nint g(int x) {\n  return x - 1;\n}";
    var module = try glsl.compile(a, src);
    defer module.deinit(a);
    const f = module.find("f") orelse return error.MissingFunction;
    const g = module.find("g") orelse return error.MissingFunction;

    var mod = aa.link.Module{};
    defer mod.deinit(a);
    try mod.addFunction(a, "f", f);
    try mod.addFunction(a, "g", g);
    const obj = try aa.object.writeModuleWithDebug(a, &mod, "shader.glsl");
    defer a.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "o.o", .data = obj });
    const res = std.process.run(a, std.testing.io, .{ .argv = &.{ "readelf", "--debug-dump=info", "--debug-dump=decodedline", "o.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) {
        std.debug.print("readelf failed:\n{s}\n", .{res.stderr});
        return error.ReadelfFailed;
    }
    // The DWARF has the CU + two subprogram DIEs, and the line table names the source file.
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_compile_unit") != null);
    try std.testing.expect(std.mem.count(u8, res.stdout, "DW_TAG_subprogram") >= 2);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "shader.glsl") != null);
}

test "dwarf end-to-end: GLSL -> code + line table -> .debug_line (readelf-validated)" {
    // The full source-line pipeline: lex line numbers -> AST -> debug.line IR attrs -> isel
    // records (byte offset, line) rows -> DWARF .debug_line -> binutils decodes it. Confirms
    // real GLSL source lines map to real machine-code offsets.
    const a = std.testing.allocator;
    const dwarf = target.dwarf;
    // decl on line 2, return on line 3.
    const src = "int f(int a) {\n  int b = a + 5;\n  return b * 2;\n}";
    var module = try glsl.compile(a, src);
    defer module.deinit(a);
    const func = module.find("f") orelse return error.MissingFunction;

    const cl = try target.aarch64.isel.selectFunctionWithLines(a, func);
    defer a.free(cl.code);
    defer a.free(cl.lines);
    try std.testing.expect(cl.lines.len >= 2); // at least lines 2 and 3

    const base: u64 = 0x1000;
    const rows = try a.alloc(dwarf.LineRow, cl.lines.len);
    defer a.free(rows);
    for (cl.lines, 0..) |e, i| rows[i] = .{ .address = base + e.offset, .line = e.line };
    const elf = try dwarf.emitLineElf(a, "shader.glsl", rows, base + cl.code.len * 4);
    defer a.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "l.o", .data = elf });
    const res = std.process.run(a, std.testing.io, .{ .argv = &.{ "readelf", "--debug-dump=decodedline", "l.o" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    // The decoded line table names the file and lists source lines 2 and 3.
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "shader.glsl") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "2") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "3") != null);
}

fn runF32(allocator: std.mem.Allocator, src: []const u8, name: []const u8, comptime Fn: type, arg: anytype) !f32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find(name) orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return @call(.auto, buf.entry(Fn, 0), arg);
}

fn runI32(allocator: std.mem.Allocator, src: []const u8, name: []const u8, comptime Fn: type, arg: anytype) !i32 {
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.find(name) orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    return @call(.auto, buf.entry(Fn, 0), arg);
}

test "glsl->vectorize->NEON: a vec4 add fuses and runs to the right sum" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const opt = @import("vulcan-opt");
    // The frontend scalarizes `a + b` to four parallel fadds, which the auto-vectorizer
    // fuses into one NEON fadd.4s. End to end: GLSL -> IR -> vectorize -> aarch64 -> run.
    const src = "float f(vec4 a, vec4 b) { vec4 c = a + b; return c.x + c.y + c.z + c.w; }";
    var module = try glsl.compile(allocator, src);
    defer module.deinit(allocator);
    const f = module.findMut("f") orelse return error.MissingFunction;

    try std.testing.expect(try opt.vectorize.run(allocator, f));
    var has_vec = false;
    for (0..f.instCount()) |i| {
        if (f.opcodeMut(@enumFromInt(i)).* == .arith) {
            const res = f.instResult(@enumFromInt(i)).?;
            if (f.types.type_kind(f.valueType(res)) == .vector) has_vec = true;
        }
    }
    try std.testing.expect(has_vec); // the four scalar fadds became one vector op

    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const Fn = *const fn (f32, f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    const r = @call(.auto, buf.entry(Fn, 0), .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 4), @as(f32, 10), @as(f32, 20), @as(f32, 30), @as(f32, 40) });
    try std.testing.expectEqual(@as(f32, 110), r); // (1+10)+(2+20)+(3+30)+(4+40)
}

test "glsl: scalar float arithmetic" {
    const allocator = std.testing.allocator;
    const r = try runF32(allocator, "float f(float x) { return x * 2.0 + 1.0; }", "f", *const fn (f32) callconv(.c) f32, .{@as(f32, 20.0)});
    try std.testing.expectEqual(@as(f32, 41.0), r);
}

test "glsl: locals and reassignment" {
    const allocator = std.testing.allocator;
    // y = x + 5, y = y * 2, return y   for x = 3 -> 16
    const src = "float f(float x) { float y = x + 5.0; y = y * 2.0; return y; }";
    try std.testing.expectEqual(@as(f32, 16.0), try runF32(allocator, src, "f", *const fn (f32) callconv(.c) f32, .{@as(f32, 3.0)}));
}

test "glsl: int arithmetic and mod" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(i32, 6), try runI32(allocator, "int f(int a, int b) { return a / b; }", "f", *const fn (i32, i32) callconv(.c) i32, .{ @as(i32, 20), @as(i32, 3) }));
    try std.testing.expectEqual(@as(i32, 2), try runI32(allocator, "int f(int a, int b) { return a % b; }", "f", *const fn (i32, i32) callconv(.c) i32, .{ @as(i32, 20), @as(i32, 3) }));
}

test "glsl: int->float promotion in mixed expression" {
    const allocator = std.testing.allocator;
    // x * 2 + 1 with x float: the int literals promote to float -> 41.0
    try std.testing.expectEqual(@as(f32, 41.0), try runF32(allocator, "float f(float x) { return x * 2 + 1; }", "f", *const fn (f32) callconv(.c) f32, .{@as(f32, 20.0)}));
}

test "glsl: type-constructor conversion" {
    const allocator = std.testing.allocator;
    // truncate a float to int: int(x) for x = 3.7 -> 3
    try std.testing.expectEqual(@as(i32, 3), try runI32(allocator, "int f(float x) { return int(x); }", "f", *const fn (f32) callconv(.c) i32, .{@as(f32, 3.7)}));
}

test "glsl: comparison yields 0/1" {
    const allocator = std.testing.allocator;
    // a < b -> 1/0
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, "int f(float a, float b) { return a < b; }", "f", *const fn (f32, f32) callconv(.c) i32, .{ @as(f32, 2.0), @as(f32, 5.0) }));
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, "int f(float a, float b) { return a < b; }", "f", *const fn (f32, f32) callconv(.c) i32, .{ @as(f32, 5.0), @as(f32, 2.0) }));
}

// Lighting builtins: distance, reflect, faceforward.

test "glsl: distance and reflect" {
    const allocator = std.testing.allocator;
    // distance((0,0,0),(3,4,0)) = 5
    const F6 = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 5.0), try runF32(allocator, "float f(vec3 a, vec3 b){return distance(a, b);}", "f", F6, .{ @as(f32, 0), @as(f32, 0), @as(f32, 0), @as(f32, 3), @as(f32, 4), @as(f32, 0) }));
    // reflect((1,-1,0),(0,1,0)) = (1,-1,0) - 2*(-1)*(0,1,0) = (1,1,0), sum = 2
    try std.testing.expectEqual(@as(f32, 2.0), try runF32(allocator, "float f(vec3 i, vec3 n){return dot(reflect(i, n), vec3(1.0));}", "f", F6, .{ @as(f32, 1), @as(f32, -1), @as(f32, 0), @as(f32, 0), @as(f32, 1), @as(f32, 0) }));
}

test "glsl->spir-v: reflect round-trips" {
    const allocator = std.testing.allocator;
    const F6 = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 2.0), try runSpirvF32(allocator, "float f(vec3 i, vec3 n){return dot(reflect(i, n), vec3(1.0));}", F6, .{ @as(f32, 1), @as(f32, -1), @as(f32, 0), @as(f32, 0), @as(f32, 1), @as(f32, 0) }));
}

// refract(I, N, eta): the glmark2 light-refract scene's builtin. Verify a known transmitted
// case AND the total-internal-reflection (k < 0 -> zero vector) case.
test "glsl: refract (transmission + total-internal-reflection)" {
    const allocator = std.testing.allocator;
    // Head-on incidence: I=(0,0,-1) (unit, into the surface), N=(0,0,1) (unit, outward),
    // eta=1.0 -> the ray passes straight through unchanged: refract = (0,0,-1).
    // d = dot(N,I) = -1, k = 1 - 1*(1 - 1) = 1, sqrt(k)=1, scale = 1*(-1)+1 = 0.
    // result = 1*I - 0*N = I = (0,0,-1). dot(result, (1,1,1)) = -1.
    const F6 = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    const sum_src = "float f(vec3 i, vec3 n){return dot(refract(i, n, 1.0), vec3(1.0));}";
    try std.testing.expectEqual(@as(f32, -1.0), try runF32(allocator, sum_src, "f", F6, .{ @as(f32, 0), @as(f32, 0), @as(f32, -1), @as(f32, 0), @as(f32, 0), @as(f32, 1) }));

    // Total internal reflection: I=(1,0,0) along the surface, N=(0,0,1), eta=2: d=0,
    // k = 1 - 4*(1) = -3 < 0, so refract returns the zero vector (dot with ones = 0).
    const tir_src = "float f(vec3 i, vec3 n, float e){return dot(refract(i, n, e), vec3(1.0));}";
    const F7 = *const fn (f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, tir_src, "f", F7, .{ @as(f32, 1), @as(f32, 0), @as(f32, 0), @as(f32, 0), @as(f32, 0), @as(f32, 1), @as(f32, 2) }));

    // A genuine bend: I=(0.6,0,-0.8) (unit), N=(0,0,1), eta=0.5. d=-0.8, d*d=0.64,
    // k = 1 - 0.25*(1-0.64) = 0.91, sqrt(k)=0.953939, scale = 0.5*(-0.8)+sqrt = 0.553939.
    // result.x = 0.5*0.6 = 0.3, result.z = 0.5*(-0.8) - 0.553939 = -0.953939.
    const xz_src = "float f(vec3 i, vec3 n){vec3 r = refract(i, n, 0.5); return r.x * 100.0 + r.z;}";
    const got = try runF32(allocator, xz_src, "f", F6, .{ @as(f32, 0.6), @as(f32, 0), @as(f32, -0.8), @as(f32, 0), @as(f32, 0), @as(f32, 1) });
    // expected = 0.3*100 + (-0.953939) = 29.046061
    try std.testing.expect(@abs(got - 29.046061) < 0.001);
}

test "glsl->spir-v: refract round-trips" {
    const allocator = std.testing.allocator;
    const F6 = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    // Head-on eta=1.0 transmission as above: dot(refract, ones) = -1.
    try std.testing.expectEqual(@as(f32, -1.0), try runSpirvF32(allocator, "float f(vec3 i, vec3 n){return dot(refract(i, n, 1.0), vec3(1.0));}", F6, .{ @as(f32, 0), @as(f32, 0), @as(f32, -1), @as(f32, 0), @as(f32, 0), @as(f32, 1) }));
}

// Matrices (scalarized column-major): construction, mat*vec, mat*mat.

test "glsl: mat3 * vec3 (identity and scaling)" {
    const allocator = std.testing.allocator;
    // identity * v -> v, sum the result
    const src = "float f(vec3 v) { mat3 m = mat3(1.0); return dot(m * v, vec3(1.0)); }";
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, src, "f", F3, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3) })); // 1+2+3
    // a diagonal scale matrix mat3(2.0) doubles each component
    const src2 = "float f(vec3 v) { mat3 m = mat3(2.0); return dot(m * v, vec3(1.0)); }";
    try std.testing.expectEqual(@as(f32, 12.0), try runF32(allocator, src2, "f", F3, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3) }));
}

test "glsl: mat2 from explicit columns, mat*vec" {
    const allocator = std.testing.allocator;
    // m columns: col0=(1,0), col1=(0,1) is identity. col0=(2,3),col1=(4,5)
    // m * (1,1) = col0*1 + col1*1 = (2+4, 3+5) = (6, 8), sum = 14
    const src = "float f() { mat2 m = mat2(2.0, 3.0, 4.0, 5.0); vec2 v = vec2(1.0, 1.0); return dot(m * v, vec2(1.0)); }";
    const F0 = *const fn () callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 14.0), try runF32(allocator, src, "f", F0, .{}));
}

test "glsl: mat2 * mat2" {
    const allocator = std.testing.allocator;
    // A = [[1,2],[3,4]] col-major comps (1,2,3,4): col0=(1,2), col1=(3,4)
    // A*A then apply to (1,0) and sum: (A*A)*e0 = A*(A*e0) = A*col0 = A*(1,2)=col0*1+col1*2=(1+6,2+8)=(7,10) sum 17
    const src = "float f() { mat2 a = mat2(1.0, 2.0, 3.0, 4.0); mat2 b = a * a; return dot(b * vec2(1.0, 0.0), vec2(1.0)); }";
    const F0 = *const fn () callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 17.0), try runF32(allocator, src, "f", F0, .{}));
}

test "glsl->spir-v: mat3*vec3 round-trips" {
    const allocator = std.testing.allocator;
    const src = "float f(vec3 v) { mat3 m = mat3(2.0); return dot(m * v, vec3(1.0)); }";
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 12.0), try runSpirvF32(allocator, src, F3, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3) }));
}

// Vectors are scalarized: a vecN parameter becomes N float parameters, so a function
// taking vectors and returning a scalar JITs and runs natively.

test "glsl: dot product of two vec3 (scalarized params)" {
    const allocator = std.testing.allocator;
    // dot([1,2,3],[4,5,6]) = 4 + 10 + 18 = 32
    const src = "float f(vec3 a, vec3 b) { return dot(a, b); }";
    const Fn = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    const r = try runF32(allocator, src, "f", Fn, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 4), @as(f32, 5), @as(f32, 6) });
    try std.testing.expectEqual(@as(f32, 32.0), r);
}

test "glsl: swizzle selects a component" {
    const allocator = std.testing.allocator;
    // f(a) = a.y + a.z   for a = (10, 20, 40) -> 60
    const src = "float f(vec3 a) { return a.y + a.z; }";
    const Fn = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 60.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 10), @as(f32, 20), @as(f32, 40) }));
}

test "glsl: component-wise vector arithmetic then dot" {
    const allocator = std.testing.allocator;
    // f(a,b) = dot(a + b, vec3(1.0))  = sum of (a+b) components
    // a=(1,2,3) b=(4,5,6) -> (5,7,9) -> 21
    const src = "float f(vec3 a, vec3 b) { vec3 c = a + b; return dot(c, vec3(1.0)); }";
    const Fn = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 21.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 4), @as(f32, 5), @as(f32, 6) }));
}

test "glsl: vector times scalar broadcasts" {
    const allocator = std.testing.allocator;
    // f(a, s) = dot(a * s, vec3(1.0))  = (a.x+a.y+a.z) * s
    // a=(1,2,3) s=2 -> 6*2 = 12
    const src = "float f(vec3 a, float s) { return dot(a * s, vec3(1.0)); }";
    const Fn = *const fn (f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 12.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 2) }));
}

test "glsl: swizzle assignment (writing components)" {
    const allocator = std.testing.allocator;
    // build a vec3, write components via swizzle, sum them
    const src = "float f() { vec3 v = vec3(0.0); v.x = 1.0; v.yz = vec2(2.0, 3.0); return dot(v, vec3(1.0)); }";
    const F0 = *const fn () callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, src, "f", F0, .{})); // 1+2+3
}

test "glsl->spir-v: swizzle assignment round-trips" {
    const allocator = std.testing.allocator;
    const src = "float f(vec3 a) { vec3 v = a; v.x = a.z; return v.x; }";
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 7.0), try runSpirvF32(allocator, src, F3, .{ @as(f32, 1), @as(f32, 2), @as(f32, 7) })); // v.x = a.z = 7
}

test "glsl: vec4 from a vec3 and a scalar, swizzled back" {
    const allocator = std.testing.allocator;
    // f(a) = vec4(a, 1.0).w + vec4(a, 1.0).x  for a=(5,6,7) -> 1 + 5 = 6
    const src = "float f(vec3 a) { vec4 p = vec4(a, 1.0); return p.w + p.x; }";
    const Fn = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 5), @as(f32, 6), @as(f32, 7) }));
}

// Early return: one branch returns, the other falls through to the rest of the function.

test "glsl: early return in a branch" {
    const allocator = std.testing.allocator;
    const src = "float f(float x) { if (x < 0.0) { return 0.0; } return x * 2.0; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, src, "f", F1, .{@as(f32, -5.0)}));
    try std.testing.expectEqual(@as(f32, 20.0), try runF32(allocator, src, "f", F1, .{@as(f32, 10.0)}));
}

test "glsl->spir-v: early return round-trips" {
    const allocator = std.testing.allocator;
    const src = "float f(float x) { if (x < 0.0) { return 0.0; } return x * 2.0; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 0.0), try runSpirvF32(allocator, src, F1, .{@as(f32, -5.0)}));
    try std.testing.expectEqual(@as(f32, 20.0), try runSpirvF32(allocator, src, F1, .{@as(f32, 10.0)}));
}

// Nested control flow (if inside loop, nested if): the lowering jumps from each branch's
// actual end block, so composing constructs works on the host JIT.

test "glsl: if inside a for loop (count values above a threshold)" {
    const allocator = std.testing.allocator;
    // count = 0, for i in 1..n: if (i > t) count = count + 1  -> number of i in (t, n]
    const src = "float f(float n, float t) { float count = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { if (i > t) { count = count + 1.0; } } return count; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 4.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 10), @as(f32, 6) })); // 7,8,9,10
    try std.testing.expectEqual(@as(f32, 10.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 10), @as(f32, 0) }));
}

test "glsl: nested if/else" {
    const allocator = std.testing.allocator;
    // sign-like: x<0 -> -1, x>0 -> 1, else 0
    const src = "float f(float x) { float s; if (x < 0.0) { s = -1.0; } else { if (x > 0.0) { s = 1.0; } else { s = 0.0; } } return s; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, -1.0), try runF32(allocator, src, "f", F1, .{@as(f32, -7.0)}));
    try std.testing.expectEqual(@as(f32, 1.0), try runF32(allocator, src, "f", F1, .{@as(f32, 3.0)}));
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, src, "f", F1, .{@as(f32, 0.0)}));
}

test "glsl: nested loops (multiplication via addition)" {
    const allocator = std.testing.allocator;
    // acc = 0, for i in 1..a: for j in 1..b: acc += 1  -> a*b
    const src = "float f(float a, float b) { float acc = 0.0; for (float i = 1.0; i <= a; i = i + 1.0) { for (float j = 1.0; j <= b; j = j + 1.0) { acc = acc + 1.0; } } return acc; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 12.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 3), @as(f32, 4) }));
}

// continue: skip the rest of the body, jump to the continue block (increment + back-edge).

test "glsl: continue skips part of the body" {
    const allocator = std.testing.allocator;
    // sum only the even values of i in 1..n (skip odd i via continue)
    const src = "float f(float n) { float sum = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { float h = i * 0.5; if (h != floor(h)) { continue; } sum = sum + i; } return sum; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, src, "f", F1, .{@as(f32, 4)})); // 2 + 4
    try std.testing.expectEqual(@as(f32, 20.0), try runF32(allocator, src, "f", F1, .{@as(f32, 9)})); // 2+4+6+8
}

test "glsl->spir-v: continue round-trips" {
    const allocator = std.testing.allocator;
    // skip i <= t via continue, sum the rest (no floor() so the SPIR-V reader can decode it)
    const src = "float f(float n, float t) { float sum = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { if (i <= t) { continue; } sum = sum + i; } return sum; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 12.0), try runSpirvF32(allocator, src, F2, .{ @as(f32, 5), @as(f32, 2) })); // 3+4+5
}

// break: jump out of the loop to its exit (the exit gets phis merging the break edges).

test "glsl: break out of a loop" {
    const allocator = std.testing.allocator;
    // accumulate triangular sums, break when the sum exceeds a limit
    const src = "float f(float n, float limit) { float sum = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { sum = sum + i; if (sum > limit) { break; } } return sum; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 100), @as(f32, 5) })); // 1,3,6 -> break
    try std.testing.expectEqual(@as(f32, 10.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 4), @as(f32, 100) })); // 1,3,6,10, no break
}

test "glsl->spir-v: break round-trips" {
    const allocator = std.testing.allocator;
    const src = "float f(float n, float limit) { float sum = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { sum = sum + i; if (sum > limit) { break; } } return sum; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runSpirvF32(allocator, src, F2, .{ @as(f32, 100), @as(f32, 5) }));
}

// for/while lower to a header/body/exit loop with loop-phi block params, host JIT.

test "glsl: for loop sums 1..n" {
    const allocator = std.testing.allocator;
    // sum = 0, for (i = 1, i <= n, i = i + 1) sum = sum + i  -> n(n+1)/2
    const src = "float f(float n) { float sum = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { sum = sum + i; } return sum; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 55.0), try runF32(allocator, src, "f", F1, .{@as(f32, 10.0)})); // 1+..+10
    try std.testing.expectEqual(@as(f32, 15.0), try runF32(allocator, src, "f", F1, .{@as(f32, 5.0)}));
}

test "glsl: while loop (repeated doubling)" {
    const allocator = std.testing.allocator;
    // count how the value grows: x starts at a, doubles while < b, return x
    const src = "float f(float a, float b) { float x = a; while (x < b) { x = x * 2.0; } return x; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 16.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 1), @as(f32, 10) })); // 1,2,4,8,16
    try std.testing.expectEqual(@as(f32, 12.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 3), @as(f32, 10) })); // 3,6,12
}

test "glsl: loop with a loop-invariant variable" {
    const allocator = std.testing.allocator;
    // k is never modified in the loop, acc accumulates k each iteration
    const src = "float f(float n) { float k = 2.0; float acc = 0.0; for (float i = 0.0; i < n; i = i + 1.0) { acc = acc + k; } return acc; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 8.0), try runF32(allocator, src, "f", F1, .{@as(f32, 4.0)})); // 2*4
}

// Operators: logical && || !, bool literals, compound assignment, ++/--.

test "glsl: logical operators and bool" {
    const allocator = std.testing.allocator;
    // in range [lo, hi]?  (a >= lo && a <= hi)  returned as int 0/1
    const src = "int f(float a, float lo, float hi) { return (a >= lo && a <= hi) ? 1 : 0; }";
    const F3 = *const fn (f32, f32, f32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, src, "f", F3, .{ @as(f32, 5), @as(f32, 0), @as(f32, 10) }));
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, src, "f", F3, .{ @as(f32, 15), @as(f32, 0), @as(f32, 10) }));
}

test "glsl: logical not and or" {
    const allocator = std.testing.allocator;
    // !(a < b) || (a == 0)
    const src = "int f(float a, float b) { return (!(a < b) || a == 0.0) ? 1 : 0; }";
    const F2 = *const fn (f32, f32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, src, "f", F2, .{ @as(f32, 5), @as(f32, 2) })); // !(5<2)=true
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, src, "f", F2, .{ @as(f32, 1), @as(f32, 2) })); // !(1<2)=false, 1!=0
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, src, "f", F2, .{ @as(f32, 0), @as(f32, 2) })); // a==0
}

test "glsl: compound assignment and increment" {
    const allocator = std.testing.allocator;
    // sum += i in a loop using i++
    const src = "float f(float n) { float sum = 0.0; for (float i = 1.0; i <= n; i++) { sum += i; } return sum; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 55.0), try runF32(allocator, src, "f", F1, .{@as(f32, 10.0)}));
    // *= and -=
    const src2 = "float f(float x) { float y = x; y *= 3.0; y -= 2.0; return y; }";
    try std.testing.expectEqual(@as(f32, 13.0), try runF32(allocator, src2, "f", F1, .{@as(f32, 5.0)})); // 5*3-2
}

test "glsl->spir-v: logical operators round-trip" {
    const allocator = std.testing.allocator;
    const src = "int f(float a, float b) { return (a > 0.0 && b > 0.0) ? 1 : 0; }";
    const F2 = *const fn (f32, f32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 1), try runSpirvI32(allocator, src, F2, .{ @as(f32, 1), @as(f32, 2) }));
    try std.testing.expectEqual(@as(i32, 0), try runSpirvI32(allocator, src, F2, .{ @as(f32, 1), @as(f32, -2) }));
}

// if/else lowers to a diamond CFG with block-param (phi) merges, runs on the host JIT.

test "glsl: if/else selects the larger (max)" {
    const allocator = std.testing.allocator;
    const src = "float f(float a, float b) { float m; if (a < b) { m = b; } else { m = a; } return m; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 5.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 2), @as(f32, 5) }));
    try std.testing.expectEqual(@as(f32, 5.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 5), @as(f32, 2) }));
}

test "glsl: if without else (abs via unary negate)" {
    const allocator = std.testing.allocator;
    const src = "float f(float a) { float r = a; if (a < 0.0) { r = -a; } return r; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 3.0), try runF32(allocator, src, "f", F1, .{@as(f32, -3.0)}));
    try std.testing.expectEqual(@as(f32, 5.0), try runF32(allocator, src, "f", F1, .{@as(f32, 5.0)}));
}

test "glsl: single-statement branches (no braces)" {
    const allocator = std.testing.allocator;
    const src = "float f(float a, float b) { float m; if (a < b) m = b; else m = a; return m; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 9.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 9), @as(f32, 4) }));
}

test "glsl: a variable left unchanged in branches needs no phi" {
    const allocator = std.testing.allocator;
    // k is set once before the if and never reassigned. m is chosen by the branch.
    const src = "float f(float a, float b) { float k = 100.0; float m = a; if (a < b) { m = b; } return m + k; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 105.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 2), @as(f32, 5) })); // 5 + 100
    try std.testing.expectEqual(@as(f32, 109.0), try runF32(allocator, src, "f", F2, .{ @as(f32, 9), @as(f32, 1) })); // 9 + 100
}

// Builtins that lower to select/arithmetic (min/max/clamp/abs/mix/cross): no new IR ops,
// so they run on the host JIT and emit valid SPIR-V.

test "glsl: min/max/clamp" {
    const allocator = std.testing.allocator;
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 2.0), try runF32(allocator, "float f(float a, float b){return min(a,b);}", "f", F2, .{ @as(f32, 2), @as(f32, 5) }));
    try std.testing.expectEqual(@as(f32, 5.0), try runF32(allocator, "float f(float a, float b){return max(a,b);}", "f", F2, .{ @as(f32, 2), @as(f32, 5) }));
    const F1 = *const fn (f32) callconv(.c) f32;
    const clampSrc = "float f(float x){return clamp(x, 0.0, 1.0);}";
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, clampSrc, "f", F1, .{@as(f32, -2.0)}));
    try std.testing.expectEqual(@as(f32, 0.5), try runF32(allocator, clampSrc, "f", F1, .{@as(f32, 0.5)}));
    try std.testing.expectEqual(@as(f32, 1.0), try runF32(allocator, clampSrc, "f", F1, .{@as(f32, 9.0)}));
}

test "glsl: abs and mix" {
    const allocator = std.testing.allocator;
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 3.0), try runF32(allocator, "float f(float x){return abs(x);}", "f", F1, .{@as(f32, -3.0)}));
    // mix(a,b,t) = a + (b-a)*t, mix(10, 20, 0.25) = 12.5
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 12.5), try runF32(allocator, "float f(float a, float b, float t){return mix(a,b,t);}", "f", F3, .{ @as(f32, 10), @as(f32, 20), @as(f32, 0.25) }));
}

test "glsl: vector min and cross" {
    const allocator = std.testing.allocator;
    // min component-wise: min((1,5,3),(4,2,6)) = (1,2,3), dot with ones = 6
    const F6 = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, "float f(vec3 a, vec3 b){return dot(min(a,b), vec3(1.0));}", "f", F6, .{ @as(f32, 1), @as(f32, 5), @as(f32, 3), @as(f32, 4), @as(f32, 2), @as(f32, 6) }));
    // cross((1,0,0),(0,1,0)) = (0,0,1), dot with ones = 1
    try std.testing.expectEqual(@as(f32, 1.0), try runF32(allocator, "float f(vec3 a, vec3 b){return dot(cross(a,b), vec3(1.0));}", "f", F6, .{ @as(f32, 1), @as(f32, 0), @as(f32, 0), @as(f32, 0), @as(f32, 1), @as(f32, 0) }));
}

// sqrt/length/normalize use the IR `unary` (sqrt) -> aarch64 fsqrt on the host, and
// OpExtInst GLSL.std.450 Sqrt in SPIR-V.

test "glsl: sqrt, length, normalize" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(f32, 4.0), try runF32(allocator, "float f(float x){return sqrt(x);}", "f", *const fn (f32) callconv(.c) f32, .{@as(f32, 16.0)}));
    // length((3,4,0)) = 5
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 5.0), try runF32(allocator, "float f(vec3 a){return length(a);}", "f", F3, .{ @as(f32, 3), @as(f32, 4), @as(f32, 0) }));
    // a normalized vector has length 1
    const n = try runF32(allocator, "float f(vec3 a){return length(normalize(a));}", "f", F3, .{ @as(f32, 3), @as(f32, 4), @as(f32, 12) });
    try std.testing.expect(@abs(n - 1.0) < 0.0001);
}

// More builtins: floor/ceil/fract/sign/step/mod/radians/degrees/smoothstep (existing IR).

test "glsl: floor/ceil/fract/sign" {
    const allocator = std.testing.allocator;
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 3.0), try runF32(allocator, "float f(float x){return floor(x);}", "f", F1, .{@as(f32, 3.7)}));
    try std.testing.expectEqual(@as(f32, 4.0), try runF32(allocator, "float f(float x){return ceil(x);}", "f", F1, .{@as(f32, 3.2)}));
    const fr = try runF32(allocator, "float f(float x){return fract(x);}", "f", F1, .{@as(f32, 3.25)});
    try std.testing.expect(@abs(fr - 0.25) < 0.0001);
    try std.testing.expectEqual(@as(f32, -1.0), try runF32(allocator, "float f(float x){return sign(x);}", "f", F1, .{@as(f32, -9.0)}));
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, "float f(float x){return sign(x);}", "f", F1, .{@as(f32, 0.0)}));
}

test "glsl: step/mod/smoothstep" {
    const allocator = std.testing.allocator;
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    // step(edge, x): 0 if x<edge else 1
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, "float f(float e, float x){return step(e, x);}", "f", F2, .{ @as(f32, 5), @as(f32, 2) }));
    try std.testing.expectEqual(@as(f32, 1.0), try runF32(allocator, "float f(float e, float x){return step(e, x);}", "f", F2, .{ @as(f32, 5), @as(f32, 9) }));
    // mod(7, 3) = 1
    const m = try runF32(allocator, "float f(float x, float y){return mod(x, y);}", "f", F2, .{ @as(f32, 7), @as(f32, 3) });
    try std.testing.expect(@abs(m - 1.0) < 0.0001);
    // smoothstep(0, 10, 5) = 0.5 (midpoint)
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    const s = try runF32(allocator, "float f(float a, float b, float x){return smoothstep(a, b, x);}", "f", F3, .{ @as(f32, 0), @as(f32, 10), @as(f32, 5) });
    try std.testing.expect(@abs(s - 0.5) < 0.0001);
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, "float f(float a, float b, float x){return smoothstep(a, b, x);}", "f", F3, .{ @as(f32, 0), @as(f32, 10), @as(f32, -5) }));
}

test "glsl->spir-v: builtins round-trip (step + sign)" {
    const allocator = std.testing.allocator;
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 1.0), try runSpirvF32(allocator, "float f(float e, float x){return step(e, x);}", F2, .{ @as(f32, 1), @as(f32, 5) }));
}

// The ternary `cond ? a : b` lowers to the IR `select` (no branches), so it runs on the
// host JIT and emits valid SPIR-V.

test "glsl: ternary selects a scalar (min via ?:)" {
    const allocator = std.testing.allocator;
    const src = "float f(float a, float b) { return a < b ? a : b; }";
    const Fn = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 2.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 2), @as(f32, 5) }));
    try std.testing.expectEqual(@as(f32, 2.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 5), @as(f32, 2) }));
}

test "glsl: nested ternary (clamp-ish)" {
    const allocator = std.testing.allocator;
    // f(x) = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x)
    const src = "float f(float x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }";
    const Fn = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, src, "f", Fn, .{@as(f32, -3.0)}));
    try std.testing.expectEqual(@as(f32, 0.5), try runF32(allocator, src, "f", Fn, .{@as(f32, 0.5)}));
    try std.testing.expectEqual(@as(f32, 1.0), try runF32(allocator, src, "f", Fn, .{@as(f32, 7.0)}));
}

test "glsl: ternary picks a vector lane via dot" {
    const allocator = std.testing.allocator;
    // f(a, b, s) = dot(s < 1.0 ? a : b, vec3(1.0))   sum of a or b's components
    // a=(1,2,3) sum 6, b=(10,20,30) sum 60
    const src = "float f(vec3 a, vec3 b, float s) { return dot(s < 1.0 ? a : b, vec3(1.0)); }";
    const Fn = *const fn (f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 10), @as(f32, 20), @as(f32, 30), @as(f32, 0.0) }));
    try std.testing.expectEqual(@as(f32, 60.0), try runF32(allocator, src, "f", Fn, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 10), @as(f32, 20), @as(f32, 30), @as(f32, 5.0) }));
}

// GLSL -> SPIR-V: emit a SPIR-V binary, read it back with the SPIR-V frontend, JIT the
// recovered IR, and run it. Closes the loop GLSL -> IR -> SPIR-V -> IR -> machine code.

fn runSpirvF32(allocator: std.mem.Allocator, src: []const u8, comptime Fn: type, arg: anytype) !f32 {
    const words = try glsl.compileToSpirv(allocator, src);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 0x07230203), words[0]); // valid SPIR-V magic
    var func = try spirv.lowerModule(allocator, words);
    defer func.deinit();
    var buf = try native.jitFunction(allocator, &func);
    defer buf.deinit();
    return @call(.auto, buf.entry(Fn, 0), arg);
}

fn runSpirvI32(allocator: std.mem.Allocator, src: []const u8, comptime Fn: type, arg: anytype) !i32 {
    const words = try glsl.compileToSpirv(allocator, src);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 0x07230203), words[0]);
    var func = try spirv.lowerModule(allocator, words);
    defer func.deinit();
    var buf = try native.jitFunction(allocator, &func);
    defer buf.deinit();
    return @call(.auto, buf.entry(Fn, 0), arg);
}

test "glsl->spir-v: float arithmetic round-trips and runs" {
    const allocator = std.testing.allocator;
    const r = try runSpirvF32(allocator, "float f(float x) { return x * 2.0 + 1.0; }", *const fn (f32) callconv(.c) f32, .{@as(f32, 20.0)});
    try std.testing.expectEqual(@as(f32, 41.0), r);
}

test "glsl->spir-v: int div/rem round-trips" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(i32, 6), try runSpirvI32(allocator, "int f(int a, int b) { return a / b; }", *const fn (i32, i32) callconv(.c) i32, .{ @as(i32, 20), @as(i32, 3) }));
    try std.testing.expectEqual(@as(i32, 2), try runSpirvI32(allocator, "int f(int a, int b) { return a % b; }", *const fn (i32, i32) callconv(.c) i32, .{ @as(i32, 20), @as(i32, 3) }));
}

test "glsl->spir-v: int->float promotion and conversion round-trip" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(f32, 41.0), try runSpirvF32(allocator, "float f(float x) { return x * 2 + 1; }", *const fn (f32) callconv(.c) f32, .{@as(f32, 20.0)}));
    try std.testing.expectEqual(@as(i32, 3), try runSpirvI32(allocator, "int f(float x) { return int(x); }", *const fn (f32) callconv(.c) i32, .{@as(f32, 3.7)}));
}

test "glsl->spir-v: ternary (select) round-trips and runs as min" {
    const allocator = std.testing.allocator;
    const src = "float f(float a, float b) { return a < b ? a : b; }";
    const Fn = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 2.0), try runSpirvF32(allocator, src, Fn, .{ @as(f32, 2), @as(f32, 5) }));
    try std.testing.expectEqual(@as(f32, 3.0), try runSpirvF32(allocator, src, Fn, .{ @as(f32, 7), @as(f32, 3) }));
}

test "glsl->spir-v: if inside a loop round-trips (nested control flow)" {
    const allocator = std.testing.allocator;
    const src = "float f(float n, float t) { float count = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { if (i > t) { count = count + 1.0; } } return count; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 4.0), try runSpirvF32(allocator, src, F2, .{ @as(f32, 10), @as(f32, 6) }));
}

test "glsl->spir-v: nested if/else round-trips" {
    const allocator = std.testing.allocator;
    const src = "float f(float x) { float s; if (x < 0.0) { s = -1.0; } else { if (x > 0.0) { s = 1.0; } else { s = 0.0; } } return s; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, -1.0), try runSpirvF32(allocator, src, F1, .{@as(f32, -7.0)}));
    try std.testing.expectEqual(@as(f32, 1.0), try runSpirvF32(allocator, src, F1, .{@as(f32, 3.0)}));
    try std.testing.expectEqual(@as(f32, 0.0), try runSpirvF32(allocator, src, F1, .{@as(f32, 0.0)}));
}

test "glsl->spir-v: nested loops round-trip" {
    const allocator = std.testing.allocator;
    const src = "float f(float a, float b) { float acc = 0.0; for (float i = 1.0; i <= a; i = i + 1.0) { for (float j = 1.0; j <= b; j = j + 1.0) { acc = acc + 1.0; } } return acc; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 12.0), try runSpirvF32(allocator, src, F2, .{ @as(f32, 3), @as(f32, 4) }));
}

test "glsl->spir-v: for loop round-trips through OpLoopMerge" {
    const allocator = std.testing.allocator;
    // sum 1..n via a loop: emits OpLoopMerge + OpPhi (back-edge), reads back, runs.
    const src = "float f(float n) { float sum = 0.0; for (float i = 1.0; i <= n; i = i + 1.0) { sum = sum + i; } return sum; }";
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 55.0), try runSpirvF32(allocator, src, F1, .{@as(f32, 10.0)}));
    try std.testing.expectEqual(@as(f32, 15.0), try runSpirvF32(allocator, src, F1, .{@as(f32, 5.0)}));
}

test "glsl->spir-v: if/else (max) round-trips through multi-block emission" {
    const allocator = std.testing.allocator;
    // A diamond CFG emits OpSelectionMerge + OpBranchConditional + OpPhi. The SPIR-V
    // frontend reads it back, and the JITed result is max(a, b).
    const src = "float f(float a, float b) { float m; if (a < b) { m = b; } else { m = a; } return m; }";
    const F2 = *const fn (f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 5.0), try runSpirvF32(allocator, src, F2, .{ @as(f32, 2), @as(f32, 5) }));
    try std.testing.expectEqual(@as(f32, 7.0), try runSpirvF32(allocator, src, F2, .{ @as(f32, 7), @as(f32, 3) }));
}

test "glsl->spir-v: scalarized vector dot round-trips and runs" {
    const allocator = std.testing.allocator;
    // A vec3 dot product emits a 6-float SPIR-V function (vectors are scalarized).
    const src = "float f(vec3 a, vec3 b) { return dot(a, b); }";
    const Fn = *const fn (f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    const r = try runSpirvF32(allocator, src, Fn, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 4), @as(f32, 5), @as(f32, 6) });
    try std.testing.expectEqual(@as(f32, 32.0), r);
}

test "glsl->spir-v: comparison (compare+select) round-trips to 0/1" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(i32, 1), try runSpirvI32(allocator, "int f(float a, float b) { return a < b; }", *const fn (f32, f32) callconv(.c) i32, .{ @as(f32, 2.0), @as(f32, 5.0) }));
    try std.testing.expectEqual(@as(i32, 0), try runSpirvI32(allocator, "int f(float a, float b) { return a < b; }", *const fn (f32, f32) callconv(.c) i32, .{ @as(f32, 5.0), @as(f32, 2.0) }));
}

// GLSL shader -> SPIR-V entry point: a fragment shader with scalar in/out variables.
// Validated structurally here (the entry point, execution mode, and interface vars),
// spirv-val conformance is confirmed out of band.

test "glsl->spir-v: fragment shader emits an entry point with in/out vars" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in float x;
        \\layout(location = 0) out float color;
        \\void main() { color = x * 2.0 + 1.0; }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 0x07230203), words[0]);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_fragment_entry = false;
    var inputs: u32 = 0;
    var outputs: u32 = 0;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.EntryPoint => saw_fragment_entry = (inst.operands[0] == op.ExecutionModel.fragment),
        op.Variable => switch (inst.operands[2]) {
            op.StorageClass.input => inputs += 1,
            op.StorageClass.output => outputs += 1,
            else => {},
        },
        else => {},
    };
    try std.testing.expect(saw_fragment_entry);
    try std.testing.expectEqual(@as(u32, 1), inputs);
    try std.testing.expectEqual(@as(u32, 1), outputs);
}

test "glsl->spir-v: fragment shader with vec3 in and vec4 out (native vectors)" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in vec3 a;
        \\layout(location = 0) out vec4 color;
        \\void main() { color = vec4(a * 2.0, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_vec3 = false;
    var saw_vec4 = false;
    var extracts: u32 = 0;
    var constructs: u32 = 0;
    var inputs: u32 = 0;
    var outputs: u32 = 0;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.TypeVector => {
            if (inst.operands[2] == 3) saw_vec3 = true;
            if (inst.operands[2] == 4) saw_vec4 = true;
        },
        op.CompositeExtract => extracts += 1,
        op.CompositeConstruct => constructs += 1,
        op.Variable => switch (inst.operands[2]) {
            op.StorageClass.input => inputs += 1,
            op.StorageClass.output => outputs += 1,
            else => {},
        },
        else => {},
    };
    try std.testing.expect(saw_vec3 and saw_vec4); // native OpTypeVector interface
    try std.testing.expectEqual(@as(u32, 3), extracts); // the vec3 input's lanes
    try std.testing.expect(constructs >= 1); // the vec4 output
    try std.testing.expectEqual(@as(u32, 1), inputs);
    try std.testing.expectEqual(@as(u32, 1), outputs);
}

test "glsl->spir-v: fragment derivatives emit OpDPdx/OpDPdy/OpFwidth" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in float v;
        \\layout(location = 0) out vec4 color;
        \\void main() { float e = abs(dFdx(v)) + abs(dFdy(v)) + fwidth(v); color = vec4(e, e, e, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var dpdx: u32 = 0;
    var dpdy: u32 = 0;
    var fwidth: u32 = 0;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.DPdx => dpdx += 1,
        op.DPdy => dpdy += 1,
        op.Fwidth => fwidth += 1,
        else => {},
    };
    try std.testing.expect(dpdx == 1 and dpdy == 1 and fwidth == 1);
}

test "glsl->spir-v: sampler2D + texture() emits OpImageSampleImplicitLod" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in vec2 uv;
        \\layout(location = 0) out vec4 color;
        \\uniform sampler2D tex;
        \\void main() { color = texture(tex, uv) * vec4(0.5, 0.5, 0.5, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var images: u32 = 0;
    var sampled: u32 = 0;
    var samples: u32 = 0;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.TypeImage => images += 1,
        op.TypeSampledImage => sampled += 1,
        op.ImageSampleImplicitLod => samples += 1,
        else => {},
    };
    try std.testing.expect(images == 1 and sampled == 1 and samples == 1);
}

test "glsl->spir-v: discard emits OpKill" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in float a;
        \\layout(location = 0) out vec4 color;
        \\void main() { if (a < 0.5) { discard; } color = vec4(a, a, a, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var kills: u32 = 0;
    while (try reader.next()) |inst| if (inst.opcode == op.Kill) {
        kills += 1;
    };
    try std.testing.expect(kills == 1);
}

test "glsl->spir-v: transcendentals emit OpExtInst (sin/cos/pow)" {
    const allocator = std.testing.allocator;
    // A fragment shader using sin/cos/pow, emits GLSL.std.450 ext instructions.
    const src =
        \\layout(location = 0) in float t;
        \\layout(location = 0) out vec4 color;
        \\void main() { float a = sin(t) * cos(t) + pow(t, 2.0); color = vec4(a, a, a, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var ext_insts: u32 = 0;
    var saw_import = false;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.ExtInst => ext_insts += 1,
        op.ExtInstImport => saw_import = true,
        else => {},
    };
    try std.testing.expect(saw_import);
    try std.testing.expectEqual(@as(u32, 3), ext_insts); // sin, cos, pow
}

test "glsl->spir-v: vertex shader with a uniform mat4 (MVP transform)" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in vec3 pos;
        \\uniform mat4 mvp;
        \\void main() { gl_Position = mvp * vec4(pos, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .vertex);
    defer allocator.free(words);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_pushconstant = false;
    var saw_block = false;
    var access_chains: u32 = 0;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.Variable => if (inst.operands[2] == op.StorageClass.push_constant) {
            saw_pushconstant = true;
        },
        op.Decorate => if (inst.operands[1] == op.Decoration.block) {
            saw_block = true;
        },
        op.AccessChain => access_chains += 1,
        else => {},
    };
    try std.testing.expect(saw_pushconstant and saw_block);
    try std.testing.expectEqual(@as(u32, 16), access_chains); // 16 mat4 floats loaded
}

test "glsl->spir-v: fragment shader with a uniform float" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in float x;
        \\layout(location = 0) out vec4 color;
        \\uniform float scale;
        \\void main() { float c = x * scale; color = vec4(c, c, c, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_pc = false;
    while (try reader.next()) |inst| if (inst.opcode == op.Variable and inst.operands[2] == op.StorageClass.push_constant) {
        saw_pc = true;
    };
    try std.testing.expect(saw_pc);
}

test "glsl->spir-v: fragment shader with if/else (multi-block entry point)" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in float x;
        \\layout(location = 0) out vec4 color;
        \\void main() { float c; if (x < 0.5) { c = 0.0; } else { c = 1.0; } color = vec4(c, c, c, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_entry = false;
    var saw_merge = false;
    var saw_cond_branch = false;
    var saw_phi = false;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.EntryPoint => saw_entry = true,
        op.SelectionMerge => saw_merge = true,
        op.BranchConditional => saw_cond_branch = true,
        op.Phi => saw_phi = true,
        else => {},
    };
    try std.testing.expect(saw_entry and saw_merge and saw_cond_branch and saw_phi);
}

test "glsl->spir-v: vertex shader writes gl_Position (BuiltIn Position)" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in vec3 pos;
        \\void main() { gl_Position = vec4(pos, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .vertex);
    defer allocator.free(words);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_vertex = false;
    var saw_builtin_position = false;
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.EntryPoint => saw_vertex = (inst.operands[0] == op.ExecutionModel.vertex),
        op.Decorate => if (inst.operands[1] == op.Decoration.builtin and inst.operands[2] == op.BuiltIn.position) {
            saw_builtin_position = true;
        },
        else => {},
    };
    try std.testing.expect(saw_vertex);
    try std.testing.expect(saw_builtin_position);
}

test "glsl->spir-v: vertex shader writes gl_PointSize (BuiltIn PointSize)" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) in vec3 pos;
        \\void main() { gl_Position = vec4(pos, 1.0); gl_PointSize = 4.0; }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .vertex);
    defer allocator.free(words);
    // Both the Position and the PointSize builtins are decorated (each exactly once).
    try std.testing.expectEqual(@as(u32, 1), try countBuiltinDecorations(words, spirv.opcodes.BuiltIn.position));
    try std.testing.expectEqual(@as(u32, 1), try countBuiltinDecorations(words, spirv.opcodes.BuiltIn.point_size));
}

fn countBuiltinDecorations(words: []const u32, builtin: u32) !u32 {
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var n: u32 = 0;
    while (try reader.next()) |inst| {
        if (inst.opcode == op.Decorate and inst.operands[1] == op.Decoration.builtin and inst.operands[2] == builtin) n += 1;
    }
    return n;
}

test "glsl->spir-v: fragment shader reads gl_FragCoord (BuiltIn FragCoord input)" {
    const allocator = std.testing.allocator;
    const src =
        \\layout(location = 0) out vec4 color;
        \\void main() { color = gl_FragCoord; }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 1), try countBuiltinDecorations(words, spirv.opcodes.BuiltIn.frag_coord));
}

test "glsl->spir-v: vertex shader reads gl_VertexIndex (BuiltIn VertexIndex input)" {
    const allocator = std.testing.allocator;
    const src =
        \\void main() { gl_Position = vec4(float(gl_VertexIndex), 0.0, 0.0, 1.0); }
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .vertex);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 1), try countBuiltinDecorations(words, spirv.opcodes.BuiltIn.vertex_index));
    try std.testing.expectEqual(@as(u32, 1), try countBuiltinDecorations(words, spirv.opcodes.BuiltIn.position));
}

// GLSL ES 1.00 (the GLES2 shading language): `attribute`/`varying`/`gl_FragColor`,
// precision statements, `texture2D`. The gradient-triangle vertical slice: the VS+FS
// the EGL/GLES path renders, compiled from GLSL ES 1.00 SOURCE (not a baked binary).

fn countInputsOutputs(words: []const u32) !struct { inputs: u32, outputs: u32 } {
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var inputs: u32 = 0;
    var outputs: u32 = 0;
    while (try reader.next()) |inst| if (inst.opcode == op.Variable) switch (inst.operands[2]) {
        op.StorageClass.input => inputs += 1,
        op.StorageClass.output => outputs += 1,
        else => {},
    };
    return .{ .inputs = inputs, .outputs = outputs };
}

fn inputLocations(allocator: std.mem.Allocator, words: []const u32) ![]u32 {
    const op = spirv.opcodes;
    // var id -> its Location decoration. Collect input vars in order, then map locations.
    var loc_of = std.AutoHashMap(u32, u32).init(allocator);
    defer loc_of.deinit();
    var input_vars: std.ArrayList(u32) = .empty;
    defer input_vars.deinit(allocator);
    var reader = try spirv.binary.Reader.init(words);
    while (try reader.next()) |inst| switch (inst.opcode) {
        op.Decorate => if (inst.operands[1] == op.Decoration.location) try loc_of.put(inst.operands[0], inst.operands[2]),
        op.Variable => if (inst.operands[2] == op.StorageClass.input) try input_vars.append(allocator, inst.operands[1]),
        else => {},
    };
    var out = try allocator.alloc(u32, input_vars.items.len);
    for (input_vars.items, 0..) |id, i| out[i] = loc_of.get(id) orelse std.math.maxInt(u32);
    return out;
}

test "glsl-es 1.00: gradient-triangle VERTEX shader (attribute/varying/gl_Position) emits a vertex entry + Position + two located inputs" {
    const allocator = std.testing.allocator;
    // The EGL gradient triangle's VS, GLSL ES 1.00. aPos at location 0, aColor at 1
    // (declaration order), vColor varying -> an Output, gl_Position -> BuiltIn Position.
    const src =
        \\#version 100
        \\attribute vec2 aPos;
        \\attribute vec3 aColor;
        \\varying vec3 vColor;
        \\void main() {
        \\    gl_Position = vec4(aPos, 0.0, 1.0);
        \\    vColor = aColor;
        \\}
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .vertex);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 0x07230203), words[0]);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_vertex = false;
    while (try reader.next()) |inst| if (inst.opcode == op.EntryPoint) {
        saw_vertex = (inst.operands[0] == op.ExecutionModel.vertex);
    };
    try std.testing.expect(saw_vertex);
    // gl_Position is the BuiltIn Position output.
    try std.testing.expectEqual(@as(u32, 1), try countBuiltinDecorations(words, op.BuiltIn.position));
    // Two attributes IN (aPos, aColor) + two outputs (gl_Position builtin + vColor varying).
    const io = try countInputsOutputs(words);
    try std.testing.expectEqual(@as(u32, 2), io.inputs);
    try std.testing.expectEqual(@as(u32, 2), io.outputs);
    // Locations are assigned in declaration order: aPos=0, aColor=1.
    const locs = try inputLocations(allocator, words);
    defer allocator.free(locs);
    try std.testing.expectEqual(@as(usize, 2), locs.len);
    try std.testing.expectEqual(@as(u32, 0), locs[0]);
    try std.testing.expectEqual(@as(u32, 1), locs[1]);
}

test "glsl-es 1.00: gradient-triangle FRAGMENT shader (precision/varying/gl_FragColor) emits a fragment entry + one in + one out" {
    const allocator = std.testing.allocator;
    // The EGL gradient triangle's FS, GLSL ES 1.00. `precision mediump float;` is parsed
    // and ignored. vColor varying -> an Input, gl_FragColor -> the color Output.
    const src =
        \\#version 100
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() {
        \\    gl_FragColor = vec4(vColor, 1.0);
        \\}
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 0x07230203), words[0]);

    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var saw_fragment = false;
    while (try reader.next()) |inst| if (inst.opcode == op.EntryPoint) {
        saw_fragment = (inst.operands[0] == op.ExecutionModel.fragment);
    };
    try std.testing.expect(saw_fragment);
    // vColor IN, gl_FragColor OUT.
    const io = try countInputsOutputs(words);
    try std.testing.expectEqual(@as(u32, 1), io.inputs);
    try std.testing.expectEqual(@as(u32, 1), io.outputs);
}

test "glsl-es 1.00: the gradient-triangle SPIR-V round-trips through the Vulcan reader (vertex+fragment)" {
    const allocator = std.testing.allocator;
    // Both stages compile, are valid SPIR-V, and lower back to Vulcan IR (the same path the
    // EGL/GLES software driver drives at draw time). This is the structural gate for render.
    const vs_src =
        \\attribute vec2 aPos;
        \\attribute vec3 aColor;
        \\varying vec3 vColor;
        \\void main() { gl_Position = vec4(aPos, 0.0, 1.0); vColor = aColor; }
    ;
    const fs_src =
        \\precision mediump float;
        \\varying vec3 vColor;
        \\void main() { gl_FragColor = vec4(vColor, 1.0); }
    ;
    const vs = try glsl.compileShaderToSpirv(allocator, vs_src, .vertex);
    defer allocator.free(vs);
    const fs = try glsl.compileShaderToSpirv(allocator, fs_src, .fragment);
    defer allocator.free(fs);
    var vf = try spirv.lowerModule(allocator, vs);
    vf.deinit();
    var ff = try spirv.lowerModule(allocator, fs);
    ff.deinit();
}

test "glsl-es 1.00: precision qualifiers on declarations + texture2D parse and compile" {
    const allocator = std.testing.allocator;
    // mediump on a local + a uniform, texture2D() spelling, the `precision` statement.
    const src =
        \\precision highp float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUv;
        \\void main() {
        \\    mediump vec4 c = texture2D(uTex, vUv);
        \\    gl_FragColor = c;
        \\}
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    const op = spirv.opcodes;
    var reader = try spirv.binary.Reader.init(words);
    var samples: u32 = 0;
    while (try reader.next()) |inst| if (inst.opcode == op.ImageSampleImplicitLod) {
        samples += 1;
    };
    try std.testing.expectEqual(@as(u32, 1), samples);
}

// glmark2 ideas/terrain frontend features: a user function with CONDITIONAL EARLY
// RETURNS (the `unitvec` pattern), mat3 from three vec3 columns, constant-bound loop
// unrolling with array indexing, user structs + arrays of structs, and uniform arrays.

// A user function whose body has multiple early returns inside `if`s, inlined at the call
// site. Each early return must short-circuit the rest of the inlined body (the glmark2
// `unitvec(vec4,vec4)` shape: return on the .w == 0 cases, else the perspective divide).
test "glsl: inlined user function with conditional early returns short-circuits" {
    const allocator = std.testing.allocator;
    // classify(x): x<0 -> -1, x==0 -> 0, x>0 -> +1, via early returns inside ifs.
    const src =
        \\float classify(float x) {
        \\    if (x < 0.0) return -1.0;
        \\    if (x == 0.0) return 0.0;
        \\    return 1.0;
        \\}
        \\float f(float x) { return classify(x) * 10.0; }
    ;
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, -10.0), try runF32(allocator, src, "f", F1, .{@as(f32, -3.0)}));
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, src, "f", F1, .{@as(f32, 0.0)}));
    try std.testing.expectEqual(@as(f32, 10.0), try runF32(allocator, src, "f", F1, .{@as(f32, 7.0)}));
}

// The `unitvec` early-return shape from ideas-logo: the && compound condition + three
// conditional early returns + a tail expression. The helper returns a float here (the
// `compile` path lowers every function standalone, and only scalar-returning functions
// lower standalone), but the inlined call in `f` exercises the exact short-circuit logic
// on scalarized vec4s. The full vec3-returning `unitvec` is exercised end-to-end by the
// live glmark2 ideas scene + the module-scope struct SPIR-V test below.
test "glsl: unitvec-shaped early returns (the ideas-logo helper) compute per path" {
    const allocator = std.testing.allocator;
    const src =
        \\float pick(vec4 v1, vec4 v2) {
        \\    if (v1.w == 0.0 && v2.w == 0.0) return (v2 - v1).x;
        \\    if (v1.w == 0.0) return -v1.x;
        \\    if (v2.w == 0.0) return v2.x;
        \\    return v2.x/v2.w - v1.x/v1.w;
        \\}
        \\float f(vec4 a, vec4 b) { return pick(a, b); }
    ;
    const F8 = *const fn (f32, f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    // Path 4 (both w != 0): b.x/b.w - a.x/a.w. a=(2,_,_,2) b=(3,_,_,3): 1 - 1 = 0.
    try std.testing.expectEqual(@as(f32, 0.0), try runF32(allocator, src, "f", F8, .{ @as(f32, 2), @as(f32, 4), @as(f32, 6), @as(f32, 2), @as(f32, 3), @as(f32, 6), @as(f32, 9), @as(f32, 3) }));
    // Path 2 (v1.w == 0, v2.w != 0): -v1.x = -1. a=(1,2,3,0) b=(0,0,0,5).
    try std.testing.expectEqual(@as(f32, -1.0), try runF32(allocator, src, "f", F8, .{ @as(f32, 1), @as(f32, 2), @as(f32, 3), @as(f32, 0), @as(f32, 0), @as(f32, 0), @as(f32, 0), @as(f32, 5) }));
    // Path 1 (both w == 0): (v2 - v1).x = 4 - 1 = 3. a=(1,1,1,0) b=(4,4,4,0).
    try std.testing.expectEqual(@as(f32, 3.0), try runF32(allocator, src, "f", F8, .{ @as(f32, 1), @as(f32, 1), @as(f32, 1), @as(f32, 0), @as(f32, 4), @as(f32, 4), @as(f32, 4), @as(f32, 0) }));
    // Path 3 (v1.w != 0, v2.w == 0): v2.x = 9. a=(1,1,1,2) b=(9,9,9,0).
    try std.testing.expectEqual(@as(f32, 9.0), try runF32(allocator, src, "f", F8, .{ @as(f32, 1), @as(f32, 1), @as(f32, 1), @as(f32, 2), @as(f32, 9), @as(f32, 9), @as(f32, 9), @as(f32, 0) }));
}

// mat3 from three vec3 columns (the terrain `mat3(vTangent, vBinormal, vNormal)` ctor),
// then mat * vec. Columns (1,0,0),(0,1,0),(0,0,1) = identity, m*(7,8,9) = (7,8,9).
test "glsl: mat3 from three vec3 columns, then mat*vec" {
    const allocator = std.testing.allocator;
    const src =
        \\float f(vec3 a, vec3 b, vec3 c, vec3 v) {
        \\    mat3 m = mat3(a, b, c);
        \\    return dot(m * v, vec3(1.0));
        \\}
    ;
    const F12 = *const fn (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32;
    // identity columns, v=(7,8,9) -> 7+8+9 = 24
    try std.testing.expectEqual(@as(f32, 24.0), try runF32(allocator, src, "f", F12, .{
        @as(f32, 1), @as(f32, 0), @as(f32, 0),
        @as(f32, 0), @as(f32, 1), @as(f32, 0),
        @as(f32, 0), @as(f32, 0), @as(f32, 1),
        @as(f32, 7), @as(f32, 8), @as(f32, 9),
    }));
}

// A constant-bound `for` loop indexing a local array by the loop variable is fully
// unrolled so the index folds (the terrain point-light loop / ideas lightSource loop
// shape). Sum arr[i] over the unrolled iterations.
test "glsl: constant-bound loop unrolls + indexes an array by the loop variable" {
    const allocator = std.testing.allocator;
    const src =
        \\float f(float a, float b, float c) {
        \\    float arr[3];
        \\    arr[0] = a; arr[1] = b; arr[2] = c;
        \\    float sum = 0.0;
        \\    for (int i = 0; i < 3; i++) { sum = sum + arr[i]; }
        \\    return sum;
        \\}
    ;
    const F3 = *const fn (f32, f32, f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 60.0), try runF32(allocator, src, "f", F3, .{ @as(f32, 10), @as(f32, 20), @as(f32, 30) }));
}

// An array of user structs, member assignment, and member read by a (unrolled) loop
// index: the precise ideas-logo `LightSourceParameters lightSource[N]` shape.
test "glsl: array of structs - member assign then indexed read in an unrolled loop" {
    const allocator = std.testing.allocator;
    const src =
        \\struct Light { vec3 color; float intensity; };
        \\float f(float r0, float i0, float r1, float i1) {
        \\    Light lights[2];
        \\    lights[0].color = vec3(r0, 0.0, 0.0);
        \\    lights[0].intensity = i0;
        \\    lights[1].color = vec3(r1, 0.0, 0.0);
        \\    lights[1].intensity = i1;
        \\    float acc = 0.0;
        \\    for (int i = 0; i < 2; i++) {
        \\        acc = acc + lights[i].color.x * lights[i].intensity;
        \\    }
        \\    return acc;
        \\}
    ;
    const F4 = *const fn (f32, f32, f32, f32) callconv(.c) f32;
    // r0*i0 + r1*i1 = 2*3 + 4*5 = 26
    try std.testing.expectEqual(@as(f32, 26.0), try runF32(allocator, src, "f", F4, .{ @as(f32, 2), @as(f32, 3), @as(f32, 4), @as(f32, 5) }));
}

// A struct constructor binding fields in order, then a member read.
test "glsl: struct constructor + member access" {
    const allocator = std.testing.allocator;
    const src =
        \\struct S { vec3 a; float b; };
        \\float f(float x) {
        \\    S s = S(vec3(x, x, x), x);
        \\    return s.a.y + s.b;
        \\}
    ;
    const F1 = *const fn (f32) callconv(.c) f32;
    try std.testing.expectEqual(@as(f32, 10.0), try runF32(allocator, src, "f", F1, .{@as(f32, 5.0)})); // x + x
}

// A fragment shader using a uniform ARRAY indexed in a constant-bound loop emits a
// push-constant block and surfaces the array uniform in the layout with array_len.
test "glsl->spir-v: uniform array indexed in a loop + layout surfaces array_len" {
    const allocator = std.testing.allocator;
    const src =
        \\precision mediump float;
        \\#define N 3
        \\uniform vec3 pointLightColor[ N ];
        \\varying vec3 vNormal;
        \\void main() {
        \\    vec3 sum = vec3(0.0);
        \\    for (int i = 0; i < N; i++) { sum += pointLightColor[i]; }
        \\    gl_FragColor = vec4(sum, 1.0);
        \\}
    ;
    var c = try glsl.compileShaderWithLayout(allocator, src, .fragment);
    defer c.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 0x07230203), c.spirv[0]);
    // One uniform member, an array of length 3, 3 floats per element.
    try std.testing.expectEqual(@as(usize, 1), c.uniforms.len);
    try std.testing.expectEqualStrings("pointLightColor", c.uniforms[0].name);
    try std.testing.expectEqual(@as(u32, 3), c.uniforms[0].array_len);
    try std.testing.expectEqual(@as(u32, 3), c.uniforms[0].float_count); // vec3 element stride
    try std.testing.expectEqual(@as(u32, 0), c.uniforms[0].offset_floats);
    try std.testing.expectEqual(@as(u32, 36), c.block_size); // 3 elems * 3 floats * 4 bytes
}

// A struct + array-of-structs declared at module scope compiles to valid SPIR-V (the
// ideas-logo fragment shader's `struct LightSourceParameters` + `lightSource[3]` shape).
test "glsl->spir-v: module-scope struct + array-of-structs compiles to valid SPIR-V" {
    const allocator = std.testing.allocator;
    const src =
        \\precision mediump float;
        \\struct LightSourceParameters { vec4 ambient; vec4 diffuse; vec4 position; };
        \\LightSourceParameters lightSource[2];
        \\uniform vec4 light0Position;
        \\uniform vec4 light1Position;
        \\varying vec3 vertex_normal;
        \\void main() {
        \\    lightSource[0] = LightSourceParameters(vec4(0.1), vec4(1.0), vec4(0.0));
        \\    lightSource[1] = LightSourceParameters(vec4(0.2), vec4(0.5), vec4(0.0));
        \\    lightSource[0].position = light0Position;
        \\    lightSource[1].position = light1Position;
        \\    vec4 acc = vec4(0.0);
        \\    for (int i = 0; i < 2; i++) {
        \\        acc += lightSource[i].diffuse * max(0.0, dot(normalize(vertex_normal), lightSource[i].position.xyz));
        \\    }
        \\    gl_FragColor = acc + lightSource[0].ambient;
        \\}
    ;
    const words = try glsl.compileShaderToSpirv(allocator, src, .fragment);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(u32, 0x07230203), words[0]);
    // It is a real fragment entry point and lowers back through the Vulcan SPIR-V reader.
    var ff = try spirv.lowerModule(allocator, words);
    ff.deinit();
}

test "glsl: bitwise and shift operators" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 3), try runI32(allocator, "int f(int x){ return x & 3; }", "f", T, .{7}));
    try std.testing.expectEqual(@as(i32, 11), try runI32(allocator, "int f(int x){ return x | 8; }", "f", T, .{3}));
    try std.testing.expectEqual(@as(i32, 4), try runI32(allocator, "int f(int x){ return x ^ 1; }", "f", T, .{5}));
    try std.testing.expectEqual(@as(i32, 12), try runI32(allocator, "int f(int x){ return x << 2; }", "f", T, .{3}));
    try std.testing.expectEqual(@as(i32, 4), try runI32(allocator, "int f(int x){ return x >> 1; }", "f", T, .{8}));
    try std.testing.expectEqual(@as(i32, -6), try runI32(allocator, "int f(int x){ return ~x; }", "f", T, .{5})); // ~5 = -6
    // Precedence: << binds tighter than |, so `x << 2 | 1` is `(x<<2) | 1` = 5 for x=1.
    try std.testing.expectEqual(@as(i32, 5), try runI32(allocator, "int f(int x){ return x << 2 | 1; }", "f", T, .{1}));
}

test "glsl: hexadecimal integer literals" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Hex masking: 0xFF keeps the low byte.
    try std.testing.expectEqual(@as(i32, 0x34), try runI32(allocator, "int f(int x){ return x & 0xFF; }", "f", T, .{0x1234}));
    // Upper-case 0X and mixed-case hex digits both parse.
    try std.testing.expectEqual(@as(i32, 0xAB), try runI32(allocator, "int f(int x){ return x & 0X00ff; }", "f", T, .{0xAB}));
    // Hex constant used directly as a value.
    try std.testing.expectEqual(@as(i32, 256), try runI32(allocator, "int f(int x){ return x + 0x100; }", "f", T, .{0}));
    // Round-trips through SPIR-V too.
    try std.testing.expectEqual(@as(i32, 0x34), try runSpirvI32(allocator, "int f(int x){ return x & 0xFF; }", T, .{0x1234}));
}

test "glsl: octal integer literals" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // 0777 is octal 511 (not decimal 777).
    try std.testing.expectEqual(@as(i32, 511), try runI32(allocator, "int f(int x){ return x + 0777; }", "f", T, .{0}));
    // 010 is octal 8.
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, "int f(int x){ return x + 010; }", "f", T, .{0}));
    // A lone 0 stays 0.
    try std.testing.expectEqual(@as(i32, 5), try runI32(allocator, "int f(int x){ return x + 0; }", "f", T, .{5}));
}

test "glsl: unsigned integer literal suffix" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // A `u`/`U` suffix on decimal, hex, and octal literals is accepted and ignored.
    try std.testing.expectEqual(@as(i32, 42), try runI32(allocator, "int f(int x){ return x + 42u; }", "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, 0x34), try runI32(allocator, "int f(int x){ return x & 0xFFU; }", "f", T, .{0x1234}));
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, "int f(int x){ return x + 010u; }", "f", T, .{0}));
}

test "glsl: function-like macro end-to-end" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    // A function-like macro must survive preprocessing and compile/run through the pipeline.
    const src =
        \\#define SQ(x) ((x)*(x))
        \\float f(float v) { return SQ(v) + SQ(2.0); }
    ;
    // f(3) = 9 + 4 = 13.
    try std.testing.expectEqual(@as(f32, 13.0), try runF32(allocator, src, "f", *const fn (f32) callconv(.c) f32, .{@as(f32, 3.0)}));
}

test "glsl: integer vectors (ivec)" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Construction + component read + integer addition.
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, "int f(int x){ ivec2 v = ivec2(x, x+1); return v.x + v.y; }", "f", T, .{3}));
    // Integer bitwise-and on components (only valid because components are ints, not f32).
    try std.testing.expectEqual(@as(i32, 5), try runI32(allocator, "int f(int x){ ivec2 v = ivec2(x, 7); return v.x & v.y; }", "f", T, .{5}));
    // Component-wise vector add.
    try std.testing.expectEqual(@as(i32, 87), try runI32(allocator, "int f(int x){ ivec2 a = ivec2(x,2); ivec2 b = ivec2(3,x); ivec2 c = a + b; return c.x*10 + c.y; }", "f", T, .{5}));
    // Integer division truncates (float division would round differently): 7/3 == 2.
    try std.testing.expectEqual(@as(i32, 2), try runI32(allocator, "int f(int x){ ivec3 v = ivec3(x, 3, 0); return v.x / v.y; }", "f", T, .{7}));
    // Splat construction + shift.
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, "int f(int x){ ivec2 v = ivec2(x); return v.x << v.y; }", "f", T, .{2}));
}

test "glsl: vector relational functions" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // all(lessThan(...)) over an int vector.
    const a1 = "int f(int x){ ivec2 a = ivec2(x,2); bvec2 c = lessThan(a, ivec2(3,5)); return all(c) ? 100 : 200; }";
    try std.testing.expectEqual(@as(i32, 100), try runI32(allocator, a1, "f", T, .{1})); // (1<3,2<5)=(T,T)
    try std.testing.expectEqual(@as(i32, 200), try runI32(allocator, a1, "f", T, .{5})); // (5<3,2<5)=(F,T)
    // any(greaterThan(...)).
    const a2 = "int f(int x){ bvec2 c = greaterThan(ivec2(x,0), ivec2(3,3)); return any(c) ? 1 : 0; }";
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, a2, "f", T, .{5})); // (T,F)
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, a2, "f", T, .{1})); // (F,F)
    // not(equal(...)) then read a lane as bool.
    const a3 = "int f(int x){ bvec2 c = not(equal(ivec2(x,2), ivec2(1,2))); return c.x ? 10 : 0; }";
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, a3, "f", T, .{1})); // eq=(T,T),not=(F,F)
    try std.testing.expectEqual(@as(i32, 10), try runI32(allocator, a3, "f", T, .{9})); // eq=(F,T),not=(T,F)
    // Float vectors compare too.
    const a4 = "int f(int x){ vec2 a = vec2(float(x), 2.0); bvec2 c = lessThan(a, vec2(3.0,5.0)); return all(c) ? 1 : 0; }";
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, a4, "f", T, .{1}));
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, a4, "f", T, .{9}));
}

test "glsl: integer vectors through function params and returns" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // The inlined helper takes ivec params and returns an ivec; both must stay integer
    // (a regression here typed the scalarized params/return as f32 and read garbage).
    const src =
        \\ivec2 addv(ivec2 a, ivec2 b) { return a + b; }
        \\int f(int x) { ivec2 r = addv(ivec2(x, 1), ivec2(2, x)); return r.x*100 + r.y; }
    ;
    try std.testing.expectEqual(@as(i32, 706), try runI32(allocator, src, "f", T, .{5})); // (7,6)
    try std.testing.expectEqual(@as(i32, 201), try runI32(allocator, src, "f", T, .{0})); // (2,1)
    // uvec through a helper too (unsigned shift semantics preserved).
    const usrc =
        \\uvec2 shr1(uvec2 v) { return uvec2(v.x >> 1u, v.y >> 1u); }
        \\int f(int x) { uvec2 r = shr1(uvec2(x, 8)); return int(r.x) - int(r.y); }
    ;
    try std.testing.expectEqual(@as(i32, 2147483643), try runI32(allocator, usrc, "f", T, .{-1})); // 0x7FFFFFFF - 4
}

test "glsl: struct member assignment" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Writing to struct members (s.a = ...) was previously rejected as a bad swizzle.
    const src = "struct P { int a; int b; }; int f(int x){ P p; p.a = x*2; p.b = x+7; return p.a*100 + p.b; }";
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, src, "f", T, .{0})); // (0, 7)
    try std.testing.expectEqual(@as(i32, 610), try runI32(allocator, src, "f", T, .{3})); // (6, 10)
    // Round-trips through SPIR-V (scalarized struct fields).
    try std.testing.expectEqual(@as(i32, 610), try runSpirvI32(allocator, src, T, .{3}));
    // Float-typed struct members via member write, then read.
    const fsrc = "struct V { float x; float y; }; float f(float t){ V v; v.x = t; v.y = t*2.0; return v.x + v.y; }";
    try std.testing.expectEqual(@as(f32, 9.0), try runF32(allocator, fsrc, "f", *const fn (f32) callconv(.c) f32, .{@as(f32, 3.0)}));
}

test "glsl: vector and matrix subscripting" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const F = *const fn (f32) callconv(.c) f32;
    // vec[i] with a constant index accesses a component.
    try std.testing.expectEqual(@as(f32, 30.0), try runF32(allocator, "float f(float x){ vec3 v = vec3(x, x*2.0, x*3.0); return v[2]; }", "f", F, .{@as(f32, 10.0)}));
    // Sum over components via an unrolled loop indexing the vector.
    try std.testing.expectEqual(@as(f32, 6.0), try runF32(allocator, "float f(float x){ vec3 v = vec3(x, x, x); float s = 0.0; for(int i=0;i<3;i=i+1){ s = s + v[i]; } return s; }", "f", F, .{@as(f32, 2.0)}));
    // mat[i] returns column i as a vector.
    try std.testing.expectEqual(@as(f32, 7.0), try runF32(allocator, "float f(float x){ mat2 m = mat2(1.0, 2.0, 3.0, 4.0); vec2 col = m[1]; return col.x + col.y; }", "f", F, .{@as(f32, 0.0)}));
    // Integer vector subscript stays integer.
    const T = *const fn (i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 3), try runI32(allocator, "int f(int x){ ivec2 v = ivec2(x, 3); return v[1]; }", "f", T, .{7}));
}

test "glsl->spir-v: integer vector loop accumulator round-trip" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const src = "int f(int n){ ivec2 acc = ivec2(0,0); for(int i=0;i<n;i=i+1){ acc = acc + ivec2(i, 1); } return acc.x*100 + acc.y; }";
    try std.testing.expectEqual(@as(i32, 0), try runSpirvI32(allocator, src, T, .{0}));
    try std.testing.expectEqual(@as(i32, 303), try runSpirvI32(allocator, src, T, .{3}));
}

test "glsl: integer vector accumulator through a loop" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // An ivec loop variable needs int-typed phi params; a regression here read garbage when
    // the loop did not execute (n=0 must return the initial 0, not stale values).
    const src = "int f(int n){ ivec2 acc = ivec2(0,0); for(int i=0;i<n;i=i+1){ acc = acc + ivec2(i, 1); } return acc.x*100 + acc.y; }";
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, src, "f", T, .{0})); // loop skipped
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, src, "f", T, .{1})); // acc=(0,1)
    try std.testing.expectEqual(@as(i32, 102), try runI32(allocator, src, "f", T, .{2})); // acc=(1,2)
    try std.testing.expectEqual(@as(i32, 303), try runI32(allocator, src, "f", T, .{3})); // acc=(3,3)
}

test "glsl: trunc and round builtins" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const F = *const fn (f32) callconv(.c) f32;
    // trunc rounds toward zero.
    try std.testing.expectEqual(@as(f32, 2.0), try runF32(allocator, "float f(float x){ return trunc(x); }", "f", F, .{@as(f32, 2.7)}));
    try std.testing.expectEqual(@as(f32, -2.0), try runF32(allocator, "float f(float x){ return trunc(x); }", "f", F, .{@as(f32, -2.7)}));
    // round-to-nearest-even.
    try std.testing.expectEqual(@as(f32, 3.0), try runF32(allocator, "float f(float x){ return round(x); }", "f", F, .{@as(f32, 2.7)}));
    try std.testing.expectEqual(@as(f32, 2.0), try runF32(allocator, "float f(float x){ return round(x); }", "f", F, .{@as(f32, 2.5)})); // even
    // Both round-trip through SPIR-V.
    try std.testing.expectEqual(@as(f32, 2.0), try runSpirvF32(allocator, "float f(float x){ return trunc(x); }", F, .{@as(f32, 2.7)}));
    try std.testing.expectEqual(@as(f32, 4.0), try runSpirvF32(allocator, "float f(float x){ return round(x); }", F, .{@as(f32, 3.5)})); // even
}

test "glsl: bitCount builtin" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const src = "int f(int x){ return bitCount(x); }";
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, src, "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, src, "f", T, .{0xFF}));
    try std.testing.expectEqual(@as(i32, 32), try runI32(allocator, src, "f", T, .{-1})); // 0xFFFFFFFF
    try std.testing.expectEqual(@as(i32, 13), try runI32(allocator, src, "f", T, .{0x12345678}));
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, src, "f", T, .{0x40000000}));
    // Component-wise on an integer vector, summed.
    const vsrc = "int f(int x){ ivec2 c = bitCount(ivec2(x, 7)); return c.x*100 + c.y; }";
    try std.testing.expectEqual(@as(i32, 803), try runI32(allocator, vsrc, "f", T, .{0xFF})); // (8,3)
}

test "glsl: packUnorm4x8 and unpackUnorm4x8 builtins" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // pack: comp0=255, comp1=0, comp2=51 (0.2*255), comp3=102 (0.4*255) -> 0x663300FF.
    try std.testing.expectEqual(@as(i32, 0x663300FF), try runI32(allocator, "int f(int x){ vec4 v = vec4(1.0, 0.0, 0.2, 0.4); return int(packUnorm4x8(v)); }", "f", T, .{0}));
    // clamp: values outside [0,1] saturate. -0.5 -> 0, 2.0 -> 255.
    try std.testing.expectEqual(@as(i32, 0xFF), try runI32(allocator, "int f(int x){ vec4 v = vec4(2.0, 0.0, 0.0, 0.0); return int(packUnorm4x8(v)); }", "f", T, .{0}));
    // unpack round-trip: byte 2 of 0x663300FF is 0x33 = 51 -> 0.2 -> *255 ~= 51.
    try std.testing.expectEqual(@as(i32, 51), try runI32(allocator, "int f(int x){ vec4 v = unpackUnorm4x8(uint(x)); return int(v.z * 255.0 + 0.5); }", "f", T, .{0x663300FF}));
    // unpack byte 0 (0xFF) -> 1.0.
    try std.testing.expectEqual(@as(i32, 255), try runI32(allocator, "int f(int x){ vec4 v = unpackUnorm4x8(uint(x)); return int(v.x * 255.0 + 0.5); }", "f", T, .{0x663300FF}));
}

test "glsl: packSnorm4x8 and 2x16 pack/unpack builtins" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Snorm: 1.0 -> 127 (0x7F), -1.0 -> -127 (0x81 byte). vec4(1,-1,0,0) -> 0x0000817F.
    try std.testing.expectEqual(@as(i32, 0x0000817F), try runI32(allocator, "int f(int x){ return int(packSnorm4x8(vec4(1.0,-1.0,0.0,0.0))); }", "f", T, .{0}));
    // Snorm round-trip: unpack lane 1 (0x81 = -127) -> -1.0.
    try std.testing.expectEqual(@as(i32, -100), try runI32(allocator, "int f(int x){ vec4 v = unpackSnorm4x8(uint(x)); return int(v.y * 100.0); }", "f", T, .{0x0000817F}));
    // Unorm 2x16: 1.0 -> 65535 (0xFFFF). vec2(1,0) -> 0x0000FFFF.
    try std.testing.expectEqual(@as(i32, 0x0000FFFF), try runI32(allocator, "int f(int x){ return int(packUnorm2x16(vec2(1.0, 0.0))); }", "f", T, .{0}));
    // Unorm 2x16 round-trip: lane 0 (0xFFFF) -> 1.0.
    try std.testing.expectEqual(@as(i32, 65535), try runI32(allocator, "int f(int x){ vec2 v = unpackUnorm2x16(uint(x)); return int(v.x * 65535.0 + 0.5); }", "f", T, .{0x0000FFFF}));
    // Snorm 2x16: 1.0 -> 32767 (0x7FFF).
    try std.testing.expectEqual(@as(i32, 0x00007FFF), try runI32(allocator, "int f(int x){ return int(packSnorm2x16(vec2(1.0, 0.0))); }", "f", T, .{0}));
}

test "glsl: bitfieldExtract and bitfieldInsert builtins" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Unsigned extract: (x >> 8) & 0xFF.
    try std.testing.expectEqual(@as(i32, 0xEF), try runI32(allocator, "int f(int x){ return int(bitfieldExtract(uint(x), 8, 8)); }", "f", T, .{@as(i32, @bitCast(@as(u32, 0xABCDEF12)))}));
    // Signed extract sign-extends: low 4 bits of 0xF is 1111 -> -1; of 7 -> 7.
    const se = "int f(int x){ return bitfieldExtract(x, 0, 4); }";
    try std.testing.expectEqual(@as(i32, -1), try runI32(allocator, se, "f", T, .{15}));
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, se, "f", T, .{7}));
    // bits == 0 extracts nothing.
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, "int f(int x){ return bitfieldExtract(x, 4, 0); }", "f", T, .{0xFFFF}));
    // Insert 0xF into bits [4,8): 0x0F -> 0xFF, 0 -> 0xF0.
    const ins = "int f(int x){ return int(bitfieldInsert(uint(x), uint(15), 4, 4)); }";
    try std.testing.expectEqual(@as(i32, 0xFF), try runI32(allocator, ins, "f", T, .{0x0F}));
    try std.testing.expectEqual(@as(i32, 0xF0), try runI32(allocator, ins, "f", T, .{0}));
    // bits == 0 inserts nothing.
    try std.testing.expectEqual(@as(i32, 15), try runI32(allocator, "int f(int x){ return int(bitfieldInsert(uint(x), uint(15), 4, 0)); }", "f", T, .{15}));
}

test "glsl: bitfieldReverse builtin" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const src = "int f(int x){ return bitfieldReverse(x); }";
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, src, "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, -2147483648), try runI32(allocator, src, "f", T, .{1})); // 1 -> 0x80000000
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, src, "f", T, .{-2147483648})); // 0x80000000 -> 1
    try std.testing.expectEqual(@as(i32, -65536), try runI32(allocator, src, "f", T, .{0x0000FFFF})); // -> 0xFFFF0000
    try std.testing.expectEqual(@as(i32, 0x12345678), try runI32(allocator, src, "f", T, .{0x1E6A2C48})); // reverse round-trips
}

test "glsl: findLSB and findMSB builtins" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const lsb = "int f(int x){ return findLSB(x); }";
    try std.testing.expectEqual(@as(i32, -1), try runI32(allocator, lsb, "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, lsb, "f", T, .{1}));
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, lsb, "f", T, .{0x80}));
    try std.testing.expectEqual(@as(i32, 3), try runI32(allocator, lsb, "f", T, .{0x12345678}));
    // findMSB on unsigned.
    const msu = "int f(int x){ return findMSB(uint(x)); }";
    try std.testing.expectEqual(@as(i32, -1), try runI32(allocator, msu, "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, 0), try runI32(allocator, msu, "f", T, .{1}));
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, msu, "f", T, .{0xFF}));
    try std.testing.expectEqual(@as(i32, 31), try runI32(allocator, msu, "f", T, .{-1})); // 0xFFFFFFFF
    // findMSB on signed: 0 and -1 give -1; negatives use the most significant 0 bit.
    const mss = "int f(int x){ return findMSB(x); }";
    try std.testing.expectEqual(@as(i32, -1), try runI32(allocator, mss, "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, -1), try runI32(allocator, mss, "f", T, .{-1}));
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, mss, "f", T, .{0xFF}));
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, mss, "f", T, .{-256})); // ~(-256)=0xFF
}

test "glsl: bit-reinterpret builtins (floatBitsToInt etc.)" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // floatBitsToInt(1.0) == 0x3F800000.
    try std.testing.expectEqual(@as(i32, 0x3F800000), try runI32(allocator, "int f(int x){ float g = 1.0; return floatBitsToInt(g); }", "f", T, .{0}));
    // Round-trip: floatBitsToInt(intBitsToFloat(x)) == x for any bit pattern.
    const rt = "int f(int x){ return floatBitsToInt(intBitsToFloat(x)); }";
    try std.testing.expectEqual(@as(i32, 1065353216), try runI32(allocator, rt, "f", T, .{1065353216}));
    try std.testing.expectEqual(@as(i32, 12345), try runI32(allocator, rt, "f", T, .{12345}));
    // Unsigned variants round-trip too.
    const rtu = "int f(int x){ return int(floatBitsToUint(uintBitsToFloat(uint(x)))); }";
    try std.testing.expectEqual(@as(i32, 777), try runI32(allocator, rtu, "f", T, .{777}));
}

test "glsl: unsigned integer vectors (uvec)" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Construction + component read + unsigned add.
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, "int f(int x){ uvec2 v = uvec2(x, 3); return int(v.x + v.y); }", "f", T, .{5}));
    // Unsigned right-shift: 0xFFFFFFFF >> 1 == 0x7FFFFFFF (a signed shift would stay -1).
    try std.testing.expectEqual(@as(i32, 2147483647), try runI32(allocator, "int f(int x){ uvec2 v = uvec2(x, 1); return int(v.x >> v.y); }", "f", T, .{-1}));
    // Unsigned division of a large-bit-pattern value.
    try std.testing.expectEqual(@as(i32, 0x7FFFFFFF), try runI32(allocator, "int f(int x){ uvec2 v = uvec2(x, 2); return int(v.x / v.y); }", "f", T, .{-2})); // 0xFFFFFFFE / 2
}

test "glsl: min/max/abs/clamp preserve integer vector semantics" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Large magnitudes an f32 cannot represent exactly: routing an ivec through float
    // min/max/abs would round (e.g. 16777217 -> 16777216).
    const big = 16777217; // 2^24 + 1, not exactly representable as f32
    try std.testing.expectEqual(@as(i32, big), try runI32(allocator, "int f(int x){ ivec2 v = ivec2(-x, 0); return abs(v).x; }", "f", T, .{big}));
    // The result must stay usable as an int (a bitwise op would break on an f32 result).
    const mn = "int f(int x){ ivec2 a=ivec2(x,0); ivec2 b=ivec2(7,0); return min(a,b).x & 3; }";
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, mn, "f", T, .{5})); // min(5,7)=5, 5&3=1
    try std.testing.expectEqual(@as(i32, 3), try runI32(allocator, mn, "f", T, .{10})); // min(10,7)=7, 7&3=3
    // clamp on an ivec stays integer.
    const cl = "int f(int x){ ivec2 v = ivec2(x, 0); return clamp(v, ivec2(2,0), ivec2(8,0)).x; }";
    try std.testing.expectEqual(@as(i32, 2), try runI32(allocator, cl, "f", T, .{-5}));
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, cl, "f", T, .{100}));
    try std.testing.expectEqual(@as(i32, 5), try runI32(allocator, cl, "f", T, .{5}));
}

test "glsl: mix with boolean select" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Scalar bool: mix(x, y, cond) picks y when cond is true.
    const s = "int f(int x){ float r = mix(5.0, 9.0, x > 3); return int(r); }";
    try std.testing.expectEqual(@as(i32, 9), try runI32(allocator, s, "f", T, .{5}));
    try std.testing.expectEqual(@as(i32, 5), try runI32(allocator, s, "f", T, .{1}));
    // Vector bvec: component-wise select of b where the lane is true.
    const v = "int f(int x){ vec2 a=vec2(1.0,2.0); vec2 b=vec2(10.0,20.0); bvec2 sel=greaterThan(vec2(float(x)), vec2(0.0,5.0)); vec2 r=mix(a,b,sel); return int(r.x)*100 + int(r.y); }";
    try std.testing.expectEqual(@as(i32, 1002), try runI32(allocator, v, "f", T, .{3})); // (T,F) -> (10,2)
    try std.testing.expectEqual(@as(i32, 102), try runI32(allocator, v, "f", T, .{-1})); // (F,F) -> (1,2)
    try std.testing.expectEqual(@as(i32, 1020), try runI32(allocator, v, "f", T, .{10})); // (T,T) -> (10,20)
}

test "glsl->spir-v: broad builtin round-trip sweep" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const F = *const fn (f32) callconv(.c) f32;
    // Builtins the frontend lowers inline (arith/select) should all round-trip through the
    // SPIR-V reader without a hardware intrinsic.
    const cases = [_]struct { src: []const u8, x: f32, want: f32 }{
        .{ .src = "float f(float x){ return abs(x); }", .x = -3.0, .want = 3.0 },
        .{ .src = "float f(float x){ return sign(x); }", .x = -2.0, .want = -1.0 },
        .{ .src = "float f(float x){ return min(x, 1.0); }", .x = 5.0, .want = 1.0 },
        .{ .src = "float f(float x){ return max(x, 1.0); }", .x = 5.0, .want = 5.0 },
        .{ .src = "float f(float x){ return clamp(x, 0.0, 1.0); }", .x = 2.0, .want = 1.0 },
        .{ .src = "float f(float x){ return mix(0.0, 10.0, x); }", .x = 0.5, .want = 5.0 },
        .{ .src = "float f(float x){ return step(1.0, x); }", .x = 2.0, .want = 1.0 },
        .{ .src = "float f(float x){ return smoothstep(0.0, 1.0, x); }", .x = 0.5, .want = 0.5 },
        .{ .src = "float f(float x){ return mod(x, 3.0); }", .x = 7.0, .want = 1.0 },
        .{ .src = "float f(float x){ return floor(x) + ceil(x); }", .x = 2.3, .want = 5.0 },
        .{ .src = "float f(float x){ return fract(x); }", .x = 2.25, .want = 0.25 },
        .{ .src = "float f(float x){ return radians(x); }", .x = 180.0, .want = std.math.pi },
        .{ .src = "float f(float x){ vec3 a = vec3(x, 0.0, 0.0); vec3 b = vec3(0.0, 1.0, 0.0); return cross(a, b).z; }", .x = 2.0, .want = 2.0 },
        .{ .src = "float f(float x){ return distance(vec2(x, 0.0), vec2(0.0, 0.0)); }", .x = 4.0, .want = 4.0 },
    };
    inline for (cases) |c| {
        try std.testing.expectApproxEqAbs(c.want, try runSpirvF32(allocator, c.src, F, .{c.x}), 0.0001);
    }
}

test "glsl->spir-v: sqrt/length/normalize round-trip and run" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const F = *const fn (f32) callconv(.c) f32;
    // sqrt lowers to OpExtInst Sqrt; the reader should map it back to the IR sqrt op (fsqrt).
    try std.testing.expectEqual(@as(f32, 4.0), try runSpirvF32(allocator, "float f(float x){ return sqrt(x); }", F, .{@as(f32, 16.0)}));
    // length(vec2(x,x)) = sqrt(2) * |x|.
    try std.testing.expectEqual(@as(f32, 5.0), try runSpirvF32(allocator, "float f(float x){ vec2 v = vec2(x, 0.0); return length(v); }", F, .{@as(f32, 5.0)}));
    // normalize then scale back: length(normalize(v)) == 1.
    try std.testing.expectEqual(@as(f32, 1.0), try runSpirvF32(allocator, "float f(float x){ vec2 v = normalize(vec2(x, x)); return length(v); }", F, .{@as(f32, 3.0)}));
}

test "glsl->spir-v: pack/unpack builtins round-trip" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Uses nearest (round) + float<->int convert through the emitter.
    try std.testing.expectEqual(@as(i32, 0x663300FF), try runSpirvI32(allocator, "int f(int x){ return int(packUnorm4x8(vec4(1.0,0.0,0.2,0.4))); }", T, .{0}));
    try std.testing.expectEqual(@as(i32, 51), try runSpirvI32(allocator, "int f(int x){ vec4 v = unpackUnorm4x8(uint(x)); return int(v.z*255.0+0.5); }", T, .{0x663300FF}));
}

test "glsl->spir-v: integer bit builtins round-trip" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 13), try runSpirvI32(allocator, "int f(int x){ return bitCount(x); }", T, .{0x12345678}));
    try std.testing.expectEqual(@as(i32, 3), try runSpirvI32(allocator, "int f(int x){ return findLSB(x); }", T, .{0x12345678}));
    try std.testing.expectEqual(@as(i32, -1), try runSpirvI32(allocator, "int f(int x){ return bitfieldExtract(x, 0, 4); }", T, .{15}));
    try std.testing.expectEqual(@as(i32, 0xF0), try runSpirvI32(allocator, "int f(int x){ return int(bitfieldInsert(uint(x), uint(15), 4, 4)); }", T, .{0}));
}

test "glsl->spir-v: unsigned integer vectors round-trip" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Exercises the int->uint reinterpret (OpBitcast) and unsigned shift through SPIR-V.
    try std.testing.expectEqual(@as(i32, 2147483647), try runSpirvI32(allocator, "int f(int x){ uvec2 v = uvec2(x, 1); return int(v.x >> v.y); }", T, .{-1}));
    try std.testing.expectEqual(@as(i32, 8), try runSpirvI32(allocator, "int f(int x){ uvec2 v = uvec2(x, 3); return int(v.x + v.y); }", T, .{5}));
}

test "glsl->spir-v: vector relational functions round-trip" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const src = "int f(int x){ bvec2 c = lessThan(ivec2(x,2), ivec2(3,5)); return all(c) ? 100 : 200; }";
    try std.testing.expectEqual(@as(i32, 100), try runSpirvI32(allocator, src, T, .{1}));
    try std.testing.expectEqual(@as(i32, 200), try runSpirvI32(allocator, src, T, .{5}));
}

test "glsl->spir-v: integer vectors round-trip and run" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Integer vectors must also survive GLSL -> SPIR-V -> IR -> JIT (the emitter should
    // treat scalarized int lanes as int, not f32).
    try std.testing.expectEqual(@as(i32, 5), try runSpirvI32(allocator, "int f(int x){ ivec2 v = ivec2(x, 7); return v.x & v.y; }", T, .{5}));
    try std.testing.expectEqual(@as(i32, 7), try runSpirvI32(allocator, "int f(int x){ ivec2 v = ivec2(x, x+1); return v.x + v.y; }", T, .{3}));
    try std.testing.expectEqual(@as(i32, 2), try runSpirvI32(allocator, "int f(int x){ ivec3 v = ivec3(x, 3, 0); return v.x / v.y; }", T, .{7}));
}

test "glsl: switch statement" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const src =
        \\int f(int x) {
        \\  int r = 0;
        \\  switch (x) {
        \\    case 0: r = 10; break;
        \\    case 1: r = 20; break;
        \\    case 2: r = 30; break;
        \\    default: r = 99; break;
        \\  }
        \\  return r;
        \\}
    ;
    try std.testing.expectEqual(@as(i32, 10), try runI32(allocator, src, "f", T, .{0}));
    try std.testing.expectEqual(@as(i32, 20), try runI32(allocator, src, "f", T, .{1}));
    try std.testing.expectEqual(@as(i32, 30), try runI32(allocator, src, "f", T, .{2}));
    try std.testing.expectEqual(@as(i32, 99), try runI32(allocator, src, "f", T, .{7}));
}

test "glsl: switch with grouped labels and no default" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Grouped (empty) labels share the next non-empty body. No default: an unmatched
    // selector leaves r untouched.
    const src =
        \\int f(int x) {
        \\  int r = -1;
        \\  switch (x) {
        \\    case 1:
        \\    case 2: r = 100; break;
        \\    case 3: r = 300; break;
        \\  }
        \\  return r;
        \\}
    ;
    try std.testing.expectEqual(@as(i32, 100), try runI32(allocator, src, "f", T, .{1}));
    try std.testing.expectEqual(@as(i32, 100), try runI32(allocator, src, "f", T, .{2}));
    try std.testing.expectEqual(@as(i32, 300), try runI32(allocator, src, "f", T, .{3}));
    try std.testing.expectEqual(@as(i32, -1), try runI32(allocator, src, "f", T, .{9}));
}

test "glsl->spir-v: switch round-trips and runs" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    const src =
        \\int f(int x) {
        \\  int r = 0;
        \\  switch (x) {
        \\    case 0: r = 10; break;
        \\    case 1: r = 20; break;
        \\    default: r = 99; break;
        \\  }
        \\  return r;
        \\}
    ;
    try std.testing.expectEqual(@as(i32, 10), try runSpirvI32(allocator, src, T, .{0}));
    try std.testing.expectEqual(@as(i32, 20), try runSpirvI32(allocator, src, T, .{1}));
    try std.testing.expectEqual(@as(i32, 99), try runSpirvI32(allocator, src, T, .{5}));
}

test "glsl: switch inside a loop with continue targeting the loop" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // `continue` in a case skips the current loop iteration (targets the enclosing for,
    // not the switch). Sum 0..n-1 but skip i == 2.
    const src =
        \\int f(int n) {
        \\  int sum = 0;
        \\  for (int i = 0; i < n; i++) {
        \\    switch (i) {
        \\      case 2: continue;
        \\      default: break;
        \\    }
        \\    sum = sum + i;
        \\  }
        \\  return sum;
        \\}
    ;
    // 0+1+3+4 = 8 (2 skipped) for n = 5.
    try std.testing.expectEqual(@as(i32, 8), try runI32(allocator, src, "f", T, .{5}));
}

test "glsl: switch rejects fall-through" {
    const allocator = std.testing.allocator;
    // A non-empty case that does not end in break would fall through: unsupported.
    const src =
        \\int f(int x) {
        \\  int r = 0;
        \\  switch (x) {
        \\    case 0: r = 10;
        \\    case 1: r = 20; break;
        \\  }
        \\  return r;
        \\}
    ;
    try std.testing.expectError(error.Unsupported, glsl.compile(allocator, src));
}

test "glsl->spir-v: bitwise and shift round-trip and run" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    // Emit GLSL bitwise -> SPIR-V (BitwiseAnd/Or/Xor/ShiftLeftLogical/ShiftRight*),
    // read it back via the SPIR-V frontend, JIT, and run.
    try std.testing.expectEqual(@as(i32, 3), try runSpirvI32(allocator, "int f(int x){ return x & 3; }", T, .{7}));
    try std.testing.expectEqual(@as(i32, 12), try runSpirvI32(allocator, "int f(int x){ return x << 2; }", T, .{3}));
    try std.testing.expectEqual(@as(i32, 4), try runSpirvI32(allocator, "int f(int x){ return x >> 1; }", T, .{8}));
    try std.testing.expectEqual(@as(i32, -6), try runSpirvI32(allocator, "int f(int x){ return ~x; }", T, .{5}));
    try std.testing.expectEqual(@as(i32, 23), try runSpirvI32(allocator, "int f(int x){ return ((x << 2) | (x & 3)) ^ (x >> 1); }", T, .{5}));
}

/// Differential: run a GLSL `int f(int)` via the direct IR path AND via the
/// GLSL->SPIR-V->IR round-trip, and require they agree. A mismatch is a bug in the
/// SPIR-V emitter or the SPIR-V frontend reader.
fn diffDirectVsSpirvI32(allocator: std.mem.Allocator, src: []const u8, x: i32) !void {
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const direct = buf.entry(*const fn (i32) callconv(.c) i32, 0)(x);

    const words = try glsl.compileToSpirv(allocator, src);
    defer allocator.free(words);
    var sf = try spirv.lowerModule(allocator, words);
    defer sf.deinit();
    var buf2 = try native.jitFunction(allocator, &sf);
    defer buf2.deinit();
    const via_spirv = buf2.entry(*const fn (i32) callconv(.c) i32, 0)(x);

    std.testing.expectEqual(direct, via_spirv) catch |e| {
        std.debug.print("\ndiff-spirv f({d}): direct={d} spirv={d}\nsrc: {s}\n", .{ x, direct, via_spirv, src });
        return e;
    };
}

test "glsl direct vs spir-v round-trip: differential (int)" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const cases = [_]struct { src: []const u8, xs: []const i32 }{
        .{ .src = "int f(int x){ return x*x + 3*x - 7; }", .xs = &.{ 0, 2, -3, 10 } },
        .{ .src = "int f(int x){ return x / 4 + x % 4; }", .xs = &.{ 10, -10, 7 } },
        .{ .src = "int f(int x){ return (x << 3) ^ (x & 6) | (x >> 1); }", .xs = &.{ 5, -1, 255 } },
        .{ .src = "int f(int x){ return ~x + (x < 0 ? 1 : 0); }", .xs = &.{ 5, -5, 0 } },
        .{ .src = "int f(int x){ if (x > 10) return 1; else if (x < 0) return -1; return x; }", .xs = &.{ 20, -3, 5 } },
        .{ .src = "int f(int x){ int s=0; for(int i=0;i<x;i=i+1) s=s+i*i; return s; }", .xs = &.{ 0, 3, 5 } },
        .{ .src = "int f(int x){ int c=0; for(int i=1;i<=x;i=i+1){ if ((i & 1) == 1) c=c+i; } return c; }", .xs = &.{ 5, 6, 0 } },
        .{ .src = "int f(int x){ x <<= 2; x |= 1; x ^= 4; return x; }", .xs = &.{ 3, -1, 0 } },
    };
    inline for (cases) |c| {
        for (c.xs) |x| try diffDirectVsSpirvI32(allocator, c.src, x);
    }
}

/// Float differential: direct GLSL->IR vs GLSL->SPIR-V->IR (exercises OpExtInst
/// GLSL.std.450 builtins on the SPIR-V path).
fn diffDirectVsSpirvF32(allocator: std.mem.Allocator, src: []const u8, x: f32) !void {
    var mod = try glsl.compile(allocator, src);
    defer mod.deinit(allocator);
    const f = mod.find("f") orelse return error.MissingFunction;
    var buf = try native.jitFunction(allocator, f);
    defer buf.deinit();
    const direct = buf.entry(*const fn (f32) callconv(.c) f32, 0)(x);

    const words = try glsl.compileToSpirv(allocator, src);
    defer allocator.free(words);
    var sf = try spirv.lowerModule(allocator, words);
    defer sf.deinit();
    var buf2 = try native.jitFunction(allocator, &sf);
    defer buf2.deinit();
    const via_spirv = buf2.entry(*const fn (f32) callconv(.c) f32, 0)(x);

    std.testing.expectEqual(direct, via_spirv) catch |e| {
        std.debug.print("\ndiff-spirv f({d}): direct={d} spirv={d}\nsrc: {s}\n", .{ x, direct, via_spirv, src });
        return e;
    };
}

test "glsl direct vs spir-v round-trip: differential (float + builtins)" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const cases = [_]struct { src: []const u8, xs: []const f32 }{
        .{ .src = "float f(float x){ return clamp(x, -1.0, 1.0); }", .xs = &.{ -2.0, 0.5, 3.0 } },
        .{ .src = "float f(float x){ return min(x, 2.0) + max(x, -2.0); }", .xs = &.{ -5.0, 0.0, 5.0 } },
        .{ .src = "float f(float x){ return sqrt(abs(x)) * sign(x); }", .xs = &.{ 4.0, -9.0, 0.0 } },
        .{ .src = "float f(float x){ return mix(1.0, 5.0, clamp(x, 0.0, 1.0)); }", .xs = &.{ -1.0, 0.25, 2.0 } },
        .{ .src = "float f(float x){ return floor(x) + fract(x) + ceil(x); }", .xs = &.{ 3.7, -1.2, 5.0 } },
        .{ .src = "float f(float x){ return step(0.0, x) + smoothstep(0.0, 1.0, x); }", .xs = &.{ -0.5, 0.5, 1.5 } },
        .{ .src = "float f(float x){ return x < 0.0 ? x*x : sqrt(x); }", .xs = &.{ -3.0, 16.0 } },
    };
    inline for (cases) |c| {
        for (c.xs) |x| try diffDirectVsSpirvF32(allocator, c.src, x);
    }
}

test "glsl: compound bitwise and shift assignments" {
    if (@import("builtin").cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const T = *const fn (i32) callconv(.c) i32;
    try std.testing.expectEqual(@as(i32, 6), try runI32(allocator, "int f(int x){ x &= 6; return x; }", "f", T, .{7}));
    try std.testing.expectEqual(@as(i32, 11), try runI32(allocator, "int f(int x){ x |= 8; return x; }", "f", T, .{3}));
    try std.testing.expectEqual(@as(i32, 4), try runI32(allocator, "int f(int x){ x ^= 1; return x; }", "f", T, .{5}));
    try std.testing.expectEqual(@as(i32, 16), try runI32(allocator, "int f(int x){ x <<= 3; return x; }", "f", T, .{2}));
    try std.testing.expectEqual(@as(i32, 10), try runI32(allocator, "int f(int x){ x >>= 2; return x; }", "f", T, .{40}));
    try std.testing.expectEqual(@as(i32, 1), try runI32(allocator, "int f(int x){ x %= 3; return x; }", "f", T, .{10}));
    // Compound assign inside a loop, combined: accumulate x|=(1<<i) then count via >>.
    try std.testing.expectEqual(@as(i32, 7), try runI32(allocator, "int f(int x){ int m=0; for(int i=0;i<x;i=i+1) m |= (1 << i); return m; }", "f", T, .{3})); // bits 0,1,2 = 7
}
