//! Execution tests for the GLSL frontend: compile GLSL source to Vulcan IR, JIT it for
//! the host via `vulcan-target.native`, and run it. The host is the JIT target, so scalar
//! GLSL functions run natively in-process.

const std = @import("std");
const glsl = @import("vulcan-glsl");
const spirv = @import("vulcan-spirv");
const native = @import("vulcan-target").native;

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
