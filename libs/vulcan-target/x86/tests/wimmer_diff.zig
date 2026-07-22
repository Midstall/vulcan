//! SP4 Task 4: the 32-bit x86 (i386, cdecl) SHARED Wimmer allocator produces EXECUTABLE code. Each
//! test builds an IR function, compiles it through `isel.compileFunctionWimmerX86` (the shared
//! allocator + the split translate/emit path, byte-addressability ebx staging, callee-saved esi,
//! div/shift guards, and for one shape the address fold), runs the bytes under qemu-i386, and asserts
//! the executed low byte equals a HAND-COMPUTED GROUND TRUTH (i32 two's-complement wrapping, exact
//! evaluation order, mod 256 because a process exit code is the low byte). This is the ship gate BEFORE
//! the production flip (Task 5): a shape that miscompiles is caught here, not in production. Ground
//! truth (not Wimmer-vs-Wimmer) is the PRIMARY assertion so a shared-allocator or emit bug diverges;
//! where cheap the pre-flip reference `selectFunction` is also run as an old-vs-Wimmer cross-check.
//! qemu-i386 is the oracle: absent (the nix sandbox), every runner returns `error.SkipZigTest` and
//! NEVER asserts it ran.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

fn i32type(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

/// Compile `func` through the shared Wimmer allocator (mutates `func`: it splits critical edges) and
/// run it under qemu-i386 with integer args, returning the low byte of the result (eax).
fn runWimmer(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64) !u8 {
    var compiled = try isel.compileFunctionWimmerX86(allocator, func);
    defer compiled.deinit(allocator);
    return harness.runCodeInt(io, allocator, compiled.code, args, harness.qemu);
}

/// Run the pre-flip reference (`selectFunction`) and the Wimmer differential path on two freshly-built
/// copies of the same single-function shape for every input, asserting BOTH match the hand-computed
/// GROUND-TRUTH `expected` (mod 256, the process exit code). `build` takes only the allocator so each
/// side gets its own untouched function (the Wimmer path mutates the IR in place). The ground truth,
/// not `ref == got`, is the load-bearing assertion so a miscompile SHARED by both paths still fails.
fn expectIntMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, inputs: []const []const i64, expected: []const i64) !void {
    const allocator = std.testing.allocator;
    std.debug.assert(inputs.len == expected.len);
    for (inputs, expected) |args, exp| {
        var ref_func = try build(allocator);
        defer ref_func.deinit();
        var wim_func = try build(allocator);
        defer wim_func.deinit();

        const want: u8 = @truncate(@as(u64, @bitCast(exp)));
        const ref = harness.runFunc(io, allocator, &ref_func, args, harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = runWimmer(io, allocator, &wim_func, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(want, ref);
        try std.testing.expectEqual(want, got);
    }
}

/// True iff `needle` occurs as a contiguous byte subsequence of `hay`. Used by the structural probes
/// to confirm a specific emitted instruction sequence (the byte-addressability staging, a callee-saved
/// push) is actually present, a belt-and-suspenders check alongside the execution ground truth.
fn containsSeq(hay: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or needle.len > hay.len) return false;
    var i: usize = 0;
    const last = hay.len - needle.len;
    while (i <= last) : (i += 1) {
        if (std.mem.eql(u8, hay[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// True iff `code` contains a `mov [esp+disp32], reg` (store: `89 XX 24 <disp32>` with XX's mod/rm
/// bits = 10/100, i.e. `(XX & 0xC7) == 0x84`, any reg) whose disp32 is later matched by a
/// `mov reg2, [esp+disp32]` (reload: `8B YY 24 <same disp32>`, same mod/rm test). A live-range-split
/// value's store and its later reload need not use the SAME register (the shared Wimmer allocator is
/// free to home each segment differently, e.g. store through edx then reload into eax), so the probe
/// deliberately does not pin either register, only that a store and a LATER reload agree on the slot
/// disp32. That is the structural signature of a value ACTUALLY round-tripping through a spill slot,
/// as opposed to staying resident in a register the whole time (in which case no matching pair would
/// appear at all). The exact disp32 is a frame-layout detail this probe deliberately does not hardcode.
fn hasSpillRoundtrip(code: []const u8) bool {
    var i: usize = 0;
    while (i + 7 <= code.len) : (i += 1) {
        if (code[i] != 0x89 or (code[i + 1] & 0xC7) != 0x84 or code[i + 2] != 0x24) continue;
        const disp = code[i + 3 .. i + 7];
        var j: usize = i + 7;
        while (j + 7 <= code.len) : (j += 1) {
            if (code[j] == 0x8B and (code[j + 1] & 0xC7) == 0x84 and code[j + 2] == 0x24 and
                std.mem.eql(u8, code[j + 3 .. j + 7], disp)) return true;
        }
    }
    return false;
}

/// A compiled function plus the symbol it is linked as, for the multi-function call shapes.
const Named = struct { name: []const u8, code: []const u8, relocs: []const isel.Reloc };

fn startOf(fns: []const Named, starts: []const usize, symbol: []const u8) usize {
    for (fns, starts) |f, s| {
        if (std.mem.eql(u8, f.name, symbol)) return s;
    }
    unreachable; // every call relocation targets one of the linked functions
}

/// Concatenate `fns` (the entry function FIRST, at offset 0), patch every call relocation into an
/// intra-image rel32 by looking its target symbol up in the linked set, and run under qemu-i386 with
/// integer args. The general form of the x86_64 template's `linkRunInt`, extended to any number of
/// functions so a caller can reach a helper that itself calls a nested helper.
fn linkAndRun(io: std.Io, allocator: std.mem.Allocator, fns: []const Named, args: []const i64) !u8 {
    std.debug.assert(fns.len > 0);
    var total: usize = 0;
    for (fns) |f| total += f.code.len;
    const code = try allocator.alloc(u8, total);
    defer allocator.free(code);
    const starts = try allocator.alloc(usize, fns.len);
    defer allocator.free(starts);

    var cursor: usize = 0;
    for (fns, 0..) |f, i| {
        starts[i] = cursor;
        @memcpy(code[cursor .. cursor + f.code.len], f.code);
        cursor += f.code.len;
    }
    for (fns, starts) |f, base| {
        for (f.relocs) |reloc| {
            const target = startOf(fns, starts, reloc.symbol);
            const site = base + reloc.offset;
            const rel: i32 = @intCast(@as(i64, @intCast(target)) - @as(i64, @intCast(site + 4)));
            std.mem.writeInt(u32, code[site .. site + 4][0..4], @bitCast(rel), .little);
        }
    }
    return harness.runCodeInt(io, allocator, code, args, harness.qemu);
}

// ---------------------------------------------------------------------------
// 1. Straight-line integer arithmetic (stack args in, result out in eax).
// ---------------------------------------------------------------------------

/// f(a, b, c) = (a + b) * (b + c) - (a * c). A handful of simultaneously-live temps, no spilling: the
/// baseline that the cdecl stack-arg prologue, the int arithmetic arms, and the eax return all work.
fn buildStraightLine(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendBlockParam(entry, t);
    const ab = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const bc = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = c } });
    const ac = try func.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const prod = try func.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = ab, .rhs = bc } });
    const res = try func.appendInst(entry, t, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = ac } });
    func.setTerminator(entry, .{ .ret = res });
    return func;
}

test "wimmer-x86-32: straight-line int arithmetic matches ground truth" {
    const inputs = [_][]const i64{ &.{ 1, 2, 3 }, &.{ 0, 0, 0 }, &.{ -5, 7, -9 }, &.{ 10, -2, 3 }, &.{ 30, -1, 2 } };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const a: i32 = @intCast(in[0]);
        const b: i32 = @intCast(in[1]);
        const c: i32 = @intCast(in[2]);
        expected[i] = (a +% b) *% (b +% c) -% (a *% c);
    }
    try expectIntMatch(std.testing.io, buildStraightLine, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 2. Register pressure: more than 4 simultaneously-live ints force spilling.
// ---------------------------------------------------------------------------

const n_fan = 12;

/// f(n) = sum_k (n*(k+1) + k) for k in 0..n_fan. All n_fan terms are created before any is consumed, so
/// far more integer values are live at once than the 4 allocatable gpr registers (eax/ecx/edx/esi):
/// the shared allocator must spill. The reverse reduction reloads every operand, so a wrong spill or
/// reload diverges from the sum.
fn buildIntPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    var a: [n_fan]Value = undefined;
    for (0..n_fan) |k| {
        const coeff = try func.appendInst(entry, t, .{ .iconst = @intCast(k + 1) });
        const prod = try func.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = coeff } });
        a[k] = try func.appendArithImm(entry, t, .add, prod, @intCast(k));
    }
    var sum = a[n_fan - 1];
    var k: usize = n_fan - 1;
    while (k > 0) {
        k -= 1;
        sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = a[k] } });
    }
    func.setTerminator(entry, .{ .ret = sum });
    return func;
}

test "wimmer-x86-32: int register pressure spills and matches ground truth" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{7}, &.{-3}, &.{20}, &.{-11} };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const n: i32 = @intCast(in[0]);
        var s: i32 = 0;
        for (0..n_fan) |k| s +%= n *% @as(i32, @intCast(k + 1)) +% @as(i32, @intCast(k));
        expected[i] = s;
    }
    try expectIntMatch(std.testing.io, buildIntPressure, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 3. Live-range split: a value defined early, split across a pressured region, then reused.
// ---------------------------------------------------------------------------

const n_split = 10;

/// f(a) = (a * 7) + sum_k ((a + k) * (k + 1)) for k in 0..n_split. `keep = a * 7` is defined FIRST but
/// used only at the very END, so its live range spans the whole pressured reduction (n_split temps live
/// at once, past the 4 gpr registers). The shared allocator must SPLIT `keep`: home it, evict it to a
/// slot across the pressure, and reload it for the final add. A wrong split reads a stale slot.
fn buildLiveRangeSplit(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const keep = try func.appendArithImm(entry, t, .mul, a, 7);
    var terms: [n_split]Value = undefined;
    for (0..n_split) |k| {
        const ak = try func.appendArithImm(entry, t, .add, a, @intCast(k));
        terms[k] = try func.appendArithImm(entry, t, .mul, ak, @intCast(k + 1));
    }
    var sum = terms[n_split - 1];
    var k: usize = n_split - 1;
    while (k > 0) {
        k -= 1;
        sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = terms[k] } });
    }
    const res = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = keep, .rhs = sum } });
    func.setTerminator(entry, .{ .ret = res });
    return func;
}

test "wimmer-x86-32: a live-range split value is reused correctly" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{3}, &.{-4}, &.{9}, &.{-15} };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const a: i32 = @intCast(in[0]);
        var s: i32 = a *% 7;
        for (0..n_split) |k| s +%= (a +% @as(i32, @intCast(k))) *% @as(i32, @intCast(k + 1));
        expected[i] = s;
    }
    try expectIntMatch(std.testing.io, buildLiveRangeSplit, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 4. A value live across a call survives in callee-saved esi (not corrupted by the clobber).
// ---------------------------------------------------------------------------

/// The leaf-most callee `h(x) = x * 2`. A real function so the caller chain makes genuine calls.
fn buildMulTwo(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// The middle callee `g(x) = h(x) + 3`. It makes its OWN nested call to `h`, so at runtime `g` really
/// writes to the caller-saved registers (eax/ecx/edx) that the outer caller must not be relying on.
fn buildGCallsH(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const hr = try func.appendCall(b, t, "h", &.{x});
    const r = try func.appendArithImm(b, t, .add, hr, 3);
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// The caller `f(a, b)`: t = b + 1 (defined BEFORE the call), cr = g(a), return t + cr. `t` is live
/// ACROSS the call, whose clobber wipes every caller-saved register, so the shared allocator keeps `t`
/// in callee-saved esi (pushed in the prologue) rather than corrupting it. g(a) = 2a + 3, so the
/// result is (b + 1) + (2a + 3) = 2a + b + 4.
fn buildAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const carried = try func.appendArithImm(entry, t, .add, b, 1);
    const cr = try func.appendCall(entry, t, "g", &.{a});
    const r = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = carried, .rhs = cr } });
    func.setTerminator(entry, .{ .ret = r });
    return func;
}

test "wimmer-x86-32: a value live across a call survives in callee-saved esi" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const inputs = [_][]const i64{ &.{ 0, 0 }, &.{ 1, 2 }, &.{ 7, -3 }, &.{ -5, 10 }, &.{ 20, 21 } };

    // The callees are compiled through the SAME Wimmer entry as the caller (once, shared across inputs).
    // This matters: the Wimmer model makes esi callee-saved, a WHOLE-PROGRAM contract, so every callee
    // must save/restore esi. Compiling the callees through the old path (which treats esi as an
    // unsaved scratch) would clobber a caller value kept in esi across the call, so the Wimmer callees
    // model the post-flip world where every function honors the contract. `g` itself relocates a call
    // to `h`, resolved by `linkAndRun`.
    var h = try buildMulTwo(allocator);
    defer h.deinit();
    var h_c = try isel.compileFunctionWimmerX86(allocator, &h);
    defer h_c.deinit(allocator);
    var g = try buildGCallsH(allocator);
    defer g.deinit();
    var g_c = try isel.compileFunctionWimmerX86(allocator, &g);
    defer g_c.deinit(allocator);

    // The Wimmer caller MUST push esi in its prologue (proof it placed the carried value in a
    // callee-saved register rather than spilling). `push esi` is the single byte 0x56.
    var probe = try buildAcrossCall(allocator);
    defer probe.deinit();
    var probe_c = try isel.compileFunctionWimmerX86(allocator, &probe);
    defer probe_c.deinit(allocator);
    try std.testing.expect(probe_c.code.len >= 1 and probe_c.code[0] == 0x56);

    for (inputs) |args| {
        var ref_caller = try buildAcrossCall(allocator);
        defer ref_caller.deinit();
        var wim_caller = try buildAcrossCall(allocator);
        defer wim_caller.deinit();

        var ref_c = isel.compile(allocator, &ref_caller) catch |e| switch (e) {
            error.Unsupported => return error.SkipZigTest,
            else => return e,
        };
        defer ref_c.deinit(allocator);
        var wim_c = try isel.compileFunctionWimmerX86(allocator, &wim_caller);
        defer wim_c.deinit(allocator);

        const a: i32 = @intCast(args[0]);
        const b: i32 = @intCast(args[1]);
        const want: u8 = @truncate(@as(u32, @bitCast((2 *% a) +% b +% 4)));

        const ref_fns = [_]Named{
            .{ .name = "f", .code = ref_c.code, .relocs = ref_c.relocs },
            .{ .name = "g", .code = g_c.code, .relocs = g_c.relocs },
            .{ .name = "h", .code = h_c.code, .relocs = h_c.relocs },
        };
        const wim_fns = [_]Named{
            .{ .name = "f", .code = wim_c.code, .relocs = wim_c.relocs },
            .{ .name = "g", .code = g_c.code, .relocs = g_c.relocs },
            .{ .name = "h", .code = h_c.code, .relocs = h_c.relocs },
        };
        const ref = linkAndRun(io, allocator, &ref_fns, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = linkAndRun(io, allocator, &wim_fns, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(want, ref);
        try std.testing.expectEqual(want, got);
    }
}

// ---------------------------------------------------------------------------
// 5. THE RISK (R1): a byte-addressable boolean forced to live in esi, then consumed.
//
// esi has no low-byte form: a `setcc esi` would wrongly encode DH (edx's high byte). The icmp arm must
// instead stage the boolean through the reserved byte-addressable ebx scratch and move it into esi.
// This shape defines a boolean BEFORE a call and consumes it AFTER, so the boolean is live across the
// call and the ONLY callee-saved register (esi) is where the allocator keeps it. If the emit wrote a
// non-existent esi low byte the boolean would be corrupted and the branch would take the wrong arm, so
// running BOTH a true and a false input is the correctness proof.
// ---------------------------------------------------------------------------

/// f(a, b): cond = a < b (a BOOLEAN), cr = g(a) = 2a + 3, then `if cond` return cr + 10 else cr + 20.
/// `cond` is defined before the call and used at the terminator after it, so it is live across the
/// call and homed in callee-saved esi (verified structurally below). The consumed boolean selects the
/// arm, so a corrupted esi low byte flips the result.
fn buildEsiBoolean(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    const cr = try func.appendCall(entry, t, "g", &.{a});
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const rt = try func.appendArithImm(then_b, t, .add, cr, 10);
    func.setTerminator(then_b, .{ .ret = rt });
    const re = try func.appendArithImm(else_b, t, .add, cr, 20);
    func.setTerminator(else_b, .{ .ret = re });
    return func;
}

test "wimmer-x86-32: a boolean forced into esi stages through ebx and selects the right arm" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Inputs exercising both the true (a < b) and the false (a >= b) branch.
    const inputs = [_][]const i64{ &.{ 1, 2 }, &.{ 5, 2 }, &.{ -3, 0 }, &.{ 4, 4 }, &.{ 10, 11 } };

    var h = try buildMulTwo(allocator);
    defer h.deinit();
    var h_c = try isel.compileFunctionWimmerX86(allocator, &h);
    defer h_c.deinit(allocator);
    var g = try buildGCallsH(allocator);
    defer g.deinit();
    var g_c = try isel.compileFunctionWimmerX86(allocator, &g);
    defer g_c.deinit(allocator);

    // Structural proof R1 is exercised: the boolean is materialized in ebx (`movzx ebx, bl` = 0F B6 DB)
    // and moved into esi (`mov esi, ebx` = 89 DE). That the boolean is homed in esi at all is proven by
    // the `push esi` prologue byte (0x56). A regression that setcc'd esi directly would drop this
    // staging pair, and a regression that spilled instead would drop the esi push.
    var probe = try buildEsiBoolean(allocator);
    defer probe.deinit();
    var probe_c = try isel.compileFunctionWimmerX86(allocator, &probe);
    defer probe_c.deinit(allocator);
    try std.testing.expect(probe_c.code.len >= 1 and probe_c.code[0] == 0x56);
    try std.testing.expect(containsSeq(probe_c.code, &.{ 0x0F, 0xB6, 0xDB, 0x89, 0xDE }));

    for (inputs) |args| {
        var wim_caller = try buildEsiBoolean(allocator);
        defer wim_caller.deinit();
        var wim_c = try isel.compileFunctionWimmerX86(allocator, &wim_caller);
        defer wim_c.deinit(allocator);

        const a: i32 = @intCast(args[0]);
        const b: i32 = @intCast(args[1]);
        const cr: i32 = (2 *% a) +% 3;
        const exp: i32 = if (a < b) cr +% 10 else cr +% 20;
        const want: u8 = @truncate(@as(u32, @bitCast(exp)));

        const wim_fns = [_]Named{
            .{ .name = "f", .code = wim_c.code, .relocs = wim_c.relocs },
            .{ .name = "g", .code = g_c.code, .relocs = g_c.relocs },
            .{ .name = "h", .code = h_c.code, .relocs = h_c.relocs },
        };
        const got = linkAndRun(io, allocator, &wim_fns, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(want, got);
    }
}

// ---------------------------------------------------------------------------
// 6. A boolean spilled to a slot, then reloaded and consumed (the spilled-boolean byte path).
//
// When the icmp result's home is a spill slot the setcc still stages through ebx (byte-addressable),
// then the clean 0/1 is stored to the slot and reloaded for its consumer. This covers the Task 1 Minor
// (that path was only structurally exercised for the esi home). The boolean is defined first and
// consumed last, across n_bpress live temporaries, so its far next-use makes the allocator spill it.
// ---------------------------------------------------------------------------

const n_bpress = 9;

/// f(a, b): cond = a < b (a BOOLEAN), then n_bpress live temps summed into acc, then `if cond` return
/// acc + 1 else acc + 2. `cond` is live from the first instruction to the terminator across the whole
/// pressured region, so the allocator spills it to a slot (verified below), and it must reload
/// correctly for the branch. acc = sum_k (a + k*b) for k in 1..=n_bpress.
fn buildSpilledBoolean(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    var temps: [n_bpress]Value = undefined;
    for (0..n_bpress) |k| {
        const kb = try func.appendArithImm(entry, t, .mul, b, @intCast(k + 1));
        temps[k] = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = kb } });
    }
    var acc = temps[n_bpress - 1];
    var k: usize = n_bpress - 1;
    while (k > 0) {
        k -= 1;
        acc = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = temps[k] } });
    }
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{acc} }, .{ .target = else_b, .args = &.{acc} });
    const p_then = try func.appendBlockParam(then_b, t);
    const rt = try func.appendArithImm(then_b, t, .add, p_then, 1);
    func.setTerminator(then_b, .{ .ret = rt });
    const p_else = try func.appendBlockParam(else_b, t);
    const re = try func.appendArithImm(else_b, t, .add, p_else, 2);
    func.setTerminator(else_b, .{ .ret = re });
    return func;
}

test "wimmer-x86-32: a spilled boolean reloads correctly and selects the right arm" {
    // Structural probe FIRST (the "verified below" this shape's doc comment promises): confirm the
    // boolean actually spills to a slot and reloads from it, mirroring how test 5 (the esi-boolean
    // shape) probes its own staging bytes. Without this, allocator heuristics moving the boolean into
    // a register instead would leave the test passing without ever exercising the reload path.
    var probe = try buildSpilledBoolean(std.testing.allocator);
    defer probe.deinit();
    var probe_c = try isel.compileFunctionWimmerX86(std.testing.allocator, &probe);
    defer probe_c.deinit(std.testing.allocator);
    try std.testing.expect(hasSpillRoundtrip(probe_c.code));

    const inputs = [_][]const i64{ &.{ 1, 2 }, &.{ 5, 1 }, &.{ -3, 2 }, &.{ 4, 4 }, &.{ 2, 3 }, &.{ 8, 1 } };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const a: i32 = @intCast(in[0]);
        const b: i32 = @intCast(in[1]);
        var acc: i32 = 0;
        for (0..n_bpress) |k| acc +%= a +% b *% @as(i32, @intCast(k + 1));
        expected[i] = if (a < b) acc +% 1 else acc +% 2;
    }
    try expectIntMatch(std.testing.io, buildSpilledBoolean, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 7. div/rem execution with the divisor pressured toward eax/edx (the Task 1 div guard).
//
// idiv/div destroy eax+edx, so a divisor allocated to eax or edx must be copied out first. The guard
// copies it to edi before `mov eax, dividend; cdq`. A missing guard computes dividend/dividend. The
// shape divides under pressure so the divisor can land in a clobbered register, and negative operands
// pin the signed truncation direction.
// ---------------------------------------------------------------------------

/// f(a, b, c, d): with c and d kept live across the divide (so eax/edx/ecx tend to be occupied and the
/// divisor b can be forced into a clobber register), q = a / b, r = a % b, return q + r*8 + c*16 + d.
/// Signed div/rem truncate toward zero. Every input has b != 0.
fn buildDivGuard(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const c = try func.appendBlockParam(entry, t);
    const d = try func.appendBlockParam(entry, t);
    const q = try func.appendInst(entry, t, .{ .arith = .{ .op = .div, .lhs = a, .rhs = b } });
    const rem = try func.appendInst(entry, t, .{ .arith = .{ .op = .rem, .lhs = a, .rhs = b } });
    const r8 = try func.appendArithImm(entry, t, .mul, rem, 8);
    const c16 = try func.appendArithImm(entry, t, .mul, c, 16);
    const s1 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = q, .rhs = r8 } });
    const s2 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = c16 } });
    const s3 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = d } });
    func.setTerminator(entry, .{ .ret = s3 });
    return func;
}

test "wimmer-x86-32: div/rem with a pressured divisor matches ground truth" {
    const inputs = [_][]const i64{ &.{ 100, 7, 1, 2 }, &.{ 40, 6, -1, 3 }, &.{ -50, 8, 2, -4 }, &.{ 123, 11, 0, 5 }, &.{ -30, -4, 3, 1 } };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const a: i32 = @intCast(in[0]);
        const b: i32 = @intCast(in[1]);
        const c: i32 = @intCast(in[2]);
        const d: i32 = @intCast(in[3]);
        expected[i] = @divTrunc(a, b) +% (@rem(a, b) *% 8) +% (c *% 16) +% d;
    }
    try expectIntMatch(std.testing.io, buildDivGuard, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 8. shl/shr execution with the shift lhs pressured toward ecx (the Task 1 shift guard).
//
// A variable shift count goes in ecx, so a shift lhs allocated to ecx must be copied out first. The
// guard copies it to ebx before ecx is overwritten with the count. A missing guard computes
// count << count. The shape shifts under pressure so the lhs can land in ecx, and mixes signed and
// unsigned right shifts to distinguish sar from shr.
// ---------------------------------------------------------------------------

/// f(x, y, z, w): keep z and w live across the shifts, s = x << y, u = x >> y (arithmetic, signed x),
/// return s + u*4 + z*32 + w. y is a small nonnegative count in every input.
fn buildShiftGuard(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const y = try func.appendBlockParam(entry, t);
    const z = try func.appendBlockParam(entry, t);
    const w = try func.appendBlockParam(entry, t);
    const s = try func.appendInst(entry, t, .{ .arith = .{ .op = .shl, .lhs = x, .rhs = y } });
    const u = try func.appendInst(entry, t, .{ .arith = .{ .op = .shr, .lhs = x, .rhs = y } });
    const u_scaled = try func.appendArithImm(entry, t, .mul, u, 4);
    const z32 = try func.appendArithImm(entry, t, .mul, z, 32);
    const s1 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = u_scaled } });
    const s2 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = z32 } });
    const s3 = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = w } });
    func.setTerminator(entry, .{ .ret = s3 });
    return func;
}

test "wimmer-x86-32: shl/shr with a pressured lhs matches ground truth" {
    const inputs = [_][]const i64{ &.{ 3, 4, 1, 2 }, &.{ -8, 1, 2, -1 }, &.{ 100, 2, 0, 5 }, &.{ -17, 3, 1, 4 }, &.{ 7, 0, 3, 2 } };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const x: i32 = @intCast(in[0]);
        const y: u5 = @intCast(in[1]);
        const z: i32 = @intCast(in[2]);
        const w: i32 = @intCast(in[3]);
        const s: i32 = x << y;
        const u: i32 = x >> y; // arithmetic right shift on a signed value (sar)
        expected[i] = s +% (u *% 4) +% (z *% 32) +% w;
    }
    try expectIntMatch(std.testing.io, buildShiftGuard, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 9. A split cdecl stack-arg param: its first segment is established from its stack slot.
//
// The entry_fixed set is empty (cdecl args arrive on the stack), so the prologue loads each stack
// argument into its allocator-chosen home. When that param's live range splits under pressure, the
// FIRST segment must be seeded from the stack-arg load. Here the raw param `a` is used only at the very
// end, across the pressured reduction, so it splits and its opening segment comes straight from the
// stack slot. A wrong seed makes the final add read garbage.
// ---------------------------------------------------------------------------

const n_argsplit = 10;

/// f(a, b): sum_k ((b + k) * (k + 1)) for k in 0..n_argsplit, then add the raw param `a` LAST. `a` is
/// live from entry to the final add, spanning the pressured reduction (n_argsplit temps at once), so it
/// splits with its first segment loaded from `[esp + arg0]`. result = a + sum.
fn buildSplitStackArg(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    var terms: [n_argsplit]Value = undefined;
    for (0..n_argsplit) |k| {
        const bk = try func.appendArithImm(entry, t, .add, b, @intCast(k));
        terms[k] = try func.appendArithImm(entry, t, .mul, bk, @intCast(k + 1));
    }
    var sum = terms[n_argsplit - 1];
    var k: usize = n_argsplit - 1;
    while (k > 0) {
        k -= 1;
        sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = terms[k] } });
    }
    const res = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = sum } });
    func.setTerminator(entry, .{ .ret = res });
    return func;
}

test "wimmer-x86-32: a split cdecl stack-arg param is seeded from its slot" {
    const inputs = [_][]const i64{ &.{ 0, 0 }, &.{ 5, 1 }, &.{ -7, 3 }, &.{ 100, -2 }, &.{ -33, 4 } };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const a: i32 = @intCast(in[0]);
        const b: i32 = @intCast(in[1]);
        var sum: i32 = 0;
        for (0..n_argsplit) |k| sum +%= (b +% @as(i32, @intCast(k))) *% @as(i32, @intCast(k + 1));
        expected[i] = a +% sum;
    }
    try expectIntMatch(std.testing.io, buildSplitStackArg, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 10. A critical edge: the block-param value needs a move on the split edge.
//
// entry has two successors (then_b, merge) and merge has two predecessors (then_b, entry), so the
// direct entry->merge edge is CRITICAL. `splitCriticalEdges` inserts a block on it, and the shared
// resolver emits the merge-param move there. A wrong edge move drops the entry-supplied value.
// ---------------------------------------------------------------------------

/// f(a): cond = a > 0; on true jump then_b, on false jump merge directly with the value a + 100 (the
/// critical entry->merge edge). then_b computes t = a * 2 and jumps merge(t). merge(p) returns p + a.
/// So p = (a > 0) ? 2a : a + 100, and the result is p + a: 3a when a > 0, else 2a + 100.
fn buildCriticalEdge(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = zero } });
    const a100 = try func.appendArithImm(entry, t, .add, a, 100);
    try func.appendIf(entry, cond, .{ .target = then_b, .args = &.{} }, .{ .target = merge, .args = &.{a100} });
    const ta = try func.appendArithImm(then_b, t, .mul, a, 2);
    try func.setJump(then_b, merge, &.{ta});
    const p = try func.appendBlockParam(merge, t);
    const r = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = a } });
    func.setTerminator(merge, .{ .ret = r });
    return func;
}

test "wimmer-x86-32: a critical-edge block-param move matches ground truth" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{-1}, &.{9}, &.{-20}, &.{40} };
    var expected: [inputs.len]i64 = undefined;
    for (inputs, 0..) |in, i| {
        const a: i32 = @intCast(in[0]);
        const p: i32 = if (a > 0) a *% 2 else a +% 100;
        expected[i] = p +% a;
    }
    try expectIntMatch(std.testing.io, buildCriticalEdge, &inputs, &expected);
}

// ---------------------------------------------------------------------------
// 11. Address fold under pressure combined with a call (fold base kept live past the clobber).
//
// Task 3 already covers fold-under-pressure. This adds the cheap fold+call combination: a foldable
// load off an alloca base whose add is DCE'd by `applyFoldRewriteX86`, with the base live across a
// call, so it must survive in callee-saved esi for the folded `[base + disp]` to read the right slot.
// Compiled through the fold-aware Wimmer entry so the fold actually fires.
// ---------------------------------------------------------------------------

/// f(a): buf0, buf1 two adjacent i32 allocas (buf0 + 4 == buf1). Store a into buf1, form the DEAD add
/// pl = buf0 + 4 (folds to buf1), call g(a) = 2a + 3, then a FOLDED load of pl and return loaded + cr.
/// buf0 is live across the call (used by the folded load after it), so it must survive in esi. Result
/// is a + (2a + 3) = 3a + 3.
fn buildFoldAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try i32type(&func);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const buf0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const buf1 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(entry, a, buf1); // buf1 = a
    const pl = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 4 } }); // = buf1
    const cr = try func.appendCall(entry, t, "g", &.{a});
    const loaded = try func.appendInst(entry, t, .{ .load = .{ .ptr = pl } }); // folded [buf0 + 4] = a
    const r = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = loaded, .rhs = cr } });
    func.setTerminator(entry, .{ .ret = r });
    return func;
}

test "wimmer-x86-32: address fold with the base live across a call matches ground truth" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{7}, &.{-3}, &.{20} };

    var h = try buildMulTwo(allocator);
    defer h.deinit();
    var h_c = try isel.compileFunctionWimmerX86(allocator, &h);
    defer h_c.deinit(allocator);
    var g = try buildGCallsH(allocator);
    defer g.deinit();
    var g_c = try isel.compileFunctionWimmerX86(allocator, &g);
    defer g_c.deinit(allocator);

    for (inputs) |args| {
        var wim_caller = try buildFoldAcrossCall(allocator);
        defer wim_caller.deinit();
        var wim_c = try isel.compileFunctionWimmerX86Fold(allocator, &wim_caller);
        defer wim_c.deinit(allocator);

        const a: i32 = @intCast(args[0]);
        const want: u8 = @truncate(@as(u32, @bitCast((3 *% a) +% 3)));

        const wim_fns = [_]Named{
            .{ .name = "f", .code = wim_c.code, .relocs = wim_c.relocs },
            .{ .name = "g", .code = g_c.code, .relocs = g_c.relocs },
            .{ .name = "h", .code = h_c.code, .relocs = h_c.relocs },
        };
        const got = linkAndRun(io, allocator, &wim_fns, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(want, got);
    }
}
