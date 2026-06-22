//! Lowers a graphics Vulcan IR function to TGSI text.
//!
//! TGSI (Tungsten Graphics Shader Infrastructure) is Gallium/virgl's textual
//! shader form: `tgsi_text_translate` parses exactly this text, so the emitted
//! bytes feed straight to a virgl shader-create command. GPU-paravirtual
//! counterpart of the nvidia SASS backend: rather than selecting a native ISA, the
//! graphics IR is rendered as TGSI declarations and opcodes.
//!
//! The graphics IR carries the same `vulcan.gpu` attribute tags the SPIR-V
//! graphics lowering produces and the nvidia isel reads:
//!   * entry-block params tagged `attr` = ATTR_GENERIC0 + loc*0x10 + comp*4
//!     (a vertex/fragment input attribute slot), one per vector component.
//!   * output stores whose pointer is a tag-carrier iconst tagged either
//!     `out_attr` (a vertex output: ATTR_POSITION for the clip-space position,
//!     ATTR_GENERIC0 + loc*0x10 + comp*4 for a varying) or `color_out` (a
//!     fragment render-target color component index 0..3).
//!
//! TGSI mapping:
//!   VS inputs   -> `DCL IN[loc]`
//!   VS position -> `DCL OUT[n], POSITION`
//!   VS varying  -> `DCL OUT[n], GENERIC[loc]`
//!   FS input    -> `DCL IN[n], GENERIC[loc], PERSPECTIVE`
//!   FS color    -> `DCL OUT[0], COLOR`
//! plus one `MOV` per output register copying its source input register, then
//! `END`.
//!
//! Only the passthrough class (output <- input, per register) is lowered. A shader
//! with arithmetic in the body returns `error.Unsupported`. General TGSI expression
//! lowering is future work.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// Attribute slot bases (shared with the nvidia encoder's interface convention).
const ATTR_POSITION: u32 = 0x70;
const ATTR_GENERIC0: u32 = 0x80;

/// The shader stage. Selected from the function's `vulcan.gpu` "stage" tag the
/// SPIR-V lowering sets (vertex / fragment). Compute has no TGSI form here.
pub const Stage = enum { vertex, fragment };

/// A decoded graphics input attribute: which IN register (the location) and which
/// of its components (x=0..w=3) a scalar param feeds.
const InputSlot = struct { reg: u32, comp: u8 };

/// A decoded graphics output: a TGSI OUT register plus its semantic.
const OutKind = enum { position, generic, color };
const OutputSlot = struct { kind: OutKind, location: u32, comp: u8 };

/// The four TGSI swizzle/writemask channel letters, by component index.
const channel = [4]u8{ 'x', 'y', 'z', 'w' };

/// Lower a graphics IR `func` to TGSI text. The text is NUL-terminated and
/// dword-padded (virglrenderer reads it as a token-aligned shader blob). The
/// caller owns the returned slice (free with `allocator`).
pub fn lower(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    const stage = stageOf(func) orelse return error.Unsupported;
    if (func.blockCount() != 1) return error.Unsupported; // single straight-line block

    const entry: Block = @enumFromInt(0);
    const params = func.blockParams(entry);

    // Inputs: each tagged param is one scalar attribute slot. Map each input param
    // Value to its (register, component). The register is the location decoded from
    // its `attr` byte slot. Track which locations are present to know how many IN[]
    // to declare.
    var in_of = std.AutoHashMapUnmanaged(Value, InputSlot){};
    defer in_of.deinit(allocator);
    var in_present = [_]bool{false} ** 32;
    for (params) |p| {
        const slot = attrTag(func, p, "attr") orelse return error.Unsupported;
        if (slot < ATTR_GENERIC0) return error.Unsupported; // only generic inputs
        const off = slot - ATTR_GENERIC0;
        const reg = off / 0x10;
        const comp: u8 = @intCast((off % 0x10) / 4);
        if (reg >= in_present.len) return error.Unsupported;
        in_present[reg] = true;
        try in_of.put(allocator, p, .{ .reg = reg, .comp = comp });
    }

    // Outputs: each tagged store is one output component. A passthrough store's
    // value is an input param. Record, per output register, which input register
    // each component copies. Components within a register must share one source
    // register (a register-wide MOV).
    const MoveSrc = struct { src_reg: u32 = 0, mask: u4 = 0, present: bool = false, kind: OutKind = .generic, location: u32 = 0 };
    var out_regs = [_]MoveSrc{.{}} ** 32;
    var out_count: u32 = 0;

    for (func.blockInsts(entry)) |inst| {
        switch (func.opcode(inst)) {
            .iconst => {
                // A tag-carrier iconst (the output slot), inspected via its store.
            },
            .store => |st| {
                const out = outputSlot(func, st.ptr) orelse return error.Unsupported;
                const reg = outReg(out);
                if (reg >= out_regs.len) return error.Unsupported;
                // The stored value must be a passthrough of an input scalar.
                const src = in_of.get(st.value) orelse return error.Unsupported;
                const e = &out_regs[reg];
                if (!e.present) {
                    e.* = .{ .src_reg = src.reg, .mask = 0, .present = true, .kind = out.kind, .location = out.location };
                    out_count += 1;
                } else if (e.src_reg != src.reg or e.kind != out.kind) {
                    return error.Unsupported; // mixed source register, not a plain MOV
                }
                if (out.comp != src.comp) return error.Unsupported; // component permutation, not a plain MOV
                e.mask |= @as(u4, 1) << @intCast(out.comp);
            },
            .load => {}, // an OpLoad of an input lowers to nothing (the param itself)
            else => return error.Unsupported, // arithmetic body: general lowering is future work
        }
    }

    var buf = Writer{ .allocator = allocator };
    defer buf.list.deinit(allocator);

    try buf.put(if (stage == .vertex) "VERT\n" else "FRAG\n");

    // Input declarations. tgsi_text_translate wants ascending IN[n] declarations.
    // Vertex inputs are vertex attributes (bare IN[n]). Fragment inputs are
    // perspective-interpolated generic varyings.
    {
        var reg: u32 = 0;
        while (reg < in_present.len) : (reg += 1) {
            if (!in_present[reg]) continue;
            if (stage == .vertex) {
                try buf.print("DCL IN[{d}]\n", .{reg});
            } else {
                try buf.print("DCL IN[{d}], GENERIC[{d}], PERSPECTIVE\n", .{ reg, reg });
            }
        }
    }

    // Output declarations.
    {
        var reg: u32 = 0;
        while (reg < out_regs.len) : (reg += 1) {
            const e = out_regs[reg];
            if (!e.present) continue;
            switch (e.kind) {
                .position => try buf.print("DCL OUT[{d}], POSITION\n", .{reg}),
                .generic => try buf.print("DCL OUT[{d}], GENERIC[{d}]\n", .{ reg, e.location }),
                .color => try buf.print("DCL OUT[{d}], COLOR\n", .{reg}),
            }
        }
    }

    // Body: one MOV per output register, then END.
    {
        var line: u32 = 1;
        var reg: u32 = 0;
        while (reg < out_regs.len) : (reg += 1) {
            const e = out_regs[reg];
            if (!e.present) continue;
            // A full-width MOV when every channel is written, otherwise a masked
            // write naming exactly the destination channels (and the same source
            // channels, since this is a component-preserving passthrough).
            if (e.mask == 0b1111) {
                try buf.print("  {d}: MOV OUT[{d}], IN[{d}]\n", .{ line, reg, e.src_reg });
            } else {
                try buf.print("  {d}: MOV OUT[{d}].", .{ line, reg });
                try buf.putMask(e.mask);
                try buf.print(", IN[{d}].", .{e.src_reg});
                try buf.putMask(e.mask);
                try buf.put("\n");
            }
            line += 1;
        }
        try buf.print("  {d}: END\n", .{line});
    }

    // NUL-terminate and dword-pad: virglrenderer reads the shader text as a
    // token-aligned blob, so the byte length must round up to a multiple of 4.
    try buf.putByte(0);
    while (buf.list.items.len % 4 != 0) try buf.putByte(0);

    return buf.list.toOwnedSlice(allocator);
}

/// Accumulating text builder over an ArrayList. Keeps TGSI emission self-contained.
const Writer = struct {
    allocator: std.mem.Allocator,
    list: std.ArrayList(u8) = .empty,

    fn put(self: *Writer, s: []const u8) Error!void {
        try self.list.appendSlice(self.allocator, s);
    }
    fn putByte(self: *Writer, b: u8) Error!void {
        try self.list.append(self.allocator, b);
    }
    fn print(self: *Writer, comptime fmt: []const u8, args: anytype) Error!void {
        var tmp: [64]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, fmt, args) catch return error.Unsupported;
        try self.put(s);
    }
    /// Emit the swizzle channel letters for a writemask (mask 0b0011 -> "xy").
    fn putMask(self: *Writer, mask: u4) Error!void {
        var c: u3 = 0;
        while (c < 4) : (c += 1) {
            if (mask & (@as(u4, 1) << @intCast(c)) != 0) try self.putByte(channel[c]);
        }
    }
};

/// The output register index a slot maps to. POSITION/varyings keep their slot's
/// location. The fragment color is OUT[0].
fn outReg(out: OutputSlot) u32 {
    return switch (out.kind) {
        .position => 0,
        .generic => out.location + 1, // OUT[0] is POSITION, varyings follow
        .color => 0,
    };
}

/// Decode an output store pointer's tag into an OutputSlot, or null if untagged.
fn outputSlot(func: *const Function, ptr: Value) ?OutputSlot {
    if (attrTag(func, ptr, "color_out")) |comp| {
        return .{ .kind = .color, .location = 0, .comp = @intCast(comp) };
    }
    if (attrTag(func, ptr, "out_attr")) |slot| {
        if (slot >= ATTR_POSITION and slot < ATTR_GENERIC0) {
            return .{ .kind = .position, .location = 0, .comp = @intCast((slot - ATTR_POSITION) / 4) };
        }
        if (slot >= ATTR_GENERIC0) {
            const off = slot - ATTR_GENERIC0;
            return .{ .kind = .generic, .location = off / 0x10, .comp = @intCast((off % 0x10) / 4) };
        }
    }
    return null;
}

/// The shader stage tagged on the function by the SPIR-V graphics lowering.
fn stageOf(func: *const Function) ?Stage {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "stage")) {
            return switch (c.value) {
                .string => |s| if (std.mem.eql(u8, s, "vertex"))
                    .vertex
                else if (std.mem.eql(u8, s, "fragment"))
                    .fragment
                else
                    null,
                else => null,
            };
        },
        else => {},
    };
    return null;
}

/// A `vulcan.gpu` integer attribute named `key` on value `v`, or null.
fn attrTag(func: *const Function, v: Value, key: []const u8) ?u32 {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) {
            return switch (c.value) {
                .int => |n| @intCast(n),
                else => null,
            };
        },
        else => {},
    };
    return null;
}

const testing = std.testing;

test "lower a passthrough vertex shader to TGSI (position + color varying)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "vertex" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // Two vec4 inputs: position (loc 0) and color (loc 1), scalarized to 4 params
    // each, tagged with their attribute byte slots.
    var pos_in: [4]Value = undefined;
    var col_in: [4]Value = undefined;
    inline for (0..4) |c| {
        pos_in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = pos_in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    inline for (0..4) |c| {
        col_in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = col_in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + 0x10 + c * 4 } } });
    }
    // Store position -> OUT[0] POSITION, color -> OUT[1] GENERIC[0].
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_POSITION + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_POSITION + c * 4 } } });
        try func.appendStore(b, pos_in[c], ptr);
    }
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(ATTR_GENERIC0 + c * 4) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
        try func.appendStore(b, col_in[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    const want =
        "VERT\n" ++
        "DCL IN[0]\n" ++
        "DCL IN[1]\n" ++
        "DCL OUT[0], POSITION\n" ++
        "DCL OUT[1], GENERIC[0]\n" ++
        "  1: MOV OUT[0], IN[0]\n" ++
        "  2: MOV OUT[1], IN[1]\n" ++
        "  3: END\n";
    try testing.expectStringStartsWith(tgsi, want);
    try testing.expectEqual(@as(u8, 0), tgsi[want.len]); // NUL-terminated
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4); // dword padded
}

test "lower a passthrough fragment shader to TGSI (interpolated varying -> color)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = "fragment" } } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // One vec4 input varying (loc 0), scalarized. The color output is the four
    // color_out components (R0..R3) copied straight from it.
    var col_in: [4]Value = undefined;
    inline for (0..4) |c| {
        col_in[c] = try func.appendBlockParam(b, f32_t);
        try func.addAttr(.{ .value = col_in[c] }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = ATTR_GENERIC0 + c * 4 } } });
    }
    inline for (0..4) |c| {
        const ptr = try func.appendInst(b, i32_t, .{ .iconst = @intCast(c) });
        try func.addAttr(.{ .value = ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, col_in[c], ptr);
    }
    func.setTerminator(b, .{ .ret = null });

    const tgsi = try lower(allocator, &func);
    defer allocator.free(tgsi);

    const want =
        "FRAG\n" ++
        "DCL IN[0], GENERIC[0], PERSPECTIVE\n" ++
        "DCL OUT[0], COLOR\n" ++
        "  1: MOV OUT[0], IN[0]\n" ++
        "  2: END\n";
    try testing.expectStringStartsWith(tgsi, want);
    try testing.expectEqual(@as(u8, 0), tgsi[want.len]);
    try testing.expectEqual(@as(usize, 0), tgsi.len % 4);
}

test {
    testing.refAllDecls(@This());
}
