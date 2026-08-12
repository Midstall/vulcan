//! `global_addr` isel: the IR op that materializes a global symbol's address. This
//! test covers instruction selection only. It asserts the shape, an `adrp`+`add`
//! pair, each with its own relocation, both naming the symbol, and that isel does
//! not hit `error.Unsupported`. A later step patches the relocation values. Both
//! immediates here are zero by construction.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const encode = @import("../encode.zig");

const Function = ir.function.Function;

/// `fn entry() i8 { let g = &sym; return *(i8*)g; }` is the minimal shape that reaches
/// the `.global_addr` arm. A bare `ret` of the address alone would let the allocator
/// discard it as dead before emission. Loading through it keeps it live.
fn buildGlobalAddr(allocator: std.mem.Allocator, sym: []const u8) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const blk = try func.appendBlock();
    const g = try func.appendGlobalAddr(blk, ptr_t, sym);
    const r = try func.appendInst(blk, i8_t, .{ .load = .{ .ptr = g } });
    func.setTerminator(blk, .{ .ret = ir.function.Ret.one(r) });
    return func;
}

test "aarch64 selects global_addr into adrp+add with two relocs" {
    const allocator = std.testing.allocator;

    var func = try buildGlobalAddr(allocator, "g");
    defer func.deinit();

    var compiled = try isel.compileFunction(allocator, &func, .{});
    defer compiled.deinit(allocator);

    // Exactly one adrp_pg relocation and one add_pgoff relocation, both naming "g".
    var pg_offset: ?usize = null;
    var pgoff_offset: ?usize = null;
    for (compiled.relocs) |reloc| {
        switch (reloc.kind) {
            .adrp_pg => {
                try std.testing.expect(pg_offset == null); // exactly one
                pg_offset = reloc.offset;
                try std.testing.expectEqualStrings("g", reloc.symbol);
            },
            .add_pgoff => {
                try std.testing.expect(pgoff_offset == null); // exactly one
                pgoff_offset = reloc.offset;
                try std.testing.expectEqualStrings("g", reloc.symbol);
            },
            .call, .got_pg, .got_lo12 => unreachable, // no calls or GOT references in this function
        }
    }
    const pg = pg_offset.?;
    const pgoff = pgoff_offset.?;

    // The add's word immediately follows the adrp's word, since isel emits them back
    // to back, and both target the same register.
    try std.testing.expectEqual(pg + 1, pgoff);
    const adrp_word = compiled.code[pg];
    const add_word = compiled.code[pgoff];
    const adrp_rd: u5 = @truncate(adrp_word);
    const add_rd: u5 = @truncate(add_word);
    const add_rn: u5 = @truncate(add_word >> 5);
    try std.testing.expectEqual(adrp_rd, add_rd);
    try std.testing.expectEqual(adrp_rd, add_rn);
    // The words really are a zero-immediate adrp/add pair, not just relocations
    // claiming so. Re-derive the expected words from the encoders with the extracted
    // rd/rn.
    const adrp_reg: encode.Reg = @enumFromInt(adrp_rd);
    try std.testing.expectEqual(encode.adrp(adrp_reg, 0), adrp_word);
    try std.testing.expectEqual(encode.addImm64(adrp_reg, @enumFromInt(add_rn), 0), add_word);
}
