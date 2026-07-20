//! addr_hi_lo fusion (Task 7), the LAST `FuseKind`. HONEST framing: unlike `fuse_cmp_branch` /
//! `fuse_shift_add`, this kind needs NO new transform. The `.global_addr` arm (isel.zig) already
//! emits `auipc rd, %pcrel_hi(sym)` immediately followed by `addi rd, rd, %pcrel_lo(.Lhi)` back to
//! back, by construction, on every path - there is nothing to fold and nothing to decline. This
//! test is a regression guard: it decodes a compiled `global_addr` and confirms the auipc/addi pair
//! really is adjacent (the invariant `caps.fuse_addr_hi_lo`'s assert in isel.zig depends on), and
//! that the flag is a pure observer - `fuse_addr_hi_lo = true` and `= false` compile the SAME
//! function to byte-identical code.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");

const Function = ir.function.Function;

/// `fn entry() ptr { return &sym; }` - the minimal shape that reaches the `.global_addr` arm.
fn buildGlobalAddr(allocator: std.mem.Allocator, sym: []const u8) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const blk = try func.appendBlock();
    const p = try func.appendGlobalAddr(blk, ptr_t, sym);
    func.setTerminator(blk, .{ .ret = p });
    return func;
}

test "riscv64 addr_hi_lo: a global address emits an adjacent auipc+addi pcrel pair" {
    const allocator = std.testing.allocator;

    var func = try buildGlobalAddr(allocator, "K");
    defer func.deinit();

    // `fuse_addr_hi_lo = true` turns on the adjacency assert (this must not panic: proving the
    // invariant holds) as well as being the shape a river-rc1.ma model would compile with.
    var compiled = try isel.compileFunction(allocator, &func, .{ .fuse_addr_hi_lo = true });
    defer compiled.deinit(allocator);

    // Exactly one auipc/pcrel_hi20 reloc and one addi/pcrel_lo12 reloc, paired.
    var hi_offset: ?usize = null;
    var lo: ?isel.Reloc = null;
    for (compiled.relocs) |reloc| {
        switch (reloc.kind) {
            .pcrel_hi20 => {
                try std.testing.expect(hi_offset == null); // exactly one hi
                hi_offset = reloc.offset;
                try std.testing.expectEqualStrings("K", reloc.symbol);
            },
            .pcrel_lo12 => {
                try std.testing.expect(lo == null); // exactly one lo
                lo = reloc;
            },
            .call => unreachable, // no calls in this function
        }
    }
    const hi = hi_offset.?;
    const lo_reloc = lo.?;

    // The pcrel_lo12 pairs back to the exact word the pcrel_hi20 patched...
    try std.testing.expectEqual(hi, lo_reloc.pair);
    // ...and its own word is exactly one word after the auipc: the adjacency the fusion needs.
    try std.testing.expectEqual(hi + 1, lo_reloc.offset);

    // The words at those offsets really are an auipc followed by an addi (not just the relocs
    // claiming so): rd is nonzero (never x0, a symbol address is always materialized somewhere)
    // and both instructions target the same register.
    const auipc_word = compiled.code[hi];
    const addi_word = compiled.code[hi + 1];
    const auipc_rd: u5 = @truncate(auipc_word >> 7);
    const addi_rd: u5 = @truncate(addi_word >> 7);
    const addi_rs1: u5 = @truncate(addi_word >> 15);
    try std.testing.expectEqual(@as(u7, 0x17), @as(u7, @truncate(auipc_word))); // auipc opcode
    try std.testing.expectEqual(@as(u7, 0x13), @as(u7, @truncate(addi_word))); // addi opcode (imm form)
    try std.testing.expectEqual(auipc_rd, addi_rd);
    try std.testing.expectEqual(auipc_rd, addi_rs1);
}

test "riscv64 addr_hi_lo: fuse_addr_hi_lo true vs false compile a global address byte-identically" {
    const allocator = std.testing.allocator;

    var func = try buildGlobalAddr(allocator, "K");
    defer func.deinit();

    // The flag only gates a debug assert on an invariant that holds unconditionally, so it must
    // never change what gets emitted.
    var on = try isel.compileFunction(allocator, &func, .{ .fuse_addr_hi_lo = true });
    defer on.deinit(allocator);
    var off = try isel.compileFunction(allocator, &func, .{ .fuse_addr_hi_lo = false });
    defer off.deinit(allocator);

    try std.testing.expectEqualSlices(u32, on.code, off.code);
    try std.testing.expectEqual(on.relocs.len, off.relocs.len);
    for (on.relocs, off.relocs) |a, b| {
        try std.testing.expectEqual(a.offset, b.offset);
        try std.testing.expectEqual(a.kind, b.kind);
        try std.testing.expectEqual(a.pair, b.pair);
        try std.testing.expectEqualStrings(a.symbol, b.symbol);
    }
}
