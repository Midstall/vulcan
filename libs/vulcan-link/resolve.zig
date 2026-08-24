//! The link driver: parse a set of relocatable objects (generic `elf.zig`), infer
//! their architecture from `e_machine`, dispatch to the matching `arch/<a>.zig` link
//! backend, and wrap the result in a static ELF executable. Architecture-generic and
//! `std`-only; each backend owns its reloc math, stub/GOT layout, RVC compression, and
//! executable parameters.

const std = @import("std");
const elf = @import("elf.zig");
const gc = @import("gc.zig");
const dynamic = @import("dynamic.zig");
const archive = @import("archive.zig");
const scriptmod = @import("script.zig");
const layout = @import("layout.zig");
const riscv64 = @import("arch/riscv64.zig");
const aarch64 = @import("arch/aarch64.zig");
const x86_64 = @import("arch/x86_64.zig");
const x86 = @import("arch/x86.zig");

pub const Error = elf.Error;
pub const Arch = elf.Arch;
pub const Image = elf.Image;
pub const ResolvedSymbol = elf.ResolvedSymbol;
pub const Resolver = elf.Resolver;

/// The error set of the script-driven link path (`linkInputsScript` +
/// `layout.computeScriptPlacement`): the generic linker errors, script-parse errors, and
/// the layout-specific diagnostics. Defined in `layout.zig` (which the engine lives in)
/// and re-exported here so a caller can drive parse -> lay out -> relocate through one set.
pub const ScriptError = layout.ScriptError;

/// One link input: a relocatable object, always included, or an `.a` archive, whose
/// members are pulled in only to satisfy an otherwise-undefined symbol.
pub const Input = union(enum) {
    object: []const u8,
    archive: []const u8,
};

/// A resolved symbol's defining-section kind, translated to the `.dynsym` type bits a
/// `-shared` export carries: a symbol defined in an executable section is a callable function
/// (`STT_FUNC`), one in a data section (`.rodata`/`.data`/`.bss`) is a data object
/// (`STT_OBJECT`). A synthesized symbol with no natural section defaults to `is_exec = true`,
/// so it falls back to `.func`.
fn symKindOf(is_exec: bool) dynamic.SymKind {
    return if (is_exec) .func else .object;
}

/// The per-architecture executable parameters used by `writeElfExec`.
fn execParams(arch: Arch) elf.ExecParams {
    return switch (arch) {
        .riscv64 => riscv64.exec_params,
        .aarch64 => aarch64.exec_params,
        .x86_64 => x86_64.exec_params,
        .x86 => x86.exec_params,
    };
}

/// Dispatch parsed objects to the matching link backend. All four architectures go through
/// the `Placement` model (compute addresses -> apply relocations -> build the `Image` from
/// the single segment, via `buildImageViaPlacement`). riscv64 synthesizes its extern
/// stub/GOT region and RVC-compresses `.text` inside `computeDefaultPlacement`, and pairs
/// the two-pass hi/lo relocations inside `applyRelocs`; the other three have no such
/// mechanisms but expose the same compute/apply pair.
fn linkArch(allocator: std.mem.Allocator, arch: Arch, parsed: []elf.ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool) Error!Image {
    return switch (arch) {
        .riscv64 => buildImageViaPlacement(riscv64, allocator, parsed, base, resolver, compress_text),
        .aarch64 => buildImageViaPlacement(aarch64, allocator, parsed, base, resolver, compress_text),
        .x86_64 => buildImageViaPlacement(x86_64, allocator, parsed, base, resolver, compress_text),
        .x86 => buildImageViaPlacement(x86, allocator, parsed, base, resolver, compress_text),
    };
}

/// Link objects through an arch backend's `Placement` model (`computeDefaultPlacement` +
/// `applyRelocs`) into an `Image`. `Mod` is one of the `arch/<a>.zig` backends; every one
/// of them exposes the same `computeDefaultPlacement`/`applyRelocs` pair, so this single
/// generic helper serves all of them. Ownership handoff: the single segment's relocated
/// bytes and the resolved symbols move into the `Image`; only the placement's wrapper
/// slices (the segment array and the `section_places` map) are freed.
fn buildImageViaPlacement(comptime Mod: type, allocator: std.mem.Allocator, parsed: []elf.ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool) Error!Image {
    // The static path never runs the GC sweep, so it passes a null liveness mask (every
    // section is live) and the layout stays byte-identical.
    var placement = try Mod.computeDefaultPlacement(allocator, parsed, base, resolver, compress_text, null);
    errdefer placement.deinit(allocator);
    try Mod.applyRelocs(allocator, &placement, parsed);

    // The default placement is always exactly one segment (the whole image at `base`).
    const seg0 = placement.segments[0];
    const image: Image = .{ .code = seg0.bytes, .symbols = placement.symbols, .base = seg0.vaddr, .memsz = seg0.memsz };
    // `seg0.bytes` and `placement.symbols` now belong to `image`; free only the wrappers.
    allocator.free(placement.segments);
    allocator.free(placement.section_places);
    allocator.free(placement.section_place_base);
    return image;
}

/// Link a set of relocatable objects into one image, laid out in the order given
/// with `.text` first at `base`. The architecture is inferred from the first object's
/// `e_machine`; all objects must agree. Resolves every cross-object and intra-object
/// relocation. The caller owns the returned image.
pub fn linkObjects(allocator: std.mem.Allocator, objs: []const []const u8, base: u64) Error!Image {
    return linkObjectsResolved(allocator, objs, base, null);
}

/// Like `linkObjects`, but undefined symbols referenced by calls are bound to
/// absolute addresses via `resolver` (the JIT use case).
pub fn linkObjectsResolved(allocator: std.mem.Allocator, objs: []const []const u8, base: u64, resolver: ?Resolver) Error!Image {
    return linkImpl(allocator, objs, base, resolver, false);
}

/// Like `linkObjectsResolved`, but RVC-compresses each object's `.text` before layout
/// (the C-extension output path). The compression pass applies only when the inferred
/// architecture is `riscv64`; for other architectures this equals `linkObjectsResolved`.
pub fn linkObjectsCompressed(allocator: std.mem.Allocator, objs: []const []const u8, base: u64, resolver: ?Resolver) Error!Image {
    return linkImpl(allocator, objs, base, resolver, true);
}

fn linkImpl(allocator: std.mem.Allocator, objs: []const []const u8, base: u64, resolver: ?Resolver, compress_text: bool) Error!Image {
    var parsed = try allocator.alloc(elf.ParsedObject, objs.len);
    var parsed_n: usize = 0;
    defer {
        var i: usize = 0;
        while (i < parsed_n) : (i += 1) parsed[i].deinit(allocator);
        allocator.free(parsed);
    }
    for (objs, 0..) |obj_bytes, oi| {
        parsed[oi] = try elf.parseObject(allocator, obj_bytes);
        parsed_n = oi + 1;
    }

    if (parsed_n == 0) {
        // No inputs: an empty image at the requested base (matches the historical
        // behavior of linking zero objects).
        return .{
            .code = try allocator.alloc(u8, 0),
            .symbols = try allocator.alloc(ResolvedSymbol, 0),
            .base = base,
            .memsz = 0,
        };
    }

    const arch = parsed[0].arch;
    for (parsed[0..parsed_n]) |p| {
        if (p.arch != arch) return error.MalformedObject;
    }

    // RVC compression only applies to riscv64; other arches ignore the flag.
    const compress_here = compress_text and arch == .riscv64;
    return linkArch(allocator, arch, parsed[0..parsed_n], base, resolver, compress_here);
}

/// Link a mix of always-included objects and `.a` archives, laid out with `.text`
/// first at `base`. Each `.archive` input's members are pulled in only as needed to
/// satisfy an undefined symbol (see `linkInputsResolved`); a `.object` input is
/// always included, exactly like `linkObjects`.
pub fn linkInputs(allocator: std.mem.Allocator, inputs: []const Input, base: u64) Error!Image {
    return linkInputsResolved(allocator, inputs, base, null);
}

/// Like `linkInputs`, but undefined symbols referenced by calls are bound to absolute
/// addresses via `resolver` (the JIT use case), same as `linkObjectsResolved`.
///
/// Archive members are pulled by a symbol-driven fixpoint, not the archive's own
/// symbol index: every `.object` input is parsed and always included; every
/// `.archive` input's members are all parsed up front as pull candidates. A
/// candidate is pulled when one of its own (global, defined) ELF symbols is still
/// undefined across the included set; pulling it adds its own defined and undefined
/// symbols to the tracked sets, so a later sweep can pull a member that only a
/// just-pulled member needed. The sweep repeats until a full pass over every
/// candidate pulls nothing new. A symbol left undefined after the fixpoint surfaces
/// exactly as it does for `linkObjects` today: `error.UndefinedSymbol` out of the
/// architecture link backend (or, with a resolver, out of the resolver callback).
pub fn linkInputsResolved(allocator: std.mem.Allocator, inputs: []const Input, base: u64, resolver: ?Resolver) Error!Image {
    // The always-included objects, plus (once pulled) archive members: this is the
    // final object set handed to the architecture link backend.
    var included: std.ArrayList(elf.ParsedObject) = .empty;
    var included_n: usize = 0;
    defer {
        var i: usize = 0;
        while (i < included_n) : (i += 1) included.items[i].deinit(allocator);
        included.deinit(allocator);
    }

    for (inputs) |in| switch (in) {
        .object => |bytes| {
            try included.append(allocator, try elf.parseObject(allocator, bytes));
            included_n = included.items.len;
        },
        .archive => {},
    };

    // Every archive member is a pull candidate, parsed once up front regardless of
    // whether it ends up used (an unpulled candidate is freed at the end).
    const Candidate = struct {
        parsed: elf.ParsedObject,
        pulled: bool = false,
    };
    var candidates: std.ArrayList(Candidate) = .empty;
    var candidates_n: usize = 0;
    defer {
        var i: usize = 0;
        while (i < candidates_n) : (i += 1) {
            if (!candidates.items[i].pulled) candidates.items[i].parsed.deinit(allocator);
        }
        candidates.deinit(allocator);
    }

    for (inputs) |in| switch (in) {
        .object => {},
        .archive => |bytes| {
            const members = try archive.parseArchive(allocator, bytes);
            defer allocator.free(members);
            for (members) |m| {
                try candidates.append(allocator, .{ .parsed = try elf.parseObject(allocator, m.bytes) });
                candidates_n = candidates.items.len;
            }
        },
    };

    if (included.items.len == 0 and candidates.items.len == 0) {
        // No inputs at all: an empty image, matching `linkObjects`' zero-object case.
        return .{
            .code = try allocator.alloc(u8, 0),
            .symbols = try allocator.alloc(ResolvedSymbol, 0),
            .base = base,
            .memsz = 0,
        };
    }

    // Track every referenced-but-undefined global symbol name, and every global
    // symbol name defined so far, across the included set. Both maps borrow their
    // keys from the parsed objects' own bytes (kept alive for the whole function).
    var defined_names: std.StringHashMapUnmanaged(void) = .empty;
    defer defined_names.deinit(allocator);
    var undefined_names: std.StringHashMapUnmanaged(void) = .empty;
    defer undefined_names.deinit(allocator);

    for (included.items) |obj| try trackSymbols(allocator, &defined_names, &undefined_names, obj);

    // Fixpoint over the archive candidates: pull any not-yet-pulled member that
    // defines a still-undefined symbol, then re-sweep (a just-pulled member may
    // itself need another member). Stop once a full sweep pulls nothing new.
    var changed = true;
    while (changed) {
        changed = false;
        for (candidates.items) |*c| {
            if (c.pulled) continue;
            if (!definesNeededSymbol(c.parsed, defined_names, undefined_names)) continue;
            c.pulled = true;
            changed = true;
            try trackSymbols(allocator, &defined_names, &undefined_names, c.parsed);
        }
    }

    // Fold every pulled candidate into the included set, in the order they were
    // encountered (object inputs first, then pulled archive members).
    for (candidates.items) |*c| {
        if (!c.pulled) continue;
        try included.append(allocator, c.parsed);
        included_n = included.items.len;
    }

    if (included.items.len == 0) {
        // No `.object` inputs and the fixpoint pulled no archive member (e.g. an
        // archive whose members are all unreferenced): an empty image, matching
        // `linkObjects`' zero-object case above.
        return .{
            .code = try allocator.alloc(u8, 0),
            .symbols = try allocator.alloc(ResolvedSymbol, 0),
            .base = base,
            .memsz = 0,
        };
    }

    const arch = included.items[0].arch;
    for (included.items) |p| {
        if (p.arch != arch) return error.MalformedObject;
    }
    return linkArch(allocator, arch, included.items, base, resolver, false);
}

/// A script-driven link result: the laid-out `Placement` (segments + resolved symbols),
/// the entry address (a copy of `placement.entry`), and the inferred architecture. Owns
/// the placement; `deinit` frees it.
pub const ScriptLinked = struct {
    placement: elf.Placement,
    entry: u64,
    arch: Arch,

    pub fn deinit(self: *ScriptLinked, allocator: std.mem.Allocator) void {
        self.placement.deinit(allocator);
    }
};

/// Dispatch the per-arch `applyRelocs` for a script-laid-out placement. Mirrors
/// `linkArch`'s switch: the same backend that would patch a default placement patches the
/// script one (both are just a `Placement` + the parsed objects).
fn applyRelocsArch(allocator: std.mem.Allocator, arch: Arch, placement: *elf.Placement, parsed: []elf.ParsedObject) Error!void {
    return switch (arch) {
        .riscv64 => riscv64.applyRelocs(allocator, placement, parsed),
        .aarch64 => aarch64.applyRelocs(allocator, placement, parsed),
        .x86_64 => x86_64.applyRelocs(allocator, placement, parsed),
        .x86 => x86.applyRelocs(allocator, placement, parsed),
    };
}

/// Parse `inputs` into the final included object set: every `.object` is always included;
/// every `.archive` member is a pull candidate, pulled by the same symbol-driven fixpoint
/// as `linkInputsResolved` (a member is pulled when it defines a still-undefined symbol,
/// re-swept until nothing new is pulled). Returns the included objects (in encounter
/// order: objects first, then pulled members). The caller owns the slice and every
/// `ParsedObject` in it (free each, then the slice). Kept separate from
/// `linkInputsResolved` so the default (non-script) link path stays byte-for-byte
/// untouched.
fn collectIncludedObjects(allocator: std.mem.Allocator, inputs: []const Input) Error![]elf.ParsedObject {
    var included: std.ArrayList(elf.ParsedObject) = .empty;
    var included_n: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < included_n) : (i += 1) included.items[i].deinit(allocator);
        included.deinit(allocator);
    }

    for (inputs) |in| switch (in) {
        .object => |bytes| {
            try included.append(allocator, try elf.parseObject(allocator, bytes));
            included_n = included.items.len;
        },
        .archive => {},
    };

    const Candidate = struct {
        parsed: elf.ParsedObject,
        pulled: bool = false,
    };
    var candidates: std.ArrayList(Candidate) = .empty;
    var candidates_n: usize = 0;
    defer {
        var i: usize = 0;
        while (i < candidates_n) : (i += 1) {
            if (!candidates.items[i].pulled) candidates.items[i].parsed.deinit(allocator);
        }
        candidates.deinit(allocator);
    }

    for (inputs) |in| switch (in) {
        .object => {},
        .archive => |bytes| {
            const members = try archive.parseArchive(allocator, bytes);
            defer allocator.free(members);
            for (members) |m| {
                try candidates.append(allocator, .{ .parsed = try elf.parseObject(allocator, m.bytes) });
                candidates_n = candidates.items.len;
            }
        },
    };

    var defined_names: std.StringHashMapUnmanaged(void) = .empty;
    defer defined_names.deinit(allocator);
    var undefined_names: std.StringHashMapUnmanaged(void) = .empty;
    defer undefined_names.deinit(allocator);

    for (included.items) |obj| try trackSymbols(allocator, &defined_names, &undefined_names, obj);

    var changed = true;
    while (changed) {
        changed = false;
        for (candidates.items) |*c| {
            if (c.pulled) continue;
            if (!definesNeededSymbol(c.parsed, defined_names, undefined_names)) continue;
            c.pulled = true;
            changed = true;
            try trackSymbols(allocator, &defined_names, &undefined_names, c.parsed);
        }
    }

    for (candidates.items) |*c| {
        if (!c.pulled) continue;
        try included.append(allocator, c.parsed);
        included_n = included.items.len;
    }

    return included.toOwnedSlice(allocator);
}

/// Link a mix of always-included objects and `.a` archives under the control of a parsed
/// linker `script`, laying out sections/addresses/boundary symbols per the script rather
/// than the default contiguous placement. Parses the inputs (archive members pulled by the
/// same fixpoint as `linkInputsResolved`), infers the architecture from the first object's
/// `e_machine`, runs the generic `layout.computeScriptPlacement`, then dispatches the same
/// per-arch `applyRelocs` a default link would (the relocation math is script-agnostic: a
/// site's address is `seg.vaddr + seg_off + r.offset`, so a script-assigned VMA flows
/// through unchanged). `resolver` is accepted for the riscv64 extern path and otherwise
/// unused (current callers pass null). The caller owns the returned `ScriptLinked`.
pub fn linkInputsScript(allocator: std.mem.Allocator, inputs: []const Input, script: *const scriptmod.Script, resolver: ?Resolver) ScriptError!ScriptLinked {
    _ = resolver; // riscv64 extern-resolution seam; unused by this layout path.
    const parsed = try collectIncludedObjects(allocator, inputs);
    defer {
        for (parsed) |*p| p.deinit(allocator);
        allocator.free(parsed);
    }
    if (parsed.len == 0) return error.MalformedObject;

    const arch = parsed[0].arch;
    for (parsed) |p| {
        if (p.arch != arch) return error.MalformedObject;
    }

    var placement = try layout.computeScriptPlacement(allocator, parsed, script, arch);
    errdefer placement.deinit(allocator);
    try applyRelocsArch(allocator, arch, &placement, parsed);

    return .{ .placement = placement, .entry = placement.entry, .arch = arch };
}

/// Add `obj`'s global symbols to the tracked defined/undefined name sets.
fn trackSymbols(allocator: std.mem.Allocator, defined_names: *std.StringHashMapUnmanaged(void), undefined_names: *std.StringHashMapUnmanaged(void), obj: elf.ParsedObject) Error!void {
    for (obj.symbols) |sym| {
        if (sym.name.len == 0 or sym.local) continue;
        if (sym.defined) {
            try defined_names.put(allocator, sym.name, {});
        } else {
            try undefined_names.put(allocator, sym.name, {});
        }
    }
}

/// True iff `obj` globally defines a symbol that is referenced-but-undefined and not
/// already satisfied by something already included.
fn definesNeededSymbol(obj: elf.ParsedObject, defined_names: std.StringHashMapUnmanaged(void), undefined_names: std.StringHashMapUnmanaged(void)) bool {
    for (obj.symbols) |sym| {
        if (!sym.defined or sym.local or sym.name.len == 0) continue;
        if (undefined_names.contains(sym.name) and !defined_names.contains(sym.name)) return true;
    }
    return false;
}

// File layout for the emitted ELF64 executables: ELF header, one program header, code.
const ehdr_size: usize = 64;
const phdr_size: usize = 56;

// File layout for the emitted ELF32 (i386) executable: same shape, 32-bit sizes.
const ehdr32_size: usize = 52;
const phdr32_size: usize = 32;

/// One loadable segment as the ELF writer core sees it: a run of file `bytes` mapped at
/// `vaddr` (physical `paddr`), zero-extended to `memsz`, with ELF `p_flags` in `flags`.
/// The single-`PT_LOAD` `writeElfExec` builds one of these; `writeElfSegments` builds one
/// per `elf.Segment`. `bytes` is `const` so `writeElfExec`'s immutable `code` fits too.
const SegDesc = struct {
    vaddr: u64,
    paddr: u64,
    bytes: []const u8,
    memsz: u64,
    flags: u32,
};

/// Compute a segment's file offset (`i == 0` lands at `code_offset`, the historical
/// single-segment position; later segments pack after the previous one, page-aligned and
/// congruent to their `vaddr` modulo `p_align` as ELF requires).
fn segFileOffset(i: usize, prev_end: u64, code_offset: u64, vaddr: u64, p_align: u64) u64 {
    if (i == 0) return code_offset;
    const want = vaddr % p_align;
    return std.mem.alignForward(u64, prev_end, p_align) + want;
}

/// Emit an ELF64 `ET_EXEC` with one `PT_LOAD` per `segs` entry for `arch`, entering at
/// `entry_addr`. For a single segment this reproduces the historical `writeElfExec` bytes
/// exactly: segment 0 sits at file offset `code_offset`, and when `map_headers` is set the
/// whole file (headers included) maps one page below the segment's `vaddr` so the bytes
/// land exactly at `vaddr`. The caller owns the returned bytes.
fn emitElf64(arch: Arch, allocator: std.mem.Allocator, segs: []const SegDesc, entry_addr: u64) std.mem.Allocator.Error![]u8 {
    const params = execParams(arch);
    const code_offset: u64 = params.code_offset;

    // Assign each segment a file offset and find the total file size.
    var file_offs = try allocator.alloc(u64, segs.len);
    defer allocator.free(file_offs);
    var file_end: u64 = code_offset;
    for (segs, 0..) |seg, i| {
        const off = segFileOffset(i, file_end, code_offset, seg.vaddr, params.p_align);
        file_offs[i] = off;
        file_end = off + seg.bytes.len;
    }

    const buf = try allocator.alloc(u8, @intCast(file_end));
    @memset(buf, 0);

    buf[0] = 0x7f;
    buf[1] = 'E';
    buf[2] = 'L';
    buf[3] = 'F';
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT

    const w = std.mem.writeInt;
    w(u16, buf[16..18], 2, .little); // e_type = ET_EXEC
    w(u16, buf[18..20], params.e_machine, .little); // e_machine
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[24..32], entry_addr, .little); // e_entry
    w(u64, buf[32..40], ehdr_size, .little); // e_phoff
    w(u16, buf[52..54], ehdr_size, .little); // e_ehsize
    w(u16, buf[54..56], phdr_size, .little); // e_phentsize
    w(u16, buf[56..58], @intCast(segs.len), .little); // e_phnum

    for (segs, 0..) |seg, i| {
        const p = buf[ehdr_size + i * phdr_size ..];
        w(u32, p[0..4], 1, .little); // p_type = PT_LOAD
        w(u32, p[4..8], seg.flags, .little); // p_flags
        if (i == 0 and params.map_headers) {
            // Map the whole file (headers included) one page below `vaddr`, so the code
            // itself (at file offset `code_offset`) lands exactly at `vaddr`.
            w(u64, p[8..16], 0, .little); // p_offset
            w(u64, p[16..24], seg.vaddr - code_offset, .little); // p_vaddr
            w(u64, p[24..32], seg.paddr - code_offset, .little); // p_paddr
            w(u64, p[32..40], code_offset + seg.bytes.len, .little); // p_filesz
            w(u64, p[40..48], code_offset + @max(seg.memsz, seg.bytes.len), .little); // p_memsz
        } else {
            w(u64, p[8..16], file_offs[i], .little); // p_offset
            w(u64, p[16..24], seg.vaddr, .little); // p_vaddr
            w(u64, p[24..32], seg.paddr, .little); // p_paddr
            w(u64, p[32..40], seg.bytes.len, .little); // p_filesz
            w(u64, p[40..48], @max(seg.memsz, seg.bytes.len), .little); // p_memsz
        }
        w(u64, p[48..56], params.p_align, .little); // p_align

        @memcpy(buf[@intCast(file_offs[i])..][0..seg.bytes.len], seg.bytes);
    }
    return buf;
}

/// The ELF32 (i386) counterpart of `emitElf64`: same multi-`PT_LOAD` shape, but
/// `Elf32_Ehdr`/`Elf32_Phdr` have different field widths *and offsets* than the ELF64
/// structs above (e.g. `Elf32_Phdr` orders `p_flags` after `p_paddr`, not right after
/// `p_type`). Mirrors `vulcan-target/x86/elf.zig`'s header layout (proven against
/// `qemu-i386`). A single segment reproduces the historical `writeElfExec32` bytes exactly.
fn emitElf32(allocator: std.mem.Allocator, segs: []const SegDesc, entry_addr: u64) std.mem.Allocator.Error![]u8 {
    const params = x86.exec_params;
    const code_offset: u64 = params.code_offset;

    var file_offs = try allocator.alloc(u64, segs.len);
    defer allocator.free(file_offs);
    var file_end: u64 = code_offset;
    for (segs, 0..) |seg, i| {
        const off = segFileOffset(i, file_end, code_offset, seg.vaddr, params.p_align);
        file_offs[i] = off;
        file_end = off + seg.bytes.len;
    }

    const buf = try allocator.alloc(u8, @intCast(file_end));
    @memset(buf, 0);

    buf[0] = 0x7f;
    buf[1] = 'E';
    buf[2] = 'L';
    buf[3] = 'F';
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT

    const w = std.mem.writeInt;
    w(u16, buf[16..18], 2, .little); // e_type = ET_EXEC
    w(u16, buf[18..20], params.e_machine, .little); // e_machine
    w(u32, buf[20..24], 1, .little); // e_version
    w(u32, buf[24..28], @intCast(entry_addr), .little); // e_entry
    w(u32, buf[28..32], ehdr32_size, .little); // e_phoff
    w(u16, buf[40..42], ehdr32_size, .little); // e_ehsize
    w(u16, buf[42..44], phdr32_size, .little); // e_phentsize
    w(u16, buf[44..46], @intCast(segs.len), .little); // e_phnum

    for (segs, 0..) |seg, i| {
        const p = buf[ehdr32_size + i * phdr32_size ..];
        w(u32, p[0..4], 1, .little); // p_type = PT_LOAD
        if (i == 0 and params.map_headers) {
            // Map the whole file (headers included) one page below `vaddr`, so the code
            // itself (at file offset `code_offset`) lands exactly at `vaddr`.
            w(u32, p[4..8], 0, .little); // p_offset
            w(u32, p[8..12], @intCast(seg.vaddr - code_offset), .little); // p_vaddr
            w(u32, p[12..16], @intCast(seg.paddr - code_offset), .little); // p_paddr
            w(u32, p[16..20], @intCast(code_offset + seg.bytes.len), .little); // p_filesz
            w(u32, p[20..24], @intCast(code_offset + @max(seg.memsz, seg.bytes.len)), .little); // p_memsz
        } else {
            w(u32, p[4..8], @intCast(file_offs[i]), .little); // p_offset
            w(u32, p[8..12], @intCast(seg.vaddr), .little); // p_vaddr
            w(u32, p[12..16], @intCast(seg.paddr), .little); // p_paddr
            w(u32, p[16..20], @intCast(seg.bytes.len), .little); // p_filesz
            w(u32, p[20..24], @intCast(@max(seg.memsz, seg.bytes.len)), .little); // p_memsz
        }
        w(u32, p[24..28], seg.flags, .little); // p_flags
        w(u32, p[28..32], @intCast(params.p_align), .little); // p_align

        @memcpy(buf[@intCast(file_offs[i])..][0..seg.bytes.len], seg.bytes);
    }
    return buf;
}

/// Wrap `code` in a minimal static ELF executable (`ET_EXEC`) for `arch`: a single
/// read+write+execute `PT_LOAD` segment mapping `code` at `base` with an in-memory size
/// of `mem_size` (>= `code.len`, the tail is zero-initialized `.bss`), entering at
/// `entry_addr`. The per-architecture page size and `e_machine` come from the arch's
/// `ExecParams`. i386 (`arch == .x86`) is the one ELFCLASS32 target here (32-bit
/// header/program-header field widths *and offsets*, not just a narrowed ELF64
/// layout); every other arch gets the ELF64 layout. The caller owns the returned bytes.
///
/// This is the single-segment special case of `writeElfSegments`, kept as a stable public
/// entry point: it builds one `SegDesc` (flags = the arch's default `p_flags`) and emits
/// through the shared core, so its bytes are identical to a one-`Segment` placement.
pub fn writeElfExec(arch: Arch, allocator: std.mem.Allocator, code: []const u8, mem_size: u64, base: u64, entry_addr: u64) std.mem.Allocator.Error![]u8 {
    const params = execParams(arch);
    const segs = [_]SegDesc{.{ .vaddr = base, .paddr = base, .bytes = code, .memsz = mem_size, .flags = params.p_flags }};
    if (arch == .x86) return emitElf32(allocator, &segs, entry_addr);
    return emitElf64(arch, allocator, &segs, entry_addr);
}

/// Emit a static ELF executable (`ET_EXEC`) for `arch` with one `PT_LOAD` per segment in
/// `placement`, entering at `entry_addr`. ELF64 for riscv64/aarch64/x86_64, ELF32 for x86.
/// A single-segment placement produces bytes byte-identical to `writeElfExec(arch, ...,
/// segments[0].bytes, segments[0].memsz, segments[0].vaddr, entry_addr)` (both go through
/// the same core). The caller owns the returned bytes.
pub fn writeElfSegments(arch: Arch, allocator: std.mem.Allocator, placement: *const elf.Placement, entry_addr: u64) std.mem.Allocator.Error![]u8 {
    var segs = try allocator.alloc(SegDesc, placement.segments.len);
    defer allocator.free(segs);
    for (placement.segments, 0..) |seg, i| {
        segs[i] = .{ .vaddr = seg.vaddr, .paddr = seg.paddr, .bytes = seg.bytes, .memsz = seg.memsz, .flags = seg.flags };
    }
    if (arch == .x86) return emitElf32(allocator, segs, entry_addr);
    return emitElf64(arch, allocator, segs, entry_addr);
}

/// Emit a runnable `ET_EXEC` for a linked image, entering at symbol `entry_name`.
pub fn writeExecutable(arch: Arch, allocator: std.mem.Allocator, image: *const Image, entry_name: []const u8) Error![]u8 {
    const entry_addr = image.addressOf(entry_name) orelse return error.UndefinedSymbol;
    return writeElfExec(arch, allocator, image.code, image.memsz, image.base, entry_addr);
}

/// Link a mix of relocatable objects, `.a` archives, and shared objects into a
/// dynamic ELF: an ET_DYN shared object (`opts.mode == .shared`) exporting the link unit's
/// global defined symbols, or an ET_EXEC dynamic executable skeleton (`opts.mode == .exec`)
/// carrying a PT_INTERP and `DT_NEEDED`s. The `.object`/`.archive` inputs are linked into
/// the code image exactly as the static path would (`collectIncludedObjects` + the per-arch
/// `Placement`/`applyRelocs` via `linkArch`), and each `.shared` input is parsed (validated
/// through `readSharedExports`) and bound against for function and data imports. The code
/// image is linked at base 0 (its backend relocations are PC-/page-relative, so it stays
/// valid once `dynamic.emit` places it page-aligned at the final VADDR). The caller owns the
/// returned bytes. This is a separate entry point from the static
/// `writeElfSegments`/`linkInputs*` path.
pub fn linkDynamic(allocator: std.mem.Allocator, inputs: []const dynamic.DynInput, opts: dynamic.DynOptions) Error![]u8 {
    // Split the inputs: `.object`/`.archive` feed the code image; each `.shared` contributes
    // its exports (name -> providing soname) so an undefined CALL in the link unit that a
    // `.shared` satisfies becomes an IMPORT (a PLT/GOT + JUMP_SLOT bound by the loader).
    var code_inputs: std.ArrayList(Input) = .empty;
    defer code_inputs.deinit(allocator);
    var shared_exports: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer {
        var it = shared_exports.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.*);
        }
        shared_exports.deinit(allocator);
    }
    for (inputs) |in| switch (in) {
        .object => |b| try code_inputs.append(allocator, .{ .object = b }),
        .archive => |b| try code_inputs.append(allocator, .{ .archive = b }),
        .shared => |b| {
            var se = try dynamic.readSharedExports(allocator, b);
            defer se.deinit(allocator);
            // A `.so` with no soname cannot be named in `DT_NEEDED`; skip binding against it.
            const soname = se.soname orelse continue;
            for (se.symbols) |name| {
                if (shared_exports.contains(name)) continue; // first `.so` wins
                const kdup = try allocator.dupe(u8, name);
                errdefer allocator.free(kdup);
                const vdup = try allocator.dupe(u8, soname);
                errdefer allocator.free(vdup);
                try shared_exports.put(allocator, kdup, vdup);
            }
        },
    };

    const parsed = try collectIncludedObjects(allocator, code_inputs.items);
    defer {
        for (parsed) |*p| p.deinit(allocator);
        allocator.free(parsed);
    }
    if (parsed.len == 0) return error.MalformedObject;
    const arch = parsed[0].arch;
    for (parsed) |p| {
        if (p.arch != arch) return error.MalformedObject;
    }

    // Run the GC mark pass when `--gc-sections` is on. `live` is a per-(object, section)
    // liveness mask indexed `live_base[oi] + si`, the same flat scheme `place.zig` uses.
    // `live` is null when GC is off, and every skip below then becomes a no-op, so the output
    // stays byte-identical. `live_base` mirrors that indexing for the resolve-layer skips.
    const live: ?[]bool = if (opts.gc_sections)
        try gc.computeLiveSections(allocator, parsed, opts.entry, opts.mode == .shared)
    else
        null;
    defer if (live) |lv| allocator.free(lv);
    var live_base: []usize = &.{};
    defer if (live_base.len > 0) allocator.free(live_base);
    if (live != null) {
        live_base = try allocator.alloc(usize, parsed.len);
        var total: usize = 0;
        for (parsed, 0..) |p, i| {
            live_base[i] = total;
            total += p.sections.len;
        }
    }

    // Classify each CALL relocation to an undefined symbol that a `.shared` exports as an
    // import: record its owning object, `.text` offset, and import index; build the deduped
    // import list. A CALL to a locally defined symbol, or an undefined one no `.shared`
    // satisfies, is left in place (the former resolves, the latter surfaces as
    // `error.UndefinedSymbol` below). A "call" here is any call-type relocation to an
    // undefined `.shared` export (`isCallReloc`: AArch64 CALL26 / x86-64 PLT32 / i386 PC32 /
    // riscv64 R_RISCV_CALL); a data relocation (e.g. x86-64 PC32) is never an import.
    var imports: std.ArrayList(dynamic.Import) = .empty;
    defer imports.deinit(allocator);
    // `near` marks a riscv64 `.jal` site (a single-instruction call, +/-1MiB reach): the
    // redirect must patch just that one `jal`, not the `.call` auipc+jalr pair. Always
    // false on the other three arches (their `isCallReloc` never matches `.jal`).
    const ImportSite = struct { oi: usize, si: usize, offset: u64, import_index: u32, near: bool };
    var import_sites: std.ArrayList(ImportSite) = .empty;
    defer import_sites.deinit(allocator);

    // The data (GOT-indirect) imports: an undefined symbol referenced by a GOT-indirect reloc
    // (aarch64 `.got_pg`/`.got_lo12`, x86-64 `.gotpcrel`) that a `.shared` input exports. Each
    // distinct symbol gets one `.got` slot + GLOB_DAT; its GOT-indirect refs are recorded so the
    // emitter patches them to that slot.
    var data_imports: std.ArrayList(dynamic.DataImport) = .empty;
    defer data_imports.deinit(allocator);
    var data_ref_sites: std.ArrayList(DataRefSite) = .empty;
    defer data_ref_sites.deinit(allocator);

    // The set of globally defined names across the included objects.
    var defined_names: std.StringHashMapUnmanaged(void) = .empty;
    defer defined_names.deinit(allocator);
    for (parsed) |obj| {
        for (obj.symbols) |sym| {
            if (sym.defined and !sym.local and sym.name.len > 0) try defined_names.put(allocator, sym.name, {});
        }
    }

    for (parsed, 0..) |obj, oi| {
        // Scan each executable section's own relocs (the code relocs). An import call and a
        // GOT-indirect data ref both live in an `SHF_EXECINSTR` section. Record the owning
        // section index `si`, so a per-function object (many `.text.<name>` sections) resolves
        // each diverted site against the section it actually lives in, not the first one.
        for (obj.sections, 0..) |isec, si| {
            if ((isec.flags & elf.SHF_EXECINSTR) == 0) continue;
            // A GC-dead section is dropped, so record no import call or GOT ref against it
            // (its placement is not_placed, which would surface as a MalformedObject below).
            if (sectionDead(live, live_base, oi, si)) continue;
            for (isec.relocs) |r| {
                // A CALL reloc to an undefined `.shared` export is a FUNCTION import (PLT/JUMP_SLOT).
                if (isCallReloc(arch, r.type)) {
                    if (r.symbol >= obj.symbols.len) return error.MalformedObject;
                    const name = obj.symbols[r.symbol].name;
                    if (defined_names.contains(name)) continue; // resolved in-image
                    const soname = shared_exports.get(name) orelse continue; // not a shared export
                    const idx = importIndexOf(imports.items, name) orelse blk: {
                        const i: u32 = @intCast(imports.items.len);
                        try imports.append(allocator, .{ .name = name, .soname = soname });
                        break :blk i;
                    };
                    const near = arch == .riscv64 and r.type == .jal;
                    try import_sites.append(allocator, .{ .oi = oi, .si = si, .offset = r.offset, .import_index = idx, .near = near });
                    continue;
                }
                // A GOT-indirect reloc (aarch64 `.got_pg`/`.got_lo12`, x86-64 `.got_pcrel`) to an
                // undefined `.shared` export is a data import (GOT slot + GLOB_DAT). A GOT ref to an
                // in-image symbol (a local GOT, for a PIE) is left in place. This is not yet supported.
                const kind: ?dynamic.DataRefKind = switch (r.type) {
                    .adr_got_page => .got_pg,
                    .ld64_got_lo12_nc => .got_lo12,
                    // x86-64's plain GOTPCREL and its two relaxable variants all target a
                    // symbol's GOT slot; against a shared export they share one `.got` slot +
                    // GLOB_DAT. (An in-image REX_GOTPCRELX never reaches here - the
                    // `defined_names` guard below leaves it for `applyRelocs` to relax.)
                    .gotpcrel, .gotpcrelx, .rex_gotpcrelx => .got_pcrel,
                    .got32 => .got_abs,
                    // riscv64's GOT-indirect `auipc` (the paired `ld` reuses `pcrel_lo12_i`, so it is
                    // NOT classified by reloc type - it is derived from the `auipc` below).
                    .got_hi20 => .got_hi20,
                    else => null,
                };
                if (kind) |k| {
                    if (r.symbol >= obj.symbols.len) return error.MalformedObject;
                    const name = obj.symbols[r.symbol].name;
                    if (defined_names.contains(name)) continue; // resolved in-image (local GOT, not yet supported)
                    const soname = shared_exports.get(name) orelse continue; // not a shared export
                    const idx = dataImportIndexOf(data_imports.items, name) orelse blk: {
                        const i: u32 = @intCast(data_imports.items.len);
                        try data_imports.append(allocator, .{ .name = name, .soname = soname });
                        break :blk i;
                    };
                    try data_ref_sites.append(allocator, .{ .oi = oi, .si = si, .offset = r.offset, .import_index = idx, .kind = k });
                    // riscv64: the GOT-indirect `ld` sits one word after its `auipc` and reuses
                    // `pcrel_lo12_i` (target = the `auipc` label), so it is not itself a shared-export
                    // reloc. Derive its data ref from the `auipc` (offset + 4, same GOT slot) so the
                    // emitter patches its lo12 and it is filtered out of `applyRelocs`.
                    if (k == .got_hi20) {
                        try data_ref_sites.append(allocator, .{ .oi = oi, .si = si, .offset = r.offset + 4, .import_index = idx, .kind = .got_lo12_i });
                    }
                }
            }
        }
    }

    // No function/data IMPORTS: link the whole image, then wrap it. Still synthesize any
    // internal-target data-section pointer-init relocs (`.rela.data`/`.rela.rodata`) as
    // RELATIVE (PIE/`.so`) or direct fixups (non-PIE). The placement (not a plain `Image`) is
    // kept so `collectDataFixups` can locate each data slot and its target within the image.
    if (imports.items.len == 0 and data_imports.items.len == 0) {
        var placement = try computeDefaultPlacementArch(allocator, arch, parsed, 0, null, false, live);
        defer placement.deinit(allocator);
        try applyRelocsArch(allocator, arch, &placement, parsed);
        const fixups = try collectDataFixups(allocator, &placement, parsed, live, live_base);
        defer allocator.free(fixups);
        var exports = try allocator.alloc(dynamic.Export, placement.symbols.len);
        defer allocator.free(exports);
        for (placement.symbols, 0..) |s, i| exports[i] = .{ .name = s.name, .offset = s.address, .kind = symKindOf(s.is_exec) };
        const seg0 = placement.segments[0];
        return dynamic.emit(allocator, execParams(arch), seg0.bytes, seg0.memsz, opts, exports, &.{}, &.{}, &.{}, &.{}, fixups);
    }

    // Imports present (aarch64/x86-64/i386/riscv64 for functions; aarch64-only for data). Link
    // the code image with the import sites deferred (their relocs filtered out so `applyRelocs`
    // leaves a placeholder), then hand the imports + their final image offsets to the emitter,
    // which synthesizes the PLT/GOT and patches each ref. Arches with no PLT support surface as
    // `error.UnsupportedReloc` (they cannot reach here anyway - `isCallReloc` never fires for
    // them - but the guard keeps the intent explicit).
    if (arch != .aarch64 and arch != .x86_64 and arch != .x86 and arch != .riscv64) return error.UnsupportedReloc;
    // GOT-indirect data imports are now all four arches (aarch64 + x86-64 + i386 + riscv64).
    if (data_imports.items.len > 0 and arch != .aarch64 and arch != .x86_64 and arch != .x86 and arch != .riscv64) return error.UnsupportedReloc;
    return linkDynamicImports(allocator, arch, parsed, opts, imports.items, import_sites.items, data_imports.items, data_ref_sites.items, live, live_base);
}

/// One recorded GOT-indirect reference site (a data import's `.got_pg`/`.got_lo12` reloc):
/// its owning object index, its owning exec section index, the in-section offset, which data
/// import it addresses, and which half of the `adrp`/`ldr` pair it patches.
const DataRefSite = struct { oi: usize, si: usize, offset: u64, import_index: u32, kind: dynamic.DataRefKind };

/// True when GC is on and section `si` of object `oi` is dead (must be dropped). Returns
/// false when `live` is null (GC off), so every caller's skip is a no-op then. The index
/// `live_base[oi] + si` matches `gc.computeLiveSections` and `place.zig`.
fn sectionDead(live: ?[]const bool, live_base: []const usize, oi: usize, si: usize) bool {
    const lv = live orelse return false;
    return !lv[live_base[oi] + si];
}

/// Dispatch the per-arch `computeDefaultPlacement` for the dynamic import path. Mirrors
/// `applyRelocsArch`: the same backend that lays out a default static link lays out the
/// import path's code image.
fn computeDefaultPlacementArch(allocator: std.mem.Allocator, arch: Arch, parsed: []elf.ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool, live: ?[]const bool) Error!elf.Placement {
    return switch (arch) {
        .riscv64 => riscv64.computeDefaultPlacement(allocator, parsed, base, resolver, compress_text, live),
        .aarch64 => aarch64.computeDefaultPlacement(allocator, parsed, base, resolver, compress_text, live),
        .x86_64 => x86_64.computeDefaultPlacement(allocator, parsed, base, resolver, compress_text, live),
        .x86 => x86.computeDefaultPlacement(allocator, parsed, base, resolver, compress_text, live),
    };
}

/// The offset resolution for `linkDynamic`'s import path (factored out to keep the dispatcher
/// readable), architecture-generic over `arch`. Lays out the code image at base 0 with the
/// import CALL relocs removed, then computes each import call's final image offset from the
/// placement and emits the dynamic exe (PLT/GOT/JUMP_SLOT + call redirect via `dynamic.emit`,
/// whose per-arch `DynArch` supplies the stub encoding and redirect).
fn linkDynamicImports(
    allocator: std.mem.Allocator,
    arch: Arch,
    parsed: []elf.ParsedObject,
    opts: dynamic.DynOptions,
    imports: []const dynamic.Import,
    import_sites: anytype,
    data_imports: []const dynamic.DataImport,
    data_ref_sites: anytype,
    live: ?[]const bool,
    live_base: []const usize,
) Error![]u8 {
    // Shallow-copy each object, filtering each EXECUTABLE section's own relocs (import CALL
    // sites AND GOT-indirect data-import refs removed, so `applyRelocs` leaves each as a
    // placeholder the emitter patches; the layout is otherwise identical to a full link).
    // The filtered relocs go into a fresh owned slice per exec section, replacing that
    // section's relocs in a shallow-copied `sections` array. Non-exec sections keep their
    // original relocs (owned by `parsed`, never freed here). The copied section-array
    // wrappers and each filtered reloc slice are freed at the end.
    var filtered = try allocator.alloc(elf.ParsedObject, parsed.len);
    var filtered_sections = try allocator.alloc([]elf.ObjSection, parsed.len);
    var owned_kept: std.ArrayList([]elf.Reloc) = .empty;
    var built: usize = 0;
    defer {
        for (owned_kept.items) |s| allocator.free(s);
        owned_kept.deinit(allocator);
        var i: usize = 0;
        while (i < built) : (i += 1) allocator.free(filtered_sections[i]);
        allocator.free(filtered_sections);
        allocator.free(filtered);
    }
    for (parsed, 0..) |obj, oi| {
        const secs = try allocator.dupe(elf.ObjSection, obj.sections);
        filtered_sections[oi] = secs;
        built = oi + 1;
        for (secs, 0..) |*s, si| {
            if ((s.flags & elf.SHF_EXECINSTR) == 0) continue;
            var keep: std.ArrayList(elf.Reloc) = .empty;
            errdefer keep.deinit(allocator);
            for (s.relocs) |r| {
                if (isImportSite(arch, import_sites, oi, si, r.offset, r.type)) continue;
                if (isDataRefSite(data_ref_sites, oi, si, r.offset, r.type)) continue;
                try keep.append(allocator, r);
            }
            const kept = try keep.toOwnedSlice(allocator);
            errdefer allocator.free(kept);
            try owned_kept.append(allocator, kept);
            s.relocs = kept;
        }
        filtered[oi] = obj;
        filtered[oi].sections = secs;
    }

    // `filtered` keeps the same section count per object as `parsed`, so the `live`/`live_base`
    // indexing carries over unchanged.
    var placement = try computeDefaultPlacementArch(allocator, arch, filtered, 0, null, false, live);
    defer placement.deinit(allocator);
    try applyRelocsArch(allocator, arch, &placement, filtered);

    // Each import call's final image offset = its object's code section placement + the reloc
    // offset. The diverted site lives in the exec (`SHF_EXECINSTR`) section, so its address
    // comes from that section's place, not a per-class `.text` slot.
    var import_calls = try allocator.alloc(dynamic.ImportCall, import_sites.len);
    defer allocator.free(import_calls);
    for (import_sites, 0..) |isite, k| {
        // Resolve the diverted call against the section it actually lives in, so a per-function
        // object (many `.text.<name>` sections) redirects the right site.
        const tp = placement.sectionPlace(isite.oi, isite.si);
        if (tp.seg == elf.not_placed) return error.MalformedObject;
        import_calls[k] = .{ .site = tp.seg_off + isite.offset, .import_index = isite.import_index, .near = isite.near };
    }

    // Each data-import GOT ref's final image offset, likewise from its code section placement.
    var data_refs = try allocator.alloc(dynamic.DataImportRef, data_ref_sites.len);
    defer allocator.free(data_refs);
    for (data_ref_sites, 0..) |dsite, k| {
        const tp = placement.sectionPlace(dsite.oi, dsite.si);
        if (tp.seg == elf.not_placed) return error.MalformedObject;
        data_refs[k] = .{ .site = tp.seg_off + dsite.offset, .import_index = dsite.import_index, .kind = dsite.kind };
    }

    // The code image's defined globals (name + image offset), so the emitter resolves the
    // entry symbol (`_start`) and, for a shared object, the exports.
    var exports = try allocator.alloc(dynamic.Export, placement.symbols.len);
    defer allocator.free(exports);
    for (placement.symbols, 0..) |s, i| exports[i] = .{ .name = s.name, .offset = s.address, .kind = symKindOf(s.is_exec) };

    // Internal-target data-section pointer-init relocs, alongside the import machinery
    // (a program can carry both an imported-function PLT and an internal pointer init).
    const fixups = try collectDataFixups(allocator, &placement, filtered, live, live_base);
    defer allocator.free(fixups);

    const seg0 = placement.segments[0];
    return dynamic.emit(allocator, execParams(arch), seg0.bytes, seg0.memsz, opts, exports, imports, import_calls, data_imports, data_refs, fixups);
}

/// Gather the internal-target data-section pointer-init relocations (`R_*_ABS64` in
/// `.data`/`.rodata`) from `parsed` into `DataFixup`s the dynamic emitter turns into
/// `R_*_RELATIVE` (PIE/`.so`) or direct writes (non-PIE). `placement` must be the same
/// single-segment layout that was relocated (laid out at base 0, so a symbol's resolved
/// address equals its byte offset within the image). Each fixup's `site` is the pointer
/// slot's image offset (its section placement + the reloc offset) and `target` the referenced
/// symbol's image offset (+ addend). A data reloc whose target is not defined in-image is an
/// imported pointer (the GLOB_DAT/GOT path, out of scope here) and surfaces as
/// `error.UndefinedSymbol`. The caller owns the returned slice.
fn collectDataFixups(allocator: std.mem.Allocator, placement: *const elf.Placement, parsed: []elf.ParsedObject, live: ?[]const bool, live_base: []const usize) Error![]dynamic.DataFixup {
    var fixups: std.ArrayList(dynamic.DataFixup) = .empty;
    errdefer fixups.deinit(allocator);
    const base_vaddr = if (placement.segments.len > 0) placement.segments[0].vaddr else 0;
    for (parsed, 0..) |obj, oi| {
        // Walk every NON-exec allocatable section, not a fixed data/rodata pair. Each section
        // owns its own placement, so a pointer-init reloc resolves its site against the exact
        // section that holds it. That is what closes the last-wins gap: two sections of the
        // same class (for example two `.data` inputs) no longer collapse to one per-class slot.
        // Exec sections carry text relocs, which the arch applier patches, so this pass skips
        // them (`SHF_EXECINSTR`).
        for (obj.sections, 0..) |sec, si| {
            if ((sec.flags & elf.SHF_EXECINSTR) != 0) continue;
            if (sec.relocs.len == 0) continue;
            // A GC-dead source section is dropped, so its pointer-init fixups go with it (its
            // placement is not_placed, which would otherwise be a MalformedObject below).
            if (sectionDead(live, live_base, oi, si)) continue;
            const place = placement.sectionPlace(oi, si);
            if (place.seg == elf.not_placed) return error.MalformedObject;
            for (sec.relocs) |r| {
                if (r.symbol >= obj.symbols.len) return error.MalformedObject;
                // A `.prel32` (aarch64) or `.pc32` (x86-64) is a 32-bit PC-relative FDE pointer
                // in `.eh_frame` (a real glibc `crt1.o` carries two of them, one per FDE's
                // `initial_location`). VCC never registers the crt's unwind tables (the autolink
                // omits `crtbegin.o`, so nothing ever reads `.eh_frame`), so the site is LEFT
                // unrelocated rather than mis-applied as a 64-bit absolute pointer init. Its
                // symbol is a local `.text` section symbol (no name), so resolving it as a global
                // would fail anyway. A `.pc32` never appears in a real DATA pointer-init slot (a
                // pointer init is `.abs64`), so skipping it here only ever drops an unwind pointer.
                if (r.type == .prel32 or r.type == .pc32) continue;
                const sym = obj.symbols[r.symbol];
                // Resolve the target. A GLOBAL symbol comes from the merged table by name. A
                // LOCAL DEFINED symbol (a `.str.N` string literal, made local so per-object
                // duplicate numbering does not collide at link time) is NOT in the global table,
                // so it resolves through THIS object's own placement of the symbol's section plus
                // its in-section value - the same fallback `applyRelocs` uses for text relocs. A
                // pointer-init slot referencing an own-`.rodata` string (`char *p = "x";`) takes
                // this path.
                const target_addr = elf.findSymbol(placement.symbols, sym.name) orelse blk: {
                    if (sym.defined and sym.local and sym.section_index != std.math.maxInt(u32)) {
                        const sp = placement.sectionPlace(oi, sym.section_index);
                        if (sp.seg == elf.not_placed or sp.seg >= placement.segments.len) return error.MalformedObject;
                        break :blk placement.segments[sp.seg].vaddr + sp.seg_off + sym.value;
                    }
                    return error.UndefinedSymbol;
                };
                const site = try elf.tableOffset(place.seg_off, r.offset, 1);
                const target_off: i64 = @as(i64, @intCast(target_addr - base_vaddr)) + r.addend;
                try fixups.append(allocator, .{ .site = site, .target = @intCast(target_off) });
            }
        }
    }
    return fixups.toOwnedSlice(allocator);
}

/// Index of the import named `name` in `list`, or null (dedups the import list).
fn importIndexOf(list: []const dynamic.Import, name: []const u8) ?u32 {
    for (list, 0..) |imp, i| {
        if (std.mem.eql(u8, imp.name, name)) return @intCast(i);
    }
    return null;
}

/// Index of the data import named `name` in `list`, or null (dedups the data-import list, so
/// every ref to one symbol shares a single `.got` slot + GLOB_DAT).
fn dataImportIndexOf(list: []const dynamic.DataImport, name: []const u8) ?u32 {
    for (list, 0..) |imp, i| {
        if (std.mem.eql(u8, imp.name, name)) return @intCast(i);
    }
    return null;
}

/// True iff `(oi, si, offset)` names one of the recorded GOT-indirect data-import ref sites (a
/// `.got_pg`/`.got_lo12` reloc the emitter patches, so it is filtered before `applyRelocs`).
/// The section index `si` keeps two same-offset sites in different `.text.<name>` sections apart.
fn isDataRefSite(data_ref_sites: anytype, oi: usize, si: usize, offset: u64, typ: elf.RelocType) bool {
    // riscv64's GOT `auipc` is `got_hi20` and its paired `ld` reuses `pcrel_lo12_i` (recorded as a
    // derived ref at auipc+4); both must be filtered so `applyRelocs` never sees them.
    if (typ != .adr_got_page and typ != .ld64_got_lo12_nc and typ != .gotpcrel and typ != .gotpcrelx and typ != .rex_gotpcrelx and typ != .got32 and typ != .got_hi20 and typ != .pcrel_lo12_i) return false;
    for (data_ref_sites) |dsite| {
        if (dsite.oi == oi and dsite.si == si and dsite.offset == offset) return true;
    }
    return false;
}

/// True iff `t` is a call-type relocation that could target an imported function (and so be
/// redirected through a synthesized PLT entry): AArch64's CALL26 or x86-64's PLT32. A data
/// relocation (e.g. x86-64's PC32 `lea rd, [rip+disp32]` for a `global_addr`) is never an
/// import call, so it stays in the code image and resolves against a defined symbol.
fn isCallReloc(arch: Arch, t: elf.RelocType) bool {
    return switch (arch) {
        // i386 emits `R_386_PC32` (`.pc32`) for a `call rel32`; its `global_addr` data
        // reference is `R_386_32` (`.abs32`), never PC-relative - so on x86 a `.pc32` is
        // unambiguously a call. (On x86-64 `.pc32` is instead a data `lea`, hence PLT32.)
        .x86 => t == .pc32,
        // riscv64 lowers a far call as an `R_RISCV_CALL` (`.call`) `auipc`+`jalr` pair, or a
        // near call (VCC's codegen, +/-1MiB reach) as a single `.jal`; its data references
        // are the PCREL_HI20/LO12 pair, never `.call`/`.jal`, so either is unambiguously a
        // call redirected through the PLT.
        .riscv64 => t == .call or t == .jal,
        // aarch64 (and x86-64, which never emits `.jump26`): a `bl` CALL26 is the obvious call,
        // but a `b` JUMP26 tail-branch to an external function equally needs a PLT stub (a
        // `libc_nonshared.a` `atexit` wrapper tail-branches to `__cxa_atexit@plt`). A JUMP26
        // to an in-image symbol is skipped by the `defined_names` guard at the call site and
        // resolved directly in `applyRelocs`, so only the external case reaches the PLT. The
        // redirect keeps the B-vs-BL opcode (`patchCall26`->`applyCall26`), so a tail-branch
        // stays a tail-branch.
        else => switch (t) {
            .call26, .jump26, .plt32 => true,
            else => false,
        },
    };
}

/// True iff `(oi, si, offset, type)` names one of the recorded import call sites. The section
/// index `si` keeps two same-offset call sites in different `.text.<name>` sections apart.
fn isImportSite(arch: Arch, import_sites: anytype, oi: usize, si: usize, offset: u64, typ: elf.RelocType) bool {
    if (!isCallReloc(arch, typ)) return false;
    for (import_sites) |isite| {
        if (isite.oi == oi and isite.si == si and isite.offset == offset) return true;
    }
    return false;
}

test "linkObjects: a cross-section call resolves the callee to its OWN same-class section (not last-wins)" {
    const allocator = std.testing.allocator;

    // A hand-built ELF64/RELA x86-64 object with TWO executable sections of the same
    // class: ".text.a" holds a `call rel32` to the global `func_b`, and ".text.b" holds
    // `func_b`. This is the arbitrary-section, per-section-reloc path the old fixed
    // 4-section model could not represent: both `.text` inputs would have collapsed into
    // one blob and `func_b` would resolve to the merged base. Here each section keeps its
    // own place, so `func_b` must resolve to `.text.b`'s address, and the call in
    // `.text.a` must patch to reach it. Mirrors the hand-built fixture of the two-`.text`
    // parse test in `elf.zig`, extended with a cross-section relocation.
    const shstrtab = "\x00.text.a\x00.text.b\x00.symtab\x00.strtab\x00.rela.text.a\x00.shstrtab\x00";
    const strtab = "\x00func_a\x00func_b\x00";

    const text_a_off: u64 = 64;
    const text_a_size: u64 = 6; // E8 00 00 00 00 (call rel32) + C3 (ret)
    const text_b_off: u64 = text_a_off + text_a_size;
    const text_b_size: u64 = 1; // C3 (ret)
    const symtab_off: u64 = text_b_off + text_b_size;
    const symtab_size: u64 = 3 * 24; // null + func_a + func_b
    const strtab_off: u64 = symtab_off + symtab_size;
    const rela_off: u64 = strtab_off + strtab.len;
    const rela_size: u64 = 24; // one Elf64_Rela
    const shstrtab_off: u64 = rela_off + rela_size;
    const shoff: u64 = elf.alignUp(shstrtab_off + shstrtab.len, 8);
    const shnum: u16 = 7;
    const total: usize = @intCast(shoff + @as(u64, shnum) * 64);

    var buf = try allocator.alloc(u8, total);
    defer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    std.mem.writeInt(u16, buf[16..18], 1, .little); // e_type = ET_REL
    std.mem.writeInt(u16, buf[18..20], elf.EM_X86_64, .little);
    std.mem.writeInt(u32, buf[20..24], 1, .little); // e_version
    std.mem.writeInt(u64, buf[40..48], shoff, .little); // e_shoff
    std.mem.writeInt(u16, buf[52..54], 64, .little); // e_ehsize
    std.mem.writeInt(u16, buf[58..60], 64, .little); // e_shentsize
    std.mem.writeInt(u16, buf[60..62], shnum, .little); // e_shnum
    std.mem.writeInt(u16, buf[62..64], 6, .little); // e_shstrndx (.shstrtab is index 6)

    // `.text.a`: `call rel32` (E8, disp field at offset 1) to `func_b`, then `ret`.
    @memcpy(buf[text_a_off..][0..6], &[_]u8{ 0xE8, 0, 0, 0, 0, 0xC3 });
    // `.text.b`: `ret`.
    buf[@intCast(text_b_off)] = 0xC3;
    @memcpy(buf[@intCast(strtab_off)..][0..strtab.len], strtab);
    @memcpy(buf[@intCast(shstrtab_off)..][0..shstrtab.len], shstrtab);

    const writeSym = struct {
        fn f(b: []u8, off: u64, idx: usize, name: u32, info: u8, shndx: u16, value: u64, size: u64) void {
            const e = b[@intCast(off + idx * 24)..][0..24];
            std.mem.writeInt(u32, e[0..4], name, .little);
            e[4] = info;
            e[5] = 0;
            std.mem.writeInt(u16, e[6..8], shndx, .little);
            std.mem.writeInt(u64, e[8..16], value, .little);
            std.mem.writeInt(u64, e[16..24], size, .little);
        }
    }.f;
    // func_a @ .text.a (shndx=1), func_b @ .text.b (shndx=2). Both STB_GLOBAL|STT_FUNC.
    writeSym(buf, symtab_off, 1, 1, 0x12, 1, 0, text_a_size);
    writeSym(buf, symtab_off, 2, 8, 0x12, 2, 0, text_b_size);

    // The one relocation: `.rela.text.a` patches the call's disp32 (offset 1 in .text.a)
    // to reach `func_b` (symbol index 2), `R_X86_64_PLT32` (4) with addend -4.
    const rela = buf[@intCast(rela_off)..][0..24];
    std.mem.writeInt(u64, rela[0..8], 1, .little); // r_offset
    std.mem.writeInt(u64, rela[8..16], (@as(u64, 2) << 32) | 4, .little); // r_info = sym 2, PLT32
    std.mem.writeInt(i64, rela[16..24], -4, .little); // r_addend

    const writeShdr = struct {
        fn f(b: []u8, sh: u64, idx: u16, name: u32, typ: u32, flags: u64, off: u64, size: u64, link: u32, info: u32) void {
            const e = b[@intCast(sh + @as(u64, idx) * 64)..][0..64];
            std.mem.writeInt(u32, e[0..4], name, .little);
            std.mem.writeInt(u32, e[4..8], typ, .little);
            std.mem.writeInt(u64, e[8..16], flags, .little);
            std.mem.writeInt(u64, e[16..24], 0, .little); // sh_addr
            std.mem.writeInt(u64, e[24..32], off, .little);
            std.mem.writeInt(u64, e[32..40], size, .little);
            std.mem.writeInt(u32, e[40..44], link, .little);
            std.mem.writeInt(u32, e[44..48], info, .little);
            std.mem.writeInt(u64, e[48..56], 1, .little); // sh_addralign
            std.mem.writeInt(u64, e[56..64], 0, .little); // sh_entsize
        }
    }.f;
    const XF = elf.SHF_ALLOC | elf.SHF_EXECINSTR;
    writeShdr(buf, shoff, 0, 0, 0, 0, 0, 0, 0, 0); // NULL
    writeShdr(buf, shoff, 1, 1, elf.SHT_PROGBITS, XF, text_a_off, text_a_size, 0, 0); // .text.a
    writeShdr(buf, shoff, 2, 9, elf.SHT_PROGBITS, XF, text_b_off, text_b_size, 0, 0); // .text.b
    writeShdr(buf, shoff, 3, 17, elf.SHT_SYMTAB, 0, symtab_off, symtab_size, 4, 1); // .symtab -> .strtab
    writeShdr(buf, shoff, 4, 25, 3, 0, strtab_off, strtab.len, 0, 0); // .strtab
    writeShdr(buf, shoff, 5, 33, elf.SHT_RELA, 0, rela_off, rela_size, 3, 1); // .rela.text.a -> symtab, target .text.a
    writeShdr(buf, shoff, 6, 46, 3, 0, shstrtab_off, shstrtab.len, 0, 0); // .shstrtab

    const base: u64 = 0x400000;
    var image = try linkObjects(allocator, &.{buf}, base);
    defer image.deinit(allocator);

    // The two exec sections land in section-header order, `.text.a` first at `base`,
    // `.text.b` next at the 16-byte-aligned end of `.text.a` (x86-64 text alignment). So
    // `func_a` is at `base` and `func_b` at `base + 16`, each in its OWN section.
    const func_a = image.addressOf("func_a") orelse return error.UndefinedSymbol;
    const func_b = image.addressOf("func_b") orelse return error.UndefinedSymbol;
    try std.testing.expectEqual(base, func_a);
    const text_b_start = base + elf.alignUp(text_a_size, 16);
    try std.testing.expectEqual(text_b_start, func_b);
    // `func_b` resolves inside `.text.b`'s placed range, distinct from `.text.a`'s range
    // (this is the last-wins gap the old model had: it would have folded both to `base`).
    try std.testing.expect(func_b >= text_b_start and func_b < text_b_start + text_b_size);
    try std.testing.expect(func_b >= base + text_a_size); // beyond `.text.a`

    // The call's disp32 is patched so RIP-relative it lands exactly on `func_b`. The CPU's
    // RIP at the branch is the byte after the 4-byte field (`base + 1 + 4`).
    const disp = std.mem.readInt(i32, image.code[1..5], .little);
    const site_addr = base + 1;
    const call_target: u64 = @intCast(@as(i64, @intCast(site_addr + 4)) + disp);
    try std.testing.expectEqual(func_b, call_target);
}

test "linkObjects: an in-image REX_GOTPCRELX mov relaxes to a lea addressing the symbol" {
    const allocator = std.testing.allocator;

    // A hand-built ELF64/RELA x86-64 object modelling what a real glibc `crt1.o` does with
    // `main`: `_start` loads main's address with `mov main@GOTPCREL(%rip), %rdi` (REX.W 8B 3D
    // <disp32>), and `main` is defined in the SAME object. Because `main` binds locally, the
    // linker must RELAX the GOT indirection: the `mov` (8B) becomes a `lea` (8D) and the disp32
    // becomes a plain PC32 to `main`. Before this support the parse rejected the REX_GOTPCRELX
    // reloc (numeric 42) as `error.UnsupportedReloc`, which is what broke the x86_64 autolink.
    const shstrtab = "\x00.text\x00.symtab\x00.strtab\x00.rela.text\x00.shstrtab\x00";
    const strtab = "\x00_start\x00main\x00";

    const text_off: u64 = 64;
    // 48 8B 3D 00 00 00 00 (mov rdi,[rip+disp32]) + C3 (ret), then main: C3 (ret) at offset 8.
    const text_size: u64 = 9;
    const symtab_off: u64 = text_off + text_size;
    const symtab_size: u64 = 3 * 24; // null + _start + main
    const strtab_off: u64 = symtab_off + symtab_size;
    const rela_off: u64 = strtab_off + strtab.len;
    const rela_size: u64 = 24; // one Elf64_Rela
    const shstrtab_off: u64 = rela_off + rela_size;
    const shoff: u64 = elf.alignUp(shstrtab_off + shstrtab.len, 8);
    const shnum: u16 = 6;
    const total: usize = @intCast(shoff + @as(u64, shnum) * 64);

    var buf = try allocator.alloc(u8, total);
    defer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    std.mem.writeInt(u16, buf[16..18], 1, .little); // e_type = ET_REL
    std.mem.writeInt(u16, buf[18..20], elf.EM_X86_64, .little);
    std.mem.writeInt(u32, buf[20..24], 1, .little); // e_version
    std.mem.writeInt(u64, buf[40..48], shoff, .little); // e_shoff
    std.mem.writeInt(u16, buf[52..54], 64, .little); // e_ehsize
    std.mem.writeInt(u16, buf[58..60], 64, .little); // e_shentsize
    std.mem.writeInt(u16, buf[60..62], shnum, .little); // e_shnum
    std.mem.writeInt(u16, buf[62..64], 5, .little); // e_shstrndx (.shstrtab is index 5)

    @memcpy(buf[@intCast(text_off)..][0..9], &[_]u8{ 0x48, 0x8b, 0x3d, 0, 0, 0, 0, 0xc3, 0xc3 });
    @memcpy(buf[@intCast(strtab_off)..][0..strtab.len], strtab);
    @memcpy(buf[@intCast(shstrtab_off)..][0..shstrtab.len], shstrtab);

    const writeSym = struct {
        fn f(b: []u8, off: u64, idx: usize, name: u32, info: u8, shndx: u16, value: u64, size: u64) void {
            const e = b[@intCast(off + idx * 24)..][0..24];
            std.mem.writeInt(u32, e[0..4], name, .little);
            e[4] = info;
            e[5] = 0;
            std.mem.writeInt(u16, e[6..8], shndx, .little);
            std.mem.writeInt(u64, e[8..16], value, .little);
            std.mem.writeInt(u64, e[16..24], size, .little);
        }
    }.f;
    // _start @ .text (shndx=1) value 0, main @ .text value 8. Both STB_GLOBAL|STT_FUNC.
    writeSym(buf, symtab_off, 1, 1, 0x12, 1, 0, 8);
    writeSym(buf, symtab_off, 2, 8, 0x12, 1, 8, 1);

    // The one relocation: `.rela.text` patches the disp32 (offset 3 in .text) against `main`
    // (symbol index 2), `R_X86_64_REX_GOTPCRELX` (42) with addend -4 (the disp32 field width).
    const rela = buf[@intCast(rela_off)..][0..24];
    std.mem.writeInt(u64, rela[0..8], 3, .little); // r_offset
    std.mem.writeInt(u64, rela[8..16], (@as(u64, 2) << 32) | 42, .little); // r_info = sym 2, REX_GOTPCRELX
    std.mem.writeInt(i64, rela[16..24], -4, .little); // r_addend

    const writeShdr = struct {
        fn f(b: []u8, sh: u64, idx: u16, name: u32, typ: u32, flags: u64, off: u64, size: u64, link: u32, info: u32) void {
            const e = b[@intCast(sh + @as(u64, idx) * 64)..][0..64];
            std.mem.writeInt(u32, e[0..4], name, .little);
            std.mem.writeInt(u32, e[4..8], typ, .little);
            std.mem.writeInt(u64, e[8..16], flags, .little);
            std.mem.writeInt(u64, e[16..24], 0, .little); // sh_addr
            std.mem.writeInt(u64, e[24..32], off, .little);
            std.mem.writeInt(u64, e[32..40], size, .little);
            std.mem.writeInt(u32, e[40..44], link, .little);
            std.mem.writeInt(u32, e[44..48], info, .little);
            std.mem.writeInt(u64, e[48..56], 1, .little); // sh_addralign
            std.mem.writeInt(u64, e[56..64], 0, .little); // sh_entsize
        }
    }.f;
    const XF = elf.SHF_ALLOC | elf.SHF_EXECINSTR;
    writeShdr(buf, shoff, 0, 0, 0, 0, 0, 0, 0, 0); // NULL
    writeShdr(buf, shoff, 1, 1, elf.SHT_PROGBITS, XF, text_off, text_size, 0, 0); // .text
    writeShdr(buf, shoff, 2, 7, elf.SHT_SYMTAB, 0, symtab_off, symtab_size, 3, 1); // .symtab -> .strtab
    writeShdr(buf, shoff, 3, 15, 3, 0, strtab_off, strtab.len, 0, 0); // .strtab (SHT_STRTAB)
    writeShdr(buf, shoff, 4, 23, elf.SHT_RELA, 0, rela_off, rela_size, 2, 1); // .rela.text -> symtab, target .text
    writeShdr(buf, shoff, 5, 34, 3, 0, shstrtab_off, shstrtab.len, 0, 0); // .shstrtab

    const base: u64 = 0x400000;
    var image = try linkObjects(allocator, &.{buf}, base);
    defer image.deinit(allocator);

    // The `mov` (8B) relaxed to a `lea` (8D); the REX prefix (48) and ModRM (3D) are untouched.
    try std.testing.expectEqual(@as(u8, 0x48), image.code[0]);
    try std.testing.expectEqual(@as(u8, 0x8d), image.code[1]);
    try std.testing.expectEqual(@as(u8, 0x3d), image.code[2]);

    // The disp32 is a plain PC32 to `main`: RIP at the read point is the byte after the 4-byte
    // field (`base + 3 + 4`), so `rip + disp` must land exactly on `main` (`base + 8`).
    const main_addr = image.addressOf("main") orelse return error.UndefinedSymbol;
    try std.testing.expectEqual(base + 8, main_addr);
    const disp = std.mem.readInt(i32, image.code[3..7], .little);
    const site_addr = base + 3;
    const lea_target: u64 = @intCast(@as(i64, @intCast(site_addr + 4)) + disp);
    try std.testing.expectEqual(main_addr, lea_target);
}
