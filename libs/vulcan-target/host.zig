//! Runtime host-CPU detection: queries the processor executing this code for the
//! ISA extensions it supports, so a JIT can pick instruction sequences for the CPU
//! it runs on rather than the machine the binary was built on.
//!
//! Architecture is fixed at compile time (a JIT runs on its own arch, so
//! `builtin.cpu.arch` is authoritative). Only features vary across CPUs of the same
//! arch, so those are detected at runtime. x86 uses `cpuid`/`xgetbv` (works on any
//! OS, including freestanding). aarch64 and riscv64 read the kernel hardware-
//! capability vector on Linux and fall back to the architectural baseline elsewhere.

const std = @import("std");
const builtin = @import("builtin");

/// The running CPU, as an ISA tag carrying that ISA's detected feature set. Each
/// variant's payload names only features meaningful to that architecture.
pub const Host = union(enum) {
    x86: X86,
    aarch64: Aarch64,
    riscv64: Riscv64,

    pub const X86 = struct {
        sse2: bool = false,
        sse41: bool = false,
        sse42: bool = false,
        avx: bool = false,
        avx2: bool = false,
        fma: bool = false,
        avx512f: bool = false,

        fn cpuid(leaf: u32, subleaf: u32) [4]u32 {
            var eax: u32 = undefined;
            var ebx: u32 = undefined;
            var ecx: u32 = undefined;
            var edx: u32 = undefined;
            asm volatile ("cpuid"
                : [eax] "={eax}" (eax),
                  [ebx] "={ebx}" (ebx),
                  [ecx] "={ecx}" (ecx),
                  [edx] "={edx}" (edx),
                : [leaf] "{eax}" (leaf),
                  [subleaf] "{ecx}" (subleaf),
            );
            return .{ eax, ebx, ecx, edx };
        }

        fn xcr0() u64 {
            var eax: u32 = undefined;
            var edx: u32 = undefined;
            asm volatile ("xgetbv"
                : [eax] "={eax}" (eax),
                  [edx] "={edx}" (edx),
                : [ecx] "{ecx}" (@as(u32, 0)),
            );
            return (@as(u64, edx) << 32) | eax;
        }

        pub fn detect() X86 {
            var h: X86 = .{};
            const leaf1 = cpuid(1, 0);
            const ecx1 = leaf1[2];
            const edx1 = leaf1[3];
            h.sse2 = (edx1 & (1 << 26)) != 0;
            h.sse41 = (ecx1 & (1 << 19)) != 0;
            h.sse42 = (ecx1 & (1 << 20)) != 0;

            // AVX/AVX2/AVX512 are usable only if the OS enabled saving the wide vector
            // state (OSXSAVE, then XCR0 advertising the YMM/ZMM regions), else they fault.
            const osxsave = (ecx1 & (1 << 27)) != 0;
            const cpu_avx = (ecx1 & (1 << 28)) != 0;
            const cpu_fma = (ecx1 & (1 << 12)) != 0;
            if (osxsave) {
                const xcr = xcr0();
                const ymm_ok = (xcr & 0x6) == 0x6; // XMM + YMM state saved
                const zmm_ok = (xcr & 0xe6) == 0xe6; // + opmask/ZMM hi state
                if (ymm_ok) {
                    h.avx = cpu_avx;
                    h.fma = cpu_fma and cpu_avx;
                    const leaf7 = cpuid(7, 0);
                    const ebx7 = leaf7[1];
                    h.avx2 = (ebx7 & (1 << 5)) != 0;
                    h.avx512f = zmm_ok and (ebx7 & (1 << 16)) != 0;
                }
            }
            return h;
        }
    };

    pub const Aarch64 = struct {
        neon: bool = false,
        dotprod: bool = false,
        sve: bool = false,

        pub fn detect() Aarch64 {
            var h: Aarch64 = .{ .neon = true }; // Advanced SIMD is mandatory on aarch64
            const cap = hwcap();
            if (cap != 0) {
                h.neon = (cap & (1 << 1)) != 0; // HWCAP_ASIMD
                h.dotprod = (cap & (1 << 20)) != 0; // HWCAP_ASIMDDP
                h.sve = (cap & (1 << 22)) != 0; // HWCAP_SVE
            }
            return h;
        }
    };

    pub const Riscv64 = struct {
        vector: bool = false,

        pub fn detect() Riscv64 {
            const cap = hwcap();
            return .{ .vector = (cap & (1 << ('V' - 'A'))) != 0 }; // COMPAT_HWCAP_ISA_V
        }
    };

    /// Widest packed-float SIMD width (in 32-bit lanes) usable on this CPU, or 1 if
    /// only scalar code is available. A vectorizing backend sizes its lanes by this.
    /// SVE/RVV are scalable, so their guaranteed 128-bit floor of 4 lanes is reported.
    pub fn floatLanes(self: Host) u8 {
        return switch (self) {
            .x86 => |x| if (x.avx512f) 16 else if (x.avx) 8 else if (x.sse2) 4 else 1,
            .aarch64 => |a| if (a.neon or a.sve) 4 else 1,
            .riscv64 => |r| if (r.vector) 4 else 1,
        };
    }

    pub fn format(self: Host, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s}:", .{@tagName(self)});
        switch (self) {
            inline else => |set| inline for (@typeInfo(@TypeOf(set)).@"struct".fields) |f| {
                if (@field(set, f.name)) try writer.print(" {s}", .{f.name});
            },
        }
        try writer.print(" (float lanes: {d})", .{self.floatLanes()});
    }
};

/// Detect the running CPU's feature set.
pub fn detect() ?Host {
    return switch (builtin.cpu.arch) {
        .x86, .x86_64 => .{ .x86 = Host.X86.detect() },
        .aarch64 => .{ .aarch64 = Host.Aarch64.detect() },
        .riscv64 => .{ .riscv64 = Host.Riscv64.detect() },
        else => null,
    };
}

fn hwcap() usize {
    if (builtin.os.tag != .linux) return 0;
    return std.os.linux.getauxval(std.elf.AT_HWCAP);
}

test "detect reports a sane baseline for the host arch" {
    const h = detect() orelse return; // no Vulcan backend for this arch
    switch (builtin.cpu.arch) {
        .x86_64 => try std.testing.expect(h.x86.sse2), // SSE2 is baseline on x86-64
        .aarch64 => try std.testing.expect(h.aarch64.neon), // NEON is baseline on aarch64
        else => {},
    }
    try std.testing.expect(h.floatLanes() >= 1);
}
