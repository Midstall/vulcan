//! Structured annotations that ride through the pipeline for later layers
//! (codegen, the plugin engine) to act on. At the IR layer they are inert data:
//! stored, shape-validated, printed, parsed. The vocabulary is a closed core set
//! defined here plus an open namespaced bag, so target and codegen layers add
//! their own without touching this file. Attributes attach to functions, blocks,
//! instructions, and values, never to interned (shared) types.

const std = @import("std");

/// The value carried by a namespaced attribute.
pub const AttrValue = union(enum) {
    /// Presence only, no payload.
    flag,
    /// An integer payload.
    int: i64,
    /// A string payload. Owned by the function once stored.
    string: []const u8,
};

/// A namespaced attribute: `namespace.key = value`. The IR owns no namespace
/// vocabulary, target and codegen layers define their own.
pub const Custom = struct {
    namespace: []const u8,
    key: []const u8,
    value: AttrValue,
};

/// Byte order. The physical resolution happens in codegen against the target's
/// native order. In the IR it is a verified annotation on memory operations.
pub const Endianness = enum { little, big, native };

/// An attribute value: the closed core vocabulary plus the open namespaced bag.
pub const Attribute = union(enum) {
    /// Prefer inlining this function.
    @"inline",
    /// This function never returns.
    noreturn,
    /// This entity is rarely executed.
    cold,
    /// Required alignment, in bytes.
    @"align": u32,
    /// Byte order of a memory operation. Verified to sit only on loads/stores.
    endian: Endianness,
    /// A namespaced attribute from the open bag.
    custom: Custom,
};
