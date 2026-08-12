//! Shared test-harness helper for the dynamic-linking E2E tests (`aarch64`/`x86_64`/`x86`
//! `tests/dynamic.zig`): each links a real ELF `.so` + dynexe with our own linker, writes
//! them to a tmp dir, and runs the dynexe through a REAL ld.so (native on aarch64, under
//! `qemu-x86_64`/`qemu-i386` elsewhere), asserting `exit(42)`.
//!
//! `std.process.run`'s child-spawn path (`Io.Threaded` fork/exec + `waitpid`) has been
//! observed to occasionally return a `run.term` that is NOT `.exited` (e.g. `.signal`) even
//! though the produced binary is byte-correct and runs cleanly when invoked directly - a
//! harness/toolchain spawn race, not a linker defect. `runExpectExit` retries ONLY on a
//! non-`.exited` term (the child did not cleanly run at all); the instant a run DOES report
//! `.exited`, the exit code is asserted immediately with no retry, so a genuine wrong-answer
//! regression still fails on the very first clean run and can never be masked by the retry.
//!
//! Note the file durability side of this is already handled upstream of this helper:
//! `Dir.writeFile` (used by every caller to write the `.so`/dynexe into the tmp dir) opens,
//! writes, and synchronously closes the file before returning (`Io.Threaded`'s `fileClose`
//! calls `close()` directly, not a queued/async close), so the produced files are fully
//! durable and visible to the child process before this helper's `std.process.run` ever
//! spawns it.

const std = @import("std");

/// How many times to attempt the run before giving up. Only non-`.exited` terms consume an
/// attempt; a clean `.exited` result is checked and returned on immediately.
const max_attempts = 5;

/// A small backoff between retries, so a transient spawn race gets a moment to clear
/// without turning the retry loop into a busy spin.
const retry_backoff_ms = 20;

/// Run `options` (a full `std.process.RunOptions`, same as a direct `std.process.run` call)
/// up to `max_attempts` times, and assert the child's exit code equals `expected_exit`.
/// Retries only when `run.term` is not `.exited`; asserts immediately (no retry) the moment
/// a run does exit. Returns `error.TestUnexpectedResult` if every attempt fails to produce
/// a clean `.exited` term.
pub fn runExpectExit(
    allocator: std.mem.Allocator,
    io: std.Io,
    options: std.process.RunOptions,
    expected_exit: u8,
) !void {
    var attempt: usize = 0;
    while (true) {
        attempt += 1;
        const run = try std.process.run(allocator, io, options);
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);

        switch (run.term) {
            .exited => |code| {
                // A clean run: assert the real result immediately, retry never applies here
                // so a genuine wrong-exit-code regression fails on the first attempt.
                try std.testing.expectEqual(expected_exit, code);
                return;
            },
            else => {
                if (attempt >= max_attempts) return error.TestUnexpectedResult;
                std.Io.sleep(io, .fromMilliseconds(retry_backoff_ms), .awake) catch {};
                continue;
            },
        }
    }
}
