# Preserved C4 local development failure — NOT acceptance/latency evidence

`python-platform-mock.log` is the original failing CPU-only protocol-unit-test
output (exit 1, 10 tests, 7 errors), before the mock was fixed. Python 3.14
lazily calls `uname -p` within `platform.platform()`, and the globally patched
`subprocess.run` received its unexpected `timeout` keyword instead of a runner
command. The fixture now mocks `platform.platform()` explicitly; it does not
change production subprocess semantics or relax a benchmark gate.

Source was dirty development code; no pre-test source digest was captured.
None is retroactively invented. This log contains only synthetic temporary
protocol fixtures and is not GPU, latency or stage-acceptance evidence.
Subsequent 11-case C4 protocol unit tests, native CPU checks and separate clean
Release/Debug/ASan+UBSan evidence are recorded in `docs/stage11_report.md`.
