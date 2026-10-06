# Stage 6 acceptance report

**Status: complete — S6-C1/C2/C3/C4 required CPU-only gates pass.** Scope was explicitly declared before measurement in the [task book](tasks/stage-06-cpu-backend.md) and [CPU backend / frozen benchmark contract](cpu_backend.md): local named-host Release is permitted; GPU/server is not required. Stage 7–19 remain proposed. S6 establishes a baseline, not an optimization speedup claim.

## Tested identity and environment

- Tested source: **`75dc03871a35c5ac32e860513280486d76bdd529`**.
- Source digest: **`108fe30a3caf9c4b01688c2850ba5860c45e2ff0c2d4eefb09f263fb791c0c78`**, clean before/after both formal runners. Evidence/result commits are separate from this tested code; their implementation tree is unchanged.
- Host: `Wafoe-MacBook.local`, macOS 26.7 arm64, AppleClang `21.0.0.21000334`. Formal validation and timing use the same source identity. Complete flags/compiler/CMake/Python environment and command exits are in the manifests/logs below.
- Clean detached verification worktree: `/Users/wafoe/projects/me/CXXEg-stage6-verify`; fresh builds retained for audit. The original untracked `AGENTS.md` in the main working tree was not modified, ignored globally, committed or removed.
- This certifies neither Linux/LSan nor CUDA-enabled compilation/T4 integration. Apple ASan runs with leak detection disabled; actual sanitizer options are recorded. GPU gates remain mandatory for later CUDA stages.

## Changes and acceptance mapping

| Change | Code commit | Delivered / acceptance |
|---|---|---|
| S6-C1 | `81fe01d` | Internal synchronous Backend buffer/copy/prepare/execute and capability contract. Graph executor includes only the common backend interface, not CPU reference/kernel headers. Decorator test proves dispatch/first-error cleanup; S5 prepared execution uses that same executor. |
| S6-C2 | `48b7a53` | FP32 single-thread source-level ijk v0, caller-owned outputs and empty workspace. Shape/layout/binding/copy/finite semantics share S2's implementation. Backend fixture tests, 240 independent long-double-oracle random GEMMs, output guards, zero dimensions, non-square shapes, overlap/device/dtype/nonfinite/error tests and dynamic/planned graph integration pass. |
| S6-C3 | `fe34a88` | RMSNorm/Softmax/RoPE/Embedding/SwiGLU/Attention use ordinary descriptors/capability/dispatch; exact Unsupported status, no output writes/backing allocation, graph stops and cleans leases. No transformer numerical implementation. |
| S6-C4 | `75dc038` | Fresh clean Release test/production runner, raw CPU CSV + initial/final oracle checks, provenance/build/flags/hash validation, independent baseline runs and alternating graph policy pairs. Analyzer recomputes all statistics/GFLOPS and replay passes after copying the archive. |

Stable defaults remain **S2 FP64-accumulating math + dynamic allocation**. Explicit `CpuBackend(CpuMatmul::ScalarFP32V0)` selects v0; explicit provider selects prepared reuse. Neither backend dispatch nor kernel execution replaces/allocates output backing. Zero backing calls does not mean zero metadata/C++ heap activity. COPY/MATERIALIZE share S1 strided FP32/INT32 semantics; no hidden device transfer or materialization. Existing graph/state-write/pinning/failure contracts remain in force.

The frozen v0 file is [`src/runtime/cpu_scalar.cpp`](../src/runtime/cpu_scalar.cpp), SHA256 **`f185261bf89faed05ae6ab49f7525e95a9bfa1a99a6becb0c7c237678b162ebd`**. S7 must add a separate candidate and rerun this unmodified baseline paired under the same host/flags. Stage 0 oracle/CUDA kernels, source-input generator, accepted fixtures and historical results were not rewritten.

## Formal correctness evidence

[`results/cpu/local/20261006T040533995553Z-78182/manifest.json`](../results/cpu/local/20261006T040533995553Z-78182/manifest.json):

- Fresh **Release 29/29**, **Debug 29/29**, **ASan+UBSan 29/29** CTests; targeted sanitizer backend tests **6/6**. Fresh production Release also builds.
- Includes all prior S1–S5 regressions, S6 direct/backend fixture/numerical/error/reserved-primitive tests, benchmark workload self-test, **22** new mock-only analyzer/runner unit cases and **12** shared validation-runner cases. Mock timings live only in temporary unit-test directories.
- Benchmark runner independently performs another fresh Release test build/29 CTests, fresh production build and production eight-shape/graph oracle self-test. Six fresh builds' raw logs contain no compiler warning/error diagnostics; production benchmark symbols contain no Storage allocation test hooks.
- All twelve archived CMakeCache/compile-command snapshots match their retained build files byte-for-byte. No raw log/cache/CSV was manually corrected.

Commands on the clean tested source:

```bash
python3 tools/validate_cpu.py
python3 tools/run_cpu_benchmark.py
python3 tools/analyze_cpu.py results/cpu/baseline/20261006T040644002962Z-80817
```

## CPU baseline data (no improvement gate)

[`results/cpu/baseline/20261006T040644002962Z-80817/manifest.json`](../results/cpu/baseline/20261006T040644002962Z-80817/manifest.json), [generated table](../results/cpu/baseline/20261006T040644002962Z-80817/summary.md), [generated CSV](../results/cpu/baseline/20261006T040644002962Z-80817/cpu_gemm.csv), [analysis](../results/cpu/baseline/20261006T040644002962Z-80817/analysis.json):

- Required square **128/256/512/1024**: three independent baseline runs, ascending/descending/ascending order. **No timed FP64 oracle**, GPU comparison or unavailable optimization candidate. Optional 2048/4096 timings are not attempted.
- Fixed FP32 row-major/seed=42/threads=1/atol=rtol=1e-3; 3 warmup batches, 10 samples, GEMM batch=1. Whole backend-call wall time (validation + finite scan + v0 + status/output read), not bare-loop latency. Initial and final actual timed GEMM outputs agree with unchanged Stage 0 FP64 oracle.
- Fixed `[16,16] MATMUL → ADD → MUL` graph: dynamic/reuse order D/R, R/D, D/R; same inputs/backend, batch=20. Each timed graph execution performs a full independent-oracle scan and destroys outputs; prepare/input/oracle generation is outside timing. Prepared has zero execute-time backing allocations versus three dynamic calls, without a zero-C++-heap claim.
- **All three graph paired ratios are below 1: prepared reuse is slower on this small graph.** The generated summary retains each ratio and their median. No latency benefit is required for Stage 6, no default policy changes and no cherry-picked speedup claim. S7's required paired ≥1.05× geometric-mean GEMM improvement gate is deferred, not passed by these baseline measurements.
- **Compiler auto-vectorization is disclosed, not suppressed.** With production `-O3 -DNDEBUG -std=c++17 -arch arm64` and ordinary warning flags (no fast-math), AppleClang reports the v0 inner loop vectorized at width=4/interleaved=4. Thus “scalar ijk” describes source, not an absence of compiler-generated SIMD. [Actual remarks](../results/cpu/baseline/20261006T040644002962Z-80817/vectorization.log) and compile commands are archived; diagnostic recompilation is separate/untimed and does not replace the measured binary.
- 18 raw CSVs / **180 samples** (120 GEMM, 60 graph), 18 initial/final correctness JSONs, all 68 benchmark evidence artifacts SHA-verified. GFLOPS/statistics are generated from raw samples, never hand-filled. Same-host repeated samples are descriptive, not statistical-significance or cross-host generalization claims; no core affinity/exclusive-machine isolation guarantee.

## Integrity, failures and limits

[Final artifact verification](../results/cpu/verification/audit.json) records archive replay with byte-identical derived files, source/protected-path checks and snapshot/log coverage. A final active LSP probe covers touched code/docs; one sibling-import resolution false positive is recorded separately (actual CLI/import/tests pass), not hidden by an inline ignore. CMake LSP availability is not used as build proof; real fresh builds are the evidence. The final 28-path active probe returns no diagnostics after the one recorded false-positive disposition; 189 local document links resolve. Source/docs diff-check passes; staged raw-evidence diff-check reports original CMake/CTest whitespace and the generated CSV's standard CRLF. These result-only warnings are preserved in [the check log](../results/cpu/verification/staged-diff-check.log), not “fixed” by editing evidence.

[Development logs](../results/cpu/development/README.md) preserve three actual iterative failures: mock temporary-path canonicalization, positive-only Stage 0 oracle misuse and nonempty-only comparator misuse. These were fixed without changing Stage 0. Their original failed logs remain alongside successful reruns, and they are not mixed into formal performance/acceptance evidence. Early logs lacked pre-run source digests; no identity was retrospectively invented for them.

There is no pool, CPU optimization candidate, scheduler, CUDA backend, transformer numerical kernel, decoder, KV cache or quantization in Stage 6. No performance improvement is inferred from backend refactoring. Only Stage 6 is completed; Stage 7 is not started and main is not merged.
