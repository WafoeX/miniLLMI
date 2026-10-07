# Stage 11 — accepted scheduler and heterogeneous execution

**Stage 11 is complete to its required C1/C2/C4 gates.** C1 is CPU-accepted;
C2 and C4 are **T4-accepted**. C3 is `skipped_optional`. The fixed paired
execution ratio is diagnostic, not a scheduler speedup or placement-benefit
claim; no segmentation or model inference claim is made. Stage 12–19 remain
proposed, not implemented.

## Code and reproducibility

- C1: `cdadc25` — deterministic requested placement, explicit supported CPU
  fallback or strict error; no CUDA capability for unimplemented primitives.
- C2 tested source: `79df7d277415249adf6113096017462776c67d65`.
- Clean source digest: `718e76e6f3a565a2c7bce44b649a093052427131f1337b1237baee4f9d6b02ae`.
- Raw CPU evidence: [`20261007T074011978256Z-55180`](../results/scheduler/stage11-c2/20261007T074011978256Z-55180/manifest.json).
- Results commit: `35db971` (separate from tested code).
- Fresh verification worktree: `/Users/wafoe/projects/me/CXXEg-stage11-verify`.
  Source before/after is identical and clean under the normal provenance rule
  excluding `results/`; this is not a claim that the original checkout's whole
  Git status is empty (its pre-existing untracked `AGENTS.md` remains untouched).

## Actual local checks

macOS 26.7 / arm64, AppleClang `21.0.0.21000334`, Python `3.14.6`:

| Configuration | CTests | Scope |
|---|---:|---|
| Fresh Release | 33/33 pass | CPU-only, no performance data |
| Fresh Debug | 33/33 pass | CPU-only |
| Fresh ASan + UBSan | 33/33 pass | CPU-only; macOS `detect_leaks=0`, no LSan claim |

All nine captured configure/build/test commands exit 0; retained artifact
hashes revalidated 9/9. No compiler warnings/errors in these fresh build logs.
The scheduler tooling test contains 16 protocol cases, including full-output
and trace validation, changed tolerance/counters/workspace, missing captures,
source changes, clean-source guard and preservation/hashing of failed runs.
GPU test C++ host syntax also passes locally; **CUDA/nvcc compilation and GPU
execution remain unverified locally** because no CUDA SDK/GPU is available.
CUDA-header editor diagnostics on the unchanged GPU-info target are environment
limitations, not grounds to rewrite accepted Stage 0 code.

Local scheduler tests execute the existing CPU Graph runtime and use a
metadata-only CUDA capability double whose allocate/execute methods throw.
They cover rewrite determinism, fan-out copy deduplication, per-device plan
validation, invalidation of old CPU plans, allocating INT32/empty COPY,
versioned persistent-state mutation and stale-state errors, repeat execution
with fresh external values, and escaped-output busy contexts.

[Preserved dirty-development failures](../results/scheduler/stage11-development-failures/20261007-local/README.md)
record the legacy trace regression and a Python platform-mock regression before
they were fixed. Their pre-test source digests were not captured and are not
retroactively reconstructed. They are not acceptance evidence.

## First T4 failure and adapter hotfix (historical)

Retained result commit `b5b4bc3` contains
[`20261007T075353274825Z-2587`](../results/scheduler/stage11-c2/20261007T075353274825Z-2587/manifest.json):
Tesla T4, clean source `44f05b4` before/after, all six artifact hashes verified.
Release configure/nvcc build passed; 35/36 CTests passed. The failing test is
`cuda_scheduler`: v0 manual/automatic 17×13×11 and 64×64×64 each passed two
repeats, then execution failed on the first empty case (M=0,K=3,N=2).
State-version, D2D/alias and v1 portions of that scheduler test were not reached.
No full snapshot capture or C2 acceptance was produced.

Code inspection identifies the boundary mismatch: Runtime shape inference
allows zero dimensions, while preserved Stage 0 launchers call `elements()`,
which rejects nonpositive dimensions. The CUDA **adapter**, not Stage 0,
now returns success for empty M/N outputs and clears a nonempty K=0 output
with synchronous `cudaMemsetAsync` completion before returning. No baseline
kernel change, hidden host math, allocation or tolerance relaxation is made.
New direct adapter tests cover M/N/K=0 and repeated NaN/nonzero-poisoned K=0
outputs for v0, v1 and cuBLAS. They require fresh T4 execution; locally only
host C++ syntax and CPU regressions can run. Graph failures now log the exact
case, mode, status and failed node. Three GCC indentation warnings in snapshot
serialization are also fixed. Original failed artifacts are not edited.

Hotfix source `bdf0fb60eb5fa5192f45ccedd458acea2894c4df` was retested in a
fresh clean worktree: Release/Debug/ASan+UBSan each 33/33 pass, nine commands
exit 0, source before/after identical and clean. Evidence:
[`20261007T081343953783Z-61306`](../results/scheduler/stage11-c2/20261007T081343953783Z-61306/manifest.json),
source digest `10109728170d2e744ed8a8c2a3d119a9d2d014467c81da5f02b0f6e6efe533bc`,
9/9 retained hashes verified, separate results commit `39743c0`. Both GPU test
host C++ syntax checks also pass without warnings. These are CPU-only checks;
at that time the newly added CUDA regression cases had not yet run on T4.

The patched CUDA target cannot be checked by the local Mac CUDA-mode checker:
there is no CUDA SDK/libdevice, and local CMake has `ENABLE_CUDA=OFF`. These
explicitly handled toolchain/target-scope diagnostics are not a claim of clean
CUDA syntax. The following independent T4 capture now verifies that build
and all regression tests; local CUDA-mode checks remain unavailable.

## C2 accepted T4 capture

- Result commit: `d4af778e94ed060572547b5e745e05c621c80947`.
- Run: [`20261007T081857631689Z-12587`](../results/scheduler/stage11-c2/20261007T081857631689Z-12587/manifest.json).
- Tested source: `6497290f9bba837f6dbb57c0272f7ac854eec2ce`, clean before/after.
- Source digest: `f0bbea7dde7db159fbe0c6fc59479da3638886ffabb325d3ca2225185af70bb8`, independently replayed from Git including executable bits.
- Tesla T4 CC 7.5; fresh Release/nvcc build without compiler warnings.
- All eight commands exit 0; all **36/36 CTests** pass, including the direct v0/v1/cuBLAS empty-shape regressions, persistent-state versions, D2D and output-alias lifetime tests.
- **41/41 artifact hashes** verified locally; 32 full-output/manual/automatic trace snapshots validated as 16 pairs. Local replay of `validate_snapshots` exactly matches saved `validation.json`.

Observed per-execution counters match in manual/automatic and v0/v1:

| M,K,N | Actual copies | Actual copy bytes | Backend dispatches | Executed switches |
|---|---:|---:|---:|---:|
| 17,13,11 | 4 | 2952 | 9 | 2 |
| 64,64,64 | 4 | 65536 | 9 | 2 |
| 0,3,2 | 1 | 24 | 6 | 2 |
| 2,0,3 | 2 | 48 | 7 | 2 |

All full outputs agree with CPU references under the frozen .001 absolute/
relative tolerances, trace counters independently match, and intermediate
backing allocations are zero during execution. Dispatches count backend API
calls, not CUDA kernel launches. This is correctness/integration acceptance,
not latency evidence. Three pre-hotfix failed T4 attempts on `44f05b4` are
retained (`075353…`, `075648…`, `075917…`); the two additionally returned failed
manifests also have all six hashes verified. Failures were not discarded.

## C4 frozen protocol

C2 now satisfies the dependency of C4. `bench/bench_scheduler.cpp` and
`tools/run_scheduler_benchmark.py` implement a fixed 64×64×64 CUDA-v0 paired
Release manual/automatic comparison: three independent paired processes,
M/A–A/M–M/A order, 3 warmup executions, 10 samples × batch 5. Each timed execution
validates full output/counters; initial/final snapshots retain complete outputs,
references and actual traces. The same snapshot writer is shared with C2 tests.
`tools/analyze_scheduler.py` independently reconstructs the deterministic inputs
and FP64 oracle, validates all 60 rows / 12 snapshots, then generates the report
and paired CSV. No raw or derived evidence is overwritten.

Production timing requires clean source, Release, CUDA and `BUILD_TESTING=OFF`.
The runner first executes a fresh testing build's 38 CPU/GPU CTests, then uses a
separate fresh production build. Timer scope includes synchronous execution,
actual-counter checks, full oracle scan and output release; prepare, input/oracle
construction and trace/I/O are excluded. Tensor memory includes both contexts,
external inputs and oracle; graph-owned peak live bytes and output retention
are independently replayed from logical BlockAllocate/BlockFree traces, not
confused with actual backing allocations. Sample SD/ranges/CV are retained;
Colab contention/clocks are not controlled. Metadata/driver/library memory is
not measured.
No C4 GPU latency numbers were captured locally. The independent server capture
below now supplies the required evidence; no speedup gate applies. C3 grouping
remains skipped; default dynamic CPU, FP64 CPU math and CUDA v0 selection are
unchanged. See [Colab reproduction](stage11_colab.md).

## C4 clean local verification (not GPU acceptance)

- Tested code: `1a331e694bbed7398c11f521c88b9102a1d340e7`, clean before/after;
  digest `67b6a59b874be57d6d1d06d7f0c3427f51ac664a5a5b2cc09d435e1e4aba5aaa`.
- Separate results commit: `472ec1b`.
- [Fresh CPU-only capture](../results/scheduler/stage11-c2/20261007T092335811153Z-73363/manifest.json):
  Release/Debug/ASan+UBSan each **35/35 CTests** pass, nine commands exit 0,
  **9/9 hashes** verified, no compiler warnings/errors. This reuses the C2
  CPU-capture tool/family; it is not new C2 GPU or C4 latency evidence.
  macOS 26.7 / arm64 / AppleClang 21; ASan `detect_leaks=0`, no LSan claim.
- [Additional host checks](../results/scheduler/stage11-c4-host-checks/20261007T092335811153Z-73363/manifest.json):
  fresh CPU Release `BUILD_TESTING=OFF` build, untimed self-test, clean/non-test
  probe, actual CPU timing refusal (expected exit 1, no output directory), and
  both CUDA-interface C++ host syntax checks with `-Werror`; **10/10 hashes**
  verified. No nvcc, CUDA SDK, GPU link or GPU execution is certified locally.
- C4 tooling contains **11 CPU-only protocol tests** with temporary synthetic
  timing fixtures, including frozen controls, malformed/missing/duplicate rows,
  full independent oracle, actual trace/lifetime accounting, wrong GPU/test/CPU
  probes, source changes including finalization, hash coverage, no overwrite,
  both CTest summary formats, and failed-capture retention. The production C++
  input hash independently agrees with Python (`4bc396fd55be2a5c`). Its oracle
  and trace checks also agree with the existing C2 T4 64-square captures;
  **those historical snapshots are not substituted for C4 timing**.
- Active changed-path C++/Python/Markdown checks return no findings. CMake LSP
  was unavailable; native fresh configure/build/test validates that path instead.
- [Preserved C4 dirty-development mock failure](../results/scheduler/stage11-development-failures/20261007-c4-local/README.md)
  remains stored with its provenance limitation; it is not acceptance evidence.
  Verification worktree `/Users/wafoe/projects/me/CXXEg-stage11-c4-verify` retains
  the builds. Original untracked `AGENTS.md` remains untouched.

Construction/rewrite/preparation costs are excluded from the C4 timer, so its
ratios cannot claim to accelerate graph construction. Only steady execution
of equivalent already-prepared graphs is compared. Local checks alone did not
close C4; the following fresh T4 capture and independent review do.

## C4 accepted T4 capture

- Result commit: `a84fbd82132629e34b3548d681cf347bc1430fc0` (data only).
- Run: [`20261007T101728202151Z-1834`](../results/scheduler/stage11-c4/20261007T101728202151Z-1834/manifest.json).
- Tested source: `906af4d3bf73f3156d0fe5dececa305247c46d89`, clean and identical
  before/after. Digest `2e1f15f1e4d19cca85a5bd1a66c2cb788f7e9da672d863c648f7efb85b59f0fc`
  independently replayed from all 189 source-tree files, including executable bits.
- Tesla T4 CC 7.5 / SM75; all **14 commands exit 0**. Fresh testing Release
  build passes **38/38 CTests**, including CUDA scheduler/adapter regressions
  and the new mixed-CUDA benchmark self-test. A separate fresh production
  Release build uses `BUILD_TESTING=OFF`; captured compile commands confirm
  no `RUNTIME_TESTING` in its benchmark and no fast-math flags in either build.
  Configure/build logs have no compiler/CMake warnings or errors.
- **36/36 artifact hashes** and complete file coverage verified. All **60 raw
  timing rows / 12 full snapshots** pass independent input/FP64 oracle,
  graph/dependency, copy/dispatch, node-completion and logical-lifetime checks.
  Saved `analysis.json` replays within the declared 1e-12 numeric tolerance;
  regenerated report is byte-identical and paired CSV controls/numbers agree
  (same tolerance for floating serialization). Original artifacts are unchanged.
- Generated [paired report](../results/scheduler/stage11-c4/20261007T101728202151Z-1834/report.md),
  [paired CSV](../results/scheduler/stage11-c4/20261007T101728202151Z-1834/scheduler.csv)
  and [analysis](../results/scheduler/stage11-c4/20261007T101728202151Z-1834/analysis.json)
  retain all samples, dispersion, controls and provenance.

The manual/automatic median-latency ratios are **1.022333×, 1.004480×,
0.981100×**; their median is **1.004480×**. The third pair is slower for the
automatic graph and is retained. With uncontrolled Colab contention/clocks
and reported sample dispersion, these numbers do **not** demonstrate a
meaningful or statistically established scheduling benefit. No sufficient
speedup gate exists for C4, so correctness plus the complete fixed-protocol
report satisfies its acceptance criteria without further optimization.

Both modes execute 4 copies / 65536 bytes / 9 backend API dispatches / 2
executed switches, with zero execution-time intermediate backing allocations.
Peak graph-owned live bytes are 65536, retained output 16384; each context
reserves CPU 49152 + CUDA 65536. Both contexts, shared host inputs and oracle
together account for 311296 tensor bytes, workspace 0; driver/library/metadata
heaps are excluded explicitly. Equal effective placement and identical CUDA v0
kernels isolate the comparison from kernel changes; prepare/rewrite cost is
not timed. C3 segmentation stays skipped, with no switch-reduction claim.

Required C1/C2/C4 are now accepted. All historical failed runs and the slower
third pair remain preserved, baseline code/default execution remain unchanged,
and Stage 11 is complete. This does not complete M5 or implement Stage 12.

Contracts and limits: [scheduler](scheduler.md), [task book](tasks/stage-11-scheduler.md),
[roadmap](roadmap.md). Stage 0 v0 and frozen CPU baseline source were not edited.
