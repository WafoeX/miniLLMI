# Stage 11 C4 — accepted Colab T4 paired benchmark reproduction

**C2 is T4-accepted**: result `d4af778`, run `20261007T081857631689Z-12587`,
36/36 tests, 41/41 hashes and 16 full snapshot pairs verified.
**Stage 11 is complete to required C1/C2/C4.** C4 result `a84fbd8`, run
`20261007T101728202151Z-1834`, is independently verified: clean tested source
`906af4d`, 38/38 tests, 36/36 hashes, 60 raw rows and 12 full snapshots.
Median paired ratio 1.004480× is diagnostic only; the slower third pair is
retained. C3 is `skipped_optional`; there is **no scheduler speedup gate**.
See [acceptance report](stage11_report.md). Instructions below reproduce the
accepted protocol in a new run; no further run is needed to close Stage 11.

## 1. Update a clean T4 checkout

Select a **T4 GPU** runtime. Use the existing checkout after committing/pushing
previous results; do not discard failures or reset source edits:

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage11
git pull --ff-only origin feat/scheduler-stage11
git rev-parse HEAD
git status --short
nvidia-smi
nvcc --version
```

If the previous checkout is unavailable, create a **new** one:

```bash
%%bash
set -euo pipefail
cd /content
git clone --branch feat/scheduler-stage11 https://github.com/WafoeX/miniLLMI.git miniLLMI-stage11-c4
```

For a new checkout, substitute `/content/miniLLMI-stage11-c4` below. Do not run
against an older checkout that lacks `tools/run_scheduler_benchmark.py`.

## 2. Capture C4

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage11
python3 tools/run_scheduler_benchmark.py
```

The runner:

- Refuses dirty source or a non-T4 device; records actual CC 7.5 / SM75,
  GPU/driver, host/Python and compiler environment.
- Creates two **fresh Release** builds: testing ON for all **38 CPU/GPU CTests**
  and testing OFF for the actual benchmark. Never times a test-instrumented,
  stale, dirty or CPU-only binary.
- Freezes one 64×64×64 mixed graph and deterministic identical inputs; CPU
  FP64-reference backend plus **CUDA v0** in both modes. Manual explicit COPY
  versus automatic insertion, with identical effective placement/kernels.
- Runs **three independent paired processes**, ordered M/A, A/M, M/A;
  each mode uses 3 warmup executions, 10 samples × batch 5.
- Records 60 raw latency rows with actual counters and full initial/final
  output/reference/trace snapshots (12 total). Every timed execution checks
  the entire CPU output and actual counters. An independent Python FP64 oracle
  also validates all serialized references/outputs; tolerance stays .001/.001.
- Generates `analysis.json`, `scheduler.csv`, and `report.md` only from raw
  artifacts; never hand-edit metrics or overwrite earlier runs.

Timer scope is synchronous graph execution + counter checks + full oracle scan
+ output release, **not bare-kernel/model latency**. Prepare, oracle/input
construction, trace and file I/O are excluded. Both prepared contexts remain
resident; the report includes their per-device backing, inputs and reference
storage, while explicitly excluding unmeasured driver/library/metadata heaps.
Peak graph-owned live bytes, reserved backing and total tensor residency are
separate quantities. Sample ranges/SD/CV are reported; shared-host contention
and clocks are not controlled. No segmentation, switch-reduction or
kernel-improvement claim is made.

Output: `results/scheduler/stage11-c4/<run-id>/`. The manifest stores unchanged
clean tested source/digest, every command/exit code, build caches/compile
commands and SHA-256 of **all** artifacts. Failed/partial runs are retained and
also print their result directory. Push failures too; do not delete/retry in
place or change the frozen protocol to force a favourable result.

## 3. Push the result in a separate cell

Authenticate Git securely by your usual method. Never expose a token in chat,
committed files or notebook output.

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage11
git add results/scheduler/stage11-c4
git commit -m "bench(scheduler): record Stage 11 C4 T4 paired capture"
git push origin feat/scheduler-stage11
git rev-parse HEAD
```

For a new reproduction, send its **result commit and run ID** for independent
source/hash/test/oracle/trace/derived verification; do not replace the accepted
capture or its tested-source identity. A correct but slower automatic mode is
valid evidence, not a reason to discard a run.

A completed capture can be verified without rewriting evidence:

```bash
python3 tools/analyze_scheduler.py results/scheduler/stage11-c4/<run-id>
```

## Historical C2 / CPU-only reproduction

`python3 tools/run_scheduler_validation.py` reproduces C2's fresh T4 correctness
suite and 32 snapshots; it never produces C4 latency evidence. The first failed
empty-shape attempts and adapter fix are preserved in the [report](stage11_report.md).
The original Stage 0 launchers remain unchanged.

On a clean source checkout, `python3 tools/run_scheduler_validation.py --cpu-only`
captures fresh Release/Debug/ASan+UBSan tests including the new C4 CPU fallback
self-test and protocol mocks. `bench_scheduler --probe` reports build controls;
`--self-test` is untimed. CPU-only timing is refused. macOS ASan uses
`detect_leaks=0` (no LSan claim). None of these checks certifies GPU execution.
