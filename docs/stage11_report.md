# Stage 11 — Local implementation record (T4 pending)

**Stage 11 is not complete.** C1 is CPU-accepted; C2 code and local correctness
checks pass, but required mixed-backend T4 acceptance is pending. C3 is
`skipped_optional`; C4 is `not_started` until C2 passes. No latency/speedup,
segmentation benefit, CUDA correctness or model inference claim is made here.

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

## What still needs T4

Follow [the Colab procedure](stage11_colab.md):

```bash
python3 tools/run_scheduler_validation.py
```

The ready C2 suite validates manual versus automatic CPU→CUDA→CPU graphs,
rectangular/empty cases, v0 and explicit v1, full FP64-accumulating CPU reference
outputs, actual trace/copy bytes/counts/backend switches, repeated versioned
state writes, D2D, metadata aliases and output lifetime. These are **test
requirements**, not observed GPU results. Full outputs and traces are saved
for independent local revalidation after the result commit is pushed.

Once C2 T4 evidence passes, implement the dependent C4 fixed paired Release
manual/automatic latency benchmark and report. Stage 11 completion needs that
server evidence too; no scheduler speedup gate applies. C3 grouping remains
skipped unless later tracing justifies it. The runtime's default dynamic CPU
path, FP64 CPU math and Stage 0 CUDA v0 selection remain unchanged.

Contracts and limits: [scheduler](scheduler.md), [task book](tasks/stage-11-scheduler.md),
[roadmap](roadmap.md). Stage 0 v0 and frozen CPU baseline source were not edited.
