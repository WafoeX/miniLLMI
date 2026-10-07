# Stage 11 C2 — Colab T4 acceptance

**Status:** C1 is CPU-accepted; C2 code/local tests are ready, T4 acceptance
pending. This run validates correctness/copies/lifetimes, **not** C4 latency.
Per the task DAG, C4 starts only after C2 passes. C3 is `skipped_optional`.

## 1. Fresh clean checkout

Select a **T4 GPU** runtime in Colab. In a shell cell:

```bash
%%bash
set -euo pipefail
cd /content
# A NEW checkout; do not reuse a dirty previous stage checkout.
git clone --branch feat/scheduler-stage11 https://github.com/WafoeX/miniLLMI.git miniLLMI-stage11
cd miniLLMI-stage11
git pull --ff-only
git rev-parse HEAD
git status --short
nvidia-smi
nvcc --version
```

If this directory already exists, use `git pull --ff-only` on the same branch
and verify no source edits/untracked code. Do not reset away earlier failures.
The tested commit will be recorded automatically; result commits come later.

## 2. Build and capture

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage11
python3 tools/run_scheduler_validation.py
```

The runner refuses dirty source or a non-T4 device; derives SM75 from the actual
T4 CC 7.5 query, makes a **fresh Release build**, runs all CPU/GPU CTests, checks
CUDA device 0 independently and runs `test_cuda_scheduler` again to save 32
full-output/manual/automatic trace JSON snapshots. It independently validates
all 16 pairs against CPU references, expected actual copy bytes/counts,
dispatches/switches and graph protocol. No benchmarks or profiler tools run.

Raw output: `results/scheduler/stage11-c2/<run-id>/`. The manifest retains clean
source before/after, host/Python/GPU/build commands, exit codes, and SHA-256 of
all artifacts, including failures/partial runs. `validation.json` is generated
from snapshots, not hand-edited. Test coverage also includes versioned state
writes, fresh values on repeated execution, D2D, CUDA metadata aliases, and
escaped output lifetime blocking. CUDA v0 remains the default; v1 is explicit.

A success prints the result directory. A failure also prints/preserves its
result directory before returning nonzero. Do not delete failed attempts;
send the full directory/logs so the failure can be diagnosed.

## 3. Push evidence separately

Authenticate Git securely through your usual Colab method; do not paste a
GitHub token into chat, committed files or notebook output. Then:

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage11
git add results/scheduler/stage11-c2
git commit -m "test(scheduler): record Stage 11 C2 T4 validation"
git push origin feat/scheduler-stage11
git rev-parse HEAD
```

Send the result commit and run ID. We will fetch/verify hashes, tested source,
CPU/GPU test logs, snapshots/recomputed correctness and copy counters locally.
**Only then** is C2 T4-accepted and the dependent C4 benchmark/report implemented.
Stage 11 completion requires C4 server evidence as well; no speedup gate applies.

## CPU-only reproducibility

On a clean source checkout/worktree:

```bash
python3 tools/run_scheduler_validation.py --cpu-only
```

This captures fresh Release/Debug/ASan+UBSan and all applicable CTests. macOS
sets ASan `detect_leaks=0` (no LSan claim); Linux enables leak checking.
CPU-only success never certifies CUDA or Stage 11 mixed-backend acceptance.
