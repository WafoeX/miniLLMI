# Stage 9 T4 / Colab acceptance

This is the mandatory GPU gate for Stage 9. It uses a fresh clean worktree/build and writes immutable raw evidence under `results/gemm/stage9/`. Do not run it until the Stage 9 source commit is pushed and checked out. It does **not** run Nsight; profiling is Stage 10.

## 1. Select and verify the runtime

In Colab: **Runtime → Change runtime type → T4 GPU**. Then run the following one cell. It fails rather than pretending another GPU is T4.

```bash
%%bash
set -euo pipefail
nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in
  *T4*) ;;
  *) echo "Stage 9 requires a T4; actual GPU: $GPU_NAME" >&2; exit 1 ;;
esac
nvcc --version
cmake --version
python3 --version
```

## 2. Clone exactly the pushed branch/commit

Replace `BRANCH` with the pushed feature branch (or use the exact commit after checking it out). Never reuse a previous `build-*` directory or a previous Stage 9 result directory.

```bash
%%bash
set -euo pipefail
cd /content
rm -rf miniLLMI
# If this repository is private, authenticate Git before this command.
git clone https://github.com/WafoeX/miniLLMI.git miniLLMI
cd miniLLMI
git checkout BRANCH
git pull --ff-only
git status --short
git rev-parse HEAD
git log -1 --oneline
```

`git status --short` must be empty before the validation runner starts. Do not add generated files, edit source, or manually create metrics.

## 3. Run the complete acceptance runner

The script makes its own fresh Release CUDA build for SM75, runs GPU CTest (including V1 boundary/offset/guard/dispatch checks), captures three alternating paired v0/V1/cuBLAS repetitions, and writes raw samples plus a verified derived summary.

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI
python3 tools/run_cuda_gemm_stage9.py
```

Capture the printed result directory and inspect its outcome. The following command works for the newest run and reruns the verifier without altering raw data:

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI
RUN="$(find results/gemm/stage9 -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
printf 'RUN=%s\n' "$RUN"
cat "$RUN/manifest.json"
cat "$RUN/summary.md"
python3 tools/analyze_cuda_gemm.py "$RUN"
```

## 4. Acceptance decision and handoff

Accept S9-C1/C2 only if all are true:

1. `manifest.json` has `"status": "passed"`, unchanged `source`/`source_after`, and a real T4 identity;
2. fresh Release CUDA build and all GPU CTests pass;
3. every V0/V1/cuBLAS initial/final full-output check passes for each mandatory size;
4. `analysis.json` has `candidate_wins_every_run: true` and `gate_passed: true` (median paired geometric-mean V0/V1 speedup ≥1.05×);
5. all raw data, failures if any, logs, CSVs and hashes are preserved.

If the candidate is correct but `gate_passed` is false, report it as an implementation pass / performance miss. Do not delete slow rows or enable it as a default. The task permits exactly one bounded C3 optional follow-up (at most four declared legal configurations) only after recording that miss; do not silently lower the gate.

After a pass, commit the evidence separately from the tested source, retaining the tested source commit/digest in the result manifest. Send back the source commit, result commit, `summary.md`, `analysis.json`, and any failure log if the run did not pass.
