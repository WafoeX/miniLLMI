# CUDA GEMM profiling protocol

Stage 10 profiles the accepted Stage 9 selected kernel (`sgemm_v1_tiled`) **against** the preserved V0 baseline (`sgemm_v0_naive`) at one declared representative shape, 4096³. It explains the existing performance-selection evidence; profiler timings are not benchmark data and cannot replace the Stage 9 paired 512–4096 CUDA Event gate.

No Stage 10 T4 profile is checked in yet. Do not fill in metric values, bottlenecks, or an Observation → Hypothesis → Change → Result card until the exact T4 capture has completed.

## Capture contract

`tools/run_cuda_gemm_stage10.py` makes a fresh Release SM75 build, requires a clean source tree and an actual T4, runs both GPU CTests, then captures all four required profiles:

| Tool | Kernel | Required output |
|---|---|---|
| Nsight Systems | V0 baseline | `.nsys-rep`, `stats.txt`, command/environment/provenance |
| Nsight Compute | V0 baseline | `.ncu-rep`, `metrics.csv`, `available_metrics.txt`, command/environment/provenance |
| Nsight Systems | V1 selected kernel | `.nsys-rep`, `stats.txt`, command/environment/provenance |
| Nsight Compute | V1 selected kernel | `.ncu-rep`, `metrics.csv`, `available_metrics.txt`, command/environment/provenance |

Each individual capture is isolated under `results/profiling/stage10/<run-id>/profiles/<tool>/<unique-id>/`. It records the exact command, binary/source provenance, `nsys`/`ncu` versions through the environment capture, an SHA-256 of its binary report, and optional external-artifact location. The runner writes a top-level manifest with every retained file hash. It reuses one freshly generated full FP64 oracle only between the four profiler captures; every capture still performs its own initial and final full-output validation. This cache is a profiling-time practicality, not formal timing evidence.

Run a single registered kernel manually only when diagnosing an individual capture:

```bash
# BUILD_DIR must contain a clean Release CUDA build.
./scripts/profile_gemm.sh nsys v0
./scripts/profile_gemm.sh ncu v0
./scripts/profile_gemm.sh nsys v1
./scripts/profile_gemm.sh ncu v1
```

Aliases `naive`/`tiled` remain accepted. The script maps them to the canonical registry names and profiles only that selected launch; V1 is permitted under `--experiment profiling` solely for this separated evidence path. `cublas` is also supported for diagnostic use, but it is not a required Stage 10 card.

## Evidence interpretation

Use the actual names, units, and denominators supplied by the installed Nsight version. A useful card links:

1. the V0/V1 Nsight Systems timeline observations (launches, API/synchronization/copies, and kernel durations);
2. the corresponding Nsight Compute resource, occupancy, warp-stall, global-load, and shared-memory observations when those metrics are available;
3. the already accepted Stage 9 paired benchmark result, including V0/V1 and cuBLAS ratios; and
4. the exact profile directories plus binary-report SHA-256 values from the Stage 10 manifest.

Do not claim a metric unavailable from the exported report, infer a T4 peak from another GPU, turn a profiler duration into a benchmark result, or invent causality from a single metric. The falsifiable Stage 9 hypothesis was that cooperative shared 16×16 tiles reduce redundant global loads compared with V0; the required profile can support or qualify that explanation, but cannot change the accepted multi-size timing result.

## Archive and follow-up

The binary `.nsys-rep` and `.ncu-rep` files are intentionally Git-ignored. Preserve them in external storage before ending the Colab session, pass that storage location with `--artifact-location` during the runner, and commit only the text/CSV/log/manifest evidence afterwards. The expected source procedure and acceptance checks are in [the Stage 10 Colab guide](stage10_colab.md).

Once actual T4 artifacts exist, author the two evidence cards in a separate documentation/result commit. Keep `CudaMatmul::Stage0Naive` as the runtime default: V1 is the correctness-passing, evidence-backed selected custom candidate, not an automatic fallback or a claim to match cuBLAS.
