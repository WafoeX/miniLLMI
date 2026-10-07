# CUDA GEMM profiling protocol

Stage 10 profiles the accepted Stage 9 selected kernel (`sgemm_v1_tiled`) **against** the preserved V0 baseline (`sgemm_v0_naive`) at one declared representative shape, 4096³. It explains the existing performance-selection evidence; profiler timings are not benchmark data and cannot replace the Stage 9 paired 512–4096 CUDA Event gate.

The required T4 capture is recorded in [`results/profiling/stage10/20261007T040853604947Z-2951/`](../results/profiling/stage10/20261007T040853604947Z-2951/). Its source is `785ab17af06fc0676dbb8a25c6a6d3e8983930aa` (digest `c33b9ecbed732361ab021328b12470e8c6c5b0dc16edef76fbe74faf0cb549f7`), clean before and after capture. The detailed evidence cards and integrity record are in [the Stage 10 report](stage10_report.md).

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

## T4 evidence cards — 4096³

The following are observations from the actual Nsight exports, not benchmark measurements. Both V0 and V1 profile runs had a 16×16 block, 256 threads/block, 40-SM Tesla T4 (CC 7.5), 10 warmups, 50 Event samples, and passing initial/final full-output checks. Their profiler output timestamps, sampling overhead, and launch timings differ, so none of the timing values below is used as a speedup claim.

| Observation | V0 (`sgemm_v0_naive`) | V1 (`sgemm_v1_tiled`) | Evidence |
|---|---:|---:|---|
| Nsight Systems kernel instances / median duration | 61 / 446.533 ms | 61 / 259.218 ms | `stats.txt` (timeline only) |
| NCU `Memory Throughput` | 59.87 Gbyte/s | 91.51 Gbyte/s | `metrics.csv` |
| NCU `Compute (SM) Throughput` | 63.67% | 74.37% | `metrics.csv` |
| NCU `Registers Per Thread` | 52 register/thread | 41 register/thread | `metrics.csv` |
| NCU `Static Shared Memory Per Block` | 0 byte/block | 2.05 Kbyte/block | `metrics.csv` |
| NCU `Achieved Occupancy` | 99.88% | 99.89% | `metrics.csv` |
| NCU primary sampled stall interpretation | LG memory queue, 29.6 cycles / 80.3% | MIO queue, 17.1 cycles / 53.2% | `WarpStateStats` rule text |

**Observation → hypothesis → change → result.** The V0 NCU export reports that its global-load pattern uses 18.0 of 32 bytes per sector on average, and its sampled primary stall is waiting for the LG memory queue. The pre-existing Stage 9 hypothesis was that a cooperative shared 16×16 tile could reduce repeated global reads. V1 implements that fixed shared tile, and its profile records 2.05 Kbyte static shared memory/block, lower register use, greater reported memory/SM throughput, and a different primary MIO-queue stall. This supports the narrower explanation that V1 changed the memory-access/resource behavior in the expected direction; it does **not** establish a universal bottleneck or identify a further tuning target. The formal performance result remains the Stage 9 three-repeat, four-size CUDA Event gate: 1.543855× median paired geometric-mean V0/V1 speedup.

The exact binary-report SHA-256 values, profile directories, command logs, and text/CSV exports are linked in the Stage 10 manifest. Keep `CudaMatmul::Stage0Naive` as the runtime default: V1 is the correctness-passing, evidence-backed selected custom candidate, not an automatic fallback or a claim to match cuBLAS.

## Archive

The binary `.nsys-rep`, generated `trace.sqlite`, and `.ncu-rep` files are intentionally Git-ignored. The manifest verifies every tracked artifact; external binary reports remain at the recorded Google Drive location and in the run archive. The expected source procedure and acceptance checks are in [the Stage 10 Colab guide](stage10_colab.md).
