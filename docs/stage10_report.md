# Stage 10 profiling evidence report

**Status: complete — S10-C1/C2/C3 pass.** This report records the required one-shape Nsight comparison and reproducible selection record for the already accepted Stage 9 V1 candidate. It does not turn profiler timings into benchmark results or authorize more tuning.

## Provenance and integrity

- Profile source: `785ab17af06fc0676dbb8a25c6a6d3e8983930aa`, source digest `c33b9ecbed732361ab021328b12470e8c6c5b0dc16edef76fbe74faf0cb549f7`, clean before and after capture.
- Result commit: `b1a8496bd3b085a706e70a7c1796e3442b055b59`.
- Run: [`20261007T040853604947Z-2951`](../results/profiling/stage10/20261007T040853604947Z-2951/), Tesla T4 / CC 7.5, fresh Release CUDA SM75 build.
- All runner commands exited zero: configure, build, two GPU CTests, and four profile commands. Every individual capture has `profile_exit_status=0` and passed initial/final full-output checks.
- The top-level manifest has 125 artifact hashes. All 119 Git-retained files re-hash exactly. The six absent Git files are intentionally ignored binary profiler artifacts: two `.nsys-rep`, two generated `trace.sqlite`, and two `.ncu-rep` files. Their per-report SHA-256 values remain in the profile manifests; the external run archive SHA-256 is `e49001522c60fa1d84cfca7fecbe79822e601e3942eccfbe01fe773f821edba1`.

## Required V0/V1 comparison

The runner captured nsys and NCU separately for `sgemm_v0_naive` and `sgemm_v1_tiled` at M=N=K=4096, with 10 warmups and 50 Event samples. The profiler-specific raw timing rows are retained only as provenance and are not a benchmark comparison.

| Profile | V0 report SHA-256 | V1 report SHA-256 |
| --- | --- | --- |
| Nsight Systems | `51afa2d5487f3bfbffeb849595561236ad1b2716c3e7ba128cebecef02fd8511` | `53a60bc7b1a04811d5db95d98e7abe7ad440e278cb22c0ef2dedfc2f00691450` |
| Nsight Compute | `81d117363bb9c53ccd88e3961308823c5b12a33931757040081cfb4888b82277` | `162eb8f3630625118b493462971d3abbf7c2bf767cf220bd31a6d814745b84f3` |

The actual metric names, units, observations, hypothesis, change, and scoped result are recorded in [the profiling protocol](profiling.md#t4-evidence-cards--4096). The Nsight Systems exports show the expected one selected kernel and 61 kernel instances in each capture. The NCU exports distinguish V0's reported global-load/LG-queue behavior from V1's shared-tile/MIO-queue behavior, while showing essentially equal achieved occupancy. This is explanatory evidence only.

## S10-C3 selection policy

The selection is reproducible from the accepted Stage 9 analyzer output, not from a fastest profiler sample: `sgemm_v1_tiled` passed all correctness checks and had V0/V1 run-level geometric-mean speedups of 1.678212×, 1.543855×, and 1.479965× over the mandatory 512/1024/2048/4096 workloads. The 1.543855× median is above the 1.05× gate and every retained run/size wins. `cublas` remains a reference ceiling, not an automatic target.

The selected custom configuration is therefore V1's fixed 16×16×16 shared tile when the caller explicitly requests `CudaMatmul::Stage9Tiled`; all experimental registry entries stay callable. The stable runtime default and simple fallback remain `CudaMatmul::Stage0Naive`, with no hidden size-based auto-selection. This preserves the Stage 9 dispatch contract and avoids selecting from a one-shape profiler capture.

## Limits

The Stage 9 paired CUDA Event results—not these profiler durations—satisfy the 1.05× performance gate. The profiler did not capture cuBLAS because Stage 10 requires only V0 and the selected custom kernel. No new kernel configuration was tuned or selected from one profiler sample; optional Stage 9 variants remain skipped. Stop CUDA GEMM tuning here.
