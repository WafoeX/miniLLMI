# Stage 11 C4 — fixed mixed graph paired capture

Tested source: `906af4d3bf73f3156d0fe5dececa305247c46d89` / `2e1f15f1e4d19cca85a5bd1a66c2cb788f7e9da672d863c648f7efb85b59f0fc`.

64×64×64 FP32, CUDA v0 and identical manual/automatic placement/kernels. Three independent paired processes; order M/A, A/M, M/A; 3 warmup executions, 10 samples × batch 5.

| Repeat | Order | Manual median ms | Automatic median ms | Manual/automatic | Manual sample SD ms | Automatic sample SD ms |
|---|---|---:|---:|---:|---:|---:|
| 1 | manual,automatic | 0.4954078 | 0.4845854 | 1.022333 | 0.020326468 | 0.00723173723 |
| 2 | automatic,manual | 0.4836605 | 0.4815033 | 1.004480 | 0.0618834478 | 0.0220398097 |
| 3 | manual,automatic | 0.478524 | 0.4877425 | 0.981100 | 0.0173514723 | 0.0116489163 |

Median paired ratio: **1.004480×**. No speedup gate; slow runs are retained.

Actual execution in both modes: 4 copies / 65536 bytes / 9 backend dispatches / 2 executed switches; zero execute-time intermediate backing allocations.

Wall timer includes graph execution, counter validation, full-output oracle scan and output release. Backend calls synchronize before returning. Prepare, input/oracle construction, trace and file I/O are outside timing. This is NOT bare-kernel or model latency.

Sample ranges/SD/CV are recorded in analysis.json and scheduler.csv. Colab/shared-host contention and clocks are not controlled; capture-start GPU/driver state is retained in gpu-info.log. No statistical significance claim.

Peak live graph-owned bytes 65536 (including copy buffers and pinned output, excluding external inputs); retained output 16384 bytes. BlockAllocate/BlockFree trace events are logical requests/releases, NOT backing allocation calls.

Each context reserves CPU 49152 + CUDA 65536 bytes (including copies and pinned output); external host inputs 65536 bytes; oracle 16384 bytes. Both contexts + shared inputs + oracle remain resident: 311296 tensor bytes. Workspace 0. C++ metadata heap, CUDA/cuBLAS context/library overhead and driver memory are not measured by this tensor accounting.

All 60 raw sample rows and 12 complete initial/final trace/output snapshots validated, including an independent Python FP64 oracle. Latency ratios describe only steady execution of explicit/manual vs automatically rewritten prepared graphs with the same kernel and placement. Construction/rewrite/prepare is excluded; no construction-cost, kernel-improvement or placement-benefit claim. C3 segmentation skipped; no switch reduction or end-to-end model claim.

Raw samples and snapshots: `repeat-1/`, `repeat-2/`, `repeat-3/`. Source/environment/build provenance and hashes: `manifest.json`.
