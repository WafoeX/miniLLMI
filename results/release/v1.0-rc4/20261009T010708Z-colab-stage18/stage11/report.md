# Stage 11 C4 — fixed mixed graph paired capture

Tested source: `ec116ce078eb7601e7cccc27a03f0c005eac9c7e` / `b819584a2efee94a6428d05d36b428343ef25779a8cb044a67c02942c96d3dd4`.

64×64×64 FP32, CUDA v0 and identical manual/automatic placement/kernels. Three independent paired processes; order M/A, A/M, M/A; 3 warmup executions, 10 samples × batch 5.

| Repeat | Order | Manual median ms | Automatic median ms | Manual/automatic | Manual sample SD ms | Automatic sample SD ms |
|---|---|---:|---:|---:|---:|---:|
| 1 | manual,automatic | 0.8537313 | 0.7926992 | 1.076993 | 0.14506672 | 0.078449056 |
| 2 | automatic,manual | 0.8826089 | 0.8856386 | 0.996579 | 0.0149785 | 0.021918109 |
| 3 | manual,automatic | 0.8702758 | 0.8414159 | 1.034299 | 0.049479673 | 0.0144505618 |

Median paired ratio: **1.034299×**. No speedup gate; slow runs are retained.

Actual execution in both modes: 4 copies / 65536 bytes / 9 backend dispatches / 2 executed switches; zero execute-time intermediate backing allocations.

Wall timer includes graph execution, counter validation, full-output oracle scan and output release. Backend calls synchronize before returning. Prepare, input/oracle construction, trace and file I/O are outside timing. This is NOT bare-kernel or model latency.

Sample ranges/SD/CV are recorded in analysis.json and scheduler.csv. Colab/shared-host contention and clocks are not controlled; capture-start GPU/driver state is retained in gpu-info.log. No statistical significance claim.

Peak live graph-owned bytes 65536 (including copy buffers and pinned output, excluding external inputs); retained output 16384 bytes. BlockAllocate/BlockFree trace events are logical requests/releases, NOT backing allocation calls.

Each context reserves CPU 49152 + CUDA 65536 bytes (including copies and pinned output); external host inputs 65536 bytes; oracle 16384 bytes. Both contexts + shared inputs + oracle remain resident: 311296 tensor bytes. Workspace 0. C++ metadata heap, CUDA/cuBLAS context/library overhead and driver memory are not measured by this tensor accounting.

All 60 raw sample rows and 12 complete initial/final trace/output snapshots validated, including an independent Python FP64 oracle. Latency ratios describe only steady execution of explicit/manual vs automatically rewritten prepared graphs with the same kernel and placement. Construction/rewrite/prepare is excluded; no construction-cost, kernel-improvement or placement-benefit claim. C3 segmentation skipped; no switch reduction or end-to-end model claim.

Raw samples and snapshots: `repeat-1/`, `repeat-2/`, `repeat-3/`. Source/environment/build provenance and hashes: `manifest.json`.
