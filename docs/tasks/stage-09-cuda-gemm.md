# Stage 9 — CUDA SGEMM optimization experiments

**Implementation status:** S9-C1/C2 are T4 accepted; the required clean-source correctness and three paired performance runs pass the CUDA improvement gate. See [the CUDA SGEMM contract](../cuda_gemm.md), [acceptance evidence](../stage9_report.md), and [Colab procedure](../stage9_colab.md). C3–C6 are `skipped_optional` because V1 passed.

**Goal:** add selectable kernels without changing Stage 0 v0 or conditions. **Prerequisite:** S8-C5. **Gate:** required C1/C2 plus roadmap CUDA improvement gate; each attempted kernel has T4 correctness/raw data. C3–C6 are optional and stop once one custom path suffices; there is no v6 requirement.

## S9-C1 — Common kernel registry and config schema
- **Goal/type:** Register names/config (`BM,BN,BK,TM,TN,threads,vector`) and emit them in CSV. *Integration.*
- **Depends/files:** S8; GEMM registry/runner/analyzer/tests.
- **Scope/not:** no new optimized kernel. Extend schemas/analyzers versionedly so accepted Stage 0 rows/reports remain readable; do not change v0 source, generator, FP32 pedantic comparison or tolerance.
- **Verify/baseline/metrics:** config validation/unique kernel selection; Stage 0 v0 baseline.
- **Accept/commit/risk:** same runner compares all candidates under identical conditions. `refactor(cuda): add GEMM kernel registry`. Risk: CSV schema migration.

## S9-C2 — V1 shared-memory tiled SGEMM
- **Goal/type:** Cooperative tiled load/compute with bounds handling. *Correctness + Performance + Profiling.*
- **Depends/files:** C1; `sgemm_v1_tiled.cu`, tests/bench.
- **Scope/not:** no per-thread output tile, vector load, double buffer.
- **Verify/baseline/metrics:** all Stage 0 sizes; correctness also on tiny, non-square, non-tile-multiple and K-tail shapes, repeated launches and offset/alignment cases. Paired v0/candidate/cuBLAS same-run raw data; v0 speedup, cuBLAS %, shared bytes, errors; Nsight comparison follows S10.
- **Accept/commit/risk:** implementation pass requires correctness; stage benefit requires roadmap ≥1.05× gate, not a cuBLAS target. Slower results remain. If this passes, skip C3–C6; one bounded legal config follow-up is allowed on a miss. `perf(cuda): add shared-memory tiled sgemm`. Risk: barriers/boundary loads.

## S9-C3 — Optional V2 bounded legal tile sweep
- **Goal/type:** At most 4 configs declared before timing; for one output/thread use legal BM×BN such as 16×16 or 32×32 and declared BK values. Large 64/128 output tiles require C4 thread tiles; never map one thread/output to >1024 threads. *Correctness + Performance.*
- **Depends/files:** C2; parameter header/sweep script/CSV.
- **Scope/not:** no presumed winner or thread tiles. Validate thread count and shared-memory limits against queried T4 limits before launch; reject illegal configs and preserve diagnostics.
- **Verify/baseline/metrics:** same shape/input/10+50; GFLOPS, register/shared use, occupancy; baseline V1.
- **Accept/commit/risk:** all attempted configurations and failures saved. `perf(cuda): sweep block-tiled GEMM shapes`. Risk: compile-time parameter explosion.

## S9-C4 — Optional V3 register/thread tile
- **Goal/type:** Each thread accumulates a small output tile; at most 4 predeclared legal configs, not a Cartesian sweep of every block/thread tile. *Correctness + Performance + Profiling.*
- **Depends/files:** C2 (C3 optional); kernel/tests/sweep.
- **Scope/not:** no float4/double buffer.
- **Verify/baseline/metrics:** prior verified V1 or V2 baseline; errors, registers/thread, achieved occupancy, eligible warps, GFLOPS. Block/thread tile mapping must respect device launch/resource limits.
- **Accept/commit/risk:** select only data-supported stable config. `perf(cuda): add register tiled GEMM`. Risk: register spill/occupancy collapse.

## S9-C5 — Optional V4 vectorized/coalesced loads
- **Goal/type:** aligned `float4` path plus scalar safe tail/fallback. *Correctness + Performance.*
- **Depends/files:** C4; kernel/dispatch/tests.
- **Scope/not:** no undefined alignment cast and no input-padding assumption.
- **Verify/baseline/metrics:** aligned/unaligned/nonmultiple shapes; V3 baseline; global-load metrics.
- **Accept/commit/risk:** runtime chooses scalar fallback when preconditions fail. `perf(cuda): add vectorized GEMM load path`. Risk: alignment and boundary correctness.

## S9-C6 — Optional V5 double-buffer experiment
- **Goal/type:** shared-memory ping-pong/software pipeline as experimental-only variant. *Correctness + Performance + Profiling.*
- **Depends/files:** C5; kernel/bench/profile/docs.
- **Scope/not:** not automatically default.
- **Verify/baseline/metrics:** V4 baseline; GFLOPS, registers, shared memory, occupancy/stalls; retain slower data.
- **Accept/commit/risk:** implementation pass ≠ enablement; enable only if stable benefit. `perf(cuda): evaluate double-buffered GEMM`. Risk: overhead/regression; fallback V4.
