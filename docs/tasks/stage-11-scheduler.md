# Stage 11 — Backend scheduler and heterogeneous execution

**Goal:** choose capable backends and make device transitions explicit. **Prerequisite:** S5, S6, S8. **Gate:** required C1/C2/C4; mixed graph values equal CPU references and copies are counted. C3 segmentation is optional; no scheduler speedup gate.

## Execution status

- C1: **CPU-accepted**, `cdadc25` (`feat(scheduler): add backend capability placement`), Release 32/32 CTests at that Change.
- C2: **T4-accepted** on clean `6497290`, result `d4af778`, run `20261007T081857631689Z-12587`: 36/36 CTests, 41/41 hashes and 16 manual/automatic snapshot pairs verified. [Acceptance evidence](../stage11_report.md), [contract](../scheduler.md), [Colab capture](../stage11_colab.md).
- C3: `skipped_optional`; no measured avoidable overhead yet justifies segmentation.
- C4: `not_started`, now unblocked by verified C2 acceptance. Required server benchmark/report still pending. Stage 11 is **not complete**.

## S11-C1 — Capability and placement policy
- **Goal/type:** Map `(op,dtype,layout,device)` to supported backend; deterministic policy selects CPU/CUDA. *Functional.*
- **Depends/files:** S6/S8; `scheduler.hpp/.cpp`, tests.
- **Scope/not:** first prefer user-requested placement when supported, otherwise deterministic documented CPU placement/error. Unsupported CUDA primitives are explicit CPU placements, never hidden fallback within a kernel. No cost model/autotuner or automatic copies/segments yet.
- **Verify/baseline/metrics:** unsupported fallback/error and deterministic placement; N/A.
- **Accept/commit/risk:** policy never claims CUDA implementation absent. `feat(scheduler): add backend capability placement`. Risk: policy leaks into operator semantics.

## S11-C2 — Explicit COPY-node insertion
- **Goal/type:** Insert H2D/D2H/D2D copies for cross-device edges and validate storage lifetime. *Integration + Correctness.*
- **Depends/files:** C1/S8-C2; graph rewrite/tests.
- **Scope/not:** no implicit execution-time transfer. Rewrite placement/copies first, revalidate topology/aliases, then recompute liveness and per-device storage plans including copy destinations and workspace. CPU-only plans from S5 cannot be reused unchanged. Deduplicate fan-out copies to the same destination device where lifetimes allow.
- **Verify/baseline/metrics:** CPU→CUDA→CPU graph results, copy bytes/count; baseline manually constructed copies.
- **Accept/commit/risk:** every cross-device use is covered by a traceable version-correct copy; consumers may share a deduplicated copy. Writes to external KV invalidate cached copy versions and force correct dependency/completion ordering. `feat(scheduler): insert explicit cross-device copies`. Risk: duplicate copies for fan-out.

## S11-C3 — Optional backend segmentation
- **Goal/type:** group already contiguous same-backend nodes; only attempt if tracing identifies avoidable overhead. Grouping alone does not eliminate transfers or prove fewer actual backend switches. *Performance + Integration.*
- **Depends/files:** C2; segment planner/tests/bench.
- **Scope/not:** no asynchronous overlap/pipelining.
- **Verify/baseline/metrics:** per-node placement baseline; copies, switches, graph median latency.
- **Accept/commit/risk:** segmentation cannot cross dependency/device boundary. `perf(scheduler): add backend segmentation`. Risk: transfer of large intermediates can outweigh grouping.

## S11-C4 — Scheduler benchmark/report
- **Goal/type:** fixed mixed graph benchmark with trace/copy data. *Performance + Documentation.*
- **Depends/files:** C2; include C3 only if attempted; results/analyzer/docs.
- **Scope/not:** no end-to-end model claim.
- **Verify/baseline/metrics:** same graph/input/placement/kernels/Release; compare manual explicit copies vs automatic insertion for correctness/counts. Compare per-node vs grouped execution only if C3 attempted. Record actual copy bytes, executed switches and latency, not segment-count proxies.
- **Accept/commit/risk:** report separates placement effect from kernel change. `bench(scheduler): record heterogeneous scheduling`. Risk: PCIe/Colab topology variability.
