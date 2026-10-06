# Stage 5 — Graph lifetime planner (gallocr)

**Goal:** plan intermediate tensor storage before execution. **Prerequisite:** S4-C4 and frozen graph semantics. **Gate:** planned graphs preserve outputs and meet the roadmap buffer-allocation/arena-capacity improvement gate. This is not a claim of zero C++ heap allocation.

**Execution scope:** explicitly CPU-only host planner/reference integration and CPU-only memory/latency diagnostic; local named-host Release measurements are permitted. CUDA planning/server performance is not claimed. Frozen workloads, policies and timing boundaries are in [planner contracts](../planner.md). S5-C4 is `skipped_optional` (default disabled; not necessary for the required gate).

## S5-C1 — Lifetime analysis
- **Goal/type:** Compute birth, first/last use, consumer counts; mark graph inputs, outputs, aliases/views. *Functional + Correctness.*
- **Depends/files:** S3/S4; `memory_planner.hpp/.cpp`, tests.
- **Scope/not:** analysis only; no allocation/reuse.
- **Verify/baseline/metrics:** chain/diamond/multi-output/view intervals; N/A.
- **Accept/commit/risk:** external outputs stay live through execution; aliases extend base lifetime. `feat(planner): analyze graph tensor lifetimes`. Risk: last-use off-by-one.

## S5-C2 — Slot assignment and reuse plan
- **Goal/type:** Map non-overlapping intervals to aligned arena slots with a simple deterministic free-list policy. *Memory + Correctness.*
- **Depends/files:** C1/S4; planner/tests.
- **Scope/not:** CPU storage first; no inplace overwrite yet. Treat producer inputs and output as simultaneously live during execution; a normal output cannot reuse an input whose last consumer is that very node. Byte spans/alignment/device govern reuse; do not require identical shapes when sequential tensors safely fit one span.
- **Verify/baseline/metrics:** plan validity checker, peak planned bytes/reuse count; baseline dynamic allocation.
- **Accept/commit/risk:** simultaneous live tensors never overlap; reuse only same-device sufficiently sized/aligned byte spans, rebind checked tensor metadata, and extend lifetimes of all aliases. `feat(planner): assign reusable arena slots`. Risk: views and non-contiguous spans.

## S5-C3 — Planned executor integration
- **Goal/type:** Allocate arena at graph-build/prepare time, bind tensor storage at execute time. *Integration + Memory.*
- **Depends/files:** C2; executor/tests.
- **Scope/not:** execute-time buffer resize prohibited; replan is explicit. prepare preallocates intermediate/copy/workspace buffers and metadata needed for execution. Outputs are pinned until an explicit release or next documented execute, with snapshot/copy available to retain old results. Cache and weights are persistent external storage, never reusable intermediates.
- **Verify/baseline/metrics:** allocation hooks report 0 dynamic intermediate allocations; compare output baseline.
- **Accept/commit/risk:** changing input shape invalidates plan safely. `feat(planner): execute graph from memory plan`. Risk: stale plan after graph mutation.

## S5-C4 — Optional inplace eligibility and implementation
- **Goal/type:** Add conservative op-specific inplace rules (e.g., safe elementwise sole-consumer output). *Memory + Correctness.*
- **Depends/files:** C3; planner/op registry/tests.
- **Scope/not:** default disabled; never inplace graph I/O, views, or multiply-consumed values.
- **Verify/baseline/metrics:** alias/reference comparison; inplace count/peak reduction vs C3.
- **Accept/commit/risk:** enable only after exact conditions pass. `feat(planner): add conservative inplace reuse`. Risk: semantic corruption; fallback remains C3.

## S5-C5 — Gallocr benchmark/report
- **Goal/type:** Compare last-use dynamic execution, prepare-only no-reuse slots, planned reuse, optional inplace. *Performance + Memory.*
- **Depends/files:** C3; include C4 only if attempted; allocator benchmark/analyzer/docs.
- **Scope/not:** fixed graphs, no unmeasured general claim.
- **Verify/baseline/metrics:** backing-buffer alloc/free counts, peak live intermediate bytes, prepared arena capacity, total resident bytes, reuse/inplace count, median graph latency; same pinned I/O/weights. Dynamic execution uses last-use cleanup. Separately compare two prepared plans with identical alignment/capacity accounting: one slot per intermediate vs reusable slots. Predeclare a chain/diamond graph with several disjoint same-sized intermediates; report a decoder graph separately when available.
- **Accept/commit/risk:** roadmap gate requires >0→0 execute-time intermediate buffer allocations vs dynamic execution and ≥20% capacity reduction vs the no-reuse prepared plan. Do not claim an arena beats the dynamic live-byte lower bound: arena capacity is at least its simultaneously live byte requirement. Report prepare time separately; latency may be slower, and inplace is not needed. `bench(planner): compare graph allocation policies`. Risk: allocator changes can affect cache behavior.
