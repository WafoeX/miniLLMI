# Stage 4 — Arena allocator

**Status:** S4-C1–C4 implemented; [contracts](../arena.md) and [actual acceptance evidence](../stage4_report.md). Benchmark scope is explicitly CPU-only host diagnostics, not CUDA/server/model/planner performance. Dynamic default remains stable; arena is opt-in, slower valid runs retained.

**Goal:** independently correct reusable arena before graph liveness. **Prerequisite:** S3-C4. **Gate:** allocator invariants pass under deterministic and randomized tests.

## S4-C1 — Allocation interface and accounting
- **Goal/type:** Define `Arena::allocate/free`, aligned `Block`, counters, capacity/peak/live bytes. *Functional + Memory.*
- **Depends/files:** S1; `arena.hpp/.cpp`, `test_arena.cpp`.
- **Scope/not:** host arena only; no graph/lifetime/CUDA pool.
- **Verify/baseline/metrics:** alignment, exhaustion, zero-size policy; baseline N/A.
- **Accept/commit/risk:** every returned range is in capacity and accounting is exact. `feat(arena): add aligned arena interface`. Risk: alignment/overflow.

## S4-C2 — Simple reusable free list
- **Goal/type:** Deterministic aligned first-fit allocation, splitting and adjacent coalescing. Best fit is an optional policy, not an acceptance requirement. *Functional + Memory.*
- **Depends/files:** C1; arena/tests.
- **Scope/not:** one-threaded deterministic allocator; no compaction.
- **Verify/baseline/metrics:** fragmentation sequences, randomized allocate/free invariant checker; baseline bump/no-reuse; metrics reuse and peak bytes.
- **Accept/commit/risk:** no overlapping blocks; fully freed arena coalesces to one block. `feat(arena): add free-list split and coalesce`. Risk: free-list corruption.

## S4-C3 — Graph executor allocation-policy seam
- **Goal/type:** Make S3 executor select dynamic baseline or supplied allocation provider. *Integration + Memory.*
- **Depends/files:** S3-C3/S4-C2; executor/provider adapter/tests.
- **Scope/not:** no new planner; reuse S3 last-use cleanup semantics. Arena capacity failure is explicit, not a hidden malloc fallback. One arena-owning Storage is shared by views; views never individually free spans or outlive the execution context's storage owner.
- **Verify/baseline/metrics:** same graph values under both policies; dynamic malloc/free count vs arena count.
- **Accept/commit/risk:** no runtime API bypasses provider. `refactor(graph): inject allocation provider`. Risk: views must not free arena spans.

## S4-C4 — Arena benchmark and evidence
- **Goal/type:** Synthetic allocation workloads + fixed graph baseline CSV. *Performance + Memory.*
- **Depends/files:** C3; `bench/bench_allocator.cpp`, `results/allocator/`, analyzer.
- **Scope/not:** benchmark policy, not gallocr claim.
- **Verify/baseline/metrics:** dynamic vs arena: alloc/free calls, peak/live bytes, median execution; fixed input/build.
- **Accept/commit/risk:** raw samples/provenance saved and slower result retained. `bench(arena): add allocation baseline benchmark`. Risk: microbenchmark not representative; include graph workload.
