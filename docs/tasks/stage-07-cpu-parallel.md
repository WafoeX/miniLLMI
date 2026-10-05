# Stage 7 — CPU parallel execution

**Goal:** evolve CPU GEMM without hiding algorithm changes. **Prerequisite:** S6-C4. **Gate:** required C1/C3/C5, race-free tests and roadmap CPU speedup gate. C2/C4 are optional; a FIFO pool is sufficient.

## S7-C1 — Loop-order and cache-block experiments
- **Goal/type:** Start with separately selectable `ikj`; cache blocking is optional if loop reorder already meets the sufficient-benefit gate, otherwise use a bounded set of block sizes. *Correctness + Performance.*
- **Depends/files:** S6; CPU kernels/bench/tests.
- **Scope/not:** still one thread; no pool or handwritten SIMD.
- **Verify/baseline/metrics:** reference equality; median/GFLOPS versus CPU `ijk`; tile candidates recorded.
- **Accept/commit/risk:** choose stable default only if data supports it. `perf(cpu): add loop-reordered blocked matmul`. Risk: unequal numerical accumulation; use tolerance and label order.

## S7-C2 — Optional static parallel comparison
- **Goal/type:** Fixed partitioned `std::thread` GEMM with deterministic output ownership. *Correctness + Performance.*
- **Depends/files:** C1; parallel kernel/tests.
- **Scope/not:** deliberately creates threads per call as baseline; no persistent pool.
- **Verify/baseline/metrics:** thread counts 1/2/4/available, no overlap/race; speedup vs S6 `ijk` and vs C1.
- **Accept/commit/risk:** CPU-only tests pass under TSAN where available. `perf(cpu): add static threaded matmul baseline`. Risk: oversubscription.

## S7-C3 — Persistent thread pool and barrier
- **Goal/type:** Queue tasks, explicit shutdown, barrier/future completion. *Functional + Correctness.*
- **Depends/files:** C1 (C2 optional); `thread_pool.hpp/.cpp`, unit tests. Integrate disjoint row/tile tasks with C1 matmul through CPU backend, not only a standalone pool.
- **Scope/not:** FIFO/chunk scheduler first; no stealing.
- **Verify/baseline/metrics:** task exactly-once, exception/status propagation, repeated graph execution and matmul at 1/2/4/available capped host threads. Compare same C1 algorithm single-thread vs pool, plus S6 `ijk`; per-call threads only if C2 attempted. Record creation count/latency.
- **Accept/commit/risk:** no deadlock on zero/one task or shutdown. `feat(cpu): add persistent thread pool`. Risk: lifetime/shutdown races.

## S7-C4 — Optional work stealing (only after FIFO integration)
- **Goal/type:** Per-worker deque, bounded chunks, steal only when local queue empty. *Functional + Performance.*
- **Depends/files:** C3; pool/tests/bench.
- **Scope/not:** no lock-free requirement; correctness and observability first.
- **Verify/baseline/metrics:** imbalanced synthetic work, steal count, completion, latency versus FIFO; static comparison only if C2 was attempted.
- **Accept/commit/risk:** stable fallback FIFO remains configurable. `perf(cpu): add work-stealing chunk scheduler`. Risk: synchronization overhead may regress uniform GEMM.

## S7-C5 — CPU scaling report
- **Goal/type:** Analyze scaling, thread/power controls, and selected stable kernel. *Performance + Documentation.*
- **Depends/files:** C1/C3; include C2/C4 only if attempted; results/analyzer/docs/interview.
- **Scope/not:** no claim of GPU comparison.
- **Verify/baseline/metrics:** same host/build/input, raw samples, speedup/efficiency; retain regression rows.
- **Accept/commit/risk:** selected policy passes roadmap ≥1.05× CPU baseline gate; separate algorithm benefit from pool scaling, retaining small-size single-thread fallback when supported by data. No minimum pool scaling efficiency; stop at sufficient benefit. `bench(cpu): record parallel GEMM scaling`. Risk: noisy shared host; disclose variance.
