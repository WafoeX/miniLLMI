# Stage 8 — CUDA backend abstraction

**Goal:** make Stage 0 GEMM a runtime backend capability rather than benchmark-only code. **Prerequisite:** S1–S3 and S6-C1. **Gate:** T4 integration tests prove buffer/copy/dispatch correctness.

## S8-C1 — CUDA storage and RAII
- **Goal/type:** Implement CUDA `Storage` allocator/deleter, stream ownership, checked CUDA status conversion. *Functional + Correctness.*
- **Depends/files:** S1/S6; `cuda_backend.hpp/.cu`, tests.
- **Scope/not:** no caching allocator; use one backend-owned stream initially. Document async completion and storage retention until queued work completes. Teardown waits for owned work before freeing buffers/stream; no accidental per-op global synchronization.
- **Verify/baseline/metrics:** allocation failure injection, RAII teardown, T4 leak/error tests; N/A.
- **Accept/commit/risk:** device identity/capacity preserved; host-only build remains CUDA-off compatible. `feat(cuda): add CUDA storage and error handling`. Risk: context teardown order.

## S8-C2 — Explicit copy operations
- **Goal/type:** H2D/D2H/D2D COPY with stream/event contract and contiguity checks. *Functional + Correctness.*
- **Depends/files:** C1/S1-C5; CUDA copy code/tests.
- **Scope/not:** caller requests copies; scheduler insertion deferred. Initially support contiguous transfers; strided input requires an explicit materialization node. Pageable-host transfers may synchronize and must be labelled; do not promise async overlap without pinned storage. Source/destination remain alive until stream/event completion.
- **Verify/baseline/metrics:** round trip, nonzero offset, D2D alias prohibition, async completion; bytes/time recorded only in copy benchmark.
- **Accept/commit/risk:** no implicit host fallback; synchronization ownership documented. `feat(cuda): add explicit tensor copy operations`. Risk: lifetime while async transfer is pending.

## S8-C3 — CUDA operator dispatch and Stage 0 adapter
- **Goal/type:** Register CUDA MATMUL dispatch adapter to v0/cuBLAS and COPY; unify error/correctness contracts. *Integration + Correctness.*
- **Depends/files:** C2/S2/S6; backend/operator adapter/tests.
- **Scope/not:** no optimized v1+ yet; Stage 0 kernels remain behaviorally unchanged.
- **Verify/baseline/metrics:** graph CPU input→H2D→matmul→D2H matches CPU; baseline Stage 0 v0.
- **Accept/commit/risk:** runtime GEMM uses same kernel source and records selected implementation. `refactor(cuda): dispatch Stage 0 GEMM through backend`. Risk: accidental benchmark-path divergence.

## S8-C4 — CUDA graph-executor integration
- **Goal/type:** Execute homogeneous CUDA graph with planned CUDA storage where supported. *Integration + Memory.*
- **Depends/files:** C3/S5; executor/planner/tests.
- **Scope/not:** homogeneous CUDA supported-op graph (e.g. two 2D MATMUL nodes), not a claim of CUDA ADD/MUL/transformer support. Prepare one device arena Storage and checked views using the S5 byte-slot plan, with CUDA allocation supplied by backend; no second device planner. Scheduler/mixed placement deferred.
- **Verify/baseline/metrics:** fixed graph output, zero execute-time intermediate `cudaMalloc` after prepare; count allocations/copies and workspace. Create/warm cuBLAS handles/workspace before timing and report backend-owned memory separately, not as invisible allocation.
- **Accept/commit/risk:** plan is device-specific and invalidation safe. `feat(cuda): execute planned CUDA graph`. Risk: stream synchronization bugs.

## S8-C5 — T4 validation artifact
- **Goal/type:** Server build/test/copy/GEMM evidence. *Correctness + Integration.*
- **Depends/files:** C1–C4; scripts/results/docs.
- **Scope/not:** no performance optimization claim.
- **Verify/baseline/metrics:** Release, CUDA tests, raw provenance, leak/error status; baseline N/A.
- **Accept/commit/risk:** result commit is separate from code commit. `test(cuda): record backend integration validation`. Risk: toolkit mismatch; preserve failure logs.
