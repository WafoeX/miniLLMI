# Stage 6 — CPU backend

**Goal:** move reference operator execution behind the common backend contract. **Prerequisite:** S3; planner integration follows S5. **Gate:** CPU backend runs core graph correctly without model-specific kernels.

**Acceptance scope (declared before measurements): CPU-only, local named-host Release allowed; GPU/server not required.** See [CPU backend and frozen benchmark contract](../cpu_backend.md). S6 creates a baseline, not an optimization speedup claim: three independent 128/256/512/1024 baseline runs; graph dynamic/reuse pairs alternate order. FP64 oracle is untimed. Default S2 FP64 math and dynamic allocation remain stable. Formal gate status is recorded in [Stage 6 evidence](../stage6_report.md).

## S6-C1 — Backend interface and capability contract
- **Goal/type:** Define backend buffer/copy/prepare/execute interface and capability query. *Functional.*
- **Depends/files:** S1–S3; `backend.hpp`, `cpu_backend.hpp/.cpp`, tests.
- **Scope/not:** CPU backend only; scheduler/CUDA deferred.
- **Verify/baseline/metrics:** unsupported op/device diagnostics; N/A.
- **Accept/commit/risk:** graph executor can call a backend without including CPU kernel headers. `feat(cpu): add backend interface and CPU capability`. Risk: interface too CUDA-specific.

## S6-C2 — CPU core-op dispatch
- **Goal/type:** CPU ADD/MUL/COPY/MATMUL dispatch using S2 reference semantics. *Functional + Correctness.*
- **Depends/files:** C1/S2; CPU kernels/tests.
- **Scope/not:** single-thread FP32-accumulating scalar `ijk` performance baseline, no pool/SIMD hand code. Retain Stage 0 FP64-accumulating oracle for correctness only; neither rewrite it nor time it as the CPU performance baseline. Backend writes caller-provided output/workspace and reuses S2 semantics.
- **Verify/baseline/metrics:** operator fixtures and graph results; CPU GEMM baseline `ijk` GFLOPS/median.
- **Accept/commit/risk:** dispatch validates device/contiguity and returns errors, never silent copies. `feat(cpu): dispatch core operators through backend`. Risk: duplicate reference/kernel logic; keep one source of semantics.

## S6-C3 — CPU transformer primitive dispatch
- **Goal/type:** Add backend hooks/placeholders for later RMSNorm/Softmax/Embedding/SwiGLU; implement only when S12 starts. *Functional.*
- **Depends/files:** C1/S2-C4; capability tests.
- **Scope/not:** interface contract, not early transformer implementation.
- **Verify/baseline/metrics:** exact unsupported status until implementation; N/A.
- **Accept/commit/risk:** no model-level special dispatch. `feat(cpu): reserve transformer operator dispatch contract`. Risk: premature ABI; keep internal.

## S6-C4 — CPU benchmark harness
- **Goal/type:** Extend provenance/raw CSV framework for CPU GEMM and graph runs. *Performance + Integration.*
- **Depends/files:** C2; `bench/bench_cpu.cpp`, scripts/analyzer/results.
- **Scope/not:** same inputs/control fields; no comparison to GPU or timed FP64 oracle. Predeclared local CPU-only scope; zero pool/hand-SIMD/optimization candidates.
- **Verify/baseline/metrics:** timed square 128/256/512/1024 scalar `ijk`; small/non-square/boundary correctness tests. 2048/4096 CPU timings are optional to avoid an impractical scalar test budget. Record warmup/iterations, threads, GFLOPS and correctness; roadmap paired-run protocol applies.
- **Accept/commit/risk:** baseline is immutable and distinguishable from Stage 0 CPU oracle. `bench(cpu): add scalar GEMM baseline runner`. Risk: compiler auto-vectorization must be reported, not suppressed.
