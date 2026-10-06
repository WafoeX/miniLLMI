# Stage 7 CPU GEMM candidate and FIFO pool

Stage 7 is **CPU-only**; it does not require a GPU or T4 acceptance run. It preserves `src/runtime/cpu_scalar.cpp` as the immutable S6 `ijk` FP32 baseline. The explicit C1 candidate is `cpu-ikj-fp32-c1`: same row-major FP32 semantics, but `i → k → j`; caller-owned output is zeroed before multiplication so K=0 remains defined. It is never silently selected by `CpuBackend{}`; the stable default remains the FP64 reference path.

`cpu-ikj-fp32-c3-fifo` adds a persistent FIFO `ThreadPool`; row tasks have disjoint output ownership. The pool reports task failures/exceptions, serializes batches, accepts zero tasks and joins workers on destruction. Its creation is outside the GEMM timer. FIFO thread counts are explicit; no work stealing or per-call `std::thread` baseline was attempted.

`tools/run_cpu_parallel_benchmark.py` makes a fresh Release test + production build, runs three alternating v0/C1 paired sequences at 128/256/512/1024, and additionally runs FIFO at 1/2/4/available-capped threads. Timed GEMM retains the Stage 6 boundary: full backend execute including validation and finite checks, excluding allocation/input/oracle/CSV. `tools/analyze_cpu_parallel.py` verifies hashes, raw schema, initial/final oracle results and recomputes median timing/GFLOPS and paired geometric means.
