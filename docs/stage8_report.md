# Stage 8 acceptance report

**Status: complete — S8-C1 through S8-C5 pass.** Stage 8 is a correctness/integration gate, not a CUDA performance claim.

## Delivered

- **C1:** `CudaBackend` owns CUDA storage, a nonblocking stream and cuBLAS handle. Storage lifetime is represented by the established shared `Storage` type; backend teardown synchronizes owned work before release.
- **C2:** Explicit contiguous FP32 H2D/D2H/D2D copies reject aliased D2D storage and strided bindings. Calls synchronize before return for pageable-host safety; no asynchronous-overlap claim is made.
- **C3:** MATMUL dispatch selects the unchanged Stage 0 naive adapter by default or explicit cuBLAS. CPU input → H2D → GEMM → D2H agrees with the reference.
- **C4:** `CudaPlannedAllocationProvider` uses the S5 homogeneous-device plan and a single CUDA reservation. The fixed two-MATMUL CUDA graph test verifies output and zero execute-time intermediate allocation count.
- **C5:** T4 Release evidence is retained below.

## T4 evidence

[`results/cuda/validation/20261006T134916590693Z-15776`](../results/cuda/validation/20261006T134916590693Z-15776/) was captured from clean source **`d8a05515a5c372b766afab901f6b66fdf44dcf5c`** (digest `be67a7170013373e62fe57aa23b9715deeac66e97db7c407d9c25febac75c6c6`), before the separate evidence commit `8a89676`.

- Tesla T4, compute capability 7.5, 15,637,086,208 bytes global memory; CUDA runtime/driver 13.0 and nvcc 13.0.88.
- Fresh Release `ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, `BUILD_TESTING=ON` configuration and build passed.
- GPU CTest: **2/2 passed** — preserved `cuda_gemm` and Stage 8 `cuda_backend`.
- The manifest records every command, exit code, source-before/after identity and SHA-256 for all raw logs. No performance or leak-rate metric is claimed.

Limits: this does not certify mixed-device scheduling, non-contiguous CUDA materialization, CUDA ADD/MUL, a caching allocator, performance versus cuBLAS, Linux sanitizers, or later CUDA transformer functionality. See [the CUDA backend contract](cuda_backend.md).
