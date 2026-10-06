# Stage 8 CUDA backend contract

`CudaBackend` is an explicit CUDA:0 backend with one backend-owned nonblocking stream and a cuBLAS handle. It uses the existing `Tensor`/shared `Storage` representation; CUDA allocations are wrapped in `Storage` deleters that select the owning device and call `cudaFree`. Backend teardown synchronizes owned stream work before destroying the handle and stream.

Stage 8 supports contiguous FP32 H2D, D2H and non-aliased D2D copies, plus rank-2 MATMUL through the preserved Stage 0 naive kernel or explicit cuBLAS selection. Pageable-host copies synchronize before return: this makes source/destination lifetime and completion explicit, but makes **no overlap claim**. Strided tensors require an explicit MATERIALIZE path; there is no hidden host fallback or scheduler copy insertion.

`CudaPlannedAllocationProvider` consumes the existing S5 homogeneous-device memory plan and owns a single CUDA reservation. It is explicit prepare/replan state: the graph must remain structurally valid and CUDA FP32; returned output aliases pin the reservation; execute-time intermediate allocation count is zero. The regular `PlannedAllocationProvider` remains CPU-only and rejects CUDA plans. CUDA Stage 8 implements homogeneous MATMUL graphs only; ADD/MUL, mixed placement and scheduler insertion remain later-stage work.
