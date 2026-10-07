# Stage 9 acceptance report

**Status: complete — S9-C1/C2 pass.** The fixed 16×16×16 shared-memory V1 SGEMM passed the required T4 correctness and paired-performance gate. This accepts Stage 9 only; Stage 10 still requires separate Nsight evidence. `CudaMatmul::Stage0Naive` remains the stable default: V1 is explicitly selectable, with no automatic policy change.

## Accepted T4 evidence

[`results/gemm/stage9/20261007T011346974425Z-10247`](../results/gemm/stage9/20261007T011346974425Z-10247/) was captured from clean source **`3cd5a66dcaf7dc3234b9780f90d85dc6f32ab0bd`** (digest `7eeb66acc223c51811b79b925f398a8a82750758f11b08f115e0841b6caaabb0`) and committed separately in **`bcac7590ef6a8c80d4fd1d0963fe2b38237b252e`**.

- Tesla T4 (compute capability 7.5, 40 SMs, 15,637,086,208 bytes global memory); CUDA/nvcc 13.0.88.
- Fresh Release `ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, `BUILD_TESTING=ON` configuration and build passed.
- GPU CTest: **2/2 passed** — `cuda_backend` and V1 registry/boundary/tail/zero/offset/guard/event/dispatch coverage in `cuda_gemm`.
- Three alternating paired v0/V1/cuBLAS runs covered 512/1024/2048/4096, FP32 pedantic math, seed 42, 10 warmups and 50 Event samples per kernel/shape. Initial and final full-output checks passed for every row.
- Per-run geometric-mean v0/V1 speedups were **1.678212×**, **1.543855×**, and **1.479965×**. Their median is **1.543855×**; every run and every individual size won (smallest paired ratio: **1.343504×**), exceeding the required 1.05× gate.
- The passed manifest records identical source-before/source-after provenance, four complete FP64 oracle-cache files, and SHA-256 hashes for 89 retained artifacts. The committed raw samples and derived `analysis.json`/`summary.md` remain the evidence source.

## Decision and retained failure

S9-C1/C2 are accepted. V1 is the selected Stage 9 custom candidate; C3–C6 are `skipped_optional` because the required gate passed. Do not claim that V1 approaches cuBLAS, and do not use profiler timing as benchmark data.

The earlier failed attempt at [`20261007T004627006659Z-2068`](../results/gemm/stage9/20261007T004627006659Z-2068/) is retained. Its build, GPU tests, and benchmark repetitions completed, but its analyzer failed before derived output because it incorrectly required raw timing-row derived fields to equal summary-row fields. The later source commit `3cd5a66` fixes that analyzer check; the accepted run above used the corrected source.

See [the CUDA SGEMM contract](cuda_gemm.md), [the Stage 9 task](tasks/stage-09-cuda-gemm.md), and [the Stage 9 T4/Colab procedure](stage9_colab.md).
