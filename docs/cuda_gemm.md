# Stage 9 CUDA SGEMM contract

Stage 9 preserves [`sgemm_v0_naive.cu`](../backend/cuda/sgemm_v0_naive.cu) verbatim and adds a single explicitly selectable custom candidate. It does not change the Stage 0 generator, CPU FP64 reference, mixed error rule, FP32 pedantic cuBLAS comparison, or Stage 0 baseline CSV schema.

## Registry and dispatch

`stage0::GemmKernel` is the common selection boundary for `bench_gemm` and `CudaBackend`:

| Canonical name | Selection aliases | BM×BN×BK | TM×TN | threads | vector | static shared bytes |
|---|---|---:|---:|---:|---:|---:|
| `sgemm_v0_naive` | `naive`, `v0` | 1×1×1 | 1×1 | 256 | 1 | 0 |
| `sgemm_v1_tiled` | `tiled`, `v1` | 16×16×16 | 1×1 | 256 | 1 | 2048 |
| `cublas` | `cublas` | 0×0×0 | 0×0 | 0 | 0 | 0 |

`CudaBackend(..., CudaMatmul::Stage9Tiled)` calls the same registry path as the benchmark. The stable default remains `CudaMatmul::Stage0Naive`; there is no automatic kernel selection or hidden fallback.

## V1 semantics and safety

`sgemm_v1_tiled` has 16×16 threads. Each block cooperatively loads a 16×16 A tile and a 16×16 B tile into static shared memory, zero-fills M/N/K out-of-range lanes, synchronizes before and after each tile’s accumulation, then stores only in-range C elements. It has one accumulator per thread; it intentionally has no register output tile, vector load, double buffer, Tensor Core, input-padding requirement, or alignment cast.

The GPU tests cover registry lookup/configuration, v0/v1/cuBLAS agreement with the unchanged CPU reference, tiny and rectangular K-tail cases, zero inputs, repeated launches, input offsets (including a non-16-byte-aligned float offset), output guards, Event timing, and runtime-backend dispatch. Device launch resources are checked against queried limits before a timed benchmark launches.

## Versioned evidence schema

Stage 0 continues to use its untouched `results/gemm/baseline.csv` schema and [`tools/analyze_results.py`](../tools/analyze_results.py). `bench_gemm --experiment stage9` defaults to `results/gemm/stage9.csv` and writes the versioned `stage9-gemm-v1` schema: the existing provenance/timing/error fields plus `threads`, `vector`, `shared_bytes`, and `kernel_order`.

`tools/run_cuda_gemm_stage9.py` requires a clean committed source tree, makes a clean Release SM75 build, runs GPU CTest, then captures three independent paired all-kernel sequences over 512/1024/2048/4096. It alternates v0/V1 order and uses 10 warmups and 50 CUDA Event samples per kernel/shape. To fit the recorded two-worker T4/Colab host without weakening the protocol, it computes each full single-thread FP64 `ijk` oracle once in a new `reference-cache/` inside that run directory, then reuses that immutable full output only for the later paired repetitions. Every kernel in every repetition still has its own initial and final full-output comparison; the cache, raw samples, correctness records, command logs, device snapshots, manifest, and SHA-256 digests are retained. `tools/analyze_cuda_gemm.py` recomputes timing/GFLOPS, validates raw/config/provenance equality, and derives the required median paired geometric-mean v0/V1 speedup. It never treats cuBLAS as a target.

Stage 9 remains **unaccepted** until a real T4 run passes correctness and the roadmap’s ≥1.05× gate in all three paired runs. C3–C6 are intentionally not implemented: they are optional and must remain skipped if V1 passes.
