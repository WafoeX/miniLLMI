# Stage 1–19 implementation blueprint

> **Status.** Stage 0 is accepted. Its source identity is `f7aaec4` / digest `cc1414…20dc0c`; reproducible T4 results and profiler evidence are committed through `f0f1de7`. Stage 1 Tensor/storage, Stage 2 operators and Stage 3 validated CPU graphs are implemented; see [the Tensor contract](tensor.md), [Stage 1 evidence](stage1_report.md), [operator contracts](operators.md), [Stage 2 evidence](stage2_report.md), [graph contracts](graph.md) and [Stage 3 evidence](stage3_report.md). Stage 4 host arena/provider is implemented; see [arena contracts](arena.md) and [Stage 4 evidence](stage4_report.md). Stage 5 CPU lifetime/slot planner and prepared execution are implemented and pass the required memory gate; see [planner contracts](planner.md) and [Stage 5 evidence](stage5_report.md). S5-C4 inplace is `skipped_optional`; dynamic remains default and measured small-graph latency regresses. Stage 6 common backend/CPU dispatch and immutable FP32 ijk baseline are implemented and CPU-only accepted; see [CPU backend contracts](cpu_backend.md) and [Stage 6 evidence](stage6_report.md). Stage 7 loop-reordered CPU GEMM and FIFO pool are implemented and CPU-only accepted; see [the CPU parallel contract](cpu_parallel.md) and [Stage 7 evidence](stage7_report.md). Default FP64 math/dynamic allocation remain stable; transformer primitives stay Unsupported. Stage 8 CUDA storage/copies/Stage 0 dispatch and homogeneous planned MATMUL graphs are implemented and T4 accepted; see [the CUDA backend contract](cuda_backend.md) and [Stage 8 evidence](stage8_report.md). Stage 9 C1/C2 registry plus V1 shared-tile SGEMM are **T4 accepted**: clean-source correctness, three paired 512–4096 runs and the ≥1.05× CUDA gate pass; see [the CUDA SGEMM contract](cuda_gemm.md), [acceptance evidence](stage9_report.md), and [Colab procedure](stage9_colab.md). C3–C6 are `skipped_optional` because V1 passed. Stage 10 C1–C3 are **T4 accepted**: a clean-source 4096³ V0/V1 Nsight Systems/Compute comparison, hashed external profiler artifacts, and the Stage 9 repeat-based explicit V1 selection record pass; see [the profiling protocol](profiling.md) and [acceptance report](stage10_report.md). Stage 11–19 remain **Proposed Design**, not implemented functionality.

## 1. Architecture and invariants

```text
Tokenizer + model loader
          │
      Decoder model ── creates ──> Graph (nodes, tensors, operator descriptors)
                                      │
                          Scheduler / copy insertion
                                      │
                                Memory planner
                         ┌────────────┴────────────┐
                    CPU backend                CUDA backend
                         │                          │
                  CPU kernels/pool        CUDA buffers/kernels/cuBLAS
                         └────────── Tensor + shared Storage ──────────┘
```

**Layering rules**

1. `Tensor` owns metadata only; shared `Storage` owns a CPU or CUDA buffer. Views share storage and change only metadata/offset.
2. Operators describe semantics, shapes, attributes, and backend-neutral inputs/outputs. Backends own execution; model code must not call CUDA kernels directly.
3. Graph owns node/tensor topology, planner owns intermediate storage assignment, scheduler owns backend placement and explicit copies.
4. Transformer code only composes existing operators/graph nodes. It may not introduce a second tensor allocator, matmul stack, or CUDA dispatch path.
5. Runtime defaults always use the last verified stable implementation. Experimental kernels and policies remain callable explicitly and retain their data even when slower.

## 2. Dependency DAG and execution order

```text
S1 -> S2 -> S3 -> S4 -> S5
             └-> S6 -> S7
S1 + S3 + S6-C1 -> S8 (C4 additionally needs S5) -> S9 -> S10
S5 + S6 + S8 -> S11 -> S12 -> S13 -> S14
S13 -> S15
S15 + stable matmul -> S16-C1..C3
S14 + S15 -> S17 (INT8 mode additionally needs S16-C3)
S16-C3 + S17-C2 -> S16-C4
S7 + S9/S10 + S16-C4 + S17 -> S18 -> S19
```

Execute one Change at a time. The DAG describes readiness, not authorization for concurrent implementation. A stage gate covers only its required Changes; explicitly skipped optional Changes never block a downstream dependency. S6 can follow S3 before S4/S5; S15 needs S13 (not merely S1/S2). CLI float mode must not wait for quantization benchmarking, and INT8 benchmarking must not block that CLI. Each experiment owns a unique run directory; never concurrently append an aggregate CSV.

## 3. Stage map

| Stage | Goal | Principal deliverable | Gate |
|---|---|---|---|
| 1 | Tensor/storage | views, strides, CPU/CUDA storage API | view/lifetime tests |
| 2 | Operators | typed backend-neutral op descriptors | shape/error tests |
| 3 | Graph | validated DAG + sequential executor | topology/error tests |
| 4 | Arena | verified aligned reuse arena (first fit is sufficient) | allocator invariants |
| 5 | Planner | graph liveness → reusable arena slots | zero execute-time intermediate buffer allocations |
| 6 | CPU backend | correct scalar CPU operator path | op/CPU-GEMM correctness |
| 7 | CPU optimization | loop reorder + explicit FIFO persistent pool | CPU-only measured improvement vs CPU v0 |
| 8 | CUDA backend | storage, copies, Stage 0 dispatch, homogeneous planned MATMUL graph | T4 CUDA integration tests |
| 9 | CUDA SGEMM | one improved custom kernel under common dispatch | correctness + improvement vs preserved v0 |
| 10 | Profiling/tuning | v0/selected-kernel Nsight comparison | one explained selection; bounded tuning |
| 11 | Scheduler | deterministic placement and explicit copies | mixed-backend graph correctness |
| 12 | Transformer ops | independently tested primitive ops | reference-vector agreement |
| 13 | Decoder | tiny decoder using graph/runtime | deterministic logits |
| 14 | KV cache | prefill/decode split and cache benchmark | speedup + cache correctness |
| 15 | Tokenizer/model file | loader + tokenizer interfaces | tiny model round-trip |
| 16 | INT8 weight-only | offline quantizer and correct dequant path | bounded error + smaller artifact; throughput diagnostic |
| 17 | CLI | reproducible end-to-end generation | scripted smoke + model run |
| 18 | Final evaluation | frozen release-candidate evidence | one tested source identity; separate result commits |
| 19 | Documentation | README + interview evidence map | claim-to-evidence audit |

Detailed, executable task books are in [`docs/tasks/`](tasks/README.md).

## 4. Baseline matrix

| Module | Baseline | Optimization / comparison | Metrics | Result path |
|---|---|---|---|---|
| CPU GEMM | frozen S6 single-thread FP32 `ijk` | loop order/blocking + FIFO pool; other schedulers optional | median ms, GFLOPS, speedup, threads | `results/cpu/` |
| CUDA GEMM | preserved Stage 0 v0, rerun with candidate | shared tile; further variants only if needed; cuBLAS is a reference ceiling, not a target | median ms, GFLOPS, v0 speedup, cuBLAS %, errors, resources | `results/gemm/` |
| Allocator | S3 last-use dynamic allocation; additionally prepare-only no-reuse slots for capacity comparison | arena + planned reuse; inplace optional | buffer alloc/free count, peak live/reserved bytes, reuse, latency | `results/allocator/` |
| Scheduler | explicit/manual placement with same kernels | capability policy + copy insertion; segments optional | copy count/bytes, switches, graph latency | `results/scheduler/` |
| KV cache | full-prefix recomputation | K/V cache prefill + decode | prefill ms, decode ms/token, tok/s, bytes | `results/kv_cache/` |
| INT8 | FP32 (or separately labelled FP16) weights | per-output-channel INT8 weight-only | bytes, compression, tensor/logit errors, tok/s | `results/quant/` |

## 5. Experiment discipline

Every result is appended, never manually rewritten, and includes timestamp, run ID, tested commit/digest, dirty status, build type, host/compiler, backend/device, configuration, correctness result, and raw samples where timing is measured. Comparisons require equal input, shapes, dtype/math mode, tolerance, warmup/iterations, device and timing boundary; INT8 is a labelled dtype exception with fixed error limits. A slower but correct experiment is an **implementation pass / performance miss**, retained in the data; it does not satisfy a required improvement gate. Historical Stage 0 data is provenance, not a substitute for a paired v0 rerun on the current server.

| Experiment | Inputs | Fixed controls | Variable | Outputs |
|---|---|---|---|---|
| CPU GEMM | timed 128/256/512/1024 square; small/non-square correctness | FP32/input/compiler/host | algorithm, threads | `results/cpu/baseline/<run-id>/cpu_gemm.csv` |
| CUDA sweep | 512–4096, same Stage 0 generator | T4, Release, FP32, Event, 10/50 | kernel config | `results/gemm/baseline.csv`, raw runs |
| Allocator | fixed synthetic and model graphs | graph/input/lifetimes | allocation policy | `results/allocator/allocator.csv` |
| Scheduler | fixed mixed graph | graph/input/device | policy/segments | `results/scheduler/scheduler.csv` |
| KV | contexts 128/256/512 + 32 fixed continuation IDs; 1024 optional | model/seed/batch/placement | cache off/on | `results/kv_cache/kv_cache.csv` |
| Quant | fixed weight/model/input suite | quant scheme/version | FP versus INT8 | `results/quant/quant.csv` |
| End-to-end | fixed prompts/max tokens/sampling | RC commit/model | CPU/CUDA/quant mode | `results/inference/inference.csv` |

### Sufficient improvement and stopping rule

Before measuring, freeze the candidate, timing boundaries, workloads, tolerances and thread/placement policy. Use at least **3 independent paired runs**; each run stores baseline and candidate raw samples (CUDA uses Stage 0 10/50). CPU/graph/model timing uses monotonic wall clock, at least 3 warmups and 10 samples; small workloads may use a fixed recorded batch of executions per sample. Synchronize GPU work before stopping graph/model wall time. Alternate baseline/candidate order across runs, disclose contention and variance, and never drop a slow/failed run. Invalid environment runs remain stored, with a reason and a replacement paired run.

For each workload, compute the median across independent **paired median-latency ratios** (`baseline_ms / candidate_ms`). For GEMM, summarize workloads by geometric mean, not by pooling different-size samples. These modest gates are engineering checks, not statistical significance claims:

| Required benefit | Sufficient gate (plus all correctness tests) |
|---|---|
| CPU GEMM | ≥1.05× geometric-mean speedup over S6 `ijk` on 128/256/512/1024, candidate wins in every retained valid run-level geometric mean (at least 3 runs); record single-thread algorithm and same-algorithm pool scaling separately |
| Custom CUDA GEMM | ≥1.05× geometric mean over unmodified v0 on 512/1024/2048/4096, candidate wins in every retained valid run-level geometric mean (at least 3 runs); no requirement to approach or beat cuBLAS |
| Planner | execute-time intermediate backing-buffer allocations reduce from >0 to 0 vs last-use dynamic execution; reusable prepared arena capacity ≤80% of prepare-only no-reuse slot capacity on a predeclared reuse-rich graph; no latency speedup required |
| KV cache | ≥1.05× median paired decode speedup at context 512 with 32 fixed continuation IDs, and each independent paired run >1×; other contexts reported, not cherry-picked |
| INT8 | quantized eligible-weight payload **including scales** ≤35% of FP32 bytes, and frozen tensor/logit error tests pass; total file size, resident/dequant workspace and throughput also reported, with no required throughput gain |

CPU/CUDA candidate regressions >10% on any mandatory size require a predeclared simple stable fallback for that size; publish both candidate-only and actually selected-policy results. The selected policy must also pass the aggregate gate; do not conceal regressions behind the fallback. If variation obscures a 5% gain, label it inconclusive and repeat under controlled conditions rather than declare success. Functional, performance and optional-skip statuses are separate.

Once these benefits pass, **stop optimization** and proceed to integration. No target for maximum GFLOPS, occupancy, cuBLAS percentage, CPU efficiency, allocator latency or text quality. A failed gate allows one bounded follow-up experiment set (at most 4 declared configs); continued misses require an explicit scope/acceptance decision, not an endless sweep or a silent lower threshold. Stage dependencies still respect their required acceptance gates; a performance miss is not stage completion. Independent Changes whose listed prerequisites already pass may proceed, but final acceptance requires every required benefit.

## 6. Git and server protocol

For one Change, create `feat/<area>-<short-name>` (experiments use `perf/` or `bench/`), make narrow commits, and run applicable local tests before push. Server procedure is always: `git pull --ff-only`, record `git rev-parse HEAD`, clean Release build, unit/correctness tests, then only the Change's benchmark/profile command. Commit textual/raw results separately from tested code; retain the tested-code provenance in CSV. Do not mix profiler timing with unprofiled benchmark data. `results/` and `tools/analyze_results.py` extend Stage 0 rather than being replaced.

Local CPU tests are mandatory for S1–S7, S11–S19 where applicable. T4 validation is mandatory for all CUDA storage/copy/dispatch/kernels/profiling and all CUDA-backed integration, KV, quant, and final performance claims.

## 7. Milestones

| Milestone | Entry | Complete when | Demo |
|---|---|---|---|
| M1 Runtime foundation | Stage 0 accepted | S1–S3 pass | validated CPU graph of add/matmul |
| M2 Memory + CPU runtime | M1 | required S4–S7 Changes pass | planned graph has no intermediate buffer alloc during execute; improved CPU GEMM |
| M3 CUDA runtime | M2 | S8 + stable CUDA graph path | CPU↔CUDA graph/copy correctness |
| M4 CUDA performance | M3 | S9 benefit gate + required S10 evidence pass | one improved custom GEMM versus v0; cuBLAS for context |
| M5 Transformer runtime | M2/M3 + scheduler | S11–S15 pass | tiny decoder prefill/decode from model file |
| M6 Optimized inference | M5 | S16–S17 pass | INT8 and float CLI generation comparison |
| M7 Evaluation/package | M6 | S18–S19 pass | tagged RC report and interview trail |

## 8. Scope control

**Must:** one tensor/storage representation; core operators and validated graph; simple arena/liveness reuse; scalar CPU baseline + improved CPU path + FIFO pool; CUDA storage/copies and one improved custom SGEMM plus unchanged v0/cuBLAS; deterministic scheduler/copy insertion; FP32 tiny decoder on CPU and an explicit T4 mixed path (CUDA projections, unsupported primitives on CPU); KV cache; custom model loader + byte tokenizer; correct INT8 artifact/dequant path; greedy CLI; paired results and short evidence docs.

**Optional, never upstream gates:** inplace (S5-C4), per-call thread baseline (S7-C2), work stealing (S7-C4), additional CUDA variants (S9-C3..C6), extra tuning beyond one profile comparison, scheduler segmentation (S11-C3), BPE (S15-C4), fused INT8/INT4 (S16-C5), sampling beyond greedy, full CUDA transformer primitive coverage, FP16 and external pretrained-model loaders. Record `skipped_optional` and rationale, not an implementation pass. Do not invent a v6 task.

**Out of scope:** multi-GPU/distributed execution, batching server, paged/Flash attention, MoE, training, tensor/pipeline parallelism, general-language quality claims from random tiny weights, or GGUF/TinyLlama/llama.cpp compatibility without a separate approved scope change.

### Frozen tiny-model contract

Before S2-C4 fixtures, document a batch=1, bias-free FP32 MHA decoder: 2 layers, hidden=64, heads=4, head_dim=16, SwiGLU intermediate=128, vocab=258, max_seq=1088 (1024+32 fits), RMSNorm epsilon=1e-5, RoPE base=10000 with explicit rotation convention. Byte IDs 0..255 plus BOS=256/EOS=257 are the initial tokenizer. Use a versioned deterministic offline weight generator; store actual tensors/fixture checksums so C++ does not depend on reproducing another language's RNG. Declare projection weight layout and quantization output-channel axis once. Random weights prove runtime behavior, **not language quality**; external trained weights are optional.

S2/S12 must explicitly cover integer token IDs, graph-visible slices/transposes/materialization, per-head 2D attention matmuls (batched matmul is optional), cache append/copy writes and causal position offsets. Full CUDA primitive coverage is not required: capabilities determine explicit copies, never hidden model-level kernels. Dynamic shapes use explicit prepare/replan; max-capacity cache and active sequence length must not cause reads of uninitialized positions.

## 9. Final acceptance

The project is complete only when a tagged Release Candidate **source commit** reproducibly builds Release on CPU and T4; all required tests pass; separate result commits preserve that tested source identity; float and INT8 tiny-model greedy CLI generation work; and all five sufficient-benefit gates in §5 pass. Scheduler/full-CUDA/INT8 throughput gains are not completion requirements. One v0/selected-kernel Nsight comparison explains the principal CUDA choice. Final tables are generated/traceable to raw data, and concise evidence pages map each actual claim to code, commands, artifacts and limits. Optional skips are listed explicitly; required missing/failed/inconclusive results block completion, rather than being labelled success.
