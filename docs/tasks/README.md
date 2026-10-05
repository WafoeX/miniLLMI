# Change task books

Stage 1 is implemented (see [acceptance evidence](../stage1_report.md)); Stage 2–19 task books remain **Proposed Design**, not implemented functionality. Execute one Change at a time after its listed prerequisites pass. Follow the corrected DAG and sufficient-benefit gates in [the roadmap](../roadmap.md). An optional Change may be recorded as `skipped_optional` with a reason; downstream gates include required Changes only. Completing a stage does not require exploring every variant.

## Required Change record

Every implementation PR/commit must retain the Change ID and demonstrate scope, tests, acceptance and metrics. For an optimization, separately record **implementation status** (correct/failed), **benefit status** (pass/miss/inconclusive against the roadmap gate), and optional skip status. Preserve baseline implementations and all failed/slow runs. Passing correctness alone cannot close a required improvement gate; do not optimize further once the modest benefit gate passes.

Required benefits are CPU and custom CUDA GEMM speedup, planned intermediate allocation/memory reduction, KV decode speedup, and bounded-error INT8 weight compression. No requirement to beat cuBLAS, accelerate every module, reach ideal occupancy, or improve INT8 throughput. The minimum model/backend contract, paired-run protocol and metrics are defined centrally in the roadmap; task-specific statements below must not weaken them.

**Optional Changes:** S5-C4, S7-C2/C4, S9-C3..C6, S11-C3, S15-C4, S16-C5. Reports that depend on these include them only if attempted. S17 float CLI depends on S14/S15, not all of S16; S16-C4 uses S17-C2 only after S16-C3 exists.

**Allocation vocabulary:** count backing-buffer allocations/frees (CPU allocator and `cudaMalloc`) separately from C++ metadata/container heap activity. `prepare` may allocate; warmed `execute` must not allocate intermediate backing buffers. Inputs, weights, pinned outputs, persistent KV, copy buffers and backend workspaces are separately accounted; never subtract them from total resident memory without disclosure. Report peak live intermediate bytes, reserved arena capacity, and total resident bytes distinctly. Dynamic baseline frees dead intermediates at last use; keeping everything until graph end is not a fair memory baseline.

本轮问题与修订依据见 [任务书审查记录](../taskbook_review.md)。

## Task books

1. [Stage 1 — Tensor and Storage](stage-01-tensor.md)
2. [Stage 2 — Operator definitions](stage-02-operators.md)
3. [Stage 3 — Graph and sequential executor](stage-03-graph.md)
4. [Stage 4 — Arena allocator](stage-04-arena.md)
5. [Stage 5 — Graph memory planner](stage-05-planner.md)
6. [Stage 6 — CPU backend](stage-06-cpu-backend.md)
7. [Stage 7 — CPU parallel execution](stage-07-cpu-parallel.md)
8. [Stage 8 — CUDA backend](stage-08-cuda-backend.md)
9. [Stage 9 — CUDA SGEMM experiments](stage-09-cuda-gemm.md)
10. [Stage 10 — Profiling and tuning](stage-10-profiling.md)
11. [Stage 11 — Backend scheduler](stage-11-scheduler.md)
12. [Stage 12 — Transformer operators](stage-12-transformer-ops.md)
13. [Stage 13 — Decoder model](stage-13-decoder.md)
14. [Stage 14 — KV cache](stage-14-kv-cache.md)
15. [Stage 15 — Tokenizer and model file](stage-15-tokenizer-loader.md)
16. [Stage 16 — INT8 weight-only](stage-16-int8.md)
17. [Stage 17 — LLM CLI](stage-17-cli.md)
18. [Stage 18 — Final evaluation](stage-18-evaluation.md)
19. [Stage 19 — Documentation and interview trail](stage-19-documentation.md)

All performance data extends Stage 0 provenance/raw-result conventions without rewriting accepted Stage 0 data or its analyzer behavior. New result families have separate versioned schemas. GPU claims require T4; CPU-only experiments name their host explicitly. Source commits and later artifact commits are distinct; final data retains the tested source identity.
