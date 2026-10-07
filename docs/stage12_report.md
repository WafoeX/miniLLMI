# Stage 12 — accepted transformer operators

**Stage 12 is complete to its required C1–C5 correctness/integration gates.**
It adds only the declared primitive/runtime path: no decoder, loader, KV-cache,
performance result, hidden CUDA transformer kernel, or language-quality claim.

## Source and evidence

- Tested source: `ff3c4cde70becb7a8778daa1947a761c35bdbe31`
  (`test(ops): add transformer conformance matrix`).
- Separate T4 result commit:
  [`bf663b08bb0f54acb0fdf8565ca91551ff640051`](https://github.com/WafoeX/miniLLMI/blob/bench/stage12-t4-conformance/results/transformer/stage12-c5/colab-t4.log).
  Its sole parent is exactly the tested source commit, independently verified
  after HTTPS fetch. The captured shell variable was blank in the log, but the
  result commit parent supplies the immutable source identity; no metric or
  benchmark interpretation depends on it.
- T4: Tesla T4, NVIDIA driver `580.82.07`, CUDA `13.0`; no running GPU
  processes were reported before the retained capture.

## Required Change acceptance

| Change | Evidence | Status |
|---|---|---|
| C1 RMSNorm and SiLU/SwiGLU | frozen FP64-generated reference vectors; finite/prewrite/no-backing-allocation tests | accepted |
| C2 causal Softmax | stable max-subtracted, causal-offset, all-masked and masked-nonfinite tests | accepted |
| C3 RoPE and embedding | interleaved RoPE/reference vectors plus checked INT32 token bounds | accepted |
| C4 attention lowering | graph-visible per-head narrow/materialize/reshape/transpose/MATMUL/MUL/Softmax/MATMUL and checked persistent-range COPY writes; attention fixture agrees | accepted |
| C5 conformance matrix | local CPU Release 37/37 CTests; clean T4 `gpu` 5/5 and `transformer` 3/3 CTests | accepted |

`cuda_transformer_ops` is the required mixed-path test. It asserts a CUDA
projection MATMUL, exactly two H2D inputs plus one D2H result transfer, and CPU
Softmax/RMSNorm thereafter via the existing scheduler, graph executor, planner,
Tensor and Storage layers. It does not claim CUDA coverage for the other
primitives, nor introduce a model-level copy/kernel path.

The retained T4 log has two successful passes: GPU-labelled tests
`scheduler_benchmark_workloads`, `cuda_transformer_ops`, `cuda_scheduler`,
`cuda_backend`, `cuda_gemm` (5/5); and transformer-labelled tests
`transformer_ops`, `attention_graph`, `cuda_transformer_ops` (3/3).

## Limits and next stage

Stage 12 is functional conformance only. No timing was recorded or implied;
CUDA primitive coverage beyond projection MATMUL/COPY remains unsupported by
design. The stable defaults remain FP64 CPU reference math, dynamic allocation
and CUDA v0. Stage 13 remains the next unimplemented dependent Change.

See [the primitive task book](tasks/stage-12-transformer-ops.md),
[operator contract](operators.md), [T4 procedure](stage12_colab.md), and
[roadmap](roadmap.md).
