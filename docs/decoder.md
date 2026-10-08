# Tiny decoder contract — Stage 13

Stage 13 assembles the frozen version-1 tiny model from existing Tensor, Graph,
planner, backend and scheduler layers. It adds no tokenizer/model-file parser,
KV cache, autoregressive loop, fused attention kernel or language-quality claim.

## Configuration and parameters

`model::DecoderConfig::tiny()` is batch 1, two layers, hidden 64, four MHA
heads of width 16, SwiGLU width 128, vocab 258 and maximum sequence 1088.
RMSNorm epsilon is `1e-5`; RoPE is interleaved with base 10000. Bias, GQA/MQA,
batch greater than one, inconsistent head geometry, odd rotary width and invalid
numeric constants are rejected.

`model::ParameterTable` binds the 21 canonical CPU FP32 contiguous tensors by
name and exact shape. It consumes the committed `tiny-weights-v1.bin` bytes in
tests but is not a file loader; the Stage 15 format remains unimplemented.
Projection layout is `W[in,out]`.

## Graph lowering

`build_decoder_block` emits pre-norm
RMSNorm→Q/K/V→RoPE→causal attention→O projection→residual→RMSNorm→
gate/up→SwiGLU→down→residual. There is no opaque `ATTENTION` node. Each head
uses explicit NARROW, MATERIALIZE, RESHAPE, transpose, QKᵀ, same-shaped scaling,
Softmax and probability×V nodes. Instead of adding a new concat operator, each
head context multiplies its corresponding row range of `O`; those projected
`[T,64]` results are added. This is algebraically identical to
`concat(heads) @ O` and keeps every intermediate graph-visible and plannable.

`build_decoder_prefill` adds INT32 embedding, two blocks, final RMSNorm and LM
head, yielding logits `[T,258]`. It builds and freezes one shape-specific graph.
Changing active sequence length requires a new graph and allocation provider;
a stale plan is rejected before execution. For a fixed shape, callers may
mutate the shared token-ID storage and reuse the prepared provider after prior
output aliases are released.

## Placement and memory

Learned rank-2 projections carry a placement hint. CPU execution uses the
stable FP64-accumulating reference backend. On T4, the Stage 11 scheduler places
exactly 21 learned projections on CUDA v0 and inserts explicit transfers around
CPU-only Embedding, RMSNorm, RoPE, Softmax, SwiGLU and ordinary attention
operations. Model code never calls a backend, CUDA API or copy routine.

The dynamic executor remains the default baseline. Explicit
`PlannedAllocationProvider`/`ScheduledAllocationProvider` preparation owns
intermediate backing; warmed execution must report zero backing allocations.
Persistent parameters, token IDs, fixture oracle and returned logits are
accounted separately from planned intermediate capacity.

## Fixtures and tolerances

`tools/generate_decoder_fixture.py` is a stdlib-only scalar oracle with explicit
FP32 rounding at every runtime-equivalent write. It consumes the frozen Stage 12
weight bytes and writes `block0-output.bin` and `logits.bin` under
`tests/fixtures/decoder-v1/`, with hashes and token IDs `[256,0,1,257]` in
`fixture.json`. CPU tolerance is atol `2e-6`, rtol `2e-5`; the predeclared mixed
CUDA tolerance is atol `2e-4`, rtol `2e-3`. Random fixture weights validate
runtime behavior only.

Stage 13 latency is diagnostic. `bench_decoder` reports prepare separately,
warmed same-shape execution, and shape-change end-to-end time including graph
build/rewrite/prepare. No speedup gate or external-runtime comparison exists.
