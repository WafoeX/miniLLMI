# Scheduler contract (Stage 11)

## C1 — Deterministic capability policy

`Scheduler` borrows the existing CPU backend and optionally one CUDA backend.
`place` is metadata-only: a requested registered backend wins only when its
operator/dtype capability and input layout are supported. Current CUDA compute
coverage is contiguous FP32 rank-2 MATMUL plus COPY; no CUDA ADD/MUL/transformer
primitive is claimed. Input shape/attribute semantics are validated by Graph
inference, not redefined by placement. Noncontiguous CUDA bindings need explicit
MATERIALIZE; placement does not insert materialization or change kernels.

Default unsupported/missing CUDA requests become explicit CPU placements if CPU
supports the operation. `PlacementFallback::Error` rejects them instead. A
primitive unsupported by both backends is an error, not a hidden fallback.
There is no cost model, autotuning, multi-GPU, stream overlap, or default-kernel
change. C1 does not allocate, copy, rewrite, or execute a graph.

Local C1 acceptance: CPU Release build and 32/32 CTests, including repeated
placement, unavailable/unsupported CUDA, strict errors, dtype/layout rejection,
and no early Stage 12 support. CUDA tests here use a metadata-only capability
double, never GPU execution evidence. Subsequent mixed execution needs T4.
