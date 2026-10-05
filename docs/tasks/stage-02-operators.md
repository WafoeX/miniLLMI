# Stage 2 — Operator definitions and shape semantics

**Goal:** define one declarative operator layer so graph/model code expresses semantics and backends provide implementations. **Prerequisite:** S1-C6. **Stage gate:** CPU-only operator validation and reference execution tests pass; no graph scheduler or CUDA dispatch.

## S2-C1 — Operator schema and attributes
- **Goal / type:** Define `OpCode`, immutable `OpDesc`, input/output tensor IDs or handles, typed attributes, and structured `Status`. *Functional + Correctness.*
- **Depends / files:** S1; proposed `include/runtime/operator.hpp`, `src/operator.cpp`, `tests/test_operator.cpp`.
- **Scope / not scope:** Descriptor has no `execute()` and no backend pointer; backend hint is advisory only. No graph topology yet.
- **Verify / baseline / metrics:** Attribute type/range and wrong-arity cases; baseline N/A.
- **Accept / commit / risk:** All later op types serialize/print deterministically and reject malformed attributes. `feat(ops): add backend-neutral operator descriptors`. Risk: over-general variant attributes; keep a narrow typed set.

## S2-C2 — Central shape and dtype inference
- **Goal / type:** Add pure validation/inference for ADD, MUL, MATMUL, COPY and contiguous requirements. *Correctness.*
- **Depends / files:** C1; `shape_inference.hpp/.cpp`, tests.
- **Scope / not scope:** FP32 exact same-shape elementwise first; no implicit broadcasting. MATMUL is 2D `A[M,K] * B[K,N]`; COPY has explicit source/destination range/layout and overlap policy. Graph-visible view/narrow/transpose aliases and explicit materialization support head splitting and cache ranges; these are metadata/copy operations, not hidden allocations. Reject INT32 arithmetic while permitting INT32 IDs for Embedding.
- **Verify / baseline / metrics:** Valid/non-square/zero/overflow/mismatch cases; compare inferred output shape with hand references. Baseline N/A.
- **Accept / commit / risk:** Operators cannot be formed with unchecked shape/dtype contracts. `feat(ops): add shape and dtype inference`. Risk: adding broadcasting later; version the semantics rather than silently changing them.

## S2-C3 — CPU reference operator kernels
- **Goal / type:** Implement deterministic CPU reference ADD/MUL/MATMUL/COPY over Stage 1 tensors. *Functional + Correctness.*
- **Depends / files:** C2; proposed `ops/{elementwise,matmul,copy}.cpp`, tests.
- **Scope / not scope:** Correctness oracle only; single-thread scalar loops, no backend interface or optimization. Kernels write to caller-provided validated output storage; allocating convenience wrappers belong to the executor/reference harness, enabling later planned execution without duplicate kernels.
- **Verify / baseline / metrics:** Hand vectors, non-square matrices, strided inputs requiring explicit contiguous policy, NaN/alias rules. Baseline is direct scalar reference; no performance claim.
- **Accept / commit / risk:** Values and output metadata agree with independent test reference; output storage ownership is correct. `feat(ops): add CPU reference core operators`. Risk: accidentally treating reference code as optimized CPU path.

## S2-C4 — Transformer-op descriptors and validators
- **Goal / type:** Add schemas/inference only for RMSNorm, Softmax, RoPE, Embedding, SwiGLU, Attention. *Functional + Correctness.*
- **Depends / files:** C1–C2; operator/shape files, tests.
- **Scope / not scope:** Freeze the roadmap tiny-model contract and FP32 projection layout (`W[in,out]` for matmul, quant output-channel axis=1); Embedding consumes INT32 IDs. Define head slice/view and contiguous-copy lowering to ordinary 2D MATMUL (no required batched matmul), RoPE positions, Softmax absolute query/key offsets, and checked writes into persistent cache destinations. Attention is a graph composition, not a mandatory opaque backend kernel. Do not execute kernels or build the model yet.
- **Verify / baseline / metrics:** Rank/head-dimension/epsilon/causal-mask validation cases. Baseline N/A.
- **Accept / commit / risk:** Every future model op has a documented shape contract before backend code exists. `feat(ops): define transformer operator contracts`. Risk: attention layouts; freeze one tiny-decoder layout and translate at loader boundary later.

## S2-C5 — Operator conformance fixtures
- **Goal / type:** Establish compact binary/text reference fixtures generated offline, plus test harness metadata. *Correctness + Integration.*
- **Depends / files:** C3–C4; `tests/fixtures/`, fixture reader, `docs/operators.md`.
- **Scope / not scope:** Runtime never imports Python/PyTorch; fixture-generation provenance is documented. No performance fixtures.
- **Verify / baseline / metrics:** Fixture checksum, dtype/shape metadata, tolerances, CPU reference results. Baseline N/A.
- **Accept / commit / risk:** Later CPU/CUDA implementations consume identical fixtures. `test(ops): add versioned operator conformance fixtures`. Risk: fixture generator bugs; store source script/version and small hand-checked vectors.
