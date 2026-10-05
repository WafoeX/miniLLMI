# Stage 12 — Transformer operator implementations

**Goal:** implement primitives through existing tensor/operator/backend paths. **Prerequisite:** S11 plus S2-C5 fixtures. **Gate:** CPU primitives and the declared mixed CPU/CUDA path meet fixture tolerance. CPU supplies all primitives; T4 supplies CUDA projection MATMUL/COPY and explicit scheduler transfers. Additional CUDA primitive coverage is optional; no model-specific tensor path.

## S12-C1 — RMSNorm and SiLU/SwiGLU
- **Goal/type:** CPU implementations, op dispatch, fixture tests; CUDA follows only when dispatch-ready. *Functional + Correctness.*
- **Depends/files:** S2/S6/S8; `ops/rmsnorm.cpp`, `ops/silu.cpp`, tests.
- **Scope/not:** no fused transformer block. RMSNorm scales the last dimension; SwiGLU explicitly computes SiLU(gate)*up with same-shaped tensors. Freeze fixture/tensor/logit tolerances before optimization; no post-hoc widening.
- **Verify/baseline/metrics:** fixture max/mean error; N/A.
- **Accept/commit/risk:** epsilon/layout explicit. `feat(ops): implement RMSNorm and SwiGLU primitives`. Risk: reduction stability.

## S12-C2 — Stable causal softmax
- **Goal/type:** row-wise max-subtracted softmax with causal mask contract. *Correctness.*
- **Depends/files:** C1; softmax/tests.
- **Scope/not:** no FlashAttention/fusion. Causal validity is `key_absolute_position <= query_absolute_position`, so one-token decode at position p sees cached keys 0..p. Define all-masked rows as zero output (and test); reject invalid nonfinite unmasked inputs explicitly.
- **Verify/baseline/metrics:** extreme logits/mask/all-masked rules; fixture error; N/A.
- **Accept/commit/risk:** no NaN from valid inputs. `feat(ops): add stable causal softmax`. Risk: mask semantics.

## S12-C3 — RoPE and embedding
- **Goal/type:** positional rotation and embedding gather with explicit layouts/dtypes. *Correctness.*
- **Depends/files:** S2/S6; rope/embedding/tests.
- **Scope/not:** no tokenizer/model loader. INT32 IDs are bounds checked; Q/K head slices and selected split/interleaved RoPE layout use the S2 contract. Position is explicit, not inferred from query length during decode.
- **Verify/baseline/metrics:** reference vectors, boundary positions/IDs; N/A.
- **Accept/commit/risk:** layout matches S2 contract. `feat(ops): implement RoPE and embedding`. Risk: interleaved versus split rotary convention.

## S12-C4 — Attention reference path
- **Goal/type:** QKᵀ scaling, causal mask, softmax, V matmul using graph operators. *Integration + Correctness.*
- **Depends/files:** C1–C3/S11; attention compose/tests.
- **Scope/not:** no cache policy/fused attention. Batch=1 MHA uses per-head 2D QKᵀ and probability×V graph nodes with explicit slice/transpose/materialization; do not require undocumented batched matmul or hidden broadcasting. Implement checked COPY writes into declared external destinations so S14 can append K/V without model-level memcpy/CUDA calls.
- **Verify/baseline/metrics:** tiny fixture logits/output; baseline decomposed graph, record intermediate bytes later.
- **Accept/commit/risk:** attention never bypasses backend/planner. `feat(ops): compose causal attention from graph operators`. Risk: shape/layout explosion.

## S12-C5 — Operator conformance matrix
- **Goal/type:** run every primitive CPU/CUDA capability path against fixtures. *Correctness + Integration.*
- **Depends/files:** C1–C4; test runner/results.
- **Scope/not:** performance tuning deferred; no requirement to add CUDA RMSNorm/Softmax/RoPE/Embedding/SwiGLU. A T4 integration fixture must run at least projection MATMUL on CUDA and other primitives on CPU through explicit copy insertion.
- **Verify/baseline/metrics:** dtype/device/tolerance coverage; N/A.
- **Accept/commit/risk:** unsupported combinations report capability status. `test(ops): add transformer conformance matrix`. Risk: fixtures must be versioned.
