# Stage 16 — INT8 weight-only quantization

**Goal:** implement per-output-channel symmetric INT8 without breaking float path. **Prerequisite:** S15-C2 and stable matmul dispatch. **Gate:** required C1–C4; float/INT8 artifacts satisfy frozen error limits and roadmap eligible-weight compression gate. Throughput/resident-memory gains are not required; C5 is optional.

## S16-C1 — Offline per-channel quantizer
- **Goal/type:** for `W[in,out]`, compute each output column's scale=max(abs(column))/127, round-to-nearest ties-to-even and clamp to [-127,127]; all-zero column has scale=1 and q=0. Reject NaN/Inf and invalid scales. *Functional + Correctness.*
- **Depends/files:** S1/S15; `quant/int8.cpp`, tool/tests.
- **Scope/not:** quantize 2D attention/MLP/LM-head projection weights only; keep embeddings/norm weights FP32. No activation quantization/INT4 or unsupported alternate weight layout.
- **Verify/baseline/metrics:** hand columns, ties, zero columns, tiny/large finite values, dequant MAE/max; reconstructed weight error ≤scale/2 plus documented FP rounding allowance. FP32 baseline and output axis are explicit.
- **Accept/commit/risk:** metadata records axis/scales/version. `feat(quant): add per-channel INT8 quantizer`. Risk: axis convention.

## S16-C2 — Quantized tensor/model metadata
- **Goal/type:** extend model format with INT8 payload + FP scale tensor and loader validation. *Functional + Integration.*
- **Depends/files:** C1/S15; tensor/model loader/tests.
- **Scope/not:** INT8 payload uses the existing Storage/Tensor metadata (add DType::INT8 here) plus explicit scale/axis descriptors; no parallel quantized runtime. Validate payload dtype, finite positive FP32 scales, output dimension and checksums; no silent reinterpretation.
- **Verify/baseline/metrics:** save/load bytes/checksums; compression ratio = float bytes/int8+scales.
- **Accept/commit/risk:** loader rejects incompatible scale shape. `feat(quant): store INT8 weights and scales`. Risk: alignment/ownership.

## S16-C3 — Correct dequantized matmul path
- **Goal/type:** graph/prepare-visible dequantization to approved float workspace then call existing backend matmul. The initial stable path may dequantize once during model prepare and reuse that buffer; label it `prepare_dequant`, not native INT8 GEMM. *Correctness + Integration.*
- **Depends/files:** C2/S6/S8; dispatch/tests.
- **Scope/not:** not fused; workspace is planner/accounting-visible. If reused across executions, it is persistent model workspace, not a dead intermediate. Runtime dequant is optional; include it in timed execution if used. Count simultaneous INT8/scales/FP32 buffers and device replicas; file compression does not imply resident-memory reduction.
- **Verify/baseline/metrics:** FP32 baseline; before benchmark freeze fixture errors: matmul and tiny-model logits must be finite and satisfy abs(error) ≤1e-2 + 1e-2*abs(FP32 reference) at every tested element. Record MAE/max, workspace/resident bytes and dequant/prepare latency. Test both zero and deterministic random weights/inputs; failure requires diagnosis, not silent tolerance widening.
- **Accept/commit/risk:** int8 never bypasses backend/scheduler. `feat(quant): run weight-only INT8 matmul`. Risk: temporary dequant can lose performance.

## S16-C4 — Quant benchmark and quality suite
- **Goal/type:** compare float/INT8 model size, peak memory, errors, fixed-prompt generation and tok/s. *Correctness + Performance + Memory.*
- **Depends/files:** C3 and S17-C2 with INT8 mode; S17 float/INT8 integration does not depend on this benchmark. results/analyzer.
- **Scope/not:** random tiny-model text is a smoke test, not a quality claim. Quantized greedy tokens may differ from float; validate logits at identical teacher-forced token IDs and each mode's own deterministic token fixture, not forced identical text.
- **Verify/baseline/metrics:** fixed model/prompt/seed/release; raw CSV and samples.
- **Accept/commit/risk:** eligible INT8 weight payload including scales is ≤35% of corresponding FP32 bytes and C3 frozen error tests pass. Report total artifact size, peak resident/workspace, prepare/load latency and tok/s regardless of regressions; no INT8 acceleration requirement. Stop without C5 when these gates pass. `bench(quant): record INT8 weight-only tradeoffs`. Risk: tiny model not representative.

## S16-C5 — Optional fused or INT4 investigation
- **Goal/type:** only if C4 identifies dequant workspace bottleneck. *Experimental.*
- **Depends/files:** C4/S10; separate kernels/profiles.
- **Scope/not:** cannot replace stable INT8; INT4 remains optional.
- **Verify/baseline/metrics:** C3 baseline, correctness/error/perf profile.
- **Accept/commit/risk:** retain if slower/failed. `perf(quant): evaluate fused INT8 path`. Risk: scope explosion.
