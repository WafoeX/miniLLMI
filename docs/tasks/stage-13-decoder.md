# Stage 13 — Tiny decoder-only transformer

**Goal:** assemble existing graph operators into a deterministic tiny decoder. **Prerequisite:** S12-C5 and S5 planner. **Gate:** model output matches frozen fixture and uses graph/runtime layers.

> **Implementation status.** C1–C3 and the C4 CPU path are locally accepted on
> `feat/decoder-stage13`. The required clean T4 mixed-path C4 capture is pending,
> so Stage 13 is not yet complete. See [`../stage13_report.md`](../stage13_report.md)
> and [`../stage13_colab.md`](../stage13_colab.md).

## S13-C1 — Model config and parameter binding
- **Goal/type:** Define tiny decoder config and named parameter table mapping tensors to graph inputs. *Functional.*
- **Depends/files:** S1/S12; `model/config.hpp`, decoder files/tests.
- **Scope/not:** no file format/tokenizer yet; test constructs tensors from the frozen roadmap tiny-model fixture (2 layers/64 hidden/4 heads/128 FFN/258 vocab). No external pretrained model or text-quality promise; unsupported GQA/batch>1/bias forms fail explicitly.
- **Verify/baseline/metrics:** invalid heads/dimensions/names; N/A.
- **Accept/commit/risk:** config validates rotary/head dimensions. `feat(model): add tiny decoder config and parameter binding`. Risk: hardcoding one topology.

## S13-C2 — Transformer block graph builder
- **Goal/type:** Build RMSNorm→QKV→RoPE→attention→residual→MLP graph from descriptors. *Integration + Correctness.*
- **Depends/files:** C1/S12; block builder/tests.
- **Scope/not:** no direct backend calls and no cache.
- **Verify/baseline/metrics:** node/order/shape validation and fixture output; baseline decomposed reference.
- **Accept/commit/risk:** every intermediate is graph-visible/plannable. `feat(model): build decoder transformer block graph`. Risk: residual alias lifetime.

## S13-C3 — Multi-layer prefill and logits
- **Goal/type:** Compose blocks, final norm and LM head for fixed short sequences. *Integration + Correctness.*
- **Depends/files:** C2; decoder/tests/bench hook.
- **Scope/not:** no autoregressive loop/KV cache/model loader.
- **Verify/baseline/metrics:** deterministic logits fixture, planned-vs-dynamic outputs, short and boundary sequence lengths; allocation metrics. A new active shape uses explicit prepare/replan. Benchmark prepare separately and also include it in end-to-end timing when shape changes require it; never hide no-cache rebuild cost or pretend decode never replans.
- **Accept/commit/risk:** graph prepare/execute reuse works across same shapes. `feat(model): add tiny decoder prefill graph`. Risk: peak memory.

## S13-C4 — Tiny-model integration evidence
- **Goal/type:** CPU and T4 mixed-path run recording (CUDA projections + explicit copies; CPU primitives). *Integration + Memory.*
- **Depends/files:** C3; tests/results/docs.
- **Scope/not:** not benchmark claim versus external runtimes.
- **Verify/baseline/metrics:** logits tolerance, planned allocation count/peak bytes, graph latency.
- **Accept/commit/risk:** provenance links model fixture/version. `test(model): record tiny decoder integration`. Risk: fixture drift.
