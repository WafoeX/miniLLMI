# Stage 14 — KV cache

**Goal:** add explicit per-layer cache and split prefill/decode. **Prerequisite:** S13-C4. **Gate:** cached logits match full-prefix within frozen tolerance, cache memory is reported, and roadmap context-512 decode improvement gate passes. No paging/fused attention target.

## S14-C1 — KV cache metadata/storage
- **Goal/type:** Define `[layer,batch,head,seq,head_dim]` (or documented alternative), capacity, position, device storage. *Functional + Memory.*
- **Depends/files:** S1/S8/S13; `kv_cache.hpp/.cpp`, tests.
- **Scope/not:** fixed batch=1 capacity max_seq=1088; no paging or cache-buffer reallocation during decode. Active ranges may trigger explicit graph reprepare using the existing planner; disclose that time/allocation separately. Weights/cache stay on documented devices; scheduler copy buffers are counted.
- **Verify/baseline/metrics:** bounds, reset, per-layer isolation, byte count; N/A.
- **Accept/commit/risk:** cache is persistent Storage with checked active views, separate from reusable intermediates. Reset invalidates valid ranges; capacity overflow fails before partial writes. Planner never recycles cache spans. `feat(kv): add KV cache storage and metadata`. Risk: layout mismatch with attention.

## S14-C2 — Prefill write path
- **Goal/type:** write K/V for a prompt and expose valid sequence range. *Correctness + Integration.*
- **Depends/files:** C1/S13; decoder/attention/tests.
- **Scope/not:** no decode optimization. Cache writes are graph-visible ordered COPY/state updates using S12 primitives; all layers commit the same valid length only after successful execution, with reset/invalidation on partial failure.
- **Verify/baseline/metrics:** compare cache values to noncached intermediate fixture; prefill ms/bytes.
- **Accept/commit/risk:** append positions are checked. `feat(kv): populate cache during prefill`. Risk: stale cache reset.

## S14-C3 — Decode attention with cache
- **Goal/type:** one-token Q attends cached K/V, appends new K/V. *Correctness + Performance.*
- **Depends/files:** C2; attention/decoder/tests.
- **Scope/not:** single batch first, no fused kernel.
- **Verify/baseline/metrics:** logits versus full-prefix recomputation; baseline no-cache decode, decode ms/token/tok/s.
- **Accept/commit/risk:** append exactly at position p, RoPE uses p, Q attends keys 0..p including its own new K/V, and no future/uninitialized entries are read. Test multiple successive steps, reset, prompt=capacity and overflow; compare every step's logits, not only final tokens. `feat(kv): add cached decode attention`. Risk: position/RoPE mismatch.

## S14-C4 — KV benchmark sweep
- **Goal/type:** mandatory contexts 128/256/512, optional 1024, same 32 fixed continuation token IDs for cache off/on (teacher forced). *Performance + Memory.*
- **Depends/files:** C3; bench/analyzer/results.
- **Scope/not:** do not mix prefill and decode metrics. Decode wall time includes required prepare/replan, dispatch, attention, KV writes, device transfers and completion, but excludes tokenizer/printing. Both paths use identical model/input continuation/placement/kernels; do not compare CPU no-cache vs CUDA cached. Cache allocation/initial prepare and prefill are reported separately.
- **Verify/baseline/metrics:** prefill latency, decode ms/token, tokens/s, bytes, correctness; raw CSV.
- **Accept/commit/risk:** roadmap context-512 ≥1.05× paired decode gate passes on the predeclared CPU path; T4 mixed-path correctness and timing are also recorded without requiring its speedup. Report all contexts and total persistent/transient memory; a miss remains open, not a successful cache-performance claim. `bench(kv): compare cached and full-prefix decode`. Risk: tiny model may hide gains; report it.
