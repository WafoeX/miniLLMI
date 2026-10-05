# Stage 17 — LLM CLI and end-to-end inference

**Goal:** one reproducible user entry point over tokenizer, loader, model, scheduler, and cache. **Prerequisite:** S14/S15 for float mode; S16-C3 for the required final INT8 mode, never S16-C4. **Gate:** scripted smoke generates deterministic tokens from a documented tiny model.

## S17-C1 — CLI contract and configuration
- **Goal/type:** parse model/prompt/max-tokens/backend/quant/seed options; structured error/status. *Functional.*
- **Depends/files:** S15; `app/llm_cli.cpp`, tests/docs.
- **Scope/not:** no server/UI/streaming API. Default deterministic greedy, batch=1, explicit backend/quant/cache mode and maximum context budget. Sampling/temperature/top-k are optional; a seed flag alone does not require a sampling implementation.
- **Verify/baseline/metrics:** help, invalid flags/files, deterministic config serialization; N/A.
- **Accept/commit/risk:** defaults never hide backend/quant mode. `feat(cli): add reproducible inference CLI`. Risk: configuration drift.

## S17-C2 — Prefill/decode generation loop
- **Goal/type:** tokenize, prefill, greedy next token, KV decode, decode text; optional sampling only after the greedy path passes. *Integration + Correctness.*
- **Depends/files:** C1/S14; app/tests.
- **Scope/not:** fixed batch=1; no continuous batching. Define logits→first generated token timing, BOS/EOS/max-token behavior, and reject prompt+requested tokens exceeding max_seq before partial cache writes. Quant mode consumes C3 directly; keep no-cache option for fair paired inference.
- **Verify/baseline/metrics:** fixed seed expected token IDs/logits, cache reset/reuse; no-cache baseline option.
- **Accept/commit/risk:** all model compute travels graph/scheduler. `feat(cli): add autoregressive generation loop`. Risk: EOS/max-token semantics.

## S17-C3 — End-to-end benchmark runner
- **Goal/type:** scripted fixed-prompt measurements separate prefill/decode and report provenance. *Performance + Integration.*
- **Depends/files:** C2; `bench/bench_inference.cpp`, scripts/analyzer/results.
- **Scope/not:** not a general quality benchmark. Separate model load/prepare, prefill, first-token latency and decode; include required shape reprepare, copies and synchronization in measured model steps, exclude terminal I/O. Teacher-forced fixed continuation compares cache/quant fairly; greedy generated-token run is separate smoke evidence.
- **Verify/baseline/metrics:** prompt lengths/contexts, tok/s, ms/token, peak memory, copy stats; compare one change at a time: cache off/on with same dtype/placement, float/INT8 with same cache/placement, and CPU/mixed T4 as a separately labelled device comparison. Never attribute a bundled dtype+cache+device gain to one component.
- **Accept/commit/risk:** profiler data remains separate. `bench(cli): add reproducible inference runner`. Risk: I/O timing contamination.

## S17-C4 — End-to-end regression fixtures
- **Goal/type:** small model/vocab/prompt fixture and CI CPU smoke. *Correctness + Integration.*
- **Depends/files:** C2; tests/fixtures.
- **Scope/not:** no large model stored in Git.
- **Verify/baseline/metrics:** expected tokens and determinism, CPU-only build; N/A.
- **Accept/commit/risk:** fixture provenance/license documented. `test(cli): add tiny model generation regression`. Risk: golden output changes must be justified.
