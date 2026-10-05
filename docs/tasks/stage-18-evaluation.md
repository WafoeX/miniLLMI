# Stage 18 — Release-candidate final evaluation

**Goal:** freeze one code commit and generate a coherent evidence set. **Prerequisite:** required Changes including S16-C4/S17 and all roadmap benefit gates, clean source tree, selected stable paths; optional skips recorded. **Gate:** source tag and separately committed artifacts reproduce the required benefits.

## S18-C1 — Release-candidate freeze
- **Goal/type:** select source commit, record tool/model/fixture versions and all predeclared controls, create annotated `v1.0-rc1` (or next unused RC tag). *Integration.*
- **Depends/files:** S17; release checklist/docs.
- **Scope/not:** tag names are immutable: start `v1.0-rc1`, then rc2/etc after fixes, never force-move an RC tag. No source/performance edits after freeze; evidence-only commits are allowed and reference the source tag/hash/digest. Fixes restart required evaluation on a new RC.
- **Verify/baseline/metrics:** clean provenance and reproducible build; N/A.
- **Accept/commit/risk:** all final CSV rows reference RC source identity. `docs(release): freeze v1.0 evaluation contract`. Risk: mixing result commits with tested code.

## S18-C2 — Functional/correctness matrix
- **Goal/type:** run CPU and T4 tests for tensor, graph, planner, backends, ops, decoder, KV, loader, INT8, CLI. *Correctness + Integration.*
- **Depends/files:** C1; scripts/results.
- **Scope/not:** no benchmark score from test timing. Matrix covers implemented capability paths; optional unsupported dtype/op/backend combinations are expected explicit errors, not promised CUDA coverage. CPU-only build and T4 mixed tiny-decoder/cache/INT8 integration are required.
- **Verify/baseline/metrics:** pass/fail logs with environment/provenance; N/A.
- **Accept/commit/risk:** any failure blocks report. `test(release): record full RC validation`. Risk: server variability; retain logs.

## S18-C3 — Final performance/memory suite
- **Goal/type:** rerun defined CPU/CUDA/allocator/scheduler/KV/quant/inference experiments on RC. *Performance + Memory.*
- **Depends/files:** C2; benchmark scripts/analyzer/results. Correctness passes before formal performance measurements.
- **Scope/not:** exact predeclared controls only; no parameter fishing. Rerun paired baselines/candidates in the same RC build; historical Stage 0 CSV remains unchanged. Required suite includes five roadmap benefit gates, scheduler counts/correctness, inference smoke and one selected CUDA profile comparison; skipped optional variants do not need final reruns.
- **Verify/baseline/metrics:** baseline matrix metrics, raw samples, repeats, profiler where selected.
- **Accept/commit/risk:** required unavailable/failed/inconclusive gates block completion; optional skips remain labelled. Do not claim success merely because a report honestly lists misses. `bench(release): record RC performance suite`. Risk: long server session; checkpoint raw runs.

## S18-C4 — Generated report audit
- **Goal/type:** generate final tables/charts/docs strictly from CSV and validate links/hashes. *Documentation + Integration.*
- **Depends/files:** C2–C3; analyzer/docs/tests.
- **Scope/not:** no manual performance number edits. Artifact commits are not the tested RC source commit; a manifest maps source tag/hash/digest, model/fixture hashes, runner commands, raw paths and artifact commits. One source identity across final candidate runs avoids the impossible requirement that measured results already exist in the tested source commit.
- **Verify/baseline/metrics:** report regeneration diff, missing-data failure tests.
- **Accept/commit/risk:** README tables are traceable to all required paired raw runs and five benefit gates; aggregate calculation, fallback disclosures, hashes/links and optional skip list are checked. `docs(release): generate final evidence report`. Risk: stale scripts.
