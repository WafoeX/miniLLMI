# Stage 10 — Profiling and tuning evidence

**Goal:** explain the selected sufficient improvement, not maximize every profiler metric. **Prerequisite:** a correct S9-C2 path. **Gate:** one v0/selected-kernel comparison contains Observation → Hypothesis → Change → Result and raw artifacts. Extra tuning cards are optional.

## S10-C1 — Profile runner generalization
- **Goal/type:** Extend Stage 0 nsys/ncu scripts to selected registered kernels/configs and record tool/version/command. *Profiling + Integration.*
- **Depends/files:** S9-C1; scripts/results/tests.
- **Scope/not:** profiler data never enters formal benchmark CSV.
- **Verify/baseline/metrics:** profile a known v0 run and validate output/artifact hashes; baseline Stage 0 profile.
- **Accept/commit/risk:** unique dirs, nonzero failure, external-archive location support. `feat(profile): support registered GEMM kernels`. Risk: profiler capability changes.

## S10-C2 — Kernel evidence cards
- **Goal/type:** Generate/author per-kernel cards with resource, throughput, occupancy, stall/load observations. *Profiling + Documentation.*
- **Depends/files:** C1; `docs/profiling/`, analyzer.
- **Scope/not:** required cards cover v0 and the selected custom kernel on one declared representative shape (4096), plus relevant boundary correctness tests outside profiling. Extra variants/shapes are optional. Report actual metric names/units only; no copied T4 peak claims or occupancy target.
- **Verify/baseline/metrics:** nsys timeline + ncu CSV linked to run/hash; baseline prior stable kernel.
- **Accept/commit/risk:** every optimization commit states a falsifiable hypothesis and outcome. `docs(profile): record GEMM tuning evidence`. Risk: profiling perturbation mislabeled benchmark.

## S10-C3 — Configuration selection policy
- **Goal/type:** Choose default only from correctness-passing, repeat-tested configurations; preserve experimental registry entries. *Performance + Integration.*
- **Depends/files:** required S9-C1/C2 (optional variants only if attempted) and C2; dispatch/docs/tests.
- **Scope/not:** no cherry-picking one best sample; selection rule uses medians/repeats.
- **Verify/baseline/metrics:** all four mandatory CUDA sizes and three paired runs per roadmap; 4096 profile explains, but does not replace, the multi-size gate. Record v0 and cuBLAS ratios and candidate/fallback selections.
- **Accept/commit/risk:** roadmap modest CUDA gate passes and policy is reproducible from CSV; stop here, even if cuBLAS is much faster. `perf(cuda): select evidence-backed stable GEMM`. Risk: overfitting one T4 condition.
