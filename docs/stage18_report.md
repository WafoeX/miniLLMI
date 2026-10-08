# Stage 18 — release-candidate final evaluation

**Status: C1 frozen; C2–C4 await the required clean Tesla T4 capture.** This
page is the release contract, not a performance-result substitution. The
annotated source tag `v1.0-rc3` is immutable; resolve its exact source identity
with `git rev-parse v1.0-rc3^{commit}` and its tree digest with
`python3 tools/provenance.py --root .` in a clean checkout.

## C1 — frozen contract

The `v1.0-rc3` tagged source contains only the stable runtime path and the release
orchestration/validator. It retains the Stage 0 naive SGEMM, FP64 CPU reference
math, dynamic allocation default, CUDA V0 default, and ordinary graph →
scheduler → backend execution. The new release runner does not create an
alternate tensor, allocator, scheduler, model, or CUDA dispatch path.

`v1.0-rc3` has these immutable controls:

| Area | Required control |
|---|---|
| Source | clean detached checkout at the annotated RC tag; one commit and provenance digest across every row |
| Build | fresh Release CMake test and production trees; no fast-math; T4 SM75 |
| Tests | full CTest plus GPU label and CLI/decoder/KV/INT8 focused matrix before performance commands |
| CPU GEMM | Stage 7 paired V0/IKJ, 128/256/512/1024, 3 runs × 10 samples |
| CUDA GEMM | Stage 9 V0/V1/cuBLAS, 512/1024/2048/4096, 3 paired runs, 10 warmups × 50 Event samples |
| Profile | Stage 10 one V0/V1 `nsys` + `ncu` comparison, kept separate from timing |
| Planner | Stage 5 dynamic/no-reuse/reuse raw memory gate, 3 paired runs |
| Scheduler | Stage 11 fixed 64³ manual/automatic C4 protocol; correctness/counts, no speedup gate |
| KV | context 128/256/512 CPU pairs and mixed context-512 diagnostic; 3 warmups × 10 samples |
| Quant/inference | canonical V1/V2 artifacts, frozen teacher-forced error tests, all six Stage 17 configurations |

`tools/verify_stage18_evidence.py` calls the existing stage-specific raw-data
analyzers and rejects a changed source identity, missing samples, failed gates,
incomplete profile coverage, incorrect error/compression accounting, and missing
inference configurations. It generates the final table/report from those raw
artifacts; performance numbers must never be typed into a report by hand.
`tests/test_stage18_evidence.py` gives the aggregation schema negative coverage.

## Pending C2–C4 evidence

GPU-backed functional validation and final performance claims require the T4;
local CPU results are not a substitute. Run the exact procedure in
[the Stage 18 Colab guide](stage18_colab.md). The resulting branch must be a
direct child of `v1.0-rc3` and change only its unique
`results/release/v1.0-rc3/<run-id>/` directory. Preserve a failing capture as
well; any failed/inconclusive required gate blocks release completion.

## Declared optional skips

`S5-C4` inplace, `S7-C2` static per-call threads, `S7-C4` work stealing,
`S9-C3..C6` extra CUDA variants, `S11-C3` segmentation, `S15-C4` BPE, and
`S16-C5` fused/INT4 remain `skipped_optional`. The final report may not turn
these into required work or omit them.
