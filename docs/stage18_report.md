# Stage 18 — release-candidate final evaluation

**Status: accepted to required C1–C4 on 2026-10-09.** The annotated source tag
`v1.0-rc4` is immutable; resolve its exact source identity with
`git rev-parse v1.0-rc4^{commit}` and its tree digest with
`python3 tools/provenance.py --root .` in a clean checkout.

## C1 — frozen contract

The `v1.0-rc4` tagged source contains only the stable runtime path and the release
orchestration/validator. It retains the Stage 0 naive SGEMM, FP64 CPU reference
math, dynamic allocation default, CUDA V0 default, and ordinary graph →
scheduler → backend execution. The new release runner does not create an
alternate tensor, allocator, scheduler, model, or CUDA dispatch path.

`v1.0-rc4` has these immutable controls:

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

## Verified C2–C4 evidence

The clean Tesla T4 capture is retained at
`results/release/v1.0-rc4/20261009T010708Z-colab-stage18/` on branch
`results/stage18-20261009T010708Z-colab-stage18`, evidence commit
`9f9c5d2d7503a77de9c03ca77c1f235b704175fa`. Its sole parent is
`ec116ce078eb7601e7cccc27a03f0c005eac9c7e` (`v1.0-rc4`), and its 1,992 changed
paths are all beneath that one result directory. Every row names the one source
digest `b819584a2efee94a6428d05d36b428343ef25779a8cb044a67c02942c96d3dd4`.

C2 passed the full fresh CTest suite (61/61), GPU label (8/8), and release
focused matrix (7/7), all from the clean T4 build. C3's generated raw-data
report records the required gates: planner `passed`; CPU GEMM paired geomean
`5.334258x`; CUDA GEMM paired geomean `1.538073x`; context-512 KV paired median
`13.470029x`; and mixed INT8 eligible payload including scales `26.432500%`.

C4 was independently regenerated with `tools/verify_stage18_evidence.py`; both
`stage18_report.md` and `stage18_report.json` match byte-for-byte. The result
manifest lists 1,987 evidence artifacts. Six size-excluded Nsight files are
retained in the companion archive
`stage18-v1.0-rc4-20261009T010708Z-colab-stage18-evidence.tar.gz` (SHA-256
`de91f2190964dbc46e0478e40642e050a5a24ea912aaf90a3ef183e2946e7185`); the
archive contains 2,268 entries, including those six files, and all 1,987 listed
artifact hashes verify. The Git result commit plus this checksum-addressed
archive are the complete evidence set; do not treat the Git checkout alone as a
complete profiler-artifact package.

## Declared optional skips

`S5-C4` inplace, `S7-C2` static per-call threads, `S7-C4` work stealing,
`S9-C3..C6` extra CUDA variants, `S11-C3` segmentation, `S15-C4` BPE, and
`S16-C5` fused/INT4 remain `skipped_optional`. The final report may not turn
these into required work or omit them.
