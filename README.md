# mini-llm-runtime

A C++17 tiny-decoder inference runtime with a graph/planner/scheduler/backend
architecture. **Stages 0–18 are accepted; Stage 19 documentation is being
finalized.** The authoritative status and limits are the
[roadmap](docs/roadmap.md) and [Stage 18 release evidence](docs/stage18_report.md),
not historical text in individual reports.

## What is implemented

- One metadata-only `Tensor` / shared-`Storage` representation with views,
  explicit CPU↔CUDA copies, typed operators and a validated frozen DAG.
- Dynamic last-use execution plus opt-in prepared memory planning; planned
  warmed execution has no intermediate backing allocations.
- CPU scalar/reference and explicit optimized paths, CUDA storage/copies, the
  preserved Stage 0 V0 GEMM and explicit Stage 9 V1 SGEMM.
- Deterministic placement/copy insertion, graph-composed transformer primitives,
  a tiny decoder with persistent K/V cache, strict model files, byte tokenizer,
  weight-only INT8 loading, and a greedy CLI.
- Reproducible result tooling that preserves raw samples, environment,
  provenance, hashes, failed runs and generated reports.

The [architecture and operation guide](docs/architecture.md) maps each layer to
its source and contract. The [interview evidence index](docs/interview/README.md)
and [resume claim matrix](docs/resume_evidence.md) provide concise, auditable
discussion entry points.

## Accepted release evidence

The immutable source tag is `v1.0-rc4` (`ec116ce078eb7601e7cccc27a03f0c005eac9c7e`).
Its clean T4 capture passed **61/61** full CTests, **8/8** GPU-labelled CTests,
and **7/7** focused release checks. The generated report's required benefit
gates are planner memory, CPU GEMM, custom CUDA GEMM, context-512 KV decode,
and eligible INT8 payload including scales; the exact values, source digest,
result commit and external profiler archive checksum are in
[Stage 18](docs/stage18_report.md).

These facts do **not** imply an overall inference speedup, scheduler speedup,
INT8 throughput gain, INT8 resident-memory reduction, or language-quality
claim. Stage-specific limits and retained slower/failed experiments remain part
of the record.

## Build and test locally (CPU-only)

Requirements: CMake ≥3.24, C++17 compiler, Python ≥3.8, Git and Bash.

```bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-local --parallel 4
ctest --test-dir build-local --output-on-failure --no-tests=error
python3 tools/check_documentation.py
```

This validates the CPU build and documentation links only. It neither compiles
CUDA nor substitutes for GPU validation.

## T4 reproducibility

Use a clean clone and a visible Tesla T4 for CUDA integration. The frozen RC
measurement matrix is documented in [Stage 18 Colab instructions](docs/stage18_colab.md).

Every accepted performance or GPU claim names a tested source and separate
result commit. Do not hand-edit metrics, reuse stale binaries, or combine
measurements across source identities. Profiler timing is not benchmark timing.

## Boundaries

The frozen tiny model is batch-1, bias-free FP32 with deterministic byte tokens.
No external/pretrained-model compatibility, BPE, sampling, training, batching
service, distributed execution, Flash/paged attention, or fused INT8/INT4
kernel is implemented. Optional work is explicitly marked `skipped_optional`
in the [roadmap](docs/roadmap.md) and [Stage 18 report](docs/stage18_report.md).
