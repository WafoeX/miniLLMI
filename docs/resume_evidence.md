# Resume claim matrix

Use only a bullet below, or a stricter qualification of it. Values are tied to
the immutable RC4 source and generated evidence; do not substitute an earlier
stage value or turn a component metric into an overall inference claim.

| Proposed resume bullet | Change IDs / implementation source | Tests and raw evidence | Exact support | Limit that must stay with the claim |
|---|---|---|---|---|
| Built a C++17 graph-based tiny-decoder runtime with CPU/CUDA backends, explicit heterogeneous copies, planned memory and a deterministic greedy CLI. | S1–S17; final release source [`v1.0-rc4`](stage18_report.md) `ec116ce078eb7601e7cccc27a03f0c005eac9c7e`. | RC4 [Stage 18 capture](stage18_report.md): 61/61 full CTests, 8/8 GPU-labelled, 7/7 focused; code map in [architecture](architecture.md). | Functional/runtime integration claim only. | Do not claim pretrained-model compatibility, language quality, batching, distributed execution, or full-CUDA transformer primitives. |
| Reduced execute-time intermediate backing allocations from 11 to 0 and prepared reuse capacity from 3072 to 512 bytes on the predeclared planner chain. | S5-C1/C2/C3/C5; tested source `4473e3002b93374222974043c73190d904a833dc`. | [Stage 5 report](stage5_report.md) and its linked raw planner manifests/CSV. | The defined CPU planner-memory gate passed. | Prepared tiny-graph execution was slower; this is not zero C++ heap allocation, lower RSS, or an overall speedup. |
| Measured a 5.334258× median paired CPU-GEMM gate and a 1.538073× custom CUDA V1-over-V0 paired geometric-mean gate in the final controlled RC4 capture. | S7-C1/C3/C5 and S9-C1/C2; RC4 source above. | [Stage 18 generated report](stage18_report.md), which replays Stage 7/9 raw data; original context is [Stage 7](stage7_report.md) and [Stage 9](stage9_report.md). | Required component microbenchmark gates passed with frozen controls. | Not an end-to-end LLM throughput result; V1 is not claimed to equal cuBLAS; profiler timings are excluded. |
| Implemented persistent K/V cache decode and passed the RC4 context-512 CPU paired median gate at 13.470029×. | S14-C1–C4; integration retained in RC4. | [Stage 14 report](stage14_report.md) and regenerated final [Stage 18 report](stage18_report.md). | Fixed 32-token teacher-forced continuation, three paired runs, CPU gate. | Do not call the mixed diagnostic timing an acceleration result or imply every context/backend receives that ratio. |
| Implemented auditable per-output-channel INT8 weight-only artifacts at 26.432500% of eligible FP32 payload including scales, while preserving frozen error limits. | S16-C1–C4; C4 source `0c258548177eac8d8ae042d0fba402dbbdedcd65`; RC4 integration. | [Stage 16 report](stage16_report.md), [Stage 18 report](stage18_report.md), raw V1/V2 accounting and error files named there. | Correct artifact compression with scales included. | Persistent FP32 prepare-dequant workspace makes resident parameters larger; no native INT8 GEMM or INT8 throughput benefit is claimed. |
| Produced reproducible release evidence: source/digest checks, raw-sample validators, generated report regeneration and an external-profiler archive checksum. | S18-C1–C4; `v1.0-rc4` and result commit `9f9c5d2d7503a77de9c03ca77c1f235b704175fa`. | [Stage 18 report](stage18_report.md), [Colab procedure](stage18_colab.md), `tools/verify_stage18_evidence.py`. | The report regenerates byte-for-byte from the complete evidence set. | Six large Nsight artifacts are in the specified companion archive, not the Git result commit alone. |

## Audit rules

1. Keep the workload, metric and comparison in the same sentence as a number.
2. Link an interviewer to this matrix and then to the report/raw artifact; do
   not quote a number without source identity.
3. State retained regressions when discussing planner, scheduler, mixed path or
   INT8. The project deliberately stopped optimization after the five required
   benefit gates passed.
4. Before using this text externally, run the [Stage 19 reproducibility
   checklist](stage19_colab.md). If its required fresh T4 result is absent,
   describe the historical RC4 evidence accurately and do not claim Stage 19
   final acceptance.
