# Stage 17 — CLI and end-to-end inference

**Status:** Stage 17 is **T4-accepted to its required C1–C4 gate**. A clean
source T4 capture validates the mixed scheduler path, CLI smoke, and raw C3
records. C3 has no speedup gate; its measurements are diagnostic and do not
claim a cache, INT8, or mixed-backend performance benefit. The reproduction
protocol remains in [the Stage 17 Colab guide](stage17_colab.md).

## Implementation source and scope

The locally verified implementation source is
`0c258548177eac8d8ae042d0fba402dbbdedcd65` (`bench(quant): add INT8 tradeoff
suite`). It is on `feat/int8-stage16`, preserves the Stage 0 naive SGEMM
baseline, and adds no alternate tensor, allocator, graph, or CUDA dispatch
path.

| Change | Status | Evidence |
|---|---|---|
| C1 CLI contract | pass | `llm_cli` requires a visible `--backend`, `--quant`, and `--cache` mode; emits deterministic JSON configuration; validates help, bad flags and model/quant format mismatch. `--seed` is recorded but greedy mode does not pretend to sample. |
| C2 greedy prefill/decode | pass | `model::generate_greedy` uses byte tokenization, ordinary decoder graphs, `KVCache`, planned storage, and scheduler rewriting when mixed is selected. It rejects the complete prompt-plus-generation budget before cache mutation. |
| C3 inference runner | pass, diagnostic performance | `bench_inference` emits raw JSONL for three independent ten-sample teacher-forced runs (three warmups), with load/prepare, prefill, first-token, decode, memory and copy accounting. Greedy output is a separate smoke record. |
| C4 regression fixture | pass | The committed Stage-15 V1 tiny model, fixed `Stage17` prompt and CPU greedy fixture produce IDs `[73,18,3,234]`; direct CLI and library paths assert it. |

`--quant float` requires a V1 FP32 file and `--quant int8` a V2 INT8 file. V2
continues to load source INT8/scales plus the persistent FP32
`prepare_dequant` workspace, then uses the same decoder/scheduler/backend
route as float; it is not a native INT8 GEMM claim. Quantized and float greedy
text are intentionally not required to match.

## Local verification

macOS, AppleClang 21.0.0.21000334, Release, CUDA disabled:

```text
cmake -S . -B build-stage17-release -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-stage17-release --parallel 4
ctest --test-dir build-stage17-release --output-on-failure
# 53/53 passed
```

A separate Debug ASan+UBSan build passed all four focused CTest entries:
`cli_tools`, `generation`, `inference_benchmark_tools`, and
`quant_benchmark_tools`. LSP diagnostics are clean for the new/changed C++
files.

The benchmark-tool test only validates real local runner output/schema; its
record correctly marks the worktree dirty while development is in progress.
No local timing values are acceptance data and none are claimed here.

## T4 acceptance evidence

Evidence commit [`3872ba9`](../results/inference/stage17-c3/20261008T132035Z-colab-stage17/)
is the direct child of the tested source and changes only the 20 files in its
unique result directory. `tested_commit.txt` names `0c258548…edcd65`,
`source_status.txt` is empty, and the committed V1/V2 artifact hashes match
the recorded SHA-256 values.

Tesla T4 (CC 7.5), driver 580.82.07 / CUDA 13.0 / nvcc 13.0.88, GNU 13.3.0,
Release: full CTest **60/60**, GPU label **8/8**, and focused CLI/INT8 mixed
suite **5/5** passed. The evidence has 18 valid teacher-forced records across
CPU float/INT8 cache-off/on and mixed float/INT8 cache-on configurations: each
has three runs, three warmups, ten positive raw samples, clean source identity,
and zero execute-time backing allocations. Six independent greedy smoke rows
all completed; the fixed benchmark prompt produced `[6,174,228,25]` in those
records.

Mixed rows retain explicit graph-visible transfer accounting (573 executed
copies / 2,539,696 bytes and zero execute allocations). CPU cache-off/on and
mixed rows include slow/noisy retained samples, so this report makes **no
cache, INT8, or mixed speedup claim**.

## Stage 16-C4 integration boundary

Stage 17-C2 unblocked Stage 16-C4. `bench_quantization` compares a V1 float
file and a V2 INT8 file on identical teacher-forced prompt/continuation IDs,
checks every returned logit against the existing frozen
`1e-2 + 1e-2*abs(FP32)` limit, records MAE/max error and artifact/resident/
prepare-dequant accounting, and enforces the eligible-payload-with-scales
≤35% gate. `bench_inference` supplies its separate raw tok/s and memory/copy
records for one dtype/cache/device change at a time; the verified C4 outcome
is recorded in [the Stage 16 report](stage16_report.md).
