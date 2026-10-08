# Stage 17 — CLI and end-to-end inference

**Status:** implementation and CPU functional verification are complete for C1–C4.
T4 mixed-backend conformance and the C3 server capture remain pending; therefore
this is **not yet a full Stage 17 acceptance** and contains no GPU or throughput
claim. The reproducible T4 procedure and result-push protocol are in
[the Stage 17 Colab guide](stage17_colab.md).

## Implementation source and scope

The locally verified implementation source is
`6005f94043c9dd4e19cb01f31ad235610680e3c1` (`test(cli): add tiny model
generation regression`). It is on `feat/int8-stage16`, preserves the Stage 0 naive SGEMM
baseline, and adds no alternate tensor, allocator, graph, or CUDA dispatch
path.

| Change | Status | Evidence |
|---|---|---|
| C1 CLI contract | local pass | `llm_cli` requires a visible `--backend`, `--quant`, and `--cache` mode; emits deterministic JSON configuration; validates help, bad flags and model/quant format mismatch. `--seed` is recorded but greedy mode does not pretend to sample. |
| C2 greedy prefill/decode | local pass | `model::generate_greedy` uses byte tokenization, ordinary decoder graphs, `KVCache`, planned storage, and scheduler rewriting when mixed is selected. It rejects the complete prompt-plus-generation budget before cache mutation. |
| C3 inference runner | local pass | `bench_inference` emits raw JSONL for three independent ten-sample teacher-forced runs (three warmups), with load/prepare, prefill, first-token, decode, memory and copy accounting. Greedy output is a separate smoke record. |
| C4 regression fixture | local pass | The committed Stage-15 V1 tiny model, fixed `Stage17` prompt and CPU greedy fixture produce IDs `[73,18,3,234]`; direct CLI and library paths assert it. |

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

## Stage 16-C4 integration boundary

Stage 17-C2 unblocks Stage 16-C4. `bench_quantization` compares a V1 float
file and a V2 INT8 file on identical teacher-forced prompt/continuation IDs,
checks every returned logit against the existing frozen
`1e-2 + 1e-2*abs(FP32)` limit, records MAE/max error and artifact/resident/
prepare-dequant accounting, and enforces the eligible-payload-with-scales
≤35% gate. `bench_inference` supplies its separate raw tok/s and memory/copy
records for one dtype/cache/device change at a time. T4 evidence is still
required before declaring C4 or either full stage accepted.
