# Stage 15 acceptance — tokenizer and model file

**Status:** CPU-functional acceptance complete for required C1–C3. Stage 15
has no performance gate and no CUDA-specific functionality; therefore its gate
is the local, deterministic saved-model/tokenizer-to-logits round-trip. C4 BPE
is `skipped_optional` by the task book.

## Scope and source

Tested implementation source: `dc8cc8a` (`feat(tokenizer): add deterministic
vocabulary tokenizer`) plus its direct loader/container parents `464b80d` and
`6125118`. It preserves the existing decoder/runtime layers: file loading ends
in `ParameterTable`/normal CPU `Tensor` storage, and tokenizer output is normal
INT32 decoder input.

| Change | Status | Evidence |
|---|---|---|
| C1 versioned container | pass | strict `MLLMRTF\0` v1 parser rejects truncated, overlap, duplicate-name, unsupported-version, and over-budget inputs (`model_file`) |
| C2 writer/loader/binding | pass | deterministic 21-tensor artifact, byte checksum equality to frozen weights, and loaded-model planned logits equality (`model_loader`) |
| C3 byte tokenizer | pass | ASCII, valid multibyte UTF-8, invalid bytes, BOS/EOS policy, unknown IDs, and vocabulary/version mismatch (`tokenizer`) |
| C4 BPE | `skipped_optional` | explicitly optional; byte tokenizer is the retained required path |

The committed v1 artifact has 461,056 FP32 payload bytes and SHA-256
`f91f89491735d324193295bb489f16ff6301186d957e04029c7a1402be729ee8`.

## Verification

On macOS / AppleClang 21.0.0.21000334, CMake Release with CUDA disabled passed
**48/48** CTests. The focused ASan+UBSan Debug build passed **3/3** Stage-15
labelled tests (`model_file`, `model_loader`, `tokenizer`). The loader test
constructs a decoder graph from a loaded artifact, drops the `LoadedModel`, then
runs a planned execution with zero intermediate backing allocations and compares
frozen logits exactly within the existing Stage 13 tolerance.

A clean Colab server replay tested source `e9991465258661d7d03888a42351fecb7409ce26`.
Its direct-child evidence commit `dcdcc8e` preserves
`results/loader/stage15-c3/20261008T095009Z-colab-stage15/stage15_ctest.log`:
focused Release CTest **3/3** passes (`model_file`, `model_loader`,
`tokenizer`) in `/content/miniLLMI/build-stage15-release`. The recorded
`tested_commit.txt` exactly matches that evidence commit's parent; the result
commit changes only this evidence directory.

No GPU claim, timing metric, throughput result, or language-quality claim is
made. The reproducible clone/build/test/push procedure is in
[the Stage 15 server guide](stage15_colab.md).
