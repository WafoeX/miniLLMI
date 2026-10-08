# Stage 16 acceptance — INT8 weight-only

**Status:** C1–C3 are locally accepted pending the requested clean T4 replay.
C4 is **not started** because its declared dependency, Stage 17-C2 (CLI INT8
mode), does not yet exist. Consequently Stage 16 as a whole is not closed and
this report makes no throughput, GPU-performance, or text-quality claim. C5 is
`skipped_optional`.

## Source and scope

Tested code source: `29b2d58` (`test(quant): cover mixed CUDA prepare dequant`)
and its C1–C3 parents `7308583`, `423f3c1`, and `7d5b4d3`. The source does not
change `sgemm_v0_naive`, model layering, or float-default behavior.

| Change | Status | Evidence |
|---|---|---|
| C1 per-channel quantizer | local pass | `quantization`: W[in,out], output axis 1, ties-to-even, clamp, all-zero scales=1, nonfinite/invalid metadata rejection, element error bound |
| C2 INT8 container metadata | local pass | V2 records explicit INT8/scales/checksum descriptors; strict loader inspection rejects bad V2 checksum and invalid shape/layout metadata |
| C3 prepare dequant path | local pass; T4 replay pending | explicit persistent FP32 `prepare_dequant` `Tensor` feeds unchanged MATMUL backend; CPU direct MATMUL and tiny-model logits meet frozen error limits; T4 test target is `cuda_quantized_decoder` |
| C4 benchmark/quality suite | blocked | requires Stage 17-C2; no benchmark or performance metric was run |
| C5 fused/INT4 | `skipped_optional` | stable non-fused path retained; no C4 bottleneck data exists |

## Representation and accounting

Eligible weights are the fourteen two-layer attention/MLP projections plus
`lm_head` (15 tensors total). Embeddings and norm tensors remain FP32.
The canonical tiny model has 393,728 eligible FP32 bytes. A locally generated
V2 artifact has 98,432 INT8 value bytes plus 5,640 FP32 scale bytes:
**104,072 bytes / 26.43%**, satisfying the roadmap's ≤35% eligible-payload
gate. Its complete payload is 171,400 bytes (173,578-byte file including
metadata), versus 461,056 FP32 payload bytes. This is artifact accounting, not
a timing result.

The initial stable implementation deliberately retains source values/scales and
a persistent FP32 dequantized workspace while a `LoadedModel` is live:
98,432 INT8 + 5,640 scale + 393,728 FP32 workspace = **497,800 bytes** for the
eligible set. Together with 67,328 bytes of ineligible FP32 parameters, the
model-resident parameter storage is 565,128 bytes before device replicas.
Thus smaller file payload is **not** claimed as lower resident memory.

## Local verification

AppleClang 21.0.0.21000334, macOS, Release, CUDA disabled:

```text
cmake -S . -B build-stage16-release -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-stage16-release -j4
ctest --test-dir build-stage16-release --output-on-failure
# 49/49 passed
```

The ASan+UBSan Debug focused suite (`quantization`, `model_file`,
`model_loader`) passed 3/3. `model_loader` checks every eligible reconstructed
weight against `scale/2 + FP rounding allowance`, a direct dequantized MATMUL
against `1e-2 + 1e-2*abs(reference)`, and all tiny-model frozen logits against
the same bound. Execute-time intermediate backing allocations remain zero; the
prepare-dequant allocation happens before graph preparation/execution.

The local machine has no CUDA toolkit, so it cannot compile/run the new T4
mixed-path target. The exact clean-server procedure and result-branch protocol
are in [the Stage 16 Colab guide](stage16_colab.md).
