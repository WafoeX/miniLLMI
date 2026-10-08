# Stage 16 acceptance — INT8 weight-only

**Status:** Stage 16 is **T4-accepted to its required C1–C4 gate**. C4 has
clean-source T4 artifact, error, memory and diagnostic throughput evidence;
INT8 throughput is not a required benefit and no acceleration claim is made.
C5 is `skipped_optional`.

## Source and scope

C1–C3 were originally accepted at `29b2d58` (`test(quant): cover mixed CUDA
prepare dequant`) and its C1–C3 parents `7308583`, `423f3c1`, and `7d5b4d3`.
C4 is accepted at descendant source `0c258548177eac8d8ae042d0fba402dbbdedcd65`
(`bench(quant): add INT8 tradeoff suite`). Neither source changes
`sgemm_v0_naive`, model layering, or float-default behavior.

| Change | Status | Evidence |
|---|---|---|
| C1 per-channel quantizer | pass | `quantization`: W[in,out], output axis 1, ties-to-even, clamp, all-zero scales=1, nonfinite/invalid metadata rejection, element error bound |
| C2 INT8 container metadata | pass | V2 records explicit INT8/scales/checksum descriptors; strict loader inspection rejects bad V2 checksum and invalid shape/layout metadata |
| C3 prepare dequant path | pass | explicit persistent FP32 `prepare_dequant` `Tensor` feeds unchanged MATMUL backend; CPU direct MATMUL and tiny-model logits meet frozen error limits; clean T4 mixed test passes |
| C4 benchmark/quality suite | pass | Clean T4 `bench_quantization` records V1/V2 file/resident/dequant accounting and frozen teacher-forced errors; six raw C3 configuration files retain separate CPU/mixed cache/dtype timing rows. Evidence: [Stage 17 / C4 capture](../results/inference/stage17-c3/20261008T132035Z-colab-stage17/). |
| C5 fused/INT4 | `skipped_optional` | stable non-fused path retained; C4 has no required throughput gate and does not justify further kernel scope |

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

## T4 acceptance evidence

Clean-source evidence commit [`38af605`](../results/quant/stage16-c3/20261008T104029Z-colab-stage16/)
is the direct child of tested source `29b2d58ec087dddc9cde922c2a948233def9c07e`
and changes only its eight result files. `tested_commit.txt` names that source
exactly and `source_status.txt` is empty.

On Tesla T4 (CC 7.5), driver 580.82.07 / CUDA 13.0 / nvcc 13.0.88, GNU 13.3.0,
a clean Release CUDA build passed the focused Stage 16 suite **4/4**
(`quantization`, `model_file`, `model_loader`, `cuda_quantized_decoder`) and
the GPU-labelled regression suite **8/8**. The mixed C3 test runs the V2 model
through the ordinary scheduler/backend route: all 21 learned projections are
placed on CUDA, explicit copies remain graph-visible, prepared execute has zero
intermediate backing allocations, and all returned logits satisfy the frozen
`1e-2 + 1e-2*abs(FP32 reference)` bound. Raw configure/build/test/device logs
are retained in that result directory.

## C4 T4 evidence

Evidence commit [`3872ba9`](../results/inference/stage17-c3/20261008T132035Z-colab-stage17/)
is a direct child of C4 source `0c258548…edcd65`, has empty source status, and
changes only its 20-result-file directory. On Tesla T4 (CC 7.5), driver
580.82.07 / CUDA 13.0 / nvcc 13.0.88, GNU 13.3.0, Release, full CTest passed
**60/60**, GPU label **8/8**, and focused CLI/quant/mixed suite **5/5**.

For identical teacher-forced IDs, both CPU and mixed C4 records pass the frozen
logit limit. CPU: MAE `0.000181204`, max absolute error `0.000777937`; mixed:
MAE `0.000181205`, max `0.000777923`. Eligible INT8 plus scales are
`104,072 / 393,728 = 26.4325%`, passing the ≤35% gate. V2 is 173,578 bytes
versus V1 462,382 bytes on disk, but resident parameters are **565,128** bytes
versus **461,056** bytes because the 393,728-byte persistent prepare-dequant
workspace coexists with INT8/scales. Peak live intermediate bytes are equal
for matched cache-on runs (37,504 CPU; 89,232 mixed). This is correct
weight-only file compression, **not** a resident-memory-reduction claim.

The raw CPU/mixed inference rows report tok/s and retain all slow samples; no
INT8, cache, or mixed speedup gate exists and none is claimed. The original
C1–C3 replay procedure remains in [the Stage 16 Colab guide](stage16_colab.md);
the combined measurement protocol is [the Stage 17 Colab guide](stage17_colab.md).
