# Stage 14 — KV cache acceptance report

**Stage 14 C1–C4 is accepted.** The tested source is
[`f37d7dc15794cfec3ce13fa551ea33e06dfb800b`](https://github.com/WafoeX/miniLLMI/commit/f37d7dc15794cfec3ce13fa551ea33e06dfb800b).
Its clean T4 result commit is the direct child
[`edf6e4a66d390d816bf2f0564ca373128f775627`](https://github.com/WafoeX/miniLLMI/commit/edf6e4a66d390d816bf2f0564ca373128f775627),
which changes only
[`results/kv_cache/stage14-c4/20261008T084022-17967/`](../results/kv_cache/stage14-c4/20261008T084022-17967/).

## Implementation

- C1: `KVCache` owns persistent CPU FP32 K and V storage with fixed
  `[layer,batch,seq,head,head_dim]` capacity and checked layer/position/range
  views. Reset invalidates all active ranges without reallocating; overflow is
  rejected before metadata changes.
- C2/C3: prefill and one-token decode use graph-visible persistent-state COPY
  writes. The cache validates/commits its active range only after every layer
  succeeds; a failure invalidates it. Decode applies RoPE at its append
  position and attends the written active K/V range only.
- C4: the benchmark uses contexts 128/256/512, the same fixed 32
  teacher-forced IDs, 3 warmups and 10 raw samples per variant/run. It includes
  graph construction, required prepare/replan, dispatch, cache writes and
  transfers in decode time, while reporting cache prefill separately.

## T4 evidence

Tesla T4 (CC 7.5), driver 580.82.07 and CUDA 13.0 built fresh Release SM75
`BUILD_TESTING=ON` and independent `BUILD_TESTING=OFF` configurations. The
clean source digest is
`e67689d3efdf62131042bd9294f850fa1281cd2327a3de9a597c557c89aa7257`.

All tests pass: **51/51** full, **7/7** GPU, and **4/4** KV-labelled. The
mixed cached-decode test confirms scheduler/copy correctness. The production
CPU paired run retains all 18 raw records; every context-512 run-level ratio
exceeds 1 and their median is **10.744146×**, exceeding the required 1.05×:

| Context | paired median-latency ratios | median |
|---:|---|---:|
| 128 | 3.241388, 3.306909, 3.282527 | 3.282527× |
| 256 | 5.909522, 5.987119, 5.937329 | 5.937329× |
| 512 | 10.196216, 10.744146, 14.347848 | **10.744146×** |

The separate T4 mixed diagnostic record at context 512 retains 3 warmups and
10 samples: median **54.439433 ms/token** / **18.369038 tok/s**, 1,114,112
persistent cache bytes, 101 copies and 1,273,608 copy bytes. It is
correctness/timing evidence only—not a mixed-path speedup claim.

No local CPU result is used as T4 acceptance evidence. Stage 15 is the next
proposed dependent change.
