# KV cache contract — Stage 14

`model::KVCache` owns two persistent CPU FP32 tensors, K and V, with documented
physical shape `[layer,batch,seq,head,head_dim]`, fixed batch `1`, two layers,
capacity `1088`, four heads and head width `16`. Its 4,456,448 bytes include
both K and V stores. It is independent of planner-owned intermediate storage.

Logical positions are stored newest-to-oldest in physical sequence order. A
graph-visible `COPY` writes the newly appended `[1,H,D]` slot, then a `VIEW`
from that write exposes the full active range. This creates an explicit graph
version dependency without an unmodelled concat/copy. A single-query attention
has no causal future entries in that active range; key/value order is the same
permutation for softmax columns and values, so its output is unchanged.

- `reset()` makes all ranges invalid without reallocating storage.
- `commit_append(position,count)` changes the valid length only after every
  layer has executed. It rejects stale positions and capacity overflow before
  mutating metadata.
- `execute_cached_decoder` resets the cache on any graph failure. Scheduled
  callers must use `finalize_cached_decoder` after executing the rewritten
  graph; this applies the same commit/reset rule.
- Prefill writes every K/V position through persistent-state `COPY` nodes.
  Decode uses RoPE position `p`, writes exactly position `p`, and attends the
  written active range `0..p` (physically reversed). Cache, token bindings,
  and CPU-only transformer primitives remain CPU-resident on the mixed path;
  the scheduler owns all CUDA projection transfers.

`bench_kv_cache` declares contexts 128/256/512 and fixed 32 teacher-forced
continuation IDs. It times full-prefix recomputation and cache decode as paired
runs, includes graph build/prepare/dispatch/KV writes/transfers in decode time,
and reports prefill separately. The CPU benefit gate is evaluated only from
three paired T4 runs at context 512; no local timing is an acceptance result.
