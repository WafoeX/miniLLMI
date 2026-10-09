# Stage 18 — RC final evaluation (generated)

- Release tag: `v1.0-rc4`
- Tested commit: `ec116ce078eb7601e7cccc27a03f0c005eac9c7e`
- Source digest: `b819584a2efee94a6428d05d36b428343ef25779a8cb044a67c02942c96d3dd4`

## Required benefit gates

| Gate | Generated result |
|---|---:|
| Planner execute allocation/capacity | passed |
| CPU GEMM paired geomean | 5.334258x |
| CUDA GEMM paired geomean | 1.538073x |
| KV cache context-512 paired median | 13.470029x |
| INT8 eligible payload including scales | 26.432500% |

## Limits

No scheduler, mixed-path, cache-outside-the-KV-gate, or INT8-throughput speedup is claimed. INT8 resident memory includes a persistent prepare-dequant workspace and is not claimed smaller.

## Optional skips

- `S5-C4 inplace`
- `S7-C2 static per-call threads`
- `S7-C4 work stealing`
- `S9-C3..C6 extra variants`
- `S11-C3 segmentation`
- `S15-C4 BPE`
- `S16-C5 fused/INT4`
