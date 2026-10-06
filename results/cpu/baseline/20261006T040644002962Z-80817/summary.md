# Stage 6 CPU-only baseline

Single-thread FP32 ijk v0, full backend call. Unchanged FP64 oracle is untimed. Three independent baseline runs, not a speedup experiment. No GPU comparison.

| Shape | Repeat | Median ms | GFLOPS |
|---|---:|---:|---:|
| 128³ | 0 | 0.979313 | 4.282907 |
| 256³ | 0 | 9.568812 | 3.506646 |
| 512³ | 0 | 86.993313 | 3.085702 |
| 1024³ | 0 | 1049.891437 | 2.045434 |
| 1024³ | 1 | 1057.309958 | 2.031082 |
| 512³ | 1 | 87.895854 | 3.054017 |
| 256³ | 1 | 9.703083 | 3.458121 |
| 128³ | 1 | 0.993521 | 4.221656 |
| 128³ | 2 | 0.982792 | 4.267743 |
| 256³ | 2 | 9.679688 | 3.466479 |
| 512³ | 2 | 88.119416 | 3.046269 |
| 1024³ | 2 | 1056.951229 | 2.031772 |

Graph [16,16] MATMUL → ADD → MUL, same backend, dynamic/reuse paired ratios: 0.719645, 0.777296, 0.729053; median 0.729053. Includes validation, oracle scan, output destruction. No required latency gate.
