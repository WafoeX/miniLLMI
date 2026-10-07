# Stage 9 CUDA SGEMM paired analysis

| Repeat | Kernel order | Geometric-mean v0/v1 speedup |
|---:|---|---:|
| 0 | v0,v1,cublas | 1.678212x |
| 1 | v1,v0,cublas | 1.543855x |
| 2 | v0,v1,cublas | 1.479965x |

Median paired geometric-mean speedup: 1.543855x; gate passed: True.
