# Stage 9 CUDA SGEMM paired analysis

| Repeat | Kernel order | Geometric-mean v0/v1 speedup |
|---:|---|---:|
| 0 | v0,v1,cublas | 1.538073x |
| 1 | v1,v0,cublas | 1.557618x |
| 2 | v0,v1,cublas | 1.480778x |

Median paired geometric-mean speedup: 1.538073x; gate passed: True.
