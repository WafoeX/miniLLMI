# Stage 9 CUDA SGEMM paired analysis

| Repeat | Kernel order | Geometric-mean v0/v1 speedup |
|---:|---|---:|
| 0 | v0,v1,cublas | 1.627423x |
| 1 | v1,v0,cublas | 1.567992x |
| 2 | v0,v1,cublas | 1.483893x |

Median paired geometric-mean speedup: 1.567992x; gate passed: True.
