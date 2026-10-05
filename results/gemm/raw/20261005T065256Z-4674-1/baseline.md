# Stage 0 — CUDA GEMM Baseline（自动生成）

数据来源：`/content/miniLLMI/results/gemm/baseline.csv`；逐次 CUDA Event 数据：`/content/miniLLMI/results/gemm/raw/<run_id>/`。
这里只报告已运行的尺寸，不等于所有 Stage 0 验收项已完成。未比较跨 run/跨 commit 的数据。

## 定义和计时方法

- CPU Reference：单线程 ijk、FP64 累加后转 FP32；全矩阵校验，不参与性能比较。
- Naive：每个 CUDA thread 计算一个 C 元素，无 shared memory/float4/register blocking。
- cuBLAS：row-major C=A×B 映射为 column-major Cᵀ=Bᵀ×Aᵀ，alpha=1、beta=0。
- 两者 FP32；cuBLAS 使用 PEDANTIC_MATH（不是不受约束的最高性能上限）。
- 数据传输、CPU reference、分配、校验、文件写入不在 Event 区间内；同一 stream 逐轮同步。
- warmup≥10、iterations≥30；主指标 median，std 为总体标准差（ddof=0）。
- GFLOPS=2MNK/(median_ms×10⁶)；cuBLAS Ratio 是百分比，不是 0～1 的小数。
- 混合逐元素判据 |actual-ref|≤atol+rtol×|ref|；relative_error 分母 max(|ref|,10⁻¹²)。

## Run `20261005T065256Z-4674-1`

- Timestamp：2026-10-05T06:52:59Z
- Commit：`f7aaec42368ab348050ea2184e937b28b860298c`；source SHA256：`cc1414e52e4d27fb04e3ffd597dd3313bb6d8956f1ddcbedf9762e801a20dc0c`
- GPU：Tesla T4；UUID：`1f4259c1a5da4d6c1714a04a189fcdf1`；CC：7.5
- CUDA runtime/driver API：13000/13000；cuBLAS：130101
- Compiler：GNU 13.3.0；nvcc：NVIDIA 13.0.88；arch：native
- Release；warmup=10；iterations=50；seed=42
- 完整环境、configure/build 日志及 CMakeCache：见对应 raw run。

### M=512, N=512, K=512

Input FNV-1a：`3760d84e5a172613`；atol=0.001，rtol=0.001。

| Kernel | median ms | mean ms | std ms | min ms | max ms | GFLOPS | vs naive | cuBLAS % | max abs error | mean abs error | max relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sgemm_v0_naive | 0.69472 | 0.694976 | 0.00141536 | 0.692256 | 0.699456 | 386.394 | 1.000x | 13.287% | 8.10623e-06 | 5.36468e-07 | 0.0681954 |
| cublas | 0.092304 | 0.0945722 | 0.00579017 | 0.089024 | 0.12416 | 2908.17 | 7.526x | 100.000% | 8.10623e-06 | 5.36468e-07 | 0.0681954 |

### M=1024, N=1024, K=1024

Input FNV-1a：`67c6933e2adeaecc`；atol=0.001，rtol=0.001。

| Kernel | median ms | mean ms | std ms | min ms | max ms | GFLOPS | vs naive | cuBLAS % | max abs error | mean abs error | max relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sgemm_v0_naive | 5.08685 | 5.48337 | 1.16537 | 5.05085 | 8.98982 | 422.164 | 1.000x | 9.503% | 2.14577e-05 | 1.06977e-06 | 0.168646 |
| cublas | 0.483408 | 0.483608 | 0.00699007 | 0.473376 | 0.511936 | 4442.38 | 10.523x | 100.000% | 5.24521e-06 | 5.35028e-07 | 0.0550131 |

### M=2048, N=2048, K=2048

Input FNV-1a：`451256a4b3d1d959`；atol=0.001，rtol=0.001。

| Kernel | median ms | mean ms | std ms | min ms | max ms | GFLOPS | vs naive | cuBLAS % | max abs error | mean abs error | max relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sgemm_v0_naive | 37.2888 | 37.3561 | 0.266819 | 36.8863 | 37.9698 | 460.725 | 1.000x | 10.777% | 4.00543e-05 | 2.14267e-06 | 1.57826 |
| cublas | 4.01851 | 4.05347 | 0.432989 | 2.82637 | 4.82765 | 4275.18 | 9.279x | 100.000% | 2.67029e-05 | 1.60441e-06 | 4.59695 |

### M=4096, N=4096, K=4096

Input FNV-1a：`a92f5e0931254f02`；atol=0.001，rtol=0.001。

| Kernel | median ms | mean ms | std ms | min ms | max ms | GFLOPS | vs naive | cuBLAS % | max abs error | mean abs error | max relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sgemm_v0_naive | 324.25 | 323.819 | 3.31694 | 315.905 | 329.945 | 423.867 | 1.000x | 10.688% | 9.34601e-05 | 4.29002e-06 | 6.69842 |
| cublas | 34.6574 | 34.7339 | 0.274238 | 34.079 | 35.0694 | 3965.64 | 9.356x | 100.000% | 9.34601e-05 | 4.29002e-06 | 6.69842 |

## 边界

不推断优化版本趋势、模型速度或量化效果；Stage 0 没有这些实现。
Kernel 固定先 naive 后 cuBLAS；GPU 频率/热状态/其他进程会影响结果，请保存 nvidia-smi 前后快照并重复完整 run。
Profiling 时间不用于此表；保留失败/退化实验，不自动删数据。
