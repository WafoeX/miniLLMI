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

## Run `20261005T065240Z-4331-1`

- Timestamp：2026-10-05T06:52:43Z
- Commit：`f7aaec42368ab348050ea2184e937b28b860298c`；source SHA256：`cc1414e52e4d27fb04e3ffd597dd3313bb6d8956f1ddcbedf9762e801a20dc0c`
- GPU：Tesla T4；UUID：`1f4259c1a5da4d6c1714a04a189fcdf1`；CC：7.5
- CUDA runtime/driver API：13000/13000；cuBLAS：130101
- Compiler：GNU 13.3.0；nvcc：NVIDIA 13.0.88；arch：native
- Release；warmup=10；iterations=50；seed=42
- 完整环境、configure/build 日志及 CMakeCache：见对应 raw run。

### M=31, N=33, K=17

Input FNV-1a：`97d676edc3f2c56f`；atol=0.001，rtol=0.001。

| Kernel | median ms | mean ms | std ms | min ms | max ms | GFLOPS | vs naive | cuBLAS % | max abs error | mean abs error | max relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sgemm_v0_naive | 0.007712 | 0.00752704 | 0.0009541 | 0.006144 | 0.012288 | 4.51011 | 1.000x | 212.448% | 1.19209e-07 | 1.70509e-08 | 4.54889e-05 |
| cublas | 0.016384 | 0.0167616 | 0.00216124 | 0.014336 | 0.03088 | 2.12292 | 0.471x | 100.000% | 1.19209e-07 | 1.70509e-08 | 4.54889e-05 |

## 边界

不推断优化版本趋势、模型速度或量化效果；Stage 0 没有这些实现。
Kernel 固定先 naive 后 cuBLAS；GPU 频率/热状态/其他进程会影响结果，请保存 nvidia-smi 前后快照并重复完整 run。
Profiling 时间不用于此表；保留失败/退化实验，不自动删数据。
