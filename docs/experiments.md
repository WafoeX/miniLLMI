# Stage 0 实验规范与 baseline.csv

## 固定条件

正式默认：FP32、连续 row-major、alpha=1/beta=0，M=N=K 为 512、1024、2048、4096；seed=42、warmup=10、iterations=50，Release、cuBLAS FP32 pedantic。A/B 在同一个 run/shape 内复用，Naive 与 cuBLAS 配对。CPU Reference 全矩阵 ijk，仅作校验。

输入：mt19937 的每次输出取高 24 位，除以 2²⁴ 后减 0.5，范围 [-0.5,0.5)。相同 seed/shape 可复现输入；input_hash 使用按 FP32 bit pattern 的固定小端字节顺序 FNV-1a-64（用于对比输入身份，不是密码学证明）。

Naive 是优化前原始 baseline；禁止为了提高后续 speedup 修改它。此阶段只报告 Naive 和 cuBLAS，不报告不存在的优化版本。

## CSV 字段

路径：`results/gemm/baseline.csv`，成功运行后追加生成，不预填实验行。

| 字段 | 含义/单位 |
|---|---|
| timestamp | 本次 run 开始的 UTC ISO 时间，各尺寸共享 run timestamp |
| run_id | 唯一 run，禁止同一个 ID 重写 |
| commit | binary 构建时对应的完整代码 Git hash，不是结果提交 hash |
| source_digest | 构建源码树 SHA256 |
| source_dirty | 正式成功实验必须为 0 |
| experiment | `baseline` 或 `profiling`；Profiler 数据不进入正式表 |
| gpu / gpu_uuid / device | 实际设备名、CUDA UUID 十六进制、可见 CUDA 设备索引 |
| compute_capability | 实际 major.minor |
| cuda | cudaRuntimeGetVersion 的整数，如 版本编码，而非猜测 nvcc 字符串 |
| cuda_driver | cudaDriverGetVersion API 支持版本整数；实际 NVIDIA 驱动版本另存 nvidia-smi |
| cublas_version | cublasGetVersion 的实际编码 |
| compiler / cuda_compiler | CMake 捕获的 host compiler 与 CUDA compiler ID/version |
| cuda_architectures | 实际构建配置，通常 native；完整解析值和命令见 build log |
| build_type | 必须 `Release` |
| kernel | `sgemm_v0_naive` 或 `cublas` |
| m / n / k | 矩阵尺寸；A=M×K，B=K×N，C=M×N |
| dtype / layout / math_mode | `fp32` / `row-major` / `fp32_pedantic` |
| alpha / beta | 固定 1 / 0，计时中不累积前一轮 C |
| warmup / iterations | 实際 warmup 数和 Event 样本数，必须 ≥10/≥30；正式默认 10/50 |
| seed / input_hash | 随机输入配置和实际 A/B 内容身份 |
| block_x / block_y | Naive launch block=16/16；cuBLAS 内部配置未公开，因此留空 |
| bm / bn / bk | Stage 0 未使用 shared/block tile，留空而非伪造 |
| tm / tn | Naive=1/1；cuBLAS 未声明 thread tile，留空 |
| atol / rtol | 逐元素混合容差，默认均 1e-3 |
| min_ms / max_ms | 单次 Event elapsed time 最小/最大值，毫秒 |
| median_ms / mean_ms / std_ms | 中位数/均值/总体标准差 ddof=0，毫秒 |
| gflops | `2*M*N*K/(median_ms*1e6)`，自动计算 |
| speedup_vs_naive | 本行 GFLOPS / 同 run 同 shape Naive GFLOPS；Naive=1 |
| cublas_ratio | 本行 GFLOPS / 同组 cuBLAS GFLOPS ×100，百分比；cuBLAS=100 |
| max_error / mean_error | 最终 GPU 输出对 CPU oracle 的 max/mean absolute error |
| relative_error | max(abs(actual-reference)/max(abs(reference),1e-12)) |
| violations / nonfinite | 超混合容差/非有限元素数量，成功必须均为 0 |
| status | `ok`、`failed_correctness` 等；失败行无成功 time/GFLOPS |

raw sample CSV 在上述 schema 后增加 `sample_index`（0-based）和 `elapsed_ms`，status=`timing_sample`；汇总统计字段在 raw sample 内留空。raw samples 是计时事实，不表示最终 correctness 已通过。`correctness.csv` 保存 `passed_initial`、`passed_final` 或失败状态，所有错误数字都保存。

独立只运行一个 kernel 时，缺少的 comparison 字段留空。正式 Runner 强制两个 kernel；分析器要求两者都存在才生成配对报告。

## 误差与失败

任何非有限 actual/reference 都失败。有限元素必须满足：

```text
abs(actual-reference) <= atol + rtol*abs(reference)
```

相对误差仅作诊断，接近零参考值会放大。误差异常时先检查 row-major 映射、leading dimension、边界、alpha/beta、数学模式，不能悄悄放宽阈值使失败实验“通过”。更改容差是一个新实验条件，必须记录并保留旧失败实验。

初始化校验、计时后校验、CUDA API/launch/Event/CSV 写入错误返回非零。`failure.txt`/runner status 与已有 raw 数据保留；分析器要求完整成功标记，否则拒绝报告。正式四尺寸过程中若中途失败，之前成功尺寸的行不会被删除，但不把整个 run 当成验收成功。

## 控制变量和重复

- 关闭不必要的 GPU 任务，记录运行前后 nvidia-smi -q，包括 P-state、温度、实际时钟、ECC 和其他进程。
- 不主动改变 power limit、锁频或驱动配置；若手动改变必须记录并作为另一组实验。
- 先运行小尺寸 smoke，再四尺寸完整实验；CPU oracle 的耗时不进入 CUDA timing。
- 固定先 Naive 后 cuBLAS，各自 warmup。顺序、热状态和频率可能产生系统误差；建议独立重复整个 run。
- `REPEATS=3 ./scripts/run_gemm_benchmark.sh` 创建三个 run，不把它们错误配成一个 baseline。
- 不把 Debug 与 Release、Profiler 与非 Profiler、其他 GPU/数学模式/尺寸的实验直接求 speedup。
- median 仍可能受热降频、其他进程影响；仅凭事件正确不能证明控制变量完全一致，必须审查环境快照。

分析工具逐 run 逐 shape 输出表，不挑选“最好看”的一条，不自动删除 failed/slow run。最新报告在 `docs/baseline.md`，历史报告在每个 raw run 内；可对多个成功 run 再独立分析，保留范围而非伪造统一数字。

## 发布和追溯

结果提交前确认原始数据齐全、报告生成成功、未含隐私信息（例如环境 PATH）。提交 `results/` 下 CSV/文本构建证据与生成的 baseline.md。不要提交 build binary。Profiler 二进制通常很大，`.gitignore` 排除它们；外部归档后提交 SHA256、位置、工具版本和文本导出。

改变代码并提交后必须重新 `build_server.sh`；单纯 `git pull` 不会让旧 binary 变成新 commit。跑完实验再提交结果时，CSV 继续保留原先被测试代码 commit，不要手工“更新”为结果提交的 hash。
