# mini-llm-runtime — Stage 0

基于 C++17/CUDA 的推理引擎项目。目前**仅实现 Stage 0：环境、可信 GEMM Baseline 与实验基础设施**。不包含 Tensor、Graph、Allocator、Transformer、KV Cache 或量化实现。

**状态：Git 已初始化；本地 CPU-only Release 构建与测试已验证。CUDA 编译、GPU 正确性、T4 性能和远程 Push/Pull 尚待实测，不能视为 Stage 0 全部验收。**

## 本地（不需要 NVIDIA GPU）

依赖：CMake ≥3.24、C++17 编译器、Python ≥3.8、Git、Bash。无外部测试框架下载。

```bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build-local --parallel 4
BUILD_DIR=build-local ./scripts/run_tests.sh
# 编辑器可读取本地编译命令：文件已被 .gitignore 排除
ln -sfn build-local/compile_commands.json compile_commands.json
```

CPU Reference、数值校验、统计、CSV、结果分析、源码版本记录可在本地验证。CUDA 文件需要 Toolkit 才能完整静态检查；CPU-only 测试通过不代表 CUDA 通过。

## T4 服务器

需要实际 NVIDIA 驱动/可见 GPU、支持宿主编译器的 CUDA Toolkit ≥11.0、cuBLAS；Nsight 工具用于单独 Profiling。参数以实际查询为准。

```bash
git pull --ff-only origin main
git rev-parse HEAD
nvidia-smi
nvcc --version
./scripts/build_server.sh
./scripts/run_tests.sh
# 首先小规模检查（不代表四个正式尺寸已验收）
./scripts/run_gemm_benchmark.sh --sizes 512
# 正式：512、1024、2048、4096；warmup=10、iterations=50
./scripts/run_gemm_benchmark.sh
# 可选：独立重复完整实验，不混合不同 run 的配对结果
REPEATS=3 ./scripts/run_gemm_benchmark.sh
```

CMake 默认查询可见 GPU 的 `native` architecture，不硬编码 T4 参数。`BUILD_DIR` 默认 `build/`，`JOBS` 默认 4。CMakeCache、完整 configure/build 命令、日志、编译命令和环境自动归档。正式 Benchmark 拒绝 Debug、未提交源码及陈旧二进制。

**注意：每个尺寸都运行完整单线程 ijk CPU Reference，4096³ 的 CPU 校验可能很慢。其时间不计入 GPU 性能。不要因等待过久就跳过验证或用未经验证数据生成报告。**

## Baseline 与数据

- CPU Reference：FP32 输入/输出、FP64 累加、单线程 ijk，仅作正确性参考。
- `sgemm_v0_naive`：一个 thread 计算一个元素，16×16 launch block；无高级优化。
- `cublas`：row-major 映射、FP32 `CUBLAS_PEDANTIC_MATH`、alpha=1、beta=0。
- 同输入/seed、同 stream；CUDA Event 逐轮同步；分配、拷贝、CPU 校验和 CSV 写入不计入计时间隔。
- `median` 为主指标；同时保存 min/max/mean/总体 stddev、max/mean absolute error 与 relative error。
- NaN/Inf、超阈值误差、CUDA API 错误导致非零退出；失败日志及已有原始数据保留。

真实 GPU 数据运行后才创建 `results/gemm/baseline.csv`。每次原始计时、环境、构建信息、校验记录和报告保存在 `results/gemm/raw/<run_id>/`。分析工具核对原始样本并重新计算 GFLOPS/比值，生成 [docs/baseline.md](docs/baseline.md)；README 不手填性能数字。

```bash
python3 tools/analyze_results.py --run-id <实际-run-id> \
  --csv results/gemm/baseline.csv --raw-root results/gemm/raw --output docs/baseline.md
```

`cuBLAS ratio` 单位是 **%**。此处 cuBLAS 是同 FP32 pedantic 口径的参考，不代表其他数学模式的性能上限。Stage 0 不设参考设备上的绝对 GFLOPS 或优化版本趋势目标。

## Profiling（与正式计时分开）

```bash
./scripts/profile_gemm.sh nsys naive
./scripts/profile_gemm.sh ncu naive
```

采集仅用于分析 baseline；Profiler 的 Event 时间带扰动，标记 `experiment=profiling`，不得用于正式性能比较。详见 [docs/profiling.md](docs/profiling.md)。

## 文件与操作文档

- [Stage 0 文件映射、Git/服务器流程、验收 checklist](docs/stage0.md)
- [核心设计](docs/architecture.md)
- [CSV 字段、实验方法与失败策略](docs/experiments.md)
- [本地验收报告](docs/stage0_report.md)

实验结果提交时保留原始 CSV、文本环境与日志：

```bash
git add results/ docs/baseline.md
git commit -m "bench: record T4 stage 0 baselines"
git push origin main
```

`build*/` 和 Profiler 大型二进制不进 Git。Profiler 二进制请保存到外部制品存储，提交文件校验和、获取位置及导出的文本指标。仅在 Stage 0 所有验收项真实通过后才能进入 Stage 1；当前停在 Stage 0。
