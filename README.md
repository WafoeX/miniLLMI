# mini-llm-runtime — Stage 3

基于 C++17/CUDA 的推理引擎项目。**Stage 0 已通过 T4 验收；Stage 1 统一 Tensor/shared Storage；Stage 2 实现 backend-neutral 算子与 core reference；Stage 3 实现冻结 DAG、确定性拓扑、顺序 CPU executor、最后使用释放/alias 保活及有界执行 trace。** Tensor 支持 FP32/INT32、checked shape/stride/offset、CPU storage、零拷贝变换和显式 CPU copy。Transformer 仍只有契约/validators 和离线 expected fixtures，尚无 Transformer kernels、Arena/planner、scheduler、模型执行、KV Cache 或量化实现。

Stage 0 naive GEMM 与历史结果保持不变。Stage 1 的 gate 是 CPU-only Debug/Release、属性测试与 ASan/UBSan；不要求 CUDA allocation 或服务器性能数据。后续按 [路线图](docs/roadmap.md) 与 [Change 任务书](docs/tasks/README.md) 执行。

## Stage 3 验收（CPU-only，无需 GPU）

```bash
# clean 已提交源码；fresh Release/Debug/ASan+UBSan/production 原始证据
python3 tools/validate_graph.py
# 开发源码带未跟踪文件时，仅作 labelled development validation：
# python3 tools/validate_graph.py --allow-dirty
# 当前 build 中单独运行 Stage 3 tests：
# cmake --build build-local --target check_graph
```

[Graph / executor / trace 契约](docs/graph.md)、[Stage 3 验收记录](docs/stage3_report.md)。动态 baseline 每个 NewTensor 节点在 execute 内分配输出，最后使用释放、graph outputs pin backing，alias 保留 base；没有隐式 copy、arena 或性能收益声明。所有执行复用 S2 CPU reference。

## Stage 2 验收（无需 GPU / Colab 服务器）

```bash
# clean 已提交源码；新建唯一 CPU 验收 run，保存 Debug/Release/sanitizer/production 原始证据
python3 tools/validate_operators.py
python3 tools/generate_operator_fixtures.py --check
# 开发源码带未跟踪文件时，仅作 labelled development validation：
# python3 tools/validate_operators.py --allow-dirty
```

[Operator / tiny-model 契约](docs/operators.md)、[Stage 2 验收记录](docs/stage2_report.md)。无隐藏 materialization、buffer allocation 或 CUDA host fallback；没有性能收益声明。用户选择不进行 Stage 1 服务器复验，本阶段亦无强制 GPU gate，未执行服务器验收。

## Stage 1 验收

```bash
# 要求源码已提交；会新建唯一 run 目录，保存原始日志、source identity 与构建快照
python3 tools/validate_tensor.py
# 若仍有未跟踪源码（例如本地 AGENTS.md），仅作明确标记的开发验证：
# python3 tools/validate_tensor.py --allow-dirty
```

参见 [Tensor API 契约](docs/tensor.md)、[Stage 1 验收记录](docs/stage1_report.md) 与 [全新 Colab / Drive 恢复指南](docs/stage1_colab.md)。这里没有 Stage 1 吞吐或性能提升声明。

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

`build*/` 和 Profiler 大型二进制不进 Git。Profiler 二进制保存到外部制品存储，保留校验和、获取位置及导出文本。Stage 0 已验收；历史报告中的待验收文字是当时状态，当前状态以路线图和各阶段验收记录为准。Stage 1 不重跑或改写 Stage 0 性能数据。
