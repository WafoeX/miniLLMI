# mini-llm-runtime — Stage 13 implementation / T4 pending

基于 C++17/CUDA 的推理引擎项目，**Stage 0–12 已通过各自必需验收项**；Stage 13 的 C1–C3 与 C4 CPU 路径已在本地完成，必需的 clean T4 混合路径证据待 Colab 验收，因此 Stage 13 尚未关闭。详见 [路线图](docs/roadmap.md)。已有统一 Tensor/shared Storage、backend-neutral 算子、冻结 DAG/顺序 executor、有界 trace、arena/生命周期内存规划、CPU backend/显式优化路径、CUDA storage/copy/GEMM、Nsight 对比及异构 scheduler。

默认保留 **FP64 CPU reference math、dynamic allocation 与 CUDA v0**；Stage 0 与冻结 CPU 基线不改写。Stage 13 新增配置/21 个命名参数绑定、91-node decoder block 与 185-node 两层 no-cache prefill/logits 图；所有 attention/MLP 中间量均走原 graph/planner/backend 层。没有 tokenizer/model-file loader、KV Cache、自回归循环、量化、性能收益或语言质量声明；Stage 14–19 尚未实现。

## Stage 13 状态（C1–C3 与 C4 CPU 已完成，T4 C4 待验收）

冻结 `[256,0,1,257]` logits 与独立 Python 标量 oracle 一致；动态与 planned CPU 输出一致，planned warmed execute 的中间 backing allocation 为 0。shape 改变必须显式重建/replan，同 shape 可在输出释放后复用。混合路径源码声明 21 个 learned projection 使用 CUDA hint，其余 primitive 通过 scheduler 显式回到 CPU；真实 T4 编译、COPY 计数和数值容差必须按 [Stage 13 Colab 步骤](docs/stage13_colab.md) 验收。参见 [decoder 契约](docs/decoder.md)、[临时验收报告](docs/stage13_report.md) 与 [Stage 13 任务书](docs/tasks/stage-13-decoder.md)。

## Stage 12 验收（C1–C5 完成）

CPU Release **37/37 CTests** 通过。Tesla T4 上 GPU 标签 **5/5**、transformer 标签 **3/3** 通过；`cuda_transformer_ops` 验证 CUDA projection MATMUL、两次 H2D/一次 D2H 显式调度传输以及 CPU Softmax/RMSNorm。参见 [Stage 12 验收记录](docs/stage12_report.md)、[primitive 任务书](docs/tasks/stage-12-transformer-ops.md) 和 [Colab 步骤](docs/stage12_colab.md)。这是功能/集成验收而非基准测试。

## Stage 11 验收（C1/C2/C4 完成，C3 可选跳过）

C1 已 CPU 验收；C2 T4 验收覆盖显式传输、状态版本和 alias 生命周期；C4 在 clean `906af4d` 上完成三轮固定 64×64×64 CUDA-v0 手动 COPY／自动插入配对执行，结果提交 `a84fbd8`。**38/38 CTests、36/36 文件哈希、60 条原始计时记录和 12 份完整输出/trace 快照**独立复核通过。

配对比值中位数 **1.004480×** 仅作诊断；第三轮自动图稍慢且保留，没有调度器提速门槛或收益宣称。每次执行实际 4 copies / 65536 bytes / 9 backend dispatches / 2 switches，执行期中间 backing 分配为 0。计时包含同步执行、counter/full-output 检查和 output release，不含构图/rewrite/prepare/trace I/O，也不是裸 kernel 或模型延迟。

[Scheduler 契约](docs/scheduler.md) · [Stage 11 验收记录](docs/stage11_report.md) · [Colab 复现步骤](docs/stage11_colab.md) · [生成的 T4 配对报告](results/scheduler/stage11-c4/20261007T101728202151Z-1834/report.md)

以下保留早期阶段的历史验收入口，不表示项目仍停留在该阶段。

Stage 0 naive GEMM 与历史结果保持不变。Stage 1 的 gate 是 CPU-only Debug/Release、属性测试与 ASan/UBSan；不要求 CUDA allocation 或服务器性能数据。后续按 [路线图](docs/roadmap.md) 与 [Change 任务书](docs/tasks/README.md) 执行。

## Stage 6 验收（明确 CPU-only，无需 GPU）

```bash
python3 tools/validate_cpu.py
python3 tools/run_cpu_benchmark.py
# dirty 源码只允许开发正确性验证，不允许计时：
# python3 tools/validate_cpu.py --allow-dirty
```

[CPU backend / 冻结基线契约](docs/cpu_backend.md)、[Stage 6 验收记录](docs/stage6_report.md)。同一 clean source 的 Release/Debug/ASan+UBSan 各 29/29 通过；128/256/512/1024 完成三轮独立 Release 基线，原始样本、GFLOPS、compiler auto-vectorization 备注和 provenance 已归档。Transformer primitives 仍明确 `Unsupported`；小图 prepared reuse 更慢，数据保留且不改默认。本节仅记录 Stage 6 历史基线，不是 CPU/GPU 对比。Stage 7 后续 CPU-only 验收另见 [CPU 并行契约](docs/cpu_parallel.md) 与 [Stage 7 报告](docs/stage7_report.md)。

## Stage 5 验收（明确 CPU-only，无需 GPU）

```bash
python3 tools/validate_planner.py
python3 tools/run_planner_benchmark.py
# dirty 源码仅允许正确性开发验证：
# python3 tools/validate_planner.py --allow-dirty
```

[Planner 契约与冻结基准](docs/planner.md)、[Stage 5 验收记录](docs/stage5_report.md)。Release/Debug/ASan+UBSan 各 23/23 通过；预声明 chain 的执行期中间 backing 分配 11→0，prepared capacity 3072→512 bytes（降低 83.33%）。与最后使用释放的 dynamic 对比，计划执行在本机小图上**更慢**，原始数据全部保留；无延迟收益 gate。默认仍为 dynamic，inplace 为 `skipped_optional`；零 backing 分配不等于零 C++ heap 分配，不宣称模型或 CUDA 性能收益。

## Stage 4 验收（明确 CPU-only，无需 GPU）

```bash
# clean 已提交源码；fresh Release/Debug/ASan+UBSan/production 正确性证据
python3 tools/validate_arena.py
# 独立 CPU-only 诊断基准：3 paired runs / 3 warmups / 10 samples / batch 20
python3 tools/run_allocator_benchmark.py
# 有本地未跟踪源码时，用 clean worktree；dirty 只允许正确性验证，不允许计时：
# python3 tools/validate_arena.py --allow-dirty
```

[Arena / provider / benchmark 契约](docs/arena.md)、[Stage 4 验收记录](docs/stage4_report.md)。默认 executor 仍为 S3 dynamic baseline；arena 明确 opt-in，容量不足无 malloc fallback，持有输出/alias 时禁止覆盖同一 context。基准含 synthetic、chain、diamond，较慢结果保留；不宣称 Stage 5 planner、零 C++ heap 或模型吞吐收益。

## Stage 3 验收（CPU-only，无需 GPU）

```bash
# clean 已提交源码；fresh Release/Debug/ASan+UBSan/production 原始证据
python3 tools/validate_graph.py
# 开发源码带未跟踪文件时，仅作 labelled development validation：
# python3 tools/validate_graph.py --allow-dirty
# 当前 build 中单独运行 Stage 3 tests：
# cmake --build build-local --target check_graph
```

[Graph / executor / trace 契约](docs/graph.md)、[Stage 3 验收记录](docs/stage3_report.md)。动态 baseline 每个 NewTensor 节点在 execute 内分配输出，最后使用释放、graph outputs pin backing，alias 保留 base；没有隐式 copy 或自动切换到 arena；S4 仅通过显式 provider 选择策略。所有执行复用 S2 CPU reference。

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
./scripts/profile_gemm.sh nsys v0
./scripts/profile_gemm.sh ncu v0
```

采集仅用于分析；Profiler 的 Event 时间带扰动，标记 `experiment=profiling`，不得用于正式性能比较。Stage 10 的 T4 V0/V1 全量采集使用 `python3 tools/run_cuda_gemm_stage10.py`；详见 [profiling protocol](docs/profiling.md) 和 [Colab acceptance guide](docs/stage10_colab.md)。

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
