# Stage 0 验收报告（本地实现与验证，GPU 验收待完成）

## 实现功能

完整 C++17/CMake 双模式工程，CPU oracle，Naive CUDA、cuBLAS wrapper，GPU 实际信息查询，RAII 与 CUDA Event Timer，严格误差检查，CSV 汇总/原始样本，版本 provenance，服务器环境/构建/测试/Benchmark/Profiling 脚本，原始数据复算和 baseline 文档生成。

未实现 Tensor、Graph Allocator、Transformer、KV Cache、INT8 或任何 CUDA 优化版本。

## 修改文件

当前项目从空目录创建。完整文件清单和 16 项交付映射见 [stage0.md](stage0.md)；主要入口 `CMakeLists.txt`、`src/common.cpp`、`backend/cuda/sgemm_v0_naive.cu`、`backend/cuda/cublas_gemm.cu`、`bench/bench_gemm.cpp`、`bench/gpu_info.cpp`、`scripts/`、`tools/`、`tests/`。

## 核心设计

common 无 CUDA 依赖；host-only Benchmark/GPU 查询/测试编排使用 `.cpp`；device kernel 用 `.cu`。CPU Reference 全矩阵单线程 ijk，FP64 累加到 FP32，不计性能。Naive 一 thread 一输出，正常 Release 编译，不故意降速。cuBLAS row-major 通过转置等价映射，无额外数据转置；数学模式固定 FP32 pedantic。

同 stream Event 计时，正确性前后检查；原始样本追加保存。官方实验需源码 clean、完整 commit、Release、binary/current-source/build-log 身份一致。更多边界见 [architecture.md](architecture.md)。

## Baseline、实验变量与控制变量

- Correctness Baseline：CPU Reference。
- Performance Baseline：`sgemm_v0_naive`。
- Performance Reference：同 FP32 pedantic 口径的 cuBLAS SGEMM。
- 变量：Naive vs cuBLAS；矩阵尺寸 512、1024、2048、4096，逐尺寸配对。
- 固定：同 A/B、seed=42、FP32、row-major、alpha=1/beta=0、warmup=10、iterations=50、同设备 UUID、同编译/版本/容差。
- 主性能指标：median_ms 对应 GFLOPS；另外保存 min/max/mean/总体 std。

## 本地测试方法与实测结果

环境：macOS/Darwin 25.6.0 arm64，AppleClang 21.0.0，CMake 4.4.3，Python 3.14.6。本机没有 nvcc/nvidia-smi/NVIDIA GPU，未安装 CUDA 依赖。

实际执行：

```bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build-local --parallel 4
ctest --test-dir build-local --output-on-failure
python3 tests/test_tools.py
for f in scripts/*.sh; do bash -n "$f" || exit; done
```

- Release Configure/Build：退出 0。
- CTest：`common`、`tools` 两个测试入口全部通过；`tools` 包含 20 个 Python unittest。
- 额外本地 ASan+UBSan Release 构建和相同 CTest：全部通过，无运行时 sanitizer 报错；这不是 GPU 验证或性能数据。
- CPU C++ 测试：手算/非方阵/零输入/覆盖旧输出，确定性输入、hash、错误容差、NaN/Inf、median/std/GFLOPS、CSV 转义/追加/schema mismatch。
- Python：原始样本复算、条件不一致、伪造 GFLOPS、错误比值、失败状态、缺失/篡改样本、无完整 run、恶意 run path、Debug/未知 commit/dirty、数值异常、CLI 失败不覆盖旧报告、provenance 与头文件稳定性。
- Bash 路径测试：在临时测试仓库模拟**失败**工具调用；验证环境采集失败仍保存错误输出、重复采集不覆盖、陈旧二进制在 Benchmark 前失败、释放 runner 锁且不产生性能 CSV。测试 fixture 不进入真实 `results/`，不充当 GPU 性能。
- 新测试发现 runner 依赖已有 `results/gemm/raw` 目录的问题，已修复为自动创建 parent；修复后的 20 个 Python 测试与 CTest 全部通过。
- Bash 语法检查：通过。shellcheck 本机未安装，不能声称独立 shellcheck 完整验收。
- 主动 LSP 探测用于本地代码诊断；CUDA 曾因 Toolkit/libdevice 缺失产生连带错误，根因已标记待服务器验证，不添加代码忽略注释。后续 host-only 入口改为 `.cpp`，本地探测没有返回新诊断；**这不证明 CUDA headers/device code 已完整检查**。CMake LSP 曾不可用，实际 CMake Configure/Build 是本地构建依据。

原始本地验证日志在 `build-local/`；确认代码提交后另保存带真实 commit 的本地验证证据到 `results/environment/local/`。这些是本地功能验证，不是 GPU Benchmark 数字。

## GPU 实验命令（待实际执行）

```bash
git pull --ff-only origin main
git rev-parse HEAD
nvidia-smi
nvcc --version
./scripts/build_server.sh
./scripts/run_tests.sh
./scripts/run_gemm_benchmark.sh --sizes 512
./scripts/run_gemm_benchmark.sh
REPEATS=3 ./scripts/run_gemm_benchmark.sh
./scripts/profile_gemm.sh nsys naive
./scripts/profile_gemm.sh ncu naive
```

脚本保存完整环境、配置、日志、逐轮样本和代码身份；成功后由 `tools/analyze_results.py` 自动生成 [baseline.md](baseline.md)。4096³ 的单线程 CPU reference 可能耗时较长，但不能因此跳过正确性。

## 实验结果、Baseline 比较、性能变化、原因分析

**暂无 T4 实验数据。** 不报告 GFLOPS、speedup、cuBLAS ratio 或 profiling 瓶颈，不编造默认结果。尚未创建真实 `results/gemm/baseline.csv`，baseline.md 仍是待实验声明。

实现提供采集与分析能力，不能替代实际服务器实验；性能变化和因果原因只能待 nsys/ncu+CSV 返回后分析。

## 是否达到验收标准

本地实现和 CPU/工具功能验证通过，Git 已初始化。**Stage 0 整体验收尚未通过**：未提供远程 URL/服务器访问环境，无法验证 Push/Pull；没有 NVIDIA GPU/Toolkit，无法验证 CUDA Release Build、GPU Correctness、四尺寸 Benchmark、独立重复和 Profiling。

完整 checklist 见 [stage0.md](stage0.md)。不得把 CPU-only 通过或编辑器静态提示写成 T4 实测通过。

## 当前 Git Commit

本报告和实现随首次 `feat:` 提交；精确 code commit 由随后保存的 `results/environment/local/provenance.json` 和最终交付消息给出。仓库当前 HEAD 以 `git rev-parse HEAD` 为准。编译/实验时程序自动嵌入当时的代码 commit；不会用结果提交的 hash 回填旧实验。

## 下一阶段

当前停止在 Stage 0。先提供远程仓库 URL，完成 Push/Pull 和 T4 验收；仅全部 checklist 通过之后，才考虑 Stage 1，不在本次实现。
