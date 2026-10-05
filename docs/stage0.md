# Stage 0 操作指南与验收 checklist

## 1. 文件目录与 16 项交付映射

```text
CMakeLists.txt
README.md
.gitignore
include/stage0/
  common.hpp             # CPU reference / correctness / statistics / CSV 接口
  build_info.hpp.in      # 编译器、Release、CUDA arch + 构建时 provenance
  cuda_utils.hpp         # CUDA/cuBLAS 错误处理、RAII、CUDA Event Timer
  gemm_cuda.hpp          # 两个 GEMM 的统一入口
src/common.cpp           # CPU Reference、误差、统计、GFLOPS、CSV
backend/cuda/
  sgemm_v0_naive.cu       # 每 thread 一个 C[i,j]
  cublas_gemm.cu          # 正确 row-major 映射
bench/
  gpu_info.cpp           # cudaGetDeviceProperties，host-only C++17
  bench_gemm.cpp         # 输入→CPU参考→两者校验→warmup→计时→再校验→CSV
scripts/
  common.sh
  collect_environment.sh
  build_server.sh
  run_tests.sh
  run_gemm_benchmark.sh
  run_benchmarks.sh       # Stage 0 仅转发 GEMM；不假装其他模块存在
  profile_gemm.sh         # baseline 单独 Profiling
tools/
  provenance.py
  analyze_results.py
tests/
  test_common.cpp
  test_cuda_gemm.cpp
  test_tools.py
docs/
  stage0.md
  architecture.md
  experiments.md
  profiling.md
  baseline.md            # 实测后自动生成；初始只有未实测声明
  stage0_report.md
results/
  environment/
  gemm/raw/
  cpu/ allocator/ kv_cache/ quant/ inference/  # 空占位目录，尚未实现
  profiling/nsys/
  profiling/ncu/
```

| 请求项 | 完整实现/说明 |
|---|---|
| 1 文件目录 | 本页 |
| 2 CMakeLists | 根 `CMakeLists.txt` |
| 3 GPU 信息查询 | `bench/gpu_info.cpp` |
| 4 CPU Reference | `src/common.cpp::cpu_reference` |
| 5 Naive CUDA | `backend/cuda/sgemm_v0_naive.cu` |
| 6 cuBLAS | `backend/cuda/cublas_gemm.cu` |
| 7 Timer | `include/stage0/cuda_utils.hpp::EventTimer` |
| 8 Correctness | `src/common.cpp::compare` + `bench/bench_gemm.cpp` |
| 9 CSV | `src/common.cpp::append_csv`、`summary_columns` |
| 10 环境脚本 | `scripts/collect_environment.sh` |
| 11 构建脚本 | `scripts/build_server.sh` |
| 12 Benchmark Runner | `scripts/run_gemm_benchmark.sh` |
| 13 本地 Git 流程 | 下节 |
| 14 服务器流程 | 下节 |
| 15 CSV 字段 | [experiments.md](experiments.md) |
| 16 验收 checklist | 本页末尾 |

## 2. 本地开发 → Commit → Push

当前仓库已经在本地初始化，分支 `main`，不需要重复 `git init`。没有远程 URL 时不创建虚假 remote，也无法验证 Push。

首次绑定你提供的仓库地址：

```bash
git remote add origin <你的真实仓库URL>
git push -u origin main
```

后续每个可验收功能：

```bash
git switch -c feat/stage0-fix
# 编码后
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build-local --parallel 4
BUILD_DIR=build-local ./scripts/run_tests.sh
for f in scripts/*.sh; do bash -n "$f" || exit; done
git diff --check
git status --short
git add <此次功能涉及的文件>
git commit -m "fix: correct stage 0 benchmark validation"
git push -u origin feat/stage0-fix
```

服务器拉取该 feature branch 后测试；测试通过再合并。优化阶段应另开 `perf:` 提交；当前不实现任何优化版本。

## 3. 服务器首次部署及更新

```bash
# 首次
git clone <你的真实仓库URL> ~/mini-llm-runtime
cd ~/mini-llm-runtime
# 后续：需确保工作树可安全拉取，先提交实验结果
git pull --ff-only origin main
git rev-parse HEAD
nvidia-smi
nvcc --version

# 编译前的命令输出也会在构建后的环境采集中再次保存
./scripts/build_server.sh
./scripts/run_tests.sh
./build/gpu_info
# 单独环境采集（生成唯一目录及最新的规范文件名）
./scripts/collect_environment.sh

# 快速 smoke：非方阵、非 tile 整数倍，验证 row-major 和边界
./scripts/run_gemm_benchmark.sh --m 31 --n 33 --k 17
# 正式 Stage 0 验收：默认四种尺寸 512/1024/2048/4096
./scripts/run_gemm_benchmark.sh
# 独立重复整个实验，保持 seed/warmup/iterations 一致
REPEATS=3 ./scripts/run_gemm_benchmark.sh

# 单独 baseline Profiling；不要拿 profiler 计时作为正式数据
./scripts/profile_gemm.sh nsys naive
./scripts/profile_gemm.sh ncu naive

# 审查数据及日志，再提交。实验 commit 与被测试代码 commit 是两个概念。
git status --short
git add results/ docs/baseline.md
git commit -m "bench: record T4 stage 0 baseline and raw samples"
git push origin main
```

Feature branch 的服务器流程把 `main` 换成实际 feature branch，先 `git switch --track origin/<branch>`。禁止用 `reset --hard` 覆盖尚未提交的实验。

脚本可从任意工作目录调用，会定位项目根。配置环境变量：`BUILD_DIR`、`JOBS`、`PYTHON`、`REPEATS`；例如 `JOBS=8 ./scripts/build_server.sh`。默认 CUDA architecture=`native`，来自可见 GPU。仅在自动检测确实不可用时，**先用真实查询得到 CC**，再显式传 `-DCMAKE_CUDA_ARCHITECTURES=<实际架构>`，不能拿参考机参数充当测量值。

`collect_environment.sh` 在缺少 GPU、nvcc 或 `gpu_info` 时保存错误输出并返回非零；不会伪造设备参数。CUDA 13 已移除 `cudaDeviceProp.memoryClockRate`，其他设备属性仍用 CUDA API 查询；显存时钟来自带 UUID 的 `nvidia-smi` 实际查询，并追加在 `gpu_info.txt`。

## 4. 每次 run 产物

```text
results/environment/{gpu_info,nvidia_smi,nvcc_version,system_info}.txt # 最新快照
results/environment/builds/<build_id>/                              # 完整构建证据
results/gemm/baseline.csv                                           # 追加汇总，不覆盖
results/gemm/raw/<run_id>/
  current_source.json / current_binary.json
  source_after.json
  build/                                                           # CMakeCache、完整日志、编译命令
  environment/                                                     # 本次不可变环境快照
  ctest.log / benchmark.log / command.txt / argv.txt
  nvidia_smi_before.txt / nvidia_smi_after.txt
  build_info.json
  <M>x<N>x<K>/
    correctness.csv                                                # 初始+最终各一次
    sgemm_v0_naive_samples.csv                                      # 50 个逐轮样本
    cublas_samples.csv
    summary.csv
  completed.txt                                                    # 所有请求尺寸都成功才生成
  runner_status.txt
  baseline.md                                                      # 本次报告
  failure.txt                                                      # 仅失败时存在，保留其他原始证据
```

Runner 序列化 CSV 写入，检测陈旧二进制、源码变更、非 Release、缺失构建证据。直接调用 `bench_gemm` 时必须自行保证单写入者；建议始终使用 Runner。

默认四种尺寸都会完整 CPU 校验，尤其 4096³ 的 ijk 很慢。单尺寸 smoke 成功不能替代四尺寸验收。每轮 CUDA 计时使用同一个 stream 上的 start/stop Event，同步 stop 后读取；不以 `std::chrono` 测 GPU。

## 5. Stage 0 验收 checklist

### 已在本地验证

- [x] Git 初始化，主分支 `main`。
- [x] CPU-only C++17 Release Configure/Build 成功。
- [x] CPU Reference：手算矩阵、非方阵、零输入、覆盖旧 C。
- [x] 误差检查：混合绝对/相对容差，超阈值、NaN、Inf 失败。
- [x] min/max/mean/median/总体 stddev 和 GFLOPS 计算单元测试。
- [x] CSV 转义、追加与 schema 检查单元测试。
- [x] 分析器校验原始样本、拒绝篡改统计/比值/版本/缺失样本/失败实验。
- [x] 源码 commit+digest+dirty 状态记录单元测试。
- [x] Bash 脚本语法检查；README 没有未经实测的性能数字。

### 必须在远程/服务器完成，不得提前打勾

- [ ] 本地成功 Push，服务器成功 Pull 同一 commit。
- [ ] `nvidia-smi`、nvcc、编译器、CMake、设备实际属性完整保存。
- [ ] T4 CUDA Release 编译成功，无未处理 CUDA 编译诊断。
- [ ] CUDA CTest（Naive/cuBLAS 非方阵、尾块、零值、输出 guard、Event）成功。
- [ ] 512、1024、2048、4096 四尺寸 Naive/cuBLAS 全部通过 CPU Reference。
- [ ] Event 时间为正；warmup=10；每 kernel 每尺寸 iterations=50；逐轮样本全部保存。
- [ ] 独立完整重复实验成功，不混合不同 run 的配对结果。
- [ ] GFLOPS、Naive/cuBLAS 百分比自动计算，原始数据与汇总一致。
- [ ] 汇总包含完整被测试代码 commit、源码 digest、GPU UUID、编译配置。
- [ ] 对失败/退化实验保留原始证据，不视为成功数据。
- [ ] `docs/baseline.md` 由真实 CSV 和 raw 样本自动生成。
- [ ] Baseline 的 nsys/ncu 文件及文本指标归档；没有把 profiler 时间混进正式数据。
- [ ] 服务器结果 commit+push，本地拉取后可重新核对报告。

**以上全部真实通过才进入 Stage 1。当前实现与指导停在 Stage 0。**
