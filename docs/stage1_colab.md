# Stage 1 — 全新 Colab / Google Drive 复验指南

**是否必须服务器验收：不必须。** Stage 1 gate 是 CPU-only Debug/Release、随机属性测试和 ASan/UBSan；CUDA 分配/copy/backend 到 S8 才验收。这里提供额外 Linux 复验和可选 T4 CUDA 构建回归；不要为了 Stage 1 重跑 4096³ baseline 或 Nsight。

运行时丢失只会丢 `/content`。Drive 中旧 Stage 0 的结果/profiler 制品不需要重新生成或移动。**旧源码备份没有本次 Stage 1 实现**：先传送当前提交，再在 `/content` 新建 checkout 和新 build；不能复用旧 CMakeCache/二进制。

## 0. 本地传送本次代码（二选一）

### A. 上传 Git bundle 到 Drive（不依赖 GitHub push；推荐）

在本地仓库执行：

```bash
mkdir -p build-transfer
# 本分支包含 Stage 1 源码、验收记录及历史 Stage 0 文本结果；不包含忽略的大型 profiler 二进制
# bundle 输出是新制品，不要覆盖旧 Drive 归档
git bundle create build-transfer/stage1.bundle feat/tensor-stage1
(cd build-transfer && shasum -a 256 stage1.bundle > stage1.bundle.sha256)
```

把这两个文件上传到 Drive 的**新目录**（例如 `MyDrive/miniLLMI-stage1-transfer/`）。用 Git bundle 可保留源码提交历史；只有 ZIP 且没有 `.git` 时，版本/dirty provenance 无法满足默认验收脚本。

### B. 先 push，再用 HTTPS clone

```bash
git push -u origin feat/tensor-stage1
```

若未执行 push，不能假设 GitHub 已有本次实现。Colab 不自动继承本地 SSH key；公共仓库用 HTTPS，私有仓库请使用自己的安全认证方式，不要把 token 写进结果日志。

## 1. 新 Colab：挂载 Drive（Python cell）

CPU 复验可选择普通 CPU runtime；仅第 5 节可选回归需要 GPU runtime，选择 T4 并实际检查设备。

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. 恢复最新仓库（bash cell；选一个来源）

bundle 方式：

```bash
%%bash
set -euo pipefail
TRANSFER=/content/drive/MyDrive/miniLLMI-stage1-transfer
cd "$TRANSFER"
sha256sum --check stage1.bundle.sha256
# 若目录已存在则主动停止，避免覆盖任何已有结果
test ! -e /content/miniLLMI-stage1
git clone --branch feat/tensor-stage1 "$TRANSFER/stage1.bundle" /content/miniLLMI-stage1
cd /content/miniLLMI-stage1
git rev-parse HEAD
git status --short
```

或 GitHub 方式（仅在 push 成功后）：

```bash
%%bash
set -euo pipefail
test ! -e /content/miniLLMI-stage1
git clone --branch feat/tensor-stage1 --single-branch \
  https://github.com/WafoeX/miniLLMI.git /content/miniLLMI-stage1
cd /content/miniLLMI-stage1
git rev-parse HEAD
git status --short
```

已有 Drive 旧仓库/归档可以继续保留作为历史证据；**不要**把它的 `.git`、旧 build 或散落源码覆盖到这个 checkout。受版本管理的 Stage 0 文本结果已随 Git 历史传入；外部 nsys/ncu 二进制留在旧 Drive 制品位置即可。

## 3. 依赖和正式 CPU-only 复验（bash cell）

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage1
# Colab 通常已有这些工具；安装缺失的基础依赖，不安装 CUDA/Nsight
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git python3 python3-pip
# 确保 CMake >= 3.24；使用 Python 分发的 CMake，不降级 CUDA Toolkit
python3 -m pip install --quiet 'cmake>=3.24,<5'
export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$HOME/.local/bin:$PATH"
cmake --version
c++ --version
python3 tools/provenance.py
# 默认拒绝未提交源码。此处不要使用 --allow-dirty 来绕过错误。
python3 tools/validate_tensor.py --jobs 2
```

自动依次执行：fresh Release → fresh Debug → ASan/UBSan（含 `check_tensor_sanitizers`）→ `BUILD_TESTING=OFF` Release。前三种模式各运行完整 CTest；正式成功标志为末尾 `Stage 1 CPU validation: PASS`，每个命令 exit=0，manifest 的 `status=passed`、`source_dirty=false` 且 source 前后相同。详细范围/预期测试数见 [验收记录](stage1_report.md)。若 Colab 的 Linux sanitizer 失败，保留原始日志并排查，不得用 macOS 结果冒充 Linux 通过或关掉 sanitizer 后报成功。

输出：`results/tensor/local/<唯一-run_id>/`，含 source/manifest、完整 build/CTest 日志、CMakeCache 与 compile_commands 快照。构建目录在被忽略的 `build-stage1-validation/`，不污染源码。**本地验收不等于这些 Colab 命令已被实际执行。**

## 4. 不论成功或失败，立刻保存到 Drive（单独 bash cell）

即使上一个 cell 失败也执行；这是新的归档，不覆盖旧结果：

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage1
ARCHIVE="/content/drive/MyDrive/miniLLMI-stage1-validation/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ARCHIVE"
# 保存 checkout 身份，以及所有新增/历史文本结果；失败日志也保留
git rev-parse HEAD > "$ARCHIVE/checkout-commit.txt"
python3 tools/provenance.py > "$ARCHIVE/source-at-archive.json"
cp -a results "$ARCHIVE/"
git bundle create "$ARCHIVE/checkout.bundle" --all
(cd "$ARCHIVE" && sha256sum checkout.bundle > checkout.bundle.sha256)
printf 'Saved to %s\n' "$ARCHIVE"
```

若要把服务器新证据提交回 Git，使用单独 `test:` 结果提交，保留 manifest 的 tested source identity；不要改写旧日志/metrics。bundle clone 的 origin 是 Drive bundle 文件，不能向它 push；如需 GitHub push，应先配置真实 GitHub remote 和认证。

## 5. 可选：T4 CUDA-enabled 构建/正确性回归（不是 Stage 1 gate）

只验证项目还能在 CUDA-enabled 配置下构建，不实现或认证 Stage 1 CUDA allocation。GPU runtime 丢失后 NVCC/驱动/架构以新运行时实际查询为准；不要复用旧 Toolkit 路径/缓存：

```bash
%%bash
set -euo pipefail
cd /content/miniLLMI-stage1
CHECK="results/tensor/colab-cuda/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$CHECK"
exec > >(tee "$CHECK/session.log") 2>&1
trap 'rc=$?; printf "exit_status=%s\n" "$rc" > "$CHECK/status.txt"' EXIT
nvidia-smi
nvcc --version
python3 tools/provenance.py
BUILD_DIR=build-colab-stage1-cuda JOBS=2 ./scripts/build_server.sh
TEST_LOG="$CHECK/ctest.log" BUILD_DIR=build-colab-stage1-cuda ./scripts/run_tests.sh
```

确认 `nvidia-smi` 实际为 T4 才能标记 T4 复验；工具缺失、设备不同或构建失败都原样记录，不假定沿用旧环境。此步骤不需要 baseline benchmark/profiling。完成/失败后再次执行第 4 节保存结果；`results/environment/` 构建记录与 `results/tensor/colab-cuda/` 日志也会随整个 results 目录保存。sanitizer 保持单独 CPU build，不能与 `ENABLE_CUDA=ON` 合用。
