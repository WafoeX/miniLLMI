# Stage 1 — Tensor/Storage 验收记录

**结论：S1-C1–C6 完成，CPU-only stage gate 通过。** 不进入 Stage 2；不实现 CUDA 分配或设备 copy，不声明吞吐提升。Stage 0 的 `sgemm_v0_naive`、cuBLAS/reference 实现和历史结果未修改。

## 冻结源码与最终证据

- Tested source commit：`f0aa8ac7fc86e3cd9bc485f9e637f90d1f118bde`。
- Source SHA-256 digest：`c3677499650bb58118c44f2d5af8e944130fd394d878e2ff602567c0d9647728`。
- 最终 run：[`20261005T111202319377Z-91158`](../results/tensor/local/20261005T111202319377Z-91158/manifest.json)，`status=passed`、`source_dirty=false`，source 前后完全相同。
- Host：本地 macOS / Darwin arm64；AppleClang 21，CMake 4.4.3，Python 3.14.6。精确 host/OS、版本、命令、编译选项与 sanitizer 环境见 manifest、configure/build 日志和字节保留的 CMakeCache/compile_commands。
- 原工作树已有未跟踪的 `AGENTS.md`，保持原状；正式复验使用仅含已提交源码的 detached worktree `build-stage1-clean-source`，没有通过隐藏未跟踪文件伪造 clean。
- 数据/本文为后续单独结果提交；tested source identity 不因结果提交而改写。
- [机器生成 audit](../results/tensor/local/20261005T111202319377Z-91158/audit.json) 验证源码前后一致、Stage 0 保护路径 diff 为空、文档本地链接有效；[production nm 原始输出](../results/tensor/local/20261005T111202319377Z-91158/production-nm.log) 与编译命令共同确认无 test-only hook symbols/defines。

复现精确测试源码：

```bash
git worktree add --detach build-stage1-repro f0aa8ac7fc86e3cd9bc485f9e637f90d1f118bde
(cd build-stage1-repro && python3 tools/validate_tensor.py)
```

## 最终验收结果

| 配置 | 结果 | 原始日志 |
|---|---|---|
| Fresh CPU Release | 5/5 CTest；构建无 warning/error | [build](../results/tensor/local/20261005T111202319377Z-91158/release-build.log)、[CTest](../results/tensor/local/20261005T111202319377Z-91158/release-ctest.log) |
| Fresh CPU Debug | 5/5 CTest；构建无 warning/error | [build](../results/tensor/local/20261005T111202319377Z-91158/debug-build.log)、[CTest](../results/tensor/local/20261005T111202319377Z-91158/debug-ctest.log) |
| Fresh Debug ASan+UBSan | 5/5 CTest；sanitizer target 3/3 tensor-labelled tests；无 sanitizer failure | [target](../results/tensor/local/20261005T111202319377Z-91158/sanitizer-target.log)、[CTest](../results/tensor/local/20261005T111202319377Z-91158/sanitizer-ctest.log) |
| Release `BUILD_TESTING=OFF` | Production library 构建通过，无 test-only API；无 warning/error | [build](../results/tensor/local/20261005T111202319377Z-91158/production-build.log)、[compile commands](../results/tensor/local/20261005T111202319377Z-91158/production-compile_commands.json) |

完整 CTest 是 `tensor`、`tensor_properties`、`tensor_validation`、原 Stage 0 `common` 和 `tools`。属性测试每个配置执行 600 个固定 seed=20903 (`0x51a7`) 案例，rank 0–8，交替 FP32/INT32；runner failure-path 单测 6 个，原工具单测 20 个。

**可验证的内存/copy事实（不是性能收益声明）：**

- reshape/view/narrow/slice/transpose：新增 CPU backing-buffer allocation=0，别名共享存储，源销毁后 view 继续有效；最后引用仅释放一次。
- 非空 strided `contiguous()`：明确新增 1 个 backing buffer，逻辑值一致且输出独立；already-contiguous 返回共享 alias，empty 不分配 buffer。
- 固定 3×2 FP32 copy：实际复制 24 logical bytes，新增 buffer=0；支持带 padding 的 strided destination，未写入孔洞。
- 单测/属性测试完成后 CPU live buffer 回到初始值；allocation failure injection 和自定义 deleter 生命周期通过。
- 所有计数由测试断言和原始输出产生，非手填性能 metrics；不把 metadata/container 堆分配计为 backing buffer。

macOS 不支持的 LeakSanitizer 被显式禁用 (`detect_leaks=0`)，**不声称 LSan 已通过**；ASan/UBSan 和所有权计数已运行。Linux runner 默认 `detect_leaks=1`，留给 Colab 实际复验。

## Change 对应与验收

| Change | 源码提交 | 接受依据 |
|---|---|---|
| S1-C1 metadata | `943616a` | scalar/empty/rank/negative/device/dtype/numel/byte/span 溢出分类；先确定单位与语义再进入 C2 |
| S1-C2 storage | `2a09229` | shared lifetime、alias、零容量、自定义 deleter、bad_alloc 注入、ASan/UBSan |
| S1-C3 construction/access | `bdd8626` | 1D–4D/INT32/scalar/empty/const access；offset/stride/capacity/alignment/type/device 错误拒绝 |
| S1-C4 views | `39c1694` | reshape/view/narrow/positive-step slice 共享存储，mutation/refcount/lifetime/zero buffer delta |
| S1-C5 transpose/copy | `1c491cc` | 2D/3D permutation、独立 materialization、strided copy、地址 span overlap 拒绝，CUDA host fallback 拒绝 |
| S1-C6 freeze | `f0aa8ac` | [公开契约](tensor.md)、600 property cases、[证据脚本](../tools/validate_tensor.py)、CPU sanitizer target、完整最终 clean-source 复验 |

`include/runtime/{dtype,device,shape,storage,tensor,copy}.hpp` 与 `src/runtime/{storage,tensor,copy}.cpp` 是唯一 Tensor/storage 路径。`copy_cpu` 为将来 S6 COPY/materialization 复用的 CPU leaf，不是另一个 dispatch。正 stride 的非重叠证明采用保守规则；重叠/零 stride/exotic 未能证明布局按文档拒绝。CUDA Device 元数据可在无 Toolkit 的 CPU build 中编译并测试，但不代表 CUDA allocation/integration 已验收。

## 开发日志与失败保留

逐 Change 开发验证在 [`results/tensor/local/`](../results/tensor/local/) 的 `c1-*` 到 `c5-*` 目录；这些记录明确为 dirty 开发源码，最终 clean-source run 才是完整验收依据。C3 曾有 range-loop-copy 编译 warning，已改为 const reference，并保留原 warning 与复验日志。

C6 的两个真实失败也未删除/修改：

1. [`20261005T104347214226Z-84114`](../results/tensor/local/20261005T104347214226Z-84114/manifest.json)：property test template 中 dependent `auto` 类型触发编译错误；改为显式 `Tensor` 类型后通过。
2. [`20261005T105341356551Z-87359`](../results/tensor/local/20261005T105341356551Z-87359/manifest.json)：runner 测试过宽地 mock 共享 `subprocess.run`，误影响 `platform.platform()`；改为 runner 私有 `run_process` 调用点后通过。

修复后的 dirty 开发成功 run 与最后正式 clean run 均保留。测试 mock 输出仅写临时目录，从未作为真实构建证据提交。CMakeCache、build 日志原样保存，包括生成器自带的 EOF 空行或命令尾空格；不为通过 Git whitespace 检查改写原始快照。

## 静态检查和边界

- 修改的 C++/Python/Markdown 19 个文件主动 LSP probe 无诊断。CMake LSP 不可用，不据此声称 CMake 静态检查通过；以四种真实 configure/build 结果验证。
- Colab 指南 7 个 bash block 已做 `bash -n`，仅语法检查，**没有实际执行 Colab**。
- 本地无 CUDA Toolkit，保留 Stage 0 CUDA 文件的缺失头/级联诊断按环境限制 defer；本阶段不修改这些源文件、不伪造 CUDA 通过。
- 不声称 CUDA-enabled 编译、T4 GPU correctness 或 Linux sanitizer 已在本次实现中实测。Stage 1 无强制服务器 gate；额外复验见 [全新 Colab / Drive 指南](stage1_colab.md)。
- 没有 CPU/GPU 吞吐/延迟 benchmark，没有 GEMM 优化，没有 CUDA allocation、Graph、Arena、Transformer、KV 或 INT8。
