# Stage 2 — Operator definitions 验收记录

**结论：S2-C1–C5 完成，CPU-only stage gate 通过。** 未进入 Stage 3 Graph；未实现 CUDA dispatch 或 Transformer execution kernels。遵照用户决定，不进行 Stage 1 服务器复验；Stage 2 亦无需 GPU/Colab gate，**本次未运行服务器验收**。

## Tested source 与正式证据

- Tested source commit：`48e18d2fd06303a59330866b69a91b215e6fcd5d`。
- Source SHA-256 digest：`7719aa199e5c48185094161234a2c0b90aaeac407357e935d192d71ec94a4cc6`。
- 最终 run：[`20261005T121705133744Z-8132`](../results/operators/local/20261005T121705133744Z-8132/manifest.json)，`stage=2`、`status=passed`、`source_dirty=false`，source 前后一致；每个命令 exit=0。
- 本地 macOS 26.7 arm64，AppleClang 21.0.0.21000334，CMake 4.4.3，Python 3.14.6；精确 host/环境/命令/编译配置见 manifest、configure/build 原始日志和缓存快照。
- 原工作树未跟踪的 `AGENTS.md` 保持未修改；正式测试使用仅含已提交源码的 detached worktree `build-stage2-clean-source`。
- 本报告与 raw evidence 是后续独立结果提交，不改写 tested source identity。

```bash
# 精确复现本次被测源码（新目录，避免旧缓存/未跟踪文件）：
git worktree add --detach build-stage2-repro 48e18d2fd06303a59330866b69a91b215e6fcd5d
(cd build-stage2-repro && python3 tools/validate_operators.py)
# 正常使用当前已提交/clean checkout：
# python3 tools/validate_operators.py
```

## 最终验收

| 配置 | 实际结果 | 原始证据 |
|---|---|---|
| Fresh CPU Release | **11/11 CTest**，build 无 warning/error | [build](../results/operators/local/20261005T121705133744Z-8132/release-build.log)、[CTest](../results/operators/local/20261005T121705133744Z-8132/release-ctest.log) |
| Fresh CPU Debug | **11/11 CTest**，build 无 warning/error | [build](../results/operators/local/20261005T121705133744Z-8132/debug-build.log)、[CTest](../results/operators/local/20261005T121705133744Z-8132/debug-ctest.log) |
| Fresh Debug ASan+UBSan | **11/11 CTest**，operator sanitizer target **6/6**，无 sanitizer failure | [target](../results/operators/local/20261005T121705133744Z-8132/sanitizer-target.log)、[CTest](../results/operators/local/20261005T121705133744Z-8132/sanitizer-ctest.log) |
| Release `BUILD_TESTING=OFF` | production library build 通过，无 test-only hooks/defines | [build](../results/operators/local/20261005T121705133744Z-8132/production-build.log)、[nm](../results/operators/local/20261005T121705133744Z-8132/production-nm.log) |
| Offline fixtures regeneration | committed artifacts 与重新生成 bytes 完全一致 | [check](../results/operators/local/20261005T121705133744Z-8132/fixture-regeneration.log) |

[机器生成 audit](../results/operators/local/20261005T121705133744Z-8132/audit.json) 核验 source 前后相同、8 个 build snapshot 字节一致、production test hook 缺席及 Stage 0/Stage 1 保护路径 diff 为空。CMakeCache 和生成日志保持原样，包括 EOF 空行或命令尾空格，不为 whitespace 检查修改 raw snapshots。

11 个 CTest 为 Stage 1 的 `tensor`、`tensor_properties`、`tensor_validation`，原 Stage 0 `common`、`tools`，以及 `operator`、`shape_inference`、`operator_reference`、`transformer_contracts`、`operator_fixtures`、`operator_fixture_tools`。

- 每种完整测试配置重跑 Stage 1 **600** seeded property cases，rank 0–8、FP32/INT32。
- CPU reference：hand vectors、scalar/empty/K=0/rectangular cases，加 **100** fixed-seed (`0x5203`) MATMUL，与独立 long-double indexing 比较。
- Conformance：**17** 个案例；**8** 个 core 数值执行通过，**9** 个 Transformer metadata/shape 验证通过，日志明确 `numeric=pending-S12`。**没有将这些 9 个案例声称为 Transformer kernel 通过**。
- C++ fixture reader 的 **10** 个 malformed cases 均拒绝；Python fixture工具 **9** 个单测（hash/版本/范围/路径/再生成/手算锚点等）通过。
- 共用 evidence runner **8** 个单测（原 6 + Stage 2 target/family 与 unsupported stage），原 Stage 0 工具 **20** 单测通过。mock/损坏测试 artifacts 仅在临时目录，不冒充真实构建结果。

macOS `detect_leaks=0` 是因为 LeakSanitizer 不可用，**不声称 LSan/Linux sanitizer 已通过**。实际执行 ASan/UBSan、ownership/buffer counters；Linux runner 默认启用 LSan，但本次没有服务器/Linux 运行。

## Change → 提交 → 验收

| Change | 源码提交 | 接受依据 |
|---|---|---|
| S2-C1 schemas | `d5789f0` | immutable-to-callers OpDesc、typed attrs、IDs/arity、structured Status、版本化 deterministic JSON；无 execute/backend pointer |
| S2-C2 inference | `a98e11d` | ADD/MUL/MATMUL shape/dtype/layout/overflow；COPY explicit destination、views/aliases/materialization；零 backing allocation 的纯 metadata 验证 |
| S2-C3 core reference | `5728e3a` | single-thread ADD/MUL、FP64 correctness MATMUL、复用 Stage 1 COPY；caller-provided outputs、alias/nonfinite/error policy，无隐藏 buffer allocation |
| S2-C4 Transformer contracts | `6a6f279` | RMSNorm/Softmax/RoPE/Embedding/SwiGLU/Attention schema/inference；位置/head geometry/epsilon/causal/cache-write/2D head-lowering 契约；tiny model/W[in,out]/quant axis 冻结 |
| S2-C5 fixtures | `48e18d2` | offline generator/version/SHA、bounded reader/harness、17 fixtures、实际 tiny weight payload、failure-path tests、公开文档、shared evidence runner、最终 clean-source 复验 |

完整规范见 [docs/operators.md](operators.md)。实施路径为同一 `runtime::Tensor` + shared `Storage` 上的 `operator.hpp/.cpp`、`shape_inference.hpp/.cpp`、`reference.hpp/.cpp`。`OutputContract` 是计划输出要求，不是另一条 Tensor/runtime 路径；NewTensor 输出由调用方/未来 planner 分配，Alias/Write 持有既有 Stage 1 Tensor。已有 `copy_cpu` 的 span/self-copy 判定仅抽取为共用 metadata helpers，原策略未改变；Stage 1 storage/tensor/dtype/device/shape 代码与历史证据未修改。

Reference MATMUL 的 FP64 scalar accumulate 仅作 correctness oracle，不替代 S6 将建立的 FP32 `ijk` 性能 baseline。算术输入须显式 contiguous；COPY 可处理 checked strided range，跨设备 COPY 只能在 metadata 层声明，CPU reference 明确拒绝执行。Transformer descriptors 当前 `reference::execute` 返回 Unsupported，未引入提前 kernel/model implementation。

## Fixture provenance 和范围

[`tests/fixtures/operators-v1/manifest.json`](../tests/fixtures/operators-v1/manifest.json) 记录生成脚本版本及 SHA-256、文件 bytes/hash、case shape/dtype/attributes/layout/tolerances。weight pack 是 **21 tensors / 461056 bytes FP32**，逐 tensor 记录范围与 SHA，projection layout **W[in,out]，output-channel axis=1**；norm scales=1，seed `0x5205` 的 xorshift32/dyadic generator。真实文件已提交，不要求 C++ 重现 Python RNG。

全部生成是 offline stdlib Python；C++ runtime 不 import Python/PyTorch，不借 fixture generator 执行 kernel。这只是 fixture raw pack，不是模型 loader、S15 模型文件或语言质量证据。Transformer expected tensors 为未来 S12 固定 oracle；FP32 误差阈值已在契约中预先冻结，INT32 exact。当前没有模型/logit运行或优化数据。

## 记录保留、静态检查与边界

- S2-C1–C4 逐项 Release/sanitizer 开发验证位于 [`results/operators/local/`](../results/operators/local/) 的 `c1-*` 到 `c4-*`；明确 dirty 开发源码。C5 dirty run、preflight 与 final clean run 均保存，不冒充 clean 正式验收。本阶段各实际 build/test run 均通过；malformed/corruption/failure tests 是预期的负例，原输出保留在 test logs。
- 修改的 C++/Python/Markdown **26** 文件主动 LSP probe 无诊断。CMake LSP unavailable，不声称其静态检查通过；以四种真实 configure/build 验证。
- Stage 0 naive/cuBLAS/reference、历史 GEMM/profiler 数据保持不变；Stage 1 全套回归通过。未运行任何 benchmark/profiling，没有吞吐/延迟改善声明。
- 无 GPU/服务器验收，也不需要为 Stage 2 重跑旧 T4 baseline 或新 Colab GPU 命令。未来 S8 开始涉及 CUDA storage/copy/backend 时，才提供对应全新 Colab 命令并要求真实 T4 数据。
- 未进入 Stage 3：没有图拓扑、executor、arena/planner、backend registry/scheduler、Transformer kernels、KV policy、tokenizer/loader、量化或模型执行。后续只可按 [roadmap](roadmap.md) 的依赖继续。
