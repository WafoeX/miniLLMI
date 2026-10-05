# Stage 4 — CPU arena 验收记录

**结论：S4-C1–C4 完成，host allocator invariant/integration gate 通过。** 用户要求先推送 Stage 3：`origin/feat/graph-stage3` 已确认是 `5d8f0efe097cfef10f9350a16abcde2d9dcf94cf`。随后在 `feat/arena-stage4` 实现本阶段；未进入 Stage 5+，未运行 GPU/服务器。基准明确 CPU-only，允许本地主机诊断；不是 CUDA/model/gallocr 性能声明。Stage 4 没有 latency improvement gate，较慢结果全部保留。

## Tested source / 正式证据

- Tested source：`d5d118d8027edacf0a6a3604c825efe212fc2d9c`。
- Source SHA-256：`793eeb9f8d13e0c20abe2334586911bdb765a4270a0bc27d496f6f9c022fdb21`。
- 正确性：[20261005T141011117404Z-37164](../results/allocator/local/20261005T141011117404Z-37164/manifest.json)，stage=4/status=passed/source_dirty=false，source 前后一致。
- CPU-only 基准：[20261005T141047459888Z-39363](../results/allocator/cpu/20261005T141047459888Z-39363/manifest.json)，status=passed/source_dirty=false，binary commit/digest/build/testing/backend 与源码匹配；记录 binary SHA、命令、host、scope、timing boundary 和 alignment 差异。
- 本机 macOS 26.7 arm64、AppleClang 21.0.0.21000334、CMake 4.4.3、Python 3.14.6；精确环境见原始 manifests/configure/build/cache。
- 两个正式 run 均在仅含已提交源码的 detached worktree `build-stage4-clean-source` 执行；原 `AGENTS.md` 未修改/提交。本报告与证据是独立结果提交，不改写 tested source identity。

```bash
git worktree add --detach build-stage4-repro d5d118d8027edacf0a6a3604c825efe212fc2d9c
(cd build-stage4-repro && python3 tools/validate_arena.py && python3 tools/run_allocator_benchmark.py)
# 对已保存数据重新分析（只能验证相同输出，不覆盖不同旧制品）：
python3 tools/analyze_allocator.py results/allocator/cpu/20261005T141047459888Z-39363
```

## 正确性与构建 gate

| 配置 | 实际结果 | 原始证据 |
|---|---|---|
| Fresh Release | **19/19 CTest** | [build](../results/allocator/local/20261005T141011117404Z-37164/release-build.log)、[CTest](../results/allocator/local/20261005T141011117404Z-37164/release-ctest.log) |
| Fresh Debug | **19/19 CTest** | [build](../results/allocator/local/20261005T141011117404Z-37164/debug-build.log)、[CTest](../results/allocator/local/20261005T141011117404Z-37164/debug-ctest.log) |
| Fresh ASan+UBSan | **19/19 CTest**，arena sanitizer target **4/4** | [target](../results/allocator/local/20261005T141011117404Z-37164/sanitizer-target.log)、[CTest](../results/allocator/local/20261005T141011117404Z-37164/sanitizer-ctest.log) |
| Production Release | BUILD_TESTING=OFF，test hooks/defines 缺席 | [build](../results/allocator/local/20261005T141011117404Z-37164/production-build.log)、[nm](../results/allocator/local/20261005T141011117404Z-37164/correctness-production-nm.log) |
| Benchmark fresh Release | 计时前重新 **19/19 CTest**，另建无 hook 的 production binary | [CTest](../results/allocator/cpu/20261005T141047459888Z-39363/ctest.log)、[probe](../results/allocator/cpu/20261005T141047459888Z-39363/binary-probe.log)、[correctness](../results/allocator/cpu/20261005T141047459888Z-39363/production-correctness.log) |
| Fixtures 再生成 | S2 committed bytes 一致 | [check](../results/allocator/local/20261005T141011117404Z-37164/fixture-regeneration.log) |

[生成 audit](../results/allocator/local/20261005T141011117404Z-37164/audit.json) 核验：12 份构建缓存/编译命令快照字节一致、6 次 build 无 warning/error、两个 production runtime libraries 无测试 hooks/defines、binary SHA 匹配、source 前后一致、180 samples 重新分析得到字节一致的 derived artifacts、保护路径无 diff。

19 CTest = 既有 15 + arena、allocation_provider、allocator_benchmark_workloads、allocator_tools。各完整配置覆盖：

- 对齐 backing/offset、size 0/capacity 0、invalid alignment、checked overflow、exhaustion、payload vs padding 计数、foreign/expired/default/stale/double free、rewind 前置条件、正确 aligned deletion。
- first-fit alignment prefix/suffix split、两侧 coalesce、总 free 足够但碎片不能 fit 的拒绝、复用地址不复活旧 ticket；保留 BumpNoReuse baseline。
- **6 seeds / 18,000** allocate/free 操作：独立 byte bitmap 检查完整 partition、无 free/live overlap、真实 live byte sentinel、精确 payload/live-block accounting；最终 coalesce 到单块，期间无新 backing allocation。
- 同一 ADD→RESHAPE→MATERIALIZE→RESHAPE→MATMUL 图的 dynamic/arena 数值一致、alias/base 最后使用、output pin、复制 alias 也阻止重入、释放所有 handles 后复用、provider 销毁后 Storage 仍随 output 活着；容量不足无 fallback、首错 cleanup、FP32→INT32 复用的 C++17 object lifetime、empty output。
- 实际 backing hooks 核对 one prepared arena / zero execute backing calls，span counters 不混同 backing；trace 记录真实 offset/capacity 与 BlockAllocate/BlockFree，旧 dynamic golden trace 仍通过。
- Benchmark untimed synthetic/chain/diamond 独立 oracle；allocator analyzer/runner **10** mock-only 临时测试：slower retained、idempotent generation、bad counts/numeric/checksums、缺 pair/sample、dirty/stale source、拒绝覆盖原制品、build failure 日志保留。
- 原有 600 Tensor properties、100 MATMUL cases、60×12 seeded DAG nodes、8 core numerical fixtures +9 Transformer Unsupported、9 fixture-tool tests、共用 correctness-runner **10** tests、Stage 0 工具 **20** tests 全部回归。

macOS ASan 使用 detect_leaks=0，因为 LSan 不可用；**不声称 Linux/LSan 或 CUDA-enabled/T4 已验证**。本阶段无需 GPU；后续 CUDA gate 未豁免。

## Change → source commit

| Change | 提交 | 接受依据 |
|---|---|---|
| S4-C1 | `2ae2e9e` | aligned Arena/Block API、exact accounting、zero/error policy，扩展同一 Storage ownership/hooks，原 allocate_cpu 不变 |
| S4-C2 | `f103a55` | deterministic first fit、split/coalesce、独立随机 partition/sentinel checker、bump baseline 保留 |
| S4-C3 | `c3bbfc8` | provider seam 复用唯一 executor/Tensor/Storage/CPU reference；物理 root 的 S3 last-use/alias lifetime、显式 capacity/busy failure、outputs 不被覆盖 |
| S4-C4 | `d5d118d` | CPU-only workload/benchmark、clean Release/provenance/negative-path safeguards、raw CSV + source-aware analyzer、公开契约 |

接口与生命周期细则：[arena.md](arena.md)。Arena 一块 Storage，views 不单独归还 spans；provider 在没有任何 output/copied Tensor Storage handle 后才允许下一轮退休 pinned spans。NewTensor 全经过 provider，默认仍为 S3 dynamic，显式 arena 不隐藏 malloc fallback。没有 graph liveness/slot plan/arena resize/inplace/CUDA/parallel path。

## CPU-only measured diagnostic

[raw/source-aware allocator.csv](../results/allocator/cpu/20261005T141047459888Z-39363/allocator.csv)、[analysis.json](../results/allocator/cpu/20261005T141047459888Z-39363/analysis.json)、[自动 summary](../results/allocator/cpu/20261005T141047459888Z-39363/summary.md)。3 independent paired runs；policy order D/A、A/D、D/A；每个 policy/workload 3 warmups、10 samples、每 sample 20 calls；共 18 raw CSV / **180 samples**，所有 commands exit=0。

基准版本、输入、容量、边界在计时前冻结于 source/docs/manifest：synthetic 8 次 fragment/reuse pattern；[8,8] chain 12 节点；diamond 5 节点含 identity MATMUL；arena capacity=1024。Synthetic 两侧 align64；Graph dynamic 保留 S3 malloc alignment，arena align64，明确是整个 allocation policy 比较，不是 alignment-isolated experiment。Tracer disabled；prepare/input construction/CSV I/O 在计时外；整个 execute/oracle/output destruction 在计时内。没有 exclusive CPU 保证，报告所有波动。

以下比值直接引用自动分析（四舍五入），每 workload 是三个 `dynamic median_ms / arena median_ms` 的中位数，不是两个「跨 run 中位数」的比值：

| Workload | 三个 paired ratios | Median paired ratio | 解读 |
|---|---|---:|---|
| synthetic | 1.4876 / 1.3445 / 1.6436 | 1.4876 | 此固定小微基准观察到 arena 较快，不外推模型 |
| chain | 1.1115 / 0.9649 / 1.0300 | 1.0300 | 有慢 run，表现混合；不声明可靠 graph speedup |
| diamond | 1.0886 / 0.9571 / 0.9655 | 0.9655 | arena 较慢，全部保留，不更换默认策略或调低 gate |

每 call 的 measured counters（execute snapshot，pinned output 的最终 free 在返回后，不计入 snapshot frees）：

| Workload | Dynamic backing alloc/free | Arena execute backing alloc/free | 两侧 span alloc/release | 两侧 peak payload bytes | Arena prepared backing/capacity |
|---|---:|---:|---:|---:|---:|
| synthetic | 48 / 48 | 0 / 0 | 48 / 48 | 640 | 1 / 1024 |
| chain | 12 / 11 | 0 / 0 | 12 / 11 | 512 | 1 / 1024 |
| diamond | 5 / 4 | 0 / 0 | 5 / 4 | 768 | 1 / 1024 |

Arena reservation **没有**优于 dynamic peak lower bound。Graph external backing 分别 512/768 bytes；peak tensor/arena backing 为 chain D=1024/A=1536、diamond D=1536/A=1792；synthetic D=640/A=1024。这不是 process RSS，排除 metadata/allocator implementation overhead，arena outputs 已包含在 capacity 中不重复相加。Online arena 不等于 S5 planner，也不满足/声称 S5 的 no-reuse-prepared capacity reduction gate；还有 execute-time C++ metadata allocation。

## Evidence discipline / checks / boundaries

- [各 Change dirty development evidence](../results/allocator/development/) 和 [完整 dirty correctness run](../results/allocator/local/20261005T140742411783Z-34796/manifest.json) 保留，不作为 clean timing evidence。
- C4 首次 allocator-tools test 失败（全局 patch subprocess.run 波及 platform.platform 内部 subprocess）。修复为 runner-local run_process callable，与既有 correctness runner 同一 isolation 模式；原 failed CTest/raw provenance 保留，后续所有真实 suites 通过。
- 最终主动 LSP probe 覆盖 27 路径：26 clean outcomes、CMake unavailable；另有 1 sibling analyzer import 误报经 disposition 隐藏，不把该位置声称为静态 clean。实际 CLI --help、单测、正式 runner 都成功 import，已标 false-positive、没有 inline ignore。76 个本地文档链接均存在；[final checks](../results/allocator/local/20261005T141011117404Z-37164/final-checks.json) 也确认 tested source 后无实现变更。临时 unused includes / Python formatter/JSON guard findings 均修复。保护的 CUDA 文件曾收到旧环境 diagnostics，fresh probe 没有当前 findings；没有据此声称 CUDA 编译/运行通过。CMake 以真实 builds 为准。
- Raw logs/CMakeCache 原样保留，包括 EOF 空行/尾空格；自动 summary 的表格格式 advisory 不改变其计算值，不手改 derived metrics。
- Stage 0 naive/cuBLAS/common/bench/scripts/analyzer、原 Tensor/copy/operator/inference/reference/graph topology 以及所有既有 fixtures/results 保持不变。Storage 仅新增 aligned API；executor/trace 仅为本任务注入 provider 扩展，旧默认语义有真实回归。
- `feat/arena-stage4` 源码与结果分开提交，本阶段未 push/merge main；此前 Stage 3 推送已完成。Stage 4 到此停止，后续阶段另行授权。
