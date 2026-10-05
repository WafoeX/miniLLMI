# Stage 3 — Graph / sequential CPU executor 验收记录

**结论：S3-C1–C4 完成，CPU-only stage gate 通过，M1 runtime foundation 完成。** 无需 GPU，本次未运行服务器/GPU；未进入 Stage 4+，没有 arena、planner、scheduler、Transformer kernel 或模型执行。没有性能收益声明。

## Tested source 与证据

- Tested source：`533ba1e12133d7c29a2b28544f01b108d5c26f01`。
- Source SHA-256：`4cde07bc83c83f46abc05cb4b3069d245bf80013b036e59b6269b62efcf5d559`。
- 正式 run：[20261005T131248311191Z-22617](../results/graph/local/20261005T131248311191Z-22617/manifest.json)，`stage=3`、`status=passed`、`source_dirty=false`、source 前后一致、全部命令 exit=0。
- 本地 macOS 26.7 arm64、AppleClang 21.0.0.21000334、CMake 4.4.3、Python 3.14.6；精确配置与命令见 manifest、原始 configure/build 日志及快照。
- 正式验收使用只含已提交源码的 detached worktree `build-stage3-clean-source`；原工作树未跟踪 `AGENTS.md` 未修改、未提交。
- 本报告与原始证据为后续独立结果提交，不改写 tested source identity。

```bash
# 精确复现被测源码，避免未跟踪文件/旧缓存：
git worktree add --detach build-stage3-repro 533ba1e12133d7c29a2b28544f01b108d5c26f01
(cd build-stage3-repro && python3 tools/validate_graph.py)
# 正常 clean checkout：python3 tools/validate_graph.py
```

## 正式验收

| 配置 | 实际结果 | 原始日志 |
|---|---|---|
| Fresh Release CPU | **15/15 CTest**，build 无 warning/error | [build](../results/graph/local/20261005T131248311191Z-22617/release-build.log)、[CTest](../results/graph/local/20261005T131248311191Z-22617/release-ctest.log) |
| Fresh Debug CPU | **15/15 CTest**，build 无 warning/error | [build](../results/graph/local/20261005T131248311191Z-22617/debug-build.log)、[CTest](../results/graph/local/20261005T131248311191Z-22617/debug-ctest.log) |
| Fresh Debug ASan+UBSan | **15/15 CTest**，graph sanitizer target **4/4** | [target](../results/graph/local/20261005T131248311191Z-22617/sanitizer-target.log)、[CTest](../results/graph/local/20261005T131248311191Z-22617/sanitizer-ctest.log) |
| Release `BUILD_TESTING=OFF` | production build 通过，test hooks/define 缺席 | [build](../results/graph/local/20261005T131248311191Z-22617/production-build.log)、[nm](../results/graph/local/20261005T131248311191Z-22617/production-nm.log) |
| S2 offline fixture 再生成 | committed bytes 完全一致 | [check](../results/graph/local/20261005T131248311191Z-22617/fixture-regeneration.log) |

[机器生成 audit](../results/graph/local/20261005T131248311191Z-22617/audit.json) 核验源码一致、8 份缓存/编译命令快照字节一致、production hooks/define 缺席、build warning/error 为空及保护路径无 diff。保留原始 CMakeCache/日志的 EOF 空行或命令尾空格，不为 whitespace 检查篡改 raw artifacts。

15 个 CTest = 既有 S1/S2/Stage 0 的 11 个 + `graph`、`graph_executor`、`graph_fixtures`、`graph_trace`。每种完整配置覆盖：

- Graph ownership：重复/保留 ID、unknown tensor、producer 唯一性、named boundaries、插入失败事务性、重复 operand 的唯一 consumer edge。
- Freeze：diamond、不同插入顺序得到相同 min-ready-ID order、disconnected/unused roots、cycle、missing producer、invalid output shape/dtype/device、moved-from schema、冻结后所有 builder mutation 拒绝；freeze 零 backing allocation。
- State：COPY 显式 persistent binding/new version；alias/base/range、ordered/disjoint writes、unordered overlap、stale reader/output、prior ordered read、同 Storage state 声明一致、不同 wrapper 的真实物理 overlap 拒绝。
- Executor：非方阵 ADD→MATMUL 与直接 S2 reference 相同；alias chain/base 延寿、last-use frees、输出 pin、多名字同 alias、未使用输出即时释放、重复执行、empty/scalar/pass-through、explicit materialization/state COPY、exact-self zero movement。
- Error：未冻结图拒绝；NonFinite/Unsupported 首错停止、清空所有 outputs、释放 dynamic buffers；注入 CPU allocation failure → ResourceExhausted；simulated CUDA metadata 拒绝且不读取设备数据。先完成的外部 state write 在后续错误时不回滚，此边界有测试。
- **60** 个 fixed-seed (`0x5303`) DAG，每图 **12** 个 ADD/MUL/MATERIALIZE 节点，逆序插入、与独立标量数组预期逐元素一致；真实 Storage hooks 核对 allocation/free/output 生命周期。
- S2 fixture checksum/provenance 验证后，**8** 个 core 数值案例通过冻结图执行；**9** 个 Transformer 案例的 inference 通过，execution 明确 **Unsupported**，不声称 Transformer kernels 数值通过。
- Trace：完整 diamond golden event sequence/metadata；trace 与 backing counters 一致；first-N bound、dropped/reset、limit=0/nullptr、alias delayed base free、explicit copy/state self-copy、首错清理。Trace 不持有 Tensor/Storage。
- 全套回归：S1 **600** seeded tensor properties、S2 **100** MATMUL cases、**10** malformed fixture reader cases、**9** Python fixture-tool tests、共用 evidence-runner **9** 单测、Stage 0 工具 **20** 单测。

macOS 使用 `detect_leaks=0`，因为 LeakSanitizer 不可用；**不声称 LSan/Linux 已验证**。本次确实执行 ASan/UBSan、ownership/buffer counters。CUDA-enabled build 和 T4 integration 未实测，后续 CUDA gate 不豁免。

## Change → 提交 → 接受依据

| Change | 源码提交 | 接受依据 |
|---|---|---|
| S3-C1 ownership | `e7ac561` | graph-owned value records/IDs/links/names、无 dangling node pointers；中间声明无 backing allocation |
| S3-C2 validation | `5098c45` | 单一 S2 pure inference、transactional freeze、确定性拓扑、cycle/output/state-version diagnostics；无调度/执行 |
| S3-C3 executor | `c5ecaeb` | 复用唯一 Tensor/Storage/reference；per-node allocation、last-use/alias/output lifetimes、首错停止与输出清理、fixture agreement |
| S3-C4 trace | `533ba1e` | bounded diagnostics/common event vocabulary、golden trace/disabled equivalence、60 seeded DAGs、共享 evidence runner 与公开契约 |

完整 API/语义见 [graph.md](graph.md)。为无 backing 的 graph freeze，提取 S1 原有 checked layout/view 逻辑到 `Layout` 元数据记录，Tensor 调用同一实现；S2 `infer_operator` 与 `infer_layout` 复用唯一推断语义。`Layout` 无指针、数据访问或 allocator，不是第二条执行路径。Tensor 的公开表示/API、Storage/copy 实现不变；`tensor.cpp` 的布局委托与 S2 inference adapter 属于显式必要重构，不能声称这些文件 byte-identical。全部旧回归通过，S2 对不同 wrapped Storage 的物理 overlap 检查保留。

## 动态 baseline 实测计数（非性能数据）

Verbose test logs 中实际输出/断言：

| 固定 workload | backing alloc | execute 内 free | peak dynamic-owned bytes | 返回时 live bytes | copy events/bytes |
|---|---:|---:|---:|---:|---:|
| ADD[2,3] → MATMUL[3,4] | 2 | 1 | 56 | 32 | 0 / 0 |
| diamond（每 Tensor 2 FP32） | 4 | 3 | 24 | 8 | 0 / 0 |

计数来自实际 executor 并与 Storage hooks 核对，不是 latency/throughput 指标。包括新产生的 graph-output buffers；排除外部 inputs/weights/state 和 C++ metadata/container allocations，**不是总 resident memory**。输出 destructor 在 execute 之后，其 frees 不计入 execute 内 free。所有新输出分配在 execute 内；未来 S5 必须沿用相同 timing boundary 与 lifetime/output semantics。没有跑 benchmark/profiling，也没有 arena/reuse/performance claim。

## 失败保留、证据纪律与边界

- [development evidence](../results/graph/development/) 保留 C1–C4 逐项开发日志和 source-before/source-after；[完整 dirty run](../results/graph/local/20261005T131114546233Z-20276/manifest.json) 明确 `source_dirty=true`，不冒充正式验收。
- 初次 C1 build 的 explicit Device default-parameter 编译错误已修复，原失败 log 保留；当时没有 runner 级即时 source capture，manifest 透明记录后续重建 provenance，**不作为正式接受证据**。
- 正式验收后的 ad-hoc audit 初次 regex 误用旧 CTest summary 格式，断言失败也保留为 [audit-preflight-failure.log](../results/graph/local/20261005T131248311191Z-22617/audit-preflight-failure.log)。真实 build/tests 都通过；修正 audit 支持实际 CMake 4.4.3 summary，不修改 raw logs。
- 最终主动 LSP probe 覆盖 28 路径：26 clean、报告中的 Git hash 片段被 typos 拼写器误报 1 information（已标记 false-positive）、CMake LSP unavailable；无未解决真实诊断。63 个本地文档链接均存在。早期测试 namespace 误报由切换 S3 compile database 解决，trace-header 引入的临时 undefined-type 由头文件声明同步解决。CMake 以四种真实 configure/build 结果为验收依据。
- Stage 0 naive/cuBLAS/common/bench/scripts/analyzer、既有 fixtures 和所有历史 Stage 0–2 结果保持不变。Stage 3 结果独立目录，不覆盖旧证据。
- 分支 `feat/graph-stage3`；源码/结果分开提交。未 push/merge main。Stage 3 无需 GPU；S8+ CUDA storage/copy/dispatch/integration 仍须真实 T4 验证并通知用户。
