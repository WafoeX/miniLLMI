# Stage 1–19 任务书审查与修订

本次只修订规划，不实现 Stage 1+，不声称已经获得性能提升。目标是**正确、可复现、相对明确 baseline 有足够提升即可**，不是最优内核或生产级 LLM。

权威要求见 [roadmap](roadmap.md) 和 [任务书索引](tasks/README.md)。Stage 0 已接受的源码、原始数据和报告保持不变。

## 发现的不足与处理

| 不足 | 修订 | 位置 |
|---|---|---|
| Must/Should/Optional 与实际依赖不一致：后续报告强制依赖 inplace、work stealing 等 | 明确 optional 跳过状态；报告只依赖必需 Change，跳过不阻塞后续 | roadmap §2/§8、S5/S7/S9/S11 |
| 只保存性能结果，未定义最终“有效果提升”；也没有停止条件 | 定义适度收益门槛、3 次独立配对实验、回退/噪声口径；达标即停止，未达标只允许有界追加实验 | roadmap §5 |
| CUDA V2 要求 64/128 block tile，却尚无 thread tile，一输出一线程会超过线程上限 | 无 thread tile 时只扫合法小 tile；大输出 tile 留给可选 register/thread tile；校验实际设备资源 | S9-C3/C4 |
| INT8 benchmark 依赖 CLI，路线图又让 CLI 等待整个 INT8 Stage，形成循环 | S16-C3 → CLI INT8 模式 → S16-C4；float CLI 只依赖 cache/loader | roadmap §2、S16/S17 |
| Tensor/算子尚不足以实现既定 Transformer：token dtype、分头、转置、cache 写入和 decode mask 未定义 | 增加 INT32 IDs、checked slice/view、显式 materialization、逐头 2D MATMUL、带绝对位置的 mask；外部状态写入产生逻辑版本并保留 DAG 顺序 | S1–S3、S12/S14 |
| 未固定 tiny model，CUDA primitive “稍后实现”也无明确最低覆盖 | 固定小型 MHA 配置、byte tokenizer 和 fixture 身份；CPU 全功能，T4 至少 CUDA projection + 显式跨设备 COPY；全 CUDA primitive 可选 | roadmap §8、S12/S13/S15 |
| “零分配”混淆 backing buffer、metadata、workspace；arena、输出、cache 生命周期不清 | 限定 warmed execute 的中间 buffer 分配；prepare、输出、cache、cuBLAS/反量化 workspace 分别计数；显式 replan | tasks/README、S3/S5/S8/S14/S16 |
| allocator baseline 可能将所有中间值保留至结束，夸大复用收益 | dynamic baseline 按 last use 释放；另外设合法的 prepare-only no-reuse 对照衡量预留容量，绝不宣称 arena 容量低于同时 live 的字节下界 | S3/S5 |
| placement 插入 COPY 后仍可能沿用旧内存计划；CUDA async 生命周期不清 | COPY rewrite 后重验图并重新规划；单 stream 起步，明确 storage 保活、teardown 等待和 pageable copy 的限制 | S8/S11 |
| CPU scalar 4096 固定 benchmark 成本高，算法/线程收益混淆 | 必测 128/256/512/1024，2048/4096 CPU 可选；区分 FP32 性能 baseline 与 Stage 0 FP64 oracle，分别记录算法和 pool scaling | S6/S7 |
| KV off/on 输入轨迹、计时范围和设备可能不同，无法归因 | 使用相同 32 个 continuation IDs、相同 model/kernel/placement；decode 包含必要 reprepare/copy/sync，prefill 独立 | S14/S17 |
| INT8 未冻结 axis/error，解压 workspace 可能抵消内存/速度收益 | 固定 W[in,out] 的输出列量化、round/zero policy 和误差阈值；允许 prepare dequant 并完整披露 resident workspace；压缩达标不要求吞吐提升 | S16 |
| 要求“同一 commit 拥有最终结果”与源码/结果分开提交矛盾；RC tag 可被反复覆盖 | 不可变 RC 源码 tag + 后续证据提交/manifest；所有 required miss 阻塞最终完成，optional 明确跳过 | roadmap §9、S18/S19 |
| 强制大量面试页、逐级内核卡片，文档/调优成本偏离目标 | 必需一组 v0/选中内核 profile 和简短证据索引；额外页面/变体可选 | S10/S19 |

## “足够提升”而非“极致”的验收

- CPU/custom CUDA GEMM：规定尺寸总体加速至少 **1.05×**，至少 3 次独立配对运行方向一致；cuBLAS 仅作参考，不要求追平。
- Planner：execute 中间 buffer 分配 **>0 → 0**；reuse-rich 固定图的预留 arena 容量较 prepare-only no-reuse 对照至少降低 **20%**。动态 last-use baseline 的峰值仍单独报告，不要求 latency 加速。
- KV：预先声明的 CPU context=512、32-token decode 至少 **1.05×**，所有独立配对 run 均更快；其他 contexts 和 T4 mixed path 不隐去。
- INT8：可量化 projection 的 payload **含 scales ≤ FP32 的 35%**，通过固定 tensor/logit 误差测试；总文件、workspace、resident memory、吞吐均披露，不要求 INT8 更快。
- work stealing、inplace、vectorized/double-buffer GEMM、BPE、fused quant、全 CUDA Transformer 不作为完成条件。

以上数字是未来实现的**工程验收门槛**，不是已测结果，也不构成显著性/泛化承诺。若噪声掩盖小幅增益，应标为 inconclusive；持续未达标需要显式调整范围，不能静默降门槛或无限调参。

## 本次交付边界

仅纳入 roadmap、19 个 Stage 任务书、索引及本审查记录。未改实现/benchmark/analyzer/Stage 0 结果；本次验证的是文档链接、Change ID/依赖一致性、Markdown 和提交差异，不是运行时 correctness 或性能。

本次静态验证：22 个 Markdown 文件、24 个本地链接、85 个唯一且连续的 Change ID；按各 Change 的 Depends 字段展开 367 条依赖边，无未知 ID、循环或必做 Change 直接依赖可选 Change。Stage header 与条件性 INT8 CLI 依赖另做人工复核。22 文件主动 Markdown LSP 检查无诊断。未运行 C++/CUDA 测试或 benchmark，因为此次没有实现变更。
