# Stage 0 Baseline Profiling

当前没有 T4 实测 profile。以下是采集与分析规范，不是已观测结论；不要将预期瓶颈写成真实测量。

## 先 Benchmark，再整体 nsys，最后单 kernel ncu

```bash
./scripts/run_gemm_benchmark.sh
./scripts/profile_gemm.sh nsys naive
./scripts/profile_gemm.sh ncu naive
```

脚本固定 M=N=K=4096，warmup=10、iterations=50，要求 clean source 与 Release，记录 command、commit/digest、环境、工具版本，输出到唯一 `results/profiling/<nsys|ncu>/<id>/`。计时 CSV 标记 `profiling`，不送进正式 baseline 报告。

`nsys` 采集 CUDA/cuBLAS/osrt，查看 CPU 提交、CUDA API、Event/同步、内存拷贝和 kernel 时间线。完整 CPU oracle 也出现在进程时间中，不能把它的耗时误认为 CUDA GEMM 延迟。

`ncu --set full` 采集实际可用的 section/指标。Naive 用 kernel-name regex 过滤；初始 correctness 一次 + warmup 十次，所以 `--launch-skip 11 --launch-count 1` 采集首个被计时 launch。cuBLAS 的内部 kernel 名及 launch 数可能随版本改变；其一 API 不一定只有一个 kernel，不能盲目把同一个 skip 值当作“首个 measured GEMM”。首次做 cuBLAS Profiling 时先在 nsys 确认 launch，再按实际 kernel/filter 调整并保存修改命令。

工具版本、驱动权限和实际 CC 决定指标可用性；如果 ncu 报性能计数器权限错误，保存错误日志，请服务器管理员解决权限。不能把权限失败写成成功采集。

## 导出文本与归档

以下 `<实际目录>`/变量需替换为本次真实路径：

```bash
NSYS_DIR=results/profiling/nsys/<实际目录>
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$NSYS_DIR/trace.nsys-rep" \
  > "$NSYS_DIR/stats.txt"
sha256sum "$NSYS_DIR/trace.nsys-rep" > "$NSYS_DIR/trace.nsys-rep.sha256"

NCU_DIR=results/profiling/ncu/<实际目录>
ncu --import "$NCU_DIR/metrics.ncu-rep" --page details --csv > "$NCU_DIR/metrics.csv"
ncu --query-metrics > "$NCU_DIR/available_metrics.txt"
sha256sum "$NCU_DIR/metrics.ncu-rep" > "$NCU_DIR/metrics.ncu-rep.sha256"
```

nsys stats report 名称如因版本变化报错，查询当前 `nsys stats --help-reports`，记录真实可用名称。Nsight 二进制 `.nsys-rep`/`.ncu-rep` 被 Git 排除；保留到外部制品存储后，在同目录 `artifact_location.txt` 写获取位置和校验和，提交文本导出与日志。禁止采完只保存一张截图然后删除原始 profile。

## 要观察而非预设的指标

| 方面 | 记录内容 | 能回答什么 |
|---|---|---|
| SM / memory throughput | 当前工具实际提供的吞吐率及单位 | 是否受计算/带宽约束 |
| Occupancy | 理论/实际 active warps、限制因子 | blocks、register、shared memory 等限制 |
| Warp stalls | stall reason、issue active、eligible warps | 等待访存/依赖/指令调度情况 |
| Global loads | sectors/requests、load transactions、cache hit | 合并访问、重用和缓存行为 |
| Shared memory | static/dynamic bytes、bank conflicts（若适用） | Stage 0 Naive 没有用户 shared tile；后续才做对比 |
| Registers | registers/thread 和寄存器限制 | 资源压力与 occupancy 的关系 |

T4 的吞吐率、SM 数、时钟和硬件限制以 `gpu_info` 和 profile 实际数据为准。不要用 4060 的峰值或 SM 数计算这里的利用率。“Global Load Efficiency” 不一定是当前 Nsight 的同名字段，应保留实际指标名、分母和单位，而不是臆造一列。

## 本阶段报告模板

```text
代码 commit / source digest：
run_id：
GPU UUID / CC / CUDA / Nsight 版本：
正式 Benchmark median / GFLOPS / cuBLAS Ratio（引用 CSV 行）：
nsys artifact + SHA256：
ncu artifact + SHA256：
实际指标名、数值、单位：
观测到的瓶颈：
证据与解释：
其他解释/实验局限：
后续优化假设（仅记录，不在 Stage 0 实现）：
```

后续阶段才执行“baseline→profile→提出优化→独立 perf commit→同条件 benchmark→再次 profile”的因果闭环。当前只保留 baseline 证据，不实现 shared/thread tile 等优化，也不预先声称它们一定更快。
