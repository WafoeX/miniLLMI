# Stage 0 核心设计

本阶段只构建 Baseline，不建立 Tensor、Operator/Graph 或 Transformer runtime。

## 职责边界

- `stage0_common`：标准 C++17 CPU oracle、确定性输入、误差检查、统计与 CSV。无 CUDA 依赖。
- `stage0_cuda`：Naive kernel 的 `.cu` 文件与 cuBLAS row-major wrapper，链接 CUDA runtime/cuBLAS。
- `bench_gemm.cpp`：host-only C++17 编排；CUDA 调用只通过明确 wrapper 和 runtime API。
- `gpu_info.cpp`：实际设备查询，不写设备规格常量。
- Python：只负责源码 provenance 和实测 CSV 的核对/报告，不代替 CUDA Timer 或生成性能值。
- Bash：干净源码→Release Build→环境快照→CTest→Benchmark→原始数据→报告。

关闭 `ENABLE_CUDA` 后仅构建 common 与 CPU/工具测试；没有占位 CUDA 模拟实现。不存在用 CPU 调用假装 GPU 测试成功的路径。

## 数值与 layout

所有矩阵是连续 row-major FP32；C=A×B，alpha=1，beta=0。CPU reference 使用单线程 ijk，FP64 累加后转 FP32，避免仅用另一个 FP32 reduction 作为唯一数值 oracle。它不参与性能计时，也不声称是 CPU 性能 baseline（CPU 优化将在后续阶段单独建立）。

cuBLAS 使用 column-major API，因此传 `(N,M,K,B,ld=N,A,ld=K,C,ld=N)`，对应 Cᵀ=BᵀAᵀ，不额外 transpose。非方阵测试是防止这个映射被方阵掩盖的关键。

Naive 默认 launch block 16×16（算法配置，不是硬编码 GPU 属性），每 thread 一个 output，FP32 累加；允许编译器正常 FMA 和标准 Release 优化，不故意关闭编译优化或使用异常访存来制造低 baseline。

cuBLAS 使用 `CUBLAS_PEDANTIC_MATH` 和 `CUBLAS_ATOMICS_NOT_ALLOWED`，明确控制 FP32 数学口径。与默认/TF32/Tensor Core 模式不是同一实验组；T4 的 actual runtime/CC 仍必须查询后保存。该参考并非所有数学模式中绝对最快的 cuBLAS。

## CUDA 资源与 Timer

DeviceBuffer、Stream、BlasHandle、EventTimer 都不允许复制，用 RAII 回收资源；初始化失败会释放已经创建的资源。析构不抛异常，所有正常执行路径的 CUDA/cuBLAS API 和 launch 状态明确检查；异步执行错误由 stream/event 同步检查暴露。若 CUDA context 已因 fatal error 失效，析构仅做 best-effort 清理，进程非零退出。

两个 kernel 和 cuBLAS handle 使用同一非默认 nonblocking stream。H2D/D2H 都在该 stream 排队并同步；不依赖隐式 default-stream 顺序。

每轮 Event：record start→launch→record stop→synchronize stop→elapsed time。区间包含设备 stream 上这次调用的执行/可能的提交空隙，不能声称完全消除了所有 host dispatch 影响。分配、拷贝、reference、CSV 写入不在区间内。无 fast-math 编译参数。

## 正确性优先于性能

每个 shape：生成同一 A/B→完整 CPU reference→分别检查 Naive/cuBLAS→各自 warmup→各自 50 轮计时→检查最终输出→汇总统计。初次验证 C 预填 NaN，避免未写输出或 beta=0 行为错误被零初始化掩盖。独立 CUDA 测试还保护输出 buffer 的前后 guard。

逐元素失败规则：非有限值，或 `abs(actual-ref) > atol + rtol*abs(ref)`。默认 atol=rtol=1e-3，CLI 可显式设置，两者必须相同，并随数据保存。相对误差在参考值接近零时可能很大；它单独报告，不作为错误的唯一判据。误差超限不能自动放宽阈值重跑后只保留好结果。

若 initial correctness 失败，不计时。若 final correctness 失败，已经保存的 raw timing 仍保留但不发布成功 GFLOPS；保存失败状态并退出。计时异常、API 错误、I/O 错误同样失败。

## 版本与可复现性

`tools/provenance.py` 读取 Git HEAD、源码 SHA256、dirty 状态，生成构建时 header；每次 build 刷新，避免仅在 configure 时记录旧 commit。digest 包括代码、测试、构建脚本、静态文档和可执行权限，不包括 ignored build 目录、实验输出 `results/`、自动生成的 `docs/baseline.md`。

基线 Binary 内嵌被构建的代码身份。Runner 比较当前源码与 binary 身份、构建日志身份，前后再比较源码快照。只允许 clean source、40 位 commit、Release。结果提交后 HEAD 变化，必须重新 build，不能拿旧 binary 冒充新提交。结果 commit 不等于被测试代码 commit。

完整构建 flags 和依赖版本记录在 CMakeCache、compile_commands 和 verbose build log。GPU 指标来自 actual API，不假设 SM 数、带宽或显存容量。CUDA 13 中显存时钟 property 被移除，环境脚本补充 UUID 关联的 `nvidia-smi` 时钟查询，不填猜测值。

## 数据与并发

CSV 表头严格校验，追加输出；原始样本逐轮记录，保留失败证据。Runner 使用目录锁序列化同仓库的正式/Profiling 实验。直接执行 binary 时 CSV writer 本身不跨进程加锁，调用者必须串行运行。若 runner 被 SIGKILL，锁可能残留；先确认没有运行中的实验，再人工 `rmdir results/.stage0.lock`。

分析工具拒绝不同输入、尺寸、GPU UUID、commit、数学模式、warmup/iterations、容差之间的配对；从 raw 样本重算统计、GFLOPS、比值，确认 initial/final correctness 和 run 完成标记后才生成文档。没有数据就非零退出，不生成示例性能。

## 参考资料

- [CUDA Runtime Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [cuBLAS Data Layout、Math Mode、SGEMM](https://docs.nvidia.com/cuda/cublas/index.html)
- [CUDA Device Properties](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
- [CMake CUDA architectures](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)

GPU 环境未提供前，不宣称 CUDA 编译、正确性或性能已验收。
