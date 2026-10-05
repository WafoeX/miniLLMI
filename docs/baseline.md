# Stage 0 Baseline — 等待真实 GPU 实验

尚未在 T4 上运行 CUDA Benchmark，因此没有性能数字，也没有预填 `results/gemm/baseline.csv`。

完成服务器 Release Build、CUDA 正确性测试和默认四尺寸 Benchmark 后，`scripts/run_gemm_benchmark.sh` 会核对逐次原始样本，自动更新本文件。此占位声明不是实验报告，不代表 Stage 0 已全部验收。

方法、CSV 字段和验收标准见 [experiments.md](experiments.md)、[stage0.md](stage0.md)。
