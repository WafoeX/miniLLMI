#!/usr/bin/env python3
"""Stage 0 only: verify raw CUDA samples, then generate a baseline report. No invented data."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PAIR_FIELDS = (
    "timestamp", "run_id", "commit", "source_digest", "source_dirty", "experiment", "gpu", "gpu_uuid", "device",
    "compute_capability", "cuda", "cuda_driver", "cublas_version", "compiler", "cuda_compiler", "cuda_architectures",
    "build_type", "m", "n", "k", "dtype", "layout", "math_mode", "alpha", "beta", "warmup", "iterations",
    "seed", "input_hash", "atol", "rtol",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"invalid CSV header: {path}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"ragged CSV: {path}")
        return rows


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer: {value!r}") from error


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid number: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"nonfinite numeric value: {value}")
    return result


def equal_number(actual, expected, field):
    if not math.isclose(finite(actual), expected, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"{field} disagrees with raw samples/formula: {actual} vs {expected}")


def safe_id(value):
    if not value or len(value) > 120 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("unsafe run ID")
    return value


def validate_row(row, raw_root):
    if (row["status"] != "ok" or row["experiment"] != "baseline" or row["build_type"] != "Release"
            or row["source_dirty"] != "0" or row["dtype"] != "fp32" or row["layout"] != "row-major"
            or row["math_mode"] != "fp32_pedantic" or row["alpha"] != "1" or row["beta"] != "0"):
        raise ValueError("not a successful controlled Stage 0 baseline")
    commit = row["commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("missing full commit hash")
    if len(row["source_digest"]) != 64 or not row["gpu_uuid"]:
        raise ValueError("missing source digest/GPU identity")
    m, n, k = (parse_int(row[x]) for x in ("m", "n", "k"))
    if min(m, n, k) <= 0 or parse_int(row["warmup"]) < 10 or parse_int(row["iterations"]) < 30:
        raise ValueError("invalid dimensions/warmup/iterations")
    if row["kernel"] not in ("sgemm_v0_naive", "cublas"):
        raise ValueError("not a Stage 0 kernel")
    if parse_int(row["violations"]) != 0 or parse_int(row["nonfinite"]) != 0:
        raise ValueError("failed correctness cannot produce a performance report")
    for field in ("max_error", "mean_error", "relative_error", "atol", "rtol"):
        if finite(row[field]) < 0:
            raise ValueError("negative error/tolerance")
    run_id = safe_id(row["run_id"])
    run_dir = Path(raw_root) / run_id
    if not (run_dir / "completed.txt").is_file() or (run_dir / "failure.txt").exists():
        raise ValueError("run was not fully completed successfully")
    shape = f"{m}x{n}x{k}"
    samples = read_csv(Path(raw_root) / run_id / shape / f"{row['kernel']}_samples.csv")
    if len(samples) != parse_int(row["iterations"]):
        raise ValueError("raw sample count mismatch")
    times = []
    for i, sample in enumerate(samples):
        if any(sample[field] != row[field] for field in PAIR_FIELDS) or sample["kernel"] != row["kernel"]:
            raise ValueError("raw sample metadata differs from summary")
        if sample["status"] != "timing_sample" or parse_int(sample["sample_index"]) != i:
            raise ValueError("raw sample sequence/status mismatch")
        ms = finite(sample["elapsed_ms"])
        if ms <= 0:
            raise ValueError("nonpositive sample time")
        times.append(ms)
    stats = {"min_ms": min(times), "max_ms": max(times), "mean_ms": statistics.mean(times),
             "median_ms": statistics.median(times), "std_ms": statistics.pstdev(times)}
    for field, expected in stats.items():
        equal_number(row[field], expected, field)
    perf = 2.0 * m * n * k / (stats["median_ms"] * 1e6)
    equal_number(row["gflops"], perf, "gflops")
    correctness = read_csv(Path(raw_root) / run_id / shape / "correctness.csv")
    relevant = [r for r in correctness if r["kernel"] == row["kernel"]]
    if [r["status"] for r in relevant] != ["passed_initial", "passed_final"]:
        raise ValueError("missing initial/final correctness evidence")
    for item in relevant:
        if any(item[field] != row[field] for field in PAIR_FIELDS):
            raise ValueError("correctness metadata mismatch")
        if parse_int(item["violations"]) or parse_int(item["nonfinite"]):
            raise ValueError("raw correctness evidence failed")
    for field in ("max_error", "mean_error", "relative_error", "violations", "nonfinite"):
        if relevant[-1][field] != row[field]:
            raise ValueError("final correctness differs from summary")
    return perf


def render(rows, raw_root, csv_path, run_id=None):
    selected = [r for r in rows if r.get("experiment") == "baseline" and
                (run_id is None or r.get("run_id") == run_id)]
    if not selected:
        raise ValueError("no real baseline rows; report not generated")
    failed = [r for r in selected if r.get("status") != "ok"]
    if failed:
        raise ValueError(f"{len(failed)} failed baseline rows retained; report refused")
    groups = defaultdict(dict)
    for row in selected:
        key = (row["run_id"], parse_int(row["m"]), parse_int(row["n"]), parse_int(row["k"]))
        if row["kernel"] in groups[key]:
            raise ValueError("duplicate kernel within one run/shape")
        perf = validate_row(row, raw_root)
        groups[key][row["kernel"]] = (row, perf)
    lines = ["# Stage 0 — CUDA GEMM Baseline（自动生成）", "",
             f"数据来源：`{csv_path}`；逐次 CUDA Event 数据：`{raw_root}/<run_id>/`。",
             "这里只报告已运行的尺寸，不等于所有 Stage 0 验收项已完成。未比较跨 run/跨 commit 的数据。", "",
             "## 定义和计时方法", "",
             "- CPU Reference：单线程 ijk、FP64 累加后转 FP32；全矩阵校验，不参与性能比较。",
             "- Naive：每个 CUDA thread 计算一个 C 元素，无 shared memory/float4/register blocking。",
             "- cuBLAS：row-major C=A×B 映射为 column-major Cᵀ=Bᵀ×Aᵀ，alpha=1、beta=0。",
             "- 两者 FP32；cuBLAS 使用 PEDANTIC_MATH（不是不受约束的最高性能上限）。",
             "- 数据传输、CPU reference、分配、校验、文件写入不在 Event 区间内；同一 stream 逐轮同步。",
             "- warmup≥10、iterations≥30；主指标 median，std 为总体标准差（ddof=0）。",
             "- GFLOPS=2MNK/(median_ms×10⁶)；cuBLAS Ratio 是百分比，不是 0～1 的小数。",
             "- 混合逐元素判据 |actual-ref|≤atol+rtol×|ref|；relative_error 分母 max(|ref|,10⁻¹²)。", ""]
    current = None
    for (rid, m, n, k), kernels in sorted(groups.items()):
        if set(kernels) != {"sgemm_v0_naive", "cublas"}:
            raise ValueError("both naive and cuBLAS required for a baseline comparison")
        naive, ng = kernels["sgemm_v0_naive"]
        blas, bg = kernels["cublas"]
        if any(naive[field] != blas[field] for field in PAIR_FIELDS):
            raise ValueError("uncontrolled naive/cuBLAS comparison")
        for row, perf in [(naive, ng), (blas, bg)]:
            equal_number(row["speedup_vs_naive"], perf / ng, "speedup_vs_naive")
            equal_number(row["cublas_ratio"], 100.0 * perf / bg, "cublas_ratio")
        if rid != current:
            current = rid
            lines.extend([f"## Run `{rid}`", "", f"- Timestamp：{naive['timestamp']}",
                          f"- Commit：`{naive['commit']}`；source SHA256：`{naive['source_digest']}`",
                          f"- GPU：{naive['gpu']}；UUID：`{naive['gpu_uuid']}`；CC：{naive['compute_capability']}",
                          f"- CUDA runtime/driver API：{naive['cuda']}/{naive['cuda_driver']}；cuBLAS：{naive['cublas_version']}",
                          f"- Compiler：{naive['compiler']}；nvcc：{naive['cuda_compiler']}；arch：{naive['cuda_architectures']}",
                          f"- Release；warmup={naive['warmup']}；iterations={naive['iterations']}；seed={naive['seed']}",
                          "- 完整环境、configure/build 日志及 CMakeCache：见对应 raw run。", ""])
        lines.extend([f"### M={m}, N={n}, K={k}", "", f"Input FNV-1a：`{naive['input_hash']}`；atol={naive['atol']}，rtol={naive['rtol']}。", "",
                      "| Kernel | median ms | mean ms | std ms | min ms | max ms | GFLOPS | vs naive | cuBLAS % | max abs error | mean abs error | max relative |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row, perf in [(naive, ng), (blas, bg)]:
            values = [row["kernel"], *[f"{finite(row[f]):.6g}" for f in ("median_ms", "mean_ms", "std_ms", "min_ms", "max_ms")],
                      f"{perf:.6g}", f"{perf/ng:.3f}x", f"{100*perf/bg:.3f}%",
                      *[f"{finite(row[f]):.6g}" for f in ("max_error", "mean_error", "relative_error")]]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(["## 边界", "", "不推断优化版本趋势、模型速度或量化效果；Stage 0 没有这些实现。",
                  "Kernel 固定先 naive 后 cuBLAS；GPU 频率/热状态/其他进程会影响结果，请保存 nvidia-smi 前后快照并重复完整 run。",
                  "Profiling 时间不用于此表；保留失败/退化实验，不自动删数据。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/gemm/baseline.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("results/gemm/raw"))
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, default=Path("docs/baseline.md"))
    args = parser.parse_args()
    try:
        report = render(read_csv(args.csv), args.raw_root, args.csv, args.run_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(report, encoding="utf-8")
        temp.replace(args.output)
        print(f"Verified raw samples; wrote {args.output}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"analyze_results: {error}\n")


if __name__ == "__main__":
    main()
