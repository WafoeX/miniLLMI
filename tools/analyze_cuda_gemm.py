#!/usr/bin/env python3
"""Verify raw Stage 9 v0/v1/cuBLAS paired GEMM evidence and derive its gate."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

BASE_COLUMNS = (
    "timestamp", "run_id", "commit", "source_digest", "source_dirty", "experiment", "gpu", "gpu_uuid", "device",
    "compute_capability", "cuda", "cuda_driver", "cublas_version", "compiler", "cuda_compiler", "cuda_architectures",
    "build_type", "kernel", "m", "n", "k", "dtype", "layout", "math_mode", "alpha", "beta", "warmup", "iterations",
    "seed", "input_hash", "block_x", "block_y", "bm", "bn", "bk", "tm", "tn", "atol", "rtol", "min_ms", "max_ms",
    "median_ms", "mean_ms", "std_ms", "gflops", "speedup_vs_naive", "cublas_ratio", "max_error", "mean_error",
    "relative_error", "violations", "nonfinite", "status",
)
COLUMNS = ("schema_version",) + BASE_COLUMNS + ("threads", "vector", "shared_bytes", "kernel_order")
KERNEL_CONFIGS = {
    "sgemm_v0_naive": {"block_x": "16", "block_y": "16", "bm": "1", "bn": "1", "bk": "1", "tm": "1", "tn": "1", "threads": "256", "vector": "1", "shared_bytes": "0"},
    "sgemm_v1_tiled": {"block_x": "16", "block_y": "16", "bm": "16", "bn": "16", "bk": "16", "tm": "1", "tn": "1", "threads": "256", "vector": "1", "shared_bytes": "2048"},
    "cublas": {"block_x": "0", "block_y": "0", "bm": "0", "bn": "0", "bk": "0", "tm": "0", "tn": "0", "threads": "0", "vector": "0", "shared_bytes": "0"},
}
PAIR_FIELDS = ("timestamp", "run_id", "commit", "source_digest", "source_dirty", "experiment", "gpu", "gpu_uuid", "device",
               "compute_capability", "cuda", "cuda_driver", "cublas_version", "compiler", "cuda_compiler", "cuda_architectures",
               "build_type", "m", "n", "k", "dtype", "layout", "math_mode", "alpha", "beta", "warmup", "iterations",
               "seed", "input_hash", "atol", "rtol", "schema_version", "kernel_order")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read JSON: " + str(path)) from error


def rows(path, expected=COLUMNS):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("CSV schema mismatch: " + str(path))
        result = list(reader)
    if any(None in row or None in row.values() for row in result):
        raise ValueError("ragged CSV: " + str(path))
    return result


def number(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid number: " + repr(value)) from error
    if not math.isfinite(parsed):
        raise ValueError("nonfinite number: " + repr(value))
    return parsed


def integer(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid integer: " + repr(value)) from error
    if str(parsed) != value:
        raise ValueError("invalid integer: " + repr(value))
    return parsed


def verify_kernel_row(row, order):
    if row["schema_version"] != "stage9-gemm-v1" or row["experiment"] != "stage9" or row["status"] != "ok":
        raise ValueError("not a successful Stage 9 row")
    if row["kernel"] not in KERNEL_CONFIGS or row["kernel_order"] != order:
        raise ValueError("unknown kernel or uncontrolled order")
    if any(row[field] != value for field, value in KERNEL_CONFIGS[row["kernel"]].items()):
        raise ValueError("kernel configuration differs from registry")
    if row["source_dirty"] != "0" or row["build_type"] != "Release" or row["dtype"] != "fp32" or row["layout"] != "row-major" or row["math_mode"] != "fp32_pedantic":
        raise ValueError("uncontrolled benchmark mode")
    if integer(row["warmup"]) != 10 or integer(row["iterations"]) != 50 or integer(row["violations"]) or integer(row["nonfinite"]):
        raise ValueError("invalid timing controls or correctness result")
    for field in ("max_error", "mean_error", "relative_error", "atol", "rtol"):
        if number(row[field]) < 0:
            raise ValueError("negative error or tolerance")


def statistics_from_samples(summary, raw):
    if len(raw) != 50:
        raise ValueError("raw sample coverage mismatch")
    samples = []
    for index, sample in enumerate(raw):
        if sample["status"] != "timing_sample" or integer(sample["sample_index"]) != index:
            raise ValueError("bad sample sequence")
        if any(sample[field] != summary[field] for field in COLUMNS):
            raise ValueError("raw sample metadata differs from summary")
        elapsed = number(sample["elapsed_ms"])
        if elapsed <= 0:
            raise ValueError("nonpositive CUDA event result")
        samples.append(elapsed)
    values = {"min_ms": min(samples), "max_ms": max(samples), "mean_ms": statistics.mean(samples),
              "median_ms": statistics.median(samples), "std_ms": statistics.pstdev(samples)}
    for field, expected in values.items():
        if not math.isclose(number(summary[field]), expected, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(field + " disagrees with raw samples")
    m, n, k = (integer(summary[field]) for field in ("m", "n", "k"))
    gflops = 2.0 * m * n * k / (values["median_ms"] * 1e6)
    if not math.isclose(number(summary["gflops"]), gflops, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("GFLOPS disagrees with raw samples")
    return values["median_ms"]


def verify_measurement(root, item, sizes):
    raw_root = root / item["raw_dir"]
    aggregate = rows(root / item["csv"])
    expected_count = len(sizes) * len(KERNEL_CONFIGS)
    if len(aggregate) != expected_count:
        raise ValueError("aggregate summary coverage mismatch")
    result = {}
    for size in sizes:
        shape = f"{size}x{size}x{size}"
        summary = rows(raw_root / shape / "summary.csv")
        correctness = rows(raw_root / shape / "correctness.csv")
        if len(summary) != len(KERNEL_CONFIGS) or {row["kernel"] for row in summary} != set(KERNEL_CONFIGS):
            raise ValueError("missing kernels in paired shape")
        if any(row not in aggregate for row in summary):
            raise ValueError("aggregate row absent from shape summary")
        paired = list(summary)
        if any(row["run_id"] != item["run_id"] or integer(row["m"]) != size or integer(row["n"]) != size or integer(row["k"]) != size for row in paired):
            raise ValueError("shape/run identity mismatch")
        if any(any(row[field] != paired[0][field] for field in PAIR_FIELDS) for row in paired[1:]):
            raise ValueError("uncontrolled kernel comparison")
        by_kernel = {row["kernel"]: row for row in paired}
        for kernel, summary_row in by_kernel.items():
            verify_kernel_row(summary_row, item["order"])
            raw_columns = COLUMNS + ("sample_index", "elapsed_ms")
            raw = rows(raw_root / shape / (kernel + "_samples.csv"), raw_columns)
            median = statistics_from_samples(summary_row, raw)
            evidence = [row for row in correctness if row["kernel"] == kernel]
            if [row["status"] for row in evidence] != ["passed_initial", "passed_final"]:
                raise ValueError("missing initial/final correctness evidence")
            if any(any(row[field] != summary_row[field] for field in COLUMNS if field not in ("min_ms", "max_ms", "median_ms", "mean_ms", "std_ms", "gflops", "speedup_vs_naive", "cublas_ratio", "status")) for row in evidence):
                raise ValueError("correctness metadata mismatch")
            result[(size, kernel)] = median
        naive, tiled, cublas = (by_kernel[name] for name in ("sgemm_v0_naive", "sgemm_v1_tiled", "cublas"))
        for row in paired:
            expected_speedup = number(row["gflops"]) / number(naive["gflops"])
            expected_ratio = 100 * number(row["gflops"]) / number(cublas["gflops"])
            if not math.isclose(number(row["speedup_vs_naive"]), expected_speedup, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("speedup formula mismatch")
            if not math.isclose(number(row["cublas_ratio"]), expected_ratio, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("cuBLAS ratio formula mismatch")
    return result


def analyze(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["status"] not in ("running", "passed") or manifest.get("source") != manifest.get("source_after", manifest.get("source")):
        raise ValueError("incomplete or changed source")
    if manifest["status"] == "passed":
        for name, expected in manifest.get("artifact_sha256", {}).items():
            if digest(root / name) != expected:
                raise ValueError("artifact digest mismatch: " + name)
    config = manifest["configuration"]
    if tuple(config["sizes"]) != (512, 1024, 2048, 4096) or config["warmup"] != 10 or config["iterations"] != 50:
        raise ValueError("unexpected Stage 9 protocol")
    if len(manifest["measurements"]) != 3 or tuple(item["order"] for item in manifest["measurements"]) != tuple(config["orders"]):
        raise ValueError("missing paired repetitions")
    all_rows = []
    for item in manifest["measurements"]:
        measured = verify_measurement(root, item, config["sizes"])
        ratios = [measured[(size, "sgemm_v0_naive")] / measured[(size, "sgemm_v1_tiled")] for size in config["sizes"]]
        all_rows.append({"repeat": item["repeat"], "order": item["order"], "ratios": ratios,
                         "geomean_speedup": math.prod(ratios) ** (1 / len(ratios))})
    gate = statistics.median(row["geomean_speedup"] for row in all_rows)
    result = {"status": "passed", "source": manifest["source"], "paired_runs": all_rows,
              "median_paired_geomean_speedup": gate,
              "candidate_wins_every_run": all(row["geomean_speedup"] > 1 for row in all_rows),
              "gate_passed": gate >= 1.05 and all(row["geomean_speedup"] > 1 for row in all_rows)}
    return result


def main():
    root = Path(sys.argv[1])
    result = analyze(root)
    table = "# Stage 9 CUDA SGEMM paired analysis\n\n"
    table += "| Repeat | Kernel order | Geometric-mean v0/v1 speedup |\n|---:|---|---:|\n"
    for row in result["paired_runs"]:
        table += f"| {row['repeat']} | {row['order']} | {row['geomean_speedup']:.6f}x |\n"
    table += f"\nMedian paired geometric-mean speedup: {result['median_paired_geomean_speedup']:.6f}x; gate passed: {result['gate_passed']}.\n"
    for name, content in (("analysis.json", json.dumps(result, indent=2) + "\n"), ("summary.md", table)):
        target = root / name
        if target.exists() and target.read_text() != content:
            raise ValueError("refusing to overwrite derived artifact")
        target.write_text(content)
    print(json.dumps({"gate_passed": result["gate_passed"], "speedup": result["median_paired_geomean_speedup"]}))


if __name__ == "__main__":
    try:
        main()
    except (IndexError, OSError, ValueError, KeyError, TypeError) as error:
        print("analyze_cuda_gemm:", error, file=sys.stderr)
        sys.exit(1)
