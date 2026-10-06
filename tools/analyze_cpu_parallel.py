#!/usr/bin/env python3
"""Validate Stage 7 paired CPU data and derive the required speedup summaries."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


COLUMNS = ("sample", "ms", "batch", "threads", "m", "n", "k", "seed", "input_hash", "algorithm", "workload", "policy", "checksum")
NAMES = {"v0": "cpu-ijk-fp32-v0", "ikj": "cpu-ikj-fp32-c1", "fifo": "cpu-ikj-fp32-c3-fifo"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read JSON: " + str(path)) from error


def samples(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("CSV schema mismatch")
        rows = list(reader)
    if len(rows) != 10 or [row["sample"] for row in rows] != list(map(str, range(10))):
        raise ValueError("sample coverage mismatch")
    values = [float(row["ms"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("invalid timing")
    return rows, values


def analyze(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["status"] != "passed" or manifest["source"] != manifest["source_after"]:
        raise ValueError("incomplete or changed source")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("artifact digest mismatch: " + name)
    rows = []
    for item in manifest["measurements"]:
        raw, correctness = root / item["raw"], root / item["correctness"]
        document = read_json(correctness)
        if document["status"] != "passed" or any(document[key]["violations"] or document[key]["nonfinite"] for key in ("initial", "final")):
            raise ValueError("correctness failure")
        raw_rows, values = samples(raw)
        expected = NAMES[item["algorithm"]]
        if any(row["algorithm"] != expected or row["threads"] != str(item["threads"]) or row["workload"] != "gemm" or row["policy"] != "caller" or row["m"] != str(item["size"]) for row in raw_rows):
            raise ValueError("raw controls mismatch")
        rows.append({**item, "algorithm_name": expected, "median_ms": statistics.median(values),
                     "gflops": 2 * item["size"] ** 3 / (statistics.median(values) * 1e6), "raw": item["raw"]})
    pairs = []
    for repeat in range(3):
        ratios = []
        for size in manifest["configuration"]["sizes"]:
            base = next(row for row in rows if row["repeat"] == repeat and row["algorithm"] == "v0" and row["size"] == size)
            candidate = next(row for row in rows if row["repeat"] == repeat and row["algorithm"] == "ikj" and row["size"] == size)
            ratios.append(base["median_ms"] / candidate["median_ms"])
        pairs.append({"repeat": repeat, "ratios": ratios, "geomean": math.prod(ratios) ** (1 / len(ratios))})
    speedup = statistics.median(pair["geomean"] for pair in pairs)
    result = {"status": "passed", "source": manifest["source"], "host": manifest["host"], "configuration": manifest["configuration"],
              "rows": rows, "paired_geomeans": pairs, "median_paired_geomean_speedup": speedup,
              "candidate_wins_every_run": all(pair["geomean"] > 1 for pair in pairs),
              "gate_passed": speedup >= 1.05 and all(pair["geomean"] > 1 for pair in pairs)}
    return result


def main():
    root = Path(sys.argv[1])
    result = analyze(root)
    table = "# Stage 7 CPU-only scaling\n\n| Algorithm | Threads | Size | Repeat | Median ms | GFLOPS |\n|---|---:|---:|---:|---:|---:|\n"
    for row in result["rows"]:
        table += f"| {row['algorithm_name']} | {row['threads']} | {row['size']} | {row['repeat']} | {row['median_ms']:.6f} | {row['gflops']:.6f} |\n"
    table += "\nPaired v0/C1 per-run geometric-mean speedups: " + ", ".join(f"{pair['geomean']:.6f}" for pair in result["paired_geomeans"])
    table += f"; median {result['median_paired_geomean_speedup']:.6f}; gate passed: {result['gate_passed']}.\n"
    for name, text in (("analysis.json", json.dumps(result, indent=2) + "\n"), ("summary.md", table)):
        path = root / name
        if path.exists() and path.read_text() != text:
            raise ValueError("refusing to overwrite derived artifact")
        path.write_text(text)
    print(json.dumps({"gate_passed": result["gate_passed"], "speedup": result["median_paired_geomean_speedup"]}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print("analyze_cpu_parallel:", error, file=sys.stderr)
        sys.exit(1)
