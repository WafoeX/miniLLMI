#!/usr/bin/env python3
"""Recompute S6 CPU v0 baselines; no oracle/GPU timing or S7 speedup claim."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from pathlib import Path

CONFIG = {"runs": 3, "sizes": [128, 256, 512, 1024], "graph_size": 16, "warmup": 3,
          "samples": 10, "gemm_batch": 1, "graph_batch": 20, "threads": 1, "seed": 42,
          "atol": 0.001, "rtol": 0.001, "algorithm": "cpu-ijk-fp32-v0", "workload_version": 1}
SCOPE = "CPU-only Stage 6 scalar baseline; local host allowed; no GPU or speedup gate"
COLUMNS = ("sample", "ms", "batch", "threads", "m", "n", "k", "seed", "input_hash", "algorithm", "workload", "policy", "checksum")
BUILD_LABELS = ("cmake-version", "tests-configure", "tests-build", "ctest", "production-configure", "production-build", "binary-probe", "production-correctness", "production-symbols", "vectorization")


def measurements():
    result = []
    for repeat in range(3):
        for size in CONFIG["sizes"] if repeat % 2 == 0 else reversed(CONFIG["sizes"]):
            result.append({"repeat": repeat, "workload": "gemm", "policy": "caller", "size": size})
        for policy in ("dynamic", "reuse") if repeat % 2 == 0 else ("reuse", "dynamic"):
            result.append({"repeat": repeat, "workload": "graph", "policy": policy, "size": 16})
    for item in result:
        item["label"] = "repeat-{repeat}-{workload}-{policy}-{size}".format(**item)
        item["raw"] = item["label"] + ".csv"
        item["correctness"] = item["label"] + ".json"
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read CPU evidence JSON: " + str(path)) from error


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
            "mean_ms": statistics.mean(samples), "stddev_ms": statistics.pstdev(samples)}


def read_samples(path, expected):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("raw CSV schema mismatch")
        rows = list(reader)
    if len(rows) != CONFIG["samples"] or [r["sample"] for r in rows] != list(map(str, range(10))):
        raise ValueError("raw sample coverage mismatch")
    samples, checksums = [], []
    for row in rows:
        if set(row) != set(COLUMNS) or any(row.get(k) != str(v) for k, v in expected.items()):
            raise ValueError("raw measurement controls mismatch")
        try:
            ms, checksum = float(row["ms"]), float(row["checksum"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid numeric raw sample") from error
        if not math.isfinite(ms) or ms <= 0 or not math.isfinite(checksum):
            raise ValueError("nonfinite/nonpositive raw timing or checksum")
        samples.append(ms)
        checksums.append(checksum)
    if len(set(checksums)) != 1:
        raise ValueError("deterministic checksum changed across samples")
    return samples, checksums[0]


def analyze(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    source, binary = manifest["source"], manifest["binary"]
    if (manifest["schema_version"] != 1 or manifest["scope"] != SCOPE or manifest["status"] not in ("measured", "passed") or
            (not isinstance(source["source_dirty"], bool) or source["source_dirty"]) or source != manifest["source_after"] or
            not re.fullmatch(r"[0-9a-f]{40}", source["commit"]) or not re.fullmatch(r"[0-9a-f]{64}", source["source_digest"]) or
            manifest["configuration"] != CONFIG or manifest["measurements"] != measurements()):
        raise ValueError("invalid source/configuration/completion provenance")
    if (any(binary[k] != source[k] for k in source) or binary["build_type"] != "Release" or (not isinstance(binary["testing"], bool) or binary["testing"]) or
            binary["benchmark"] != "cpu-v1" or binary["backend"] != "cpu" or binary["algorithm"] != CONFIG["algorithm"] or binary["threads"] != 1 or
            not re.fullmatch(r"[0-9a-f]{64}", manifest["binary_sha256"]) or read_json(root / "binary-probe.log") != binary):
        raise ValueError("invalid binary identity")
    expected_logs = [label + ".log" for label in BUILD_LABELS] + [m["label"] + ".log" for m in measurements()]
    commands = manifest["commands"]
    if [c["log"] for c in commands] != expected_logs or any(c["exit_code"] != 0 for c in commands):
        raise ValueError("invalid build/measurement command coverage")
    expected_files = set(expected_logs) | {mode + "-" + filename for mode in ("tests", "production") for filename in ("CMakeCache.txt", "compile_commands.json")} | {m[k] for m in measurements() for k in ("raw", "correctness")}
    if set(manifest["artifact_sha256"]) != expected_files:
        raise ValueError("artifact identity coverage mismatch")
    for name, sha in manifest["artifact_sha256"].items():
        if Path(name).name != name or digest(root / name) != sha:
            raise ValueError("artifact digest mismatch: " + name)
    for mode, testing in (("tests", True), ("production", False)):
        cache = (root / (mode + "-CMakeCache.txt")).read_text().splitlines()
        required = ("CMAKE_BUILD_TYPE:STRING=Release", "ENABLE_CUDA:BOOL=OFF", "ENABLE_SANITIZERS:BOOL=OFF", "BUILD_TESTING:BOOL=" + ("ON" if testing else "OFF"))
        if any(setting not in cache for setting in required):
            raise ValueError("build configuration mismatch")
        comp = read_json(root / (mode + "-compile_commands.json"))
        if not isinstance(comp, list) or not comp or not any(Path(c["file"]).name == "cpu_scalar.cpp" for c in comp):
            raise ValueError("missing scalar compile command")
        for c in comp:
            flags = c.get("command", " ".join(c.get("arguments", [])))
            if (not testing and "RUNTIME_TESTING" in flags) or any(flag in flags.split() for flag in ("-ffast-math", "-Ofast")):
                raise ValueError("test hooks/fast math in production flags")
    if "cpu_allocation_counts" in (root / "production-symbols.log").read_text():
        raise ValueError("production has allocation test hooks")
    exe = commands[6]["argv"][0]
    if commands[6]["argv"] != [exe, "--probe"] or commands[7]["argv"] != [exe, "--self-test"]:
        raise ValueError("correctness/probe binary mismatch")
    # Read bytes in this archive; validate original absolute argv against its
    # recorded root, so moved archives replay without rewriting historical logs.
    recorded_root = Path(manifest["recorded_output_root"])
    if not recorded_root.is_absolute():
        raise ValueError("recorded output root must be absolute")
    hashes, checksums, rows = {}, {}, []
    for index, measurement in enumerate(measurements()):
        command = commands[len(BUILD_LABELS) + index]
        size, workload, policy = measurement["size"], measurement["workload"], measurement["policy"]
        expected_argv = [exe, "--workload", workload, "--policy", policy, "--size", str(size), "--raw", str(recorded_root / measurement["raw"]), "--correctness", str(recorded_root / measurement["correctness"])]
        if command["argv"] != expected_argv:
            raise ValueError("measurement argv/order mismatch")
        correctness = read_json(root / measurement["correctness"])
        expected = {"workload": workload, "policy": policy, "algorithm": CONFIG["algorithm"], "m": size, "n": size, "k": size,
                    "seed": 42, "warmup": 3, "samples": 10, "batch": 1 if workload == "gemm" else 20, "threads": 1, "atol": 0.001, "rtol": 0.001,
                    "schema_version": 1, "oracle": "unchanged-stage0-fp64-untimed", "status": "passed"}
        if any(correctness.get(k) != v for k, v in expected.items()):
            raise ValueError("correctness controls mismatch")
        for phase in ("initial", "final"):
            metrics = correctness[phase]
            if (not isinstance(metrics["passed"], bool) or not metrics["passed"]) or metrics["violations"] != 0 or metrics["nonfinite"] != 0:
                raise ValueError("correctness failed")
            for key in ("max_abs", "mean_abs", "max_relative"):
                if not math.isfinite(metrics[key]) or metrics[key] < 0:
                    raise ValueError("invalid error metric")
        identity = correctness["input_hash"]
        if not re.fullmatch(r"[0-9a-f]{16}", identity):
            raise ValueError("invalid input hash")
        condition = (workload, size)
        if hashes.setdefault(condition, identity) != identity:
            raise ValueError("paired inputs differ")
        raw_expected = {k: expected[k] for k in ("workload", "policy", "algorithm", "m", "n", "k", "seed", "batch", "threads")}
        raw_expected["input_hash"] = identity
        samples, checksum = read_samples(root / measurement["raw"], raw_expected)
        if checksums.setdefault(condition, checksum) != checksum:
            raise ValueError("repeated/paired checksums differ")
        summary = stats(samples)
        rows.append({"run_id": manifest["run_id"], "timestamp": manifest["timestamp"], "tested_commit": source["commit"], "source_digest": source["source_digest"],
                     "source_dirty": False, "build_type": "Release", "host": manifest["host"], "platform": manifest["platform"], "compiler": binary["compiler"],
                     "backend": "cpu", "repeat": measurement["repeat"], **raw_expected, "warmup": 3, "samples": 10, **summary,
                     "gflops": 2 * size**3 / (summary["median_ms"] * 1e6) if workload == "gemm" else None,
                     "initial_max_abs": correctness["initial"]["max_abs"], "final_max_abs": correctness["final"]["max_abs"], "checksum": checksum,
                     "correctness": "passed", "raw": measurement["raw"], "correctness_file": measurement["correctness"]})
    ratios = []
    for repeat in range(3):
        policies = {r["policy"]: r["median_ms"] for r in rows if r["repeat"] == repeat and r["workload"] == "graph"}
        ratios.append(policies["dynamic"] / policies["reuse"])
    return {"source": source, "status": "passed", "scope": SCOPE, "benefit_gate": "not_applicable_baseline_only_S7_deferred", "graph_paired_ratios": ratios,
            "graph_median_paired_ratio": statistics.median(ratios), "rows": rows}


def save(root):
    root = Path(root).resolve()
    result = analyze(root)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    table = "# Stage 6 CPU-only baseline\n\nSingle-thread FP32 ijk v0, full backend call. Unchanged FP64 oracle is untimed. Three independent baseline runs, not a speedup experiment. No GPU comparison.\n\n| Shape | Repeat | Median ms | GFLOPS |\n|---|---:|---:|---:|\n"
    for row in result["rows"]:
        if row["workload"] == "gemm":
            table += f"| {row['m']}³ | {row['repeat']} | {row['median_ms']:.6f} | {row['gflops']:.6f} |\n"
    table += "\nGraph [16,16] MATMUL → ADD → MUL, same backend, dynamic/reuse paired ratios: " + ", ".join(f"{r:.6f}" for r in result["graph_paired_ratios"]) + f"; median {result['graph_median_paired_ratio']:.6f}. Includes validation, oracle scan, output destruction. No required latency gate.\n"
    for name, text in (("cpu_gemm.csv", stream.getvalue()), ("analysis.json", json.dumps(result, indent=2) + "\n"), ("summary.md", table)):
        path = root / name
        if path.exists() and path.read_bytes() != text.encode():
            raise ValueError("refusing to overwrite different derived artifact: " + name)
        if not path.exists():
            path.write_bytes(text.encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        result = save(args.run)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print("analyze_cpu: " + str(error), file=sys.stderr)
        return 1
    print("CPU baseline: PASS; rows=" + str(len(result["rows"])) + "; graph paired ratios=" + str(result["graph_paired_ratios"]))
    return 0

if __name__ == "__main__":
    sys.exit(main())
