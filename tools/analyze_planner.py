#!/usr/bin/env python3
"""Recompute CPU-only S5 evidence, including the predeclared memory gate."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

WORKLOADS = ("chain", "diamond")
POLICIES = ("dynamic", "no_reuse", "reuse")
CONFIG = {"runs": 3, "samples": 10, "warmup": 3, "batch": 20, "prepare_batch": 1, "alignment": 64, "workload_version": 1, "gate_workload": "chain"}
COLUMNS = ("sample", "ms", "batch", "requests", "releases", "backing_allocs", "backing_frees", "intermediate_backing_allocs", "output_destroy_backing_frees", "peak_payload_bytes", "peak_intermediate_bytes", "output_live_bytes", "reused", "inplace", "arena_capacity", "external_bytes", "peak_resident_bytes", "prepared_backing_allocs", "checksum", "correctness")
PREPARE_COLUMNS = ("sample", "ms", "batch", "arena_capacity", "prepared_backing_allocs", "correctness")
EXPECTED = {"chain": (12, 512, 512, 3072, 512, 10), "diamond": (5, 768, 768, 1280, 768, 2)}


def checksum(workload):
    values = [0.125 + i / 16 for i in range(64)]
    if workload == "diamond":
        return sum(value * 4 for value in values)
    for _ in range(6):
        values = [value * 0.5 + 0.5 for value in values]
    return sum(values)


def expected_counters(workload, policy):
    requests, peak, external, no_reuse, reuse, reused = EXPECTED[workload]
    dynamic = policy == "dynamic"
    capacity = 0 if dynamic else (reuse if policy == "reuse" else no_reuse)
    return {"batch": 20, "requests": requests, "releases": requests - 1,
            "backing_allocs": requests if dynamic else 0, "backing_frees": requests - 1 if dynamic else 0,
            "intermediate_backing_allocs": requests - 1 if dynamic else 0, "output_destroy_backing_frees": 1 if dynamic else 0,
            "peak_payload_bytes": peak, "peak_intermediate_bytes": peak, "output_live_bytes": 256,
            "reused": reused if policy == "reuse" else 0, "inplace": 0, "arena_capacity": capacity,
            "external_bytes": external, "peak_resident_bytes": external + (peak if dynamic else capacity),
            "prepared_backing_allocs": 0 if dynamic else 1}


def read_samples(path, columns, expected, expected_checksum=None, zero=False):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError("raw CSV header mismatch")
        rows = list(reader)
    if (len(rows) != 10 or [row["sample"] for row in rows] != list(map(str, range(10))) or
            any(set(row) != set(columns) or any(value is None for value in row.values()) for row in rows)):
        raise ValueError("raw sample coverage mismatch")
    samples = []
    for row in rows:
        if row["correctness"] != "passed" or any(row[key] != str(value) for key, value in expected.items()):
            raise ValueError("raw correctness/counter mismatch")
        try:
            value = float(row["ms"])
            actual_checksum = float(row["checksum"]) if expected_checksum is not None else None
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid numeric raw sample") from error
        if not math.isfinite(value) or (value != 0 if zero else value <= 0):
            raise ValueError("invalid raw timing")
        if expected_checksum is not None and actual_checksum != expected_checksum:
            raise ValueError("independent checksum mismatch")
        samples.append(value)
    return samples


def stats(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
            "mean_ms": statistics.mean(samples), "stddev_ms": statistics.pstdev(samples)}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read planner JSON: " + str(path)) from error


def analyze(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    source = manifest["source"]
    if (manifest["schema_version"] != 1 or manifest["scope"] != "CPU-only Stage 5 planner memory gate" or
            manifest["status"] not in ("measured", "passed") or source["source_dirty"] or
            source != manifest["source_after"] or manifest["configuration"] != CONFIG):
        raise ValueError("invalid source/configuration/completion provenance")
    binary = manifest["binary"]
    if (binary["commit"] != source["commit"] or binary["source_digest"] != source["source_digest"] or binary["source_dirty"] or
            binary["testing"] or binary["build_type"] != "Release" or binary["backend"] != "cpu" or binary["benchmark"] != "planner-v1" or
            len(manifest["binary_sha256"]) != 64):
        raise ValueError("invalid binary provenance")
    if read_json(root / "binary-probe.log") != binary:
        raise ValueError("archived binary probe mismatch")
    if manifest["pairs"] != [{"pair": i, "order": list(POLICIES if i % 2 == 0 else reversed(POLICIES))} for i in range(3)]:
        raise ValueError("invalid pair coverage/order")
    commands = manifest["commands"]
    expected_labels = ["cmake-version", "tests-configure", "tests-build", "ctest", "production-configure", "production-build", "binary-probe", "production-correctness"]
    expected_labels += [f"pair-{pair}-{workload}-{policy}" for pair in range(3) for workload in WORKLOADS for policy in manifest["pairs"][pair]["order"]]
    if [command["log"] for command in commands] != [label + ".log" for label in expected_labels]:
        raise ValueError("invalid build/measurement command coverage")
    for command in commands:
        if command["exit_code"] != 0 or not (root / command["log"]).is_file():
            raise ValueError("invalid command/log provenance")
    for mode, testing in (("tests", True), ("production", False)):
        cache = (root / (mode + "-CMakeCache.txt")).read_text()
        for setting in ("CMAKE_BUILD_TYPE:STRING=Release", "ENABLE_CUDA:BOOL=OFF", "ENABLE_SANITIZERS:BOOL=OFF", "BUILD_TESTING:BOOL=" + ("ON" if testing else "OFF")):
            if setting not in cache.splitlines():
                raise ValueError("build cache configuration mismatch")
        compilation = read_json(root / (mode + "-compile_commands.json"))
        if not isinstance(compilation, list) or not compilation:
            raise ValueError("missing compile command coverage")
        if not testing and any("RUNTIME_TESTING" in command.get("command", " ".join(command.get("arguments", []))) for command in compilation):
            raise ValueError("production compilation contains test hooks")
    benchmark_binary = commands[6]["argv"][0]
    if commands[7]["argv"] != [benchmark_binary, "--self-test"]:
        raise ValueError("production correctness binary mismatch")
    # Archive replay can happen in another checkout/host. Verify the original
    # absolute command paths against their recorded output root, but read raw
    # bytes relative to the currently supplied archive. Never edit old argv.
    first_argv = commands[8]["argv"]
    if len(first_argv) != 9 or not Path(first_argv[6]).is_absolute():
        raise ValueError("invalid recorded measurement path")
    recorded_root = Path(first_argv[6]).parent.parent
    for command in commands[8:]:
        argv = command["argv"]
        if argv[0] != benchmark_binary:
            raise ValueError("measurement binary mismatch")
        label = command["log"].removesuffix(".log") if sys.version_info >= (3, 9) else command["log"][:-4]
        pair, workload, policy = label.split("-")[1:]
        expected_tail = ["--workload", workload, "--policy", policy, "--raw", str(recorded_root / f"pair-{pair}/{workload}-{policy}.csv"),
                         "--prepare-raw", str(recorded_root / f"pair-{pair}/{workload}-{policy}-prepare.csv")]
        if argv[1:] != expected_tail:
            raise ValueError("measurement argv/order mismatch")
    expected_raw = {f"pair-{pair}/{workload}-{policy}{suffix}.csv" for pair in range(3) for workload in WORKLOADS for policy in POLICIES for suffix in ("", "-prepare")}
    if set(manifest["raw_sha256"]) != expected_raw:
        raise ValueError("raw file digest coverage mismatch")
    for filename, digest in manifest["raw_sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("raw digest mismatch")
    rows = []
    ratios = {workload: [] for workload in WORKLOADS}
    capacity_ratios = {}
    for pair in range(3):
        for workload in WORKLOADS:
            medians = {}
            counters = {}
            for policy in POLICIES:
                expected = expected_counters(workload, policy)
                raw = f"pair-{pair}/{workload}-{policy}.csv"
                prepare_raw = f"pair-{pair}/{workload}-{policy}-prepare.csv"
                execute = read_samples(root / raw, COLUMNS, expected, checksum(workload))
                prep_expected = {"batch": 1, "arena_capacity": expected["arena_capacity"], "prepared_backing_allocs": expected["prepared_backing_allocs"]}
                prep = read_samples(root / prepare_raw, PREPARE_COLUMNS, prep_expected, zero=policy == "dynamic")
                medians[policy] = statistics.median(execute)
                counters[policy] = expected
                rows.append({"run_id": manifest["run_id"], "timestamp": manifest["timestamp"], "tested_commit": source["commit"], "source_digest": source["source_digest"],
                             "source_dirty": False, "build_type": "Release", "host": manifest["host"], "compiler": binary["compiler"], "backend": "cpu", "pair": pair,
                             "workload": workload, "policy": policy, "raw": raw, "prepare_raw": prepare_raw, **expected, **stats(execute),
                             **{"prepare_" + key: value for key, value in stats(prep).items()}, "checksum": checksum(workload), "correctness": "passed"})
            ratios[workload].append(medians["dynamic"] / medians["reuse"])
            capacity_ratios[workload] = counters["reuse"]["arena_capacity"] / counters["no_reuse"]["arena_capacity"]
    gate = all(row["intermediate_backing_allocs"] > 0 if row["policy"] == "dynamic" else row["intermediate_backing_allocs"] == 0 for row in rows if row["workload"] == "chain") and capacity_ratios["chain"] <= 0.8
    if not gate:
        raise ValueError("required chain planner memory gate failed")
    return {"source": source, "status": "passed", "scope": manifest["scope"], "memory_gate": "passed", "inplace_status": "skipped_optional", "decoder_status": "unavailable_pending_S13",
            "capacity_ratios": capacity_ratios, "paired_median_ratios": ratios, "median_paired_ratio": {key: statistics.median(values) for key, values in ratios.items()}, "rows": rows}


def save(root):
    root = Path(root).resolve()
    result = analyze(root)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    table = "# Stage 5 CPU-only planner memory gate\n\nMemory gate: passed on predeclared chain. No latency gate, model or CUDA claim. Slower results retained. Ratio = median of three paired dynamic/reuse median-latency ratios. Resident bytes exclude metadata/RSS.\n\n| Workload | Capacity reuse / no-reuse | Paired latency ratios | Median ratio |\n|---|---:|---|---:|\n"
    for workload in WORKLOADS:
        table += f"| {workload} | {result['capacity_ratios'][workload]:.6f} | {', '.join(f'{value:.4f}' for value in result['paired_median_ratios'][workload])} | {result['median_paired_ratio'][workload]:.4f} |\n"
    table += "\nOptional inplace: skipped_optional. Decoder workload: unavailable_pending_S13. Prepare statistics and exact counts: planner.csv / analysis.json.\n"
    for name, content in (("planner.csv", stream.getvalue()), ("analysis.json", json.dumps(result, indent=2) + "\n"), ("summary.md", table)):
        path = root / name
        if path.exists() and path.read_bytes() != content.encode():
            raise ValueError("refusing to overwrite different derived artifact: " + name)
        if not path.exists():
            path.write_bytes(content.encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        result = save(args.run)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print("analyze_planner: " + str(error), file=sys.stderr)
        return 1
    print("Planner memory gate: PASS", result["capacity_ratios"], "latency diagnostics:", result["median_paired_ratio"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
