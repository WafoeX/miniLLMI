#!/usr/bin/env python3
"""Recompute Stage 4 CPU-only allocator evidence; reject incomplete/corrupt runs."""
import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path

WORKLOADS = ("synthetic", "chain", "diamond")
POLICIES = ("dynamic", "arena")
COLUMNS = ("sample", "ms", "batch", "requests", "releases", "backing_allocs", "backing_frees", "peak_payload_bytes", "output_live_bytes", "reused", "arena_capacity", "external_bytes", "peak_backing_bytes", "prepared_backing_allocs", "checksum", "correctness")
EXPECTED = {"synthetic": (48, 48, 640, 0, 0, 44), "chain": (12, 11, 512, 256, 512, 10), "diamond": (5, 4, 768, 256, 768, 2)}


def analyze(root):
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read allocator manifest: " + str(error)) from error
    source = manifest["source"]
    if (manifest["schema_version"] != 1 or manifest["scope"] != "CPU-only allocator diagnostic" or manifest["status"] not in ("measured", "passed")
            or source["source_dirty"] or source != manifest["source_after"] or manifest["configuration"] != {"runs": 3, "samples": 10, "warmup": 3, "batch": 20, "alignment": 64, "arena_capacity": 1024, "workload_version": 1}):
        raise ValueError("invalid source/configuration/completion provenance")
    binary = manifest["binary"]
    if (binary["commit"] != source["commit"] or binary["source_digest"] != source["source_digest"] or binary["source_dirty"] or binary["testing"]
            or binary["build_type"] != "Release" or binary["backend"] != "cpu" or any(command["exit_code"] != 0 for command in manifest["commands"])):
        raise ValueError("invalid binary/build/command provenance")
    if manifest["pairs"] != [{"pair": run, "order": ["dynamic", "arena"] if run % 2 == 0 else ["arena", "dynamic"]} for run in range(3)]:
        raise ValueError("invalid pair coverage/order")
    summaries = []
    checksums = {}
    ratios = {workload: [] for workload in WORKLOADS}
    for run in range(3):
        for workload in WORKLOADS:
            medians = {}
            for policy in POLICIES:
                path = root / f"pair-{run}/{workload}-{policy}.csv"
                with path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    if tuple(reader.fieldnames or ()) != COLUMNS:
                        raise ValueError("raw CSV header mismatch")
                    rows = list(reader)
                if len(rows) != 10 or any(set(row) != set(COLUMNS) or any(value is None for value in row.values()) for row in rows) or [row["sample"] for row in rows] != list(map(str, range(10))):
                    raise ValueError("raw sample coverage mismatch")
                requests, releases, peak, live, external, reuse = EXPECTED[workload]
                expected = {"batch": 20, "requests": requests, "releases": releases, "backing_allocs": requests if policy == "dynamic" else 0,
                            "backing_frees": releases if policy == "dynamic" else 0, "peak_payload_bytes": peak, "output_live_bytes": live,
                            "reused": reuse if policy == "arena" else 0, "arena_capacity": 1024 if policy == "arena" else 0,
                            "external_bytes": external, "peak_backing_bytes": external + (1024 if policy == "arena" else peak),
                            "prepared_backing_allocs": 1 if policy == "arena" else 0}
                for row in rows:
                    if row["correctness"] != "passed" or any(row[key] != str(value) for key, value in expected.items()):
                        raise ValueError("raw correctness/counter mismatch")
                try:
                    samples = [float(row["ms"]) for row in rows]
                    values = [float(row["checksum"]) for row in rows]
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError("invalid numeric raw sample") from error
                if any(not math.isfinite(value) or value <= 0 for value in samples) or any(not math.isfinite(value) for value in values) or len(set(values)) != 1:
                    raise ValueError("invalid raw timings/checksums")
                if workload in checksums and checksums[workload] != values[0]:
                    raise ValueError("paired checksum mismatch")
                checksums[workload] = values[0]
                median = statistics.median(samples)
                medians[policy] = median
                summaries.append({"run_id": manifest["run_id"], "timestamp": manifest["timestamp"], "tested_commit": source["commit"], "source_digest": source["source_digest"],
                                  "source_dirty": False, "build_type": "Release", "host": manifest["host"], "compiler": binary["compiler"], "backend": "cpu", "pair": run,
                                  "workload": workload, "policy": policy, "raw": str(path.relative_to(root)), **expected,
                                  "median_ms": median, "min_ms": min(samples), "max_ms": max(samples), "mean_ms": statistics.mean(samples), "stddev_ms": statistics.pstdev(samples),
                                  "checksum": values[0], "correctness": "passed"})
            ratios[workload].append(medians["dynamic"] / medians["arena"])
    result = {"source": source, "status": "passed", "scope": "CPU-only diagnostic; no planner or latency improvement gate", "paired_median_ratios": ratios,
              "median_paired_ratio": {workload: statistics.median(values) for workload, values in ratios.items()}, "rows": summaries}
    return result


def save(root):
    root = Path(root)
    result = analyze(root)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    table = "# Stage 4 CPU-only allocation diagnostic\n\nNo latency benefit gate or planner claim. Slower results are retained. Ratios are median across three independent paired median-latency ratios (dynamic / arena).\n\n| Workload | Paired ratios | Median ratio |\n|---|---|---:|\n"
    for workload in WORKLOADS:
        table += f"| {workload} | {', '.join(f'{value:.4f}' for value in result['paired_median_ratios'][workload])} | {result['median_paired_ratio'][workload]:.4f} |\n"
    for name, content in (("allocator.csv", stream.getvalue()), ("analysis.json", json.dumps(result, indent=2) + "\n"), ("summary.md", table)):
        path = root / name
        if path.exists():
            if path.read_bytes() != content.encode():
                raise ValueError("refusing to overwrite existing different derived artifact: " + name)
        else:
            path.write_bytes(content.encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        result = save(args.run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("analyze_allocator: " + str(error), file=sys.stderr)
        return 1
    print("Allocator analysis: PASS", result["median_paired_ratio"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
