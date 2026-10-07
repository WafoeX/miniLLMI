#!/usr/bin/env python3
"""Validate Stage 11 C4 fixed protocol and derive results without an improvement gate."""
import argparse
import csv
import functools
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path

from run_scheduler_validation import normalize_nodes, reject_nonfinite

BOUNDARY = "execute+counts+full_output_check+output_release;synchronous;prepare_oracle_trace_io_excluded"
COLUMNS = ["schema_version", "repeat", "order", "mode", "sample", "ms", "warmup", "samples", "batch", "m", "k", "n",
           "copies", "copy_bytes", "backend_dispatches", "backend_switches", "allocations", "peak_live_bytes", "live_bytes", "cpu_capacity", "cuda_capacity",
           "external_bytes", "oracle_bytes", "workspace_bytes", "runtime_tensor_resident_bytes", "input_hash", "checksum", "commit",
           "source_digest", "source_dirty", "build_type", "compiler", "cuda_compiler", "cuda_architecture", "cpu_backend", "cuda_backend",
           "timing_boundary", "correctness"]
COUNTS = {"copies": 4, "copy_bytes": 65536, "backend_dispatches": 9, "backend_switches": 2, "allocations": 0}
CAPACITIES = {"cpu:0": 49152, "cuda:0": 65536}
COMMANDS = ["gpu-query", "gpu-info", "nvcc-version", "cmake-version", "configure-testing", "build-testing", "ctest",
            "configure-production", "build-production", "probe", "self-test", "paired-1", "paired-2", "paired-3"]


def load_json(path):
    try:
        return json.loads(Path(path).read_text(), parse_constant=reject_nonfinite)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("invalid JSON " + str(path)) from error


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("FP32 conversion failed") from error


@functools.lru_cache(maxsize=1)
def inputs():
    return tuple(tuple(((i * 17 + salt) % 31 - 15) / 32 for i in range(4096)) for salt in (1, 2, 3, 4))


@functools.lru_cache(maxsize=1)
def input_hash():
    value = 14695981039346656037
    for tensor in inputs():
        for element in tensor:
            for byte in struct.pack("<f", element):
                value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


@functools.lru_cache(maxsize=1)
def cpu_reference():
    x, bias, weight, scale = inputs()
    activation = [f32(a + bias[i]) for i, a in enumerate(x)]
    # Independent single-thread FP64 oracle, then exact declared FP32 ADD/MUL.
    product = [f32(sum(activation[row * 64 + k] * weight[k * 64 + col] for k in range(64)))
               for row in range(64) for col in range(64)]
    return tuple(f32(f32(value + value) * scale[i]) for i, value in enumerate(product))


def expected_nodes():
    def inp(i):
        return ("input", i)
    def node(i):
        return ("node", i)
    def copy_to(device):
        return {"overlap": "reject_except_exact_self", "destination": device}
    return [("ADD", {}, [inp(0), inp(1)]), ("COPY", copy_to("cuda:0"), [node(0)]),
            ("COPY", copy_to("cuda:0"), [inp(2)]), ("MATMUL", {}, [node(1), node(2)]),
            ("MATMUL", {}, [node(1), node(2)]), ("COPY", copy_to("cpu:0"), [node(3)]),
            ("COPY", copy_to("cpu:0"), [node(4)]), ("ADD", {}, [node(5), node(6)]), ("MUL", {}, [node(7), inp(8)])]


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("nonfinite or nonnumeric artifact value")
    return value


def snapshot(path):
    data = load_json(path)
    if data["schema_version"] != 1 or data["trace_dropped"] or data["workspace_bytes"] or data["atol"] != .001 or data["rtol"] != .001:
        raise ValueError("snapshot protocol changed")
    if data["counts"] != COUNTS or {c["device"]: c["bytes"] for c in data["device_capacities"]} != CAPACITIES:
        raise ValueError("snapshot counters/capacities changed")
    if len(data["nodes"]) != 9 or len({n["id"] for n in data["nodes"]}) != 9 or any(n["descriptor"]["version"] != 1 for n in data["nodes"]):
        raise ValueError("unexpected graph node protocol")
    if normalize_nodes(data["nodes"]) != expected_nodes():
        raise ValueError("different graph/input dependencies")
    trace = data["trace"]
    dispatch = [e["kind"] for e in trace if e["kind"] in ("BackendCPU", "BackendCUDA")]
    if dispatch != ["BackendCPU"] + ["BackendCUDA"] * 6 + ["BackendCPU"] * 2:
        raise ValueError("actual backend dispatch sequence changed")
    copy_events = [e for e in trace if e["kind"] == "Copy"]
    if len(copy_events) != 4 or any(e["bytes"] != 16384 for e in copy_events):
        raise ValueError("actual copy count/bytes changed")
    if any(e["kind"] in ("Failure", "Allocate") for e in trace):
        raise ValueError("failed execution or backing allocation")
    for kind in ("NodeBegin", "NodeEnd"):
        if [e["node"] for e in trace if e["kind"] == kind] != [n["id"] for n in data["nodes"]]:
            raise ValueError("incomplete node execution trace")
    live = {}
    peak = 0
    requests = releases = 0
    for event in trace:
        if event["kind"] == "BlockAllocate":
            if event["tensor"] in live or event["bytes"] != 16384:
                raise ValueError("invalid graph-owned allocation request")
            live[event["tensor"]] = event["bytes"]
            requests += 1
            peak = max(peak, sum(live.values()))
        elif event["kind"] == "BlockFree":
            if live.pop(event["tensor"], None) != event["bytes"]:
                raise ValueError("invalid graph-owned release")
            releases += 1
    if requests != 9 or releases != 8 or peak != 65536 or sum(live.values()) != 16384:
        raise ValueError("different graph-owned peak/output memory")
    values, oracle = data["output"], data["cpu_reference"]
    expected = cpu_reference()
    if len(values) != 4096 or len(oracle) != 4096:
        raise ValueError("incomplete full output")
    for i, value in enumerate(values):
        value = finite(value)
        if f32(finite(oracle[i])) != expected[i] or abs(value - expected[i]) > .001 + .001 * abs(expected[i]):
            raise ValueError("independent CPU reference/output mismatch")
    return data


def number(row, key):
    try:
        value = float(row[key])
    except (ValueError, TypeError) as error:
        raise ValueError("invalid number " + key) from error
    return finite(value)


def analyze(directory, source):
    directory = Path(directory)
    if source["source_dirty"]:
        raise ValueError("dirty source")
    probe = load_json(directory / "probe.log")
    if probe["benchmark"] != "scheduler-c4-v1" or probe["schema_version"] != 1 or probe["testing"] or not probe["cuda_enabled"] or probe["source_dirty"]:
        raise ValueError("not a clean production CUDA benchmark")
    if probe["commit"] != source["commit"] or probe["source_digest"] != source["source_digest"] or probe["build_type"] != "Release" or probe["cuda_architecture"] != "75" or probe["input_hash"] != input_hash():
        raise ValueError("stale/different probe provenance")
    if not probe["compiler"] or not probe["cuda_compiler"]:
        raise ValueError("missing compiler identity")
    pairs = []
    expected_checksum = sum(cpu_reference())
    for repeat in (1, 2, 3):
        folder = directory / f"repeat-{repeat}"
        wanted = {"samples.csv", "manual-initial.json", "manual-final.json", "automatic-initial.json", "automatic-final.json"}
        if {p.name for p in folder.iterdir()} != wanted:
            raise ValueError("missing/extra pair artifacts")
        with (folder / "samples.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != COLUMNS:
                raise ValueError("CSV schema mismatch")
            rows = list(reader)
        if any(set(row) != set(COLUMNS) or any(value is None for value in row.values()) for row in rows):
            raise ValueError("malformed CSV row")
        order = "automatic,manual" if repeat == 2 else "manual,automatic"
        if len(rows) != 20 or [(r["mode"], r["sample"]) for r in rows] != [(mode, str(i)) for mode in order.split(",") for i in range(10)]:
            raise ValueError("missing, reordered or duplicate samples")
        fixed = {"schema_version": "1", "repeat": str(repeat), "order": order, "warmup": "3", "samples": "10", "batch": "5",
                 "m": "64", "k": "64", "n": "64", "peak_live_bytes": "65536", "live_bytes": "16384", "cpu_capacity": "49152", "cuda_capacity": "65536", "external_bytes": "65536",
                 "oracle_bytes": "16384", "workspace_bytes": "0", "runtime_tensor_resident_bytes": "311296", "input_hash": input_hash(),
                 "commit": source["commit"], "source_digest": source["source_digest"], "source_dirty": "false", "build_type": "Release",
                 "compiler": probe["compiler"], "cuda_compiler": probe["cuda_compiler"], "cuda_architecture": "75",
                 "cpu_backend": "cpu-reference-fp64", "cuda_backend": "cuda-stage0-naive", "timing_boundary": BOUNDARY, "correctness": "passed"}
        fixed.update({key: str(value) for key, value in COUNTS.items()})
        for row in rows:
            if any(row.get(key) != value for key, value in fixed.items()):
                raise ValueError("CSV provenance/control/counter mismatch")
            if number(row, "ms") <= 0 or not math.isclose(number(row, "checksum"), expected_checksum, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError("invalid latency or output checksum")
        medians, dispersion = {}, {}
        for mode in ("manual", "automatic"):
            for phase in ("initial", "final"):
                snapshot(folder / f"{mode}-{phase}.json")
            times = [number(row, "ms") for row in rows if row["mode"] == mode]
            medians[mode] = statistics.median(times)
            dispersion.update({mode + "_min_ms": min(times), mode + "_max_ms": max(times), mode + "_sample_sd_ms": statistics.stdev(times),
                               mode + "_cv_percent": 100 * statistics.stdev(times) / statistics.mean(times)})
        controls = {key: rows[0][key] for key in COLUMNS if key not in ("repeat", "order", "mode", "sample", "ms", "checksum")}
        pairs.append({"repeat": repeat, "order": order, "manual_median_ms": medians["manual"], "automatic_median_ms": medians["automatic"],
                      "manual_over_automatic": medians["manual"] / medians["automatic"], **dispersion, **controls})
    return {"schema_version": 1, "source": source, "shape_mkn": [64, 64, 64], "warmups": 3, "samples": 10, "batch": 5,
            "kernel": "cuda-stage0-naive", "input_hash": input_hash(), "copies": COUNTS,
            "cpu_capacity_bytes": 49152, "cuda_capacity_bytes": 65536, "external_bytes": 65536, "oracle_bytes": 16384,
            "runtime_tensor_resident_bytes": 311296, "peak_live_graph_owned_bytes": 65536, "output_live_bytes": 16384, "workspace_bytes": 0, "timing_boundary": BOUNDARY, "pairs": pairs,
            "median_paired_ratio": statistics.median(p["manual_over_automatic"] for p in pairs),
            "correctness": "passed", "speedup_gate": None, "c3": "skipped_optional"}


def generate(directory, source):
    directory = Path(directory)
    data = analyze(directory, source)
    targets = [directory / name for name in ("analysis.json", "scheduler.csv", "report.md")]
    if any(p.exists() for p in targets):
        raise ValueError("derived artifacts must be new; never overwrite evidence")
    targets[0].write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    with targets[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["pairs"][0]))
        writer.writeheader()
        writer.writerows(data["pairs"])
    lines = ["# Stage 11 C4 — fixed mixed graph paired capture", "", f"Tested source: `{source['commit']}` / `{source['source_digest']}`.",
             "", "64×64×64 FP32, CUDA v0 and identical manual/automatic placement/kernels. Three independent paired processes; order M/A, A/M, M/A; 3 warmup executions, 10 samples × batch 5.",
             "", "| Repeat | Order | Manual median ms | Automatic median ms | Manual/automatic | Manual sample SD ms | Automatic sample SD ms |", "|---|---|---:|---:|---:|---:|---:|"]
    for p in data["pairs"]:
        lines.append(f"| {p['repeat']} | {p['order']} | {p['manual_median_ms']:.9g} | {p['automatic_median_ms']:.9g} | {p['manual_over_automatic']:.6f} | {p['manual_sample_sd_ms']:.9g} | {p['automatic_sample_sd_ms']:.9g} |")
    lines += ["", f"Median paired ratio: **{data['median_paired_ratio']:.6f}×**. No speedup gate; slow runs are retained.",
              "", "Actual execution in both modes: 4 copies / 65536 bytes / 9 backend dispatches / 2 executed switches; zero execute-time intermediate backing allocations.",
              "", "Wall timer includes graph execution, counter validation, full-output oracle scan and output release. Backend calls synchronize before returning. Prepare, input/oracle construction, trace and file I/O are outside timing. This is NOT bare-kernel or model latency.",
              "", "Sample ranges/SD/CV are recorded in analysis.json and scheduler.csv. Colab/shared-host contention and clocks are not controlled; capture-start GPU/driver state is retained in gpu-info.log. No statistical significance claim.",
              "", "Peak live graph-owned bytes 65536 (including copy buffers and pinned output, excluding external inputs); retained output 16384 bytes. BlockAllocate/BlockFree trace events are logical requests/releases, NOT backing allocation calls.",
              "", "Each context reserves CPU 49152 + CUDA 65536 bytes (including copies and pinned output); external host inputs 65536 bytes; oracle 16384 bytes. Both contexts + shared inputs + oracle remain resident: 311296 tensor bytes. Workspace 0. C++ metadata heap, CUDA/cuBLAS context/library overhead and driver memory are not measured by this tensor accounting.",
              "", "All 60 raw sample rows and 12 complete initial/final trace/output snapshots validated, including an independent Python FP64 oracle. Latency ratios describe only explicit manual vs automatic construction/execution with the same kernel and placement; no kernel improvement or placement-benefit claim. C3 segmentation skipped; no switch reduction or end-to-end model claim.",
              "", "Raw samples and snapshots: `repeat-1/`, `repeat-2/`, `repeat-3/`. Source/environment/build provenance and hashes: `manifest.json`.", ""]
    targets[2].write_text("\n".join(lines))
    return data


def verify(directory):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    if manifest["status"] != "passed" or manifest["source_before"] != manifest["source_after"] or manifest["gpu_name"] not in ("Tesla T4", "NVIDIA T4") or manifest["cuda_architecture"] != "75":
        raise ValueError("not an accepted clean T4 capture")
    if [c["name"] for c in manifest["commands"]] != COMMANDS or any(c.get("exit_code") != 0 for c in manifest["commands"]):
        raise ValueError("missing/failed/out-of-order command")
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != directory / "manifest.json"}
    if set(manifest["artifact_sha256"]) != files:
        raise ValueError("incomplete artifact hash coverage")
    required = {name + ".log" for name in COMMANDS} | {"testing-CMakeCache.txt", "production-CMakeCache.txt", "testing-compile_commands.json", "production-compile_commands.json", "analysis.json", "scheduler.csv", "report.md"}
    if not required <= files or not re.search(r"100% tests passed(?:, 0 tests failed)? out of 38(?:\D|$)", (directory / "ctest.log").read_text()):
        raise ValueError("missing build metadata or full fresh 38-test suite")
    for name, sha in manifest["artifact_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
            raise ValueError("artifact hash mismatch: " + name)
    computed = analyze(directory, manifest["source_before"])
    saved = load_json(directory / "analysis.json")
    # Keep hashed derived artifacts unchanged; compare numbers with an explicit
    # round-trip tolerance for Python-version last-bit serialization differences.
    def equal(a, b):
        if type(a) is float and type(b) in (int, float):
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(v, b[i]) for i, v in enumerate(a))
        return a == b
    if not equal(computed, saved):
        raise ValueError("derived analysis replay mismatch")
    return computed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, allow_nan=False))
