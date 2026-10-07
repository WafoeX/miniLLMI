#!/usr/bin/env python3
"""Stage 11 C2 correctness capture, not the dependent C4 latency benchmark."""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from provenance import inspect

CASES = ((17, 13, 11), (64, 64, 64), (0, 3, 2), (2, 0, 3))
KERNELS = ("cuda-stage0-naive", "cuda-stage9-tiled")


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def normalize_nodes(nodes):
    # External logical input IDs are fixed by the declared workload; normalize
    # intermediate IDs by producer position, NEVER by physical storage address.
    tensors = {i: ("input", i) for i in (0, 1, 2, 8)}
    normalized = []
    for index, node in enumerate(nodes):
        desc = node["descriptor"]
        normalized.append((desc["op"], desc["attributes"], [tensors[i] for i in desc["inputs"]]))
        for output in desc["outputs"]:
            if output in tensors:
                raise ValueError("duplicate logical producer in captured graph")
            tensors[output] = ("node", index)
    return normalized


def reject_nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)


def validate_snapshots(directory):
    expected = {f"{kernel}-{m}-{k}-{n}-repeat{repeat}-{mode}.json"
                for kernel in KERNELS for m, k, n in CASES for repeat in (0, 1)
                for mode in ("automatic", "manual")}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError("incomplete or extra C2 snapshot artifacts")
    records = []
    for kernel in KERNELS:
        for m, k, n in CASES:
            for repeat in (0, 1):
                pair = []
                for mode in ("automatic", "manual"):
                    name = f"{kernel}-{m}-{k}-{n}-repeat{repeat}-{mode}.json"
                    try:
                        data = json.loads((directory / name).read_text(), parse_constant=reject_nonfinite)
                    except (OSError, UnicodeError, ValueError) as error:
                        raise ValueError("invalid snapshot " + name + ": " + str(error)) from error
                    if data["schema_version"] != 1 or data["trace_dropped"] or data["workspace_bytes"]:
                        raise ValueError("unsupported/incomplete trace or workspace")
                    if data["atol"] != .001 or data["rtol"] != .001:
                        raise ValueError("changed correctness tolerance")
                    values, oracle = data["output"], data["cpu_reference"]
                    if len(values) != m * n or len(oracle) != len(values):
                        raise ValueError("wrong full-output reference shape")
                    for index, value in enumerate(values):
                        reference = oracle[index]
                        if not math.isfinite(value) or not math.isfinite(reference) or abs(value - reference) > .001 + .001 * abs(reference):
                            raise ValueError("full-output CPU reference mismatch")
                    trace = data["trace"]
                    copies = [e for e in trace if e["kind"] == "Copy"]
                    dispatch = [e["kind"] for e in trace if e["kind"] in ("BackendCPU", "BackendCUDA")]
                    counts = data["counts"]
                    actual = {"copies": len(copies), "copy_bytes": sum(e["bytes"] for e in copies),
                              "backend_dispatches": len(dispatch),
                              "backend_switches": sum(dispatch[i] != current for i, current in enumerate(dispatch[1:])),
                              "allocations": sum(e["kind"] == "Allocate" for e in trace)}
                    if counts != actual:
                        raise ValueError("trace/counter mismatch")
                    wanted_copies = sum(1 for size in (m * k, k * n, m * n, m * n) if size > 0)
                    if counts != {"copies": wanted_copies, "copy_bytes": 4 * (m * k + k * n + 2 * m * n),
                                  "backend_dispatches": 5 + wanted_copies, "backend_switches": 2, "allocations": 0}:
                        raise ValueError("frozen workload execution counts changed")
                    if len(data["nodes"]) != 9 or any(e["kind"] == "Failure" for e in trace):
                        raise ValueError("unexpected graph or execution failure")
                    pair.append(data)
                if pair[0]["counts"] != pair[1]["counts"] or normalize_nodes(pair[0]["nodes"]) != normalize_nodes(pair[1]["nodes"]):
                    raise ValueError("manual/automatic graph protocol mismatch")
                if pair[0]["cpu_reference"] != pair[1]["cpu_reference"]:
                    raise ValueError("different pair oracle")
                records.append({"kernel": kernel, "shape_mkn": [m, k, n], "repeat": repeat,
                                "counts": pair[0]["counts"], "correctness": "passed"})
    return {"scope": "Stage 11 C2 correctness only; not C4 timing acceptance", "pairs": records, "status": "passed"}


def capture(root, cpu_only=False):
    root = Path(root).resolve()
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("requires committed clean source (results excluded); use a clean checkout/worktree")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = root / "results/scheduler/stage11-c2" / run_id
    output.mkdir(parents=True, exist_ok=False)
    build_base = root / "build-stage11-validation" / run_id
    manifest = {"schema_version": 1, "scope": "Stage 11 C1/C2 CPU-only" if cpu_only else "Stage 11 C2 T4 integration correctness",
                "cpu_only": cpu_only, "run_id": run_id, "timestamp": stamp.isoformat(), "source_before": source,
                "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
                "python": sys.version, "commands": [], "status": "running", "c4_status": "not_started"}
    def save():
        dump(output / "manifest.json", manifest)

    def run(label, argv, env=None):
        entry = {"label": label, "argv": list(map(str, argv)), "log": label + ".log"}
        manifest["commands"].append(entry)
        save()
        with (output / entry["log"]).open("w") as stream:
            process = subprocess.run(entry["argv"], cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry["exit_code"] = process.returncode
        save()
        if process.returncode:
            raise RuntimeError("failed command: " + label)
        return (output / entry["log"]).read_text()

    try:
        save()
        architecture = None
        if not cpu_only:
            run("nvidia-smi", ["nvidia-smi"])
            devices = run("gpu-query", ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
            first = devices.strip().splitlines()[0].split(",")
            if len(first) != 2 or first[0].strip() not in ("Tesla T4", "NVIDIA T4") or first[1].strip() != "7.5":
                raise RuntimeError("mandatory acceptance device is T4 CC 7.5; capture preserved but not accepted")
            architecture = first[1].strip().replace(".", "")
            manifest["gpu_name"] = first[0].strip()
            manifest["cuda_architecture"] = architecture
            run("nvcc-version", ["nvcc", "--version"])
        modes = ("Release", "Debug", "Sanitizers") if cpu_only else ("Release",)
        for mode in modes:
            build = build_base / mode.lower()
            command = ["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=" + ("Debug" if mode == "Sanitizers" else mode),
                       "-DBUILD_TESTING=ON", "-DENABLE_CUDA=" + ("OFF" if cpu_only else "ON")]
            if architecture:
                command.append("-DCMAKE_CUDA_ARCHITECTURES=" + architecture)
            if mode == "Sanitizers":
                command.append("-DENABLE_SANITIZERS=ON")
            run(mode.lower() + "-configure", command)
            run(mode.lower() + "-build", ["cmake", "--build", build, "--parallel", "2"])
            env = os.environ.copy()
            if mode == "Sanitizers":
                env["ASAN_OPTIONS"] = "detect_leaks=" + ("0" if sys.platform == "darwin" else "1")
                env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
                manifest["sanitizer_environment"] = {k: env[k] for k in ("ASAN_OPTIONS", "UBSAN_OPTIONS")}
            run(mode.lower() + "-tests", ["ctest", "--test-dir", build, "--output-on-failure", "--verbose"], env)
            if not cpu_only:
                info = run("gpu-info", [build / "gpu_info", "--device", "0"])
                if not re.search(r"GPU Name: (Tesla T4|NVIDIA T4)\n", info) or "Compute Capability: 7.5\n" not in info:
                    raise RuntimeError("CUDA device 0 does not match T4 query")
                run("scheduler-capture", [build / "test_cuda_scheduler", output / "cases"])
                dump(output / "validation.json", validate_snapshots(output / "cases"))
        after = inspect(root)
        if after != source:
            raise RuntimeError("source changed during validation")
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        try:
            manifest["source_after"] = inspect(root)
        except Exception as error:
            manifest["source_after_error"] = str(error)
        manifest["artifact_sha256"] = hashes(output)  # failures and partial captures also retained
        save()
        print(output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    capture(Path(__file__).resolve().parents[1], args.cpu_only)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("run_scheduler_validation:", error, file=sys.stderr)
        sys.exit(1)
