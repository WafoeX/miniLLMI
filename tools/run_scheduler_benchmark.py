#!/usr/bin/env python3
"""Capture Stage 11 C4: clean T4 Release, three paired fixed-kernel processes."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from provenance import inspect

ROOT = Path(__file__).resolve().parents[1]


def load_analyzer():
    # Standalone CLI tools and file-loaded unit tests share a path-based module,
    # without adding a second package/import-path dependency.
    spec = importlib.util.spec_from_file_location("scheduler_c4_analyzer", ROOT / "tools/analyze_scheduler.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scheduler C4 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze_scheduler = load_analyzer()


def capture(root, result_base, build_base):
    root = Path(root).resolve()
    before = inspect(root)
    if before["source_dirty"]:
        raise ValueError("clean source checkout required")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{os.getpid()}"
    output, builds = Path(result_base).resolve() / run_id, Path(build_base).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    builds.mkdir(parents=True, exist_ok=False)
    testing, production = builds / "testing", builds / "production"
    manifest = {"schema_version": 1, "stage": "stage11-c4", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "run_id": run_id, "status": "running", "platform": platform.platform(), "python": sys.version,
                "source_before": before, "commands": [], "scope": "T4-only fixed 64x64x64; no speedup gate", "build_directory": str(builds)}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    def run(name, argv):
        log = output / (name + ".log")
        item = {"name": name, "argv": [str(x) for x in argv], "log": log.name}
        manifest["commands"].append(item)
        save()
        print(f"[{name}] {' '.join(item['argv'])}", flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(item["argv"], cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        item["exit_code"] = completed.returncode
        save()
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {log}")
        return log.read_text()

    try:
        gpu = run("gpu-query", ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader,nounits"])
        rows = list(csv.reader(gpu.splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 4:
            raise RuntimeError("requires one visible T4 GPU")
        manifest["gpu_name"], capability, manifest["driver_version"], manifest["gpu_memory_mib"] = [x.strip() for x in rows[0]]
        if manifest["gpu_name"] not in ("Tesla T4", "NVIDIA T4") or capability != "7.5":
            raise RuntimeError("C4 evidence requires T4 compute capability 7.5")
        manifest["cuda_architecture"] = "75"
        save()
        run("gpu-info", ["nvidia-smi", "-q"])
        run("nvcc-version", ["nvcc", "--version"])
        run("cmake-version", ["cmake", "--version"])
        for directory, enabled, label in ((testing, "ON", "testing"), (production, "OFF", "production")):
            run("configure-" + label, ["cmake", "-S", root, "-B", directory, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=ON",
                                      "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DBUILD_TESTING=" + enabled])
            run("build-" + label, ["cmake", "--build", directory, "--parallel", "2"])
            if enabled == "ON":
                run("ctest", ["ctest", "--test-dir", directory, "--output-on-failure"])
        executable = production / "bench_scheduler"
        run("probe", [executable, "--probe"])
        probe = analyze_scheduler.load_json(output / "probe.log")
        if probe["source_dirty"] or probe["testing"] or not probe["cuda_enabled"] or probe["build_type"] != "Release" or probe["commit"] != before["commit"] or probe["source_digest"] != before["source_digest"] or probe["cuda_architecture"] != "75" or probe["input_hash"] != analyze_scheduler.input_hash():
            raise RuntimeError("production probe rejected; never time CPU/test/dirty/stale binary")
        run("self-test", [executable, "--self-test"])
        for repeat in (1, 2, 3):
            order = "automatic,manual" if repeat == 2 else "manual,automatic"
            run(f"paired-{repeat}", [executable, "--output", output / f"repeat-{repeat}", "--repeat", str(repeat), "--order", order])
        after = inspect(root)
        manifest["source_after"] = after
        if after != before:
            raise RuntimeError("source changed during capture")
        analyze_scheduler.generate(output, before)
        manifest["analysis"] = {"function": "analyze_scheduler.generate", "status": "passed", "raw_samples": 60, "full_snapshots": 12}
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        try:
            manifest["source_after"] = inspect(root)
            if manifest["source_after"] != before:
                manifest["status"] = "failed"
                manifest["error"] = "source changed during capture"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["source_after_error"] = str(error)
        for directory, label in ((testing, "testing"), (production, "production")):
            for name in ("CMakeCache.txt", "compile_commands.json"):
                if (directory / name).is_file():
                    shutil.copyfile(directory / name, output / (label + "-" + name))
        manifest["artifact_sha256"] = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in sorted(output.rglob("*")) if path.is_file() and path != output / "manifest.json"}
        save()
        print("result directory: " + str(output), flush=True)
    if manifest["status"] != "passed":
        raise RuntimeError(manifest.get("error", "capture finalization failed"))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    capture(args.root, args.root / "results/scheduler/stage11-c4", args.root / "build-stage11-benchmark")
