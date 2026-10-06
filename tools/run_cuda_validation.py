#!/usr/bin/env python3
"""Capture a clean T4 Stage 8 Release build/test validation artifact."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from provenance import inspect


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("CUDA validation requires committed clean source")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = root / "results/cuda/validation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    build = root / "build-cuda-validation" / run_id
    manifest = {"schema_version": 1, "scope": "Stage 8 T4 CUDA integration validation; no performance claim",
                "run_id": run_id, "timestamp": stamp.isoformat(), "source": source, "host": platform.node(),
                "platform": platform.platform(), "machine": platform.machine(), "commands": [], "status": "running"}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(label, argv):
        entry = {"label": label, "argv": list(map(str, argv)), "log": label + ".log"}
        manifest["commands"].append(entry)
        save()
        with (output / entry["log"]).open("w") as stream:
            process = subprocess.run(entry["argv"], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        entry["exit_code"] = process.returncode
        save()
        if process.returncode:
            raise RuntimeError("failed command: " + label)

    try:
        run("nvidia-smi", ["nvidia-smi"])
        run("nvcc-version", ["nvcc", "--version"])
        run("configure", ["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=ON",
                              "-DCMAKE_CUDA_ARCHITECTURES=75", "-DBUILD_TESTING=ON"])
        run("build", ["cmake", "--build", build, "--parallel", "2"])
        run("gpu-tests", ["ctest", "--test-dir", build, "--output-on-failure", "-L", "gpu"])
        run("gpu-info", [build / "gpu_info"])
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            raise RuntimeError("source changed during CUDA validation")
        manifest["artifact_sha256"] = {path.name: digest(path) for path in output.iterdir() if path.name != "manifest.json"}
        manifest["status"] = "passed"
        save()
        print(output)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["source_after"] = inspect(root)
        save()
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("run_cuda_validation:", error, file=sys.stderr)
        sys.exit(1)
