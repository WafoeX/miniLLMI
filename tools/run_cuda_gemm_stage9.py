#!/usr/bin/env python3
"""Capture the three paired T4 Stage 9 v0/v1/cuBLAS validation runs."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from provenance import inspect

SIZES = (512, 1024, 2048, 4096)
ORDERS = ("v0,v1,cublas", "v1,v0,cublas", "v0,v1,cublas")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("Stage 9 validation requires committed clean source")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = root / "results/gemm/stage9" / run_id
    output.mkdir(parents=True, exist_ok=False)
    build = root / "build-cuda-stage9" / run_id
    manifest = {
        "schema_version": 1,
        "scope": "Stage 9 T4 paired v0/v1 tiled/cuBLAS correctness and performance; no Nsight claim",
        "run_id": run_id, "timestamp": stamp.isoformat(), "source": source,
        "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "configuration": {"sizes": SIZES, "warmup": 10, "iterations": 50, "seed": 42,
                          "atol": .001, "rtol": .001, "orders": ORDERS,
                          "candidate": "sgemm_v1_tiled", "baseline": "sgemm_v0_naive"},
        "commands": [], "measurements": [], "status": "running",
    }

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
        run("nvidia-smi-before", ["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"])
        gpu_name = (output / "nvidia-smi-before.log").read_text().split(",", 1)[0].strip()
        if "T4" not in gpu_name:
            raise RuntimeError("Stage 9 acceptance requires T4; actual GPU: " + gpu_name)
        manifest["gpu_name"] = gpu_name
        save()
        run("nvcc-version", ["nvcc", "--version"])
        run("configure", ["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=ON",
                          "-DCMAKE_CUDA_ARCHITECTURES=75", "-DBUILD_TESTING=ON"])
        run("build", ["cmake", "--build", build, "--parallel", "2"])
        run("gpu-tests", ["ctest", "--test-dir", build, "--output-on-failure", "-L", "gpu"])
        run("gpu-info", [build / "gpu_info"])
        binary = build / "bench_gemm"
        for repeat, order in enumerate(ORDERS):
            raw = output / "raw" / f"repeat-{repeat}"
            csv = output / f"repeat-{repeat}.csv"
            benchmark_id = f"{run_id}-r{repeat}"
            manifest["measurements"].append({"repeat": repeat, "order": order, "raw_dir": str(raw.relative_to(output)),
                                             "csv": csv.name, "run_id": benchmark_id})
            run(f"benchmark-{repeat}", [binary, "--experiment", "stage9", "--kernel", "all", "--order", order,
                                          "--sizes", ",".join(map(str, SIZES)), "--warmup", "10", "--iterations", "50",
                                          "--seed", "42", "--atol", "0.001", "--rtol", "0.001", "--run-id", benchmark_id,
                                          "--raw-dir", raw, "--csv", csv])
        run("nvidia-smi-after", ["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"])
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            raise RuntimeError("source changed during Stage 9 validation")
        run("analyze", [sys.executable, root / "tools/analyze_cuda_gemm.py", output])
        manifest["artifact_sha256"] = {str(path.relative_to(output)): digest(path) for path in output.rglob("*")
                                       if path.name != "manifest.json" and path.is_file()}
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
        print("run_cuda_gemm_stage9:", error, file=sys.stderr)
        sys.exit(1)
