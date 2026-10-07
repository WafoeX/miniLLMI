#!/usr/bin/env python3
"""Capture the required T4 Nsight evidence for the selected Stage 9 GEMM kernel."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from provenance import inspect

KERNELS = ("v0", "v1")
TOOLS = ("nsys", "ncu")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv):
    if not argv:
        return None
    if len(argv) == 2 and argv[0] == "--artifact-location" and argv[1]:
        return argv[1]
    raise ValueError("usage: run_cuda_gemm_stage10.py [--artifact-location LOCATION]")


def main(argv):
    artifact_location = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("Stage 10 validation requires committed clean source")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = root / "results/profiling/stage10" / run_id
    output.mkdir(parents=True, exist_ok=False)
    build = root / "build-cuda-stage10" / run_id
    manifest = {
        "schema_version": 1,
        "scope": "Stage 10 T4 Nsight v0/selected V1 comparison; profiler timing is not benchmark evidence",
        "run_id": run_id,
        "timestamp": stamp.isoformat(),
        "source": source,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "configuration": {"shape": [4096, 4096, 4096], "warmup": 10, "iterations": 50,
                          "kernels": ["sgemm_v0_naive", "sgemm_v1_tiled"], "tools": list(TOOLS),
                          "reference_cache": "fresh-per-run; full FP64 oracle reused only between profile captures"},
        "artifact_location": artifact_location,
        "commands": [],
        "profiles": [],
        "status": "running",
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

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

    try:
        run("nvidia-smi-before", ["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"])
        gpu_name = (output / "nvidia-smi-before.log").read_text().split(",", 1)[0].strip()
        if "T4" not in gpu_name:
            raise RuntimeError("Stage 10 acceptance requires T4; actual GPU: " + gpu_name)
        manifest["gpu_name"] = gpu_name
        save()
        run("nvcc-version", ["nvcc", "--version"])
        run("nsys-version", ["nsys", "--version"])
        run("ncu-version", ["ncu", "--version"])
        run("configure", ["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=ON",
                          "-DCMAKE_CUDA_ARCHITECTURES=75", "-DBUILD_TESTING=ON"])
        run("build", ["cmake", "--build", build, "--parallel", "2"])
        run("gpu-tests", ["ctest", "--test-dir", build, "--output-on-failure", "-L", "gpu"])
        environment = os.environ.copy()
        environment.update({"BUILD_DIR": str(build), "PROFILE_OUTPUT_ROOT": str(output / "profiles"),
                            "PROFILE_REFERENCE_CACHE_DIR": str(output / "reference-cache")})
        for tool in TOOLS:
            for kernel in KERNELS:
                before = set((output / "profiles" / tool).glob("*")) if (output / "profiles" / tool).exists() else set()
                command = [root / "scripts/profile_gemm.sh", tool, kernel]
                location = None
                if artifact_location:
                    location = artifact_location.rstrip("/") + "/" + tool + "/" + kernel
                    command += ["--artifact-location", location]
                run(f"profile-{tool}-{kernel}", command, environment)
                created = set((output / "profiles" / tool).glob("*")) - before
                if len(created) != 1:
                    raise RuntimeError("expected one unique profile directory for " + tool + "/" + kernel)
                directory = created.pop()
                profile = json.loads((directory / "profile_manifest.json").read_text())
                expected_kernel = "sgemm_v0_naive" if kernel == "v0" else "sgemm_v1_tiled"
                if profile["tool"] != tool or profile["kernel"] != expected_kernel:
                    raise RuntimeError("profile metadata mismatch for " + tool + "/" + kernel)
                manifest["profiles"].append({"tool": tool, "kernel": expected_kernel,
                                             "directory": str(directory.relative_to(output)),
                                             "report": profile["report"], "report_sha256": profile["report_sha256"],
                                             "artifact_location": location})
                save()
        if len(list((output / "reference-cache").glob("*.f32"))) != 1:
            raise RuntimeError("Stage 10 reference cache coverage mismatch")
        run("nvidia-smi-after", ["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"])
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            raise RuntimeError("source changed during Stage 10 validation")
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
        main(sys.argv[1:])
    except Exception as error:
        print("run_cuda_gemm_stage10:", error, file=sys.stderr)
        sys.exit(1)
