#!/usr/bin/env python3
"""Stage 7 CPU-only paired GEMM evidence runner (v0, C1 ikj, C3 FIFO pool)."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from provenance import inspect

SIZES = (128, 256, 512, 1024)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("requires committed clean source")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    out = root / "results/cpu/parallel" / run_id
    out.mkdir(parents=True)
    build = root / "build-cpu-parallel" / run_id
    available = min(os.cpu_count() or 1, 8)
    pool_threads = tuple(dict.fromkeys((1, 2, 4, available)))
    measurements = []
    for repeat in range(3):
        ordered = ("v0", "ikj") if repeat % 2 == 0 else ("ikj", "v0")
        order_sizes = SIZES if repeat % 2 == 0 else tuple(reversed(SIZES))
        for algorithm in ordered:
            for size in order_sizes:
                measurements.append((repeat, algorithm, 1, size))
        for threads in pool_threads:
            for size in order_sizes:
                measurements.append((repeat, "fifo", threads, size))
    manifest = {
        "schema_version": 1,
        "scope": "CPU-only Stage 7 local scaling; no GPU comparison",
        "run_id": run_id,
        "timestamp": stamp.isoformat(),
        "source": source,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "configuration": {"sizes": SIZES, "warmup": 3, "samples": 10, "seed": 42,
                          "atol": .001, "rtol": .001, "pool_threads": pool_threads,
                          "available_threads_capped": available},
        "measurements": [],
        "commands": [],
        "status": "running",
        "recorded_output_root": str(out),
    }

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def command(argv, label, cwd=root):
        log = out / (label + ".log")
        entry = {"argv": list(map(str, argv)), "cwd": str(cwd), "log": log.name}
        manifest["commands"].append(entry)
        save()
        with log.open("w") as stream:
            process = subprocess.run(entry["argv"], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        entry["exit_code"] = process.returncode
        save()
        if process.returncode:
            raise RuntimeError("failed command: " + label)

    try:
        command(["cmake", "-S", root, "-B", build / "tests", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DBUILD_TESTING=ON"], "tests-configure")
        command(["cmake", "--build", build / "tests", "--parallel", "4"], "tests-build")
        command(["ctest", "--test-dir", build / "tests", "--output-on-failure"], "ctest")
        command(["cmake", "-S", root, "-B", build / "production", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DBUILD_TESTING=OFF"], "production-configure")
        command(["cmake", "--build", build / "production", "--parallel", "4"], "production-build")
        binary = build / "production/bench_cpu"
        probe = out / "binary-probe.json"
        with probe.open("w") as stream:
            process = subprocess.run([binary, "--probe"], stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError("failed binary probe")
        manifest["binary"] = json.loads(probe.read_text())
        manifest["binary_sha256"] = sha(binary)
        command([binary, "--self-test"], "production-correctness")
        for repeat, algorithm, threads, size in measurements:
            label = f"r{repeat}-{algorithm}-t{threads}-{size}"
            raw = label + ".csv"
            correctness = label + ".json"
            manifest["measurements"].append({"repeat": repeat, "algorithm": algorithm, "threads": threads,
                                             "size": size, "raw": raw, "correctness": correctness})
            command([binary, "--workload", "gemm", "--policy", "caller", "--size", str(size),
                     "--algorithm", algorithm, "--threads", str(threads), "--raw", out / raw,
                     "--correctness", out / correctness], label)
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            raise RuntimeError("source changed during benchmark")
        manifest["artifact_sha256"] = {path.name: sha(path) for path in out.iterdir() if path.name != "manifest.json"}
        manifest["status"] = "passed"
        save()
        print(out)
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
        print("run_cpu_parallel_benchmark:", error, file=sys.stderr)
        sys.exit(1)
