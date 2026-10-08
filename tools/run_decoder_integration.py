#!/usr/bin/env python3
"""Capture Stage 13 C4 clean-source CPU and T4 mixed decoder evidence."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from provenance import inspect

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_REL = Path("tests/fixtures/operators-v1/tiny-weights-v1.bin")
LOGITS_REL = Path("tests/fixtures/decoder-v1/logits.bin")
FIXTURE_REL = Path("tests/fixtures/decoder-v1/fixture.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_benchmark(path):
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("decoder benchmark must emit exactly one JSON record")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("decoder benchmark emitted invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("decoder benchmark record must be an object")
    return value


def validate_benchmark(value, mode, source):
    required = {"schema_version", "stage", "mode", "commit", "source_digest", "source_dirty",
                "build_type", "testing", "cuda_enabled", "correctness", "nodes", "execute_allocations",
                "peak_live_bytes", "cpu_capacity_bytes", "cuda_capacity_bytes", "copies", "copy_bytes",
                "backend_dispatches", "backend_switches", "prepare_ms", "warmups", "samples",
                "execute_samples_ms", "shape_change_end_to_end_samples_ms", "execute_median_ms",
                "shape_change_end_to_end_median_ms"}
    missing = required - set(value)
    if missing:
        raise ValueError("decoder benchmark missing fields: " + ",".join(sorted(missing)))
    if value["schema_version"] != 1 or value["stage"] != "stage13-c4" or value["mode"] != mode:
        raise ValueError("decoder benchmark identity mismatch")
    if value["commit"] != source["commit"] or value["source_digest"] != source["source_digest"]:
        raise ValueError("decoder benchmark source provenance mismatch")
    if value["source_dirty"] or value["build_type"] != "Release" or value["testing"]:
        raise ValueError("decoder benchmark requires clean non-testing Release build")
    if not value["cuda_enabled"] or value["correctness"] != "passed" or value["execute_allocations"] != 0:
        raise ValueError("decoder benchmark build/correctness/allocation gate failed")
    if value["nodes"] <= 0 or value["prepare_ms"] <= 0 or value["execute_median_ms"] <= 0 or value["shape_change_end_to_end_median_ms"] <= 0:
        raise ValueError("decoder benchmark metrics must be positive")
    if value["warmups"] != 3 or value["samples"] != 10:
        raise ValueError("decoder benchmark protocol requires 3 warmups and 10 samples")
    sample_fields = (("execute_samples_ms", "execute_median_ms"),
                     ("shape_change_end_to_end_samples_ms", "shape_change_end_to_end_median_ms"))
    for samples_field, median_field in sample_fields:
        values = value[samples_field]
        if not isinstance(values, list) or len(values) != value["samples"]:
            raise ValueError(f"decoder benchmark {samples_field} count mismatch")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or
               not math.isfinite(item) or item <= 0 for item in values):
            raise ValueError(f"decoder benchmark {samples_field} contains invalid latency")
        replayed = statistics.median(values)
        if not math.isclose(value[median_field], replayed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"decoder benchmark {median_field} does not match raw samples")
    if mode == "mixed":
        if value.get("cuda_projection_nodes") != 21 or value.get("inserted_copy_nodes", 0) <= 0:
            raise ValueError("mixed decoder CUDA projection/copy topology mismatch")
        if value["copies"] <= 0 or value["copy_bytes"] <= 0 or value["cuda_capacity_bytes"] <= 0:
            raise ValueError("mixed decoder copy/CUDA memory metrics missing")
    elif value["copies"] != 0 or value["cuda_capacity_bytes"] != 0:
        raise ValueError("CPU decoder unexpectedly reports CUDA activity")


def capture(root, result_base, build_base):
    root = Path(root).resolve()
    before = inspect(root)
    if before["source_dirty"]:
        raise ValueError("clean source checkout required")
    weights, logits, fixture_path = root / WEIGHTS_REL, root / LOGITS_REL, root / FIXTURE_REL
    try:
        fixture = json.loads(fixture_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read decoder fixture metadata") from error
    if fixture["model_contract_version"] != 1 or fixture["source_weights_sha256"] != sha256(weights):
        raise ValueError("decoder fixture weight provenance mismatch")
    if fixture["artifacts"][logits.name] != sha256(logits):
        raise ValueError("decoder logits fixture hash mismatch")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{os.getpid()}"
    output = Path(result_base).resolve() / run_id
    builds = Path(build_base).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    builds.mkdir(parents=True, exist_ok=False)
    testing, production = builds / "testing", builds / "production"
    manifest = {
        "schema_version": 1,
        "stage": "stage13-c4",
        "run_id": run_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "platform": platform.platform(),
        "python": sys.version,
        "source_before": before,
        "fixture": {
            "model_contract_version": fixture["model_contract_version"],
            "generator": fixture["generator"],
            "token_ids": fixture["token_ids"],
            "weights_sha256": sha256(weights),
            "logits_sha256": sha256(logits),
        },
        "scope": "deterministic tiny decoder CPU and T4 mixed-path conformance; latency diagnostic only",
        "commands": [],
        "build_directory": str(builds),
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    def run(name, argv):
        log = output / f"{name}.log"
        item = {"name": name, "argv": [str(value) for value in argv], "log": log.name}
        manifest["commands"].append(item)
        save()
        print(f"[{name}] {' '.join(item['argv'])}", flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(item["argv"], cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        item["exit_code"] = completed.returncode
        save()
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {log}")
        return log

    try:
        query = run("gpu-query", ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader,nounits"])
        fields = [field.strip() for field in query.read_text().strip().split(",")]
        if len(fields) != 4 or fields[0] not in ("Tesla T4", "NVIDIA T4") or fields[1] != "7.5":
            raise RuntimeError("Stage 13 C4 evidence requires one T4 with compute capability 7.5")
        manifest["gpu"] = {"name": fields[0], "compute_capability": fields[1], "driver": fields[2], "memory_mib": fields[3]}
        run("gpu-info", ["nvidia-smi", "-q"])
        run("nvcc-version", ["nvcc", "--version"])
        run("cmake-version", ["cmake", "--version"])
        for directory, enabled, label in ((testing, "ON", "testing"), (production, "OFF", "production")):
            run("configure-" + label, ["cmake", "-S", root, "-B", directory, "-DCMAKE_BUILD_TYPE=Release",
                                         "-DENABLE_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DBUILD_TESTING=" + enabled,
                                         "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"])
            run("build-" + label, ["cmake", "--build", directory, "--parallel", "2"])
        run("ctest-all", ["ctest", "--test-dir", testing, "--output-on-failure"])
        run("ctest-gpu", ["ctest", "--test-dir", testing, "--output-on-failure", "-L", "gpu"])
        run("ctest-decoder", ["ctest", "--test-dir", testing, "--output-on-failure", "-L", "decoder"])
        executable = production / "bench_decoder"
        cpu_log = run("decoder-cpu", [executable, "--weights", weights, "--expected", logits, "--mode", "cpu"])
        mixed_log = run("decoder-mixed", [executable, "--weights", weights, "--expected", logits, "--mode", "mixed"])
        cpu, mixed = load_benchmark(cpu_log), load_benchmark(mixed_log)
        validate_benchmark(cpu, "cpu", before)
        validate_benchmark(mixed, "mixed", before)
        manifest["metrics"] = {"cpu": cpu, "mixed": mixed}
        (output / "metrics.json").write_text(json.dumps(manifest["metrics"], indent=2, allow_nan=False) + "\n")
        after = inspect(root)
        manifest["source_after"] = after
        if after != before:
            raise RuntimeError("source changed during Stage 13 capture")
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
                manifest["error"] = "source changed during Stage 13 capture"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["source_after_error"] = str(error)
        for directory, label in ((testing, "testing"), (production, "production")):
            for name in ("CMakeCache.txt", "compile_commands.json"):
                if (directory / name).is_file():
                    shutil.copyfile(directory / name, output / f"{label}-{name}")
        manifest["artifact_sha256"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json"
        }
        save()
        print("result directory: " + str(output), flush=True)
    if manifest["status"] != "passed":
        raise RuntimeError(manifest.get("error", "capture finalization failed"))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    capture(args.root, args.root / "results/decoder/stage13-c4", args.root / "build-stage13-integration")
