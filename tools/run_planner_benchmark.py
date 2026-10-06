#!/usr/bin/env python3
"""Explicit CPU-only S5 memory gate: clean Release, three retained paired runs."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from subprocess import STDOUT
from subprocess import run as run_process

from analyze_planner import CONFIG, POLICIES, WORKLOADS, save
from provenance import inspect


def run_benchmark(root, results_root, jobs=4):
    root = Path(root).resolve()
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("planner benchmark requires committed clean source; use a clean worktree")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = Path(results_root).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    build_root = root / "build-planner-benchmark" / run_id
    manifest = {"schema_version": 1, "scope": "CPU-only Stage 5 planner memory gate", "run_id": run_id, "timestamp": timestamp, "source": source,
                "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
                "configuration": CONFIG, "timing_boundary": "steady_clock execute incl validation/begin/reference/oracle/output destruction; prepare and CSV excluded. Prepare constructor timed separately, destruction excluded.",
                "environment_note": "Named local CPU host; no exclusive-core/process isolation guarantee. Retain slow runs. No CUDA/model claim.",
                "alignment_note": "Prepared policies both 64; dynamic retains S3 malloc alignment. Not an alignment-isolated latency experiment.",
                "commands": [], "pairs": [], "raw_sha256": {}, "status": "running"}

    def persist():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def command(argv, label):
        entry = {"argv": list(map(str, argv)), "cwd": str(root), "log": label + ".log"}
        manifest["commands"].append(entry)
        persist()
        with (output / entry["log"]).open("w") as stream:
            process = run_process(entry["argv"], cwd=root, stdout=stream, stderr=STDOUT)
        entry["exit_code"] = process.returncode
        persist()
        print(label + ": exit=" + str(process.returncode), flush=True)
        if process.returncode:
            raise RuntimeError("command failed; raw log preserved: " + str(output / entry["log"]))
        return output / entry["log"]

    persist()
    print("CPU planner evidence: " + str(output), flush=True)
    try:
        command(["cmake", "--version"], "cmake-version")
        for mode, testing in (("tests", True), ("production", False)):
            build = build_root / mode
            command(["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DENABLE_SANITIZERS=OFF", "-DBUILD_TESTING=" + ("ON" if testing else "OFF")], mode + "-configure")
            for name in ("CMakeCache.txt", "compile_commands.json"):
                shutil.copy2(build / name, output / (mode + "-" + name))
            command(["cmake", "--build", build, "--parallel", str(jobs), "--verbose"], mode + "-build")
            if testing:
                command(["ctest", "--test-dir", build, "--output-on-failure", "--verbose"], "ctest")
        binary = build_root / "production/bench_planner"
        probe = command([binary, "--probe"], "binary-probe")
        try:
            identity = json.loads(probe.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("cannot parse benchmark identity") from error
        if (identity["commit"] != source["commit"] or identity["source_digest"] != source["source_digest"] or identity["source_dirty"] or
                identity["build_type"] != "Release" or identity["testing"] or identity["backend"] != "cpu" or identity["benchmark"] != "planner-v1"):
            raise RuntimeError("stale/dirty/non-Release/test-hook benchmark binary")
        manifest["binary"] = identity
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        command([binary, "--self-test"], "production-correctness")
        for pair in range(3):
            order = list(POLICIES if pair % 2 == 0 else reversed(POLICIES))
            manifest["pairs"].append({"pair": pair, "order": order})
            (output / f"pair-{pair}").mkdir()
            for workload in WORKLOADS:
                for policy in order:
                    raw = output / f"pair-{pair}/{workload}-{policy}.csv"
                    prepare_raw = output / f"pair-{pair}/{workload}-{policy}-prepare.csv"
                    command([binary, "--workload", workload, "--policy", policy, "--raw", raw, "--prepare-raw", prepare_raw], f"pair-{pair}-{workload}-{policy}")
                    for path in (raw, prepare_raw):
                        manifest["raw_sha256"][str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            raise RuntimeError("source changed during benchmark")
        manifest["status"] = "measured"
        persist()
        save(output)
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            manifest["status"] = "failed"
            manifest["error"] = "source changed during benchmark"
        persist()
    if manifest["status"] != "passed":
        raise RuntimeError(manifest["error"])
    print("CPU planner memory gate: PASS (no latency benefit gate)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    try:
        run_benchmark(root, args.results_root or root / "results/planner/cpu", args.jobs)
    except Exception as error:
        print("run_planner_benchmark: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
