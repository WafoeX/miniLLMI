#!/usr/bin/env python3
"""S6 clean Release CPU-only scalar baseline + alternating graph policy pairs."""
import argparse
import datetime
import json
import os
import platform
import shlex
import shutil
import sys
from pathlib import Path
from subprocess import STDOUT
from subprocess import run as run_process

from analyze_cpu import (
    BUILD_LABELS,
    CONFIG,
    SCOPE,
    digest,
    measurements,
    read_json,
    save,
)
from provenance import inspect


def vectorization_command(compilation, compiler, build):
    entry = next(c for c in compilation if Path(c["file"]).name == "cpu_scalar.cpp")
    argv = list(entry["arguments"]) if "arguments" in entry else shlex.split(entry["command"])
    position = argv.index("-o")
    argv[position + 1] = str(build / "cpu-scalar-vectorization.o")
    if "Clang" in compiler:
        argv.extend(["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize"])
    elif "GNU" in compiler:
        argv.append("-fopt-info-vec-all")
    else:
        raise RuntimeError("compiler vectorization disclosure currently supports GNU/Clang only")
    return argv, entry["directory"]


def run_benchmark(root, results_root, jobs=4):
    root = Path(root).resolve()
    source = inspect(root)
    if source["source_dirty"]:
        raise RuntimeError("CPU benchmark requires committed clean source; use a clean worktree")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = Path(results_root).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    build_root = root / "build-cpu-benchmark" / run_id
    manifest = {"schema_version": 1, "scope": SCOPE, "run_id": run_id, "timestamp": stamp.isoformat(), "source": source,
                "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
                "configuration": CONFIG, "measurements": measurements(), "recorded_output_root": str(output),
                "timing_boundary": {"gemm": "steady_clock complete Backend::execute incl metadata and finite checks, volatile output[0] read; allocation/input generation/FP64 oracle/copy/CSV excluded",
                                    "graph": "steady_clock execute_graph incl provider validation/begin, backend checks, full oracle scan, volatile output[0] read, output destruction; prepare/input/oracle generation/CSV excluded"},
                "environment_note": "Local CPU-only host permitted by predeclared S6 task. No affinity/exclusive-core guarantee. Ordinary Release compiler optimization including auto-vectorization, no suppression/fast-math. No GPU/oracle comparison or S7 speedup claim.",
                "commands": [], "artifact_sha256": {}, "status": "running"}

    def persist():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def command(argv, label, cwd=None):
        entry = {"argv": list(map(str, argv)), "cwd": str(cwd or root), "log": label + ".log"}
        manifest["commands"].append(entry)
        persist()
        with (output / entry["log"]).open("w") as stream:
            process = run_process(entry["argv"], cwd=entry["cwd"], stdout=stream, stderr=STDOUT)
        entry["exit_code"] = process.returncode
        persist()
        print(label + ": exit=" + str(process.returncode), flush=True)
        if process.returncode:
            raise RuntimeError("command failed; raw log preserved: " + str(output / entry["log"]))
        return output / entry["log"]

    persist()
    print("CPU v0 evidence: " + str(output), flush=True)
    try:
        command(["cmake", "--version"], "cmake-version")
        for mode, testing in (("tests", True), ("production", False)):
            build = build_root / mode
            command(["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DENABLE_SANITIZERS=OFF", "-DBUILD_TESTING=" + ("ON" if testing else "OFF")], mode + "-configure")
            for filename in ("CMakeCache.txt", "compile_commands.json"):
                shutil.copy2(build / filename, output / (mode + "-" + filename))
            command(["cmake", "--build", build, "--parallel", str(jobs), "--verbose"], mode + "-build")
            if testing:
                command(["ctest", "--test-dir", build, "--output-on-failure", "--verbose"], "ctest")
        binary = build_root / "production/bench_cpu"
        identity = read_json(command([binary, "--probe"], "binary-probe"))
        if (any(identity[k] != source[k] for k in source) or identity["build_type"] != "Release" or identity["testing"] or
                identity["backend"] != "cpu" or identity["benchmark"] != "cpu-v1" or identity["algorithm"] != CONFIG["algorithm"] or identity["threads"] != 1):
            raise RuntimeError("stale/dirty/non-Release/test-hook CPU benchmark binary")
        manifest["binary"] = identity
        manifest["binary_sha256"] = digest(binary)
        command([binary, "--self-test"], "production-correctness")
        command(["nm", "-g", binary], "production-symbols")
        argv, cwd = vectorization_command(read_json(output / "production-compile_commands.json"), identity["compiler"], build_root / "production")
        command(argv, "vectorization", cwd)
        for item in measurements():
            command([binary, "--workload", item["workload"], "--policy", item["policy"], "--size", str(item["size"]),
                     "--raw", output / item["raw"], "--correctness", output / item["correctness"]], item["label"])
        manifest["source_after"] = inspect(root)
        if manifest["source_after"] != source:
            raise RuntimeError("source changed during benchmark")
        manifest["artifact_sha256"] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"}
        if [c["log"] for c in manifest["commands"][:len(BUILD_LABELS)]] != [label + ".log" for label in BUILD_LABELS]:
            raise RuntimeError("internal command ordering mismatch")
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
    print("CPU baseline: PASS (baseline only, S7 improvement gate deferred)", flush=True)
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
        run_benchmark(root, args.results_root or root / "results/cpu/baseline", args.jobs)
    except Exception as error:
        print("run_cpu_benchmark: " + str(error), file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
