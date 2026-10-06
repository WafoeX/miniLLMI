#!/usr/bin/env python3
"""CPU-only Stage 1 correctness evidence; no timings/benchmark claims."""
import argparse
import datetime
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from subprocess import STDOUT
from subprocess import run as run_process

from provenance import inspect


def configuration(stage):
    configurations = {1: ("tensor", "check_tensor_sanitizers"),
                      2: ("operators", "check_operator_sanitizers"),
                      3: ("graph", "check_graph_sanitizers"),
                      4: ("allocator", "check_arena_sanitizers"),
                      5: ("planner", "check_planner_sanitizers"),
                      6: ("cpu", "check_cpu_backend_sanitizers")}
    if stage not in configurations:
        raise ValueError("only CPU Stage 1/2/3/4/5/6 validation is supported")
    return configurations[stage]


def validate(root, results_root, jobs, allow_dirty, stage=1):
    _, sanitizer_target = configuration(stage)
    source = inspect(root)
    if source["source_dirty"] and not allow_dirty:
        raise RuntimeError("Commit source first (including untracked source), or use --allow-dirty for labelled development evidence")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + str(os.getpid())
    output = results_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    build_root = root / ("build-stage" + str(stage) + "-validation") / run_id
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    # Apple ASan does not provide LSan; never claim a leak-sanitizer pass there.
    env.setdefault("ASAN_OPTIONS", "halt_on_error=1:detect_leaks=" + ("0" if sys.platform == "darwin" else "1"))
    env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
    manifest = {
        "schema_version": 1, "stage": stage, "run_id": run_id, "scope": "CPU-only correctness, no performance claim",
        "source": source, "host": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "sanitizer_environment": {key: env[key] for key in ("ASAN_OPTIONS", "UBSAN_OPTIONS")},
        "commands": [], "status": "running",
    }
    (output / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def run(command, label):
        log = output / (label + ".log")
        entry = {"argv": [str(arg) for arg in command], "cwd": str(root), "log": log.name}
        manifest["commands"].append(entry)
        save()
        with log.open("w", encoding="utf-8") as stream:
            # A local callable also makes runner-only mocks independent of platform/provenance subprocesses.
            process = run_process(entry["argv"], cwd=root, env=env, stdout=stream, stderr=STDOUT)
        entry["exit_code"] = process.returncode
        save()
        print(label + ": exit=" + str(process.returncode), flush=True)
        if process.returncode:
            raise RuntimeError("Command failed; preserved full log: " + str(log))

    save()
    print("Evidence: " + str(output), flush=True)
    try:
        run(["cmake", "--version"], "cmake-version")
        run(["git", "--version"], "git-version")
        for mode, build_type, sanitizer, testing in (
            ("release", "Release", False, True), ("debug", "Debug", False, True),
            ("sanitizer", "Debug", True, True), ("production", "Release", False, False),
        ):
            build = build_root / mode
            run(["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=" + build_type,
                 "-DENABLE_CUDA=OFF", "-DENABLE_SANITIZERS=" + ("ON" if sanitizer else "OFF"),
                 "-DBUILD_TESTING=" + ("ON" if testing else "OFF")], mode + "-configure")
            for filename in ("CMakeCache.txt", "compile_commands.json"):
                shutil.copy2(build / filename, output / (mode + "-" + filename))
            run(["cmake", "--build", build, "--parallel", str(jobs), "--verbose"], mode + "-build")
            if testing:
                if sanitizer:
                    run(["cmake", "--build", build, "--target", sanitizer_target], "sanitizer-target")
                run(["ctest", "--test-dir", build, "--output-on-failure", "--verbose"], mode + "-ctest")
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        after = inspect(root)
        manifest["source_after"] = after
        if after != source:
            manifest["status"] = "failed"
            manifest["error"] = "Source provenance changed during validation"
        save()
    if manifest["status"] != "passed":
        raise RuntimeError(manifest.get("error", "validation failed"))
    print("Stage " + str(stage) + " CPU validation: PASS (Release/Debug/ASan+UBSan/production build)", flush=True)
    return output


def main(stage=1):
    family, _ = configuration(stage)
    parser = argparse.ArgumentParser(description="CPU-only Stage " + str(stage) + " correctness evidence; no performance claims")
    parser.add_argument("--results-root", type=Path, help="Defaults to results/" + family + "/local under repository")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--allow-dirty", action="store_true", help="Development evidence only; records source_dirty=true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    try:
        validate(root, (args.results_root or root / "results" / family / "local").resolve(), args.jobs, args.allow_dirty, stage)
    except Exception as error:
        print("validate_" + family + ": " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
