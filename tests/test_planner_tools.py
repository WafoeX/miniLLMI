#!/usr/bin/env python3
"""Mock-only temporary analyzer/runner tests; never real performance evidence."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("analyze_planner")
runner = load("run_planner_benchmark")
validation = load("validate_tensor")
SOURCE = {"commit": "mock-only-commit", "source_digest": "mock-only-digest", "source_dirty": False}
BINARY = {"schema_version": 1, **SOURCE, "build_type": "Release", "testing": False, "backend": "cpu", "compiler": "mock-only", "benchmark": "planner-v1"}


def write_raw(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class PlannerTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest = {"schema_version": 1, "scope": "CPU-only Stage 5 planner memory gate", "status": "measured", "source": SOURCE,
                         "source_after": SOURCE, "configuration": analysis.CONFIG, "binary": BINARY, "binary_sha256": "0" * 64,
                         "commands": [], "raw_sha256": {}, "pairs": [], "run_id": "mock-only", "timestamp": "mock-only", "host": "mock-only"}
        for label in ("cmake-version", "tests-configure", "tests-build", "ctest", "production-configure", "production-build", "binary-probe", "production-correctness"):
            self.command(["mock-only"], label)
        (self.root / "binary-probe.log").write_text(json.dumps(BINARY))
        self.manifest["commands"][6]["argv"] = ["mock-only", "--probe"]
        self.manifest["commands"][7]["argv"] = ["mock-only", "--self-test"]
        for mode in ("tests", "production"):
            (self.root / (mode + "-CMakeCache.txt")).write_text("CMAKE_BUILD_TYPE:STRING=Release\nENABLE_CUDA:BOOL=OFF\nENABLE_SANITIZERS:BOOL=OFF\nBUILD_TESTING:BOOL=" + ("ON" if mode == "tests" else "OFF") + "\n")
            (self.root / (mode + "-compile_commands.json")).write_text(json.dumps([{"command": "mock-only-cxx"}]))
        for pair in range(3):
            order = list(analysis.POLICIES if pair % 2 == 0 else reversed(analysis.POLICIES))
            self.manifest["pairs"].append({"pair": pair, "order": order})
            for workload in analysis.WORKLOADS:
                for policy in order:
                    expected = analysis.expected_counters(workload, policy)
                    raw = self.root / f"pair-{pair}/{workload}-{policy}.csv"
                    prep = self.root / f"pair-{pair}/{workload}-{policy}-prepare.csv"
                    write_raw(raw, analysis.COLUMNS, [{"sample": i, "ms": (2 if policy == "dynamic" else 4) * (1 + i / 10), **expected,
                              "checksum": analysis.checksum(workload), "correctness": "passed"} for i in range(10)])
                    write_raw(prep, analysis.PREPARE_COLUMNS, [{"sample": i, "ms": 0 if policy == "dynamic" else 1, "batch": 1,
                              "arena_capacity": expected["arena_capacity"], "prepared_backing_allocs": expected["prepared_backing_allocs"], "correctness": "passed"} for i in range(10)])
                    for path in (raw, prep):
                        self.digest(path)
                    self.command(["mock-only", "--workload", workload, "--policy", policy, "--raw", str(raw), "--prepare-raw", str(prep)], f"pair-{pair}-{workload}-{policy}")
        self.persist()

    def command(self, argv, label):
        (self.root / (label + ".log")).write_text("mock-only command\n")
        self.manifest["commands"].append({"argv": argv, "exit_code": 0, "log": label + ".log"})

    def persist(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def digest(self, path):
        self.manifest["raw_sha256"][str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def corrupt(self, column, value, prep=False):
        path = self.root / ("pair-0/chain-reuse-prepare.csv" if prep else "pair-0/chain-reuse.csv")
        with path.open(newline="") as stream:
            data = list(csv.DictReader(stream))
        data[0][column] = value
        write_raw(path, analysis.PREPARE_COLUMNS if prep else analysis.COLUMNS, data)
        self.digest(path)
        self.persist()

    def test_slower_runs_retained_and_memory_gate(self):
        result = analysis.save(self.root)
        self.assertEqual(result["memory_gate"], "passed")
        self.assertEqual(result["median_paired_ratio"], dict.fromkeys(analysis.WORKLOADS, 0.5))
        self.assertEqual(result["capacity_ratios"], {"chain": 1 / 6, "diamond": 0.6})
        self.assertEqual(len(result["rows"]), 18)
        before = (self.root / "planner.csv").read_bytes()
        analysis.save(self.root)
        self.assertEqual(before, (self.root / "planner.csv").read_bytes())

    def test_invalid_counters(self):
        for key in ("backing_allocs", "intermediate_backing_allocs", "inplace", "arena_capacity", "peak_resident_bytes"):
            self.corrupt(key, "99")
            with self.assertRaisesRegex(ValueError, "counter"):
                analysis.analyze(self.root)
            self.corrupt(key, str(analysis.expected_counters("chain", "reuse")[key]))

    def test_invalid_timings(self):
        for value in ("nan", "inf", "-1", "0", "not-a-number"):
            self.corrupt("ms", value)
            with self.assertRaises(ValueError):
                analysis.analyze(self.root)

    def test_prepare_invalid_or_missing(self):
        self.corrupt("ms", "nan", prep=True)
        with self.assertRaises(ValueError):
            analysis.analyze(self.root)
        (self.root / "pair-0/chain-reuse-prepare.csv").unlink()
        with self.assertRaises(OSError):
            analysis.analyze(self.root)

    def test_independent_checksum(self):
        self.corrupt("checksum", "99")
        with self.assertRaisesRegex(ValueError, "checksum"):
            analysis.analyze(self.root)

    def test_missing_sample(self):
        path = self.root / "pair-0/chain-reuse.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        self.digest(path)
        self.persist()
        with self.assertRaisesRegex(ValueError, "coverage"):
            analysis.analyze(self.root)

    def test_raw_digest(self):
        path = self.root / "pair-0/chain-reuse.csv"
        path.write_text(path.read_text().replace(",4.0,", ",4.1,"))
        with self.assertRaisesRegex(ValueError, "digest"):
            analysis.analyze(self.root)

    def test_source_or_binary_rejected(self):
        for key, value in (("source_dirty", True), ("testing", True), ("build_type", "Debug"), ("source_digest", "stale")):
            self.manifest["binary"] = {**BINARY, key: value}
            self.persist()
            with self.assertRaisesRegex(ValueError, "binary"):
                analysis.analyze(self.root)
        self.manifest["binary"] = BINARY
        self.manifest["source_after"] = {**SOURCE, "source_digest": "changed"}
        self.persist()
        with self.assertRaisesRegex(ValueError, "provenance"):
            analysis.analyze(self.root)

    def test_order_commands_logs(self):
        self.manifest["commands"][-1]["argv"][2] = "chain"
        self.persist()
        with self.assertRaisesRegex(ValueError, "argv"):
            analysis.analyze(self.root)
        self.manifest["commands"][-1]["argv"][2] = "diamond"
        self.persist()
        (self.root / "production-build.log").unlink()
        with self.assertRaisesRegex(ValueError, "log"):
            analysis.analyze(self.root)

    def test_production_build_flags(self):
        path = self.root / "production-CMakeCache.txt"
        original = path.read_text()
        path.write_text(original.replace("Release", "Debug"))
        with self.assertRaisesRegex(ValueError, "cache"):
            analysis.analyze(self.root)
        path.write_text(original)
        (self.root / "production-compile_commands.json").write_text(json.dumps([{"command": "mock-only -DRUNTIME_TESTING=1"}]))
        with self.assertRaisesRegex(ValueError, "hooks"):
            analysis.analyze(self.root)

    def test_probe_disagrees(self):
        (self.root / "binary-probe.log").write_text(json.dumps({**BINARY, "compiler": "other"}))
        with self.assertRaisesRegex(ValueError, "probe"):
            analysis.analyze(self.root)

    def test_invalid_json(self):
        (self.root / "manifest.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "JSON"):
            analysis.analyze(self.root)

    def test_refuse_overwrite(self):
        (self.root / "planner.csv").write_text("original\n")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            analysis.save(self.root)
        self.assertEqual((self.root / "planner.csv").read_text(), "original\n")

    def test_dirty_runner_refused(self):
        with patch.object(runner, "inspect", return_value={**SOURCE, "source_dirty": True}), self.assertRaisesRegex(RuntimeError, "clean"):
            runner.run_benchmark(self.root, self.root / "evidence")
        self.assertFalse((self.root / "evidence").exists())

    def test_runner_failure_preserved(self):
        def fake(argv, **kwargs):
            kwargs["stdout"].write("mock-only build evidence\n")
            if "-B" in argv:
                directory = Path(argv[argv.index("-B") + 1])
                directory.mkdir(parents=True)
                for name in ("CMakeCache.txt", "compile_commands.json"):
                    (directory / name).write_text("mock-only\n")
            return subprocess.CompletedProcess(argv, 7 if "--build" in argv else 0)
        with patch.object(runner, "inspect", return_value=SOURCE), patch.object(runner, "run_process", side_effect=fake), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "command failed"):
            runner.run_benchmark(self.root, self.root / "evidence")
        output, = (self.root / "evidence").iterdir()
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["commands"][-1]["exit_code"], 7)
        self.assertIn("mock-only", (output / "tests-build.log").read_text())

    def test_stage5_shared_validation_config(self):
        self.assertEqual(validation.configuration(5), ("planner", "check_planner_sanitizers"))


if __name__ == "__main__":
    unittest.main()
