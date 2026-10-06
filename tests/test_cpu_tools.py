#!/usr/bin/env python3
"""Mock-only temporary S6 analyzer/runner tests, never performance evidence."""
import contextlib
import csv
import importlib.util
import io
import json
import shutil
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
        raise RuntimeError("cannot load tool module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("analyze_cpu")
runner = load("run_cpu_benchmark")

SOURCE = {"commit": "a" * 40, "source_digest": "b" * 64, "source_dirty": False}
BINARY = {"schema_version": 1, **SOURCE, "build_type": "Release", "testing": False, "backend": "cpu", "compiler": "AppleClang MOCK-ONLY", "benchmark": "cpu-v1", "algorithm": "cpu-ijk-fp32-v0", "threads": 1}


def mock_measurement(raw, correctness, workload, policy, size):
    batch = 1 if workload == "gemm" else 20
    expected = {"workload": workload, "policy": policy, "algorithm": "cpu-ijk-fp32-v0", "m": size, "n": size, "k": size,
                "seed": 42, "input_hash": "0123456789abcdef", "batch": batch, "threads": 1}
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=analysis.COLUMNS)
        writer.writeheader()
        for sample in range(10):
            writer.writerow({"sample": sample, "ms": (1 if policy != "reuse" else 2) + sample / 100, "checksum": 0.25, **expected})
    metrics = {"max_abs": 0, "mean_abs": 0, "max_relative": 0, "violations": 0, "nonfinite": 0, "passed": True}
    correctness.write_text(json.dumps({**expected, "schema_version": 1, "atol": 0.001, "rtol": 0.001, "warmup": 3, "samples": 10,
                                      "oracle": "unchanged-stage0-fp64-untimed", "initial": metrics, "final": metrics, "status": "passed"}))


class CpuTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "source"
        self.root.mkdir()
        self.results = Path(self.temp.name).resolve() / "mock-only-results"
        self.output = self.run_mock()
        self.manifest = analysis.read_json(self.output / "manifest.json")

    def fake_run(self, argv, cwd, stdout, stderr):
        argv = list(map(str, argv))
        stdout.write("MOCK-ONLY fixture, NOT performance evidence\n")
        if argv[:2] == ["cmake", "-S"]:
            build = Path(argv[argv.index("-B") + 1])
            build.mkdir(parents=True)
            testing = "-DBUILD_TESTING=ON" in argv
            (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\nENABLE_CUDA:BOOL=OFF\nENABLE_SANITIZERS:BOOL=OFF\nBUILD_TESTING:BOOL=" + ("ON" if testing else "OFF") + "\n")
            (build / "compile_commands.json").write_text(json.dumps([{"file": str(self.root / "src/runtime/cpu_scalar.cpp"), "directory": str(build), "command": "mock-cxx -O3 -DNDEBUG -o scalar.o -c " + str(self.root / "src/runtime/cpu_scalar.cpp")}]))
            (build / "bench_cpu").write_bytes(b"mock-only-binary")
        elif "--probe" in argv:
            stdout.seek(0)
            stdout.truncate()
            stdout.write(json.dumps(BINARY))
        elif "--workload" in argv:
            mock_measurement(Path(argv[argv.index("--raw") + 1]), Path(argv[argv.index("--correctness") + 1]), argv[argv.index("--workload") + 1], argv[argv.index("--policy") + 1], int(argv[argv.index("--size") + 1]))
        return subprocess.CompletedProcess(argv, 0)

    def run_mock(self, source=SOURCE, side_effect=None):
        with patch.object(runner, "inspect", return_value=source), patch.object(runner, "run_process", side_effect=side_effect or self.fake_run), contextlib.redirect_stdout(io.StringIO()):
            return runner.run_benchmark(self.root, self.results)

    def persist(self):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest))

    def rehash(self, name):
        self.manifest["artifact_sha256"][name] = analysis.digest(self.output / name)
        self.persist()

    def assert_rejected(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            analysis.analyze(self.output)

    def test_baseline_raw_statistics_and_slower_graph_retained(self):
        result = analysis.analyze(self.output)
        self.assertEqual(len(result["rows"]), 18)
        self.assertLess(result["graph_median_paired_ratio"], 1)
        row = result["rows"][0]
        self.assertAlmostEqual(row["gflops"], 2 * 128**3 / (row["median_ms"] * 1e6))
        self.assertEqual(result["benefit_gate"], "not_applicable_baseline_only_S7_deferred")
        before = {name: (self.output / name).read_bytes() for name in ("analysis.json", "cpu_gemm.csv", "summary.md")}
        analysis.save(self.output)
        self.assertTrue(all((self.output / name).read_bytes() == value for name, value in before.items()))

    def test_archive_replay_without_historical_path_rewrite(self):
        moved = Path(self.temp.name) / "moved"
        shutil.copytree(self.output, moved)
        self.assertEqual(analysis.analyze(moved), analysis.analyze(self.output))
        analysis.save(moved)

    def test_raw_digest_tamper(self):
        path = self.output / self.manifest["measurements"][0]["raw"]
        path.write_text(path.read_text().replace("1.0,", "900.0,", 1))
        self.assert_rejected()

    def corrupt_raw(self, field, value):
        name = self.manifest["measurements"][0]["raw"]
        path = self.output / name
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0][field] = value
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=analysis.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self.rehash(name)

    def test_rehashed_nan_timing(self):
        self.corrupt_raw("ms", "nan")
        self.assert_rejected()

    def test_rehashed_nonnumeric_timing(self):
        self.corrupt_raw("ms", "not-a-number")
        self.assert_rejected()

    def test_invalid_json_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot read CPU evidence JSON"):
            analysis.read_json(self.output / "missing.json")

    def test_rehashed_negative_timing(self):
        self.corrupt_raw("ms", "-1")
        self.assert_rejected()

    def test_rehashed_thread_mismatch(self):
        self.corrupt_raw("threads", "8")
        self.assert_rejected()

    def test_rehashed_input_mismatch(self):
        self.corrupt_raw("input_hash", "ffffffffffffffff")
        self.assert_rejected()

    def test_rehashed_checksum_mismatch(self):
        self.corrupt_raw("checksum", "99")
        self.assert_rejected()

    def test_rehashed_sample_coverage(self):
        self.corrupt_raw("sample", "9")
        self.assert_rejected()

    def test_final_correctness_failure(self):
        name = self.manifest["measurements"][0]["correctness"]
        document = analysis.read_json(self.output / name)
        document["final"]["violations"] = 1
        (self.output / name).write_text(json.dumps(document))
        self.rehash(name)
        self.assert_rejected()

    def test_order_mismatch(self):
        self.manifest["measurements"].reverse()
        self.persist()
        self.assert_rejected()

    def test_raw_path_argv_mismatch(self):
        self.manifest["commands"][-1]["argv"][-1] = "/wrong.json"
        self.persist()
        self.assert_rejected()

    def test_source_identity_changed(self):
        self.manifest["source_after"] = dict(SOURCE, source_digest="c" * 64)
        self.persist()
        self.assert_rejected()

    def test_dirty_source_rejected_before_run(self):
        with self.assertRaisesRegex(RuntimeError, "clean source"):
            self.run_mock(dict(SOURCE, source_dirty=True))

    def test_stale_binary_failure_preserved(self):
        def stale(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            if "--probe" in args[0]:
                kwargs["stdout"].seek(0)
                kwargs["stdout"].truncate()
                kwargs["stdout"].write(json.dumps(dict(BINARY, commit="c" * 40)))
            return result
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.run_mock(side_effect=stale)
        failed = [analysis.read_json(p / "manifest.json") for p in self.results.iterdir() if p != self.output]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["status"], "failed")

    def test_command_failure_preserved(self):
        def fail(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            if "--self-test" in args[0]:
                return subprocess.CompletedProcess(args[0], 9)
            return result
        with self.assertRaisesRegex(RuntimeError, "raw log preserved"):
            self.run_mock(side_effect=fail)
        failed = [p for p in self.results.iterdir() if p != self.output][0]
        self.assertEqual(analysis.read_json(failed / "manifest.json")["status"], "failed")
        self.assertTrue((failed / "production-correctness.log").is_file())

    def test_production_hooks_rejected(self):
        name = "production-symbols.log"
        (self.output / name).write_text("runtime::testing::cpu_allocation_counts")
        self.rehash(name)
        self.assert_rejected()

    def test_compiler_flags_rejected(self):
        name = "production-compile_commands.json"
        commands = analysis.read_json(self.output / name)
        commands[0]["command"] += " -ffast-math"
        (self.output / name).write_text(json.dumps(commands))
        self.rehash(name)
        self.assert_rejected()

    def test_existing_derived_not_overwritten(self):
        (self.output / "summary.md").write_text("OLD mock-only")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            analysis.save(self.output)
        self.assertEqual((self.output / "summary.md").read_text(), "OLD mock-only")

    def test_vectorization_uses_original_release_flags(self):
        commands = [{"file": "cpu_scalar.cpp", "command": "c++ -O3 -DNDEBUG -o old.o -c cpu_scalar.cpp", "directory": "mock-only"}]
        argv, _ = runner.vectorization_command(commands, "GNU MOCK", Path("mock-only-build"))
        self.assertIn("-O3", argv)
        self.assertIn("-fopt-info-vec-all", argv)
        self.assertNotIn("old.o", argv)

if __name__ == "__main__":
    unittest.main()
