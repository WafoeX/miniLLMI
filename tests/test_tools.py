#!/usr/bin/env python3
"""Synthetic timing fixtures live only in temporary test dirs, never results/ or reports."""
import csv
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {name}")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


analyze = module("analyze_results")
provenance = module("provenance")


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name)
        self.rows = []
        for kernel, milliseconds in [("sgemm_v0_naive", 4.0), ("cublas", 2.0)]:
            row = dict.fromkeys(analyze.PAIR_FIELDS, "test")
            row.update(timestamp="TEST-FIXTURE", run_id="test-run", commit="a"*40, source_digest="b"*64,
                       source_dirty="0", experiment="baseline", build_type="Release", dtype="fp32",
                       layout="row-major", math_mode="fp32_pedantic", alpha="1", beta="0", m="8", n="8", k="8",
                       warmup="10", iterations="30", seed="42", atol="0.001", rtol="0.001", kernel=kernel,
                       max_error="0", mean_error="0", relative_error="0", violations="0", nonfinite="0", status="ok")
            times = [milliseconds + i * 0.001 for i in range(30)]
            median = statistics.median(times)
            row.update(min_ms=str(min(times)), max_ms=str(max(times)), median_ms=str(median),
                       mean_ms=str(statistics.mean(times)), std_ms=str(statistics.pstdev(times)),
                       gflops=str(2*8**3/(median*1e6)))
            self.rows.append(row)
            samples = [dict(row, sample_index=str(i), elapsed_ms=str(t), status="timing_sample") for i, t in enumerate(times)]
            save(self.raw / "test-run" / "8x8x8" / f"{kernel}_samples.csv", samples)
        ng, bg = (float(r["gflops"]) for r in self.rows)
        for row in self.rows:
            row.update(speedup_vs_naive=str(float(row["gflops"])/ng), cublas_ratio=str(100*float(row["gflops"])/bg))
        correctness = []
        for row in self.rows:
            correctness.extend([dict(row, status="passed_initial"), dict(row, status="passed_final")])
        save(self.raw / "test-run" / "8x8x8" / "correctness.csv", correctness)
        (self.raw / "test-run" / "completed.txt").write_text("test fixture only")

    def report(self):
        return analyze.render(self.rows, self.raw, "TEMPORARY-TEST-FIXTURE.csv")

    def test_report_from_raw(self):
        text = self.report()
        self.assertIn("sgemm_v0_naive", text)
        self.assertIn("100.000%", text)
        self.assertIn("a"*40, text)

    def test_no_data_no_report(self):
        with self.assertRaises(ValueError):
            analyze.render([], self.raw, "none")

    def test_changed_comparison_conditions(self):
        self.rows[1]["seed"] = "999"
        with self.assertRaises(ValueError):
            self.report()

    def test_wrong_gflops(self):
        self.rows[0]["gflops"] = "1000000"
        with self.assertRaises(ValueError):
            self.report()

    def test_failure_not_success(self):
        self.rows[0]["status"] = "failed_correctness"
        with self.assertRaises(ValueError):
            self.report()

    def test_timing_raw_tamper(self):
        path = self.raw / "test-run" / "8x8x8" / "sgemm_v0_naive_samples.csv"
        rows = analyze.read_csv(path)
        rows[0]["elapsed_ms"] = "999"
        save(path, rows)
        with self.assertRaises(ValueError):
            self.report()

    def test_missing_raw_samples(self):
        (self.raw / "test-run" / "8x8x8" / "cublas_samples.csv").unlink()
        with self.assertRaises(OSError):
            self.report()

    def test_invalid_build_and_commit(self):
        for key, value in [("build_type", "Debug"), ("commit", "unknown"), ("source_dirty", "1"), ("nonfinite", "1")]:
            old = self.rows[0][key]
            self.rows[0][key] = value
            with self.assertRaises(ValueError):
                self.report()
            self.rows[0][key] = old

    def test_duplicate_or_single_kernel(self):
        self.rows.append(self.rows[0].copy())
        with self.assertRaises(ValueError):
            self.report()
        self.rows = self.rows[:1]
        with self.assertRaises(ValueError):
            self.report()

    def test_ratio_wrong_units(self):
        self.rows[1]["cublas_ratio"] = "1"
        with self.assertRaises(ValueError):
            self.report()

    def test_path_traversal(self):
        self.rows[0]["run_id"] = "../escape"
        with self.assertRaises(ValueError):
            self.report()

    def test_no_report_without_completed_run(self):
        (self.raw / "test-run" / "completed.txt").unlink()
        with self.assertRaises(ValueError):
            self.report()

    def test_bad_numeric_input(self):
        for key, value in [("iterations", "not-a-number"), ("gflops", "nan"), ("median_ms", "inf")]:
            old = self.rows[0][key]
            self.rows[0][key] = value
            with self.assertRaises(ValueError):
                self.report()
            self.rows[0][key] = old

    def test_cli_missing_csv_preserves_existing_report(self):
        report = self.raw / "old.md"
        report.write_text("previous real report fixture")
        result = subprocess.run([sys.executable, str(ROOT / "tools/analyze_results.py"),
                                 "--csv", str(self.raw / "missing.csv"), "--output", str(report)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report.read_text(), "previous real report fixture")


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Stage0 Test")
        self.run_git("config", "user.email", "test@example.invalid")
        self.run_git("config", "commit.gpgsign", "false")
        (self.root / "main.cpp").write_text("// fixture\n")
        self.run_git("add", "main.cpp")
        self.run_git("commit", "-qm", "test fixture")

    def run_git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=True)

    def test_clean_and_dirty_source(self):
        initial = provenance.inspect(self.root)
        self.assertFalse(initial["source_dirty"])
        self.assertEqual(len(initial["commit"]), 40)
        (self.root / "main.cpp").write_text("// changed\n")
        changed = provenance.inspect(self.root)
        self.assertTrue(changed["source_dirty"])
        self.assertNotEqual(changed["source_digest"], initial["source_digest"])
        self.assertEqual(changed["commit"], initial["commit"])

    def test_results_and_report_excluded(self):
        before = provenance.inspect(self.root)
        (self.root / "results").mkdir()
        (self.root / "results" / "test.csv").write_text("test fixture")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "baseline.md").write_text("generated report fixture")
        self.assertEqual(before, provenance.inspect(self.root))

    def test_untracked_source_included(self):
        before = provenance.inspect(self.root)
        (self.root / "new.cpp").write_text("// fixture\n")
        after = provenance.inspect(self.root)
        self.assertTrue(after["source_dirty"])
        self.assertNotEqual(before["source_digest"], after["source_digest"])

    def test_header_stable(self):
        header = self.root / "results" / "provenance.hpp"
        cmd = [sys.executable, str(ROOT / "tools" / "provenance.py"), "--root", str(self.root), "--header", str(header)]
        subprocess.run(cmd, check=True, capture_output=True)
        timestamp = header.stat().st_mtime_ns
        subprocess.run(cmd, check=True, capture_output=True)
        self.assertEqual(timestamp, header.stat().st_mtime_ns)
        self.assertIn(json.dumps(provenance.inspect(self.root)["commit"]), header.read_text())


class ScriptFailureTests(unittest.TestCase):
    """Mocks only exercise failure paths in a TEMPORARY repo; never emulate successful GPU data."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="stage0-script-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for dirname in ("scripts", "tools"):
            (self.root / dirname).mkdir()
        for name in ("common.sh", "collect_environment.sh", "run_gemm_benchmark.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        shutil.copy2(ROOT / "tools/provenance.py", self.root / "tools/provenance.py")
        (self.root / ".gitignore").write_text("/build/\n/fakebin/\n")
        for args in [("init", "-q"), ("config", "user.name", "Stage0 Test"),
                     ("config", "user.email", "test@example.invalid"), ("config", "commit.gpgsign", "false"),
                     ("add", "."), ("commit", "-qm", "temporary script fixture")]:
            subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)
        self.bin = self.root / "fakebin"
        self.bin.mkdir()
        self.environment = dict(os.environ, PYTHON=sys.executable, BUILD_DIR=str(self.root / "build"),
                                PATH=str(self.bin) + os.pathsep + os.environ["PATH"])

    def mock(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(0o755)

    def test_failed_environment_capture_preserved(self):
        self.mock(self.bin / "nvidia-smi", "echo TEST-FIXTURE-not-a-GPU >&2; exit 77")
        self.mock(self.bin / "nvcc", "echo TEST-FIXTURE-not-a-Toolkit >&2; exit 77")
        out = self.root / "results/environment/test-capture"
        cmd = ["bash", str(self.root / "scripts/collect_environment.sh"), str(out)]
        result = subprocess.run(cmd, env=self.environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXIT_STATUS=77", (out / "nvidia_smi.txt").read_text())
        self.assertIn("EXIT_STATUS=77", (out / "nvcc_version.txt").read_text())
        self.assertIn("executable missing", (out / "gpu_info.txt").read_text())
        original = (out / "system_info.txt").read_bytes()
        retry = subprocess.run(cmd, env=self.environment, capture_output=True, text=True)
        self.assertNotEqual(retry.returncode, 0)
        self.assertEqual(original, (out / "system_info.txt").read_bytes())

    def test_stale_binary_refused_before_benchmark(self):
        self.mock(self.bin / "nvidia-smi", "exit 88")
        metadata = json.dumps({"commit": "0"*40, "source_digest": "b"*64,
                               "source_dirty": False, "build_type": "Release"})
        self.mock(self.root / "build/bench_gemm", "cat <<'JSON'\n" + metadata + "\nJSON")
        result = subprocess.run(["bash", str(self.root / "scripts/run_gemm_benchmark.sh")],
                                env=self.environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Stale binary", result.stderr)
        self.assertFalse((self.root / "results/gemm/baseline.csv").exists())
        self.assertFalse((self.root / "results/.stage0.lock").exists())
        status = list((self.root / "results/gemm/raw").glob("*/runner_status.txt"))
        self.assertEqual(len(status), 1)
        self.assertIn("runner_exit_status=1", status[0].read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
