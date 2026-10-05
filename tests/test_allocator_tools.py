#!/usr/bin/env python3
"""Temporary mock-only analyzer/runner cases, never real benchmark evidence."""
import contextlib
import csv
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
        raise RuntimeError("cannot load tools module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("analyze_allocator")
runner = load("run_allocator_benchmark")
SOURCE = {"commit": "mock-only-commit", "source_digest": "mock-only-digest", "source_dirty": False}
BINARY = {"schema_version": 1, "commit": SOURCE["commit"], "source_digest": SOURCE["source_digest"], "source_dirty": False, "build_type": "Release", "testing": False, "backend": "cpu", "compiler": "mock"}


def rows(workload, policy):
    requests, releases, peak, live, external, reuse = analysis.EXPECTED[workload]
    return [{"sample": str(i), "ms": str((2 if policy == "dynamic" else 4) * (1 + i / 10)), "batch": "20", "requests": str(requests), "releases": str(releases),
             "backing_allocs": str(requests if policy == "dynamic" else 0), "backing_frees": str(releases if policy == "dynamic" else 0),
             "peak_payload_bytes": str(peak), "output_live_bytes": str(live), "reused": str(reuse if policy == "arena" else 0), "arena_capacity": "1024" if policy == "arena" else "0",
             "external_bytes": str(external), "peak_backing_bytes": str(external + (1024 if policy == "arena" else peak)), "prepared_backing_allocs": "1" if policy == "arena" else "0",
             "checksum": "336", "correctness": "passed"} for i in range(10)]


def raw(path, workload, policy):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=analysis.COLUMNS)
        writer.writeheader()
        writer.writerows(rows(workload, policy))


class AllocatorTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifest = {"schema_version": 1, "scope": "CPU-only allocator diagnostic", "status": "measured", "source": SOURCE, "source_after": SOURCE,
                    "configuration": {"runs": 3, "samples": 10, "warmup": 3, "batch": 20, "alignment": 64, "arena_capacity": 1024, "workload_version": 1},
                    "binary": BINARY, "commands": [{"exit_code": 0}], "run_id": "mock-only", "timestamp": "mock-only", "host": "mock-only",
                    "pairs": [{"pair": i, "order": ["dynamic", "arena"] if i % 2 == 0 else ["arena", "dynamic"]} for i in range(3)]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        for i in range(3):
            for workload in analysis.WORKLOADS:
                for policy in analysis.POLICIES:
                    raw(self.root / f"pair-{i}/{workload}-{policy}.csv", workload, policy)

    def corrupt(self, column, value):
        path = self.root / "pair-0/chain-arena.csv"
        with path.open(newline="") as stream:
            data = list(csv.DictReader(stream))
        data[0][column] = value
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=analysis.COLUMNS)
            writer.writeheader()
            writer.writerows(data)

    def test_slower_retained_and_idempotent_generation(self):
        result = analysis.save(self.root)
        self.assertEqual(len(result["rows"]), 18)
        self.assertEqual(result["median_paired_ratio"], dict.fromkeys(analysis.WORKLOADS, 0.5))
        before = (self.root / "allocator.csv").read_bytes()
        analysis.save(self.root)
        self.assertEqual(before, (self.root / "allocator.csv").read_bytes())

    def test_corrupt_count(self):
        self.corrupt("backing_allocs", "1")
        with self.assertRaisesRegex(ValueError, "counter"):
            analysis.analyze(self.root)

    def test_invalid_timings(self):
        for value in ("nan", "inf", "-1", "0", "not-a-number"):
            self.corrupt("ms", value)
            with self.assertRaises(ValueError):
                analysis.analyze(self.root)

    def test_checksum_mismatch(self):
        self.corrupt("checksum", "99")
        with self.assertRaisesRegex(ValueError, "checksum"):
            analysis.analyze(self.root)

    def test_missing_pair(self):
        (self.root / "pair-2/diamond-arena.csv").unlink()
        with self.assertRaises(OSError):
            analysis.analyze(self.root)

    def test_missing_sample(self):
        path = self.root / "pair-0/chain-arena.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "coverage"):
            analysis.analyze(self.root)

    def test_dirty_source_and_stale_binary(self):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["binary"]["source_digest"] = "stale"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "binary"):
            analysis.analyze(self.root)
        manifest["binary"] = dict(BINARY)
        manifest["source"] = {**SOURCE, "source_dirty": True}
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance"):
            analysis.analyze(self.root)

    def test_refuse_overwrite(self):
        (self.root / "allocator.csv").write_text("original artifact\n")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            analysis.save(self.root)
        self.assertEqual((self.root / "allocator.csv").read_text(), "original artifact\n")

    def test_runner_dirty_refused(self):
        with patch.object(runner, "inspect", return_value={**SOURCE, "source_dirty": True}), self.assertRaisesRegex(RuntimeError, "clean"):
            runner.run_benchmark(self.root, self.root / "evidence")
        self.assertFalse((self.root / "evidence").exists())

    def test_runner_build_failure_preserved(self):
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


if __name__ == "__main__":
    unittest.main()
