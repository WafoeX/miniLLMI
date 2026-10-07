#!/usr/bin/env python3
"""CPU-only protocol tests. Synthetic captures live only in temporary directories."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location("run_scheduler_validation", TOOLS / "run_scheduler_validation.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scheduler validation runner")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

SOURCE = {"commit": "a" * 40, "source_digest": "b" * 64, "source_dirty": False}


class SchedulerTools(unittest.TestCase):
    def snapshots(self, directory):
        nodes = []
        descriptions = (("ADD", [0, 1], {}), ("COPY", [9], {"destination": "cuda:0"}),
                        ("COPY", [2], {"destination": "cuda:0"}), ("MATMUL", [10, 11], {}),
                        ("MATMUL", [10, 11], {}), ("COPY", [12], {"destination": "cpu:0"}),
                        ("COPY", [13], {"destination": "cpu:0"}), ("ADD", [14, 15], {}), ("MUL", [16, 8], {}))
        for i, (op, inputs, attrs) in enumerate(descriptions):
            nodes.append({"id": i, "descriptor": {"op": op, "inputs": inputs, "outputs": [9 + i], "attributes": attrs}})
        for kernel in runner.KERNELS:
            for m, k, n in runner.CASES:
                copies = [4 * m * k, 4 * k * n, 4 * m * n, 4 * m * n]
                copies = [value for value in copies if value]
                trace = [{"kind": "BackendCPU", "bytes": 0}]
                trace += [{"kind": "BackendCUDA", "bytes": 0} for _ in range(2 + len(copies))]
                trace += [{"kind": "BackendCPU", "bytes": 0} for _ in range(2)]
                trace += [{"kind": "Copy", "bytes": value} for value in copies]
                data = {"schema_version": 1, "atol": .001, "rtol": .001, "workspace_bytes": 0, "trace_dropped": 0,
                        "counts": {"copies": len(copies), "copy_bytes": sum(copies), "backend_dispatches": 5 + len(copies),
                                   "backend_switches": 2, "allocations": 0}, "nodes": nodes, "trace": trace,
                        "output": [0.] * (m * n), "cpu_reference": [0.] * (m * n)}
                for repeat in (0, 1):
                    for mode in ("manual", "automatic"):
                        runner.dump(directory / f"{kernel}-{m}-{k}-{n}-repeat{repeat}-{mode}.json", data)

    def mutate_capture(self, update):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshots(root)
            path = root / "cuda-stage0-naive-17-13-11-repeat0-automatic.json"
            data = json.loads(path.read_text())
            update(data)
            runner.dump(path, data)
            with self.assertRaises((ValueError, KeyError)):
                runner.validate_snapshots(root)

    def test_complete_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshots(root)
            self.assertEqual(len(runner.validate_snapshots(root)["pairs"]), 16)

    def test_missing_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshots(root)
            next(root.iterdir()).unlink()
            with self.assertRaises(ValueError):
                runner.validate_snapshots(root)

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshots(root)
            next(root.iterdir()).write_text("{")
            with self.assertRaises(ValueError):
                runner.validate_snapshots(root)

    def test_full_reference_error(self):
        self.mutate_capture(lambda d: d["output"].__setitem__(-1, 10.))

    def test_counter_error(self):
        self.mutate_capture(lambda d: d["counts"].__setitem__("copy_bytes", 1))

    def test_missing_trace(self):
        self.mutate_capture(lambda d: d.__setitem__("trace_dropped", 1))

    def test_changed_tolerance(self):
        self.mutate_capture(lambda d: d.__setitem__("atol", 10.))

    def test_workspace_not_ignored(self):
        self.mutate_capture(lambda d: d.__setitem__("workspace_bytes", 1))

    def test_wrong_reference_length(self):
        self.mutate_capture(lambda d: d["cpu_reference"].pop())

    def test_different_graph(self):
        def change(data):
            data["nodes"] = copy.deepcopy(data["nodes"])
            data["nodes"][0]["descriptor"]["op"] = "MUL"
        self.mutate_capture(change)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('[NaN]', parse_constant=runner.reject_nonfinite)

    def test_clean_source_guard(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(runner, "inspect", return_value={**SOURCE, "source_dirty": True}):
            with self.assertRaises(RuntimeError):
                runner.capture(temp, True)
            self.assertFalse((Path(temp) / "results").exists())

    def mock_capture(self, cpu_only, fail_label=None, changed_source=False, gpu="Tesla T4, 7.5\n"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def run(argv, cwd, env, stdout, stderr):
                label = Path(stdout.name).stem
                stdout.write(gpu if label == "gpu-query" else "mock command output\n")
                return subprocess.CompletedProcess(argv, 1 if label == fail_label else 0)
            sources = [SOURCE, {**SOURCE, "source_digest": "c" * 64}, {**SOURCE, "source_digest": "c" * 64}] if changed_source else None
            with mock.patch.object(runner, "inspect", side_effect=sources, return_value=SOURCE), \
                    mock.patch.object(runner.subprocess, "run", side_effect=run), \
                    mock.patch.object(runner.platform, "platform", return_value="mock-platform"), \
                    mock.patch.object(runner.platform, "node", return_value="mock-host"), \
                    mock.patch.object(runner.platform, "machine", return_value="mock-machine"):
                if fail_label or changed_source or not cpu_only:
                    with self.assertRaises(RuntimeError):
                        runner.capture(root, cpu_only)
                else:
                    runner.capture(root, cpu_only)
            manifest_path = next((root / "results/scheduler/stage11-c2").glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            for name, sha in manifest["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest(), sha)
            return manifest

    def test_cpu_capture_protocol(self):
        manifest = self.mock_capture(True)
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(len(manifest["commands"]), 9)
        self.assertTrue(manifest["cpu_only"])
        self.assertEqual(manifest["c4_status"], "not_started")

    def test_failure_retained_and_hashed(self):
        manifest = self.mock_capture(True, "release-build")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["commands"][-1]["exit_code"], 1)
        self.assertEqual(len(manifest["artifact_sha256"]), 2)

    def test_source_change_rejected(self):
        self.assertEqual(self.mock_capture(True, changed_source=True)["status"], "failed")

    def test_non_t4_rejected(self):
        manifest = self.mock_capture(False, gpu="NVIDIA A100, 8.0\n")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["commands"]), 2)


if __name__ == "__main__":
    unittest.main()
