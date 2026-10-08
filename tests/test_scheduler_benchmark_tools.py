#!/usr/bin/env python3
"""CPU-only C4 protocol mocks. Synthetic timings/artifacts are temporary, NOT evidence."""
import copy
import csv
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
spec = importlib.util.spec_from_file_location("scheduler_c4_runner_test", TOOLS / "run_scheduler_benchmark.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load C4 runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
analyzer = runner.analyze_scheduler
SOURCE = {"commit": "a" * 40, "source_digest": "b" * 64, "source_dirty": False}


def dump(path, data):
    Path(path).write_text(json.dumps(data) + "\n")


def probe():
    return {"schema_version": 1, "benchmark": "scheduler-c4-v1", **SOURCE, "build_type": "Release", "compiler": "mock-cpu",
            "cuda_compiler": "mock-nvcc", "cuda_architecture": "75", "testing": False, "cuda_enabled": True, "input_hash": analyzer.input_hash()}


def pair(directory, repeat):
    directory = Path(directory)
    directory.mkdir()
    nodes = []
    descriptions = (("ADD", [0, 1], {}), ("COPY", [9], {"overlap": "reject_except_exact_self", "destination": "cuda:0"}),
                    ("COPY", [2], {"overlap": "reject_except_exact_self", "destination": "cuda:0"}), ("MATMUL", [10, 11], {}),
                    ("MATMUL", [10, 11], {}), ("COPY", [12], {"overlap": "reject_except_exact_self", "destination": "cpu:0"}),
                    ("COPY", [13], {"overlap": "reject_except_exact_self", "destination": "cpu:0"}), ("ADD", [14, 15], {}), ("MUL", [16, 8], {}))
    for i, (op, inputs, attrs) in enumerate(descriptions):
        nodes.append({"id": i, "descriptor": {"version": 1, "op": op, "inputs": inputs, "outputs": [9 + i], "attributes": attrs}})
    trace = [{"kind": "BackendCPU", "bytes": 0}] + [{"kind": "BackendCUDA", "bytes": 0} for _ in range(6)]
    trace += [{"kind": "BackendCPU", "bytes": 0} for _ in range(2)] + [{"kind": "Copy", "bytes": 16384} for _ in range(4)]
    trace += [{"kind": kind, "node": i, "bytes": 0} for i in range(9) for kind in ("NodeBegin", "NodeEnd")]
    logical = [("BlockAllocate", 9), ("BlockAllocate", 10), ("BlockFree", 9), ("BlockAllocate", 11),
               ("BlockAllocate", 12), ("BlockAllocate", 13), ("BlockFree", 10), ("BlockFree", 11),
               ("BlockAllocate", 14), ("BlockFree", 12), ("BlockAllocate", 15), ("BlockFree", 13),
               ("BlockAllocate", 16), ("BlockFree", 14), ("BlockFree", 15), ("BlockAllocate", 17), ("BlockFree", 16)]
    trace += [{"kind": kind, "tensor": tensor, "bytes": 16384} for kind, tensor in logical]
    data = {"schema_version": 1, "atol": .001, "rtol": .001, "workspace_bytes": 0, "trace_dropped": 0, "nodes": nodes,
            "device_capacities": [{"device": key, "bytes": value} for key, value in analyzer.CAPACITIES.items()],
            "counts": analyzer.COUNTS, "trace": trace, "output": analyzer.cpu_reference(), "cpu_reference": analyzer.cpu_reference()}
    for mode in ("manual", "automatic"):
        for phase in ("initial", "final"):
            dump(directory / f"{mode}-{phase}.json", data)
    rows = []
    order = "automatic,manual" if repeat == 2 else "manual,automatic"
    for mode in order.split(","):
        for sample in range(10):
            rows.append({"schema_version": "1", "repeat": str(repeat), "order": order, "mode": mode, "sample": str(sample),
                         "ms": str((2 if mode == "manual" else 1) + sample / 100), "warmup": "3", "samples": "10", "batch": "5",
                         "m": "64", "k": "64", "n": "64", **{key: str(value) for key, value in analyzer.COUNTS.items()},
                         "peak_live_bytes": "65536", "live_bytes": "16384", "cpu_capacity": "49152", "cuda_capacity": "65536", "external_bytes": "65536", "oracle_bytes": "16384",
                         "workspace_bytes": "0", "runtime_tensor_resident_bytes": "311296", "input_hash": analyzer.input_hash(),
                         "checksum": str(sum(analyzer.cpu_reference())), "commit": SOURCE["commit"], "source_digest": SOURCE["source_digest"],
                         "source_dirty": "false", "build_type": "Release", "compiler": "mock-cpu", "cuda_compiler": "mock-nvcc",
                         "cuda_architecture": "75", "cpu_backend": "cpu-reference-fp64", "cuda_backend": "cuda-stage0-naive",
                         "timing_boundary": analyzer.BOUNDARY, "correctness": "passed"})
    write_rows(directory, rows)


def write_rows(directory, rows):
    with (directory / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=analyzer.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    dump(root / "probe.log", probe())
    for repeat in (1, 2, 3):
        pair(root / f"repeat-{repeat}", repeat)


class SchedulerBenchmarkTools(unittest.TestCase):
    def test_complete_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            result = analyzer.generate(root, SOURCE)
            self.assertEqual(len(result["pairs"]), 3)
            self.assertIsNone(result["speedup_gate"])
            self.assertEqual(result["c3"], "skipped_optional")
            self.assertGreater(result["median_paired_ratio"], 1)
            with self.assertRaises(ValueError):
                analyzer.generate(root, SOURCE)

    def test_csv_controls(self):
        changes = {"ms": "nan", "warmup": "0", "samples": "9", "batch": "1", "order": "automatic,manual", "m": "63",
                   "copies": "3", "copy_bytes": "0", "backend_switches": "1", "backend_dispatches": "8",
                   "allocations": "1", "peak_live_bytes": "0", "live_bytes": "0", "workspace_bytes": "16", "cpu_capacity": "0", "runtime_tensor_resident_bytes": "0",
                   "checksum": "0", "input_hash": "0" * 16, "commit": "c" * 40, "source_digest": "c" * 64,
                   "source_dirty": "true", "build_type": "Debug", "compiler": "stale", "cuda_compiler": "stale",
                   "cuda_architecture": "80", "cpu_backend": "cpu-simd", "cuda_backend": "cuda-v1-tiled",
                   "timing_boundary": "kernel-only", "correctness": "failed"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            with (root / "repeat-1/samples.csv").open(newline="") as stream:
                original = list(csv.DictReader(stream))
            for key, value in changes.items():
                with self.subTest(key=key):
                    rows = copy.deepcopy(original)
                    rows[0][key] = value
                    write_rows(root / "repeat-1", rows)
                    with self.assertRaises(ValueError):
                        analyzer.analyze(root, SOURCE)

    def test_missing_duplicate_and_zero_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            with (root / "repeat-1/samples.csv").open(newline="") as stream:
                original = list(csv.DictReader(stream))
            for rows in (original[:-1], original + [original[0]], [original[0]] + original[:-1]):
                write_rows(root / "repeat-1", rows)
                with self.assertRaises(ValueError):
                    analyzer.analyze(root, SOURCE)
            rows = copy.deepcopy(original)
            rows[0]["ms"] = "0"
            write_rows(root / "repeat-1", rows)
            with self.assertRaises(ValueError):
                analyzer.analyze(root, SOURCE)

    def test_extra_csv_field_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "repeat-1/samples.csv"
            lines = path.read_text().splitlines()
            lines[1] += ",unexpected"
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(ValueError):
                analyzer.analyze(root, SOURCE)

    def test_snapshot_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "repeat-1/automatic-final.json"
            original = json.loads(path.read_text())
            def corrupt_graph(data):
                data["nodes"][0]["descriptor"]["op"] = "MUL"
            changes = [lambda d: d["output"].__setitem__(-1, 100), lambda d: d["cpu_reference"].__setitem__(-1, 100),
                       lambda d: d["output"].pop(), lambda d: d.__setitem__("atol", 10), lambda d: d.__setitem__("trace_dropped", 1),
                       lambda d: d["counts"].__setitem__("copy_bytes", 0), lambda d: d["trace"].__setitem__(0, {"kind": "BackendCUDA"}),
                       lambda d: d["trace"].append({"kind": "Allocate"}), lambda d: d["trace"].append({"kind": "Copy", "bytes": 1}),
                       lambda d: d["trace"].pop(), lambda d: d.__setitem__("schema_version", 2), corrupt_graph]
            for change in changes:
                data = copy.deepcopy(original)
                change(data)
                dump(path, data)
                with self.assertRaises(ValueError):
                    analyzer.analyze(root, SOURCE)
            path.unlink()
            with self.assertRaises(ValueError):
                analyzer.analyze(root, SOURCE)

    def test_dirty_and_cpu_test_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            for key, value in (("cuda_enabled", False), ("testing", True), ("build_type", "Debug"), ("source_dirty", True), ("commit", "c" * 40)):
                data = probe()
                data[key] = value
                dump(root / "probe.log", data)
                with self.assertRaises(ValueError):
                    analyzer.analyze(root, SOURCE)
            with self.assertRaises(ValueError):
                analyzer.analyze(root, {**SOURCE, "source_dirty": True})

    def mock_capture(self, fail=None, changed=False, gpu="Tesla T4, 7.5, mock-driver, 15360\n", cpu_probe=False, modern_ctest=False, ctest_count=38, late_change=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def run(argv, cwd, stdout, stderr, check):
                label = Path(stdout.name).stem
                if label == "gpu-query":
                    stdout.write(gpu)
                elif label == "probe":
                    data = probe()
                    if cpu_probe:
                        data["cuda_enabled"] = False
                    stdout.write(json.dumps(data))
                elif label == "ctest":
                    stdout.write(f"100% tests passed out of {ctest_count}\n" if modern_ctest else f"100% tests passed, 0 tests failed out of {ctest_count}\n")
                else:
                    stdout.write("mock C4 command, not real evidence\n")
                if label.startswith("configure-"):
                    folder = Path(argv[argv.index("-B") + 1])
                    folder.mkdir(parents=True)
                    (folder / "CMakeCache.txt").write_text("mock cache\n")
                    (folder / "compile_commands.json").write_text("[]\n")
                if label.startswith("paired-") and fail != label:
                    pair(Path(argv[argv.index("--output") + 1]), int(argv[argv.index("--repeat") + 1]))
                return subprocess.CompletedProcess(argv, 1 if fail == label else 0)
            changed_source = {**SOURCE, "source_digest": "c" * 64}
            side_effect = [SOURCE, SOURCE if late_change else changed_source, changed_source] if changed or late_change else None
            with mock.patch.object(runner, "inspect", return_value=SOURCE, side_effect=side_effect), \
                    mock.patch.object(runner.subprocess, "run", side_effect=run), \
                    mock.patch.object(runner.platform, "platform", return_value="mock-platform"):
                if fail or changed or late_change or cpu_probe or "T4" not in gpu:
                    with self.assertRaises(RuntimeError):
                        runner.capture(root, root / "results", root / "build")
                else:
                    runner.capture(root, root / "results", root / "build")
            path = next((root / "results").glob("*/manifest.json"))
            manifest = json.loads(path.read_text())
            for name, sha in manifest["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((path.parent / name).read_bytes()).hexdigest(), sha)
            if manifest["status"] == "passed":
                if ctest_count < 38:
                    with self.assertRaises(ValueError):
                        analyzer.verify(path.parent)
                else:
                    self.assertEqual(analyzer.verify(path.parent)["correctness"], "passed")
                    first = next(iter(manifest["artifact_sha256"]))
                    manifest["artifact_sha256"].pop(first)
                    dump(path, manifest)
                    with self.assertRaises(ValueError):
                        analyzer.verify(path.parent)
            return manifest

    def test_runner_success(self):
        manifest = self.mock_capture()
        self.assertEqual(len(manifest["commands"]), 14)
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(self.mock_capture(modern_ctest=True)["status"], "passed")
        self.assertEqual(self.mock_capture(modern_ctest=True, ctest_count=60)["status"], "passed")
        self.assertEqual(self.mock_capture(ctest_count=37)["status"], "passed")

    def test_runner_failures_retained(self):
        for label in ("build-testing", "ctest", "probe", "paired-2"):
            with self.subTest(command=label):
                manifest = self.mock_capture(fail=label)
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["commands"][-1]["exit_code"], 1)

    def test_wrong_gpu_and_source_change(self):
        self.assertEqual(self.mock_capture(gpu="NVIDIA A100, 8.0, driver, 40000\n")["status"], "failed")
        self.assertEqual(self.mock_capture(changed=True)["status"], "failed")
        self.assertEqual(self.mock_capture(late_change=True)["status"], "failed")

    def test_never_time_cpu_binary(self):
        manifest = self.mock_capture(cpu_probe=True)
        self.assertEqual(manifest["commands"][-1]["name"], "probe")
        self.assertEqual(manifest["status"], "failed")

    def test_dirty_runner_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(runner, "inspect", return_value={**SOURCE, "source_dirty": True}):
            with self.assertRaises(ValueError):
                runner.capture(temp, Path(temp) / "results", Path(temp) / "build")
            self.assertFalse((Path(temp) / "results").exists())


if __name__ == "__main__":
    unittest.main()
