#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("decoder_integration", ROOT / "tools/run_decoder_integration.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load decoder integration tool")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DecoderToolTests(unittest.TestCase):
    def record(self, mode="cpu"):
        value = {
            "schema_version": 1, "stage": "stage13-c4", "mode": mode,
            "commit": "abc", "source_digest": "digest", "source_dirty": False,
            "build_type": "Release", "testing": False, "cuda_enabled": True,
            "correctness": "passed", "nodes": 185, "execute_allocations": 0,
            "peak_live_bytes": 4096, "cpu_capacity_bytes": 4096,
            "cuda_capacity_bytes": 0, "copies": 0, "copy_bytes": 0,
            "backend_dispatches": 185, "backend_switches": 0,
            "prepare_ms": 1.0, "warmups": 3, "samples": 10,
            "execute_samples_ms": [2.0] * 10, "execute_median_ms": 2.0,
            "shape_change_end_to_end_samples_ms": [3.0] * 10,
            "shape_change_end_to_end_median_ms": 3.0,
        }
        if mode == "mixed":
            value.update(cuda_capacity_bytes=4096, copies=10, copy_bytes=1024,
                         cuda_projection_nodes=21, inserted_copy_nodes=10)
        return value

    def test_load_and_validate(self):
        source = {"commit": "abc", "source_digest": "digest", "source_dirty": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(self.record()) + "\n")
            MODULE.validate_benchmark(MODULE.load_benchmark(path), "cpu", source)
        MODULE.validate_benchmark(self.record("mixed"), "mixed", source)

    def test_rejects_dirty_testing_or_missing_copy(self):
        source = {"commit": "abc", "source_digest": "digest", "source_dirty": False}
        for field in ("source_dirty", "testing"):
            value = self.record()
            value[field] = True
            with self.assertRaises(ValueError):
                MODULE.validate_benchmark(value, "cpu", source)
        value = self.record("mixed")
        value["copies"] = 0
        with self.assertRaises(ValueError):
            MODULE.validate_benchmark(value, "mixed", source)

    def test_rejects_missing_raw_samples_or_wrong_median(self):
        source = {"commit": "abc", "source_digest": "digest", "source_dirty": False}
        value = self.record()
        value["execute_samples_ms"] = value["execute_samples_ms"][:-1]
        with self.assertRaises(ValueError):
            MODULE.validate_benchmark(value, "cpu", source)
        value = self.record()
        value["execute_median_ms"] = 2.5
        with self.assertRaises(ValueError):
            MODULE.validate_benchmark(value, "cpu", source)
        value = self.record()
        value["shape_change_end_to_end_samples_ms"][4] = float("inf")
        with self.assertRaises(ValueError):
            MODULE.validate_benchmark(value, "cpu", source)


if __name__ == "__main__":
    unittest.main()
