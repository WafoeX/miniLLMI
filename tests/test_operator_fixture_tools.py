#!/usr/bin/env python3
"""Checksum/provenance/regeneration tests; tampered artifacts exist only in temp dirs."""
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/operators-v1"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("fixture test module unavailable")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


generator = module("fixture_generator", ROOT / "tools/generate_operator_fixtures.py")
harness = module("fixture_harness", ROOT / "tests/run_operator_fixtures.py")


class FixtureToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.copy = Path(self.temp.name) / "fixtures"
        shutil.copytree(FIXTURES, self.copy)

    def test_verified_suite_and_weight_layout(self):
        manifest = harness.verify(FIXTURES)
        self.assertEqual(len(manifest["cases"]), 17)
        self.assertEqual(sum(case["op"] in harness.CORE for case in manifest["cases"]), 8)
        weights = json.loads((FIXTURES / "tiny-weights-v1.json").read_text())
        self.assertEqual(len(weights["tensors"]), 21)
        self.assertEqual(weights["total_bytes"], 461056)

    def test_exact_offline_regeneration(self):
        for name, data in generator.artifacts().items():
            self.assertEqual((FIXTURES / name).read_bytes(), data, name)

    def test_text_checksum_corruption_rejected(self):
        with (self.copy / "operators-v1.txt").open("ab") as stream:
            stream.write(b"extra data")
        with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
            harness.verify(self.copy)

    def test_weight_checksum_corruption_rejected(self):
        path = self.copy / "tiny-weights-v1.bin"
        data = bytearray(path.read_bytes())
        data[4] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
            harness.verify(self.copy)

    def test_byte_range_validation_after_outer_checksum(self):
        path = self.copy / "tiny-weights-v1.json"
        weights = json.loads(path.read_text())
        weights["tensors"][1]["offset_bytes"] = 0
        path.write_text(json.dumps(weights))
        manifest_path = self.copy / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "byte ranges"):
            harness.verify(self.copy)

    def test_manifest_path_traversal_rejected(self):
        path = self.copy / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["../outside"] = manifest["files"].pop("operators-v1.txt")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fixed basenames"):
            harness.verify(self.copy)

    def test_unknown_version_rejected(self):
        path = self.copy / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["version"] = 999
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "version/model"):
            harness.verify(self.copy)

    def test_generator_rejects_malformed_payload(self):
        for shape, values, dtype in [([-1], [], "FP32"), ([2], [1], "FP32"), ([1], [1.2], "INT32"), ([1], [2**31], "INT32"), ([1], [math.nan], "FP32"), ([1], [1], "FP16")]:
            with self.assertRaises(ValueError):
                generator.tensor(shape, values, dtype)

    def test_independent_hand_checked_anchors(self):
        cases = {case["name"]: case for case in generator.cases()}
        self.assertEqual(cases["matmul_rectangular"]["output"]["values"], [38, 44, 50, 56, 83, 98, 113, 128])
        self.assertEqual(cases["embedding_hand"]["output"]["values"], [1, 2, 3, 10, 11, 12, 4, 5, 6])
        self.assertEqual(cases["softmax_all_masked"]["output"]["values"], [0] * 6)
        probabilities = cases["softmax_decode_offset"]["output"]["values"]
        self.assertEqual(probabilities[3], 0)
        self.assertAlmostEqual(sum(probabilities), 1, places=6)
        self.assertAlmostEqual(probabilities[0], 1 / (1 + math.e + math.e**2), places=6)
        rotary = cases["rope_interleaved_hand"]["output"]["values"]
        self.assertAlmostEqual(rotary[0], math.cos(1) - 2 * math.sin(1), places=6)
        self.assertAlmostEqual(rotary[3], 3 * math.sin(0.01) + 4 * math.cos(0.01), places=6)
        self.assertEqual(cases["swiglu_hand"]["output"]["values"][0], 0)
        self.assertAlmostEqual(cases["rmsnorm_hand_zero_row"]["output"]["values"][0], 1 / math.sqrt(2.50001), places=6)


if __name__ == "__main__":
    unittest.main()
