#!/usr/bin/env python3
"""Synthetic schema checks for Stage 18 aggregation; no synthetic result is evidence."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage18", ROOT / "tools/verify_stage18_evidence.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Stage 18 verifier")
stage18 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage18)
SOURCE = {"commit": "a" * 40, "source_digest": "b" * 64}


def dump(path, value):
    Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")


def kv_rows():
    rows = []
    for context in (128, 256, 512):
        for run in range(3):
            for variant, median in (("full_prefix", 10.0), ("kv_cache", 5.0)):
                rows.append({**SOURCE, "source_dirty": False, "stage": "stage14-c4", "mode": "cpu", "build_type": "Release",
                             "testing": False, "warmups": 3, "samples": 10, "continuation_tokens": 32, "correctness": "passed",
                             "context": context, "run": run, "variant": variant, "decode_ms_per_token_samples": [median] * 10,
                             "decode_ms_per_token_median": median})
    rows.append({**SOURCE, "source_dirty": False, "stage": "stage14-c4", "mode": "mixed", "context": 512,
                 "warmups": 3, "samples": 10, "correctness": "passed", "decode_ms_per_token_samples": [1.0] * 10})
    return rows


class Stage18EvidenceTools(unittest.TestCase):
    def test_kv_replays_samples_and_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kv.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in kv_rows()), encoding="utf-8")
            result = stage18.verify_kv(path, SOURCE)
            self.assertEqual(result["context_512_ratios"], [2.0, 2.0, 2.0])
            rows = kv_rows()
            rows[0]["decode_ms_per_token_median"] = 99.0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                stage18.verify_kv(path, SOURCE)

    def test_quant_rejects_bad_ratio_and_resident_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quant.json"
            row = {**SOURCE, "source_dirty": False, "schema_version": 1, "stage": "stage16-c4", "backend": "cpu", "cache": True,
                   "correctness": "passed", "eligible_int8_ratio": .25, "eligible_fp32_bytes": 100, "eligible_int8_scale_bytes": 25,
                   "float_resident_parameter_bytes": 100, "int8_resident_parameter_bytes": 125, "int8_prepare_dequant_bytes": 100,
                   "logit_mae": .001, "logit_max_abs_error": .002}
            dump(path, row)
            self.assertEqual(stage18.verify_quant(path, SOURCE, "cpu")["eligible_ratio"], .25)
            row["eligible_int8_ratio"] = .36
            dump(path, row)
            with self.assertRaises(ValueError):
                stage18.verify_quant(path, SOURCE, "cpu")
            row["eligible_int8_ratio"] = .25
            row["int8_resident_parameter_bytes"] = 99
            dump(path, row)
            with self.assertRaises(ValueError):
                stage18.verify_quant(path, SOURCE, "cpu")

    def test_inference_requires_all_runs_and_raw_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.jsonl"
            rows = []
            for run in range(3):
                rows.append({**SOURCE, "source_dirty": False, "stage": "stage17-c3", "variant": "teacher_forced", "backend": "cpu",
                             "quant": "float", "cache": True, "warmups": 3, "samples": 10, "independent_runs": 3, "run": run,
                             "correctness": "passed", "execute_allocations": 0, "prefill_ms_samples": [1.0] * 10,
                             "first_token_ms_samples": [1.0] * 10, "decode_ms_per_token_samples": [1.0] * 10})
            rows.append({"variant": "greedy_smoke"})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(stage18.verify_inference(path, SOURCE, "cpu", "float", True)["records"], 3)
            rows.pop(2)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                stage18.verify_inference(path, SOURCE, "cpu", "float", True)


if __name__ == "__main__":
    unittest.main()
