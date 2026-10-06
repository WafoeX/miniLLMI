#!/usr/bin/env python3
"""CPU-only schema checks for the Stage 9 CUDA evidence analyzer."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
_spec = importlib.util.spec_from_file_location("analyze_cuda_gemm", TOOLS / "analyze_cuda_gemm.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load Stage 9 analyzer")
analyzer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyzer)


class CudaGemmToolTests(unittest.TestCase):
    def test_versioned_schema_has_required_registry_fields(self):
        self.assertEqual(analyzer.COLUMNS[0], "schema_version")
        self.assertEqual(analyzer.COLUMNS[-4:], ("threads", "vector", "shared_bytes", "kernel_order"))
        self.assertEqual(set(analyzer.KERNEL_CONFIGS), {"sgemm_v0_naive", "sgemm_v1_tiled", "cublas"})
        self.assertEqual(analyzer.KERNEL_CONFIGS["sgemm_v1_tiled"], {
            "block_x": "16", "block_y": "16", "bm": "16", "bn": "16", "bk": "16",
            "tm": "1", "tn": "1", "threads": "256", "vector": "1", "shared_bytes": "2048",
        })

    def test_reader_rejects_stage0_schema_for_stage9(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(analyzer.BASE_COLUMNS)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                analyzer.rows(path)

    def test_numeric_parsers_reject_nonfinite_and_noncanonical_integer(self):
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(ValueError):
                analyzer.number(value)
        with self.assertRaises(ValueError):
            analyzer.integer("01")


if __name__ == "__main__":
    unittest.main()
