#!/usr/bin/env python3
"""Exercise Stage 16-C4 measurement accounting on a generated V2 artifact."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: test_quant_benchmark_tools.py <bench> <converter> <weights> <v1-model>")
    bench, converter, weights, float_model = sys.argv[1:]
    with tempfile.TemporaryDirectory() as directory:
        int8_model = str(Path(directory) / "tiny-int8.mllm")
        converted = subprocess.run([converter, weights, int8_model, "--int8"], text=True, capture_output=True, check=False)
        assert converted.returncode == 0, converted.stderr
        result = subprocess.run([bench, "--float-model", float_model, "--int8-model", int8_model,
                                 "--backend", "cpu", "--cache", "on"], text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        row = json.loads(result.stdout)
    assert row["stage"] == "stage16-c4" and row["correctness"] == "passed"
    assert row["eligible_int8_ratio"] <= .35 and row["eligible_int8_scale_bytes"] < row["eligible_fp32_bytes"]
    assert row["int8_resident_parameter_bytes"] > row["float_resident_parameter_bytes"]
    assert row["int8_prepare_dequant_bytes"] > 0
    assert row["logit_mae"] >= 0 and row["logit_max_abs_error"] >= row["logit_mae"]
    assert len(row["float_generated_tokens"]) == len(row["int8_generated_tokens"]) == 4


if __name__ == "__main__":
    main()
