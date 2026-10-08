#!/usr/bin/env python3
"""Validate the Stage 17-C3 JSONL schema without fabricating measurements."""
import json
import subprocess
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_inference_benchmark_tools.py <bench_inference> <model>")
    result = subprocess.run([sys.argv[1], "--model", sys.argv[2], "--backend", "cpu", "--quant", "float",
                             "--cache", "on", "--runs", "1"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(rows) == 2, rows
    record, smoke = rows
    assert record["stage"] == "stage17-c3" and record["variant"] == "teacher_forced"
    assert isinstance(record["source_dirty"], bool) and record["warmups"] == 3 and record["samples"] == 10
    assert record["run"] == 0 and record["independent_runs"] == 1
    assert record["backend"] == "cpu" and record["quant"] == "float" and record["cache"] is True
    assert len(record["continuation_tokens"]) == 4
    for field in ("prefill_ms_samples", "first_token_ms_samples", "decode_ms_per_token_samples"):
        assert len(record[field]) == 10 and all(value > 0 for value in record[field]), (field, record[field])
    assert record["execute_allocations"] == 0 and record["correctness"] == "passed"
    assert smoke["variant"] == "greedy_smoke" and len(smoke["generated_tokens"]) == 4


if __name__ == "__main__":
    main()
