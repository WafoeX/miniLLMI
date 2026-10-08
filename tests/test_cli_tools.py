#!/usr/bin/env python3
"""Stage 17-C1 command-line contract checks."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary, *args):
    return subprocess.run([binary, *args], text=True, capture_output=True, check=False)


def expect_error(binary, *args):
    result = run(binary, *args)
    assert result.returncode != 0, (args, result.stdout, result.stderr)
    message = json.loads(result.stderr)
    assert message["status"] == "error" and message["message"], message


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: test_cli_tools.py <llm_cli> <converter> <legacy-weights> <v1-model>")
    binary = str(Path(sys.argv[1]))
    converter = str(Path(sys.argv[2]))
    weights = str(Path(sys.argv[3]))
    float_model = str(Path(sys.argv[4]))
    first = run(binary, "--model", "tiny.mllm", "--prompt", "hello", "--max-tokens", "7",
                "--backend", "cpu", "--quant", "float", "--cache", "on", "--seed", "42", "--dry-run")
    assert first.returncode == 0, first.stderr
    config = json.loads(first.stdout)
    assert config == {
        "status": "configured", "model": "tiny.mllm", "prompt": "hello", "max_tokens": 7,
        "backend": "cpu", "quant": "float", "cache": True, "seed": 42, "sampling": "greedy",
    }, config
    second = run(binary, "--model", "tiny.mllm", "--prompt", "hello", "--max-tokens", "7",
                 "--backend", "cpu", "--quant", "float", "--cache", "on", "--seed", "42", "--dry-run")
    assert second.returncode == 0 and second.stdout == first.stdout, second.stderr
    help_result = run(binary, "--help")
    assert help_result.returncode == 0 and "--backend cpu|mixed" in help_result.stdout, help_result.stdout
    smoke = run(binary, "--model", float_model, "--prompt", "Stage17", "--max-tokens", "4",
                "--backend", "cpu", "--quant", "float", "--cache", "on", "--seed", "0")
    assert smoke.returncode == 0, smoke.stderr
    smoke_records = [json.loads(line) for line in smoke.stdout.splitlines()]
    assert smoke_records[-1]["generated_tokens"] == [73, 18, 3, 234], smoke_records
    expect_error(binary, "--model", "tiny.mllm", "--backend", "invalid", "--dry-run")
    expect_error(binary, "--model", "tiny.mllm", "--cache", "invalid", "--dry-run")
    expect_error(binary, "--model", "tiny.mllm", "--max-tokens", "0", "--dry-run")
    expect_error(binary, "--model", "tiny.mllm", "--seed", "-1", "--dry-run")
    expect_error(binary, "--prompt", "missing-model", "--dry-run")
    with tempfile.TemporaryDirectory() as directory:
        int8_model = str(Path(directory) / "tiny-int8.mllm")
        converted = subprocess.run([converter, weights, int8_model, "--int8"], text=True, capture_output=True, check=False)
        assert converted.returncode == 0, converted.stderr
        generated = run(binary, "--model", int8_model, "--prompt", "Stage17", "--max-tokens", "2",
                        "--backend", "cpu", "--quant", "int8", "--cache", "on")
        assert generated.returncode == 0, generated.stderr
        records = [json.loads(line) for line in generated.stdout.splitlines()]
        assert records[-1]["status"] == "generated" and len(records[-1]["generated_tokens"]) == 2, records
        expect_error(binary, "--model", int8_model, "--prompt", "x", "--quant", "float")


if __name__ == "__main__":
    main()
