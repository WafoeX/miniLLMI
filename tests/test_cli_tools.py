#!/usr/bin/env python3
"""Stage 17-C1 command-line contract checks."""
import json
import subprocess
import sys
from pathlib import Path


def run(binary, *args):
    return subprocess.run([binary, *args], text=True, capture_output=True, check=False)


def expect_error(binary, *args):
    result = run(binary, *args)
    assert result.returncode != 0, (args, result.stdout, result.stderr)
    message = json.loads(result.stderr)
    assert message["status"] == "error" and message["message"], message


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_cli_tools.py <llm_cli>")
    binary = str(Path(sys.argv[1]))
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
    expect_error(binary, "--model", "tiny.mllm", "--backend", "invalid", "--dry-run")
    expect_error(binary, "--model", "tiny.mllm", "--cache", "invalid", "--dry-run")
    expect_error(binary, "--model", "tiny.mllm", "--max-tokens", "0", "--dry-run")
    expect_error(binary, "--model", "tiny.mllm", "--seed", "-1", "--dry-run")
    expect_error(binary, "--prompt", "missing-model", "--dry-run")


if __name__ == "__main__":
    main()
