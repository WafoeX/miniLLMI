#!/usr/bin/env python3
"""Reuse S2's checksum/provenance verification for the S3 graph fixture harness."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from run_operator_fixtures import NUMERIC, ROOT, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "tests/fixtures/operators-v1")
    args = parser.parse_args()
    try:
        manifest = verify(args.root)
        print("Graph fixture checksum/provenance: PASS", flush=True)
        process = subprocess.run([str(args.binary), str(args.root / "operators-v1.txt")], text=True, capture_output=True)
        print(process.stdout, end="")
        print(process.stderr, end="", file=sys.stderr)
        if process.returncode:
            return process.returncode
        reports = re.findall(r"^graph fixture (\S+) (core|unsupported)=PASS$", process.stdout, re.MULTILINE)
        expected = [(case["name"], "core" if case["op"] in NUMERIC else "unsupported") for case in manifest["cases"]]
        if reports != expected:
            raise ValueError("graph fixture execution coverage mismatch")
        numeric = sum(case["op"] in NUMERIC for case in manifest["cases"])
        summary = f"Graph fixtures: PASS numeric_core={numeric} unsupported={len(expected) - numeric}"
        if summary not in process.stdout.splitlines():
            raise ValueError("graph fixture summary mismatch")
        return 0
    except (KeyError, ValueError, OSError) as error:
        print("run_graph_fixtures: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
