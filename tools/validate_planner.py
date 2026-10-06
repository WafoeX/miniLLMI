#!/usr/bin/env python3
"""Stage 5 CPU-only correctness evidence, separate from paired memory/timing data."""
import sys

from validate_tensor import main

if __name__ == "__main__":
    sys.exit(main(stage=5))
