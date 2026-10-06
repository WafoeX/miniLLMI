#!/usr/bin/env python3
"""Stage 6 CPU-only correctness evidence; independent of Release timing data."""
import sys

from validate_tensor import main

if __name__ == "__main__":
    sys.exit(main(stage=6))
