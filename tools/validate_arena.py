#!/usr/bin/env python3
"""Stage 4 CPU-only correctness evidence, separate from timing diagnostics."""
import sys

from validate_tensor import main

if __name__ == "__main__":
    sys.exit(main(stage=4))
