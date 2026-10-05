#!/usr/bin/env python3
"""Stage 2 entry point reusing the established CPU validation/evidence runner."""
import sys

from validate_tensor import main

if __name__ == "__main__":
    sys.exit(main(stage=2))
