#!/usr/bin/env python3
"""Verify gfx1151 Q8_0 MMQ dispatches use weight-stationary traversal."""

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} KERNEL_TRACE.csv", file=sys.stderr)
        return 2

    rows = [
        row
        for row in csv.DictReader(open(sys.argv[1], newline=""))
        if "mul_mat_q<(ggml_type)8," in row["Kernel_Name"]
    ]
    if not rows:
        print("FAIL: no Q8_0 MMQ dispatches", file=sys.stderr)
        return 1
    if not all(", true>" in row["Kernel_Name"] for row in rows):
        print("FAIL: Q8_0 MMQ has non-weight-stationary dispatches", file=sys.stderr)
        return 1

    print(f"PASS: {len(rows)} gfx1151 Q8_0 MMQ dispatches use weight-stationary traversal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
