#!/usr/bin/env python3
"""Verify gfx1151 MoE Q4_K/Q5_K MMQ dispatches use weight-stationary traversal."""

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} KERNEL_TRACE.csv", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(sys.argv[1], newline="")))
    expected = {
        "Q4_K": "mul_mat_q<(ggml_type)12,",
        "Q5_K": "mul_mat_q<(ggml_type)13,",
    }

    for label, prefix in expected.items():
        dispatches = [row for row in rows if prefix in row["Kernel_Name"]]
        if not dispatches:
            print(f"FAIL: no {label} MMQ dispatches", file=sys.stderr)
            return 1

        stationary = [row for row in dispatches if ", true>" in row["Kernel_Name"]]
        if not stationary:
            print(f"FAIL: no weight-stationary {label} MMQ dispatches", file=sys.stderr)
            return 1

    print("PASS: gfx1151 MoE Q4_K/Q5_K use weight-stationary MMQ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
