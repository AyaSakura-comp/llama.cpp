#!/usr/bin/env python3
"""Validate that prefill Gate/Up MUL_MAT_ID pairs share routing preprocessing."""

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} KERNEL_TRACE.csv", file=sys.stderr)
        return 2

    with open(sys.argv[1], newline="") as f:
        names = [row["Kernel_Name"] for row in csv.DictReader(f)]

    helpers = sum("mm_ids_helper<8>" in name for name in names)
    q4_dispatches = sum("mul_mat_q<(ggml_type)12," in name for name in names)
    if not q4_dispatches:
        print("FAIL: trace contains no Q4_K MMQ dispatches")
        return 1

    # Gate and Up contribute one Q4_K dispatch each. With shared preprocessing,
    # their pair plus the separate Down path need no more helper launches than
    # the two Q4_K dispatch streams combined. The unfused graph exceeds this by
    # roughly one helper launch per Gate/Up pair.
    if helpers > q4_dispatches:
        print(f"FAIL: {helpers} routing helpers for {q4_dispatches} Q4_K dispatches; Gate/Up preprocessing is duplicated")
        return 1

    print(f"PASS: {helpers} routing helpers for {q4_dispatches} Q4_K dispatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
