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

    # The direct MMQ epilogue represents Gate+Up with one Q4 dispatch. Account
    # for fallback Gate/Up pairs (two ordinary Q4 dispatches per helper) when
    # comparing routing-helper and Q4 launch counts.
    paired = sum("<(ggml_type)12, 32, false, true, true>" in name for name in names)
    ordinary = q4_dispatches - paired
    helper_limit = q4_dispatches + ordinary // 2 if paired else q4_dispatches
    if helpers > helper_limit:
        print(f"FAIL: {helpers} routing helpers exceed limit {helper_limit} for {q4_dispatches} Q4_K dispatches")
        return 1

    print(f"PASS: {helpers} routing helpers for {q4_dispatches} Q4_K dispatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
