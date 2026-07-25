#!/usr/bin/env python3
"""Verify the gfx1151 prefill trace uses the wider GDN column workgroup."""

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
        if "gated_delta_net_cuda<128, false" in row["Kernel_Name"]
    ]
    if not rows:
        print("FAIL: no S_v=128 scalar-decay GDN dispatches", file=sys.stderr)
        return 1

    widths = {int(row["Workgroup_Size_Y"]) for row in rows}
    if widths != {8}:
        print(f"FAIL: expected gfx1151 GDN workgroup Y=8, observed {sorted(widths)}", file=sys.stderr)
        return 1

    print(f"PASS: {len(rows)} GDN dispatches use workgroup Y=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
