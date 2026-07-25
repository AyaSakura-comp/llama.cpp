#!/usr/bin/env python3
"""Reject repeated Qwen MTP prompt LM-head projections in a rocprof kernel stats CSV.

Usage: test-mtp-prefill-lm-head.py KERNEL_STATS.csv

For an eligible multi-ubatch MTP prefill, only the global final output row needs
vocabulary logits. The one-row head uses MMQ directly, so Qwen's Q6_K output
tensor must never take the repeated full-matrix dequantize-plus-GEMM path.
"""

import csv
import sys
from pathlib import Path


Q6_K_DEQUANT_KERNEL = "dequantize_block_q6_K"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} KERNEL_STATS.csv")

    stats = Path(sys.argv[1])
    calls = 0
    with stats.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if Q6_K_DEQUANT_KERNEL in row.get("Name", ""):
                calls += int(row["Calls"])

    assert calls == 0, f"MTP prefill dequantized the full Q6_K output head {calls} times"
    print("PASS: MTP prefill performed no full Q6_K output-head dequantizations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
