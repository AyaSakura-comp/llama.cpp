#!/usr/bin/env python3
"""Regression check for MTP prefill copying full-vocabulary logits to host.

Usage: test-mtp-prefill-copy.py COPY_TRACE.csv
The trace is produced by the HIP memcpy instrumentation used for MTP profiling.
"""

import csv
import sys
from pathlib import Path


FULL_LOGITS_COPY_MIN_BYTES = 400 * 1024 * 1024


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} COPY_TRACE.csv")

    trace = Path(sys.argv[1])
    offenders = []
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row.get("kind", -1)) == 2  # hipMemcpyDeviceToHost
                and int(row.get("bytes", 0)) >= FULL_LOGITS_COPY_MIN_BYTES
            ):
                offenders.append(row)

    assert not offenders, (
        f"MTP prefill copied {len(offenders)} full-vocabulary logits buffers to host; "
        f"first copy was {int(offenders[0]['bytes']) / 2**20:.2f} MiB"
    )
    print("PASS: MTP prefill performed no full-vocabulary D2H copies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
