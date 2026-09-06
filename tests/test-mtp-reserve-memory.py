#!/usr/bin/env python3
"""Reject oversized target/draft MTP prefill scheduler reservations.

Usage: test-mtp-reserve-memory.py SERVER_LOG

A production MTP server has two contexts. They must keep only token-generation
buffers while idle and grow prompt-processing buffers to the actual prompt on
first use, rather than retaining two n_ctx-by-n_ubatch attention masks.
"""

import re
import sys
from pathlib import Path


BUFFER_RE = re.compile(
    r"sched_reserve:\s+(ROCm0|ROCm_Host) compute buffer size =\s+([0-9.]+) MiB"
)
MAX_COMBINED_COMPUTE_MIB = 11 * 512
MAX_SINGLE_HOST_COMPUTE_MIB = 1200


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} SERVER_LOG")

    entries = [(kind, float(size)) for kind, size in BUFFER_RE.findall(Path(sys.argv[1]).read_text())]
    gpu = [size for kind, size in entries if kind == "ROCm0"]
    host = [size for kind, size in entries if kind == "ROCm_Host"]

    assert len(gpu) >= 4, f"expected generic and dynamic target/draft GPU reserves, got {gpu}"
    assert len(host) >= 4, f"expected generic and dynamic target/draft host reserves, got {host}"

    # Each context logs its generic reserve followed by the replacement TG-only
    # reserve. Keep the replacement from the final target/draft pair.
    gpu = gpu[-4:][1::2]
    host = host[-4:][1::2]
    combined = sum(gpu) + sum(host)
    assert combined < MAX_COMBINED_COMPUTE_MIB, (
        f"MTP compute reservation is {combined:.2f} MiB; expected less than "
        f"{MAX_COMBINED_COMPUTE_MIB} MiB"
    )
    assert max(host) < MAX_SINGLE_HOST_COMPUTE_MIB, (
        f"MTP host compute reservation is still oversized: {host} MiB"
    )

    print(
        f"PASS: target+draft compute reserve is {combined:.2f} MiB "
        f"(GPU={sum(gpu):.2f}, host={sum(host):.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
