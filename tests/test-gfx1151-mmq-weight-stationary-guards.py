#!/usr/bin/env python3
"""Guard weight-stationary MMQ against unsupported devices and oversized grid Y."""

from pathlib import Path


SOURCE = Path(__file__).parents[1] / "ggml/src/ggml-cuda/mmq.cuh"


def main() -> int:
    text = SOURCE.read_text()
    required = {
        "exact gfx1151 scope": "(cc & 0xffff) == 0x1151",
        "grid-Y limit": "nty <= 65535",
    }
    missing = [label for label, token in required.items() if token not in text]
    if missing:
        print(f"FAIL: missing weight-stationary guards: {', '.join(missing)}")
        return 1

    print("PASS: weight-stationary MMQ is limited to gfx1151 and legal grid-Y sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
