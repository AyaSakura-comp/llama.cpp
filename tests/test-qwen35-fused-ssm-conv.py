#!/usr/bin/env python3
"""Regression check for CUDA Qwen35 chunked SSM-conv concat fusion."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_trace_csv", type=Path)
    parser.add_argument("--max-materialized", type=int, default=60,
                        help="allow autoregressive fallback concats; chunked concats must be fused")
    args = parser.parse_args()

    with args.kernel_trace_csv.open(newline="") as handle:
        names = [row["Kernel_Name"] for row in csv.DictReader(handle)]

    materialized = sum("concat_f32_non_cont<0>" in name for name in names)
    fused = sum("ssm_conv_split" in name for name in names)
    assert materialized <= args.max_materialized, (
        f"found {materialized} materialized Qwen35 convolution concats"
    )
    assert fused > 0, "no fused split-input SSM convolution kernels found"
    print(f"PASS: {fused} split-input SSM dispatches; {materialized} autoregressive fallback concats")


if __name__ == "__main__":
    main()
