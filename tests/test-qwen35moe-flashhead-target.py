#!/usr/bin/env python3
"""Source-level contract for the opt-in Qwen3.6 target FlashHead graph."""

from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src/models/qwen35moe.cpp"


def main() -> int:
    text = SOURCE.read_text()
    required = {
        "opt-in environment gate": 'getenv("LLAMA_FLASHHEAD_TARGET")',
        "target centroid scores": '"flashhead_target_centroid_logits"',
        "target candidate ids": '"flashhead_target_ids"',
        "target sparse logits": '"flashhead_target_narrow_logits"',
        "multi-column candidate union": "ggml_reshape_1d(ctx0, sel, ggml_nelements(sel))",
        "sparse target projection": "ggml_mul_mat_rows(ctx0, model.output, cur, fh_ids)",
    }
    missing = [description for description, needle in required.items() if needle not in text]
    if missing:
        print("FAIL: missing " + ", ".join(missing))
        return 1
    print("PASS: opt-in target FlashHead graph contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
