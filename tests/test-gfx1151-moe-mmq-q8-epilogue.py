#!/usr/bin/env python3
"""Static RED/GREEN guard for the gfx1151 Q4 MMQ SwiGLU-to-Q8_1 epilogue."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMQ_CUH = (ROOT / "ggml/src/ggml-cuda/mmq.cuh").read_text()
MMQ_CU = (ROOT / "ggml/src/ggml-cuda/mmq.cu").read_text()
CUDA_CU = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text()

required = {
    "MMQ epilogue": "mmq_write_back_swiglu_q8_1",
    "direct Q8 destination": "dst_swiglu_q8_1",
    "fused host dispatch": "ggml_cuda_mul_mat_q_moe_swiglu_down",
    "four-node graph fusion": "GGML_OP_GLU, GGML_OP_MUL_MAT_ID",
}
texts = {**{k: MMQ_CUH for k in ("MMQ epilogue", "direct Q8 destination")},
         "fused host dispatch": MMQ_CU,
         "four-node graph fusion": CUDA_CU}
missing = [name for name, marker in required.items() if marker not in texts[name]]
assert not missing, "missing direct quantized MMQ handoff pieces: " + ", ".join(missing)

# The optimized path must not materialize Gate, Up, or SwiGLU in graph F32 buffers.
# Gate and Up are accumulated sequentially inside one MMQ workgroup.
start = MMQ_CU.index("void ggml_cuda_mul_mat_q_moe_swiglu_down")
body = MMQ_CU[start: start + 10000]
assert "ggml_cuda_op_swiglu" not in body
assert "dst_gate->data" not in body
assert "dst_up->data" not in body
# One quantizer remains for the layer input; there must not be a second one for SwiGLU.
assert body.count("quantize_mmq_q8_1_cuda") == 1

if len(sys.argv) == 2:
    with open(sys.argv[1], newline="") as f:
        names = [row["Kernel_Name"] for row in csv.DictReader(f)]
    paired = sum("<(ggml_type)12, 32, false, true, true>" in name for name in names)
    q5 = sum("mul_mat_q<(ggml_type)13," in name for name in names)
    assert paired > 0, "trace contains no paired Gate+Up Q4_K MMQ epilogue"
    assert paired == q5, f"paired Q4_K epilogues ({paired}) do not match Q5_K Down dispatches ({q5})"
    print(f"PASS: {paired} paired Q4_K MMQ epilogues feed {q5} Q5_K Down dispatches")
else:
    print("PASS: gfx1151 Q4 MMQ epilogue emits Q8_1 directly for Q5 Down")
