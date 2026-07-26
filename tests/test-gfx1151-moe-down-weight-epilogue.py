#!/usr/bin/env python3
"""Guard the gfx1151 Q5_K Down MMQ expert-weight epilogue fusion."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMQ_CUH = (ROOT / "ggml/src/ggml-cuda/mmq.cuh").read_text()
MMQ_CU = (ROOT / "ggml/src/ggml-cuda/mmq.cu").read_text()
CUDA_CU = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text()

required = {
    "weighted MMQ writeback": (MMQ_CUH, "mmq_write_back_mma_weighted"),
    "weight pointer in MMQ args": (MMQ_CUH, "output_weights"),
    "weighted Down host dispatch": (MMQ_CU, "const ggml_tensor * weights, ggml_tensor * dst_weighted"),
    "five-node graph fusion": (CUDA_CU, "GGML_OP_GLU, GGML_OP_MUL_MAT_ID, GGML_OP_MUL"),
    "four-node paired fallback": (CUDA_CU, "{ op, op, GGML_OP_GLU, GGML_OP_MUL_MAT_ID }, { i + 3 }"),
    "isolated fallback": (CUDA_CU, "GGML_CUDA_DISABLE_GFX1151_MOE_DOWN_WEIGHT"),
}
missing = [name for name, (text, marker) in required.items() if marker not in text]
assert not missing, "missing Q5_K Down weighting fusion pieces: " + ", ".join(missing)

start = MMQ_CU.index("void ggml_cuda_mul_mat_q_moe_swiglu_down")
body = MMQ_CU[start:start + 12000]
assert "down_args.output_weights" in body
assert "ggml_tensor * dst_final = fuse_output_weight ? dst_weighted : dst_down" in body

writeback_start = MMQ_CUH.index("mmq_write_back_mma_weighted")
writeback_end = MMQ_CUH.index("mmq_swiglu_materialize", writeback_start)
writeback = MMQ_CUH[writeback_start:writeback_end]
assert "#if defined(TURING_MMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)" in writeback, \
    "weighted writeback must preserve pre-Volta CUDA compile reachability"

predicate_start = CUDA_CU.index("static bool ggml_cuda_should_fuse_moe_mmq_q8_epilogue(")
predicate_end = CUDA_CU.index("static bool ggml_cuda_should_fuse_moe_mmq_q8_epilogue_weighted", predicate_start)
predicate = CUDA_CU[predicate_start:predicate_end]
assert "GGML_CUDA_DISABLE_GFX1151_MOE_DOWN_WEIGHT" not in predicate, \
    "disabling Down weighting must preserve the deployed four-node Gate/Up/SwiGLU/Down fusion"

if len(sys.argv) == 2:
    with open(sys.argv[1], newline="") as f:
        names = [row["Kernel_Name"] for row in csv.DictReader(f)]
    weighted_q5 = sum("<(ggml_type)13, 32, false, true, false, true>" in name for name in names)
    assert weighted_q5 > 0, "trace contains no weighted Q5_K Down MMQ epilogue"
    print(f"PASS: {weighted_q5} Q5_K Down dispatches apply expert weights in their epilogue")
else:
    print("PASS: gfx1151 Q5_K Down writes weighted expert output directly")
