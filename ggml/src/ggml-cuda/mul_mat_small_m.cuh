#pragma once

#include "common.cuh"

// Fast native matrix multiplication for thin weight matrices (M <= 512, e.g. M=1, 32, 256)
// on AMD GPUs, bypassing slow rocBLAS Tensile MT32x32x8 kernels and eliminating
// FP32->FP16->FP32 roundtrip conversion overhead.

bool ggml_cuda_supports_mul_mat_small_m(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_cuda_mul_mat_small_m(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);
