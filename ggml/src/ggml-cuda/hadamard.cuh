#pragma once

#include "common.cuh"

bool ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_tensor * dst);
