#pragma once

#include "common.cuh"

// gfx11 (RDNA3/3.5) WMMA flash attention for long-context prefill with D = 256 and GQA ratio 8 (Qwen3.5/3.6 MoE
// full-attention layers). Opt-in: GGML_CUDA_GFX1151_WMMA_FA=1. Returns false (and does nothing) when the op is not
// eligible, so the caller falls back to the regular kernel selection.
bool ggml_cuda_flash_attn_ext_gfx1151(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
