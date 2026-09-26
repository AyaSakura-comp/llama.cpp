#include "mul_mat_small_m.cuh"
#include "ggml-backend-impl.h"

// 4 rows of M per warp, 4 warps per block (1 token per warp)
// Coalesced loads: all 32 lanes in a warp load a contiguous chunk of 128 elements (512 bytes)
template <int M_PER_WARP = 4, int WARPS_PER_BLOCK = 4, int WARP_SZ = 32>
__global__ void __launch_bounds__(WARP_SZ * WARPS_PER_BLOCK, 4)
mul_mat_f32_f32_small_m_kernel(
    const float * __restrict__ src0,
    const float * __restrict__ src1,
    float * __restrict__ dst,
    int M, int N, int K,
    int64_t stride_src0_row,
    int64_t stride_src1_row,
    int64_t stride_dst_row,
    int64_t stride_src0_sample,
    int64_t stride_src1_sample,
    int64_t stride_dst_sample,
    int ne02, int ne12) {

    const int batch_idx = blockIdx.z;
    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;
    const int i02 = i12 % ne02;

    const float * src0_batch = src0 + (int64_t)i02 * stride_src0_sample;
    const float * src1_batch = src1 + (int64_t)(i13 * ne12 + i12) * stride_src1_sample;
    float * dst_batch        = dst  + (int64_t)(i13 * ne12 + i12) * stride_dst_sample;

    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;

    const int token_idx = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    const int m_base    = blockIdx.x * M_PER_WARP;

    if (token_idx >= N) return;

    const float * b_token = src1_batch + (int64_t)token_idx * stride_src1_row;

    float sum[M_PER_WARP] = {0.0f};

    const int n_chunks = K / 128;

#pragma unroll 4
    for (int c = 0; c < n_chunks; ++c) {
        const int k_offset = c * 128 + lane * 4;
        const float4 b_val = *reinterpret_cast<const float4 *>(b_token + k_offset);

#pragma unroll
        for (int mi = 0; mi < M_PER_WARP; ++mi) {
            if (m_base + mi < M) {
                const float * a_row = src0_batch + (int64_t)(m_base + mi) * stride_src0_row;
                const float4 a_val = *reinterpret_cast<const float4 *>(a_row + k_offset);
                sum[mi] += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
            }
        }
    }

    // Residual elements if K % 128 != 0
    const int k_rem_start = n_chunks * 128;
    for (int k = k_rem_start + lane; k < K; k += WARP_SZ) {
        const float b_v = b_token[k];
#pragma unroll
        for (int mi = 0; mi < M_PER_WARP; ++mi) {
            if (m_base + mi < M) {
                sum[mi] += src0_batch[(int64_t)(m_base + mi) * stride_src0_row + k] * b_v;
            }
        }
    }

    // Warp reduction using hardware DPP / warp shuffle
#pragma unroll
    for (int mi = 0; mi < M_PER_WARP; ++mi) {
        sum[mi] = warp_reduce_sum<WARP_SZ>(sum[mi]);
    }

    if (lane == 0) {
#pragma unroll
        for (int mi = 0; mi < M_PER_WARP; ++mi) {
            if (m_base + mi < M) {
                dst_batch[(int64_t)token_idx * stride_dst_row + (m_base + mi)] = sum[mi];
            }
        }
    }
}

template <int M_PER_WARP = 4, int WARPS_PER_BLOCK = 4, int WARP_SZ = 32>
__global__ void __launch_bounds__(WARP_SZ * WARPS_PER_BLOCK, 4)
mul_mat_f16_f32_small_m_kernel(
    const half * __restrict__ src0,
    const float * __restrict__ src1,
    float * __restrict__ dst,
    int M, int N, int K,
    int64_t stride_src0_row,
    int64_t stride_src1_row,
    int64_t stride_dst_row,
    int64_t stride_src0_sample,
    int64_t stride_src1_sample,
    int64_t stride_dst_sample,
    int ne02, int ne12) {

    const int batch_idx = blockIdx.z;
    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;
    const int i02 = i12 % ne02;

    const half * src0_batch  = src0 + (int64_t)i02 * stride_src0_sample;
    const float * src1_batch = src1 + (int64_t)(i13 * ne12 + i12) * stride_src1_sample;
    float * dst_batch        = dst  + (int64_t)(i13 * ne12 + i12) * stride_dst_sample;

    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;

    const int token_idx = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    const int m_base    = blockIdx.x * M_PER_WARP;

    if (token_idx >= N) return;

    const float * b_token = src1_batch + (int64_t)token_idx * stride_src1_row;

    float sum[M_PER_WARP] = {0.0f};

    const int n_chunks = K / 128;

#pragma unroll 4
    for (int c = 0; c < n_chunks; ++c) {
        const int k_offset = c * 128 + lane * 4;
        const float4 b_val = *reinterpret_cast<const float4 *>(b_token + k_offset);

#pragma unroll
        for (int mi = 0; mi < M_PER_WARP; ++mi) {
            if (m_base + mi < M) {
                const half * a_row = src0_batch + (int64_t)(m_base + mi) * stride_src0_row;
                const half2 a_h0 = *reinterpret_cast<const half2 *>(a_row + k_offset);
                const half2 a_h1 = *reinterpret_cast<const half2 *>(a_row + k_offset + 2);
                const float2 a_f0 = __half22float2(a_h0);
                const float2 a_f1 = __half22float2(a_h1);
                sum[mi] += a_f0.x * b_val.x + a_f0.y * b_val.y + a_f1.x * b_val.z + a_f1.y * b_val.w;
            }
        }
    }

    const int k_rem_start = n_chunks * 128;
    for (int k = k_rem_start + lane; k < K; k += WARP_SZ) {
        const float b_v = b_token[k];
#pragma unroll
        for (int mi = 0; mi < M_PER_WARP; ++mi) {
            if (m_base + mi < M) {
                sum[mi] += __half2float(src0_batch[(int64_t)(m_base + mi) * stride_src0_row + k]) * b_v;
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < M_PER_WARP; ++mi) {
        sum[mi] = warp_reduce_sum<WARP_SZ>(sum[mi]);
    }

    if (lane == 0) {
#pragma unroll
        for (int mi = 0; mi < M_PER_WARP; ++mi) {
            if (m_base + mi < M) {
                dst_batch[(int64_t)token_idx * stride_dst_row + (m_base + mi)] = sum[mi];
            }
        }
    }
}

bool ggml_cuda_supports_mul_mat_small_m(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst) {

    if (ggml_is_quantized(src0->type)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }

    // Thin matrix check: M <= 512, inner dimension K divisible by 32
    if (src0->ne[1] > 512 || src0->ne[0] % 32 != 0) {
        return false;
    }

    // Check contiguous row strides
    const size_t ts0 = ggml_type_size(src0->type);
    if (src0->nb[0] != ts0 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }

    // Enable for batch >= 32 to replace slow rocBLAS Tensile
    if (src1->ne[1] < 32) {
        return false;
    }

    return true;
}

void ggml_cuda_mul_mat_small_m(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne02 = src0->ne[2];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const size_t ts0 = ggml_type_size(src0->type);
    const int64_t stride_src0_row    = src0->nb[1] / ts0;
    const int64_t stride_src0_sample = src0->nb[2] / ts0;

    const int64_t stride_src1_row    = src1->nb[1] / sizeof(float);
    const int64_t stride_src1_sample = src1->nb[2] / sizeof(float);

    const int64_t stride_dst_row     = dst->nb[1]  / sizeof(float);
    const int64_t stride_dst_sample  = dst->nb[2]  / sizeof(float);

    constexpr int M_PER_WARP       = 4;
    constexpr int WARPS_PER_BLOCK  = 4;
    constexpr int WARP_SIZE_VAL    = WARP_SIZE;

    const dim3 block_dims(WARP_SIZE_VAL, WARPS_PER_BLOCK, 1);
    const dim3 grid_dims(
        (M + M_PER_WARP - 1) / M_PER_WARP,
        (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        ne12 * ne13
    );

    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_F32) {
        mul_mat_f32_f32_small_m_kernel<M_PER_WARP, WARPS_PER_BLOCK, WARP_SIZE_VAL>
            <<<grid_dims, block_dims, 0, stream>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (float *) dst->data,
                (int) M, (int) N, (int) K,
                stride_src0_row, stride_src1_row, stride_dst_row,
                stride_src0_sample, stride_src1_sample, stride_dst_sample,
                (int) ne02, (int) ne12);
    } else if (src0->type == GGML_TYPE_F16) {
        mul_mat_f16_f32_small_m_kernel<M_PER_WARP, WARPS_PER_BLOCK, WARP_SIZE_VAL>
            <<<grid_dims, block_dims, 0, stream>>>(
                (const half *) src0->data,
                (const float *) src1->data,
                (float *) dst->data,
                (int) M, (int) N, (int) K,
                stride_src0_row, stride_src1_row, stride_dst_row,
                stride_src0_sample, stride_src1_sample, stride_dst_sample,
                (int) ne02, (int) ne12);
    } else {
        GGML_ABORT("unsupported src0 type for mul_mat_small_m");
    }

    CUDA_CHECK(cudaGetLastError());
}
