#pragma once

#include "common.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(GGML_USE_HIP)
typedef _Float16 half16_amdgcn __attribute__((ext_vector_type(16)));
typedef float    float8_amdgcn __attribute__((ext_vector_type(8)));

union half16_amdgcn_u {
    half16_amdgcn vec;
    float4 f4[2];
    _Float16 h[16];
};

union float8_amdgcn_u {
    float8_amdgcn vec;
    float4 f4[2];
    float f[8];
};

static constexpr bool WMMA_GFX1151_GLOBAL_KV_DEQUANT = true;

static constexpr int wmma_gfx1151_ceil_div(const int value, const int divisor) {
    return (value + divisor - 1) / divisor;
}

// Unlike flash_attn_mask_to_KV_max, this scanner treats the final partial
// FATTN_KQ_STRIDE chunk as masked padding instead of reading past n_kv.
template <int ncols1>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void flash_attn_mask_to_KV_max_tail_safe(
        const half2 * __restrict__ mask,
        int * __restrict__ KV_max,
        const int n_q,
        const int n_kv,
        const int s31,
        const int s33) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;
    const int lane     = threadIdx.x;
    const int nchunks  = wmma_gfx1151_ceil_div(n_kv, FATTN_KQ_STRIDE);
    const int q0       = jt * ncols1;

    mask += sequence * s33 + q0 * s31;

    int upper = n_kv;
    for (int chunk = nchunks - 1; chunk >= 0; --chunk) {
        const int key0 = chunk * FATTN_KQ_STRIDE;
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            if (q0 + j < n_q) {
                const half * mask_row = (const half *) mask + j * (2 * s31);
#pragma unroll
                for (int pair = 0; pair < FATTN_KQ_STRIDE / (2 * WARP_SIZE); ++pair) {
                    const int key = key0 + 2 * (lane + pair * WARP_SIZE);
                    if (key < n_kv) {
                        all_inf = all_inf && int(isinf((float) mask_row[key]));
                    }
                    if (key + 1 < n_kv) {
                        all_inf = all_inf && int(isinf((float) mask_row[key + 1]));
                    }
                }
            }
        }

        all_inf = warp_reduce_all(all_inf);
        if (!all_inf) {
            upper = min(key0 + FATTN_KQ_STRIDE, n_kv);
            break;
        }
        upper = key0;
    }

    if (lane == 0) {
        KV_max[sequence * gridDim.x + jt] = upper;
    }
#else
    GGML_UNUSED_VARS(mask, KV_max, n_q, n_kv, s31, s33);
#endif
}

template <int DKQ, int DV>
__global__ void __launch_bounds__(8 * WARP_SIZE, 1)
flash_attn_wmma_gfx1151_kernel(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    const int  * __restrict__ KV_max,
    float      * __restrict__ dst,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head_q,
    const int n_head_kv,
    const int gqa_ratio,
    const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb11, const int64_t nb12, const int64_t nb13,
    const int64_t nb21, const int64_t nb22, const int64_t nb23,
    const int64_t nb31, const int64_t nb33, const int ne33) {

#if defined(__HIP_DEVICE_COMPILE__)
    const int lane    = threadIdx.x % WARP_SIZE;
    const int wave_id = threadIdx.x / WARP_SIZE;
    const int warp_m0 = blockIdx.x * 16;
    if (warp_m0 >= n_q) return; // Uniform for the whole workgroup.

    const int head_kv    = blockIdx.y / 4;
    const int gqa_group  = blockIdx.y % 4;
    const int head_slot  = wave_id / 4;
    const int pv_quarter = wave_id % 4;
    const int head_q     = head_kv * gqa_ratio + gqa_group * 2 + head_slot;
    const int seq       = blockIdx.z;

    const int m_in_tile = lane % 16;
    const int col_idx = lane % 16;
    const int is_odd_row = (lane >= 16 ? 1 : 0);

    __shared__ _Float16 K_lds[16][256 + 16];
    __shared__ _Float16 V_lds_trans[256][16 + 2];
    __shared__ _Float16 P_lds[2][16][16 + 2];
    __shared__ float O_scale_lds[2][16];
    __shared__ float O_sum_lds[2][16];

    float row_max[8];
    float row_sum[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        row_max[i] = -1e20f;
        row_sum[i] = 0.0f;
    }

    float8_amdgcn o_acc[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            o_acc[s][i] = 0.0f;
        }
    }

    const char * K_head = K + seq * nb13 + head_kv * nb12;
    const char * V_head = V + seq * nb23 + head_kv * nb22;
    const char * mask_seq = mask ? (mask + (seq % ne33) * nb33) : nullptr;

    int max_k = n_kv;
    if (KV_max) {
        int max_k_val = KV_max[seq * gridDim.x + blockIdx.x];
        if (max_k_val < max_k) max_k = max_k_val;
    }

    for (int k_step0 = 0; k_step0 < max_k; k_step0 += 16) {
        // Cooperative loading of K tile (16 tokens x 256 dims = 4096 halfs) into K_lds
        for (int idx = threadIdx.x; idx < 512; idx += 8 * WARP_SIZE) {
            int k_row = idx >> 5; // 0..15
            int k_col8 = (idx & 31) << 3; // 0..248
            int k_global = k_step0 + k_row;
            if (k_global < n_kv) {
                const _Float16 * K_row = (const _Float16 *)(K_head + k_global * nb11);
                *((float4 *)(&K_lds[k_row][k_col8])) = *((const float4 *)(&K_row[k_col8]));
            } else {
                *((float4 *)(&K_lds[k_row][k_col8])) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // Cooperative loading of V tile into V_lds_trans [256 cols][18 rows]
        for (int idx = threadIdx.x; idx < 512; idx += 8 * WARP_SIZE) {
            int v_row = idx >> 5; // 0..15 (k dim)
            int v_col8 = (idx & 31) << 3; // 0..248
            int k_global = k_step0 + v_row;
            if (k_global < n_kv) {
                const _Float16 * V_row = (const _Float16 *)(V_head + k_global * nb21);
                float4 v_f4 = *((const float4 *)(&V_row[v_col8]));
                const _Float16 * v_h = (const _Float16 *)(&v_f4);
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    V_lds_trans[v_col8 + c][v_row] = v_h[c];
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    V_lds_trans[v_col8 + c][v_row] = (_Float16)0.0f;
                }
            }
        }
        __syncthreads();

        float8_amdgcn S_acc;
        #pragma unroll
        for (int i = 0; i < 8; ++i) S_acc[i] = 0.0f;

        if (pv_quarter == 0) {
#pragma unroll 1
            for (int s = 0; s < 16; ++s) {
                half16_amdgcn_u q_u;
                const int m_global = warp_m0 + m_in_tile;
                if (m_global < n_q) {
                    const float * Q_row = (const float *) (Q + seq * nb03 + head_q * nb02 + m_global * nb01);
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        q_u.h[k] = (_Float16) (Q_row[s * 16 + k] * scale);
                    }
                } else {
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        q_u.h[k] = (_Float16) 0.0f;
                    }
                }

                half16_amdgcn_u k_u;
                k_u.f4[0] = *((const float4 *)(&K_lds[col_idx][s * 16 + 0]));
                k_u.f4[1] = *((const float4 *)(&K_lds[col_idx][s * 16 + 8]));

                S_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_u.vec, k_u.vec, S_acc);
            }
        }

        if (pv_quarter == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m_row = 2 * i + lane / 16;
                const int m_cur = warp_m0 + m_row;
                const int n_cur = k_step0 + col_idx;
                float s_val = S_acc[i];
                if (n_cur >= n_kv || m_cur >= n_q) {
                    s_val = -1e20f;
                } else if (mask_seq) {
                    const _Float16 * mask_row = (const _Float16 *) (mask_seq + m_cur * nb31);
                    s_val += (float) mask_row[n_cur];
                }

                float r_max = s_val;
                r_max = fmaxf(r_max, __shfl_xor(r_max, 8));
                r_max = fmaxf(r_max, __shfl_xor(r_max, 4));
                r_max = fmaxf(r_max, __shfl_xor(r_max, 2));
                r_max = fmaxf(r_max, __shfl_xor(r_max, 1));

                const float new_m = fmaxf(row_max[i], r_max);
                const float exp_old = expf(row_max[i] - new_m);
                const float exp_val = expf(s_val - new_m);
                float r_sum = exp_val;
                r_sum += __shfl_xor(r_sum, 8);
                r_sum += __shfl_xor(r_sum, 4);
                r_sum += __shfl_xor(r_sum, 2);
                r_sum += __shfl_xor(r_sum, 1);

                row_sum[i] = row_sum[i] * exp_old + r_sum;
                row_max[i] = new_m;
                P_lds[head_slot][m_row][col_idx] = (_Float16) exp_val;
                if (col_idx == 0) {
                    O_scale_lds[head_slot][m_row] = exp_old;
                    O_sum_lds[head_slot][m_row] = row_sum[i];
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m_row = 2 * i + lane / 16;
            const float exp_old = O_scale_lds[head_slot][m_row];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                o_acc[s][i] *= exp_old;
            }
        }

        half16_amdgcn_u p_u;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            p_u.h[i] = P_lds[head_slot][m_in_tile][i];
        }

#pragma unroll 1
        for (int s = 0; s < 4; ++s) {
            const int v_col = (pv_quarter * 4 + s) * 16 + col_idx;
            half16_amdgcn_u v_u;
            v_u.f4[0] = *((const float4 *)(&V_lds_trans[v_col][0]));
            v_u.f4[1] = *((const float4 *)(&V_lds_trans[v_col][8]));

            o_acc[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_u.vec, v_u.vec, o_acc[s]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m_row = 2 * i + lane / 16;
        const int m_cur = warp_m0 + m_row;
        if (m_cur < n_q) {
            const float row_sum_final = O_sum_lds[head_slot][m_row];
            const float inv_sum = (row_sum_final > 0.0f) ? (1.0f / row_sum_final) : 0.0f;
            const int j_dst_unrolled = (seq * n_q + m_cur) * n_head_q + head_q;
            float * dst_row = dst + j_dst_unrolled * 256;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int v_col = (pv_quarter * 4 + s) * 16 + col_idx;
                dst_row[v_col] = o_acc[s][i] * inv_sum;
            }
        }
    }
#else
    (void)Q; (void)K; (void)V; (void)mask; (void)KV_max; (void)dst;
    (void)scale; (void)n_q; (void)n_kv; (void)n_head_q; (void)n_head_kv; (void)gqa_ratio;
    (void)nb01; (void)nb02; (void)nb03; (void)nb11; (void)nb12; (void)nb13;
    (void)nb21; (void)nb22; (void)nb23; (void)nb31; (void)nb33; (void)ne33;
#endif
}
#endif // defined(GGML_USE_HIP)

template <int DKQ, int DV>
bool launch_fattn_wmma_gfx1151(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = std::getenv("GGML_CUDA_EXPERIMENTAL_GFX1151_INTRINSIC_WMMA") != nullptr;
    if (!enabled) {
        return false;
    }
    if constexpr (DKQ != 256 || DV != 256) {
        return false;
    } else {

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if ((cc & 0xffff) != 0x1151) {
        return false;
    }

    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // Sink recurrence is not implemented in this experimental path.
    if (sinks != nullptr) {
        return false;
    }

    // Only accelerate prefill when token count is sufficient (>= 16)
    if (Q->ne[1] < 16) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) KQV->op_params + 0, sizeof(float));

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    ggml_cuda_pool_alloc<half> K_f16(pool);
    ggml_cuda_pool_alloc<half> V_f16(pool);
    ggml_cuda_pool_alloc<int>  KV_max(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (const char *) K_f16.ptr;
    }

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    if (V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
                V_data = (const char *) V_f16.ptr;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
                V_data = (const char *) V_f16.ptr;
            }
        }
    }

    const int n_q = Q->ne[1];
    const int n_kv = K->ne[1];
    const int n_head_q = Q->ne[2];
    const int n_head_kv = K->ne[2];
    const int gqa_ratio = n_head_q / n_head_kv;
    const int n_seq = Q->ne[3];
    if (n_head_kv <= 0 || n_head_q % n_head_kv != 0 || gqa_ratio != 8) {
        return false;
    }

    const int ntiles_x = (n_q + 15) / 16;

    if (mask && (n_q >= 1024 || n_seq > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);
        const dim3 blocks_num_KV_max(ntiles_x, n_seq, 1);
        const dim3 block_dim_KV_max(WARP_SIZE, 1, 1);
        const int nchunks = wmma_gfx1151_ceil_div(n_kv, FATTN_KQ_STRIDE);
        GGML_ASSERT(nchunks > 0);
        KV_max.alloc(blocks_num_KV_max.x * blocks_num_KV_max.y);
        flash_attn_mask_to_KV_max_tail_safe<16><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, n_q, n_kv, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    dim3 grid(ntiles_x, 4 * n_head_kv, n_seq);
    dim3 block(8 * WARP_SIZE, 1, 1);

    flash_attn_wmma_gfx1151_kernel<DKQ, DV><<<grid, block, 0, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? (const char *) mask->data : nullptr,
        KV_max.ptr,
        (float *) KQV->data,
        scale,
        n_q,
        n_kv,
        n_head_q,
        n_head_kv,
        gqa_ratio,
        Q->nb[1], Q->nb[2], Q->nb[3],
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1
    );
    CUDA_CHECK(cudaGetLastError());

    return true;
    }
#else
    return false;
#endif // GGML_USE_HIP
}
