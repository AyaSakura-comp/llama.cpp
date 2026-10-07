#include <string>
#include <vector>
#include <cstdio>
#include "gated_delta_net.cuh"
#include "gdn-fused.cuh"

template <int S_v, bool KDA, int num_warps = 4>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * num_warps, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    const float * q_ptr    = q + iq3 * sq3 + iq1 * sq1;
    const float * k_ptr    = k + iq3 * sq3 + iq1 * sq1;
    const float * v_ptr    = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base  = sequence * sb3 + h_idx * sb1;
    const float * beta_ptr = beta + gb_base;
    const float * g_ptr    = g    + gb_base * (KDA ? S_v : 1);
    const int64_t g_stride = sb2 * (KDA ? S_v : 1);

    for (int t = 0; t < n_tokens; t++) {
        const float beta_val = *beta_ptr;

        // Directly load k and q from L1/L2 cache into registers (vectorized, barrier-free)
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_ptr[i];
            q_reg[r] = q_ptr[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_ptr);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_ptr[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_ptr[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_ptr[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_ptr[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
        q_ptr    += sq2;
        k_ptr    += sq2;
        v_ptr    += sv2;
        beta_ptr += sb2;
        g_ptr    += g_stride;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int cc         = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size  = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps  = S_v == 128 && !KDA && GGML_CUDA_CC_IS_RDNA3_5(cc) ? 8 : 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            if (!KDA && GGML_CUDA_CC_IS_RDNA3_5(cc)) {
                gated_delta_net_cuda<128, KDA, 8><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            } else {
                gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// fused conv1d + SiLU + l2norm(q,k) + GDN (ggml_gated_delta_net_conv), gfx11 wave32 only
static void ggml_cuda_op_gated_delta_net_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * cs = dst->src[1];
    const ggml_tensor * cw = dst->src[2];
    const ggml_tensor * g  = dst->src[3];
    const ggml_tensor * b  = dst->src[4];
    const ggml_tensor * st = dst->src[5];
    GGML_ASSERT(st->ne[0] == gdnf::D && st->ne[2] == gdnf::HV && x->ne[0] == gdnf::CH && cw->ne[0] == gdnf::KW);
    const float eps = ggml_get_op_params_f32(dst, 1);
    const int T = (int) x->ne[1], n_seqs = (int) x->ne[2];
    // debug: GDN_FUSED_DUMP=<dir> dumps the first call's inputs (raw fp32) + a meta line
    static int dumped = 0, ncall = 0;
    const int want = getenv("GDN_FUSED_DUMP_CALL") ? atoi(getenv("GDN_FUSED_DUMP_CALL")) : 0;
    if (T >= 64 && ncall++ == want && getenv("GDN_FUSED_DUMP") && !dumped) {
        dumped = 1;
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        const std::string d = getenv("GDN_FUSED_DUMP");
        auto dump = [&](const ggml_tensor * t, const char * name) {
            std::vector<char> h(ggml_nbytes(t));
            CUDA_CHECK(cudaMemcpy(h.data(), t->data, h.size(), cudaMemcpyDeviceToHost));
            FILE * f = fopen((d + "/" + name + ".bin").c_str(), "wb"); fwrite(h.data(), 1, h.size(), f); fclose(f);
        };
        dump(x, "x"); dump(cs, "cstate"); dump(cw, "cw"); dump(g, "g"); dump(b, "beta"); dump(st, "state");
        FILE * f = fopen((d + "/meta.txt").c_str(), "w");
        fprintf(f, "T=%d n_seqs=%d eps=%g x_ne=%lld,%lld,%lld x_nb=%zu,%zu,%zu cs_ne=%lld,%lld,%lld cw_ne=%lld,%lld g_ne=%lld,%lld,%lld,%lld st_ne=%lld,%lld,%lld,%lld\n",
            T, n_seqs, eps, (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], x->nb[0], x->nb[1], x->nb[2],
            (long long) cs->ne[0], (long long) cs->ne[1], (long long) cs->ne[2], (long long) cw->ne[0], (long long) cw->ne[1],
            (long long) g->ne[0], (long long) g->ne[1], (long long) g->ne[2], (long long) g->ne[3],
            (long long) st->ne[0], (long long) st->ne[1], (long long) st->ne[2], (long long) st->ne[3]);
        fclose(f);
    }
    gdnf::launch_ws<8>((const float *) x->data, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
        (const float *) cs->data, (const float *) cw->data, (const float *) g->data, (const float *) b->data,
        (const float *) st->data, (float *) dst->data, T, n_seqs, eps, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("gated_delta_net_conv: HIP only");
#endif
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) == 1) {
        ggml_cuda_op_gated_delta_net_conv(ctx, dst);
        return;
    }
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}
