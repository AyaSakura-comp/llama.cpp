// gfx11 (RDNA3/3.5, tuned on gfx1151) WMMA flash attention for long-context prefill, D = DV = 256, GQA ratio 8.
//
// One workgroup = 16 tokens x 8 GQA heads of one KV head (128 query rows), 16 waves. Wave w owns 16 rows
// (w / 2) and half of DV (w % 2). The math runs transposed on v_wmma_f32_16x16x16_f16:
//   S^T = K Q^T  (A = K tile from LDS, B = Q fragments kept in registers)
//   O^T = V^T P^T (A = V^T tile from LDS, B = P^T built in registers)
// so in the C layout every lane owns one query row (lane % 16): the softmax max/sum are lane-local plus one
// permlanex16 swap with lane ^ 16, and alpha rescales O^T lane-locally. KV tiles of 16 cells are double-buffered in
// LDS; tile i+1 is prefetched into registers during iteration i-1 and written at the start of iteration i, so there
// is a single barrier per tile. 128 rows per KV read halves the L2->memory traffic of a 64-row block.
//
// Measured standalone (2048 tokens after 131072 cached cells, llama.cpp KV layout): 214 ms vs 694 ms for
// flash_attn_tile<256,256,4,8> (q4_0 KV path), fp64-reference cosine 0.9999999.
//
// gfx11 wave32 WMMA fragment layout: A lane l holds row l%16, all 16 k (lanes 16..31 repeat 0..15); B lane l holds
// column l%16, all 16 k; C/D lane l holds rows 2*i + l/16 (i = 0..7) of column l%16.

#include "fattn-gfx1151.cuh"
#include "convert.cuh"

#include <cstdlib>
#include <cstring>

namespace fa1151 {

constexpr int D    = 256;
constexpr int G    = 8;          // GQA ratio
constexpr int NTOK = 16;         // tokens per workgroup
constexpr int NW   = 16;         // waves per workgroup
constexpr int NT   = NW * 32;    // 512 threads
constexpr int BN   = 16;         // KV cells per tile
constexpr int DVH  = D / 2;      // O^T dims per wave
constexpr int QS   = D + 8;      // padded LDS row stride (halves) of the K tile

struct params {
    const float * q; const half * k; const half * v; const half * mask; float * o;
    int64_t q_nb1, q_nb2, k_nb1, k_nb2, v_nb1, v_nb2, m_nb1;
    int nq, nkv, n_head, n_head_kv;
    float scale;
};

#if defined(RDNA3)
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float    f8  __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ f8 wmma(h16 a, h16 b, f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
static __device__ __forceinline__ float swap16(float x) {   // the value held by lane ^ 16
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
}
static __device__ __forceinline__ uint32_t swap16u(uint32_t x) {
    return (uint32_t) __builtin_amdgcn_permlanex16((int) x, (int) x, 0x76543210, 0xfedcba98, false, false);
}
#endif // defined(RDNA3)

static __global__ void __launch_bounds__(NT) flash_attn_gfx1151(const params p) {
#if defined(RDNA3)
    __shared__ __align__(16) _Float16 Ks[2][BN * QS];
    __shared__ __align__(16) _Float16 Vt[2][D * BN];
    __shared__ __align__(16) _Float16 Ms[2][NTOK * BN];
    __shared__ int kv_end_s;

    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int col = lane & 15, hf = lane >> 4;
    const int rg = w >> 1, dh = w & 1;
    const int tok0 = blockIdx.x * NTOK, kvh = blockIdx.y;
    const float qscale = p.scale * 1.44269504088896341f;   // softmax in base 2
    const int R = rg * 16 + col, tl = R / G;                // this lane's query row: token tl, head R % G

    // Q as the B operand: lane holds row R, 16 consecutive dims per fragment, all 256 dims
    h16 qb[D / 16];
    {
        const int tok = tok0 + tl, h = kvh * G + R % G;
        const float * src = (const float *) ((const char *) p.q + (int64_t) min(tok, p.nq - 1) * p.q_nb1 + (int64_t) h * p.q_nb2);
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                const float4 x = *(const float4 *) (src + kk * 16 + e);
                qb[kk][e + 0] = (_Float16) (x.x * qscale); qb[kk][e + 1] = (_Float16) (x.y * qscale);
                qb[kk][e + 2] = (_Float16) (x.z * qscale); qb[kk][e + 3] = (_Float16) (x.w * qscale);
            }
        }
    }
    // KV range of this block: everything after the last cell the block's last token can see is masked
    if (t == 0) {
        const int tlast = min(NTOK, p.nq - tok0) - 1;
        const half * mrow = (const half *) ((const char *) p.mask + (int64_t) (tok0 + tlast) * p.m_nb1);
        int e = p.nkv;
        while (e > 0 && __half2float(mrow[e - 1]) == -INFINITY) {
            e--;
        }
        kv_end_s = e;
    }

    const char * kbase = (const char *) p.k + (int64_t) kvh * p.k_nb2;
    const char * vbase = (const char *) p.v + (int64_t) kvh * p.v_nb2;
    // loader: consecutive lanes -> consecutive cells (the transposed V writes don't bank-conflict); 8 dims each
    const int lc = t & 15, ld8 = (t >> 4) * 8;
    uint4 pk, pv, pm;
    auto gload = [&](int c0) {
        const int cell = c0 + lc;
        pk = make_uint4(0, 0, 0, 0); pv = pk; pm = make_uint4(0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u);
        if (cell < p.nkv) {
            pk = *(const uint4 *) (kbase + cell * p.k_nb1 + ld8 * 2);
            pv = *(const uint4 *) (vbase + cell * p.v_nb1 + ld8 * 2);
        }
        if (t < 2 * NTOK) {
            const int tok = tok0 + (t >> 1), cb = c0 + (t & 1) * 8;
            if (tok < p.nq && cb + 8 <= p.nkv) {
                pm = *(const uint4 *) ((const char *) p.mask + (int64_t) tok * p.m_nb1 + cb * 2);
            } else if (tok < p.nq) {
                _Float16 tmp[8];
                for (int e = 0; e < 8; e++) {
                    tmp[e] = cb + e < p.nkv ? *(const _Float16 *) ((const char *) p.mask + (int64_t) tok * p.m_nb1 + (cb + e) * 2) : (_Float16) -INFINITY;
                }
                pm = *(uint4 *) tmp;
            }
        }
    };
    auto lstore = [&](int buf) {
        *(uint4 *) (Ks[buf] + lc * QS + ld8) = pk;
        const _Float16 * vh = (const _Float16 *) &pv;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            Vt[buf][(ld8 + e) * BN + lc] = vh[e];
        }
        if (t < 2 * NTOK) {
            *(uint4 *) (Ms[buf] + (t >> 1) * BN + (t & 1) * 8) = pm;
        }
    };

    float m_r = -INFINITY, l_r = 0.f;   // softmax state of row R (identical in both lane halves)
    f8 acc[DVH / 16];                   // O^T: acc[j][i] = O[R][dh*128 + j*16 + 2i + hf]
#pragma unroll
    for (int j = 0; j < DVH / 16; j++) {
        acc[j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    }

    __syncthreads();
    const int kv_end = kv_end_s;
    gload(0);
    lstore(0);
    if (BN < kv_end) {
        gload(BN);
    }
    __syncthreads();

    int buf = 0;
    for (int c0 = 0; c0 < kv_end; c0 += BN, buf ^= 1) {
        // publish tile c0+BN (in registers) into the other buffer, then prefetch tile c0+2*BN
        if (c0 + BN < kv_end) {
            lstore(buf ^ 1);
            if (c0 + 2 * BN < kv_end) {
                gload(c0 + 2 * BN);
            }
        }

        // S^T: s[i] = S[R][cell c0 + 2i + hf]
        f8 s = f8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            const h16 a = *(const h16 *) (Ks[buf] + col * QS + kk * 16);
            s = wmma(a, qb[kk], s);
        }

        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float mv = (float) Ms[buf][tl * BN + 2 * i + hf];
            s[i] = mv == -INFINITY ? -INFINITY : s[i] + mv * 1.44269504088896341f;
            mx = fmaxf(mx, s[i]);
        }
        mx = fmaxf(mx, swap16(mx));
        const float mn = fmaxf(m_r, mx);
        const float alpha = mn == -INFINITY ? 1.f : exp2f(m_r - mn);
        m_r = mn;
        float ps = 0.f;
        _Float16 ph[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float e = mn == -INFINITY ? 0.f : exp2f(s[i] - mn);
            ps += e;
            ph[i] = (_Float16) e;
        }
        ps += swap16(ps);
        l_r = l_r * alpha + ps;
        // P^T B fragment: the lane needs P[R][k], k = 0..15; it holds k = 2i + hf, lane ^ 16 holds k = 2i + (1 - hf)
        h16 pb;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const uint32_t own = (uint32_t) __builtin_bit_cast(uint16_t, ph[i]) | ((uint32_t) __builtin_bit_cast(uint16_t, ph[i + 1]) << 16);
            const uint32_t oth = swap16u(own);
            const _Float16 o0 = __builtin_bit_cast(_Float16, (uint16_t) (oth & 0xFFFF));
            const _Float16 o1 = __builtin_bit_cast(_Float16, (uint16_t) (oth >> 16));
            pb[2 * i + 0] = hf ? o0 : ph[i];
            pb[2 * i + 1] = hf ? ph[i] : o0;
            pb[2 * i + 2] = hf ? o1 : ph[i + 1];
            pb[2 * i + 3] = hf ? ph[i + 1] : o1;
        }
#pragma unroll
        for (int j = 0; j < DVH / 16; j++) {
            acc[j] *= alpha;
            const h16 a = *(const h16 *) (Vt[buf] + (dh * DVH + j * 16 + col) * BN);
            acc[j] = wmma(a, pb, acc[j]);
        }
        __syncthreads();
    }

    const int tok = tok0 + tl;
    if (tok < p.nq) {
        const float inv = l_r > 0.f ? 1.f / l_r : 0.f;
        float * dst = p.o + ((int64_t) tok * p.n_head + kvh * G + R % G) * D + dh * DVH;
#pragma unroll
        for (int j = 0; j < DVH / 16; j++) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                dst[j * 16 + 2 * i + hf] = acc[j][i] * inv;
            }
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

} // namespace fa1151

static bool ggml_cuda_gfx1151_wmma_fa_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("GGML_CUDA_GFX1151_WMMA_FA");
        return e != nullptr && std::strcmp(e, "0") != 0;
    }();
    return enabled;
}

bool ggml_cuda_flash_attn_ext_gfx1151(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_gfx1151_wmma_fa_enabled()) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }

    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // eligibility: prefill-sized batches of the D=256 / GQA-8 shape, one sequence plane, plain causal-style mask
    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != fa1151::D || V->ne[0] != fa1151::D || K->ne[0] != fa1151::D ||
        Q->ne[1] < 64 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
        K->ne[2] == 0 || Q->ne[2] != fa1151::G * K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1] ||
        mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
        mask->ne[2] != 1 || mask->ne[3] != 1 || sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || mask->nb[1] % 16 != 0 ||
        K->ne[1] > INT32_MAX || Q->ne[1] > INT32_MAX) {
        return false;
    }
    if ((K->type != GGML_TYPE_F16 && ggml_get_to_fp16_cuda(K->type) == nullptr) ||
        (V->type != GGML_TYPE_F16 && ggml_get_to_fp16_cuda(V->type) == nullptr)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half> K_f16(pool);
    ggml_cuda_pool_alloc<half> V_f16(pool);

    // K/V as f16 with their original (KV-cache) strides; quantized caches are converted like launch_fattn does
    auto to_f16 = [&](const ggml_tensor * T, ggml_cuda_pool_alloc<half> & buf, const char *& data, size_t & nb1, size_t & nb2) {
        data = (const char *) T->data;
        nb1  = T->nb[1];
        nb2  = T->nb[2];
        if (T->type == GGML_TYPE_F16) {
            return;
        }
        const size_t bs = ggml_blck_size(T->type);
        const size_t ts = ggml_type_size(T->type);
        buf.alloc(ggml_nelements(T));
        if (ggml_is_contiguously_allocated(T)) {
            ggml_get_to_fp16_cuda(T->type)(data, buf.ptr, ggml_nelements(T), stream);
            nb1 = nb1 * bs * sizeof(half) / ts;
            nb2 = nb2 * bs * sizeof(half) / ts;
        } else {
            GGML_ASSERT(T->nb[0] == ts);
            ggml_get_to_fp16_nc_cuda(T->type)(data, buf.ptr, T->ne[0], T->ne[1], T->ne[2], T->ne[3],
                                              nb1 / ts, nb2 / ts, T->nb[3] / ts, stream);
            nb1 = T->ne[0] * sizeof(half);
            nb2 = T->ne[1] * nb1;
        }
        data = (const char *) buf.ptr;
    };
    const char * K_data; size_t k_nb1, k_nb2;
    const char * V_data; size_t v_nb1, v_nb2;
    to_f16(K, K_f16, K_data, k_nb1, k_nb2);
    to_f16(V, V_f16, V_data, v_nb1, v_nb2);
    if (k_nb1 % 16 != 0 || k_nb2 % 16 != 0 || v_nb1 % 16 != 0 || v_nb2 % 16 != 0) {
        GGML_ABORT("gfx1151 WMMA FA: unaligned K/V strides");
    }

    fa1151::params p;
    p.q = (const float *) Q->data; p.k = (const half *) K_data; p.v = (const half *) V_data;
    p.mask = (const half *) mask->data; p.o = (float *) dst->data;
    p.q_nb1 = Q->nb[1]; p.q_nb2 = Q->nb[2];
    p.k_nb1 = k_nb1; p.k_nb2 = k_nb2; p.v_nb1 = v_nb1; p.v_nb2 = v_nb2;
    p.m_nb1 = mask->nb[1];
    p.nq = (int) Q->ne[1]; p.nkv = (int) K->ne[1]; p.n_head = (int) Q->ne[2]; p.n_head_kv = (int) K->ne[2];
    p.scale = scale;

    const dim3 grid((p.nq + fa1151::NTOK - 1) / fa1151::NTOK, p.n_head_kv, 1);
    fa1151::flash_attn_gfx1151<<<grid, fa1151::NT, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
