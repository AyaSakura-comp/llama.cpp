// Fused Qwen3.6 GDN prefill kernel for gfx1151 (RDNA3.5, wave32):
//   causal conv1d(4) + SiLU + l2norm(q,k) + gated delta rule recurrence
// replacing CONCAT -> SSM_CONV -> SILU -> L2_NORM(q) -> L2_NORM(k) -> GATED_DELTA_NET.
//
// Math per v-head h, token t (all fp32, same as ggml):
//   xc   = SiLU( sum_k in[t+k] * w[k] ),  in = [conv_state(3) | x(T)]   (per channel)
//   q,k  = l2norm over 128 (scale = rsqrt(max(sum x^2, eps^2)))
//   S    = e^g S;  delta = beta (v - S^T k);  S += k delta^T;  o = scale_attn * S^T q
//
// Parallel layout: block = (v-head h, 64 of its 128 columns), 256 threads.
// A column j is owned by a quad of lanes (G = 4), each holding R = 32 rows of
// S[:, j] in registers -> S^T k and S^T q need only two quad DPP steps.
// Tokens are processed in chunks of C: the block cooperatively computes the
// chunk's q/k (256 channels, conv+SiLU+l2norm) and v (its 64 channels) into
// LDS, then every quad runs the recurrence reading LDS (broadcast).
//
// Software pipeline:
//   * chunk level: global loads of chunk c+1's raw qkv / g / beta are issued
//     into registers before chunk c's recurrence, so their latency hides
//     behind C tokens of math;
//   * token level: q/k rows of token t+1 are read from LDS into a second
//     register set while token t is being computed (explicit double buffer +
//     sched_group_barrier interleaving of DS reads with VALU);
//   * outputs go to an LDS staging tile and leave as one coalesced store per
//     chunk.
#pragma once
#include <stdint.h>
#if defined(GGML_USE_HIP)

namespace gdnf {

constexpr int D    = 128;   // head dim (k and v)
constexpr int HV   = 32;    // v heads
constexpr int HK   = 16;    // q/k heads
constexpr int CH   = 2 * HK * D + HV * D;  // 8192 conv channels (q | k | v)
constexpr int KW   = 4;     // conv width


// ---- DPP helpers --------------------------------------------------------
#define DPP_QUAD_XOR1 0xB1   // quad_perm [1,0,3,2]
#define DPP_QUAD_XOR2 0x4E   // quad_perm [2,3,0,1]
#define DPP_ROW_ROR4  0x124
#define DPP_ROW_ROR8  0x128

#define DPP_XMASK(m) (0x160 + (m))
template <int G>
__device__ __forceinline__ float group_sum(float x) {
    static_assert(G == 2 || G == 4 || G == 8 || G == 16, "G");
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_XMASK(1), 0xF, 0xF, false));
    if constexpr (G >= 4)  x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_XMASK(2), 0xF, 0xF, false));
    if constexpr (G >= 8)  x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_XMASK(4), 0xF, 0xF, false));
    if constexpr (G >= 16) x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_XMASK(8), 0xF, 0xF, false));
    return x;
}
__device__ __forceinline__ float quad_sum(float x) {
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_QUAD_XOR1, 0xF, 0xF, false));
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_QUAD_XOR2, 0xF, 0xF, false));
    return x;
}
__device__ __forceinline__ float wave_sum(float x) {
    x = quad_sum(x);
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_ROW_ROR4, 0xF, 0xF, false));
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), DPP_ROW_ROR8, 0xF, 0xF, false));
    // swap 16-lane rows
    x += __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x),
                                                     0x76543210, 0xfedcba98, false, false));
    return x;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// Block barrier that only orders LDS: wait for this wave's LDS ops, then s_barrier.
// Unlike __syncthreads() it does NOT drain outstanding global loads (vmcnt), so register
// prefetches issued just before the barrier stay in flight across it.
__device__ __forceinline__ void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}

// sched_group_barrier masks
#define SG_VALU   0x002
#define SG_DSREAD 0x100

template <int C, int NCOL>
struct Smem {
    alignas(16) float qk[C][2 * D];     // normalised q (0..127) | k (128..255)
    alignas(16) float v[C][NCOL];
    alignas(16) float out[C][NCOL];
    alignas(8)  float2 egb[C];          // (exp(g), beta)
};

// x      : qkv_mixed  [n_seqs][T][CH]           (row stride x_s1 floats, seq stride x_s2)
// cstate : conv state [n_seqs][CH][KW-1]         (tap 0 = oldest)
// cw     : conv weight [CH][KW]
// gl     : gate (log decay) [n_seqs][T][HV];  bt : beta [n_seqs][T][HV]
// s_in   : state [n_seqs][HV][D (col j)][D (row i)]  (ggml transposed layout)
// dst    : attn [n_seqs][T][HV][D] followed by state_out [n_seqs][HV][D][D]
// C: tokens per chunk; COLS: columns per block; MC: columns per lane; G: lanes per column group
template <int C, int COLS, int MC, int G>
__global__ void __launch_bounds__((COLS / MC) * G, 1)
gdn_fused_prefill(const float * __restrict__ x, int64_t x_s1, int64_t x_s2,
                  const float * __restrict__ cstate, const float * __restrict__ cw,
                  const float * __restrict__ gl, const float * __restrict__ bt,
                  const float * __restrict__ s_in, float * __restrict__ dst,
                  int T, int n_seqs, float eps, float scale_attn) {
    constexpr int R  = D / G;                   // rows per lane
    constexpr int NT = (COLS / MC) * G;
    constexpr int BPH = D / COLS;               // blocks per head
    __shared__ Smem<C, COLS> sm;
    const int tid  = threadIdx.x;
    const int h    = blockIdx.x / BPH;
    const int col0 = (blockIdx.x % BPH) * COLS;
    const int seq  = blockIdx.y;
    const int kh   = h % HK;

    // ---- per-thread conv roles --------------------------------------------
    // role A: q/k channels of the block's k-head (q: 0..127, k: 128..255), NA per thread
    constexpr int NA = (2 * D + NT - 1) / NT;   // channels per thread (>= 1)
    int ch_a[NA], qk_c[NA];
#pragma unroll
    for (int a = 0; a < NA; a++) {
        qk_c[a] = tid + a * NT;
        const int c = qk_c[a] < 2 * D ? qk_c[a] : 0;          // idle threads alias channel 0 (never stored)
        ch_a[a] = (c < D) ? (kh * D + c) : (HK * D + kh * D + (c - D));
    }
    // role B (tid < 64): v channel of column col0 + tid
    const bool has_v = tid < COLS;
    const int ch_b  = 2 * HK * D + h * D + col0 + (tid & (COLS - 1));

    const float * xs = x + (int64_t) seq * x_s2;
    float wa[NA][KW], wb[KW];
    float win_a[NA][KW - 1], win_b[KW - 1];   // sliding conv windows, seeded from the conv state
#pragma unroll
    for (int k = 0; k < KW; k++) wb[k] = cw[ch_b * KW + k];
#pragma unroll
    for (int k = 0; k < KW - 1; k++) win_b[k] = cstate[((int64_t) seq * CH + ch_b) * (KW - 1) + k];
#pragma unroll
    for (int a = 0; a < NA; a++) {
#pragma unroll
        for (int k = 0; k < KW; k++) wa[a][k] = cw[ch_a[a] * KW + k];
#pragma unroll
        for (int k = 0; k < KW - 1; k++) win_a[a][k] = cstate[((int64_t) seq * CH + ch_a[a]) * (KW - 1) + k];
    }

    // ---- recurrence role --------------------------------------------------
    const int cg = tid / G;             // column group
    const int gq = tid % G;             // lane within the group
    // rows are interleaved in float4 units across the G lanes: row(r) = (r/4)*4G + 4*gq + r%4,
    // so one ds_load_b128 by a group covers 4G contiguous floats (no LDS bank conflicts)
    auto roff = [&](int r) { return (r >> 2) * (4 * G) + 4 * gq; };
    float s[MC][R];
    float c = 1.0f;                     // lazy-decay scale: S_true = c * s
#pragma unroll
    for (int m = 0; m < MC; m++) {
        const int j = col0 + cg * MC + m;
        const float * sp = s_in + (((int64_t) seq * HV + h) * D + j) * D;
#pragma unroll
        for (int r = 0; r < R; r += 4) {
            const float4 v4 = *reinterpret_cast<const float4 *>(sp + roff(r));
            s[m][r] = v4.x; s[m][r + 1] = v4.y; s[m][r + 2] = v4.z; s[m][r + 3] = v4.w;
        }
    }

    // ---- register prefetch of one chunk ------------------------------------
    float ra[NA][C], rb[C], rg = 0.0f;
    auto prefetch = [&](int t0) {
#pragma unroll
        for (int t = 0; t < C; t++) {
            const int tt = t0 + t;
            const bool ok = tt < T;
            const float * row = xs + (int64_t) (ok ? tt : 0) * x_s1;
#pragma unroll
            for (int a = 0; a < NA; a++) ra[a][t] = ok ? __builtin_nontemporal_load(row + ch_a[a]) : 0.0f;
            rb[t] = (ok && has_v) ? __builtin_nontemporal_load(row + ch_b) : 0.0f;
        }
        // g / beta: threads 0..C-1 fetch g[t], C..2C-1 fetch beta[t]
        if (tid < 2 * C) {
            const int t  = tid % C;
            const int tt = t0 + t;
            const float * src = (tid < C) ? gl : bt;
            rg = tt < T ? src[((int64_t) seq * T + tt) * HV + h] : 0.0f;
        }
    };

    // conv + SiLU of the prefetched chunk into LDS, then l2norm q/k in place
    auto stage = [&](int t0, int nvalid) {
#pragma unroll
        for (int t = 0; t < C; t++) {
            if (t < nvalid) {
#pragma unroll
                for (int a = 0; a < NA; a++) {
                    const float ya = win_a[a][0] * wa[a][0] + win_a[a][1] * wa[a][1] + win_a[a][2] * wa[a][2] + ra[a][t] * wa[a][3];
                    win_a[a][0] = win_a[a][1]; win_a[a][1] = win_a[a][2]; win_a[a][2] = ra[a][t];
                    if (qk_c[a] < 2 * D) sm.qk[t][qk_c[a]] = silu(ya);
                }
                if (has_v) {
                    const float yb = win_b[0] * wb[0] + win_b[1] * wb[1] + win_b[2] * wb[2] + rb[t] * wb[3];
                    win_b[0] = win_b[1]; win_b[1] = win_b[2]; win_b[2] = rb[t];
                    sm.v[t][tid] = silu(yb);
                }
            }
        }
        if (tid < C)          sm.egb[tid].x     = __expf(rg);
        else if (tid < 2 * C) sm.egb[tid - C].y = rg;
        __syncthreads();
        // l2norm: C tokens x 2 vectors, one (token, vector) per wave iteration
        const int wave = tid >> 5, lane = tid & 31;
        for (int task = wave; task < 2 * C; task += NT / 32) {
            const int t = task >> 1, vo = (task & 1) * D;
            float4 e = *reinterpret_cast<float4 *>(&sm.qk[t][vo + lane * 4]);
            const float ss = wave_sum(e.x * e.x + e.y * e.y + e.z * e.z + e.w * e.w);
            const float sc = rsqrtf(fmaxf(ss, eps * eps));
            e.x *= sc; e.y *= sc; e.z *= sc; e.w *= sc;
            *reinterpret_cast<float4 *>(&sm.qk[t][vo + lane * 4]) = e;
        }
        __syncthreads();
    };

    float * attn_out = dst + ((int64_t) seq * T) * HV * D;

    prefetch(0);
    for (int t0 = 0; t0 < T; t0 += C) {
        const int nvalid = min(C, T - t0);
        stage(t0, nvalid);
        if (t0 + C < T) prefetch(t0 + C);      // in flight during the recurrence below

        // ---- recurrence over the chunk --------------------------------------
        // Everything token t+1 needs from LDS (k rows, v for MC columns, exp(g)/beta)
        // is issued while token t computes; q(t) is issued at the top of step(t)
        // and consumed after the S^T k reduction. sched_barrier(0) pins the issue
        // points so the scheduler cannot sink the loads next to their first use.
        struct Tok { float k[R]; float v[MC]; float2 egb; };
        auto fetch = [&](Tok & tk, int t) {
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                const float4 k4 = *reinterpret_cast<const float4 *>(&sm.qk[t][D + roff(r)]);
                tk.k[r] = k4.x; tk.k[r + 1] = k4.y; tk.k[r + 2] = k4.z; tk.k[r + 3] = k4.w;
            }
            if constexpr (MC % 4 == 0) {
#pragma unroll
                for (int m = 0; m < MC; m += 4) {
                    const float4 v4 = *reinterpret_cast<const float4 *>(&sm.v[t][cg * MC + m]);
                    tk.v[m] = v4.x; tk.v[m + 1] = v4.y; tk.v[m + 2] = v4.z; tk.v[m + 3] = v4.w;
                }
            } else {
#pragma unroll
                for (int m = 0; m < MC; m++) tk.v[m] = sm.v[t][cg * MC + m];
            }
            tk.egb = sm.egb[t];
        };
        float qq[R];
        // Lazy decay: S_true = c * S'. Per token c <- c*e^g (a block-uniform scalar), so the
        // update is a single FMA per element: S' += k * (delta / c). Renormalise S' *= c when c
        // gets tiny; tokens with extreme decay (e^g < THR) fall back to an explicit multiply.
        // (c is uniform across the block since g is per head/token -> no divergent branches.)
        constexpr float THR = 0x1p-40f;
        auto step = [&](const Tok & tk, int t) {
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                const float4 q4 = *reinterpret_cast<const float4 *>(&sm.qk[t][roff(r)]);
                qq[r] = q4.x; qq[r + 1] = q4.y; qq[r + 2] = q4.z; qq[r + 3] = q4.w;
            }
            __builtin_amdgcn_sched_barrier(0);
            const float egf = tk.egb.x, beta = tk.egb.y;
            if (c * egf < THR) {                         // renormalise (rare, uniform)
#pragma unroll
                for (int m = 0; m < MC; m++)
#pragma unroll
                    for (int r = 0; r < R; r++) s[m][r] *= c;
                c = 1.0f;
            }
            float cn;
            if (egf < THR) {                             // extreme decay: explicit this token
#pragma unroll
                for (int m = 0; m < MC; m++)
#pragma unroll
                    for (int r = 0; r < R; r++) s[m][r] *= egf;
                cn = c;
            } else {
                cn = c * egf;
            }
            const float inv_cn = 1.0f / cn;
            float kv[MC];
#pragma unroll
            for (int m = 0; m < MC; m++) {
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int r = 0; r < R; r += 2) { a0 = fmaf(s[m][r], tk.k[r], a0); a1 = fmaf(s[m][r + 1], tk.k[r + 1], a1); }
                kv[m] = a0 + a1;
            }
            float o[MC];
#pragma unroll
            for (int m = 0; m < MC; m++) {
                // delta = beta * (v - cn * (S'^T k));  S' += k * delta / cn
                const float delta = beta * (tk.v[m] - cn * group_sum<G>(kv[m]));
                const float dp    = delta * inv_cn;
                float o0 = 0.f, o1 = 0.f;
#pragma unroll
                for (int r = 0; r < R; r += 2) {
                    s[m][r]     = fmaf(tk.k[r],     dp, s[m][r]);     o0 = fmaf(s[m][r],     qq[r],     o0);
                    s[m][r + 1] = fmaf(tk.k[r + 1], dp, s[m][r + 1]); o1 = fmaf(s[m][r + 1], qq[r + 1], o1);
                }
                o[m] = group_sum<G>(o0 + o1);
            }
            if (gq == 0) {
                const float sc = scale_attn * cn;
#pragma unroll
                for (int m = 0; m < MC; m++) sm.out[t][cg * MC + m] = o[m] * sc;
            }
            c = cn;
        };

        Tok A, B;
        fetch(A, 0);
        for (int t = 0; t < nvalid; t += 2) {
            if (t + 1 < nvalid) fetch(B, t + 1);
            __builtin_amdgcn_sched_barrier(0);
            step(A, t);
            if (t + 1 < nvalid) {
                if (t + 2 < nvalid) fetch(A, t + 2);
                __builtin_amdgcn_sched_barrier(0);
                step(B, t + 1);
            }
        }
        __syncthreads();
        // coalesced chunk store of the attention outputs: out[t][h][col0 .. col0+63]
        for (int e = tid; e < nvalid * COLS; e += NT) {
            const int t = e / COLS, c = e % COLS;
            attn_out[((int64_t) (t0 + t) * HV + h) * D + col0 + c] = sm.out[t][c];
        }
        // (next stage() begins with LDS writes; its first __syncthreads orders them
        //  after these reads only for qk/v/eg/beta; out[] is rewritten after that barrier)
    }

    // final state
#pragma unroll
    for (int m = 0; m < MC; m++) {
        const int j = col0 + cg * MC + m;
        float * sp = dst + (int64_t) n_seqs * T * HV * D + (((int64_t) seq * HV + h) * D + j) * D;
#pragma unroll
        for (int r = 0; r < R; r += 4) *reinterpret_cast<float4 *>(sp + roff(r)) = make_float4(c * s[m][r], c * s[m][r + 1], c * s[m][r + 2], c * s[m][r + 3]);
    }
}

template <int C, int COLS, int MC, int G>
inline void launch(const float * x, int64_t x_s1, int64_t x_s2, const float * cstate, const float * cw,
                   const float * gl, const float * bt, const float * s_in, float * dst,
                   int T, int n_seqs, float eps, hipStream_t st) {
    dim3 grid(HV * (D / COLS), n_seqs), block((COLS / MC) * G);
    hipLaunchKernelGGL((gdn_fused_prefill<C, COLS, MC, G>), grid, block, 0, st,
                       x, x_s1, x_s2, cstate, cw, gl, bt, s_in, dst, T, n_seqs, eps, 1.0f / sqrtf((float) D));
}

} // namespace gdnf

namespace gdnf {

// ---------------------------------------------------------------------------
// Warp-specialised variant (producer/consumer software pipeline).
//   block = one v-head (128 columns); 8 consumer waves (MC=4 columns x G=8 lanes)
//   + 2 producer waves. LDS is double-buffered per chunk:
//     producers: build chunk c+1 (global loads, conv+SiLU, l2norm, e^g/beta) into
//                buf[(c+1)&1] and store chunk c-1's outputs from out[(c-1)&1];
//     consumers: run the recurrence of chunk c from buf[c&1] into out[c&1].
//   One __syncthreads per chunk hands the buffers over.
// ---------------------------------------------------------------------------
template <int C>
struct SmemWS {
    alignas(16) float qk[2][C][2 * D];
    alignas(16) float v[2][C][D];
    alignas(16) float out[2][C][D];
    alignas(8)  float2 egb[2][C];
};

template <int C>
__global__ void __launch_bounds__(320, 1)
gdn_fused_prefill_ws(const float * __restrict__ x, int64_t x_s1, int64_t x_s2,
                     const float * __restrict__ cstate, const float * __restrict__ cw,
                     const float * __restrict__ gl, const float * __restrict__ bt,
                     const float * __restrict__ s_in, float * __restrict__ dst,
                     int T, int n_seqs, float eps, float scale_attn) {
    constexpr int MC = 4, G = 8, R = D / G;
    constexpr int NCONS = (D / MC) * G;        // 256 consumer threads = 8 waves
    constexpr int NPW   = 2;                   // producer waves
    constexpr int HALF  = (C + NPW - 1) / NPW; // tokens per producer wave per chunk
    constexpr int NCHP  = 12;                  // channels per producer lane: 8 q/k + 4 v (384 / 32)
    __shared__ SmemWS<C> sm;
    const int tid = threadIdx.x;
    const int h   = blockIdx.x;
    const int seq = blockIdx.y;
    const int kh  = h % HK;
    const int nchunk = (T + C - 1) / C;
    const float * xs = x + (int64_t) seq * x_s2;
    float * attn_out = dst + ((int64_t) seq * T) * HV * D;

    if (tid >= NCONS) {
        // ============================ producer ============================
        const int pw = (tid - NCONS) >> 5, lane = tid & 31;
        int gch[NCHP];                        // global conv channel of each slot
#pragma unroll
        for (int i = 0; i < NCHP; i++) {
            const int c = lane + 32 * i;      // 0..255 q|k, 256..383 v
            gch[i] = c < D ? kh * D + c : (c < 2 * D ? HK * D + kh * D + (c - D) : 2 * HK * D + h * D + (c - 2 * D));
        }
        float w[NCHP][KW];
#pragma unroll
        for (int i = 0; i < NCHP; i++)
#pragma unroll
            for (int k = 0; k < KW; k++) w[i][k] = cw[gch[i] * KW + k];
        auto raw = [&](int tt, int i) -> float {   // conv input at token tt (tt<0 -> conv state)
            return tt >= 0 ? __builtin_nontemporal_load(xs + (int64_t) tt * x_s1 + gch[i])
                           : cstate[((int64_t) seq * CH + gch[i]) * (KW - 1) + (KW - 1 + tt)];
        };
        auto produce = [&](int ck) {          // this wave's half of chunk ck -> buf[ck&1]
            const int b  = ck & 1;
            const int t0 = ck * C + pw * HALF;
            const int t1 = min(ck * C + min((pw + 1) * HALF, C), T);
            if (t0 >= t1) return;
            float win[NCHP][KW - 1];
#pragma unroll
            for (int i = 0; i < NCHP; i++)
#pragma unroll
                for (int k = 0; k < KW - 1; k++) win[i][k] = raw(t0 - (KW - 1) + k, i);
            float nxt[NCHP];
#pragma unroll
            for (int i = 0; i < NCHP; i++) nxt[i] = raw(t0, i);
            for (int tt = t0; tt < t1; tt++) {
                float cur[NCHP];
#pragma unroll
                for (int i = 0; i < NCHP; i++) cur[i] = nxt[i];
                if (tt + 1 < t1) {                    // prefetch next row while computing this one
#pragma unroll
                    for (int i = 0; i < NCHP; i++) nxt[i] = raw(tt + 1, i);
                }
                float y[NCHP];
#pragma unroll
                for (int i = 0; i < NCHP; i++) {
                    y[i] = silu(win[i][0] * w[i][0] + win[i][1] * w[i][1] + win[i][2] * w[i][2] + cur[i] * w[i][3]);
                    win[i][0] = win[i][1]; win[i][1] = win[i][2]; win[i][2] = cur[i];
                }
                // l2norm of q (slots 0..3) and k (slots 4..7) inside the wave
                const float sq = wave_sum(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]);
                const float sk = wave_sum(y[4] * y[4] + y[5] * y[5] + y[6] * y[6] + y[7] * y[7]);
                const float rq = rsqrtf(fmaxf(sq, eps * eps)), rk = rsqrtf(fmaxf(sk, eps * eps));
                const int t = tt - ck * C;
#pragma unroll
                for (int i = 0; i < 4; i++) { sm.qk[b][t][lane + 32 * i] = y[i] * rq; sm.qk[b][t][D + lane + 32 * i] = y[4 + i] * rk; }
#pragma unroll
                for (int i = 0; i < 4; i++) sm.v[b][t][lane + 32 * i] = y[8 + i];
                if (lane == 0) {
                    const int64_t gi = ((int64_t) seq * T + tt) * HV + h;
                    sm.egb[b][t] = make_float2(__expf(gl[gi]), bt[gi]);
                }
            }
        };
        auto store_out = [&](int ck) {        // outputs of chunk ck from out[ck&1]
            const int b = ck & 1, t0 = ck * C, nv = min(C, T - t0);
            for (int e = tid - NCONS; e < nv * D; e += 32 * NPW) {
                const int t = e / D, cc = e % D;
                attn_out[((int64_t) (t0 + t) * HV + h) * D + cc] = sm.out[b][t][cc];
            }
        };
        produce(0);
        lds_barrier();                                   // buf0 ready
        for (int ck = 0; ck < nchunk; ck++) {
            if (ck >= 1) store_out(ck - 1);
            if (ck + 1 < nchunk) produce(ck + 1);
            lds_barrier();                               // hand over
        }
        store_out(nchunk - 1);
        return;
    }

    // ============================ consumer ============================
    const int cg = tid / G, gq = tid % G;
    auto roff = [&](int r) { return (r >> 2) * (4 * G) + 4 * gq; };
    float s[MC][R];
    float c = 1.0f;
#pragma unroll
    for (int m = 0; m < MC; m++) {
        const float * sp = s_in + (((int64_t) seq * HV + h) * D + cg * MC + m) * D;
#pragma unroll
        for (int r = 0; r < R; r += 4) {
            const float4 v4 = *reinterpret_cast<const float4 *>(sp + roff(r));
            s[m][r] = v4.x; s[m][r + 1] = v4.y; s[m][r + 2] = v4.z; s[m][r + 3] = v4.w;
        }
    }
    constexpr float THR = 0x1p-40f;
    struct Tok { float k[R]; float v[MC]; float2 egb; };
    float qq[R];
    lds_barrier();                                       // buf0 ready
    for (int ck = 0; ck < nchunk; ck++) {
        const int b = ck & 1, nv = min(C, T - ck * C);
        auto fetch = [&](Tok & tk, int t) {
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                const float4 k4 = *reinterpret_cast<const float4 *>(&sm.qk[b][t][D + roff(r)]);
                tk.k[r] = k4.x; tk.k[r + 1] = k4.y; tk.k[r + 2] = k4.z; tk.k[r + 3] = k4.w;
            }
            const float4 v4 = *reinterpret_cast<const float4 *>(&sm.v[b][t][cg * MC]);
            tk.v[0] = v4.x; tk.v[1] = v4.y; tk.v[2] = v4.z; tk.v[3] = v4.w;
            tk.egb = sm.egb[b][t];
        };
        auto step = [&](const Tok & tk, int t) {
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                const float4 q4 = *reinterpret_cast<const float4 *>(&sm.qk[b][t][roff(r)]);
                qq[r] = q4.x; qq[r + 1] = q4.y; qq[r + 2] = q4.z; qq[r + 3] = q4.w;
            }
            __builtin_amdgcn_sched_barrier(0);
            const float egf = tk.egb.x, beta = tk.egb.y;
            if (c * egf < THR) {
#pragma unroll
                for (int m = 0; m < MC; m++)
#pragma unroll
                    for (int r = 0; r < R; r++) s[m][r] *= c;
                c = 1.0f;
            }
            float cn;
            if (egf < THR) {
#pragma unroll
                for (int m = 0; m < MC; m++)
#pragma unroll
                    for (int r = 0; r < R; r++) s[m][r] *= egf;
                cn = c;
            } else {
                cn = c * egf;
            }
            const float inv_cn = 1.0f / cn;
            float kv[MC];
#pragma unroll
            for (int m = 0; m < MC; m++) {
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int r = 0; r < R; r += 2) { a0 = fmaf(s[m][r], tk.k[r], a0); a1 = fmaf(s[m][r + 1], tk.k[r + 1], a1); }
                kv[m] = a0 + a1;
            }
            float o[MC];
#pragma unroll
            for (int m = 0; m < MC; m++) {
                const float delta = beta * (tk.v[m] - cn * group_sum<G>(kv[m]));
                const float dp = delta * inv_cn;
                float o0 = 0.f, o1 = 0.f;
#pragma unroll
                for (int r = 0; r < R; r += 2) {
                    s[m][r]     = fmaf(tk.k[r],     dp, s[m][r]);     o0 = fmaf(s[m][r],     qq[r],     o0);
                    s[m][r + 1] = fmaf(tk.k[r + 1], dp, s[m][r + 1]); o1 = fmaf(s[m][r + 1], qq[r + 1], o1);
                }
                o[m] = group_sum<G>(o0 + o1);
            }
            if (gq == 0) {
                const float sc = scale_attn * cn;
                *reinterpret_cast<float4 *>(&sm.out[b][t][cg * MC]) = make_float4(o[0] * sc, o[1] * sc, o[2] * sc, o[3] * sc);
            }
            c = cn;
        };
        Tok A, B;
        fetch(A, 0);
        for (int t = 0; t < nv; t += 2) {
            if (t + 1 < nv) fetch(B, t + 1);
            __builtin_amdgcn_sched_barrier(0);
            step(A, t);
            if (t + 1 < nv) {
                if (t + 2 < nv) fetch(A, t + 2);
                __builtin_amdgcn_sched_barrier(0);
                step(B, t + 1);
            }
        }
        lds_barrier();                                   // hand over
    }
#pragma unroll
    for (int m = 0; m < MC; m++) {
        float * sp = dst + (int64_t) n_seqs * T * HV * D + (((int64_t) seq * HV + h) * D + cg * MC + m) * D;
#pragma unroll
        for (int r = 0; r < R; r += 4)
            *reinterpret_cast<float4 *>(sp + roff(r)) = make_float4(c * s[m][r], c * s[m][r + 1], c * s[m][r + 2], c * s[m][r + 3]);
    }
}

template <int C>
inline void launch_ws(const float * x, int64_t x_s1, int64_t x_s2, const float * cstate, const float * cw,
                      const float * gl, const float * bt, const float * s_in, float * dst,
                      int T, int n_seqs, float eps, hipStream_t st) {
    hipLaunchKernelGGL((gdn_fused_prefill_ws<C>), dim3(HV, n_seqs), dim3(320), 0, st,
                       x, x_s1, x_s2, cstate, cw, gl, bt, s_in, dst, T, n_seqs, eps, 1.0f / sqrtf((float) D));
}

} // namespace gdnf
#endif // GGML_USE_HIP
