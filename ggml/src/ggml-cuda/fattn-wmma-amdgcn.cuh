#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

namespace wmma {
    struct matrix_a {};
    struct matrix_b {};
    struct accumulator {};
    struct row_major {};
    struct col_major {};

    enum layout_t : uint32_t {
        mem_row_major = 0,
        mem_col_major = 1
    };

    template<typename Use, int m, int n, int k, typename T, typename Layout = void>
    struct fragment;

    template<int m, int n, int k, typename T>
    struct fragment<matrix_a, m, n, k, T, row_major> {
        union {
            half16 vec;
            float4 f4[2];
            T      arr[16];
        };
    };

    template<int m, int n, int k, typename T>
    struct fragment<matrix_a, m, n, k, T, col_major> {
        union {
            half16 vec;
            float4 f4[2];
            T      arr[16];
        };
    };

    template<int m, int n, int k, typename T>
    struct fragment<matrix_b, m, n, k, T, col_major> {
        union {
            half16 vec;
            float4 f4[2];
            T      arr[16];
        };
    };

    template<int m, int n, int k, typename T>
    struct fragment<matrix_b, m, n, k, T, row_major> {
        union {
            half16 vec;
            float4 f4[2];
            T      arr[16];
        };
    };

    template<int m, int n, int k, typename T, typename Layout>
    struct fragment<accumulator, m, n, k, T, Layout> {
        union {
            float8 vec;
            float4 f4[2];
            float  arr[8];
        };
    };

    template<int m, int n, int k, typename T, typename Layout, typename V>
    __device__ __forceinline__ void fill_fragment(fragment<accumulator, m, n, k, T, Layout> & f, V v) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) f.arr[i] = static_cast<float>(v);
    }

    template<int m, int n, int k, typename T, typename T_ptr>
    __device__ __forceinline__ void load_matrix_sync(
            fragment<matrix_a, m, n, k, T, row_major> & f,
            const T_ptr * ptr, int ldm) {
        const int lane = threadIdx.x;
        const float4 * row_ptr = (const float4 *)(ptr + (lane % 16) * ldm);
        f.f4[0] = row_ptr[0];
        f.f4[1] = row_ptr[1];
    }

    template<int m, int n, int k, typename T, typename T_ptr>
    __device__ __forceinline__ void load_matrix_sync(
            fragment<matrix_a, m, n, k, T, col_major> & f,
            const T_ptr * ptr, int ldm) {
        const int lane = threadIdx.x;
        const int mi = lane % 16;
        const T_ptr * p = ptr + mi;
        #pragma unroll
        for (int ki = 0; ki < 16; ++ki) {
            f.arr[ki] = p[ki * ldm];
        }
    }

    template<int m, int n, int k, typename T, typename T_ptr>
    __device__ __forceinline__ void load_matrix_sync(
            fragment<matrix_b, m, n, k, T, col_major> & f,
            const T_ptr * ptr, int ldm) {
        const int lane = threadIdx.x;
        const float4 * col_ptr = (const float4 *)(ptr + (lane % 16) * ldm);
        f.f4[0] = col_ptr[0];
        f.f4[1] = col_ptr[1];
    }

    template<int m, int n, int k, typename T, typename T_ptr>
    __device__ __forceinline__ void load_matrix_sync(
            fragment<matrix_b, m, n, k, T, row_major> & f,
            const T_ptr * ptr, int ldm) {
        const int lane = threadIdx.x;
        const int ni = lane % 16;
        const T_ptr * p = ptr + ni;
        #pragma unroll
        for (int ki = 0; ki < 16; ++ki) {
            f.arr[ki] = p[ki * ldm];
        }
    }

    template<int m, int n, int k, typename T_A, typename Layout_A, typename T_B, typename Layout_B, typename T_C, typename Layout_C, typename T_D, typename Layout_D>
    __device__ __forceinline__ void mma_sync(
            fragment<accumulator, m, n, k, T_D, Layout_D> & d,
            const fragment<matrix_a, m, n, k, T_A, Layout_A> & a,
            const fragment<matrix_b, m, n, k, T_B, Layout_B> & b,
            const fragment<accumulator, m, n, k, T_C, Layout_C> & c) {
        d.vec = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.vec, b.vec, c.vec);
    }

    template<int m, int n, int k, typename T, typename Layout, typename T_ptr>
    __device__ __forceinline__ void store_matrix_sync(
            T_ptr * ptr,
            const fragment<accumulator, m, n, k, T, Layout> & f,
            int ldm, layout_t layout) {
        if (layout == mem_col_major) {
            const int lane = threadIdx.x;
            T_ptr * col_ptr = ptr + (lane % 16) * ldm + (lane >= 16 ? 1 : 0);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                col_ptr[2 * i] = static_cast<T_ptr>(f.arr[i]);
            }
        } else {
            const int lane = threadIdx.x;
            T_ptr * row_ptr = ptr + (lane % 16) * ldm + (lane >= 16 ? 1 : 0);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                row_ptr[2 * i] = static_cast<T_ptr>(f.arr[i]);
            }
        }
    }
}
