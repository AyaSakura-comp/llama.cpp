#include "hadamard.cuh"
#include <algorithm>

template<int N>
static __global__ void fwht_f32_kernel(
    const float * __restrict__ src,
    float * __restrict__ dst,
    int64_t ne11,
    int64_t ne12,
    size_t nb11,
    size_t nb12,
    size_t nb13,
    size_t nb1,
    size_t nb2,
    size_t nb3,
    int64_t nr,
    bool is_contiguous
) {
    const int tid = threadIdx.x;
    constexpr float scale = 1.0f / (
        N == 1024 ? 32.0f :
        N == 512  ? 22.627416997969522f :
        N == 256  ? 16.0f :
        N == 128  ? 11.313708498984761f :
        N == 64   ? 8.0f :
        N == 32   ? 5.656854249492381f :
                    4.0f
    );

    __shared__ float s_data[N];

    const int64_t row_start = (int64_t)blockIdx.x;
    const int64_t grid_stride = (int64_t)gridDim.x;

    for (int64_t row = row_start; row < nr; row += grid_stride) {
        const float * src_row;
        float * dst_row;

        if (is_contiguous) {
            src_row = src + row * N;
            dst_row = dst + row * N;
        } else {
            const int64_t i13 = row / (ne11 * ne12);
            const int64_t i12 = (row - i13 * ne11 * ne12) / ne11;
            const int64_t i11 = row - i13 * ne11 * ne12 - i12 * ne11;
            src_row = (const float *) ((const char *) src + i11 * nb11 + i12 * nb12 + i13 * nb13);
            dst_row = (float *) ((char *) dst + i11 * nb1 + i12 * nb2 + i13 * nb3);
        }

        float val = src_row[tid];

        // Intra-warp stages: 1, 2, 4, 8, 16
        #pragma unroll
        for (int mask = 1; mask < 32 && mask < N; mask <<= 1) {
            float other = __shfl_xor_sync(0xffffffff, val, mask, 32);
            val = (tid & mask) ? (other - val) : (val + other);
        }

        // Inter-warp stages: 32, 64, 128, 256, 512
        if constexpr (N > 32) {
            s_data[tid] = val;
            __syncthreads();

            #pragma unroll
            for (int stage = 32; stage < N; stage <<= 1) {
                float other = s_data[tid ^ stage];
                val = (tid & stage) ? (other - val) : (val + other);
                if (stage * 2 < N) {
                    __syncthreads();
                    s_data[tid] = val;
                    __syncthreads();
                }
            }
        }

        dst_row[tid] = val * scale;

        if constexpr (N > 32) {
            if (row + grid_stride < nr) {
                __syncthreads();
            }
        }
    }
}

bool ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n = src1->ne[0];
    if (n != dst->ne[0]) {
        return false;
    }

    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nr = ne11 * ne12 * ne13;

    if (nr == 0) {
        return true;
    }

    const bool is_contiguous = ggml_is_contiguous(src1) && ggml_is_contiguous(dst);

    const float * src_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    const int64_t nblocks = std::min((int64_t)65535, nr);

    switch (n) {
        case 16:
            fwht_f32_kernel<16><<<nblocks, 16, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        case 32:
            fwht_f32_kernel<32><<<nblocks, 32, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        case 64:
            fwht_f32_kernel<64><<<nblocks, 64, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        case 128:
            fwht_f32_kernel<128><<<nblocks, 128, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        case 256:
            fwht_f32_kernel<256><<<nblocks, 256, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        case 512:
            fwht_f32_kernel<512><<<nblocks, 512, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        case 1024:
            fwht_f32_kernel<1024><<<nblocks, 1024, 0, stream>>>(
                src_d, dst_d, ne11, ne12, src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], nr, is_contiguous);
            break;
        default:
            return false;
    }

    CUDA_CHECK(cudaGetLastError());
    return true;
}
