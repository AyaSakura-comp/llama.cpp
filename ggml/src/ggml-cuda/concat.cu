#include "concat.cuh"

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x, const T * y, T * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne0, int64_t ne1, int64_t ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");
    const int64_t n = ne0 * ne1 * ne2;

    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;
            dst[i] = i0 < ne00 ? x[row * ne00 + i0] : y[row * (ne0 - ne00) + i0 - ne00];
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;
            dst[i] = i01 < src0_plane ? x[i2 * src0_plane + i01]
                                      : y[i2 * (dst_plane - src0_plane) + i01 - src0_plane];
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;
            dst[i] = i < src0_size ? x[i] : y[i - src0_size];
        }
    }
}

template <typename T>
static void concat_cuda(const T * x, const T * y, T * dst, int64_t ne00, int64_t ne01, int64_t ne02,
                        int64_t ne0, int64_t ne1, int64_t ne2, int dim, cudaStream_t stream) {
    const int64_t n = ne0 * ne1 * ne2;
    const int num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    if (dim == 0) {
        concat_cont<T, 0><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
    } else if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
    } else {
        concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
    }
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else if constexpr (dim == 0) {
            x = (const T *) (src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
        } else if constexpr (dim == 1) {
            x = (const T *) (src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
        } else if constexpr (dim == 2) {
            x = (const T *) (src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
        } else {
            x = (const T *) (src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
        }
        *(T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = *x;
    }
}


template <typename T>
static void concat_non_contiguous(const ggml_tensor * src0, const ggml_tensor * src1,
                                  ggml_tensor * dst, int dim, cudaStream_t stream) {
    dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
    auto launch = [&](auto d) {
        concat_non_cont<T, decltype(d)::value><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    };
    if (dim == 0) launch(std::integral_constant<int, 0>{});
    else if (dim == 1) launch(std::integral_constant<int, 1>{});
    else if (dim == 2) launch(std::integral_constant<int, 2>{});
    else if (dim == 3) launch(std::integral_constant<int, 3>{});
    else GGML_ABORT("Invalid dim: %d", dim);
}

template <typename T>
static void concat_typed(const ggml_tensor * src0, const ggml_tensor * src1,
                         ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        concat_non_contiguous<T>(src0, src1, dst, dim, stream);
        return;
    }
    if (dim == 3) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src0->data, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + ggml_nbytes(src0), src1->data,
                                   ggml_nbytes(src1), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    for (int i3 = 0; i3 < dst->ne[3]; ++i3) {
        concat_cuda((const T *) src0->data + i3 * src0->nb[3] / sizeof(T),
                    (const T *) src1->data + i3 * src1->nb[3] / sizeof(T),
                    (T *) dst->data + i3 * dst->nb[3] / sizeof(T), src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0], dst->ne[1], dst->ne[2], dim, stream);
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src0->type == src1->type && src0->type == dst->type);
    const int dim = ((int32_t *) dst->op_params)[0];
    if (dst->type == GGML_TYPE_F32) {
        concat_typed<float>(src0, src1, dst, dim, ctx.stream());
    } else if (dst->type == GGML_TYPE_I32) {
        concat_typed<int32_t>(src0, src1, dst, dim, ctx.stream());
    } else {
        GGML_ABORT("Unsupported concat type: %s", ggml_type_name(dst->type));
    }
}
