#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));

__global__ void test_wmma_gemm(const _Float16 * A, const _Float16 * B, float * C) {
    const int lane = threadIdx.x;
    half16 a, b;
    float8 c = {};
    for (int k = 0; k < 16; ++k) {
        a[k] = A[(lane % 16) * 16 + k];
        b[k] = B[k * 16 + lane % 16];
    }
    const float8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + lane / 16;
        const int col = lane % 16;
        C[row * 16 + col] = d[i];
    }
}

int main() {
    std::vector<_Float16> A(256), B(256);
    std::vector<float> got(256), expected(256);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            A[i * 16 + j] = (_Float16) (std::sin(i * 3.0 + j) * 2.0);
            B[i * 16 + j] = (_Float16) (std::cos(j * 5.0 - i) * 1.5);
        }
    }
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 16; ++k) sum += float(A[i * 16 + k]) * float(B[k * 16 + j]);
            expected[i * 16 + j] = sum;
        }
    }
    _Float16 * dA = nullptr, *dB = nullptr;
    float * dC = nullptr;
    if (hipMalloc(&dA, 512) != hipSuccess || hipMalloc(&dB, 512) != hipSuccess || hipMalloc(&dC, 1024) != hipSuccess) return 2;
    hipMemcpy(dA, A.data(), 512, hipMemcpyHostToDevice);
    hipMemcpy(dB, B.data(), 512, hipMemcpyHostToDevice);
    test_wmma_gemm<<<1, 32>>>(dA, dB, dC);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    hipMemcpy(got.data(), dC, 1024, hipMemcpyDeviceToHost);
    float max_error = 0.0f;
    for (int i = 0; i < 256; ++i) max_error = std::max(max_error, std::abs(got[i] - expected[i]));
    std::printf("max_error=%g\n", max_error);
    return max_error < 1.0e-3f ? 0 : 1;
}
