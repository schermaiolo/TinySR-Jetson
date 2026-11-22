#include "tinysr/cuda_kernels.hpp"
#include "tinysr/cuda_utils.hpp"

#include <cuda_fp16.h>

namespace tinysr {

__global__ void convert_kernel(const float* src, __half* dst, std::size_t count) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half(src[i]);
}

__global__ void subtract_kernel(const __half* a, const __half* b,
                                float* dst, std::size_t count) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __half2float(__hsub(a[i], b[i]));
}

void float_to_half(const float* src, void* dst, std::size_t count, cudaStream_t stream) {
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    convert_kernel<<<blocks, threads, 0, stream>>>(src, static_cast<__half*>(dst), count);
    cuda_check(cudaGetLastError(), "float_to_half kernel");
}

void subtract_half_to_float(const void* a, const void* b, float* dst,
                            std::size_t count, cudaStream_t stream) {
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    subtract_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(a), static_cast<const __half*>(b), dst, count);
    cuda_check(cudaGetLastError(), "subtract_half_to_float kernel");
}

}  // namespace tinysr
