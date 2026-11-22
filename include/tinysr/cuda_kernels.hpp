#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace tinysr {
void float_to_half(const float* src, void* dst, std::size_t count, cudaStream_t stream);
void subtract_half_to_float(const void* a, const void* b, float* dst,
                            std::size_t count, cudaStream_t stream);
}  // namespace tinysr
