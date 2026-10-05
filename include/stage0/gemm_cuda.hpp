#pragma once
#include "stage0/cuda_utils.hpp"
namespace stage0 {
inline constexpr unsigned kBlockX = 16, kBlockY = 16;
void launch_naive(const float* a, const float* b, float* c, Shape s, cudaStream_t stream);
void launch_cublas(cublasHandle_t handle, const float* a, const float* b, float* c, Shape s);
}
