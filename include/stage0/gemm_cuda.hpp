#pragma once
#include "stage0/cuda_utils.hpp"
#include <optional>
#include <string_view>

namespace stage0 {
// Stage 0 v0 remains a separate, preserved implementation.  The registry is
// the only Stage 9 selection boundary used by the benchmark and runtime.
inline constexpr unsigned kBlockX = 16, kBlockY = 16;
enum class GemmKernel { V0Naive, V1Tiled, CuBlas };
struct GemmKernelConfig {
    const char* name;
    unsigned block_x, block_y, bm, bn, bk, tm, tn, threads, vector_width;
    std::size_t shared_bytes;
};

const GemmKernelConfig& gemm_kernel_config(GemmKernel kernel);
std::optional<GemmKernel> gemm_kernel_from_name(std::string_view name);
void launch_naive(const float* a, const float* b, float* c, Shape s, cudaStream_t stream);
void launch_tiled(const float* a, const float* b, float* c, Shape s, cudaStream_t stream);
void launch_cublas(cublasHandle_t handle, const float* a, const float* b, float* c, Shape s);
void launch_gemm(GemmKernel kernel, cublasHandle_t handle, const float* a, const float* b, float* c,
                 Shape s, cudaStream_t stream);
} // namespace stage0
