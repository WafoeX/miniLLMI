#include "stage0/gemm_cuda.hpp"
#include <stdexcept>

namespace stage0 {
namespace {
constexpr GemmKernelConfig kV0Config{"sgemm_v0_naive", kBlockX, kBlockY, 1, 1, 1, 1, 1,
                                      kBlockX * kBlockY, 1, 0};
constexpr GemmKernelConfig kV1Config{"sgemm_v1_tiled", 16, 16, 16, 16, 16, 1, 1,
                                      16 * 16, 1, 2 * 16 * 16 * sizeof(float)};
constexpr GemmKernelConfig kCuBlasConfig{"cublas", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
} // namespace

const GemmKernelConfig& gemm_kernel_config(GemmKernel kernel) {
    switch (kernel) {
    case GemmKernel::V0Naive: return kV0Config;
    case GemmKernel::V1Tiled: return kV1Config;
    case GemmKernel::CuBlas: return kCuBlasConfig;
    }
    throw std::invalid_argument("unknown GEMM kernel");
}

std::optional<GemmKernel> gemm_kernel_from_name(std::string_view name) {
    if (name == "naive" || name == "v0" || name == kV0Config.name) return GemmKernel::V0Naive;
    if (name == "tiled" || name == "v1" || name == kV1Config.name) return GemmKernel::V1Tiled;
    if (name == kCuBlasConfig.name) return GemmKernel::CuBlas;
    return std::nullopt;
}

void launch_gemm(GemmKernel kernel, cublasHandle_t handle, const float* a, const float* b, float* c,
                 Shape s, cudaStream_t stream) {
    switch (kernel) {
    case GemmKernel::V0Naive: launch_naive(a, b, c, s, stream); return;
    case GemmKernel::V1Tiled: launch_tiled(a, b, c, s, stream); return;
    case GemmKernel::CuBlas: launch_cublas(handle, a, b, c, s); return;
    }
    throw std::invalid_argument("unknown GEMM kernel");
}
} // namespace stage0
