#include "stage0/gemm_cuda.hpp"
namespace stage0 {
namespace {
__global__ void sgemm_v0_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= static_cast<std::size_t>(m) || col >= static_cast<std::size_t>(n)) return;
    float acc = 0.0f;
    // One thread -> one output. No shared memory, blocking, float4 or Tensor Core.
    for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[static_cast<std::size_t>(p) * n + col];
    c[row * n + col] = acc;
}
}
void launch_naive(const float* a, const float* b, float* c, Shape s, cudaStream_t stream) {
    elements(s.m, s.k); elements(s.k, s.n); elements(s.m, s.n);
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((static_cast<unsigned>(s.n) + kBlockX - 1) / kBlockX,
                    (static_cast<unsigned>(s.m) + kBlockY - 1) / kBlockY);
    sgemm_v0_naive<<<grid, block, 0, stream>>>(a, b, c, s.m, s.n, s.k);
    CUDA_CHECK(cudaGetLastError());
}
}
