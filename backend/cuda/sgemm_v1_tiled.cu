#include "stage0/gemm_cuda.hpp"

namespace stage0 {
namespace {
constexpr unsigned kTileM = 16;
constexpr unsigned kTileN = 16;
constexpr unsigned kTileK = 16;

__global__ void sgemm_v1_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[kTileM][kTileK];
    __shared__ float tile_b[kTileK][kTileN];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kTileM + ty;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * kTileN + tx;
    float acc = 0.0F;

    for (int tile = 0; tile < k; tile += static_cast<int>(kTileK)) {
        const int a_col = tile + static_cast<int>(tx);
        const int b_row = tile + static_cast<int>(ty);
        tile_a[ty][tx] = row < static_cast<std::size_t>(m) && a_col < k
            ? a[row * static_cast<std::size_t>(k) + static_cast<std::size_t>(a_col)] : 0.0F;
        tile_b[ty][tx] = b_row < k && col < static_cast<std::size_t>(n)
            ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0F;
        __syncthreads();
        for (unsigned p = 0; p < kTileK; ++p) acc += tile_a[ty][p] * tile_b[p][tx];
        __syncthreads();
    }
    if (row < static_cast<std::size_t>(m) && col < static_cast<std::size_t>(n)) c[row * n + col] = acc;
}
} // namespace

void launch_tiled(const float* a, const float* b, float* c, Shape s, cudaStream_t stream) {
    elements(s.m, s.k); elements(s.k, s.n); elements(s.m, s.n);
    const dim3 block(kTileN, kTileM);
    const dim3 grid((static_cast<unsigned>(s.n) + kTileN - 1) / kTileN,
                    (static_cast<unsigned>(s.m) + kTileM - 1) / kTileM);
    sgemm_v1_tiled<<<grid, block, 0, stream>>>(a, b, c, s.m, s.n, s.k);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace stage0
