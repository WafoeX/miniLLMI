#include "stage0/gemm_cuda.hpp"
namespace stage0 {
void launch_cublas(cublasHandle_t handle, const float* a, const float* b, float* c, Shape s) {
    elements(s.m, s.k); elements(s.k, s.n); elements(s.m, s.n);
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C=A*B is column-major C^T=B^T*A^T; no transpose/copy kernels.
    // Column-major operands: B (N x K, ld=N), A (K x M, ld=K), C (N x M, ld=N).
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k,
                           &alpha, b, s.n, a, s.k, &beta, c, s.n));
}
}
