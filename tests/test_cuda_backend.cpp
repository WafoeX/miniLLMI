#include "runtime/cuda_backend.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }
}
int main() {
    try {
        CudaBackend cuda(0);
        auto host_a = Tensor::allocate_cpu({2, 3});
        auto host_b = Tensor::allocate_cpu({3, 2});
        auto host_c = Tensor::allocate_cpu({2, 2});
        for (std::size_t i = 0; i < host_a.numel(); ++i) host_a.data<float>()[i] = static_cast<float>(i + 1);
        for (std::size_t i = 0; i < host_b.numel(); ++i) host_b.data<float>()[i] = static_cast<float>(i + 1);
        auto a = cuda.allocate({2, 3}, DType::FP32, cuda.device());
        auto b = cuda.allocate({3, 2}, DType::FP32, cuda.device());
        auto c = cuda.allocate({2, 2}, DType::FP32, cuda.device());
        require(a.ok() && b.ok() && c.ok(), "CUDA allocation");
        success(cuda.copy(host_a, *a.tensor)); success(cuda.copy(host_b, *b.tensor));
        const OpDesc matmul(OpCode::MATMUL, {0, 1}, {2});
        success(cuda.execute(matmul, {*a.tensor, *b.tensor}, *c.tensor));
        success(cuda.copy(*c.tensor, host_c));
        const float expected[] = {22, 28, 49, 64};
        for (std::size_t i = 0; i < host_c.numel(); ++i) require(std::abs(host_c.data<float>()[i] - expected[i]) < 1e-4F, "CUDA v0 adapter result");
        auto alias = c.tensor->narrow(0, 0, 2);
        const auto aliased = cuda.copy(*c.tensor, alias);
        require(aliased.code == StatusCode::Aliasing, "CUDA D2D alias rejection");
        auto transposed = host_a.transpose(0, 1);
        require(cuda.copy(transposed, *a.tensor).code == StatusCode::InvalidArgument, "explicit materialization required");
        std::cout << "CUDA backend storage/copy/Stage0 dispatch: PASS\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "test_cuda_backend: " << error.what() << '\n'; return 1; }
}
