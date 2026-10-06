#include "runtime/cuda_backend.hpp"
#include "runtime/graph_executor.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

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
        CudaBackend tiled(0, CudaMatmul::Stage9Tiled);
        require(std::string(tiled.name()) == "cuda-stage9-tiled", "tiled runtime selection name");
        success(tiled.execute(matmul, {*a.tensor, *b.tensor}, *c.tensor));
        success(tiled.copy(*c.tensor, host_c));
        for (std::size_t i = 0; i < host_c.numel(); ++i) require(std::abs(host_c.data<float>()[i] - expected[i]) < 1e-4F, "CUDA tiled runtime result");
        auto alias = c.tensor->narrow(0, 0, 2);
        const auto aliased = cuda.copy(*c.tensor, alias);
        require(aliased.code == StatusCode::Aliasing, "CUDA D2D alias rejection");
        auto transposed = host_a.transpose(0, 1);
        require(cuda.copy(transposed, *a.tensor).code == StatusCode::InvalidArgument, "explicit materialization required");

        auto host_d = Tensor::allocate_cpu({2, 2});
        for (std::size_t i = 0; i < host_d.numel(); ++i) host_d.data<float>()[i] = static_cast<float>(i + 1);
        auto d = cuda.allocate({2, 2}, DType::FP32, cuda.device());
        require(d.ok(), "CUDA second input allocation"); success(cuda.copy(host_d, *d.tensor));
        Graph graph;
        graph.add_input(0, "a", *a.tensor); graph.add_input(1, "b", *b.tensor); graph.add_input(2, "d", *d.tensor);
        graph.add_tensor(3, {2, 2}, DType::FP32, cuda.device()); graph.add_tensor(4, {2, 2}, DType::FP32, cuda.device());
        graph.add_node(0, OpDesc(OpCode::MATMUL, {0, 1}, {3})); graph.add_node(1, OpDesc(OpCode::MATMUL, {3, 2}, {4}));
        graph.add_output("result", 4); success(graph.freeze());
        CudaPlannedAllocationProvider prepared(graph, cuda);
        { const auto result = execute_graph(graph, nullptr, &prepared, &cuda);
          require(result.ok() && result.counts.allocations == 0 && result.counts.arena_capacity_bytes == prepared.capacity(), "planned CUDA graph has no execute-time cudaMalloc");
          success(cuda.copy(result.outputs.at("result"), host_c));
          const float chained[] = {106, 156, 241, 354};
          for (std::size_t i = 0; i < host_c.numel(); ++i) require(std::abs(host_c.data<float>()[i] - chained[i]) < 1e-4F, "planned CUDA graph result"); }
        std::cout << "CUDA backend storage/copy/v0-v1 dispatch/planned graph: PASS\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "test_cuda_backend: " << error.what() << '\n'; return 1; }
}
