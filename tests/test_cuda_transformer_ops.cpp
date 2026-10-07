#include "runtime/cuda_backend.hpp"
#include "runtime/graph_executor.hpp"
#include "runtime/scheduler.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }
} // namespace
int main() {
    try {
        CudaBackend cuda;
        Scheduler scheduler(default_cpu_backend(), &cuda);
        auto a = Tensor::allocate_cpu({2, 2}), b = Tensor::allocate_cpu({2, 2}), scale = Tensor::allocate_cpu({2});
        a.data<float>()[0] = 1; a.data<float>()[1] = 2; a.data<float>()[2] = 3; a.data<float>()[3] = 4;
        b.data<float>()[0] = 1; b.data<float>()[1] = 0; b.data<float>()[2] = 0; b.data<float>()[3] = 1;
        scale.data<float>()[0] = scale.data<float>()[1] = 1;
        Graph graph; graph.add_input(0, "projection_input", a); graph.add_input(1, "projection_weight", b); graph.add_input(2, "norm_scale", scale);
        graph.add_tensor(3, {2, 2}); graph.add_tensor(4, {2, 2}); graph.add_tensor(5, {2, 2});
        graph.add_node(0, OpDesc(OpCode::MATMUL, {0, 1}, {3}, {}, cuda.device()));
        graph.add_node(1, OpDesc(OpCode::SOFTMAX, {3}, {4}, SoftmaxAttrs{}));
        graph.add_node(2, OpDesc(OpCode::RMSNORM, {4, 2}, {5}, NormAttrs{1e-5})); graph.add_output("result", 5);
        success(graph.freeze());
        const auto scheduled = scheduler.rewrite(graph); require(scheduled.ok(), scheduled.status.message.c_str());
        require(scheduled.inserted_copies.size() == 3, "CUDA projection and CPU primitives require explicit H2D/D2H copies");
        ScheduledAllocationProvider prepared(*scheduled.graph, scheduler);
        const auto result = execute_graph(*scheduled.graph, nullptr, &prepared, nullptr, &scheduler);
        require(result.ok(), result.status.message.c_str());
        require(result.outputs.at("result").device() == Device{} && result.counts.copies == 3, "mixed conformance executes explicit transfers");
        for (std::size_t row = 0; row < 2; ++row) {
            const auto x = result.outputs.at("result").data<float>()[row * 2], y = result.outputs.at("result").data<float>()[row * 2 + 1];
            require(std::isfinite(x) && std::isfinite(y) && x > 0 && y > 0, "CPU transformer primitive output is finite after CUDA projection");
        }
        std::cout << "CUDA transformer conformance: PASS copies=" << result.counts.copies << " dispatches=" << result.counts.backend_dispatches << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "test_cuda_transformer_ops: " << error.what() << '\n'; return 1; }
}
