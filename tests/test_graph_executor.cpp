#include "runtime/graph_executor.hpp"
#include "runtime/reference.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
Tensor sequence(Shape shape, float start = 1) {
    auto tensor = Tensor::allocate_cpu(shape);
    for (std::size_t i = 0; i < tensor.numel(); ++i) tensor.data<float>()[i] = start + static_cast<float>(i);
    return tensor;
}
void composed() {
    const auto a = sequence({2, 3}), b = sequence({2, 3}, -2), w = sequence({3, 4});
    auto sum = Tensor::allocate_cpu({2, 3}), expected = Tensor::allocate_cpu({2, 4});
    success(reference::execute(OpDesc(OpCode::ADD, {0, 1}, {2}), {a, b}, sum));
    success(reference::execute(OpDesc(OpCode::MATMUL, {0, 1}, {2}), {sum, w}, expected));
    Graph graph; graph.add_input(0, "a", a); graph.add_input(1, "b", b); graph.add_input(2, "w", w);
    graph.add_tensor(3, {2, 3}); graph.add_tensor(4, {2, 4});
    graph.add_node(7, OpDesc(OpCode::MATMUL, {3, 2}, {4}));
    graph.add_node(9, OpDesc(OpCode::ADD, {0, 1}, {3}));
    graph.add_output("result", 4); success(graph.freeze());
    const auto before = testing::cpu_allocation_counts();
    {
        const auto result = execute_graph(graph); success(result.status);
        require(!result.failed_node && result.counts.nodes_completed == 2, "graph execution completion");
        require(result.counts.allocations == 2 && result.counts.frees == 1 && result.counts.live_bytes == 32 &&
                result.counts.peak_live_bytes == 56 && result.counts.copies == 0, "dynamic baseline counts");
        require(testing::cpu_allocation_counts().allocations - before.allocations == result.counts.allocations &&
                testing::cpu_allocation_counts().frees - before.frees == result.counts.frees, "counts match actual backing allocator");
        for (std::size_t i = 0; i < expected.numel(); ++i)
            require(result.outputs.at("result").data<float>()[i] == expected.data<float>()[i], "ADD->MATMUL direct reference agreement");
        std::cout << "ADD->MATMUL counts: allocations=" << result.counts.allocations << " frees=" << result.counts.frees
                  << " peak_live_bytes=" << result.counts.peak_live_bytes << " copies=" << result.counts.copies << '\n';
    }
    require(testing::cpu_allocation_counts().live == before.live, "outputs free when result dies");
    auto result = execute_graph(graph); success(result.status);
    require(result.counts.allocations == 2, "repeated execution uses fresh intermediate buffers");
}
void lifetime() {
    Graph graph; graph.add_input(0, "x", sequence({2, 3}));
    graph.add_tensor(1, {2, 3}); graph.add_tensor(2, {3, 2}); graph.add_tensor(3, {2, 2}); graph.add_tensor(4, {2, 2});
    graph.add_tensor(5, {2, 2}); graph.add_tensor(6, {2, 3});
    graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    graph.add_node(1, OpDesc(OpCode::TRANSPOSE, {1}, {2}, TransposeAttrs{}));
    graph.add_node(2, OpDesc(OpCode::SLICE, {2}, {3}, SliceAttrs{0, 1, 2, 1}));
    graph.add_node(3, OpDesc(OpCode::MATERIALIZE, {3}, {4}));
    graph.add_node(4, OpDesc(OpCode::ADD, {4, 4}, {5}));
    graph.add_node(5, OpDesc(OpCode::ADD, {0, 0}, {6})); // unused output is allocated then released
    graph.add_output("dense", 5); graph.add_output("alias", 3); graph.add_output("alias-again", 3);
    success(graph.freeze());
    const auto before = testing::cpu_allocation_counts();
    {
        auto result = execute_graph(graph); success(result.status);
        require(result.counts.allocations == 4 && result.counts.frees == 2 && result.counts.live_bytes == 40 &&
                result.counts.peak_live_bytes == 64 && result.counts.copies == 1 && result.counts.copy_bytes == 16, "alias base pinned and last-use release");
        const auto& alias = result.outputs.at("alias");
        require(alias.storage() == result.outputs.at("alias-again").storage() && alias.data_offset() == 4, "named aliases retain shared base");
        require(alias.at<float>({0, 0}) == 4 && alias.at<float>({1, 1}) == 12, "view survives base TensorId last-use release");
        require(result.outputs.at("dense").data<float>()[3] == 24, "materialization is explicit");
        require(testing::cpu_allocation_counts().live == before.live + 2, "only output bases remain live");
    }
    require(testing::cpu_allocation_counts().live == before.live, "alias result release frees all bases");
    Graph unpinned; unpinned.add_input(0, "x", sequence({2, 3}));
    unpinned.add_tensor(1, {2, 3}); unpinned.add_tensor(2, {3, 2}); unpinned.add_tensor(3, {3, 2});
    unpinned.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    unpinned.add_node(1, OpDesc(OpCode::TRANSPOSE, {1}, {2}, TransposeAttrs{}));
    unpinned.add_node(2, OpDesc(OpCode::MATERIALIZE, {2}, {3}));
    unpinned.add_output("dense", 3); success(unpinned.freeze());
    const auto run = execute_graph(unpinned); success(run.status);
    require(run.counts.allocations == 2 && run.counts.frees == 1 && run.counts.live_bytes == 24 && run.counts.peak_live_bytes == 48,
            "unpinned alias extends base only through last downstream consumer");
}
void failures() {
    Graph invalid; require(execute_graph(invalid).status.code == StatusCode::InvalidArgument, "invalid graph cannot execute");
    Graph graph; auto input = sequence({2}); input.data<float>()[0] = std::numeric_limits<float>::infinity();
    graph.add_input(0, "x", input); graph.add_tensor(1, {2}); graph.add_tensor(2, {2});
    graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); graph.add_node(1, OpDesc(OpCode::MUL, {1, 1}, {2}));
    graph.add_output("result", 2); success(graph.freeze());
    const auto before = testing::cpu_allocation_counts();
    const auto failed = execute_graph(graph);
    require(failed.status.code == StatusCode::NonFinite && failed.failed_node == 0 && failed.outputs.empty() &&
            failed.counts.nodes_completed == 0 && failed.counts.allocations == 1 && failed.counts.frees == 1, "stop on first error and discard outputs");
    require(testing::cpu_allocation_counts().live == before.live, "error cleanup");
    input.data<float>()[0] = 1;
    testing::fail_next_cpu_allocation();
    const auto oom = execute_graph(graph);
    require(oom.status.code == StatusCode::ResourceExhausted && oom.failed_node == 0 && oom.outputs.empty() &&
            oom.counts.allocations == 0 && testing::cpu_allocation_counts().live == before.live, "allocation failure cleanup");
    Graph unsupported; unsupported.add_input(0, "x", sequence({1, 2}));
    unsupported.add_tensor(1, {1, 2}); unsupported.add_tensor(2, {1, 2}); unsupported.add_tensor(3, {1, 2});
    unsupported.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    unsupported.add_node(1, OpDesc(OpCode::SOFTMAX, {1}, {2}, SoftmaxAttrs{}));
    unsupported.add_node(2, OpDesc(OpCode::ADD, {2, 2}, {3}));
    unsupported.add_output("early", 1); unsupported.add_output("result", 3); success(unsupported.freeze());
    const auto no_kernel = execute_graph(unsupported);
    require(no_kernel.status.code == StatusCode::Unsupported && no_kernel.failed_node == 1 && no_kernel.outputs.empty() &&
            no_kernel.counts.nodes_completed == 1 && no_kernel.counts.allocations == 2 && no_kernel.counts.frees == 2, "unsupported registry op stops execution; all outputs discarded");
    auto owner = sequence({2});
    auto gpu = Storage::wrap(Device(DeviceType::CUDA), owner.nbytes(), owner.data<float>(), [storage = owner.storage()](void*) noexcept { (void)storage; });
    Graph noncpu; noncpu.add_input(0, "gpu", Tensor(gpu, DType::FP32, Shape{2}, Stride{1}));
    noncpu.add_output("gpu", 0); success(noncpu.freeze());
    require(execute_graph(noncpu).status.code == StatusCode::DeviceMismatch, "CPU executor rejects simulated device metadata without dereferencing GPU");
}
void state_and_empty() {
    auto state = Tensor::allocate_cpu({4}), source = sequence({4});
    Graph graph; graph.add_input(0, "source", source); graph.add_input(1, "state", state, true);
    graph.add_tensor(2, {4}); graph.add_tensor(3, {4}); graph.add_tensor(4, {4});
    graph.add_node(0, OpDesc(OpCode::COPY, {0, 1}, {2}, CopyAttrs{}));
    graph.add_node(1, OpDesc(OpCode::COPY, {2, 2}, {3}, CopyAttrs{})); // exact self is zero bytes
    graph.add_node(2, OpDesc(OpCode::ADD, {3, 3}, {4}));
    graph.add_output("state", 3); graph.add_output("computed", 4); success(graph.freeze());
    auto result = execute_graph(graph); success(result.status);
    require(result.counts.allocations == 1 && result.counts.copies == 1 && result.counts.copy_bytes == 16 &&
            result.outputs.at("state").storage() == state.storage(), "explicit state writes do not allocate or implicitly copy");
    require(state.data<float>()[3] == 4 && result.outputs.at("computed").data<float>()[3] == 8, "write version is used by reads");
    Graph empty; empty.add_input(0, "x", Tensor::allocate_cpu({0})); empty.add_tensor(1, {0});
    empty.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1})); empty.add_output("empty", 1); success(empty.freeze());
    const auto zero = execute_graph(empty); success(zero.status);
    require(zero.counts.allocations == 0 && zero.counts.frees == 0 && zero.counts.nodes_completed == 1 && zero.counts.peak_live_bytes == 0, "empty outputs allocate no backing buffers");
    Graph passthrough; passthrough.add_input(0, "x", source); passthrough.add_output("x", 0); success(passthrough.freeze());
    const auto pass = execute_graph(passthrough); success(pass.status);
    require(pass.outputs.at("x").storage() == source.storage() && pass.counts.allocations == 0, "input-only graph no hidden copy");
}
}
int main() {
    try {
        composed(); lifetime(); failures(); state_and_empty();
        require(testing::cpu_allocation_counts().live == 0, "executor test leaks backing buffers");
        std::cout << "graph executor: PASS\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
