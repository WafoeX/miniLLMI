#include "runtime/cpu_backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_executor.hpp"
#include "runtime/thread_pool.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }

void fifo_and_status() {
    ThreadPool pool(4);
    require(pool.worker_count() == 4 && pool.creation_count() == 4, "persistent workers created once");
    std::vector<std::atomic<unsigned>> calls(97);
    success(pool.parallel_for(calls.size(), [&calls](std::size_t index) {
        ++calls[index];
        return Status::success();
    }));
    for (const auto& count : calls) require(count.load() == 1, "FIFO task executed exactly once");
    require(pool.parallel_for(0, [](std::size_t) { return Status::success(); }).ok(), "zero-task pool completion");
    const auto failure = pool.parallel_for(19, [](std::size_t index) {
        return index == 7 ? Status::failure(StatusCode::Unsupported, "test failure") : Status::success();
    });
    require(failure.code == StatusCode::Unsupported && !failure.message.empty(), "task status propagated");
    const auto throwing = pool.parallel_for(1, [](std::size_t) -> Status { throw std::runtime_error("test throw"); });
    require(throwing.code == StatusCode::InvalidArgument && !throwing.message.empty(), "task exception propagated");
    success(pool.parallel_for(3, [](std::size_t) { return Status::success(); }));
}

void pool_backend() {
    auto a = Tensor::allocate_cpu({3, 4}), b = Tensor::allocate_cpu({4, 2}), out = Tensor::allocate_cpu({3, 2});
    for (std::size_t i = 0; i < a.numel(); ++i) a.data<float>()[i] = static_cast<float>(i + 1);
    for (std::size_t i = 0; i < b.numel(); ++i) b.data<float>()[i] = static_cast<float>(i + 1);
    const OpDesc mm(OpCode::MATMUL, {0, 1}, {2});
    for (std::size_t workers : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        CpuBackend backend(CpuMatmul::LoopIKJFifoPoolFP32C3, workers);
        require(backend.workers() == workers, "configured FIFO worker count");
        success(backend.execute(mm, {a, b}, out));
        const float expected[] = {50, 60, 114, 140, 178, 220};
        for (std::size_t i = 0; i < out.numel(); ++i) require(out.data<float>()[i] == expected[i], "FIFO backend output");
        Graph graph;
        graph.add_input(0, "a", a); graph.add_input(1, "b", b); graph.add_tensor(2, {3, 2});
        graph.add_node(0, mm); graph.add_output("result", 2); success(graph.freeze());
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            const auto result = execute_graph(graph, nullptr, nullptr, &backend);
            require(result.ok() && result.outputs.at("result").data<float>()[5] == 220, "repeated FIFO graph execution");
        }
    }
    bool rejected = false;
    try { (void)CpuBackend(CpuMatmul::LoopIKJFP32C1, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "non-pool algorithms reject worker configuration");
}
} // namespace

int main() {
    try {
        fifo_and_status(); pool_backend();
        std::cout << "CPU FIFO pool: PASS exactly-once/status/exception/zero/shutdown/backend workers=1,2,4\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_thread_pool: " << error.what() << '\n';
        return 1;
    }
}
