#pragma once
#include "runtime/graph.hpp"
#include <stdexcept>

namespace scheduler_workload {
using namespace runtime;
inline void check(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }
struct Inputs { Tensor x, bias, weight, scale; };
inline Inputs inputs(std::int64_t m, std::int64_t k, std::int64_t n) {
    Inputs in{Tensor::allocate_cpu({m, k}), Tensor::allocate_cpu({m, k}),
              Tensor::allocate_cpu({k, n}), Tensor::allocate_cpu({m, n})};
    int salt = 1;
    for (auto* t : {&in.x, &in.bias, &in.weight, &in.scale}) {
        for (std::size_t i = 0; i < t->numel(); ++i) t->data<float>()[i] = static_cast<float>(static_cast<int>((i * 17 + salt) % 31) - 15) / 32;
        ++salt;
    }
    return in;
}
// The same CPU ADD -> CUDA MATMUL fan-out -> CPU ADD/MUL graph.
// Both modes use exactly the same tensors, hints, kernels and node order.
inline Graph graph(const Inputs& in, bool manual, bool cpu_reference = false) {
    const Device cpu{}, gpu(DeviceType::CUDA);
    const auto mk = in.x.shape(), kn = in.weight.shape(), mn = in.scale.shape();
    Graph g;
    g.add_input(0, "x", in.x); g.add_input(1, "bias", in.bias);
    g.add_input(2, "weight", in.weight); g.add_input(8, "scale", in.scale);
    g.add_tensor(3, mk); g.add_tensor(4, mn, DType::FP32, manual ? gpu : cpu);
    g.add_tensor(5, mn, DType::FP32, manual ? gpu : cpu); g.add_tensor(6, mn); g.add_tensor(7, mn);
    g.add_node(0, OpDesc(OpCode::ADD, {0, 1}, {3}));
    TensorId a = 3, w = 2, left = 4, right = 5;
    if (manual) {
        g.add_tensor(100, mk, DType::FP32, gpu); g.add_tensor(101, kn, DType::FP32, gpu);
        g.add_node(1, OpDesc(OpCode::COPY, {3}, {100}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, gpu}));
        g.add_node(2, OpDesc(OpCode::COPY, {2}, {101}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, gpu}));
        a = 100; w = 101;
    }
    const auto requested = cpu_reference ? cpu : gpu;
    g.add_node(3, OpDesc(OpCode::MATMUL, {a, w}, {4}, {}, requested));
    g.add_node(4, OpDesc(OpCode::MATMUL, {a, w}, {5}, {}, requested));
    if (manual) {
        g.add_tensor(102, mn); g.add_tensor(103, mn);
        g.add_node(5, OpDesc(OpCode::COPY, {4}, {102}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, cpu}));
        g.add_node(6, OpDesc(OpCode::COPY, {5}, {103}, CopyAttrs{CopyOverlap::RejectExceptExactSelf, cpu}));
        left = 102; right = 103;
    }
    g.add_node(7, OpDesc(OpCode::ADD, {left, right}, {6}, {}, cpu_reference || manual ? cpu : gpu));
    g.add_node(8, OpDesc(OpCode::MUL, {6, 8}, {7}));
    g.add_output("result", 7); check(g.freeze()); return g;
}
} // namespace scheduler_workload
