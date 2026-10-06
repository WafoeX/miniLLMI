#include "runtime/cpu_backend.hpp"
#include "runtime/planned_executor.hpp"
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }
void expect(const Status& status, StatusCode code) { require(status.code == code && !status.message.empty(), "backend failure classification"); }
// Graph executor sees only the common interface, not CPU kernel declarations.
class CountingBackend final : public Backend {
    CpuBackend cpu_;
public:
    mutable std::size_t calls = 0;
    bool fail = false;
    const char* name() const noexcept override { return "test-decorator"; }
    Device device() const noexcept override { return cpu_.device(); }
    Status capability(OpCode c, Device d, DType t) const override { return cpu_.capability(c, d, t); }
    BackendBuffer allocate(Shape s, DType t, Device d) const override { return cpu_.allocate(s, t, d); }
    Status copy(const Tensor& a, Tensor& b) const override { return cpu_.copy(a, b); }
    BackendPreparation prepare(const OpDesc& d, const TensorInputs& i, const Tensor& o) const override { return cpu_.prepare(d, i, o); }
    Status execute(const OpDesc& d, const TensorInputs& i, Tensor& o, Workspace w) const override {
        ++calls;
        return fail ? Status::failure(StatusCode::Unsupported, "intentional backend failure") : cpu_.execute(d, i, o, w);
    }
};
void contracts() {
    CpuBackend cpu;
    success(cpu.capability(OpCode::ADD, Device{}, DType::FP32));
    expect(cpu.capability(OpCode::ADD, Device{}, DType::INT32), StatusCode::DTypeMismatch);
    expect(cpu.capability(OpCode::ADD, Device(DeviceType::CUDA), DType::FP32), StatusCode::DeviceMismatch);
    expect(cpu.capability(static_cast<OpCode>(999), Device{}, DType::FP32), StatusCode::Unsupported);
    auto allocated = cpu.allocate({2, 3}, DType::FP32, Device{});
    require(allocated.ok(), "backend buffer uses unified Tensor");
    require(!cpu.allocate({2}, DType::FP32, Device(DeviceType::CUDA)).ok(), "no CPU fallback for CUDA allocation");
    auto a = *allocated.tensor, b = Tensor::allocate_cpu({2, 3}), out = Tensor::allocate_cpu({2, 3});
    for (std::size_t i = 0; i < a.numel(); ++i) { a.data<float>()[i] = 2; b.data<float>()[i] = 3; }
    OpDesc add(OpCode::ADD, {0, 1}, {2});
    const auto before = testing::cpu_allocation_counts();
    const auto prep = cpu.prepare(add, {a, b}, out);
    require(prep.ok() && prep.workspace_bytes == 0, "CPU prepare has zero scratch");
    success(cpu.execute(add, {a, b}, out));
    require(testing::cpu_allocation_counts().allocations == before.allocations, "prepare/execute cannot allocate backing buffers");
    for (std::size_t i = 0; i < out.numel(); ++i) require(out.data<float>()[i] == 5, "CPU ADD through backend");
    expect(cpu.execute(add, {a, b}, out, {out.data<float>(), 4}), StatusCode::InvalidArgument);
    expect(cpu.prepare(add, {a}, out).status, StatusCode::ArityMismatch);
    OpDesc hinted(OpCode::ADD, {0, 1}, {2}, {}, Device(DeviceType::CUDA));
    expect(cpu.execute(hinted, {a, b}, out), StatusCode::DeviceMismatch);
    success(cpu.copy(a, b)); require(b.data<float>()[0] == 2, "backend explicit copy");
    auto transposed = a.transpose(0, 1);
    OpDesc view(OpCode::TRANSPOSE, {0}, {2}, TransposeAttrs{0, 1});
    success(cpu.execute(view, {a}, transposed));
    expect(cpu.execute(add, {transposed, transposed}, transposed), StatusCode::LayoutMismatch);
}
void graphs() {
    Graph graph;
    auto input = Tensor::allocate_cpu({2, 3});
    for (std::size_t i = 0; i < input.numel(); ++i) input.data<float>()[i] = 2;
    graph.add_input(0, "input", input);
    graph.add_tensor(1, {2, 3}); graph.add_tensor(2, {3, 2});
    graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
    graph.add_node(1, OpDesc(OpCode::RESHAPE, {1}, {2}, ReshapeAttrs{Shape{3, 2}}));
    graph.add_output("result", 2); success(graph.freeze());
    CountingBackend backend;
    { const auto result = execute_graph(graph, nullptr, nullptr, &backend);
      require(result.ok() && backend.calls == 2, "all graph nodes dispatch through Backend");
      require(result.outputs.at("result").data<float>()[0] == 4, "backend graph result"); }
    PlannedAllocationProvider planned(graph);
    { const auto result = execute_planned(graph, planned, nullptr, &backend);
      require(result.ok() && result.counts.allocations == 0 && backend.calls == 4, "S5 provider uses same backend executor"); }
    backend.fail = true;
    { const auto result = execute_planned(graph, planned, nullptr, &backend);
      expect(result.status, StatusCode::Unsupported);
      require(result.failed_node == 0 && result.outputs.empty() && result.counts.live_bytes == 0, "first backend error cleans output leases"); }
    backend.fail = false;
    require(execute_planned(graph, planned, nullptr, &backend).ok(), "provider reusable after backend error");
}
} // namespace
int main() {
    try {
        contracts(); graphs();
        require(testing::cpu_allocation_counts().live == 0, "backend tests leaked backing");
        std::cout << "CPU backend interface/capability/graph/provider: PASS\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "test_cpu_backend: " << e.what() << '\n'; return 1; }
}
