#include "fixture_reader.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/planned_executor.hpp"
#include <fstream>
#include <iostream>
#include <set>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void unsupported(const Status& status, OpCode code) {
    require(status.code == StatusCode::Unsupported && status.message == std::string(op_name(code)) + " CPU primitive is not implemented", "unimplemented primitive must return exact Unsupported diagnostic");
}
bool c1(OpCode code) { return code == OpCode::RMSNORM || code == OpCode::SWIGLU; }
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected verified fixture file");
        std::ifstream stream(argv[1]); const auto cases = fixtures::load(stream);
        std::set<OpCode> covered;
        for (auto mode : {CpuMatmul::ReferenceFP64, CpuMatmul::ScalarFP32V0}) {
            CpuBackend backend(mode);
            for (const auto& fixture : cases) {
                const auto code = fixture.descriptor.code();
                if (code != OpCode::RMSNORM && code != OpCode::SOFTMAX && code != OpCode::ROPE && code != OpCode::EMBEDDING && code != OpCode::SWIGLU && code != OpCode::ATTENTION) continue;
                TensorInputs inputs; for (const auto& tensor : fixture.inputs) inputs.emplace_back(tensor);
                auto out = Tensor::allocate_cpu(fixture.expected.shape(), fixture.expected.dtype());
                for (std::size_t i = 0; i < out.numel(); ++i) out.data<float>()[i] = 99;
                if (c1(code)) {
                    require(backend.capability(code, Device{}, out.dtype()).ok(), "C1 primitive capability");
                    require(backend.prepare(fixture.descriptor, inputs, out).ok(), "C1 primitive prepare");
                    require(backend.execute(fixture.descriptor, inputs, out).ok(), "C1 primitive execute");
                } else {
                    unsupported(backend.capability(code, Device{}, out.dtype()), code);
                    unsupported(backend.prepare(fixture.descriptor, inputs, out).status, code);
                    unsupported(backend.execute(fixture.descriptor, inputs, out), code);
                    for (std::size_t i = 0; i < out.numel(); ++i) require(out.data<float>()[i] == 99, "unsupported primitives cannot write output");
                }
                covered.insert(code);
            }
            Graph graph; auto x = Tensor::allocate_cpu({1, 2}); x.data<float>()[0] = 1; x.data<float>()[1] = 2;
            graph.add_input(0, "x", x); graph.add_tensor(1, {1, 2}); graph.add_tensor(2, {1, 2});
            graph.add_node(0, OpDesc(OpCode::ADD, {0, 0}, {1}));
            graph.add_node(1, OpDesc(OpCode::SOFTMAX, {1}, {2}, SoftmaxAttrs{}));
            graph.add_output("probabilities", 2); require(graph.freeze().ok(), "reserved primitive graph remains inferable");
            PlannedAllocationProvider plan(graph);
            for (AllocationProvider* provider : {static_cast<AllocationProvider*>(nullptr), static_cast<AllocationProvider*>(&plan)}) {
                const auto result = execute_graph(graph, nullptr, provider, &backend);
                require(result.status.code == StatusCode::Unsupported && result.failed_node == 1 && result.counts.nodes_completed == 1 && result.outputs.empty() && result.counts.live_bytes == 0, "graph must stop on first unsupported primitive and release bindings");
            }
        }
        require(covered.size() == 6, "all six S2 transformer descriptors covered");
        std::cout << "CPU transformer capability transition: PASS C1 enabled, C2-C4 still Unsupported\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "test_cpu_transformer_contract: " << e.what() << '\n'; return 1; }
}
