#include "fixture_reader.hpp"
#include "runtime/graph_executor.hpp"
#include <fstream>
#include <iostream>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
bool is_numeric(OpCode code) {
    return code != OpCode::ATTENTION;
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: test_graph_fixtures <verified operators-v1.txt>");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("cannot open fixture file");
        const auto cases = fixtures::load(input);
        std::size_t numeric = 0, unsupported = 0;
        for (const auto& fixture : cases) {
            Graph graph;
            for (std::size_t i = 0; i < fixture.inputs.size(); ++i)
                graph.add_input(static_cast<TensorId>(i), "input-" + std::to_string(i), fixture.inputs[i],
                                fixture.descriptor.code() == OpCode::COPY && i == 1);
            const auto output = fixture.descriptor.outputs()[0];
            graph.add_tensor(output, fixture.expected.shape(), fixture.expected.dtype());
            graph.add_node(0, fixture.descriptor); graph.add_output("result", output);
            const auto frozen = graph.freeze();
            if (!frozen.ok()) throw std::runtime_error(fixture.name + ": " + frozen.message);
            const auto before = testing::cpu_allocation_counts();
            {
                const auto result = execute_graph(graph);
                if (is_numeric(fixture.descriptor.code())) {
                    if (!result.ok()) throw std::runtime_error(fixture.name + ": " + result.status.message);
                    const auto& actual = result.outputs.at("result");
                    for (std::size_t i = 0; i < actual.numel(); ++i) {
                        if (actual.dtype() == DType::INT32)
                            require(actual.data<std::int32_t>()[i] == fixture.expected.data<std::int32_t>()[i], "graph INT32 fixture mismatch");
                        else {
                            const auto value = actual.data<float>()[i], expected = fixture.expected.data<float>()[i];
                            require(std::isfinite(value) && std::abs(static_cast<double>(value) - expected) <= fixture.atol + fixture.rtol * std::abs(expected), "graph FP32 fixture tolerance mismatch");
                        }
                    }
                    ++numeric; std::cout << "graph fixture " << fixture.name << " core=PASS\n";
                } else {
                    require(result.status.code == StatusCode::Unsupported && result.failed_node == 0 && result.outputs.empty(), "ATTENTION must compose graph operators instead of dispatching an opaque kernel");
                    ++unsupported; std::cout << "graph fixture " << fixture.name << " unsupported=PASS\n";
                }
                require(testing::cpu_allocation_counts().allocations - before.allocations == result.counts.allocations, "fixture graph allocation counter agrees with Storage");
            }
            require(testing::cpu_allocation_counts().live == before.live, "fixture graph leaves no transient backing buffers");
        }
        std::cout << "Graph fixtures: PASS numeric_core=" << numeric << " unsupported=" << unsupported << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
