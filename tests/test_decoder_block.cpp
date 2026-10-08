#include "model/decoder.hpp"
#include "runtime/graph_executor.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
using namespace runtime;
using namespace model;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }

std::map<std::string, Tensor> load_parameters(const char* path, const DecoderConfig& config) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open tiny weights");
    std::map<std::string, Tensor> tensors;
    for (const auto& spec : parameter_specs(config)) {
        auto tensor = Tensor::allocate_cpu(spec.shape);
        input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
        if (!input) throw std::runtime_error("truncated tiny weights");
        tensors.emplace(spec.name, std::move(tensor));
    }
    require(input.peek() == std::ifstream::traits_type::eof(), "trailing tiny weights");
    return tensors;
}

Tensor load_expected(const char* path, Shape shape) {
    auto tensor = Tensor::allocate_cpu(std::move(shape));
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open block fixture");
    input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
    require(static_cast<bool>(input) && input.peek() == std::ifstream::traits_type::eof(), "invalid block fixture size");
    return tensor;
}

void compare(const Tensor& actual, const Tensor& expected) {
    require(actual.shape() == expected.shape(), "block output shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto a = actual.data<float>()[index], e = expected.data<float>()[index];
        require(std::isfinite(a) && std::abs(static_cast<double>(a) - e) <= 2e-6 + 2e-5 * std::abs(e),
                "block output differs from independent fixture");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected weights and block fixture");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, load_parameters(argv[1], config));
        const std::vector<std::int32_t> token_ids{256, 0, 1, 257};
        auto hidden = Tensor::allocate_cpu({static_cast<std::int64_t>(token_ids.size()), config.hidden});
        const auto& embedding = parameters.at("token_embedding");
        for (std::size_t token = 0; token < token_ids.size(); ++token)
            for (std::int64_t channel = 0; channel < config.hidden; ++channel)
                hidden.data<float>()[token * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(channel)] =
                    embedding.data<float>()[static_cast<std::size_t>(token_ids[token]) * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(channel)];

        Graph graph;
        graph.add_input(0, "hidden", hidden);
        GraphCursor cursor{1, 0};
        const auto block = build_decoder_block(graph, config, parameters, 0,
                                               static_cast<std::int64_t>(token_ids.size()), 0, cursor);
        require(block.first_node == 0 && block.node_count == 91, "declared block node count");
        graph.add_output("hidden", block.output);
        success(graph.freeze());
        require(graph.order().size() == block.node_count, "all block nodes are ordered");
        require(graph.inputs().size() == 11, "hidden, nine parameters and scale are explicit inputs");
        std::size_t projections = 0;
        for (const auto node : graph.order()) {
            const auto& descriptor = graph.nodes().at(node).descriptor;
            require(descriptor.code() != OpCode::ATTENTION && descriptor.outputs().size() == 1,
                    "block contains only graph-visible primitive intermediates");
            if (descriptor.code() == OpCode::MATMUL && descriptor.backend_hint()) ++projections;
        }
        require(projections == 10, "learned block projections carry placement hints");
        const auto result = execute_graph(graph);
        require(result.ok(), result.status.message.c_str());
        require(result.counts.nodes_completed == block.node_count, "every block node executes");
        compare(result.outputs.at("hidden"), load_expected(argv[2], {4, config.hidden}));

        Graph invalid; invalid.add_input(0, "hidden", hidden); GraphCursor invalid_cursor{1, 0};
        try { (void)build_decoder_block(invalid, config, parameters, config.layers, 4, 0, invalid_cursor);
              throw std::runtime_error("out-of-range layer accepted"); }
        catch (const std::out_of_range&) {}
        std::cout << "Stage 13 C2 decoder block graph: PASS nodes=" << block.node_count
                  << " projections=" << projections << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_decoder_block: " << error.what() << '\n';
        return 1;
    }
}
