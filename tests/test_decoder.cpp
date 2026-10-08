#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "runtime/planned_executor.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace runtime;
using namespace model;
using decoder_fixture::require;
void success(const Status& status) { if (!status.ok()) throw std::runtime_error(status.message); }
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected weights and logits fixture");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, decoder_fixture::load_parameters(argv[1], config));
        auto ids = decoder_fixture::token_tensor({256, 0, 1, 257});
        const auto expected = decoder_fixture::load_tensor(argv[2], {4, config.vocab});
        const auto decoder = build_decoder_prefill(config, parameters, ids);
        require(decoder.sequence_length == 4 && decoder.graph.order().size() == 185, "full decoder topology");
        require(decoder.graph.inputs().size() == 24 && decoder.projection_nodes == 21, "full decoder bindings/projections");

        const auto dynamic = execute_graph(decoder.graph);
        require(dynamic.ok(), dynamic.status.message.c_str());
        require(dynamic.counts.nodes_completed == decoder.graph.order().size() && dynamic.counts.allocations > 0,
                "dynamic baseline executes every node and allocates intermediates");
        decoder_fixture::compare(dynamic.outputs.at("logits"), expected);

        PlannedAllocationProvider prepared(decoder.graph);
        const auto prepared_capacity = prepared.capacity();
        const auto before_execute = testing::cpu_allocation_counts();
        {
            const auto planned = execute_planned(decoder.graph, prepared);
            require(planned.ok(), planned.status.message.c_str());
            require(planned.counts.allocations == 0 && planned.counts.frees == 0 &&
                    planned.counts.allocation_requests > 0 && planned.counts.arena_capacity_bytes == prepared_capacity,
                    "planned decoder has zero execute-time backing allocations");
            decoder_fixture::compare(planned.outputs.at("logits"), expected);
            decoder_fixture::compare(planned.outputs.at("logits"), dynamic.outputs.at("logits"), 0, 0);
        }
        require(testing::cpu_allocation_counts().allocations == before_execute.allocations,
                "planned execute does not allocate backing storage");

        ids.data<std::int32_t>()[1] = 2;
        {
            const auto changed = execute_planned(decoder.graph, prepared);
            success(changed.status);
            decoder_fixture::require_finite(changed.outputs.at("logits"));
            require(changed.outputs.at("logits").data<float>()[config.vocab] != expected.data<float>()[config.vocab],
                    "same-shape execution observes caller-updated token storage");
        }
        ids.data<std::int32_t>()[1] = 0;
        {
            const auto repeated = execute_planned(decoder.graph, prepared);
            success(repeated.status);
            decoder_fixture::compare(repeated.outputs.at("logits"), expected);
        }

        auto single_ids = decoder_fixture::token_tensor({256});
        const auto single = build_decoder_prefill(config, parameters, single_ids);
        const auto stale = execute_planned(single.graph, prepared);
        require(stale.status.code == StatusCode::InvalidArgument && stale.counts.nodes_completed == 0,
                "new active shape rejects stale prepared plan before execution");
        PlannedAllocationProvider single_prepared(single.graph);
        {
            const auto result = execute_planned(single.graph, single_prepared);
            success(result.status);
            require(result.outputs.at("logits").shape() == Shape({1, config.vocab}), "sequence-length one executes");
            decoder_fixture::require_finite(result.outputs.at("logits"));
        }

        auto maximum_ids = Tensor::allocate_cpu({config.max_seq}, DType::INT32);
        const auto maximum = build_decoder_prefill(config, parameters, maximum_ids);
        require(maximum.graph.tensors().at(maximum.logits).shape == Shape({config.max_seq, config.vocab}),
                "maximum sequence boundary builds and freezes");
        try {
            auto too_long = Tensor::allocate_cpu({config.max_seq + 1}, DType::INT32);
            (void)build_decoder_prefill(config, parameters, std::move(too_long));
            throw std::runtime_error("sequence above max accepted");
        } catch (const std::out_of_range&) {}

        auto invalid_ids = decoder_fixture::token_tensor({static_cast<std::int32_t>(config.vocab)});
        const auto invalid = build_decoder_prefill(config, parameters, invalid_ids);
        const auto rejected = execute_graph(invalid.graph);
        require(rejected.status.code == StatusCode::OutOfRange && rejected.failed_node == 0,
                "embedding rejects invalid token before decoder execution");

        std::cout << "Stage 13 C3 tiny decoder prefill: PASS nodes=" << decoder.graph.order().size()
                  << " dynamic_allocations=" << dynamic.counts.allocations
                  << " planned_capacity_bytes=" << prepared_capacity << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_decoder: " << error.what() << '\n';
        return 1;
    }
}
