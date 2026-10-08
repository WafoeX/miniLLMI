#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/graph_executor.hpp"
#include "runtime/scheduler.hpp"

#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
using namespace model;
using decoder_fixture::require;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected weights and logits fixture");
        const auto config = DecoderConfig::tiny();
        const ParameterTable parameters(config, decoder_fixture::load_parameters(argv[1], config));
        const auto expected = decoder_fixture::load_tensor(argv[2], {4, config.vocab});
        CudaBackend cuda(0, CudaMatmul::Stage0Naive);
        Scheduler scheduler(default_cpu_backend(), &cuda);
        const auto logical = build_decoder_prefill(config, parameters,
            decoder_fixture::token_tensor({256, 0, 1, 257}), {cuda.device()});
        const auto scheduled = scheduler.rewrite(logical.graph);
        require(scheduled.ok(), scheduled.status.message.c_str());
        std::size_t cuda_placements = 0;
        for (const auto& item : scheduled.placements) {
            const auto& descriptor = logical.graph.nodes().at(item.first).descriptor;
            if (descriptor.code() == OpCode::MATMUL && descriptor.backend_hint() &&
                item.second.device == cuda.device()) ++cuda_placements;
        }
        require(cuda_placements == logical.projection_nodes && cuda_placements == 21,
                "all learned projections are placed on CUDA");
        require(!scheduled.inserted_copies.empty(), "mixed decoder must expose scheduler copies");
        bool host_to_device = false, device_to_host = false;
        for (const auto& copy : scheduled.inserted_copies) {
            host_to_device = host_to_device || (copy.from == Device{} && copy.to == cuda.device());
            device_to_host = device_to_host || (copy.from == cuda.device() && copy.to == Device{});
        }
        require(host_to_device && device_to_host, "mixed decoder requires explicit copies in both directions");
        std::size_t expected_copies = 0, expected_copy_bytes = 0;
        for (const auto node : scheduled.graph->order()) {
            const auto& descriptor = scheduled.graph->nodes().at(node).descriptor;
            if (descriptor.code() != OpCode::COPY && descriptor.code() != OpCode::MATERIALIZE) continue;
            const auto& source = scheduled.graph->tensors().at(descriptor.inputs()[0]);
            if (runtime::numel(source.shape) == 0) continue;
            ++expected_copies;
            expected_copy_bytes += runtime::nbytes(source.shape, source.dtype);
        }
        require(expected_copies > scheduled.inserted_copies.size(),
                "copy accounting includes graph materializations and explicit V boundaries");
        ScheduledAllocationProvider prepared(*scheduled.graph, scheduler);
        const auto result = execute_graph(*scheduled.graph, nullptr, &prepared, nullptr, &scheduler);
        require(result.ok(), result.status.message.c_str());
        require(result.outputs.at("logits").device() == Device{}, "mixed logits return to CPU");
        if (result.counts.allocations != 0 || result.counts.frees != 0 ||
            result.counts.copies != expected_copies || result.counts.copy_bytes != expected_copy_bytes)
            throw std::runtime_error("mixed decoder accounting: allocations=" + std::to_string(result.counts.allocations) +
                " frees=" + std::to_string(result.counts.frees) + " copies=" + std::to_string(result.counts.copies) +
                "/" + std::to_string(expected_copies) + " copy_bytes=" + std::to_string(result.counts.copy_bytes) +
                "/" + std::to_string(expected_copy_bytes));
        decoder_fixture::compare(result.outputs.at("logits"), expected, 2e-4, 2e-3);
        std::cout << "Stage 13 C4 mixed decoder: PASS logical_nodes=" << logical.graph.order().size()
                  << " scheduled_nodes=" << scheduled.graph->order().size()
                  << " cuda_projections=" << cuda_placements
                  << " copies=" << result.counts.copies
                  << " copy_bytes=" << result.counts.copy_bytes
                  << " cpu_capacity=" << prepared.plan().device_capacity_bytes.at(Device{})
                  << " cuda_capacity=" << prepared.plan().device_capacity_bytes.at(cuda.device()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_decoder: " << error.what() << '\n';
        return 1;
    }
}
