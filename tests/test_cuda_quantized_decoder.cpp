#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "model/model_file.hpp"
#include "runtime/cuda_backend.hpp"
#include "runtime/graph_executor.hpp"
#include "runtime/scheduler.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
using namespace model;
using decoder_fixture::require;

void compare_quantized_logits(const Tensor& actual, const Tensor& expected) {
    require(actual.shape() == expected.shape(), "quantized CUDA logits shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto value = actual.data<float>()[index];
        const auto reference = expected.data<float>()[index];
        const auto tolerance = 1e-2F + 1e-2F * std::fabs(reference);
        require(std::isfinite(value) && std::fabs(value - reference) <= tolerance,
                "quantized CUDA logit error exceeds frozen tolerance");
    }
}
} // namespace

int main(int argc, char** argv) {
    const auto artifact = std::filesystem::temp_directory_path() / "mini_llm_stage16_cuda_prepare_dequant.mllm";
    try {
        if (argc != 3) throw std::invalid_argument("expected weights and logits fixture");
        const auto config = DecoderConfig::tiny();
        const ParameterTable original(config, decoder_fixture::load_parameters(argv[1], config));
        write_quantized_model_file(artifact.string(), original);
        const auto quantized = load_model_file(artifact.string());
        require(quantized.prepare_dequant_weights.size() == 15, "all eligible sources retained during prepare_dequant");
        const auto expected = decoder_fixture::load_tensor(argv[2], {4, config.vocab});
        CudaBackend cuda(0, CudaMatmul::Stage0Naive);
        Scheduler scheduler(default_cpu_backend(), &cuda);
        const auto logical = build_decoder_prefill(config, quantized.parameters,
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
                "all prepare_dequant projections are placed on CUDA");
        require(!scheduled.inserted_copies.empty(), "mixed decoder must expose scheduler copies");
        ScheduledAllocationProvider prepared(*scheduled.graph, scheduler);
        const auto result = execute_graph(*scheduled.graph, nullptr, &prepared, nullptr, &scheduler);
        require(result.ok(), result.status.message.c_str());
        require(result.outputs.at("logits").device() == Device{}, "mixed logits return to CPU");
        require(result.counts.allocations == 0 && result.counts.frees == 0,
                "prepared quantized execute has no intermediate backing allocations");
        compare_quantized_logits(result.outputs.at("logits"), expected);
        std::filesystem::remove(artifact);
        std::cout << "Stage 16 C3 mixed prepare_dequant decoder: PASS cuda_projections=" << cuda_placements
                  << " copies=" << result.counts.copies << " copy_bytes=" << result.counts.copy_bytes
                  << " cpu_capacity=" << prepared.plan().device_capacity_bytes.at(Device{})
                  << " cuda_capacity=" << prepared.plan().device_capacity_bytes.at(cuda.device()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(artifact);
        std::cerr << "test_cuda_quantized_decoder: " << error.what() << '\n';
        return 1;
    }
}
