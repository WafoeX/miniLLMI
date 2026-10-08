#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "model/model_file.hpp"
#include "model/quantization.hpp"
#include "runtime/planned_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using decoder_fixture::require;

std::uint64_t checksum(const runtime::Tensor& tensor) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(tensor.data<float>());
    for (std::size_t index = 0; index < tensor.nbytes(); ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void compare_quantized_parameters(const model::ParameterTable& original, const model::LoadedModel& loaded) {
    for (const auto& spec : model::parameter_specs(original.config())) {
        if (!model::is_weight_only_int8_eligible(spec)) continue;
        const auto source = std::find_if(loaded.prepare_dequant_weights.begin(), loaded.prepare_dequant_weights.end(),
            [&](const model::PrepareDequantWeight& weight) { return weight.name == spec.name; });
        require(source != loaded.prepare_dequant_weights.end(), "eligible weight lacks prepare_dequant source");
        const auto& expected = original.at(spec.name);
        const auto& actual = loaded.parameters.at(spec.name);
        const auto* scales = source->source.scales.data<float>();
        for (std::size_t index = 0; index < expected.numel(); ++index) {
            const auto output = index % static_cast<std::size_t>(spec.shape[1]);
            const auto tolerance = scales[output] / 2.F +
                4.F * std::numeric_limits<float>::epsilon() * std::fabs(expected.data<float>()[index]);
            require(std::isfinite(actual.data<float>()[index]) &&
                        std::fabs(actual.data<float>()[index] - expected.data<float>()[index]) <= tolerance,
                    "prepare_dequant element error");
        }
    }
}

void compare_quantized_logits(const runtime::Tensor& actual, const runtime::Tensor& expected) {
    require(actual.shape() == expected.shape(), "quantized logits shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto value = actual.data<float>()[index];
        const auto reference = expected.data<float>()[index];
        const auto tolerance = 1e-2F + 1e-2F * std::fabs(reference);
        require(std::isfinite(value) && std::fabs(value - reference) <= tolerance,
                "quantized logit error exceeds frozen tolerance");
    }
}
} // namespace

int main(int argc, char** argv) {
    const auto root = std::filesystem::temp_directory_path();
    const auto temporary = root / "mini_llm_stage15_roundtrip.mllm";
    const auto quantized_temporary = root / "mini_llm_stage16_prepare_dequant.mllm";
    try {
        if (argc != 4) throw std::invalid_argument("expected legacy weights, model artifact, and logits fixture");
        const auto config = model::DecoderConfig::tiny();
        const model::ParameterTable original(config, decoder_fixture::load_parameters(argv[1], config));
        const auto persisted = model::load_model_file(argv[2]);
        require(persisted.vocabulary_version == model::BYTE_VOCABULARY_VERSION && persisted.config().vocab == config.vocab,
                "persistent model vocabulary/config");
        model::write_model_file(temporary.string(), original);
        const auto loaded = model::load_model_file(temporary.string());
        require(loaded.vocabulary_version == model::BYTE_VOCABULARY_VERSION && loaded.config().vocab == config.vocab,
                "model vocabulary/config round-trip");
        for (const auto& spec : model::parameter_specs(config)) {
            const auto& expected = original.at(spec.name);
            const auto& actual = loaded.parameters.at(spec.name);
            const auto& persisted_tensor = persisted.parameters.at(spec.name);
            require(actual.nbytes() == expected.nbytes() && checksum(actual) == checksum(expected) &&
                        checksum(persisted_tensor) == checksum(expected),
                    "persistent model payload checksum round-trip");
        }

        model::DecoderGraph decoder;
        {
            auto ephemeral = model::load_model_file(argv[2]);
            decoder = model::build_decoder_prefill(config, ephemeral.parameters,
                                                    decoder_fixture::token_tensor({256, 0, 1, 257}));
        }
        runtime::PlannedAllocationProvider plan(decoder.graph);
        const auto result = runtime::execute_planned(decoder.graph, plan);
        require(result.ok(), result.status.message.c_str());
        const auto expected_logits = decoder_fixture::load_tensor(argv[3], {4, config.vocab});
        decoder_fixture::compare(result.outputs.at("logits"), expected_logits);
        require(result.counts.allocations == 0, "loaded model planned execution allocation-free");

        model::write_quantized_model_file(quantized_temporary.string(), original);
        const auto quantized = model::load_model_file(quantized_temporary.string());
        require(quantized.prepare_dequant_weights.size() == 15, "all eligible weights retain INT8/scales sources");
        compare_quantized_parameters(original, quantized);
        const auto quantized_decoder = model::build_decoder_prefill(config, quantized.parameters,
                                                                      decoder_fixture::token_tensor({256, 0, 1, 257}));
        runtime::PlannedAllocationProvider quantized_plan(quantized_decoder.graph);
        const auto quantized_result = runtime::execute_planned(quantized_decoder.graph, quantized_plan);
        require(quantized_result.ok(), quantized_result.status.message.c_str());
        compare_quantized_logits(quantized_result.outputs.at("logits"), expected_logits);
        require(quantized_result.counts.allocations == 0, "prepare_dequant execution allocation-free");

        std::filesystem::remove(temporary);
        std::filesystem::remove(quantized_temporary);
        std::cout << "Stage 15 C2 and Stage 16 C3 model write/load/prepare_dequant/logit round-trip: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(temporary);
        std::filesystem::remove(quantized_temporary);
        std::cerr << "test_model_loader: " << error.what() << '\n';
        return 1;
    }
}
