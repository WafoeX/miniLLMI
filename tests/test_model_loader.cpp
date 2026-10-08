#include "decoder_fixture.hpp"
#include "model/decoder.hpp"
#include "model/model_file.hpp"
#include "runtime/planned_executor.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
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
} // namespace

int main(int argc, char** argv) {
    const auto temporary = std::filesystem::temp_directory_path() / "mini_llm_stage15_roundtrip.mllm";
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

        std::filesystem::remove(temporary);
        std::cout << "Stage 15 C2 model write/load/logit round-trip: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(temporary);
        std::cerr << "test_model_loader: " << error.what() << '\n';
        return 1;
    }
}
