#include "model/generation.hpp"
#include "model/model_file.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void compare_logits(const runtime::Tensor& actual, const runtime::Tensor& expected) {
    require(actual.shape().rank() == 2 && expected.shape().rank() == 2 &&
                actual.shape()[1] == expected.shape()[1],
            "generation logits shape");
    const auto width = static_cast<std::size_t>(actual.shape()[1]);
    const auto actual_offset = actual.numel() - width;
    const auto expected_offset = expected.numel() - width;
    for (std::size_t index = 0; index < width; ++index) {
        const auto tolerance = 2e-6 + 2e-5 * std::fabs(expected.data<float>()[expected_offset + index]);
        require(std::isfinite(actual.data<float>()[actual_offset + index]) &&
                    std::fabs(actual.data<float>()[actual_offset + index] - expected.data<float>()[expected_offset + index]) <= tolerance,
                "cache/no-cache logits diverged");
    }
}
} // namespace

int main(int argc, char** argv) {
    const auto temporary = std::filesystem::temp_directory_path() / "mini_llm_stage17_generation_int8.mllm";
    try {
        if (argc != 2) throw std::invalid_argument("expected V1 tiny model fixture");
        const auto loaded = model::load_model_file(argv[1]);
        const model::GenerationOptions cached{model::GenerationBackend::CPU, true, 4};
        const model::GenerationOptions uncached{model::GenerationBackend::CPU, false, 4};
        const auto first = model::generate_greedy(loaded, "Stage17", cached);
        const auto second = model::generate_greedy(loaded, "Stage17", cached);
        const auto baseline = model::generate_greedy(loaded, "Stage17", uncached);
        require(first.prompt_tokens == second.prompt_tokens && first.generated_tokens == second.generated_tokens &&
                    first.generated_tokens == baseline.generated_tokens,
                "greedy generation must be deterministic and cache-equivalent");
        require(first.generated_tokens == std::vector<std::int32_t>({73, 18, 3, 234}),
                "frozen Stage 17 float generation fixture changed");
        require(first.generated_tokens.size() == 4 && first.final_logits && second.final_logits && baseline.final_logits,
                "generation must retain a final greedy logit row");
        compare_logits(*first.final_logits, *second.final_logits);
        compare_logits(*first.final_logits, *baseline.final_logits);
        require(first.prefill_counts.allocations == 0 && first.total_counts.allocations == 0 &&
                    first.cache_persistent_bytes > 0 && baseline.cache_persistent_bytes == 0,
                "generation must use prepared graph storage and explicit persistent cache accounting");

        try {
            (void)model::generate_greedy(loaded, std::string(loaded.config().max_seq, 'x'), cached);
            throw std::runtime_error("over-budget prompt was accepted");
        } catch (const std::out_of_range&) {}

        model::write_quantized_model_file(temporary.string(), loaded.parameters);
        const auto quantized = model::load_model_file(temporary.string());
        const auto quantized_result = model::generate_greedy(quantized, "Stage17", cached);
        require(quantized_result.generated_tokens.size() == 4 && quantized_result.final_logits &&
                    quantized_result.total_counts.allocations == 0,
                "INT8 prepare_dequant model must use the ordinary generation path");
        for (std::size_t index = 0; index < quantized_result.final_logits->numel(); ++index)
            require(std::isfinite(quantized_result.final_logits->data<float>()[index]), "INT8 generation logit is nonfinite");

        std::filesystem::remove(temporary);
        std::cout << "Stage 17 C2 greedy prefill/decode generation: PASS generated="
                  << first.generated_tokens.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(temporary);
        std::cerr << "test_generation: " << error.what() << '\n';
        return 1;
    }
}
