#include "model/quantization.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class Action>
void require_rejected(Action&& action, const char* message) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_per_channel_rounding_and_error() {
    auto weight = runtime::Tensor::allocate_cpu({2, 4});
    auto* data = weight.data<float>();
    // Each column has max(abs)=1. Columns 0/1 exercise +.5 ties to even,
    // column 2 is all zero, and column 3 exercises a negative .5 tie.
    data[0] = 1.F; data[1] = 1.F; data[2] = 0.F; data[3] = -1.F;
    data[4] = 0.5F; data[5] = 62.5F / 127.F; data[6] = 0.F; data[7] = -63.5F / 127.F;
    const auto quantized = model::quantize_per_output_channel(weight);
    model::validate_per_output_channel_int8(quantized);
    const auto* q = quantized.values.data<std::int8_t>();
    const auto* scales = quantized.scales.data<float>();
    require(scales[0] == 1.F / 127.F && scales[1] == 1.F / 127.F && scales[2] == 1.F &&
                scales[3] == 1.F / 127.F,
            "per-output scales");
    require(q[0] == 127 && q[4] == 64 && q[1] == 127 && q[5] == 62 && q[2] == 0 && q[6] == 0 &&
                q[3] == -127 && q[7] == -64,
            "ties-to-even or zero-column quantization");

    const auto reconstructed = model::dequantize_per_output_channel(quantized);
    for (std::size_t index = 0; index < weight.numel(); ++index) {
        const auto output = index % 4;
        const auto allowance = scales[output] / 2.F +
            4.F * std::numeric_limits<float>::epsilon() * std::fabs(data[index]);
        require(std::fabs(reconstructed.data<float>()[index] - data[index]) <= allowance,
                "dequantized element exceeds scale/2 allowance");
    }
}

void test_validation_and_eligibility() {
    auto nonfinite = runtime::Tensor::allocate_cpu({1, 1});
    nonfinite.data<float>()[0] = std::numeric_limits<float>::infinity();
    require_rejected([&] { (void)model::quantize_per_output_channel(nonfinite); }, "infinite weight accepted");
    auto vector = runtime::Tensor::allocate_cpu({4});
    require_rejected([&] { (void)model::quantize_per_output_channel(vector); }, "non-matrix weight accepted");

    auto values = runtime::Tensor::allocate_cpu({2, 2}, runtime::DType::INT8);
    auto scales = runtime::Tensor::allocate_cpu({2});
    scales.data<float>()[0] = 1.F;
    scales.data<float>()[1] = 0.F;
    require_rejected([&] { model::validate_per_output_channel_int8({values, scales, 1}); }, "zero scale accepted");
    scales.data<float>()[1] = 1.F;
    require_rejected([&] { model::validate_per_output_channel_int8({values, scales, 0}); }, "wrong output axis accepted");

    require(model::is_weight_only_int8_eligible({"layers.0.q_proj", {64, 64}}), "q_proj must be eligible");
    require(model::is_weight_only_int8_eligible({"lm_head", {64, 258}}), "lm_head must be eligible");
    require(!model::is_weight_only_int8_eligible({"token_embedding", {258, 64}}), "embedding must stay FP32");
    require(!model::is_weight_only_int8_eligible({"layers.0.attn_norm", {64}}), "norm must stay FP32");
}

} // namespace

int main() {
    try {
        test_per_channel_rounding_and_error();
        test_validation_and_eligibility();
        std::cout << "Stage 16 C1 per-output-channel INT8 quantization: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_quantization: " << error.what() << '\n';
        return 1;
    }
}
