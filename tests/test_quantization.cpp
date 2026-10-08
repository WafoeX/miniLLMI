#include "model/quantization.hpp"
#include "runtime/graph_executor.hpp"

#include <algorithm>
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

runtime::Tensor matmul_with_existing_backend(const runtime::Tensor& input, const runtime::Tensor& weight) {
    runtime::Graph graph;
    graph.add_input(0, "input", input);
    graph.add_input(1, "weight", weight);
    graph.add_tensor(2, {input.shape()[0], weight.shape()[1]});
    graph.add_node(0, runtime::OpDesc(runtime::OpCode::MATMUL, {0, 1}, {2}));
    graph.add_output("output", 2);
    const auto frozen = graph.freeze();
    require(frozen.ok(), frozen.message.c_str());
    auto result = runtime::execute_graph(graph);
    require(result.ok(), result.status.message.c_str());
    return result.outputs.at("output");
}

void test_prepare_dequant_matmul_path() {
    auto input = runtime::Tensor::allocate_cpu({2, 3});
    const float input_values[] = {0.25F, -0.5F, 0.75F, -1.F, 0.5F, 0.125F};
    std::copy(std::begin(input_values), std::end(input_values), input.data<float>());
    auto weight = runtime::Tensor::allocate_cpu({3, 2});
    const float weight_values[] = {0.125F, -0.25F, 0.5F, 0.75F, -0.875F, 1.F};
    std::copy(std::begin(weight_values), std::end(weight_values), weight.data<float>());
    const auto expected = matmul_with_existing_backend(input, weight);
    const auto quantized = model::quantize_per_output_channel(weight);
    const auto actual = matmul_with_existing_backend(input, model::dequantize_per_output_channel(quantized));
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto tolerance = 1e-2F + 1e-2F * std::fabs(expected.data<float>()[index]);
        require(std::isfinite(actual.data<float>()[index]) &&
                    std::fabs(actual.data<float>()[index] - expected.data<float>()[index]) <= tolerance,
                "prepare_dequant matmul error exceeds frozen tolerance");
    }

    auto zero_weight = runtime::Tensor::allocate_cpu({3, 2});
    const auto zero_actual = matmul_with_existing_backend(input,
        model::dequantize_per_output_channel(model::quantize_per_output_channel(zero_weight)));
    for (std::size_t index = 0; index < zero_actual.numel(); ++index)
        require(zero_actual.data<float>()[index] == 0.F, "zero weight prepare_dequant matmul");
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
        test_prepare_dequant_matmul_path();
        test_validation_and_eligibility();
        std::cout << "Stage 16 C1 per-output-channel INT8 quantization: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_quantization: " << error.what() << '\n';
        return 1;
    }
}
