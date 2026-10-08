#include "model/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace model {
namespace {

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(std::string("INT8 quantization: ") + message);
}

std::int8_t round_to_even_int8(float value) {
    if (!std::isfinite(value)) invalid("non-finite normalized value");
    const auto lower = std::floor(static_cast<double>(value));
    const auto fraction = static_cast<double>(value) - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(std::fabs(lower), 2.0) != 0.0))
        rounded = lower + 1.0;
    rounded = std::max(-127.0, std::min(127.0, rounded));
    return static_cast<std::int8_t>(rounded);
}

bool has_projection_suffix(const std::string& name) {
    constexpr const char suffix[] = "_proj";
    return name.size() >= sizeof(suffix) - 1 &&
           name.compare(name.size() - (sizeof(suffix) - 1), sizeof(suffix) - 1, suffix) == 0;
}

void validate_weight(const runtime::Tensor& weight) {
    if (weight.dtype() != runtime::DType::FP32 || weight.device() != runtime::Device{} ||
        weight.shape().rank() != 2 || !weight.is_contiguous())
        invalid("weight must be contiguous CPU FP32 W[in,out]");
    if (weight.shape()[0] <= 0 || weight.shape()[1] <= 0) invalid("weight dimensions must be positive");
}

} // namespace

bool is_weight_only_int8_eligible(const ParameterSpec& spec) {
    return spec.shape.rank() == 2 && (spec.name == "lm_head" || has_projection_suffix(spec.name));
}

void validate_per_output_channel_int8(const QuantizedTensor& quantized) {
    const auto& values = quantized.values;
    const auto& scales = quantized.scales;
    if (quantized.output_axis != 1) invalid("only output axis 1 is supported");
    if (values.dtype() != runtime::DType::INT8 || values.device() != runtime::Device{} ||
        values.shape().rank() != 2 || !values.is_contiguous())
        invalid("values must be contiguous CPU INT8 W[in,out]");
    if (values.shape()[0] <= 0 || values.shape()[1] <= 0) invalid("value dimensions must be positive");
    if (scales.dtype() != runtime::DType::FP32 || scales.device() != runtime::Device{} ||
        scales.shape() != runtime::Shape({values.shape()[1]}) || !scales.is_contiguous())
        invalid("scales must be contiguous CPU FP32 [out]");
    for (std::size_t output = 0; output < scales.numel(); ++output) {
        const auto scale = scales.data<float>()[output];
        if (!std::isfinite(scale) || scale <= 0.F) invalid("scales must be finite and positive");
    }
}

QuantizedTensor quantize_per_output_channel(const runtime::Tensor& weight) {
    validate_weight(weight);
    const auto input = weight.data<float>();
    const auto in = static_cast<std::size_t>(weight.shape()[0]);
    const auto out = static_cast<std::size_t>(weight.shape()[1]);
    auto values = runtime::Tensor::allocate_cpu(weight.shape(), runtime::DType::INT8);
    auto scales = runtime::Tensor::allocate_cpu({weight.shape()[1]}, runtime::DType::FP32);
    auto* q = values.data<std::int8_t>();
    auto* scale_data = scales.data<float>();

    for (std::size_t output = 0; output < out; ++output) {
        float maximum = 0.F;
        for (std::size_t input_index = 0; input_index < in; ++input_index) {
            const auto value = input[input_index * out + output];
            if (!std::isfinite(value)) invalid("weight contains NaN or Inf");
            maximum = std::max(maximum, std::fabs(value));
        }
        const auto scale = maximum == 0.F ? 1.F : maximum / 127.F;
        if (!std::isfinite(scale) || scale <= 0.F) invalid("computed scale is not finite and positive");
        scale_data[output] = scale;
        for (std::size_t input_index = 0; input_index < in; ++input_index)
            q[input_index * out + output] = round_to_even_int8(input[input_index * out + output] / scale);
    }
    QuantizedTensor result{std::move(values), std::move(scales), 1};
    validate_per_output_channel_int8(result);
    return result;
}

runtime::Tensor dequantize_per_output_channel(const QuantizedTensor& quantized) {
    validate_per_output_channel_int8(quantized);
    const auto& values = quantized.values;
    const auto in = static_cast<std::size_t>(values.shape()[0]);
    const auto out = static_cast<std::size_t>(values.shape()[1]);
    auto result = runtime::Tensor::allocate_cpu(values.shape(), runtime::DType::FP32);
    const auto* q = values.data<std::int8_t>();
    const auto* scales = quantized.scales.data<float>();
    auto* output = result.data<float>();
    for (std::size_t input_index = 0; input_index < in; ++input_index)
        for (std::size_t output_index = 0; output_index < out; ++output_index)
            output[input_index * out + output_index] = static_cast<float>(q[input_index * out + output_index]) * scales[output_index];
    return result;
}

} // namespace model
