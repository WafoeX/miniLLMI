#include "transformer_ops.hpp"

#include <cmath>
#include <limits>

namespace runtime::detail {
namespace {
Status nonfinite(const char* message) {
    return Status::failure(StatusCode::NonFinite, message);
}

Status finite_fp32(const Tensor& tensor, const char* message) {
    const auto* data = tensor.data<float>();
    for (std::size_t index = 0; index < tensor.numel(); ++index)
        if (!std::isfinite(data[index])) return nonfinite(message);
    return Status::success();
}

Status rmsnorm(const TensorInputs& inputs, Tensor& output, const NormAttrs& attrs) {
    const auto finite_x = finite_fp32(inputs[0], "RMSNORM input contains NaN/Inf");
    if (!finite_x.ok()) return finite_x;
    const auto finite_scale = finite_fp32(inputs[1], "RMSNORM scale contains NaN/Inf");
    if (!finite_scale.ok()) return finite_scale;

    const auto channels = as_size(inputs[0].get().shape()[inputs[0].get().shape().rank() - 1]);
    const auto rows = output.numel() / channels;
    const auto* x = inputs[0].get().data<float>();
    const auto* scale = inputs[1].get().data<float>();
    auto* result = output.data<float>();
    for (std::size_t row = 0; row < rows; ++row) {
        double sum_squares = 0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto value = static_cast<double>(x[row * channels + channel]);
            sum_squares += value * value;
        }
        const auto inv_rms = 1.0 / std::sqrt(sum_squares / static_cast<double>(channels) + attrs.epsilon);
        if (!std::isfinite(inv_rms)) return nonfinite("RMSNORM reduction is nonfinite");
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto value = static_cast<float>(static_cast<double>(x[row * channels + channel]) * inv_rms * scale[channel]);
            if (!std::isfinite(value)) return nonfinite("RMSNORM result is nonfinite");
            result[row * channels + channel] = value;
        }
    }
    return Status::success();
}

Status swiglu(const TensorInputs& inputs, Tensor& output) {
    const auto finite_gate = finite_fp32(inputs[0], "SWIGLU gate contains NaN/Inf");
    if (!finite_gate.ok()) return finite_gate;
    const auto finite_up = finite_fp32(inputs[1], "SWIGLU up contains NaN/Inf");
    if (!finite_up.ok()) return finite_up;

    const auto* gate = inputs[0].get().data<float>();
    const auto* up = inputs[1].get().data<float>();
    auto* result = output.data<float>();
    for (std::size_t index = 0; index < output.numel(); ++index) {
        const auto value = static_cast<double>(gate[index]);
        const auto sigmoid = value >= 0 ? 1.0 / (1.0 + std::exp(-value))
                                        : std::exp(value) / (1.0 + std::exp(value));
        const auto transformed = static_cast<float>(value * sigmoid * static_cast<double>(up[index]));
        if (!std::isfinite(transformed)) return nonfinite("SWIGLU result is nonfinite");
        result[index] = transformed;
    }
    return Status::success();
}
} // namespace

Status execute_transformer_cpu(const OpDesc& descriptor, const TensorInputs& inputs, Tensor& output) {
    try {
        switch (descriptor.code()) {
        case OpCode::RMSNORM:
            return rmsnorm(inputs, output, std::get<NormAttrs>(descriptor.attrs()));
        case OpCode::SWIGLU:
            return swiglu(inputs, output);
        default:
            return Status::failure(StatusCode::Unsupported, "CPU transformer primitive is not implemented");
        }
    } catch (const std::overflow_error& error) {
        return Status::failure(StatusCode::Overflow, error.what());
    } catch (const std::invalid_argument& error) {
        return Status::failure(StatusCode::InvalidArgument, error.what());
    }
}
} // namespace runtime::detail
