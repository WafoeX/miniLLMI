#include "transformer_ops.hpp"

#include <algorithm>
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

Status softmax(const TensorInputs& inputs, Tensor& output, const SoftmaxAttrs& attrs) {
    const auto queries = as_size(inputs[0].get().shape()[0]);
    const auto keys = as_size(inputs[0].get().shape()[1]);
    const auto* scores = inputs[0].get().data<float>();
    // Validate only unmasked entries before writing any row. A masked score is
    // semantically ignored, including a NaN/Inf payload in its storage slot.
    for (std::size_t query = 0; query < queries; ++query) {
        const auto absolute_query = attrs.query_position + static_cast<std::int64_t>(query);
        for (std::size_t key = 0; key < keys; ++key) {
            const auto visible = !attrs.causal || attrs.key_position + static_cast<std::int64_t>(key) <= absolute_query;
            if (visible && !std::isfinite(scores[query * keys + key]))
                return nonfinite("SOFTMAX unmasked score contains NaN/Inf");
        }
    }

    auto* result = output.data<float>();
    for (std::size_t query = 0; query < queries; ++query) {
        const auto absolute_query = attrs.query_position + static_cast<std::int64_t>(query);
        bool any_visible = false;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t key = 0; key < keys; ++key) {
            if (attrs.causal && attrs.key_position + static_cast<std::int64_t>(key) > absolute_query) continue;
            any_visible = true;
            maximum = std::max(maximum, scores[query * keys + key]);
        }
        if (!any_visible) {
            for (std::size_t key = 0; key < keys; ++key) result[query * keys + key] = 0;
            continue;
        }
        double sum = 0;
        for (std::size_t key = 0; key < keys; ++key) {
            if (attrs.causal && attrs.key_position + static_cast<std::int64_t>(key) > absolute_query) continue;
            sum += std::exp(static_cast<double>(scores[query * keys + key] - maximum));
        }
        if (!std::isfinite(sum) || sum <= 0) return nonfinite("SOFTMAX normalization is nonfinite");
        for (std::size_t key = 0; key < keys; ++key) {
            if (attrs.causal && attrs.key_position + static_cast<std::int64_t>(key) > absolute_query) {
                result[query * keys + key] = 0;
                continue;
            }
            const auto value = static_cast<float>(std::exp(static_cast<double>(scores[query * keys + key] - maximum)) / sum);
            if (!std::isfinite(value)) return nonfinite("SOFTMAX result is nonfinite");
            result[query * keys + key] = value;
        }
    }
    return Status::success();
}

Status rope(const TensorInputs& inputs, Tensor& output, const RopeAttrs& attrs) {
    const auto finite = finite_fp32(inputs[0], "ROPE input contains NaN/Inf");
    if (!finite.ok()) return finite;
    const auto& shape = inputs[0].get().shape();
    const auto tokens = as_size(shape[0]);
    const auto dim = as_size(shape[shape.rank() - 1]);
    const auto heads = shape.rank() == 3 ? as_size(shape[1]) : 1U;
    const auto* source = inputs[0].get().data<float>();
    auto* result = output.data<float>();
    for (std::size_t token = 0; token < tokens; ++token) {
        const auto position = static_cast<double>(attrs.position + static_cast<std::int64_t>(token));
        for (std::size_t head = 0; head < heads; ++head) for (std::size_t pair = 0; pair < dim / 2; ++pair) {
            const auto theta = position / std::pow(attrs.base, 2.0 * static_cast<double>(pair) / static_cast<double>(dim));
            const auto base = (token * heads + head) * dim + 2 * pair;
            const auto even = source[base], odd = source[base + 1];
            const auto rotated_even = static_cast<float>(static_cast<double>(even) * std::cos(theta) - static_cast<double>(odd) * std::sin(theta));
            const auto rotated_odd = static_cast<float>(static_cast<double>(even) * std::sin(theta) + static_cast<double>(odd) * std::cos(theta));
            if (!std::isfinite(rotated_even) || !std::isfinite(rotated_odd)) return nonfinite("ROPE result is nonfinite");
            result[base] = rotated_even; result[base + 1] = rotated_odd;
        }
    }
    return Status::success();
}

Status embedding(const TensorInputs& inputs, Tensor& output) {
    const auto* ids = inputs[0].get().data<std::int32_t>();
    const auto* table = inputs[1].get().data<float>();
    const auto tokens = inputs[0].get().numel();
    const auto vocab = as_size(inputs[1].get().shape()[0]);
    const auto hidden = as_size(inputs[1].get().shape()[1]);
    for (std::size_t index = 0; index < tokens; ++index)
        if (ids[index] < 0 || static_cast<std::size_t>(ids[index]) >= vocab)
            return Status::failure(StatusCode::OutOfRange, "EMBEDDING token ID is outside [0,vocab)");
    const auto finite = finite_fp32(inputs[1], "EMBEDDING table contains NaN/Inf");
    if (!finite.ok()) return finite;
    auto* result = output.data<float>();
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t channel = 0; channel < hidden; ++channel)
            result[token * hidden + channel] = table[static_cast<std::size_t>(ids[token]) * hidden + channel];
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
        case OpCode::SOFTMAX:
            return softmax(inputs, output, std::get<SoftmaxAttrs>(descriptor.attrs()));
        case OpCode::ROPE:
            return rope(inputs, output, std::get<RopeAttrs>(descriptor.attrs()));
        case OpCode::EMBEDDING:
            return embedding(inputs, output);
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
