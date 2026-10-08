#pragma once

#include "model/config.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace decoder_fixture {
inline void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

inline std::map<std::string, runtime::Tensor> load_parameters(const char* path,
                                                              const model::DecoderConfig& config) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open tiny weights");
    std::map<std::string, runtime::Tensor> tensors;
    for (const auto& spec : model::parameter_specs(config)) {
        auto tensor = runtime::Tensor::allocate_cpu(spec.shape);
        input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
        if (!input) throw std::runtime_error("truncated tiny weights");
        tensors.emplace(spec.name, std::move(tensor));
    }
    require(input.peek() == std::ifstream::traits_type::eof(), "trailing tiny weights");
    return tensors;
}

inline runtime::Tensor load_tensor(const char* path, runtime::Shape shape) {
    auto tensor = runtime::Tensor::allocate_cpu(std::move(shape));
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open decoder fixture");
    input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
    require(static_cast<bool>(input) && input.peek() == std::ifstream::traits_type::eof(), "invalid decoder fixture size");
    return tensor;
}

inline runtime::Tensor token_tensor(const std::vector<std::int32_t>& values) {
    auto ids = runtime::Tensor::allocate_cpu({static_cast<std::int64_t>(values.size())}, runtime::DType::INT32);
    for (std::size_t index = 0; index < values.size(); ++index) ids.data<std::int32_t>()[index] = values[index];
    return ids;
}

inline void compare(const runtime::Tensor& actual, const runtime::Tensor& expected,
                    double atol = 2e-6, double rtol = 2e-5) {
    require(actual.shape() == expected.shape(), "decoder fixture output shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto value = actual.data<float>()[index];
        const auto reference = expected.data<float>()[index];
        require(std::isfinite(value) && std::abs(static_cast<double>(value) - reference) <=
                    atol + rtol * std::abs(reference),
                "decoder output differs from independent fixture");
    }
}

inline void require_finite(const runtime::Tensor& tensor) {
    for (std::size_t index = 0; index < tensor.numel(); ++index)
        require(std::isfinite(tensor.data<float>()[index]), "decoder output is nonfinite");
}
} // namespace decoder_fixture
