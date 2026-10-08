#include "model/model_file.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
std::map<std::string, runtime::Tensor> load_legacy_weights(const char* path, const model::DecoderConfig& config) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open legacy FP32 weights");
    std::map<std::string, runtime::Tensor> tensors;
    for (const auto& spec : model::parameter_specs(config)) {
        auto tensor = runtime::Tensor::allocate_cpu(spec.shape);
        input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
        if (!input) throw std::runtime_error("truncated legacy FP32 weights");
        tensors.emplace(spec.name, std::move(tensor));
    }
    if (input.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("legacy FP32 weights have trailing bytes");
    return tensors;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: convert_tiny_model <legacy-weights.bin> <model.mllm>");
        const auto config = model::DecoderConfig::tiny();
        model::ParameterTable parameters(config, load_legacy_weights(argv[1], config));
        model::write_model_file(argv[2], parameters);
        const auto metadata = model::inspect_model_file(argv[2]);
        std::cout << "wrote " << argv[2] << " tensors=" << metadata.tensors.size()
                  << " payload_bytes=" << metadata.payload_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "convert_tiny_model: " << error.what() << '\n';
        return 1;
    }
}
