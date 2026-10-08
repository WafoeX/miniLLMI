#include "model/config.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
using namespace runtime;
using namespace model;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

template<class F> void rejects(F&& function, const char* message) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

std::map<std::string, Tensor> load_fixture(const char* path, const DecoderConfig& config) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open tiny weight fixture");
    std::map<std::string, Tensor> tensors;
    for (const auto& spec : parameter_specs(config)) {
        auto tensor = Tensor::allocate_cpu(spec.shape);
        input.read(reinterpret_cast<char*>(tensor.data<float>()), static_cast<std::streamsize>(tensor.nbytes()));
        if (!input) throw std::runtime_error("truncated tiny weight fixture");
        tensors.emplace(spec.name, std::move(tensor));
    }
    require(input.peek() == std::ifstream::traits_type::eof(), "tiny weight fixture has trailing bytes");
    return tensors;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected tiny-weights-v1.bin");
        const auto config = DecoderConfig::tiny();
        require(config.validate().ok(), "frozen tiny config validates");
        require(parameter_specs(config).size() == 21, "frozen parameter count");
        require(parameter_bytes(config) == 461056, "frozen parameter byte count");
        auto tensors = load_fixture(argv[1], config);
        ParameterTable table(config, tensors);
        require(table.size() == 21 && table.at("lm_head").shape() == Shape({64, 258}), "fixture parameter binding");
        try { (void)table.at("unknown"); throw std::runtime_error("unknown lookup accepted"); }
        catch (const std::out_of_range&) {}

        auto bad = config; bad.batch = 2; require(!bad.validate().ok(), "batch>1 rejected");
        bad = config; bad.kv_heads = 2; require(!bad.validate().ok(), "GQA rejected");
        bad = config; bad.hidden = 63; require(!bad.validate().ok(), "head geometry rejected");
        bad = config; bad.head_dim = 15; bad.hidden = 60; require(!bad.validate().ok(), "odd rotary dimension rejected");
        bad = config; bad.bias = true; require(!bad.validate().ok(), "bias rejected");

        auto missing = tensors; missing.erase("final_norm");
        rejects([&] { ParameterTable ignored(config, missing); }, "missing name accepted");
        auto extra = tensors; extra.emplace("unexpected", Tensor::allocate_cpu({1}));
        rejects([&] { ParameterTable ignored(config, extra); }, "extra name accepted");
        auto wrong_shape = tensors; wrong_shape.at("lm_head") = Tensor::allocate_cpu({64, 257});
        rejects([&] { ParameterTable ignored(config, wrong_shape); }, "wrong shape accepted");
        auto wrong_dtype = tensors; wrong_dtype.at("final_norm") = Tensor::allocate_cpu({64}, DType::INT32);
        rejects([&] { ParameterTable ignored(config, wrong_dtype); }, "wrong dtype accepted");
        std::cout << "Stage 13 C1 decoder config/parameter binding: PASS tensors=" << table.size()
                  << " bytes=" << parameter_bytes(config) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_decoder_config: " << error.what() << '\n';
        return 1;
    }
}
