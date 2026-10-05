#pragma once

#include "runtime/operator.hpp"
#include "runtime/tensor.hpp"
#include <cmath>
#include <istream>
#include <limits>
#include <locale>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fixtures {
struct Case {
    std::string name;
    runtime::OpDesc descriptor;
    std::vector<runtime::Tensor> inputs;
    runtime::Tensor expected;
    double atol;
    double rtol;
};
template<class T> T read(std::istream& input) {
    T result;
    if (!(input >> result)) throw std::runtime_error("fixture token missing or invalid");
    return result;
}
inline void token(std::istream& input, const char* expected) {
    if (read<std::string>(input) != expected) throw std::runtime_error(std::string("fixture requires token ") + expected);
}
inline bool boolean(std::istream& input) {
    const auto value = read<int>(input);
    if (value != 0 && value != 1) throw std::runtime_error("fixture boolean must be 0/1");
    return value == 1;
}
inline runtime::OpCode opcode(const std::string& name) {
    using runtime::OpCode;
    for (auto code : {OpCode::ADD, OpCode::MUL, OpCode::MATMUL, OpCode::COPY, OpCode::MATERIALIZE,
                     OpCode::RMSNORM, OpCode::SOFTMAX, OpCode::ROPE, OpCode::EMBEDDING, OpCode::SWIGLU, OpCode::ATTENTION})
        if (name == runtime::op_name(code)) return code;
    throw std::runtime_error("fixture unknown operator");
}
inline runtime::OpAttrs attributes(std::istream& input, runtime::OpCode code) {
    using namespace runtime;
    token(input, "ATTR");
    switch (code) {
    case OpCode::RMSNORM: return NormAttrs{read<double>(input)};
    case OpCode::SOFTMAX: {
        const auto causal = boolean(input);
        const auto query = read<std::int64_t>(input), key = read<std::int64_t>(input), maximum = read<std::int64_t>(input);
        return SoftmaxAttrs{causal, query, key, maximum};
    }
    case OpCode::ROPE: {
        const auto position = read<std::int64_t>(input);
        const auto base = read<double>(input);
        return RopeAttrs{position, base, read<std::int64_t>(input)};
    }
    case OpCode::ATTENTION: {
        const auto heads = read<std::int64_t>(input), dim = read<std::int64_t>(input);
        const auto causal = boolean(input);
        const auto query = read<std::int64_t>(input), key = read<std::int64_t>(input), maximum = read<std::int64_t>(input);
        return AttentionAttrs{heads, dim, causal, query, key, maximum};
    }
    default:
        token(input, "NONE");
        if (code == OpCode::COPY) return CopyAttrs{};
        return {};
    }
}
inline runtime::Tensor tensor(std::istream& input, const char* role) {
    using namespace runtime;
    token(input, role);
    const auto dtype_name = read<std::string>(input);
    if (dtype_name != "FP32" && dtype_name != "INT32") throw std::runtime_error("fixture dtype must be FP32/INT32");
    const auto dtype = dtype_name == "FP32" ? DType::FP32 : DType::INT32;
    const auto rank = read<std::size_t>(input);
    if (rank > MAX_RANK) throw std::runtime_error("fixture rank exceeds limit");
    std::vector<std::int64_t> dims(rank);
    for (auto& dim : dims) dim = read<std::int64_t>(input);
    Shape shape(std::move(dims));
    const auto size = numel(shape);
    if (size > 1000000) throw std::runtime_error("fixture exceeds compact payload limit");
    token(input, "DATA");
    if (read<std::size_t>(input) != size) throw std::runtime_error("fixture data count differs from shape");
    auto result = Tensor::allocate_cpu(shape, dtype);
    for (std::size_t i = 0; i < size; ++i) {
        if (dtype == DType::FP32) {
            const auto value = read<float>(input);
            if (!std::isfinite(value)) throw std::runtime_error("fixture requires finite FP32 values");
            result.data<float>()[i] = value;
        } else {
            const auto value = read<std::int64_t>(input);
            if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max())
                throw std::runtime_error("fixture INT32 overflow");
            result.data<std::int32_t>()[i] = static_cast<std::int32_t>(value);
        }
    }
    token(input, "LAYOUT");
    const auto layout = read<std::string>(input);
    if (layout == "TRANSPOSE") {
        const auto first = read<std::size_t>(input), second = read<std::size_t>(input);
        return result.transpose(first, second);
    }
    if (layout != "DENSE") throw std::runtime_error("fixture unknown layout");
    return result;
}
inline std::vector<Case> load(std::istream& input) {
    input.imbue(std::locale::classic());
    token(input, "MLRT_OP_FIXTURES");
    if (read<unsigned>(input) != runtime::OP_SEMANTICS_VERSION) throw std::runtime_error("fixture unsupported semantics version");
    std::vector<Case> result;
    std::set<std::string> names;
    for (;;) {
        const auto marker = read<std::string>(input);
        if (marker == "END_FIXTURES") break;
        if (marker != "CASE" || result.size() >= 1024) throw std::runtime_error("fixture invalid CASE boundary/count");
        auto name = read<std::string>(input);
        if (!names.insert(name).second) throw std::runtime_error("fixture duplicate case name");
        const auto code = opcode(read<std::string>(input));
        const auto arity = read<std::size_t>(input);
        const auto atol = read<double>(input), rtol = read<double>(input);
        if (arity == 0 || arity > 3 || !std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
            throw std::runtime_error("fixture invalid arity/tolerance");
        auto attrs = attributes(input, code);
        std::vector<runtime::TensorId> ids;
        std::vector<runtime::Tensor> inputs;
        for (std::size_t i = 0; i < arity; ++i) { ids.push_back(static_cast<runtime::TensorId>(i)); inputs.push_back(tensor(input, "INPUT")); }
        auto expected = tensor(input, "OUTPUT");
        token(input, "END");
        result.push_back({std::move(name), runtime::OpDesc(code, std::move(ids), {static_cast<runtime::TensorId>(arity)}, std::move(attrs)),
                          std::move(inputs), std::move(expected), atol, rtol});
    }
    std::string trailing;
    if (result.empty() || input >> trailing) throw std::runtime_error("fixture empty suite or trailing tokens");
    return result;
}
} // namespace fixtures
