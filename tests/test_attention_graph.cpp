#include "fixture_reader.hpp"
#include "runtime/attention.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void compare(const Tensor& actual, const Tensor& expected, double atol, double rtol) {
    require(actual.shape() == expected.shape(), "attention output shape");
    for (std::size_t i = 0; i < actual.numel(); ++i)
        require(std::isfinite(actual.data<float>()[i]) && std::abs(static_cast<double>(actual.data<float>()[i]) - expected.data<float>()[i]) <= atol + rtol * std::abs(expected.data<float>()[i]), "attention fixture mismatch");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected verified fixture file");
        std::ifstream stream(argv[1]); const auto cases = fixtures::load(stream);
        const auto found = std::find_if(cases.begin(), cases.end(), [](const auto& item) { return item.descriptor.code() == OpCode::ATTENTION; });
        require(found != cases.end(), "attention fixture missing");
        const auto attrs = std::get<AttentionAttrs>(found->descriptor.attrs());
        std::vector<Tensor> scales;
        for (std::int64_t head = 0; head < attrs.heads; ++head) {
            auto scale = Tensor::allocate_cpu({found->inputs[0].shape()[0], found->inputs[1].shape()[0]});
            for (std::size_t i = 0; i < scale.numel(); ++i) scale.data<float>()[i] = 1.F / std::sqrt(static_cast<float>(attrs.head_dim));
            scales.push_back(std::move(scale));
        }
        auto output = Tensor::allocate_cpu(found->expected.shape());
        const auto before = testing::cpu_allocation_counts();
        const auto result = execute_causal_attention(found->inputs[0], found->inputs[1], found->inputs[2], scales, output, attrs);
        require(result.ok(), result.status.message.c_str());
        require(result.counts.nodes_completed == static_cast<std::size_t>(attrs.heads) * 18, "all declared lowering nodes execute");
        require(result.counts.copies == static_cast<std::size_t>(attrs.heads) * 5, "materializations and checked head writes are explicit");
        compare(output, found->expected, found->atol, found->rtol);
        require(testing::cpu_allocation_counts().live == before.live, "attention graph releases intermediates and result aliases");
        std::cout << "Causal attention graph composition: PASS heads=" << attrs.heads << " nodes=" << result.counts.nodes_completed << " copies=" << result.counts.copies << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "test_attention_graph: " << error.what() << '\n'; return 1; }
}
