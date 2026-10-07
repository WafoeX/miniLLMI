#include "fixture_reader.hpp"
#include "runtime/cpu_backend.hpp"
#include "runtime/reference.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace runtime;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void compare(const fixtures::Case& fixture, const Tensor& actual) {
    require(actual.shape() == fixture.expected.shape() && actual.dtype() == DType::FP32, "fixture output metadata");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const auto value = actual.data<float>()[index];
        const auto expected = fixture.expected.data<float>()[index];
        require(std::isfinite(value) && std::abs(static_cast<double>(value) - expected) <=
                    fixture.atol + fixture.rtol * std::abs(expected),
                "transformer fixture tolerance mismatch");
    }
}

bool implemented(OpCode code) {
    return code == OpCode::RMSNORM || code == OpCode::SOFTMAX || code == OpCode::ROPE || code == OpCode::EMBEDDING || code == OpCode::SWIGLU;
}

void fixture_conformance(const std::vector<fixtures::Case>& cases) {
    std::size_t covered = 0;
    for (const auto& fixture : cases) {
        if (!implemented(fixture.descriptor.code())) continue;
        TensorInputs inputs;
        for (const auto& input : fixture.inputs) inputs.emplace_back(input);
        for (const auto mode : {CpuMatmul::ReferenceFP64, CpuMatmul::ScalarFP32V0}) {
            CpuBackend backend(mode);
            auto output = Tensor::allocate_cpu(fixture.expected.shape());
            const auto before = testing::cpu_allocation_counts();
            const auto preparation = backend.prepare(fixture.descriptor, inputs, output);
            require(preparation.ok() && preparation.workspace_bytes == 0, "C1 prepare has no workspace");
            const auto status = backend.execute(fixture.descriptor, inputs, output);
            require(status.ok(), "C1 fixture execution");
            require(testing::cpu_allocation_counts().allocations == before.allocations,
                    "C1 execute cannot allocate backing storage");
            compare(fixture, output);
        }
        ++covered;
    }
    require(covered == 8, "C1-C3 fixtures cover all primitive reference vectors");
}

void softmax_mask_and_errors() {
    auto scores = Tensor::allocate_cpu({1, 2});
    auto output = Tensor::allocate_cpu({1, 2});
    scores.data<float>()[0] = 0;
    scores.data<float>()[1] = std::numeric_limits<float>::quiet_NaN();
    const auto masked = OpDesc(OpCode::SOFTMAX, {0}, {1}, SoftmaxAttrs{true, 0, 0, 1088});
    require(CpuBackend().execute(masked, {scores}, output).ok(), "masked nonfinite score is ignored");
    require(output.data<float>()[0] == 1 && output.data<float>()[1] == 0, "causal mask output is exact");
    scores.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    scores.data<float>()[1] = 0;
    output.data<float>()[0] = output.data<float>()[1] = 46;
    const auto status = CpuBackend().execute(masked, {scores}, output);
    require(status.code == StatusCode::NonFinite, "unmasked nonfinite score is rejected");
    require(output.data<float>()[0] == 46 && output.data<float>()[1] == 46, "SOFTMAX preflight leaves output unchanged");
}

void embedding_errors_are_prewrite() {
    auto ids = Tensor::allocate_cpu({2}, DType::INT32);
    auto table = Tensor::allocate_cpu({2, 2});
    auto output = Tensor::allocate_cpu({2, 2});
    ids.data<std::int32_t>()[0] = 0; ids.data<std::int32_t>()[1] = 2;
    for (std::size_t index = 0; index < table.numel(); ++index) table.data<float>()[index] = static_cast<float>(index);
    for (std::size_t index = 0; index < output.numel(); ++index) output.data<float>()[index] = 45;
    const auto status = CpuBackend().execute(OpDesc(OpCode::EMBEDDING, {0, 1}, {2}), {ids, table}, output);
    require(status.code == StatusCode::OutOfRange, "EMBEDDING rejects an invalid ID");
    for (std::size_t index = 0; index < output.numel(); ++index) require(output.data<float>()[index] == 45, "EMBEDDING preflight leaves output unchanged");
}

void errors_are_prewrite() {
    auto x = Tensor::allocate_cpu({1, 2});
    auto scale = Tensor::allocate_cpu({2});
    auto output = Tensor::allocate_cpu({1, 2});
    x.data<float>()[0] = std::numeric_limits<float>::infinity();
    x.data<float>()[1] = 1;
    scale.data<float>()[0] = scale.data<float>()[1] = 1;
    output.data<float>()[0] = output.data<float>()[1] = 47;
    const auto desc = OpDesc(OpCode::RMSNORM, {0, 1}, {2}, NormAttrs{1e-5});
    const auto status = CpuBackend().execute(desc, {x, scale}, output);
    require(status.code == StatusCode::NonFinite, "RMSNorm rejects nonfinite input");
    require(output.data<float>()[0] == 47 && output.data<float>()[1] == 47, "RMSNorm preflight leaves output unchanged");

    auto gate = Tensor::allocate_cpu({2});
    auto up = Tensor::allocate_cpu({2});
    gate.data<float>()[0] = 1;
    gate.data<float>()[1] = -1;
    up.data<float>()[0] = 1;
    up.data<float>()[1] = std::numeric_limits<float>::quiet_NaN();
    auto swiglu_output = Tensor::allocate_cpu({2});
    swiglu_output.data<float>()[0] = swiglu_output.data<float>()[1] = 48;
    const auto swiglu = OpDesc(OpCode::SWIGLU, {0, 1}, {2});
    const auto swiglu_status = reference::execute(swiglu, {gate, up}, swiglu_output);
    require(swiglu_status.code == StatusCode::NonFinite, "SwiGLU rejects nonfinite input");
    require(swiglu_output.data<float>()[0] == 48 && swiglu_output.data<float>()[1] == 48, "SwiGLU preflight leaves output unchanged");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected verified fixture file");
        std::ifstream stream(argv[1]);
        if (!stream) throw std::runtime_error("fixture file could not be opened");
        fixture_conformance(fixtures::load(stream));
        softmax_mask_and_errors();
        embedding_errors_are_prewrite();
        errors_are_prewrite();
        require(testing::cpu_allocation_counts().live == 0, "transformer C1 test leaked backing storage");
        std::cout << "Transformer C1-C3 RMSNorm/SwiGLU/Softmax/RoPE/Embedding: PASS fixture/masks/prewrite/no-allocation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_transformer_ops: " << error.what() << '\n';
        return 1;
    }
}
