#include "fixture_reader.hpp"
#include "runtime/reference.hpp"
#ifdef CPU_BACKEND_FIXTURES
#include "runtime/cpu_backend.hpp"
#endif
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void malformed_reader_tests() {
    for (const std::string& text : {
        std::string("MLRT_OP_FIXTURES 2 END_FIXTURES"),
        std::string("MLRT_OP_FIXTURES 1 END_FIXTURES extra"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad UNKNOWN 1 0 0"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad ADD 2 -1 0"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad ADD 2 0 0 ATTR NONE INPUT FP32 9"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad ADD 2 0 0 ATTR NONE INPUT FP32 1 -1"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad ADD 2 0 0 ATTR NONE INPUT INT32 1 1 DATA 1 2147483648"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad ADD 2 0 0 ATTR NONE INPUT FP32 1 1 DATA 2"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad ADD 2 0 0 ATTR NONE INPUT FP64 1 1"),
        std::string("MLRT_OP_FIXTURES 1 CASE bad SOFTMAX 1 0 0 ATTR 2 0 0 1088")}) {
        std::istringstream input(text);
        bool rejected = false;
        try { (void)fixtures::load(input); } catch (const std::exception&) { rejected = true; }
        require(rejected, "malformed fixture accepted");
    }
}
bool core(OpCode code) {
    return code == OpCode::ADD || code == OpCode::MUL || code == OpCode::MATMUL || code == OpCode::COPY || code == OpCode::MATERIALIZE;
}
void compare(const fixtures::Case& fixture, const Tensor& actual) {
    require(actual.is_contiguous() && fixture.expected.is_contiguous(), "fixture comparison requires declared contiguous output");
    double maximum = 0;
    for (std::size_t i = 0; i < actual.numel(); ++i) {
        if (actual.dtype() == DType::INT32) require(actual.data<std::int32_t>()[i] == fixture.expected.data<std::int32_t>()[i], "INT32 fixture exact mismatch");
        else {
            const auto value = actual.data<float>()[i], expected = fixture.expected.data<float>()[i];
            const double error = std::abs(static_cast<double>(value) - expected);
            maximum = std::max(maximum, error);
            require(std::isfinite(value) && error <= fixture.atol + fixture.rtol * std::abs(expected), "FP32 fixture tolerance mismatch");
        }
    }
    std::cout << "fixture " << fixture.name << " core=PASS max_abs=" << maximum << " atol=" << fixture.atol << " rtol=" << fixture.rtol << '\n';
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: test_operator_fixtures <verified operators-v1.txt>");
        malformed_reader_tests();
        std::size_t numeric = 0, metadata = 0;
        {
            std::ifstream input(argv[1]);
            if (!input) throw std::runtime_error("fixture file could not be opened");
            const auto cases = fixtures::load(input);
            for (const auto& fixture : cases) {
                TensorInputs inputs;
                for (const auto& tensor : fixture.inputs) inputs.push_back(std::cref(tensor));
                auto inferred = infer_operator(fixture.descriptor, inputs);
                if (!inferred.ok()) throw std::runtime_error(fixture.name + ": " + inferred.status.message);
                require(inferred.output->shape == fixture.expected.shape() && inferred.output->dtype == fixture.expected.dtype(), "fixture inferred dtype/shape mismatch");
                auto output = inferred.output->alias ? *inferred.output->alias : Tensor::allocate_cpu(inferred.output->shape, inferred.output->dtype);
                const auto before = testing::cpu_allocation_counts();
#ifdef CPU_BACKEND_FIXTURES
                const auto status = CpuBackend(CpuMatmul::ScalarFP32V0).execute(fixture.descriptor, inputs, output);
#else
                const auto status = reference::execute(fixture.descriptor, inputs, output);
#endif
                require(testing::cpu_allocation_counts().allocations == before.allocations, "fixture reference kernel allocated buffers");
                if (core(fixture.descriptor.code())) {
                    if (!status.ok()) throw std::runtime_error(fixture.name + ": " + status.message);
                    compare(fixture, output); ++numeric;
                } else {
                    require(status.code == StatusCode::Unsupported, "S2 must not silently execute Transformer kernels");
                    std::cout << "fixture " << fixture.name << " metadata=PASS numeric=pending-S12\n"; ++metadata;
                }
                std::cout << "descriptor " << fixture.descriptor.serialize() << '\n';
            }
        }
        require(testing::cpu_allocation_counts().live == 0, "fixture parser/harness leaked buffers");
        std::cout << "Operator fixtures: PASS numeric_core=" << numeric << " transformer_metadata=" << metadata << " malformed_reader=10\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_operator_fixtures: " << error.what() << '\n'; return 1;
    }
}
