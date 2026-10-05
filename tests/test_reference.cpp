#include "runtime/reference.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
OpDesc op(OpCode code) {
    if (code == OpCode::MATERIALIZE) return OpDesc(code, {0}, {1});
    return OpDesc(code, {0, 1}, {2}, code == OpCode::COPY ? OpAttrs(CopyAttrs{}) : OpAttrs{});
}
void success(Status status) { if (!status.ok()) throw std::runtime_error(status.message); }
void expect(Status status, StatusCode code) { require(status.code == code && !status.message.empty(), "reference failure classification"); }
void hand_vectors() {
    auto a = Tensor::allocate_cpu({4}), b = Tensor::allocate_cpu({4}), out = Tensor::allocate_cpu({4});
    for (std::size_t i = 0; i < 4; ++i) { a.data<float>()[i] = static_cast<float>(i) - 2; b.data<float>()[i] = static_cast<float>(i) + 1; }
    const auto before = testing::cpu_allocation_counts();
    const auto storage = out.storage();
    success(reference::execute(op(OpCode::ADD), {a, b}, out));
    for (std::size_t i = 0; i < 4; ++i) require(out.data<float>()[i] == a.data<float>()[i] + b.data<float>()[i], "ADD hand vector");
    success(reference::execute(op(OpCode::MUL), {a, b}, out));
    for (std::size_t i = 0; i < 4; ++i) require(out.data<float>()[i] == a.data<float>()[i] * b.data<float>()[i], "MUL hand vector");
    require(out.storage() == storage && testing::cpu_allocation_counts().allocations == before.allocations, "kernel must not replace/allocate output");
    auto scalar = Tensor::allocate_cpu({}), scalar_out = Tensor::allocate_cpu({});
    *scalar.data<float>() = 3;
    success(reference::execute(op(OpCode::MUL), {scalar, scalar}, scalar_out));
    require(*scalar_out.data<float>() == 9, "scalar arithmetic");
    auto matrix_a = Tensor::allocate_cpu({2, 3}), matrix_b = Tensor::allocate_cpu({3, 4}), matrix_out = Tensor::allocate_cpu({2, 4});
    for (std::size_t i = 0; i < 6; ++i) matrix_a.data<float>()[i] = static_cast<float>(i + 1);
    for (std::size_t i = 0; i < 12; ++i) matrix_b.data<float>()[i] = static_cast<float>(i + 1);
    success(reference::execute(op(OpCode::MATMUL), {matrix_a, matrix_b}, matrix_out));
    const float expected[] = {38, 44, 50, 56, 83, 98, 113, 128};
    for (std::size_t i = 0; i < 8; ++i) require(matrix_out.data<float>()[i] == expected[i], "non-square MATMUL hand vector");
    auto zero_a = Tensor::allocate_cpu({2, 0}), zero_b = Tensor::allocate_cpu({0, 3}), zero_out = Tensor::allocate_cpu({2, 3});
    for (std::size_t i = 0; i < 6; ++i) zero_out.data<float>()[i] = 17;
    success(reference::execute(op(OpCode::MATMUL), {zero_a, zero_b}, zero_out));
    for (std::size_t i = 0; i < 6; ++i) require(zero_out.data<float>()[i] == 0, "K=0 must overwrite with zeros");
    auto empty = Tensor::allocate_cpu({0});
    success(reference::execute(op(OpCode::ADD), {empty, empty}, empty));
}
void copies_and_errors() {
    auto owner = Tensor::allocate_cpu({2, 3}, DType::INT32);
    for (std::size_t i = 0; i < owner.numel(); ++i) owner.data<std::int32_t>()[i] = static_cast<std::int32_t>(i);
    auto transposed = owner.transpose(0, 1), dense = Tensor::allocate_cpu({3, 2}, DType::INT32);
    const auto before = testing::cpu_allocation_counts();
    success(reference::execute(op(OpCode::MATERIALIZE), {transposed}, dense));
    for (std::int64_t i = 0; i < 3; ++i) for (std::int64_t j = 0; j < 2; ++j)
        require(dense.at<std::int32_t>({i, j}) == owner.at<std::int32_t>({j, i}), "reference reuses strided Stage 1 COPY");
    require(testing::cpu_allocation_counts().allocations == before.allocations, "explicit MATERIALIZE kernel cannot allocate buffers");
    expect(reference::execute(op(OpCode::MATERIALIZE), {dense}, dense), StatusCode::Aliasing);
    success(reference::execute(op(OpCode::COPY), {dense, dense}, dense));
    auto padded_owner = Tensor::allocate_cpu({3, 4}, DType::INT32);
    auto padded = padded_owner.slice(1, 0, 2, 2);
    success(reference::execute(op(OpCode::COPY), {dense, padded}, padded));
    require(padded.at<std::int32_t>({2, 1}) == dense.at<std::int32_t>({2, 1}) && padded_owner.at<std::int32_t>({2, 3}) == 0, "copy to explicit strided range preserves padding");
    auto redirected = Tensor::allocate_cpu({3, 2}, DType::INT32);
    expect(reference::execute(op(OpCode::COPY), {dense, padded}, redirected), StatusCode::Aliasing);
    auto a = Tensor::allocate_cpu({2, 2}), b = Tensor::allocate_cpu({2, 2}), out = Tensor::allocate_cpu({2, 2});
    expect(reference::execute(op(OpCode::ADD), {a, b}, a), StatusCode::Aliasing);
    expect(reference::execute(op(OpCode::MATMUL), {a, b}, b), StatusCode::Aliasing);
    auto strided = a.transpose(0, 1);
    expect(reference::execute(op(OpCode::ADD), {strided, b}, out), StatusCode::LayoutMismatch);
    auto wrong = Tensor::allocate_cpu({4});
    expect(reference::execute(op(OpCode::ADD), {a, b}, wrong), StatusCode::ShapeMismatch);
    for (std::size_t i = 0; i < 4; ++i) out.data<float>()[i] = 99;
    a.data<float>()[3] = std::numeric_limits<float>::quiet_NaN();
    expect(reference::execute(op(OpCode::ADD), {a, b}, out), StatusCode::NonFinite);
    for (std::size_t i = 0; i < 4; ++i) require(out.data<float>()[i] == 99, "nonfinite input must fail before any write");
    a.data<float>()[3] = std::numeric_limits<float>::infinity();
    expect(reference::execute(op(OpCode::MATMUL), {a, b}, out), StatusCode::NonFinite);
    success(reference::execute(op(OpCode::COPY), {a, out}, out));
    require(std::isinf(out.data<float>()[3]), "COPY is bitwise data movement, not arithmetic finite validation");
    a.data<float>()[0] = 1; a.data<float>()[1] = std::numeric_limits<float>::max(); a.data<float>()[2] = 0; a.data<float>()[3] = 0;
    for (std::size_t i = 0; i < 4; ++i) { b.data<float>()[i] = 2; out.data<float>()[i] = 99; }
    expect(reference::execute(op(OpCode::MUL), {a, b}, out), StatusCode::NonFinite);
    require(out.data<float>()[0] == 2 && out.data<float>()[1] == 99, "result overflow partial-write policy");
    auto simulated = Storage::wrap(Device(DeviceType::CUDA), b.nbytes(), b.data<float>(), [](void*) noexcept {});
    Tensor gpu(simulated, b.dtype(), b.shape(), b.stride());
    expect(reference::execute(op(OpCode::COPY), {gpu, out}, out), StatusCode::DeviceMismatch);
    auto alias_op = OpDesc(OpCode::RESHAPE, {0}, {1}, ReshapeAttrs{Shape{4}});
    auto alias = a.reshape({4});
    expect(reference::execute(alias_op, {a}, alias), StatusCode::Unsupported);
}
void randomized_matmul() {
    std::mt19937 rng(0x5203);
    for (std::size_t trial = 0; trial < 100; ++trial) {
        const auto m = static_cast<std::int64_t>(rng() % 6), k = static_cast<std::int64_t>(rng() % 7), n = static_cast<std::int64_t>(rng() % 6);
        auto a = Tensor::allocate_cpu({m, k}), b = Tensor::allocate_cpu({k, n}), out = Tensor::allocate_cpu({m, n});
        for (std::size_t i = 0; i < a.numel(); ++i) a.data<float>()[i] = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 100;
        for (std::size_t i = 0; i < b.numel(); ++i) b.data<float>()[i] = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 100;
        success(reference::execute(op(OpCode::MATMUL), {a, b}, out));
        for (std::int64_t row = 0; row < m; ++row) for (std::int64_t column = 0; column < n; ++column) {
            long double sum = 0;
            for (std::int64_t inner = 0; inner < k; ++inner)
                sum += static_cast<long double>(a.at<float>({row, inner})) * b.at<float>({inner, column});
            const auto expected = static_cast<float>(sum);
            require(std::abs(out.at<float>({row, column}) - expected) <= 1e-6f + 1e-6f * std::abs(expected), "random independent long-double reference");
        }
    }
    std::cout << "Reference random MATMUL: PASS seed=20995 trials=100 (zero/rectangular), no performance claim\n";
}
} // namespace
int main() {
    try {
        hand_vectors(); copies_and_errors(); randomized_matmul();
        require(testing::cpu_allocation_counts().live == 0, "reference tests leaked buffers");
        std::cout << "CPU reference ADD/MUL/MATMUL/COPY/MATERIALIZE: PASS\n"; return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_reference: " << error.what() << '\n'; return 1;
    }
}
