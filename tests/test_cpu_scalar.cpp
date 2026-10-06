#include "runtime/cpu_backend.hpp"
#include "runtime/planned_executor.hpp"
#include "runtime/reference.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace runtime;
void require(bool c, const char* m) { if (!c) throw std::runtime_error(m); }
void success(const Status& s) { if (!s.ok()) throw std::runtime_error(s.message); }
void expect(const Status& s, StatusCode c) { require(s.code == c && !s.message.empty(), "scalar status classification"); }
const CpuBackend scalar(CpuMatmul::ScalarFP32V0);
const OpDesc mm(OpCode::MATMUL, {0, 1}, {2});
void shapes() {
    std::mt19937 random(0x6002);
    for (unsigned trial = 0; trial < 240; ++trial) {
        const std::int64_t m = random() % 10, n = random() % 11, k = random() % 35;
        auto a = Tensor::allocate_cpu({m, k}), b = Tensor::allocate_cpu({k, n});
        // Guard values surround even an empty output; the view is contiguous.
        auto owner = Tensor::allocate_cpu({m * n + 2});
        for (std::size_t i = 0; i < owner.numel(); ++i) owner.data<float>()[i] = -77;
        auto out = owner.narrow(0, 1, m * n).reshape({m, n});
        for (std::size_t i = 0; i < a.numel(); ++i) a.data<float>()[i] = static_cast<float>(static_cast<int>(random() % 2001) - 1000) / 1000;
        for (std::size_t i = 0; i < b.numel(); ++i) b.data<float>()[i] = static_cast<float>(static_cast<int>(random() % 2001) - 1000) / 1000;
        const auto storage = out.storage(); const auto before = testing::cpu_allocation_counts();
        success(scalar.execute(mm, {a, b}, out));
        require(out.storage() == storage && testing::cpu_allocation_counts().allocations == before.allocations, "scalar writes caller output only");
        require(owner.data<float>()[0] == -77 && owner.data<float>()[owner.numel()-1] == -77, "scalar output guards");
        for (std::int64_t i = 0; i < m; ++i) for (std::int64_t j = 0; j < n; ++j) {
            long double sum = 0;
            for (std::int64_t q = 0; q < k; ++q) sum += static_cast<long double>(a.at<float>({i, q})) * b.at<float>({q, j});
            require(std::abs(static_cast<long double>(out.at<float>({i, j})) - sum) <= 1e-5L + 1e-5L * std::abs(sum), "scalar independent long-double oracle");
        }
    }
    auto a = Tensor::allocate_cpu({1, 3}), b = Tensor::allocate_cpu({3, 1}), c = Tensor::allocate_cpu({1, 1});
    a.data<float>()[0] = 16777216; a.data<float>()[1] = 1; a.data<float>()[2] = -16777216;
    for (std::size_t i = 0; i < 3; ++i) b.data<float>()[i] = 1;
    success(scalar.execute(mm, {a, b}, c)); require(c.data<float>()[0] == 0, "v0 accumulates FP32, not S2 oracle");
    success(default_cpu_backend().execute(mm, {a, b}, c)); require(c.data<float>()[0] == 1, "stable default retains FP64 accumulation");
}
void failures_and_copies() {
    auto a = Tensor::allocate_cpu({2, 2}), b = Tensor::allocate_cpu({2, 2}), c = Tensor::allocate_cpu({2, 2});
    for (std::size_t i = 0; i < 4; ++i) { a.data<float>()[i] = 1; b.data<float>()[i] = 1; c.data<float>()[i] = 99; }
    expect(scalar.prepare(mm, {a, b}, a).status, StatusCode::Aliasing);
    expect(scalar.execute(mm, {a, b}, a), StatusCode::Aliasing);
    auto t = a.transpose(0, 1);
    expect(scalar.execute(mm, {t, b}, c), StatusCode::LayoutMismatch);
    auto wrong = Tensor::allocate_cpu({3, 2});
    expect(scalar.execute(mm, {a, b}, wrong), StatusCode::ShapeMismatch);
    auto integer = Tensor::allocate_cpu({2, 2}, DType::INT32);
    expect(scalar.execute(mm, {integer, integer}, integer), StatusCode::DTypeMismatch);
    a.data<float>()[3] = std::numeric_limits<float>::quiet_NaN();
    require(scalar.prepare(mm, {a, b}, c).ok(), "prepare is metadata-only");
    expect(scalar.execute(mm, {a, b}, c), StatusCode::NonFinite);
    for (std::size_t i = 0; i < 4; ++i) require(c.data<float>()[i] == 99, "nonfinite input rejects before writes");
    a.data<float>()[3] = std::numeric_limits<float>::infinity();
    expect(scalar.execute(mm, {a, b}, c), StatusCode::NonFinite);
    a.data<float>()[3] = std::numeric_limits<float>::max(); b.data<float>()[2] = 2;
    expect(scalar.execute(mm, {a, b}, c), StatusCode::NonFinite);
    auto fake_storage = Storage::wrap(Device(DeviceType::CUDA), b.nbytes(), b.data<float>(), [](void*) noexcept {});
    Tensor fake_gpu(fake_storage, b.dtype(), b.shape(), b.stride());
    expect(scalar.execute(mm, {a, fake_gpu}, c), StatusCode::DeviceMismatch);
    // Explicit strided INT32 movement uses exactly S1/S2 semantics.
    auto source = Tensor::allocate_cpu({2, 3}, DType::INT32);
    for (std::size_t i = 0; i < 6; ++i) source.data<std::int32_t>()[i] = static_cast<std::int32_t>(i + 1);
    auto transposed = source.transpose(0, 1), dense = Tensor::allocate_cpu({3, 2}, DType::INT32);
    success(scalar.execute(OpDesc(OpCode::MATERIALIZE, {0}, {1}), {transposed}, dense));
    for (std::int64_t i = 0; i < 3; ++i) for (std::int64_t j = 0; j < 2; ++j)
        require(dense.at<std::int32_t>({i, j}) == source.at<std::int32_t>({j, i}), "scalar materialize reuses S1 copy");
    success(scalar.copy(dense, dense));
    expect(scalar.execute(OpDesc(OpCode::MATERIALIZE, {0}, {1}), {dense}, dense), StatusCode::Aliasing);
    auto padded_owner = Tensor::allocate_cpu({3, 4}, DType::INT32);
    auto padded = padded_owner.slice(1, 0, 2, 2);
    success(scalar.copy(dense, padded));
    require(padded.at<std::int32_t>({2, 1}) == 6 && padded_owner.at<std::int32_t>({2, 3}) == 0, "COPY preserves padding");
}
void graph() {
    Graph g;
    auto a = Tensor::allocate_cpu({2, 3}), b = Tensor::allocate_cpu({3, 2});
    for (std::size_t i = 0; i < 6; ++i) { a.data<float>()[i] = static_cast<float>(i + 1); b.data<float>()[i] = static_cast<float>(i + 1); }
    g.add_input(0, "a", a); g.add_input(1, "b", b);
    for (TensorId i = 2; i <= 4; ++i) g.add_tensor(i, {2, 2});
    g.add_node(0, OpDesc(OpCode::MATMUL, {0, 1}, {2}));
    g.add_node(1, OpDesc(OpCode::ADD, {2, 2}, {3}));
    g.add_node(2, OpDesc(OpCode::MUL, {3, 3}, {4}));
    g.add_output("result", 4); success(g.freeze());
    for (auto policy : {PlanPolicy::NoReuse, PlanPolicy::Reuse}) {
        PlannedAllocationProvider provider(g, policy);
        const auto output = execute_planned(g, provider, nullptr, &scalar);
        require(output.ok() && output.counts.allocations == 0, "scalar graph planned integration");
        const float expected[] = {22,28,49,64};
        for (std::size_t i = 0; i < 4; ++i) require(output.outputs.at("result").data<float>()[i] == 4 * expected[i] * expected[i], "scalar graph oracle");
        expect(execute_planned(g, provider, nullptr, &scalar).status, StatusCode::InvalidArgument);
    }
    const auto dynamic = execute_graph(g, nullptr, nullptr, &scalar);
    require(dynamic.ok() && dynamic.counts.allocations == 3, "dynamic provider stable default");
}
} // namespace
int main() {
    try {
        shapes(); failures_and_copies(); graph();
        require(testing::cpu_allocation_counts().live == 0, "scalar tests leaked backing");
        std::cout << "CPU FP32 ijk v0: PASS seed=24578 random=240 zero/rectangular/guards/errors/copy/planner\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "test_cpu_scalar: " << e.what() << '\n'; return 1; }
}
