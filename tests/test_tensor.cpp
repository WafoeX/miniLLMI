#include "runtime/device.hpp"
#include "runtime/shape.hpp"
#include "runtime/storage.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Exception, class F> void throws(F&& fn, const char* message) {
    try { fn(); }
    catch (const Exception& error) {
        require(!std::string(error.what()).empty(), "exception must have diagnostic");
        return;
    }
    throw std::runtime_error(message);
}
void test_metadata() {
    using namespace runtime;
    require(numel(Shape{}) == 1 && nbytes(Shape{}, DType::FP32) == 4, "scalar semantics");
    require(numel(Shape{2, 0, 3}) == 0 && nbytes(Shape{0}, DType::INT32) == 0, "empty semantics");
    require(numel(Shape{2, 3, 4}) == 24 && nbytes(Shape{2, 3, 4}, DType::INT32) == 96, "byte counts");
    require(contiguous_stride(Shape{2, 3, 4}) == Stride{12, 4, 1}, "row-major strides");
    require(contiguous_stride(Shape{2, 0, 3}) == Stride{3, 3, 1}, "positive empty strides");
    require(storage_span_bytes(Shape{2, 3}, Stride{4, 1}, DType::FP32) == 28, "strided span");
    require(storage_span_bytes(Shape{}, Stride{}, DType::FP32) == 4, "scalar span");
    require(Shape(std::vector<std::int64_t>(MAX_RANK, 1)).rank() == MAX_RANK, "rank limit");
    throws<std::invalid_argument>([] { Shape bad{-1}; }, "negative dimension accepted");
    throws<std::invalid_argument>([] { Stride bad{-1}; }, "negative stride accepted");
    throws<std::invalid_argument>([] { Shape bad(std::vector<std::int64_t>(MAX_RANK + 1, 1)); }, "rank overflow accepted");
    throws<std::out_of_range>([] { (void)Shape{1}[1]; }, "axis bounds accepted");
    const auto huge = std::numeric_limits<std::int64_t>::max();
    require(numel(Shape{huge, huge, 0}) == 0, "empty numel must short-circuit");
    throws<std::overflow_error>([&] { (void)numel(Shape{huge, huge}); }, "numel overflow accepted");
    throws<std::overflow_error>([&] { (void)nbytes(Shape{huge}, DType::FP32); }, "byte overflow accepted");
    throws<std::overflow_error>([&] { (void)contiguous_stride(Shape{1, huge, 2}); }, "stride overflow accepted");
    throws<std::overflow_error>([&] { (void)storage_span_bytes(Shape{3}, Stride{huge}, DType::FP32); }, "span overflow accepted");
    throws<std::overflow_error>([] { (void)checked_add(std::numeric_limits<std::size_t>::max(), 1); }, "addition overflow accepted");
    throws<std::invalid_argument>([] { (void)storage_span_bytes(Shape{2}, Stride{}, DType::FP32); }, "rank mismatch accepted");
    throws<std::invalid_argument>([] { (void)dtype_size(static_cast<DType>(99)); }, "unknown dtype accepted");
    require(dtype_of<float>() == DType::FP32 && dtype_of<const std::int32_t>() == DType::INT32, "typed metadata");
    require(Device{} == Device(DeviceType::CPU, 0), "CPU device");
    require(Device(DeviceType::CUDA, 2).index() == 2, "backend-neutral CUDA declaration");
    throws<std::invalid_argument>([] { Device bad(DeviceType::CPU, 1); }, "invalid CPU index accepted");
    throws<std::invalid_argument>([] { Device bad(DeviceType::CUDA, -1); }, "negative device accepted");
    throws<std::invalid_argument>([] { Device bad(static_cast<DeviceType>(99)); }, "unknown device accepted");
}
void test_storage() {
    using namespace runtime;
    const auto before = testing::cpu_allocation_counts();
    std::weak_ptr<Storage> weak;
    {
        auto source = Storage::allocate_cpu(64);
        require(source->device() == Device{} && source->capacity_bytes() == 64 && source->data(), "CPU storage metadata");
        weak = source;
        auto alias = source;
        require(alias.use_count() == 2 && alias->data() == source->data(), "shared storage alias");
        source.reset();
        require(!weak.expired() && alias.use_count() == 1, "alias must keep storage alive");
        static_cast<unsigned char*>(alias->data())[63] = 42;
    }
    auto after = testing::cpu_allocation_counts();
    require(weak.expired() && after.allocations == before.allocations + 1 &&
            after.frees == before.frees + 1 && after.live == before.live, "exactly one CPU release");
    auto empty = Storage::allocate_cpu(0);
    require(empty->data() == nullptr && empty->capacity_bytes() == 0, "zero-byte storage policy");
    require(testing::cpu_allocation_counts().allocations == after.allocations, "empty buffer must not allocate");
    testing::fail_next_cpu_allocation();
    throws<std::bad_alloc>([] { (void)Storage::allocate_cpu(16); }, "failure injection ignored");
    require(testing::cpu_allocation_counts().live == before.live, "failed allocation must not leak");
    int calls = 0;
    auto token = std::make_shared<int>(7);
    std::weak_ptr<int> weak_token = token;
    {
        auto* pointer = new std::int32_t[4]{};
        auto wrapped = Storage::wrap(Device{}, 16, pointer,
            [token, &calls](void* p) noexcept { ++calls; delete[] static_cast<std::int32_t*>(p); });
        token.reset();
        auto alias = wrapped;
        wrapped.reset();
        require(!weak_token.expired() && calls == 0, "custom deleter capture/lifetime");
    }
    require(calls == 1 && weak_token.expired(), "custom deleter exactly once");
    {
        auto zero = Storage::wrap(Device{}, 0, nullptr, [&calls](void*) noexcept { ++calls; });
    }
    require(calls == 2, "zero wrapped deleter exactly once");
    int invalid_calls = 0;
    throws<std::invalid_argument>([&] {
        (void)Storage::wrap(Device{}, 4, nullptr, [&invalid_calls](void*) { ++invalid_calls; });
    }, "nonnull policy accepted");
    require(invalid_calls == 0, "invalid wrap must not transfer ownership");
    throws<std::invalid_argument>([] { (void)Storage::wrap(Device{}, 0, nullptr, {}); }, "missing deleter accepted");
}
} // namespace
int main() {
    try {
        test_metadata();
        test_storage();
        std::cout << "Tensor metadata/storage: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_tensor: " << error.what() << '\n';
        return 1;
    }
}
